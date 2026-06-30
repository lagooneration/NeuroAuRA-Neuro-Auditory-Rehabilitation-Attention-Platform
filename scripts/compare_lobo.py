"""
scripts/compare_lobo.py

Leave-One-Block-Out (LOBO) comparison:
  - LinearDecoder (classical mTRF, per-subject)
  - ZionGolumbicAdapter (global deep model, trained on all-but-one block per subject)

Protocol (mirrors AAD literature standard):
    For each subject (N=5 blocks):
        For each test block b in {1..5}:
            train_blocks = all blocks except b
            test_block  = b
            Fit LinearDecoder on train_blocks, evaluate on test_block.
            Evaluate global deep model on test_block.
    Report mean across all (subject, block) folds.

The deep model is trained ONCE on all subjects' train-side data
(i.e. 4/5 of every subject's data) and evaluated on all left-out blocks.

Usage
-----
    python scripts/compare_lobo.py \\
        --bids-root "F:/neurophile_data/ds003516" \\
        --device cuda \\
        --epochs 30

    # Only included subjects (recommended — excludes sub-004, 007, 010, 016)
    python scripts/compare_lobo.py \\
        --bids-root "F:/neurophile_data/ds003516" \\
        --device cuda \\
        --epochs 30 \\
        --incl-only
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from train_bids_real import load_bids_data_by_block

from neurophile.decoding import LinearDecoder
from neurophile.models import ZionGolumbicAdapter, GlobalCITrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("compare_lobo")

# ds003516: subjects marked 'excl' in participants.tsv
EXCLUDED = {"004", "007", "010", "016"}
ALL_SUBJECTS = [f"{i:03d}" for i in range(1, 26)]
INCLUDED_SUBJECTS = [s for s in ALL_SUBJECTS if s not in EXCLUDED]


def lobo_baseline(subjects, bids_root):
    """Per-subject LOBO: train mTRF on 4 blocks, test on 1. Average over 5 folds."""
    subject_results = {}
    for sub in subjects:
        logger.info("[Baseline] ── Subject %s ──────────────────────", sub)
        try:
            blocks = load_bids_data_by_block(bids_root, sub)
        except Exception as exc:
            logger.error("[Baseline] Skipping %s: %s", sub, exc)
            continue

        n_blocks = len(blocks)
        if n_blocks < 2:
            logger.warning("[Baseline] Subject %s has only %d block(s), skipping.", sub, n_blocks)
            continue

        fold_accs, fold_rs = [], []
        for test_b in range(n_blocks):
            train_blocks = [blocks[i] for i in range(n_blocks) if i != test_b]
            te_eeg, te_env, te_lbl = blocks[test_b]

            tr_eeg = np.concatenate([b[0] for b in train_blocks])
            tr_env = np.concatenate([b[1] for b in train_blocks])
            tr_lbl = np.concatenate([b[2] for b in train_blocks])

            model = LinearDecoder()
            trainer = GlobalCITrainer(model=model, loss_mode="reconstruction",
                                      output_dir=Path("./checkpoints"))
            trainer.train(tr_eeg, tr_env, tr_lbl)
            metrics = trainer.evaluate(te_eeg, te_env, te_lbl)
            fold_accs.append(metrics["accuracy"])
            fold_rs.append(metrics["mean_pearson_r"])
            logger.info("[Baseline] Sub %s fold %d/%d → acc=%.1f%%  r=%.4f",
                        sub, test_b + 1, n_blocks,
                        metrics["accuracy"] * 100, metrics["mean_pearson_r"])

        subject_results[sub] = {
            "accuracy": float(np.mean(fold_accs)),
            "mean_pearson_r": float(np.mean(fold_rs)),
        }
        logger.info("[Baseline] Sub %s MEAN → acc=%.1f%%  r=%.4f",
                    sub, subject_results[sub]["accuracy"] * 100,
                    subject_results[sub]["mean_pearson_r"])

    return subject_results


def lobo_deep(subjects, bids_root, device, epochs):
    """Global LOBO for deep model.

    For each fold (test block index b = 0..4):
        Train on: all subjects' blocks EXCEPT block b
        Test on:  each subject's block b

    Returns per-subject mean accuracy across folds.
    """
    # Discover max number of blocks
    logger.info("[Deep] Pre-loading block counts...")
    subject_blocks = {}
    for sub in subjects:
        try:
            blocks = load_bids_data_by_block(bids_root, sub)
            if len(blocks) >= 2:
                subject_blocks[sub] = blocks
            else:
                logger.warning("[Deep] Sub %s: only %d block(s), skipping.", sub, len(blocks))
        except Exception as exc:
            logger.error("[Deep] Skipping %s: %s", sub, exc)

    if not subject_blocks:
        raise RuntimeError("No valid subjects for deep LOBO.")

    n_folds = min(len(b) for b in subject_blocks.values())
    logger.info("[Deep] Running %d-fold LOBO across %d subjects", n_folds, len(subject_blocks))

    subject_fold_results = {sub: [] for sub in subject_blocks}

    for test_b in range(n_folds):
        logger.info("[Deep] ── Fold %d/%d (test block %d) ──────────────────────",
                    test_b + 1, n_folds, test_b + 1)

        # Build global training set: all subjects, all blocks except test_b
        all_tr_eeg, all_tr_env, all_tr_lbl = [], [], []
        for sub, blocks in subject_blocks.items():
            for i, (be, bv, bl) in enumerate(blocks):
                if i != test_b:
                    all_tr_eeg.append(be)
                    all_tr_env.append(bv)
                    all_tr_lbl.append(bl)

        tr_eeg = np.concatenate(all_tr_eeg)
        tr_env = np.concatenate(all_tr_env)
        tr_lbl = np.concatenate(all_tr_lbl)
        logger.info("[Deep] Fold %d train set: %d trials from %d subjects",
                    test_b + 1, len(tr_eeg), len(subject_blocks))

        model = ZionGolumbicAdapter(num_eeg_channels=64, audio_sampling_rate=64)
        trainer = GlobalCITrainer(
            model=model, lr=3e-4, epochs=epochs, batch_size=64,
            loss_mode="classification", device=device,
            output_dir=Path("./checkpoints"),
        )
        t0 = time.time()
        trainer.train(tr_eeg, tr_env, tr_lbl)
        logger.info("[Deep] Fold %d training done in %.1f min", test_b + 1, (time.time() - t0) / 60)

        # Evaluate on each subject's left-out block
        for sub, blocks in subject_blocks.items():
            if test_b >= len(blocks):
                continue
            te_eeg, te_env, te_lbl = blocks[test_b]
            metrics = trainer.evaluate(te_eeg, te_env, te_lbl)
            subject_fold_results[sub].append(metrics)
            logger.info("[Deep] Sub %s fold %d → acc=%.1f%%  r=%.4f",
                        sub, test_b + 1,
                        metrics["accuracy"] * 100, metrics["mean_pearson_r"])

    # Average over folds per subject
    subject_results = {}
    for sub, fold_metrics in subject_fold_results.items():
        if fold_metrics:
            subject_results[sub] = {
                "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
                "mean_pearson_r": float(np.mean([m["mean_pearson_r"] for m in fold_metrics])),
            }

    return subject_results


def print_table(baseline, deep):
    subjects = sorted(set(baseline) | set(deep))
    hdr = f"{'Sub':>5}  {'Baseline Acc':>13}  {'Baseline r':>10}  {'Deep Acc':>10}  {'Deep r':>8}  {'Delta':>8}"
    sep = "─" * len(hdr)
    logger.info(sep)
    logger.info("LOBO COMPARISON (5-fold leave-one-block-out)")
    logger.info(sep)
    logger.info(hdr)
    logger.info("─" * len(hdr))

    bl_accs, dp_accs = [], []
    for sub in subjects:
        bl = baseline.get(sub, {})
        dp = deep.get(sub, {})
        ba = bl.get("accuracy", float("nan")) * 100
        br = bl.get("mean_pearson_r", float("nan"))
        da = dp.get("accuracy", float("nan")) * 100
        dr = dp.get("mean_pearson_r", float("nan"))
        delta = da - ba
        sign = "▲" if delta >= 0 else "▼"
        logger.info(f"{sub:>5}  {ba:>12.1f}%  {br:>10.4f}  {da:>9.1f}%  {dr:>8.4f}  {sign}{abs(delta):>6.1f}%")
        if not np.isnan(ba):
            bl_accs.append(ba)
        if not np.isnan(da):
            dp_accs.append(da)

    logger.info("─" * len(hdr))
    m_bl = np.mean(bl_accs) if bl_accs else float("nan")
    m_dp = np.mean(dp_accs) if dp_accs else float("nan")
    sign = "▲" if m_dp >= m_bl else "▼"
    logger.info(f"{'MEAN':>5}  {m_bl:>12.1f}%  {'':>10}  {m_dp:>9.1f}%  {'':>8}  {sign}{abs(m_dp - m_bl):.1f}%")
    logger.info(sep)


def main():
    parser = argparse.ArgumentParser(description="LOBO: Baseline vs ZionGolumbic deep model")
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--subjects", type=str, default="all",
                        help="Comma-separated IDs, 'all', or 'included' (excludes excl subjects)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--incl-only", action="store_true",
                        help="Shortcut for --subjects included (skip excl subjects)")
    args = parser.parse_args()

    if args.incl_only or args.subjects.lower() == "included":
        subjects = INCLUDED_SUBJECTS
        logger.info("Using %d included subjects (excl: %s)", len(subjects), EXCLUDED)
    elif args.subjects.lower() == "all":
        subjects = ALL_SUBJECTS
    else:
        subjects = [s.strip() for s in args.subjects.split(",")]

    logger.info("Subjects: %s", subjects)
    logger.info("Device: %s  |  Epochs: %d", args.device, args.epochs)

    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: LinearDecoder LOBO (per-subject)")
    logger.info("=" * 60)
    baseline_results = lobo_baseline(subjects, args.bids_root)

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: ZionGolumbicAdapter LOBO (global model, CUDA)")
    logger.info("=" * 60)
    deep_results = lobo_deep(subjects, args.bids_root, args.device, args.epochs)

    print_table(baseline_results, deep_results)


if __name__ == "__main__":
    main()

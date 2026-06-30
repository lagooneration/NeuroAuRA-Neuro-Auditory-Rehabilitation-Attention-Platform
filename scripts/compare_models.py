"""
scripts/compare_models.py

Head-to-head comparison:
  - LinearDecoder (classical mTRF backward model, per-subject)
  - ZionGolumbicAdapter (deep cross-attention, global model on RTX 4060)

Protocol
--------
For each subject:
  1. Load data, split 80% train / 20% test (reproducible, no shuffle).
  2. Fit a fresh LinearDecoder on the subject's train split.
  3. Evaluate both models on the same held-out test split.

The global deep model is trained ONCE on all subjects' train splits,
then evaluated per-subject on their respective test splits.

Usage
-----
    python scripts/compare_models.py \\
        --bids-root "F:\\neurophile_data\\ds003516" \\
        --device cuda \\
        --epochs 30
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from train_bids_real import load_bids_data

from neurophile.decoding import LinearDecoder
from neurophile.models import ZionGolumbicAdapter, GlobalCITrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("compare_models")

# ── Config ────────────────────────────────────────────────────────────────────
DS003516_SUBJECTS = [f"{i:03d}" for i in range(1, 26)]


def split_data(eeg, env, labels, test_frac=0.20):
    """Deterministic 80/20 train/test split (last 20% held out)."""
    n = len(eeg)
    split = int(n * (1 - test_frac))
    return (
        eeg[:split], env[:split], labels[:split],
        eeg[split:], env[split:], labels[split:],
    )


def eval_baseline_per_subject(bids_root, subjects, test_frac=0.20):
    """Train + evaluate LinearDecoder independently per subject."""
    results = {}
    for sub in subjects:
        logger.info("[Baseline] Loading Subject %s ...", sub)
        try:
            eeg, env, labels = load_bids_data(bids_root, sub)
        except Exception as exc:
            logger.error("[Baseline] Skipping subject %s: %s", sub, exc)
            continue

        tr_eeg, tr_env, tr_lbl, te_eeg, te_env, te_lbl = split_data(eeg, env, labels, test_frac)

        model = LinearDecoder()
        trainer = GlobalCITrainer(model=model, loss_mode="reconstruction",
                                  output_dir=Path("./checkpoints"))
        trainer.train(tr_eeg, tr_env, tr_lbl)
        metrics = trainer.evaluate(te_eeg, te_env, te_lbl)

        results[sub] = metrics
        logger.info(
            "[Baseline] Subject %s → acc=%.1f%%  r=%.4f",
            sub,
            metrics["accuracy"] * 100,
            metrics["mean_pearson_r"],
        )
    return results


def train_global_deep_model(bids_root, subjects, device, epochs, test_frac=0.20):
    """Train ZionGolumbicAdapter once on all subjects' train splits.
    Returns (trainer, per-subject test splits dict).
    """
    all_train_eeg, all_train_env, all_train_lbl = [], [], []
    test_splits = {}

    for sub in subjects:
        logger.info("[Deep] Loading Subject %s ...", sub)
        try:
            eeg, env, labels = load_bids_data(bids_root, sub)
        except Exception as exc:
            logger.error("[Deep] Skipping subject %s: %s", sub, exc)
            continue

        tr_eeg, tr_env, tr_lbl, te_eeg, te_env, te_lbl = split_data(eeg, env, labels, test_frac)
        all_train_eeg.append(tr_eeg)
        all_train_env.append(tr_env)
        all_train_lbl.append(tr_lbl)
        test_splits[sub] = (te_eeg, te_env, te_lbl)

    if not all_train_eeg:
        raise RuntimeError("No valid subjects loaded for deep model training.")

    train_eeg = np.concatenate(all_train_eeg, axis=0)
    train_env = np.concatenate(all_train_env, axis=0)
    train_lbl = np.concatenate(all_train_lbl, axis=0)

    logger.info("[Deep] Pooled training set: %d trials across %d subjects",
                len(train_eeg), len(test_splits))

    model = ZionGolumbicAdapter(num_eeg_channels=64, audio_sampling_rate=64)
    trainer = GlobalCITrainer(
        model=model,
        lr=3e-4,
        epochs=epochs,
        batch_size=64,
        loss_mode="classification",
        device=device,
        output_dir=Path("./checkpoints"),
    )

    t0 = time.time()
    trainer.train(train_eeg, train_env, train_lbl)
    logger.info("[Deep] Training complete in %.1f min", (time.time() - t0) / 60)

    return trainer, test_splits


def eval_deep_per_subject(trainer, test_splits):
    """Evaluate the global deep model on each subject's held-out test split."""
    results = {}
    for sub, (te_eeg, te_env, te_lbl) in test_splits.items():
        metrics = trainer.evaluate(te_eeg, te_env, te_lbl)
        results[sub] = metrics
        logger.info(
            "[Deep] Subject %s → acc=%.1f%%  r=%.4f",
            sub,
            metrics["accuracy"] * 100,
            metrics["mean_pearson_r"],
        )
    return results


def print_comparison_table(baseline_results, deep_results):
    subjects = sorted(set(baseline_results) | set(deep_results))

    header = f"{'Sub':>5}  {'Baseline Acc':>13}  {'Baseline r':>10}  {'Deep Acc':>10}  {'Deep r':>8}  {'Delta':>8}"
    logger.info("=" * len(header))
    logger.info("COMPARISON TABLE (80/20 held-out test split)")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    bl_accs, dp_accs = [], []
    for sub in subjects:
        bl = baseline_results.get(sub, {})
        dp = deep_results.get(sub, {})
        ba = bl.get("accuracy", float("nan")) * 100
        br = bl.get("mean_pearson_r", float("nan"))
        da = dp.get("accuracy", float("nan")) * 100
        dr = dp.get("mean_pearson_r", float("nan"))
        delta = da - ba
        sign = "▲" if delta > 0 else "▼"
        logger.info(
            f"{sub:>5}  {ba:>12.1f}%  {br:>10.4f}  {da:>9.1f}%  {dr:>8.4f}  {sign}{abs(delta):>6.1f}%"
        )
        if not np.isnan(ba):
            bl_accs.append(ba)
        if not np.isnan(da):
            dp_accs.append(da)

    logger.info("-" * len(header))
    mean_bl = np.mean(bl_accs) if bl_accs else float("nan")
    mean_dp = np.mean(dp_accs) if dp_accs else float("nan")
    logger.info(
        f"{'MEAN':>5}  {mean_bl:>12.1f}%  {'':>10}  {mean_dp:>9.1f}%  {'':>8}  "
        f"{'▲' if mean_dp > mean_bl else '▼'}{abs(mean_dp - mean_bl):.1f}%"
    )
    logger.info("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Baseline vs ZionGolumbic deep model comparison")
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--subjects", type=str, default="all",
                        help="Comma-separated subject IDs or 'all'")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (cuda / cpu)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs for the deep model")
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Fraction of data held out for testing (default 0.20)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline training (load previously saved if any)")
    parser.add_argument("--skip-deep", action="store_true",
                        help="Skip deep model training (load checkpoint if exists)")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/zion_golumbic_cross_attention_global_ci.pt"),
                        help="Path to existing deep model checkpoint (used with --skip-deep)")
    args = parser.parse_args()

    if args.subjects.lower() == "all":
        subjects = DS003516_SUBJECTS
    else:
        subjects = [s.strip() for s in args.subjects.split(",")]

    logger.info("Subjects: %s", subjects)
    logger.info("Device: %s", args.device)
    logger.info("Epochs: %d", args.epochs)
    logger.info("Test fraction: %.0f%%", args.test_frac * 100)

    # ── Step 1: Linear baseline ────────────────────────────────────────────────
    if not args.skip_baseline:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Training LinearDecoder baseline (per-subject mTRF)")
        logger.info("=" * 60)
        t0 = time.time()
        baseline_results = eval_baseline_per_subject(args.bids_root, subjects, args.test_frac)
        logger.info("[Baseline] All subjects done in %.1f min", (time.time() - t0) / 60)
    else:
        logger.info("[Baseline] Skipped.")
        baseline_results = {}

    # ── Step 2: Global deep model ──────────────────────────────────────────────
    if not args.skip_deep:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Training ZionGolumbicAdapter global model on CUDA")
        logger.info("=" * 60)
        trainer, test_splits = train_global_deep_model(
            args.bids_root, subjects, args.device, args.epochs, args.test_frac
        )
    else:
        logger.info("[Deep] Loading checkpoint from %s ...", args.checkpoint)
        from neurophile.models import ZionGolumbicAdapter
        model = ZionGolumbicAdapter(num_eeg_channels=64, audio_sampling_rate=64)
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(state.get("model_state", state))
        trainer = GlobalCITrainer(model=model, device=args.device,
                                  output_dir=Path("./checkpoints"))
        # Still need test splits
        test_splits = {}
        for sub in subjects:
            try:
                eeg, env, labels = load_bids_data(args.bids_root, sub)
                *_, te_eeg, te_env, te_lbl = split_data(eeg, env, labels, args.test_frac)
                test_splits[sub] = (te_eeg, te_env, te_lbl)
            except Exception as exc:
                logger.error("Skipping %s: %s", sub, exc)

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Evaluating deep model per subject")
    logger.info("=" * 60)
    deep_results = eval_deep_per_subject(trainer, test_splits)

    # ── Step 3: Print table ────────────────────────────────────────────────────
    print_comparison_table(baseline_results, deep_results)


if __name__ == "__main__":
    main()

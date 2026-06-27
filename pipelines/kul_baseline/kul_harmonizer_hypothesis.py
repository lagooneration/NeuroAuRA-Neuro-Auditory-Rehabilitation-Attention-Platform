"""
pipelines/kul_baseline/kul_harmonizer_hypothesis.py

Tests the harmonizer hypothesis on the KUL dataset.

KEY ADVANTAGE over the BIDS version:
    The KUL dataset has perfectly time-aligned audio envelopes (wavA, wavB)
    that are ALREADY matched sample-for-sample to the EEG. This eliminates
    the temporal alignment problem (the 800ms peak lag we saw with BIDS).
    Each trial's wavA and wavB are guaranteed to be synchronized.

Hypothesis:
    Does harmonizing wavA or wavB before computing the TRF correlation
    improve the |r| relative to the dry KUL envelope?

    If yes → the brain tracked harmonic structure in the speech.
    If no  → the brain only tracked the fundamental amplitude envelope.

Usage:
    # Fast (Pearson correlation only):
    python pipelines/kul_baseline/kul_harmonizer_hypothesis.py \\
        --mat-file "P:\\...\\S1_data_preproc.mat" --skip-ridge

    # Full (includes Ridge decoder comparison):
    python pipelines/kul_baseline/kul_harmonizer_hypothesis.py \\
        --mat-file "P:\\...\\S1_data_preproc.mat"
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.signal as ss
from scipy.stats import pearsonr

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipelines.kul_baseline.kul_linear_decoder import load_kul_trials, decode_spatiotemporal
from pipelines.eq_hypothesis.run_harmonizer_hypothesis import (
    apply_harmonizer,
    HARMONIZER_CONFIGS,
)
import copy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("kul_harmonizer")


def decode_with_harmonizer(data: dict, harmonic_voices: list) -> dict:
    """
    Run the Spatiotemporal Ridge AAD decoder using harmonized envelopes
    instead of the raw KUL wavA/wavB envelopes.
    """
    # Create a copy of the data with harmonized envelopes
    data_harm = copy.deepcopy(data)
    for i in range(data["n_trials"]):
        data_harm["wavA"][i] = apply_harmonizer(data["wavA"][i], harmonic_voices)
        data_harm["wavB"][i] = apply_harmonizer(data["wavB"][i], harmonic_voices)

    # Run the full Ridge LOOCV decoder
    res = decode_spatiotemporal(data_harm)
    
    # Calculate attended vs unattended mean correlation from LOOCV details
    details = res["details"]
    mean_r_attended = np.mean([
        d["rA"] if d["label"] == 1 else d["rB"] for d in details
    ])
    mean_r_unattended = np.mean([
        d["rB"] if d["label"] == 1 else d["rA"] for d in details
    ])
    
    res["mean_r_attended"] = float(mean_r_attended)
    res["mean_r_unattended"] = float(mean_r_unattended)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mat-file", type=Path,
        default=Path("data/raw/kul/DATA_preproc.zip.unzip/S1_data_preproc.mat"),
    )
    args = parser.parse_args()

    if not args.mat_file.exists():
        logger.error("File not found: %s", args.mat_file)
        return

    # ── Load KUL data ─────────────────────────────────────────────────────────
    data = load_kul_trials(args.mat_file)

    # ── Run each harmonizer config ────────────────────────────────────────────
    results = []
    for label, voices, description in HARMONIZER_CONFIGS:
        clean_label = label.replace("\n", " ")
        logger.info("Testing harmonizer config: %s ...", clean_label)
        res = decode_with_harmonizer(data, voices)
        results.append({
            "label": clean_label,
            "description": description,
            "n_harmonics": len(voices),
            **res,
        })
        logger.info(
            "  Accuracy=%.1f%% | Attended r=%.4f | Unattended r=%.4f",
            res["accuracy"] * 100, res["mean_r_attended"], res["mean_r_unattended"],
        )

    # ── Sort and print results ────────────────────────────────────────────────
    results.sort(key=lambda x: x["accuracy"], reverse=True)
    baseline_acc = next(r["accuracy"] for r in results if "Dry" in r["label"])
    baseline_r = next(r["mean_r_attended"] for r in results if "Dry" in r["label"])

    print("\n" + "=" * 80)
    print(f"KUL HARMONIZER HYPOTHESIS — {args.mat_file.stem}")
    print(f"Published benchmark (Das et al. 2019): ~85–90%")
    print("=" * 80)
    print(f"{'Rank':<5} {'Config':<35} {'Accuracy':>10} {'Δ Acc':>8} {'Att. r':>8} {'Unatt. r':>10}")
    print("-" * 80)
    for i, r in enumerate(results):
        delta = r["accuracy"] - baseline_acc
        delta_str = f"+{delta*100:.1f}%" if delta >= 0 else f"{delta*100:.1f}%"
        winner = " ← BEST" if i == 0 else ""
        print(
            f"{i+1:<5} {r['label']:<35} {r['accuracy']*100:>9.1f}% {delta_str:>8} "
            f"{r['mean_r_attended']:>8.4f} {r['mean_r_unattended']:>10.4f}{winner}"
        )
    print("=" * 80)

    winner = results[0]
    print(f"\n📊 VERDICT:")
    print(f"   Best config: '{winner['label']}'")
    print(f"   Accuracy: {winner['accuracy']*100:.1f}% vs Dry baseline: {baseline_acc*100:.1f}%")
    print(f"   Attended r: {winner['mean_r_attended']:.4f} vs Dry: {baseline_r:.4f}")
    if winner["accuracy"] > baseline_acc and "Dry" not in winner["label"]:
        improvement = ((winner["accuracy"] / baseline_acc) - 1) * 100
        print(f"   ✅ Hypothesis SUPPORTED: Harmonics improved AAD accuracy by {improvement:.1f}%!")
    else:
        print(f"   ❌ Hypothesis NOT supported: Dry audio gave same or better accuracy.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"KUL Harmonizer Hypothesis — {args.mat_file.stem}\n"
        f"(Published benchmark: ~85–90%, Das et al. 2019)",
        fontsize=12, fontweight="bold",
    )

    # Plot A: Accuracy per config
    ax = axes[0]
    labels_plot = [r["label"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    bar_colors = ["gold" if i == 0 else ("lightcoral" if "Dry" in r["label"] else "steelblue")
                  for i, r in enumerate(results)]
    bars = ax.bar(range(len(labels_plot)), accs, color=bar_colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8, label="Chance (50%)")
    ax.axhline(87.5, color="gray", linestyle=":", linewidth=1.0, label="Published (~87.5%)")
    ax.axhline(baseline_acc * 100, color="orange", linestyle="--", linewidth=1.0,
               label=f"Dry baseline ({baseline_acc*100:.1f}%)")
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AAD Accuracy (%)")
    ax.set_title("Decoding Accuracy by Harmonizer Config")
    ax.set_ylim([0, 110])
    ax.legend(fontsize=7)

    # Plot B: Attended vs Unattended r per config
    ax = axes[1]
    att_rs = [r["mean_r_attended"] for r in results]
    unatt_rs = [r["mean_r_unattended"] for r in results]
    x = np.arange(len(results))
    width = 0.35
    bars1 = ax.bar(x - width/2, att_rs, width, label="Attended stream", color="green", alpha=0.8)
    bars2 = ax.bar(x + width/2, unatt_rs, width, label="Unattended stream", color="salmon", alpha=0.8)
    ax.bar_label(bars1, fmt="%.3f", fontsize=7, padding=1)
    ax.bar_label(bars2, fmt="%.3f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in results], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean |Pearson r|")
    ax.set_title("Attended vs Unattended Stream Correlation\n(Gap = decoding signal)")
    ax.legend(fontsize=9)

    # Plot C: Discrimination gap (attended r - unattended r) — this is the true signal
    ax = axes[2]
    gaps = [r["mean_r_attended"] - r["mean_r_unattended"] for r in results]
    gap_colors = ["gold" if g == max(gaps) else "steelblue" for g in gaps]
    bars3 = ax.bar(range(len(results)), gaps, color=gap_colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars3, fmt="%.4f", fontsize=8, padding=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r["label"] for r in results], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Discrimination Gap\n(Attended r − Unattended r)")
    ax.set_title("Which Config Maximizes\nAttended vs Unattended Discrimination?")

    plt.tight_layout()
    out_path = Path("checkpoints") / f"kul_harmonizer_{args.mat_file.stem}.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logger.info("Figure saved → %s", out_path)
    print(f"\nFigure saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

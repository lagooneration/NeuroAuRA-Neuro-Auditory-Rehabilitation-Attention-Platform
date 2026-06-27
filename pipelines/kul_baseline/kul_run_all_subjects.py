"""
pipelines/kul_baseline/kul_run_all_subjects.py

Runs the full Spatiotemporal Ridge LOOCV baseline on ALL 18 KUL subjects.
The 85-90% published benchmark (Das et al. 2019) is a "grand average" — 
meaning the decoder is trained for each subject individually (because every
person's brain and EEG cap placement is unique), and then their final 
accuracy scores are averaged together.

Usage:
    python pipelines/kul_baseline/kul_run_all_subjects.py \\
        --data-dir "P:\\auditory\\neurophile\\data\\raw\\kul\\DATA_preproc.zip.unzip"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path so 'pipelines' module can be found
sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

# Import the fixed spatiotemporal decoder and loader we wrote
from pipelines.kul_baseline.kul_linear_decoder import (
    load_kul_trials,
    decode_spatiotemporal,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("kul_all_subjects")


def main():
    parser = argparse.ArgumentParser(description="Run KUL baseline on all subjects")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/raw/kul/DATA_preproc.zip.unzip"),
        help="Directory containing S1_data_preproc.mat etc."
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error("Directory not found: %s", args.data_dir)
        return

    # Find all S*.mat files and sort them (S1, S2, ..., S18)
    mat_files = list(args.data_dir.glob("S*_data_preproc.mat"))
    mat_files.sort(key=lambda p: int(p.stem.replace("S", "").split("_")[0]))

    if not mat_files:
        logger.error("No S*_data_preproc.mat files found in %s", args.data_dir)
        return

    logger.info("Found %d subject files. Beginning collective benchmark...", len(mat_files))

    results = []
    for mat_file in mat_files:
        logger.info("\n" + "="*50)
        logger.info("Processing %s", mat_file.name)
        logger.info("="*50)
        
        try:
            data = load_kul_trials(mat_file)
            # Run the Spatiotemporal Ridge LOOCV decoder
            res = decode_spatiotemporal(data)
            
            results.append({
                "subject": mat_file.stem.split("_")[0],
                "accuracy": res["accuracy"],
                "n_correct": res["n_correct"],
                "n_trials": data["n_trials"]
            })
            
            logger.info("--> %s Accuracy: %.1f%%", mat_file.stem.split("_")[0], res["accuracy"] * 100)
            
        except Exception as e:
            logger.error("Failed to process %s: %s", mat_file.name, e)

    if not results:
        return

    # Calculate Grand Average
    accuracies = [r["accuracy"] * 100 for r in results]
    grand_average = np.mean(accuracies)
    
    print("\n" + "=" * 60)
    print("KUL COLLECTIVE BENCHMARK RESULTS (DAS ET AL. 2019)")
    print("=" * 60)
    for r in results:
        print(f"{r['subject']:<5} : {r['accuracy']*100:>5.1f}%  ({r['n_correct']}/{r['n_trials']} correct)")
    print("-" * 60)
    print(f"GRAND AVERAGE : {grand_average:.1f}%")
    print(f"PUBLISHED KUL : ~85–90% (60-second trials)")
    print("=" * 60)

    # Plot the collective results
    fig, ax = plt.subplots(figsize=(10, 5))
    subjects = [r["subject"] for r in results]
    
    # Color subjects based on performance
    colors = []
    for acc in accuracies:
        if acc >= 80: colors.append("forestgreen")
        elif acc >= 65: colors.append("gold")
        else: colors.append("salmon")
        
    bars = ax.bar(subjects, accuracies, color=colors, edgecolor="black")
    ax.axhline(50, color="red", linestyle="--", label="Chance (50%)")
    ax.axhline(grand_average, color="blue", linestyle="-", linewidth=2, label=f"Grand Avg ({grand_average:.1f}%)")
    ax.axhline(87.5, color="gray", linestyle=":", label="Published Benchmark (~87.5%)")
    
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_ylim([0, 105])
    ax.set_ylabel("Decoding Accuracy (%)")
    ax.set_title("Individual Subject Decoding Accuracies (Spatiotemporal Ridge)")
    ax.legend()
    
    plt.tight_layout()
    out_path = Path("checkpoints") / "kul_collective_benchmark.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()

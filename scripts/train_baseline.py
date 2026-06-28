"""
scripts/train_baseline.py

Runs the classical Linear Decoder baseline on the BIDS dataset.
This provides the ~65-75% accuracy mark to beat.
"""
import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np

from neurophile.decoding import LinearDecoder
from neurophile.models.global_trainer import GlobalCITrainer

# Add scripts directory to path to import load_bids_data
sys.path.append(str(Path(__file__).resolve().parent))
from train_bids_real import load_bids_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("train_baseline")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, required=True, help="Path to OpenNeuro dataset")
    parser.add_argument("--subject", type=str, default="001", help="Subject ID")
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing LinearDecoder baseline...")
    model = LinearDecoder()
    
    trainer = GlobalCITrainer(
        model=model,
        loss_mode="reconstruction",
        output_dir=args.output_dir
    )
    
    logger.info("Loading data for Subject %s...", args.subject)
    if args.subject.lower() == "all":
        subjects = [f"{i:03d}" for i in range(1, 26)]
    else:
        subjects = [args.subject]

    per_subject_metrics = []

    for sub in subjects:
        logger.info("="*50)
        logger.info("Processing Subject %s...", sub)

        try:
            eeg, env, labels = load_bids_data(args.bids_root, sub)
        except Exception as e:
            logger.error("Skipping subject %s: %s", sub, e)
            continue

        # Fresh model per subject (mTRF is always subject-specific)
        model = LinearDecoder()
        trainer = GlobalCITrainer(
            model=model,
            loss_mode="reconstruction",
            output_dir=args.output_dir,
        )

        logger.info("Training on Subject %s (%d trials)...", sub, len(eeg))
        trainer.train(eeg, env, labels)

        logger.info("Evaluating Subject %s...", sub)
        metrics = trainer.evaluate(eeg, env, labels)
        per_subject_metrics.append(metrics)
        logger.info(
            "Subject %s → Accuracy: %.2f%%, Pearson r: %.4f",
            sub, metrics.get("accuracy", 0.0) * 100, metrics.get("mean_pearson_r", 0.0),
        )

    if not per_subject_metrics:
        logger.error("No subjects were successfully evaluated.")
        return

    mean_acc = float(np.mean([m["accuracy"] for m in per_subject_metrics]))
    mean_r = float(np.mean([m["mean_pearson_r"] for m in per_subject_metrics]))

    logger.info("="*50)
    logger.info("BASELINE METRICS (Subject %s, N=%d):", args.subject, len(per_subject_metrics))
    logger.info("Accuracy: %.2f%%", mean_acc * 100)
    logger.info("Pearson r: %.4f", mean_r)
    logger.info("="*50)

if __name__ == "__main__":
    main()

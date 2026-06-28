"""
scripts/run_inference.py

Loads a trained Foundation Model checkpoint and evaluates it on a test subject.
Supports both OpenNeuro BIDS datasets and KUL .mat datasets.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from neurophile.models import MesgaraniAdapter, ZionGolumbicAdapter, GlobalCITrainer
import sys

# Add project root to path so we can import from the scripts folder
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.train_bids_real import load_bids_data
from scripts.train_kul_real import load_kul_subject

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("run_inference")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mesgarani_crn_global_ci.pt"))
    parser.add_argument("--dataset-type", choices=["bids", "kul"], default="bids",
                        help="Dataset format: 'bids' for OpenNeuro .set files, 'kul' for KUL .mat files")
    parser.add_argument("--model", type=str, choices=["mesgarani", "zion_golumbic"], default="mesgarani", help="Deep learning architecture to use")
    # BIDS args
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--subject", type=str, default="001")
    # KUL args
    parser.add_argument("--mat-file", type=Path, default=None,
                        help="Path to a specific KUL .mat file (e.g. S1_data_preproc.mat)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        logger.error("Checkpoint not found at %s", args.checkpoint)
        return
        
    logger.info("Loading model from %s", args.checkpoint)
    if args.model == "mesgarani":
        model = MesgaraniAdapter(num_eeg_channels=64, audio_sampling_rate=64)
    else:
        model = ZionGolumbicAdapter(num_eeg_channels=64, audio_sampling_rate=64)
    
    # Load weights
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
        
    trainer = GlobalCITrainer(
        model=model,
        epochs=1,
        batch_size=8,
        device=args.device,
        output_dir=Path("./checkpoints")
    )
    
    logger.info("Loading test data (%s mode)...", args.dataset_type)
    if args.dataset_type == "kul":
        if args.mat_file is None:
            logger.error("--mat-file is required when using --dataset-type kul")
            return
        if not args.mat_file.exists():
            logger.error("KUL .mat file not found: %s", args.mat_file)
            return
        eeg, env, labels = load_kul_subject(args.mat_file)
    else:
        if args.bids_root is None:
            logger.error("--bids-root is required when using --dataset-type bids")
            return
            
        if args.subject.lower() == "all":
            subjects = [f"{i:03d}" for i in range(1, 26)]
        else:
            subjects = [args.subject]
            
        all_eeg, all_env, all_labels = [], [], []
        for sub in subjects:
            try:
                e, en, l = load_bids_data(args.bids_root, sub)
                all_eeg.append(e)
                all_env.append(en)
                all_labels.append(l)
            except Exception as e:
                logger.error("Skipping subject %s: %s", sub, e)
                
        if not all_eeg:
            logger.error("No valid data loaded.")
            return
            
        eeg = np.concatenate(all_eeg, axis=0)
        env = np.concatenate(all_env, axis=0)
        labels = np.concatenate(all_labels, axis=0)
    
    # Evaluate
    logger.info("Running evaluation forward pass...")
    metrics = trainer.evaluate(eeg, env, labels)
    
    logger.info("="*50)
    logger.info("FINAL INFERENCE METRICS (Subject %s):", args.subject)
    logger.info("Accuracy: %.2f%%", metrics.get("accuracy", 0.0) * 100)
    logger.info("Pearson r: %.4f", metrics.get("mean_pearson_r", 0.0))
    logger.info("="*50)

if __name__ == "__main__":
    main()

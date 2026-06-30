"""
scripts/evaluate_zero_shot.py

Evaluates the pre-trained Global CI Foundation model (ZionGolumbicAdapter) 
on the completely unseen KU Leuven dataset to prove zero-shot cross-dataset generalization.
"""
import argparse
import logging
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from neurophile.models import ZionGolumbicAdapter, GlobalCITrainer
from scripts.train_kul_real import load_kul_subject

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("evaluate_zero_shot")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/zion_golumbic_cross_attention_global_ci.pt"))
    parser.add_argument("--kul-dir", type=Path, default=Path("data/raw/kul/DATA_preproc.zip.unzip"))
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found at {args.checkpoint}!")
        return
        
    if not args.kul_dir.exists():
        logger.error(f"KUL data not found at {args.kul_dir}!")
        return

    # 1. Initialize the Model and Trainer
    model = ZionGolumbicAdapter(num_eeg_channels=64)
    logger.info(f"Loading weights from {args.checkpoint}")
    checkpoint_dict = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if "model_state" in checkpoint_dict:
        model.load_state_dict(checkpoint_dict["model_state"])
    else:
        model.load_state_dict(checkpoint_dict)
    model.to(args.device)
    model.eval()

    trainer = GlobalCITrainer(
        model=model,
        epochs=0, # No training!
        batch_size=8,
        device=args.device,
        output_dir=Path("checkpoints")
    )
    
    # 2. Evaluate on each KUL subject
    accuracies = []
    
    for sub_idx in range(1, 17): # S1 to S16
        mat_path = args.kul_dir / f"S{sub_idx}_data_preproc.mat"
        if not mat_path.exists():
            logger.warning(f"File {mat_path} missing, skipping.")
            continue
            
        eeg_arr, env_arr, label_arr = load_kul_subject(mat_path)
        
        # FIX 1: The model was trained on Z-scored EEG. KUL provides raw EEG.
        # We must Z-score the EEG data (shape: Trials, Time, Channels)
        mean_eeg = np.mean(eeg_arr, axis=1, keepdims=True)
        std_eeg = np.std(eeg_arr, axis=1, keepdims=True) + 1e-8
        eeg_arr = (eeg_arr - mean_eeg) / std_eeg
        
        # FIX 2: GlobalCITrainer's 2AFC evaluation hard-assumes that the dataset 
        # is ordered [Attended, Unattended, Attended, Unattended...].
        # load_kul_subject outputs [SampleA, SampleB], regardless of which is attended.
        eeg_fixed, env_fixed, lbl_fixed = [], [], []
        
        for i in range(0, len(eeg_arr), 2):
            eeg1, env1, lbl1 = eeg_arr[i], env_arr[i], label_arr[i]
            eeg2, env2, lbl2 = eeg_arr[i+1], env_arr[i+1], label_arr[i+1]
            
            if lbl1 == 1.0:
                eeg_fixed.extend([eeg1, eeg2])
                env_fixed.extend([env1, env2])
                lbl_fixed.extend([1.0, 0.0])
            else:
                eeg_fixed.extend([eeg2, eeg1])
                env_fixed.extend([env2, env1])
                lbl_fixed.extend([1.0, 0.0])
                
        eeg_arr = np.stack(eeg_fixed)
        env_arr = np.stack(env_fixed)
        label_arr = np.array(lbl_fixed, dtype="float32")
        
        # Calculate metric via evaluate() but manually grab accuracy
        metrics = trainer.evaluate(eeg_arr, env_arr, label_arr)
        acc = metrics.get('accuracy', 0.0)
        
        logger.info(f"[Zero-Shot] Subject S{sub_idx} Accuracy: {acc*100:.1f}%")
        accuracies.append(acc)
        
    if accuracies:
        mean_acc = np.mean(accuracies) * 100
        logger.info("=" * 60)
        logger.info(f"FINAL ZERO-SHOT KUL ACCURACY: {mean_acc:.1f}%")
        logger.info("=" * 60)
    else:
        logger.error("No subjects evaluated.")

if __name__ == "__main__":
    main()

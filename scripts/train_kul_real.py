"""
scripts/train_kul_real.py

Trains the Global CI Foundation model (Mesgarani CRN Adapter) on the 
actual KU Leuven dataset, bypassing the vocoder step since the dataset 
already provides the 64 Hz envelopes.
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

from neurophile.models import MesgaraniAdapter, GlobalCITrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("train_kul_real")

def load_kul_subject(mat_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a KUL .mat file and convert it into (eeg, envelope, labels).
    
    Creates 2 samples per trial:
    - Sample 1: EEG + wavA. Label = 1 if attended, 0 if unattended
    - Sample 2: EEG + wavB. Label = 1 if attended, 0 if unattended
    """
    logger.info("Loading %s...", mat_file)
    mat = sio.loadmat(str(mat_file))
    d = mat['data'][0,0]
    
    eeg_raw = d['eeg'][0]
    wavA_raw = d['wavA'][0]
    wavB_raw = d['wavB'][0]
    events = d['event'][0] # shape (60,) array of structs
    
    eeg_list = []
    env_list = []
    label_list = []
    
    n_trials = len(eeg_raw)
    for i in range(n_trials):
        # Extract EEG and keep only first 64 channels
        eeg_t = eeg_raw[i][:, :64].astype("float32") # shape (T, 64)
        envA_t = wavA_raw[i].reshape(-1, 1).astype("float32") # shape (T, 1)
        envB_t = wavB_raw[i].reshape(-1, 1).astype("float32") # shape (T, 1)
        
        # Determine which stream was attended based on event value
        # MATLAB FieldTrip struct: ev[1] = value field, nested uint8 array
        try:
            val = int(events['eeg'][0][i][1][0][0][0][0])
            attended_val = val
        except Exception as e:
            logger.warning("Trial %d: could not parse event value (%s), assuming 1", i, e)
            attended_val = 1
        
        # Create Sample A
        eeg_list.append(eeg_t)
        env_list.append(envA_t)
        label_list.append(1.0 if attended_val == 1 else 0.0)
        
        # Create Sample B
        eeg_list.append(eeg_t)
        env_list.append(envB_t)
        label_list.append(1.0 if attended_val == 2 else 0.0)
        
    eeg_arr = np.stack(eeg_list)
    env_arr = np.stack(env_list)
    label_arr = np.array(label_list, dtype="float32")
    
    return eeg_arr, env_arr, label_arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Subject 1 for subject-dependent training
    # (Training across all 18 subjects simultaneously requires advanced transfer learning
    # because every human's brain folding and skull thickness is unique!)
    mat_path = Path("data/raw/kul/DATA_preproc.zip.unzip/S1_data_preproc.mat")
    eeg_arr, env_arr, label_arr = load_kul_subject(mat_path)
    
    n_trials = len(eeg_arr)
    logger.info("Total samples loaded: %d", n_trials)
    
    split_idx = int(n_trials * (1 - args.test_split))
    
    train_eeg, test_eeg = eeg_arr[:split_idx], eeg_arr[split_idx:]
    train_env, test_env = env_arr[:split_idx], env_arr[split_idx:]
    train_label, test_label = label_arr[:split_idx], label_arr[split_idx:]
    
    logger.info("Train samples: %d | Test samples: %d", len(train_eeg), len(test_eeg))
    
    # Initialize adapter and trainer
    model = MesgaraniAdapter(num_eeg_channels=64, audio_sampling_rate=64)
    trainer = GlobalCITrainer(
        model=model,
        epochs=args.epochs,
        batch_size=8,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Train
    logger.info("Starting training on device: %s", args.device)
    t0 = time.time()
    trainer.train(
        train_eeg, train_env, train_label,
        val_eeg=test_eeg, val_env=test_env, val_lbl=test_label
    )
    logger.info("Training complete in %.1f seconds.", time.time() - t0)
    
    # Evaluate
    logger.info("Evaluating on Train set...")
    train_metrics = trainer.evaluate(train_eeg, train_env, train_label)
    logger.info("Train-set metrics: %s", train_metrics)
    
    logger.info("Evaluating on Test set...")
    test_metrics = trainer.evaluate(test_eeg, test_env, test_label)
    logger.info("Test-set metrics: %s", test_metrics)

if __name__ == "__main__":
    main()

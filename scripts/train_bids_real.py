"""
scripts/train_bids_real.py

Trains the Global CI Foundation model natively on OpenNeuro BIDS datasets
(like ds003516) stored on an external drive.
Supports optional inline EEGLAB ICA artifact rejection (Option B) via
the --use-eeglab flag, which fires a silent MATLAB subprocess per subject.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from mne_bids import BIDSPath, read_raw_bids
except ImportError:
    print("mne-bids is required. Please install it via: pip install mne-bids")
    sys.exit(1)

from neurophile.models import MesgaraniAdapter, ZionGolumbicAdapter, GlobalCITrainer
from neurophile.preprocessing.ci_artifact.pipeline import CIArtifactPipeline, CIArtifactConfig
from neurophile.preprocessing.ci_artifact.ica_cancellation import ICACancellationConfig

# Import the MATLAB EEGLAB CLI bridge
sys.path.append(str(Path(__file__).resolve().parent))
from eeglab_bridge import run_eeglab_via_cli

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("train_bids_real")

def load_bids_data(
    bids_root: Path,
    subject: str,
    enable_ica: bool = False,
    use_eeglab: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads EEG and Audio from an OpenNeuro BIDS dataset and creates true binary classification pairs."""
    logger.info("Scanning BIDS root: %s for subject %s", bids_root, subject)
    
    import pandas as pd
    import scipy.io as sio

    # 1. Parse participants.tsv
    participants_file = bids_root / "participants.tsv"
    if not participants_file.exists():
        raise FileNotFoundError(f"Missing {participants_file}")
    
    df_part = pd.read_csv(participants_file, sep='\t')
    subject_row = df_part[df_part['participant_id'] == f"sub-{subject}"]
    if len(subject_row) == 0:
        raise ValueError(f"Subject {subject} not found in participants.tsv")
    
    attended_ch = subject_row.iloc[0]['attended_ch'].lower()
    logger.info("Subject %s attended ear: %s", subject, attended_ch)
    
    # 2. Parse events.tsv
    events_file = bids_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-AttendedSpeakerParadigmOwnName_events.tsv"
    if not events_file.exists():
        raise FileNotFoundError(f"Missing events file: {events_file}")
        
    df_events = pd.read_csv(events_file, sep='\t')
    start_triggers = df_events[df_events['value'] == 'StartTrigger']['onset'].values
    logger.info("Found %d StartTrigger events (blocks).", len(start_triggers))
    
    # 3. Load EEG
    eeg_dir = bids_root / f"sub-{subject}" / "eeg"
    set_files = list(eeg_dir.glob("*_eeg.set"))
    if not set_files:
        raise ValueError(f"No .set EEG data found for subject {subject} in {eeg_dir}")
    raw_set_file = set_files[0]
    
    if use_eeglab:
        cleaned_set_file = eeg_dir / f"sub-{subject}_eeg_cleaned.set"
        if cleaned_set_file.exists():
            logger.info("Found cached EEGLAB-cleaned file: %s", cleaned_set_file)
        else:
            logger.info("Running EEGLAB ICA silently via MATLAB for subject %s...", subject)
            t_eeg = time.time()
            success = run_eeglab_via_cli(raw_set_file, cleaned_set_file)
            if success:
                logger.info("EEGLAB ICA complete in %.1f seconds!", time.time() - t_eeg)
            else:
                logger.warning("EEGLAB ICA failed! Falling back to raw data for subject %s.", subject)
                cleaned_set_file = raw_set_file
        import mne
        raw = mne.io.read_raw_eeglab(str(cleaned_set_file), preload=True, verbose=False)
    else:
        bids_path = BIDSPath(
            subject=subject,
            task="AttendedSpeakerParadigmOwnName",
            datatype="eeg",
            suffix="eeg",
            extension=".set",
            root=bids_root
        )
        bids_path.update(check=False)
        raw = read_raw_bids(bids_path, verbose=False)
        raw.load_data()
        
    # Standard AAD Preprocessing: Bandpass filter (1-8 Hz) and Downsample (64 Hz)
    logger.info("Applying bandpass filter (1-8 Hz)...")
    raw.filter(l_freq=1.0, h_freq=8.0, verbose=False)
    
    orig_sfreq = raw.info['sfreq']
    
    logger.info("Downsampling to 64 Hz...")
    raw.resample(64.0, verbose=False)
        
    eeg_picks = raw.copy().pick_types(eeg=True).get_data()
    eeg_data_full = eeg_picks.T.astype("float32")
    
    if eeg_data_full.shape[1] > 64:
        eeg_data_full = eeg_data_full[:, :64]
    elif eeg_data_full.shape[1] < 64:
        pad_width = 64 - eeg_data_full.shape[1]
        eeg_data_full = np.pad(eeg_data_full, ((0,0), (0, pad_width)))
        
    nan_count = np.isnan(eeg_data_full).sum()
    if nan_count > 0:
        logger.warning("Scrubbing %d NaN values...", nan_count)
        np.nan_to_num(eeg_data_full, copy=False, nan=0.0)
        
    if enable_ica:
        logger.info("Applying CIArtifactPipeline...")
        config = CIArtifactConfig(stage3_enabled=True, stage3=ICACancellationConfig(kurtosis_threshold=5.0))
        pipeline = CIArtifactPipeline(fs=raw.info['sfreq'], config=config)
        eeg_data_full = pipeline.run(eeg_data_full)
        
    # Standardize (Z-score) the EEG data for the neural network!
    eeg_data_full = (eeg_data_full - np.mean(eeg_data_full, axis=0)) / (np.std(eeg_data_full, axis=0) + 1e-8)
        
    sfreq = raw.info['sfreq']
    block_duration_s = 600.0  # 10 minutes per block
    samples_per_block = int(block_duration_s * sfreq)
    
    eeg_trials = []
    env_trials = []
    labels = []
    
    window_t = 512
    
    # 4. Extract blocks and stimuli
    for k, onset in enumerate(start_triggers):
        block_idx = k + 1
        logger.info("Processing Block %d starting at %.1fs", block_idx, onset)
        
        start_sample = int(onset * sfreq)
        end_sample = start_sample + samples_per_block
        
        if end_sample > len(eeg_data_full):
            logger.warning("Block %d exceeds EEG recording length. Truncating.", block_idx)
            end_sample = len(eeg_data_full)
            
        eeg_block = eeg_data_full[start_sample:end_sample]
        
        sig1_file = bids_root / "stimuli" / f"sig1_{block_idx}.mat"
        sig2_file = bids_root / "stimuli" / f"sig2_{block_idx}.mat"
        
        if not sig1_file.exists() or not sig2_file.exists():
            logger.warning("Missing stimuli files for block %d. Skipping block.", block_idx)
            continue
            
        if attended_ch == "left":
            attended_file, unattended_file = sig1_file, sig2_file
        else:
            attended_file, unattended_file = sig2_file, sig1_file
            
        import scipy.signal
        def _load_env(filepath):
            mat = sio.loadmat(filepath)
            key = [k for k in mat.keys() if not k.startswith("__")][0]
            data = mat[key].astype("float32").reshape(-1)
            
            # Resample audio accurately based on original sfreq vs target 64 Hz
            target_fs = 64.0
            new_len = int(len(data) * target_fs / orig_sfreq)
            data = scipy.signal.resample(data, new_len)
            data = data.reshape(-1, 1).astype("float32")
            
            # Standardize (Z-score) the audio envelope!
            data = (data - np.mean(data)) / (np.std(data) + 1e-8)
            return data
            
        att_env = _load_env(attended_file)
        unatt_env = _load_env(unattended_file)
        
        # Ensure exact sample-by-sample temporal alignment by trimming to the shortest length
        min_len = min(len(eeg_block), len(att_env), len(unatt_env))
        eeg_block_aligned = eeg_block[:min_len]
        att_env_aligned = att_env[:min_len]
        unatt_env_aligned = unatt_env[:min_len]
        
        n_windows = min_len // window_t
        for i in range(n_windows):
            start = i * window_t
            end = start + window_t
            
            # Attended pair (Label = 1.0)
            eeg_trials.append(eeg_block_aligned[start:end])
            env_trials.append(att_env_aligned[start:end])
            labels.append(1.0)
            
            # Unattended pair (Label = 0.0)
            eeg_trials.append(eeg_block_aligned[start:end])
            env_trials.append(unatt_env_aligned[start:end])
            labels.append(0.0)
            
    if not eeg_trials:
        raise ValueError("No valid trials were generated!")
        
    return np.stack(eeg_trials), np.stack(env_trials), np.array(labels, dtype="float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, required=True, help="Path to the downloaded OpenNeuro dataset")
    parser.add_argument("--subject", type=str, default="001", help="Subject ID or 'all' to train on entire dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, choices=["mesgarani", "zion_golumbic"], default="mesgarani", help="Deep learning architecture to use")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--enable-ica", action="store_true", help="Enable Python CIArtifactPipeline ICA")
    parser.add_argument("--use-eeglab", action="store_true", help="Run MATLAB EEGLAB runica ICA inline before each subject (slower but more accurate)")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.subject.lower() == "all":
        # ds003516 has 25 subjects
        subjects = [f"{i:03d}" for i in range(1, 26)]
        logger.info("Sequential Deep Learning mode activated. Will iterate over %d subjects.", len(subjects))
    else:
        subjects = [args.subject]
        
    # Initialize the global model ONCE
    if args.model == "mesgarani":
        model = MesgaraniAdapter(num_eeg_channels=64, audio_sampling_rate=64)
    else:
        model = ZionGolumbicAdapter(num_eeg_channels=64, audio_sampling_rate=64)
        
    trainer = GlobalCITrainer(
        model=model,
        lr=3e-4,
        epochs=args.epochs,
        batch_size=32,
        device=args.device,
        output_dir=args.output_dir
    )
    
    t_global = time.time()
    
    all_train_eeg, all_train_env, all_train_lbl = [], [], []
    all_test_eeg, all_test_env, all_test_lbl = [], [], []
    
    for sub in subjects:
        logger.info("="*50)
        logger.info("Loading data for Subject %s...", sub)
        
        try:
            eeg, env, label = load_bids_data(
                args.bids_root, sub,
                enable_ica=args.enable_ica,
                use_eeglab=args.use_eeglab
            )
        except Exception as e:
            logger.error("Skipping subject %s due to error: %s", sub, e)
            continue
        
        n_trials = len(eeg)
        split_idx = int(n_trials * (1 - args.test_split))
        if split_idx == 0: split_idx = 1
        
        all_train_eeg.append(eeg[:split_idx])
        all_train_env.append(env[:split_idx])
        all_train_lbl.append(label[:split_idx])
        
        all_test_eeg.append(eeg[split_idx:])
        all_test_env.append(env[split_idx:])
        all_test_lbl.append(label[split_idx:])
        
    if not all_train_eeg:
        logger.error("No valid subjects loaded!")
        return
        
    train_eeg = np.concatenate(all_train_eeg, axis=0)
    train_env = np.concatenate(all_train_env, axis=0)
    train_label = np.concatenate(all_train_lbl, axis=0)
    
    logger.info("="*50)
    logger.info("Pooled Train samples: %d", len(train_eeg))
    
    t0 = time.time()
    logger.info("Starting global model training across all accumulated subjects...")
    trainer.train(train_eeg, train_env, train_label)
    logger.info("Global model training completed in %.1f seconds!", time.time() - t0)
    logger.info("Total script time: %.1f seconds", time.time() - t_global)

if __name__ == "__main__":
    main()

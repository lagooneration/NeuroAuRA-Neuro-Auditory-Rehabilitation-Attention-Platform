"""
scripts/train_global_ci_model.py
=================================
Master training script for the Global CI Foundation Model.

Implements the 6-step mathematical pipeline from the Neurophile blueprint:

  Step 1 — Ingest baseline audio from downloaded datasets
  Step 2 — Vocode to CI: CIVocoderSimulator converts NH audio → CI simulation
  Step 3 — Extract low-frequency envelope (0.5–8 Hz) from the CI audio
  Step 4 — Clean EEG: CIArtifactPipeline removes electrical stimulation spikes
  Step 5 — Feed (clean EEG, CI envelope) into the selected adapter model
  Step 6 — Compute loss (Pearson correlation or binary CE); backpropagate

The resulting checkpoint is a "Global CI Foundation Model" — biologically
primed for CI acoustics. It can be pushed to clinical edge devices as the
starting point for Federated Learning fine-tuning (Phase 4).

Usage
-----
Synthetic data (smoke test, no downloads needed):
    python scripts/train_global_ci_model.py --synthetic --epochs 5

Real KUL dataset:
    python scripts/train_global_ci_model.py \\
        --eeg-dir data/raw/kul \\
        --audio-dir data/raw/audio \\
        --model kul --epochs 50 --output-dir checkpoints/

Full CI pipeline with Stage 3 ICA:
    python scripts/train_global_ci_model.py \\
        --synthetic --epochs 10 --ci-rate 900 --enable-ica \\
        --model mesgarani
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── Neurophile imports ─────────────────────────────────────────────────────────
try:
    from neurophile.stimulus.ci_vocoder import CIVocoderSimulator
    from neurophile.preprocessing.ci_artifact.pipeline import (
        CIArtifactPipeline,
        CIArtifactConfig,
    )
    from neurophile.preprocessing.ci_artifact.ica_cancellation import (
        ICACancellationConfig,
    )
except ImportError as exc:
    print(f"Error importing neurophile: {exc}")
    print("Run: pip install -e 'p:/auditory/aad[dl]'")
    sys.exit(1)

try:
    from neurophile.models import KULAdapter, MesgaraniAdapter, GlobalCITrainer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_global_ci")

# ── Constants ─────────────────────────────────────────────────────────────────
FS_AUDIO = 44100   # Hz — standard audio sampling rate
FS_EEG = 64        # Hz — EEG sampling rate after downsampling
N_EEG_CHANNELS = 64


# ── Step helpers ──────────────────────────────────────────────────────────────

def step1_ingest_audio(audio_dir: Path | None, n_synthetic: int) -> list[np.ndarray]:
    """Step 1: Ingest baseline audio tracks.

    For real datasets, reads .wav/.flac files from ``audio_dir``.
    Falls back to synthetic Gaussian noise (valid CI vocoding target).
    """
    if audio_dir and audio_dir.exists():
        audio_files = sorted(audio_dir.glob("*.wav")) + sorted(audio_dir.glob("*.flac"))
        if audio_files:
            logger.info("Step 1: loading %d audio files from %s", len(audio_files), audio_dir)
            try:
                import soundfile as sf
                tracks = []
                for af in audio_files[:n_synthetic]:
                    audio, fs = sf.read(str(af), dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    # Resample to FS_AUDIO if needed
                    if fs != FS_AUDIO:
                        from scipy.signal import resample_poly
                        from math import gcd
                        g = gcd(int(FS_AUDIO), int(fs))
                        audio = resample_poly(audio, FS_AUDIO // g, fs // g).astype("float32")
                    tracks.append(audio)
                return tracks
            except ImportError:
                logger.warning("soundfile not installed — falling back to synthetic audio")

    # Synthetic: 60-second white-noise bursts (valid spectral content for vocoding)
    logger.info("Step 1: generating %d synthetic audio tracks (60s each)", n_synthetic)
    rng = np.random.default_rng(42)
    duration_s = 60
    return [rng.standard_normal(FS_AUDIO * duration_s).astype("float32")
            for _ in range(n_synthetic)]


def step2_vocode(
    audio_tracks: list[np.ndarray],
    n_channels: int,
    ci_rate_pps: float,
) -> list[np.ndarray]:
    """Step 2: Vocode each audio track to CI simulation."""
    logger.info(
        "Step 2: vocoding %d tracks → %d-channel CI simulation (rate=%.0f pps)",
        len(audio_tracks), n_channels, ci_rate_pps,
    )
    vocoder = CIVocoderSimulator(
        fs=FS_AUDIO,
        n_channels=n_channels,
        carrier="noise",
    )
    return [vocoder.simulate(track) for track in audio_tracks]


def step3_extract_envelope(ci_tracks: list[np.ndarray]) -> list[np.ndarray]:
    """Step 3: Extract low-frequency CI envelope (0.5–8 Hz, downsampled to EEG rate)."""
    logger.info("Step 3: extracting CI envelopes → %d Hz", FS_EEG)
    vocoder = CIVocoderSimulator(fs=FS_AUDIO)
    envelopes = []
    for track in ci_tracks:
        _, env = vocoder.simulate_and_extract_envelope(track, fs_eeg=FS_EEG)
        envelopes.append(env)
    return envelopes


def step4_clean_eeg(
    eeg_trials: list[np.ndarray],
    fs: int,
    enable_ica: bool,
    ci_rate_pps: float,
) -> list[np.ndarray]:
    """Step 4: Clean EEG via CI artifact pipeline."""
    stage3_config = None
    if enable_ica:
        stage3_config = ICACancellationConfig(
            ci_rate_pps=ci_rate_pps,
            kurtosis_threshold=5.0,
        )

    config = CIArtifactConfig(
        stage3_enabled=enable_ica,
        stage3=stage3_config or ICACancellationConfig(),
    )
    pipeline = CIArtifactPipeline(fs=fs, config=config)

    logger.info(
        "Step 4: cleaning %d EEG trials (CI pipeline, ICA=%s)",
        len(eeg_trials), enable_ica,
    )
    return [pipeline.run(trial) for trial in eeg_trials]


def step5_6_train(
    clean_eeg: list[np.ndarray],
    ci_envelopes: list[np.ndarray],
    model_name: str,
    epochs: int,
    output_dir: Path,
    device: str,
) -> None:
    """Steps 5–6: Feed data into adapter model and train."""
    if not _TORCH_AVAILABLE:
        logger.error(
            "PyTorch not installed. Install with: pip install 'neurophile[dl]'\n"
            "For sklearn LinearDecoder, use the neurophile CLI instead."
        )
        sys.exit(1)

    # ── Select adapter ────────────────────────────────────────────────────────
    adapter_map = {"kul": KULAdapter, "mesgarani": MesgaraniAdapter}
    if model_name not in adapter_map:
        logger.error("Unknown model: %s. Choose from: %s", model_name, list(adapter_map))
        sys.exit(1)

    AdapterClass = adapter_map[model_name]
    model = AdapterClass(num_eeg_channels=N_EEG_CHANNELS, audio_sampling_rate=FS_EEG)
    logger.info("Step 5: adapter = %r", model)

    # ── Prepare arrays ────────────────────────────────────────────────────────
    # Align EEG and envelope lengths (EEG may be longer due to template subtraction)
    min_len = min(
        min(e.shape[0] for e in clean_eeg),
        min(len(v) for v in ci_envelopes),
    )
    # Window into non-overlapping trials
    window_t = min(512, min_len)  # 512 EEG samples ≈ 8s at 64 Hz
    n_trials = len(clean_eeg)

    eeg_array = np.stack([e[:window_t] for e in clean_eeg]).astype("float32")   # (N, T, C)
    env_array = np.stack([
        v[:window_t].reshape(window_t, 1) for v in ci_envelopes
    ]).astype("float32")                                                          # (N, T, 1)
    # Synthetic labels: alternating attended/unattended for balanced training
    label_array = np.array([i % 2 for i in range(n_trials)], dtype="float32")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = GlobalCITrainer(
        model=model,
        epochs=epochs,
        batch_size=min(8, n_trials),
        loss_mode="classification",
        device=device,
        output_dir=output_dir,
    )

    logger.info("Step 6: training global CI model …")
    t0 = time.time()
    history = trainer.train(eeg_array, env_array, label_array)
    elapsed = time.time() - t0

    logger.info(
        "Training complete in %.1fs | final_loss=%.5f",
        elapsed, history[-1] if history else float("nan"),
    )

    # ── Evaluate on training data (smoke test) ────────────────────────────────
    metrics = trainer.evaluate(eeg_array, env_array, label_array)
    logger.info("Train-set metrics: %s", metrics)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Neurophile Global CI Foundation Model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (no dataset downloads required). Ideal for smoke testing.",
    )
    parser.add_argument(
        "--eeg-dir", type=Path, default=None,
        help="Directory containing preprocessed EEG numpy arrays (.npy).",
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=None,
        help="Directory containing audio files (.wav or .flac).",
    )
    parser.add_argument(
        "--model", choices=["kul", "mesgarani"], default="kul",
        help="AAD adapter model to train (default: kul).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=32,
        help="Number of trials for synthetic data (default: 32).",
    )
    parser.add_argument(
        "--ci-channels", type=int, default=16,
        help="Number of CI vocoder channels (default: 16).",
    )
    parser.add_argument(
        "--ci-rate", type=float, default=900.0,
        help="CI stimulation rate in pulses/second (default: 900).",
    )
    parser.add_argument(
        "--enable-ica", action="store_true",
        help="Enable ICA-based CI artifact cancellation (Stage 3).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device string (default: cpu).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./checkpoints"),
        help="Output directory for checkpoints (default: ./checkpoints).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Neurophile — Global CI Foundation Model Training")
    logger.info("  Model: %s | Epochs: %d | Device: %s", args.model, args.epochs, args.device)
    logger.info("=" * 60)

    # ── Step 1: Ingest audio ─────────────────────────────────────────────────
    audio_dir = None if args.synthetic else args.audio_dir
    audio_tracks = step1_ingest_audio(audio_dir, n_synthetic=args.n_trials)

    # ── Step 2: Vocode to CI ─────────────────────────────────────────────────
    ci_tracks = step2_vocode(audio_tracks, n_channels=args.ci_channels, ci_rate_pps=args.ci_rate)

    # ── Step 3: Extract CI envelope ──────────────────────────────────────────
    ci_envelopes = step3_extract_envelope(ci_tracks)

    # ── Step 4: Generate or load EEG ─────────────────────────────────────────
    if args.synthetic or not (args.eeg_dir and args.eeg_dir.exists()):
        logger.info("Generating synthetic EEG (%d trials, %d channels) …", args.n_trials, N_EEG_CHANNELS)
        rng = np.random.default_rng(0)
        env_len = len(ci_envelopes[0])
        # EEG is sampled at FS_EEG; scale to same length as envelope
        raw_eeg_trials = [
            rng.standard_normal((env_len, N_EEG_CHANNELS)).astype("float32")
            for _ in range(args.n_trials)
        ]
    else:
        npy_files = sorted(args.eeg_dir.glob("*.npy"))[:args.n_trials]
        logger.info("Loading %d EEG trials from %s", len(npy_files), args.eeg_dir)
        raw_eeg_trials = [np.load(str(f)) for f in npy_files]

    # ── Step 4 (cont.): Clean EEG ────────────────────────────────────────────
    clean_eeg = step4_clean_eeg(
        raw_eeg_trials,
        fs=FS_EEG,
        enable_ica=args.enable_ica,
        ci_rate_pps=args.ci_rate,
    )

    # ── Steps 5+6: Train ─────────────────────────────────────────────────────
    step5_6_train(
        clean_eeg=clean_eeg,
        ci_envelopes=ci_envelopes,
        model_name=args.model,
        epochs=args.epochs,
        output_dir=args.output_dir,
        device=args.device,
    )

    logger.info("Done. Checkpoint saved in %s", args.output_dir)


if __name__ == "__main__":
    main()

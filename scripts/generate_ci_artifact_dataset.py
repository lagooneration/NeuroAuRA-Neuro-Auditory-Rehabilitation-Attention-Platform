import argparse
import sys
import time
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from neurophile.preprocessing.ci_artifact.pipeline import CIArtifactPipeline, CIArtifactConfig
from neurophile.preprocessing.ci_artifact.template_subtraction import TemplateSubtractionConfig
from neurophile.preprocessing.ci_artifact.ica_cancellation import ICACancellationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_ci_dataset")

def generate_brain_data(n_channels: int, duration_sec: float, fs: int) -> np.ndarray:
    """Generate multi-channel synthetic brain data (1000 Hz)."""
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    data = np.zeros((len(t), n_channels))
    
    # Add random brain-like waves (alpha 8-12Hz, beta 15-30Hz)
    for c in range(n_channels):
        # Base alpha wave
        data[:, c] += np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi) * np.random.uniform(2.0, 5.0)
        # Base beta wave
        data[:, c] += np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi) * np.random.uniform(1.0, 3.0)
        # Low frequency drift
        data[:, c] += np.sin(2 * np.pi * 1 * t + np.random.rand() * 2 * np.pi) * np.random.uniform(3.0, 8.0)
        # Random noise floor
        data[:, c] += np.random.normal(0, 1.5, len(t))
        
    return data.astype("float32")

def inject_ci_artifacts(clean_data: np.ndarray, fs: int, ci_rate: int) -> np.ndarray:
    """Inject 900pps huge electrical spikes across all channels."""
    corrupted = clean_data.copy()
    
    # Calculate indices where pulses should occur
    pulse_spacing = fs / ci_rate
    pulse_indices = [int(round(i * pulse_spacing)) for i in range(int(clean_data.shape[0] / pulse_spacing))]
    
    # Filter out indices out of bounds
    pulse_indices = [idx for idx in pulse_indices if idx < clean_data.shape[0]]
    
    for idx in pulse_indices:
        # A massive electrical spike (constant 65uV, reflecting deterministic CI physics)
        corrupted[idx, :] += 65.0
        
    return corrupted

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic CI-corrupted dataset and clean it."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("F:/neurophile_data/CIArtifactData"),
                        help="Output directory (default: F:/neurophile_data/CIArtifactData)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Duration of the dataset in seconds (default: 60s)")
    parser.add_argument("--channels", type=int, default=64,
                        help="Number of EEG channels (default: 64)")
    parser.add_argument("--fs", type=int, default=10000,
                        help="Sampling rate in Hz (default: 10000)")
    parser.add_argument("--ci-rate", type=int, default=900,
                        help="Cochlear implant pulse rate in pps (default: 900)")
    
    args = parser.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("1. Generating %.1f seconds of Ground Truth Brain Data (fs=%d Hz, %d channels)...", 
                args.duration, args.fs, args.channels)
    t_start = time.time()
    ground_truth = generate_brain_data(args.channels, args.duration, args.fs)
    np.save(args.out_dir / "ground_truth_eeg.npy", ground_truth)
    logger.info("   -> Saved ground_truth_eeg.npy (%.2f MB) in %.1fs", 
                ground_truth.nbytes / 1e6, time.time() - t_start)
    
    logger.info("2. Injecting %d pps Cochlear Implant Artifacts...", args.ci_rate)
    t_start = time.time()
    corrupted = inject_ci_artifacts(ground_truth, args.fs, args.ci_rate)
    np.save(args.out_dir / "corrupted_eeg.npy", corrupted)
    logger.info("   -> Saved corrupted_eeg.npy (%.2f MB) in %.1fs", 
                corrupted.nbytes / 1e6, time.time() - t_start)
    
    logger.info("3. Running CIArtifactPipeline to scrub the electrical corruption...")
    t_start = time.time()
    # Configure pipeline: Stage 1 template subtraction, Stage 3 ICA (optimized)
    config = CIArtifactConfig(
        stage1=TemplateSubtractionConfig(epoch_post_ms=2.0),
        stage3_enabled=True,
        stage3=ICACancellationConfig(ci_rate_pps=args.ci_rate)
    )
    pipeline = CIArtifactPipeline(fs=args.fs, config=config)
    cleaned = pipeline.run(corrupted)
    
    np.save(args.out_dir / "cleaned_eeg.npy", cleaned)
    logger.info("   -> Saved cleaned_eeg.npy (%.2f MB) in %.1fs", 
                cleaned.nbytes / 1e6, time.time() - t_start)
    
    # 4. Generate Validation Plot
    logger.info("4. Generating visual verification plot...")
    plt.figure(figsize=(14, 8))
    # Zoom in to just 0.1 seconds so we can see the individual 900pps needle spikes!
    plot_sec = 0.1
    t_plot = np.linspace(0, plot_sec, int(args.fs * plot_sec), endpoint=False) 
    
    plt.subplot(3, 1, 1)
    plt.plot(t_plot, ground_truth[:len(t_plot), 0], color="black")
    plt.title("Ground Truth Brain Waves (Channel 0)")
    plt.ylabel("Amplitude (uV)")
    
    plt.subplot(3, 1, 2)
    plt.plot(t_plot, corrupted[:len(t_plot), 0], color="salmon", alpha=0.8)
    plt.title(f"Corrupted EEG (Injected with {args.ci_rate} pps distinct electrical spikes)")
    plt.ylabel("Amplitude (uV)")
    
    plt.subplot(3, 1, 3)
    # Mean-center the cleaned output just for visual comparison
    cleaned_centered = cleaned[:len(t_plot), 0] - np.mean(cleaned[:len(t_plot), 0])
    plt.plot(t_plot, cleaned_centered, color="forestgreen", label="Cleaned")
    plt.plot(t_plot, ground_truth[:len(t_plot), 0], color="black", linestyle="--", alpha=0.5, label="True Brain")
    plt.title("Cleaned EEG (Output of CI Artifact Pipeline)")
    plt.ylabel("Amplitude (uV)")
    plt.xlabel("Time (seconds)")
    plt.legend()
    
    plt.tight_layout()
    plot_path = args.out_dir / "artifact_rejection_validation.png"
    plt.savefig(plot_path, dpi=150)
    logger.info("   -> Saved visual validation to %s", plot_path)
    logger.info("Dataset generation complete! All files saved to %s", args.out_dir)

if __name__ == "__main__":
    main()

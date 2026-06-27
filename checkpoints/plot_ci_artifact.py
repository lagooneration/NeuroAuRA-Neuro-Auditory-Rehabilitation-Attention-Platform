import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from neurophile.preprocessing.ci_artifact.pipeline import CIArtifactPipeline, CIArtifactConfig
from neurophile.preprocessing.ci_artifact.template_subtraction import TemplateSubtractionConfig
from neurophile.preprocessing.ci_artifact.ica_cancellation import ICACancellationConfig

# 1. Generate clean "brain" signal (1000 Hz sample rate)
fs = 1000
duration = 1.0
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
# 10 Hz alpha wave
clean_eeg = np.sin(2 * np.pi * 10 * t) * 5.0  

# 2. Generate CI electrical artifact (900 pulses per second)
ci_rate = 900
ci_artifact = np.zeros_like(t)
pulse_indices = np.arange(0, len(t), int(fs/ci_rate))
# Add large electrical spikes (much larger than brain waves)
for idx in pulse_indices:
    if idx < len(t):
        ci_artifact[idx] = 50.0 

# Combine to create raw CI EEG (simulated)
raw_eeg = clean_eeg + ci_artifact

# Convert to (samples, channels) shape for pipeline
raw_eeg_2d = raw_eeg.reshape(-1, 1)

# 3. Run Pipeline
config = CIArtifactConfig(
    stage1=TemplateSubtractionConfig(epoch_post_ms=2.0),
    stage3_enabled=True,
    stage3=ICACancellationConfig(ci_rate_pps=900)
)
pipeline = CIArtifactPipeline(fs=fs, config=config)
cleaned_eeg_2d = pipeline.run(raw_eeg_2d)

# 4. Plot
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t[:100], raw_eeg[:100], color='salmon', label="Raw EEG (with 900Hz CI Spikes)")
plt.plot(t[:100], clean_eeg[:100], color='black', linestyle='--', alpha=0.5, label="True Brain Signal")
plt.title("Before CI Artifact Rejection")
plt.ylabel("Amplitude (uV)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t[:100], cleaned_eeg_2d[:100, 0], color='forestgreen', label="Cleaned EEG (Pipeline Output)")
plt.title("After Stage 1 (Template Subtraction) + Stage 3 (ICA)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (uV)")
plt.legend()

plt.tight_layout()
plt.savefig("checkpoints/ci_artifact_validation.png", dpi=150)
print("Plot saved to checkpoints/ci_artifact_validation.png")

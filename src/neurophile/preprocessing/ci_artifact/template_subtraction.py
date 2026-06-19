"""
neurophile.preprocessing.ci_artifact.template_subtraction
==========================================================
Stage 1: Periodic CI artifact removal via running-average template subtraction.

The cochlear implant stimulates at a constant rate (e.g. 900 pps for ACE strategy).
Each stimulation pulse produces an artifact in the EEG that is:
- Time-locked to the pulse train
- Approximately identical across pulses (within a stationary window)

By averaging across many pulses (exponential moving average), we build a
template of the artifact waveform. Subtracting this template from each
epoch removes the dominant periodic artifact component.

This stage alone typically achieves 20–30 dB SNR improvement.

References
----------
- Gilley et al. (2017) "Minimization of cochlear implant stimulus artifact in
  cortical auditory evoked potentials" doi:10.1186/1744-9081-2-21
- Mc Laughlin et al. (2013) "A neural mass model to predict the EEG of
  cochlear implant users" (uses similar template logic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

logger = logging.getLogger(__name__)


@dataclass
class TemplateSubtractionConfig:
    """Configuration for Stage 1 CI artifact template subtraction."""

    enabled: bool = True
    pulse_detection_highpass_hz: float = 100.0
    """High-pass threshold to isolate the artifact (above cortical EEG band)."""
    pulse_detection_threshold_std: float = 3.0
    """Z-score threshold for pulse peak detection."""
    template_tau_pulses: int = 50
    """Exponential moving average window in pulses. Larger = more smoothing."""
    epoch_pre_ms: float = 0.5
    """Samples before each pulse onset to include in epoch."""
    epoch_post_ms: float = 2.0
    """Samples after each pulse onset to include in epoch."""
    channels_exclude: list[str] = field(default_factory=list)
    """Channel names too contaminated to salvage (will be zeroed out)."""


class CITemplateSubtraction:
    """Remove periodic CI stimulation artifact via template subtraction.

    Parameters
    ----------
    config : TemplateSubtractionConfig
        Algorithm configuration.
    fs : int
        EEG sampling rate in Hz.

    Examples
    --------
    >>> stage1 = CITemplateSubtraction(fs=1000)
    >>> clean_eeg = stage1.fit_transform(raw_eeg)  # (n_samples, n_channels)
    """

    def __init__(
        self,
        fs: int,
        config: TemplateSubtractionConfig | None = None,
    ) -> None:
        self.fs = fs
        self.config = config or TemplateSubtractionConfig()
        self._template: np.ndarray | None = None  # (n_epoch_samples, n_channels)
        self._pulse_onsets: np.ndarray | None = None

    # ── Main entry point ──────────────────────────────────────────────────────

    def fit_transform(self, eeg: np.ndarray) -> np.ndarray:
        """Detect CI pulses, build template, subtract from EEG.

        Parameters
        ----------
        eeg : np.ndarray, shape (n_samples, n_channels)
            Raw EEG containing CI stimulation artifact.

        Returns
        -------
        clean_eeg : np.ndarray, shape (n_samples, n_channels)
            EEG with Stage 1 artifact removed.
        """
        if not self.config.enabled:
            logger.debug("Stage 1 (template subtraction) is disabled — skipping.")
            return eeg.copy()

        n_samples, n_channels = eeg.shape

        # 1. Detect pulse onsets
        pulse_onsets = self._detect_pulse_onsets(eeg)
        if len(pulse_onsets) == 0:
            logger.warning(
                "No CI pulse onsets detected. Check pulse_detection_threshold_std "
                "and pulse_detection_highpass_hz settings."
            )
            return eeg.copy()

        logger.info(
            "Detected %d CI pulse onsets. Rate ≈ %.1f pps.",
            len(pulse_onsets),
            len(pulse_onsets) / (n_samples / self.fs),
        )

        # 2. Epoch EEG around each pulse
        pre = int(np.round(self.config.epoch_pre_ms * self.fs / 1000))
        post = int(np.round(self.config.epoch_post_ms * self.fs / 1000))
        epoch_len = pre + post

        # 3. Build template with exponential moving average; subtract
        clean = eeg.copy()
        template = np.zeros((epoch_len, n_channels), dtype=np.float64)
        tau = self.config.template_tau_pulses
        alpha = 1.0 / tau   # EMA decay factor

        valid_count = 0
        for onset in pulse_onsets:
            start = onset - pre
            end = onset + post
            if start < 0 or end > n_samples:
                continue

            epoch = eeg[start:end].astype(np.float64)  # (epoch_len, n_channels)
            # Update running template (EMA)
            template = (1 - alpha) * template + alpha * epoch
            # Subtract template from this epoch in the clean signal
            clean[start:end] -= template
            valid_count += 1

        self._template = template
        self._pulse_onsets = pulse_onsets
        logger.info(
            "Template subtraction complete. %d/%d valid epochs used.",
            valid_count, len(pulse_onsets),
        )
        return clean

    # ── Pulse detection ────────────────────────────────────────────────────────

    def _detect_pulse_onsets(self, eeg: np.ndarray) -> np.ndarray:
        """Detect CI stimulation pulse onsets from the most contaminated channel.

        Strategy: high-pass filter to isolate artifact energy above cortical band,
        then find peaks exceeding a z-score threshold.

        Returns
        -------
        onsets : np.ndarray, shape (n_pulses,)
            Sample indices of detected pulse onsets.
        """
        # Use the channel with the highest high-frequency energy (most contaminated)
        hf = self._highpass(eeg, self.config.pulse_detection_highpass_hz)
        energy = np.abs(hf)
        most_contaminated = int(np.argmax(energy.std(axis=0)))
        reference = energy[:, most_contaminated]

        # Z-score normalization
        z = (reference - reference.mean()) / (reference.std() + 1e-10)

        # Find peaks above threshold
        # minimum_distance: assume at least 0.1 ms between pulses (= 10 kHz max rate)
        min_distance = max(1, int(self.fs * 0.0001))
        peaks, _ = find_peaks(z, height=self.config.pulse_detection_threshold_std,
                               distance=min_distance)
        return peaks

    def _highpass(self, eeg: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """Apply a 4th-order Butterworth high-pass filter."""
        nyq = self.fs / 2.0
        normalized = cutoff_hz / nyq
        if normalized >= 1.0:
            logger.warning(
                "High-pass cutoff %.1f Hz exceeds Nyquist %.1f Hz. Skipping filter.",
                cutoff_hz, nyq,
            )
            return eeg
        b, a = butter(4, normalized, btype="high")
        return filtfilt(b, a, eeg, axis=0)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def detected_rate_pps(self) -> float | None:
        """Estimated CI stimulation rate in pulses per second."""
        if self._pulse_onsets is None or len(self._pulse_onsets) < 2:
            return None
        intervals = np.diff(self._pulse_onsets) / self.fs
        return float(1.0 / np.median(intervals))

    @property
    def template(self) -> np.ndarray | None:
        """Final artifact template, shape (epoch_len, n_channels)."""
        return self._template

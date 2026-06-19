"""
neurophile.preprocessing.ci_artifact.quality
============================================
Post-cleaning quality metrics for the CI artifact pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

_NEURAL_BAND = (1.0, 30.0)   # Hz — cortical EEG band of interest
_ARTIFACT_BAND_HIGH = 100.0  # Hz — CI artifact sits above this


class CIArtifactQuality:
    """Compute quality metrics before and after CI artifact removal."""

    @staticmethod
    def compute(
        raw_eeg: np.ndarray,
        clean_eeg: np.ndarray,
        fs: int,
    ) -> dict[str, float]:
        """Compute a standard set of post-cleaning quality metrics.

        Parameters
        ----------
        raw_eeg, clean_eeg : np.ndarray, shape (n_samples, n_channels)
        fs : int

        Returns
        -------
        metrics : dict[str, float]
            snr_improvement_db          : dB improvement in signal-to-artifact ratio
            neural_band_power_ratio     : clean/raw power in 1–30 Hz band (should be ~1)
            artifact_residual_corr      : correlation of clean signal with raw HF content
            topographic_asymmetry_index : L/R hemisphere alpha-band power asymmetry
        """
        metrics: dict[str, float] = {}

        # ── SNR improvement ───────────────────────────────────────────────────
        raw_hf_power = CIArtifactQuality._band_power(raw_eeg, fs, _ARTIFACT_BAND_HIGH, fs / 2 - 1)
        clean_hf_power = CIArtifactQuality._band_power(clean_eeg, fs, _ARTIFACT_BAND_HIGH, fs / 2 - 1)
        # SNR improvement = reduction in HF (artifact) power in dB
        if raw_hf_power > 0 and clean_hf_power > 0:
            metrics["snr_improvement_db"] = float(
                10 * np.log10(raw_hf_power / (clean_hf_power + 1e-12))
            )
        else:
            metrics["snr_improvement_db"] = float("nan")

        # ── Neural band power ratio ───────────────────────────────────────────
        raw_neural = CIArtifactQuality._band_power(raw_eeg, fs, *_NEURAL_BAND)
        clean_neural = CIArtifactQuality._band_power(clean_eeg, fs, *_NEURAL_BAND)
        metrics["neural_band_power_ratio"] = (
            clean_neural / (raw_neural + 1e-12)
        )

        # ── Artifact residual correlation ─────────────────────────────────────
        raw_hf_signal = CIArtifactQuality._bandpass(raw_eeg, fs, _ARTIFACT_BAND_HIGH, fs / 2 - 1)
        clean_neural_signal = CIArtifactQuality._bandpass(clean_eeg, fs, *_NEURAL_BAND)
        # Mean absolute correlation across channels
        corrs = []
        for ch in range(raw_eeg.shape[1]):
            r = np.corrcoef(raw_hf_signal[:, ch], clean_neural_signal[:, ch])[0, 1]
            if not np.isnan(r):
                corrs.append(abs(r))
        metrics["artifact_residual_corr"] = float(np.mean(corrs)) if corrs else float("nan")

        # ── Topographic asymmetry index ───────────────────────────────────────
        # Rough L/R split: first half of channels = left, second half = right
        n_ch = clean_eeg.shape[1]
        mid = n_ch // 2
        alpha_left = CIArtifactQuality._band_power(clean_eeg[:, :mid], fs, 8.0, 13.0)
        alpha_right = CIArtifactQuality._band_power(clean_eeg[:, mid:], fs, 8.0, 13.0)
        metrics["topographic_asymmetry_index"] = (
            alpha_left / (alpha_right + 1e-12)
        )

        return metrics

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _band_power(eeg: np.ndarray, fs: int, f_low: float, f_high: float) -> float:
        filtered = CIArtifactQuality._bandpass(eeg, fs, f_low, f_high)
        return float(np.mean(filtered ** 2))

    @staticmethod
    def _bandpass(eeg: np.ndarray, fs: int, f_low: float, f_high: float) -> np.ndarray:
        nyq = fs / 2.0
        low = np.clip(f_low / nyq, 1e-6, 0.999)
        high = np.clip(f_high / nyq, 1e-6, 0.999)
        if low >= high:
            return np.zeros_like(eeg)
        b, a = butter(4, [low, high], btype="band")
        return filtfilt(b, a, eeg, axis=0)

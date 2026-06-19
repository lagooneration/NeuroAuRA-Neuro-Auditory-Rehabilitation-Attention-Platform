"""
neurophile.preprocessing.standard
==================================
Standard offline EEG preprocessing pipeline (MNE-Python based).

Designed for normal-hearing subjects or as the pre-processing step before
the CI artifact pipeline. Uses MNE-Python throughout.

Pipeline stages (all configurable via YAML):
1. Bandpass filter (0.5–45 Hz)
2. Notch filter (50 or 60 Hz + harmonics)
3. Bad channel detection + interpolation
4. Re-reference to average reference
5. Epoch extraction around events
6. Baseline correction
7. Amplitude-threshold artifact rejection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mne
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StandardPipelineConfig:
    """Configuration for the standard EEG preprocessing pipeline."""

    bandpass_low: float = 0.5     # Hz
    bandpass_high: float = 45.0   # Hz
    notch_hz: float = 50.0        # 50 or 60 depending on country
    notch_harmonics: bool = True
    rereference: str = "average"  # "average" | "Cz" | None
    bad_channel_method: str = "ransac"  # "ransac" | "correlation" | None
    epoch_tmin: float = -0.1      # s relative to event onset
    epoch_tmax: float = 0.8       # s relative to event onset
    baseline: tuple[float | None, float] = (None, 0.0)  # baseline correction window
    amplitude_threshold_uv: float = 150.0  # reject epochs > ±threshold µV
    resample_hz: float | None = None  # resample to this rate (None = no resample)


class StandardPipeline:
    """Offline MNE-based EEG preprocessing pipeline.

    Parameters
    ----------
    config : StandardPipelineConfig
        Pipeline configuration.

    Examples
    --------
    >>> import mne
    >>> pipeline = StandardPipeline()
    >>> raw = mne.io.read_raw_fif("my_recording.fif", preload=True)
    >>> clean_raw = pipeline.preprocess_raw(raw)
    >>> epochs = pipeline.epoch(clean_raw, events=events, event_id=event_id)
    """

    def __init__(self, config: StandardPipelineConfig | None = None) -> None:
        self.config = config or StandardPipelineConfig()

    def preprocess_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply all continuous preprocessing steps to a Raw object.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Must be preloaded (``raw.load_data()``).

        Returns
        -------
        raw : mne.io.BaseRaw
            In-place modified raw object.
        """
        cfg = self.config

        if not raw.preload:
            raw.load_data()

        # 1. Bandpass filter
        logger.info(
            "Bandpass filter: %.2f–%.2f Hz", cfg.bandpass_low, cfg.bandpass_high
        )
        raw.filter(
            l_freq=cfg.bandpass_low,
            h_freq=cfg.bandpass_high,
            method="fir",
            fir_window="hamming",
            verbose=False,
        )

        # 2. Notch filter
        freqs = [cfg.notch_hz]
        if cfg.notch_harmonics:
            nyq = raw.info["sfreq"] / 2.0
            freqs = [cfg.notch_hz * k for k in range(1, 6) if cfg.notch_hz * k < nyq]
        logger.info("Notch filter: %s Hz", freqs)
        raw.notch_filter(freqs=freqs, verbose=False)

        # 3. Resample (if requested)
        if cfg.resample_hz is not None and cfg.resample_hz != raw.info["sfreq"]:
            logger.info("Resampling to %.1f Hz", cfg.resample_hz)
            raw.resample(cfg.resample_hz, verbose=False)

        # 4. Bad channel detection
        if cfg.bad_channel_method == "correlation":
            bads = self._detect_bad_channels_correlation(raw)
            raw.info["bads"].extend(bads)
            logger.info("Bad channels detected: %s", bads)

        # 5. Interpolate bad channels
        if raw.info["bads"]:
            logger.info("Interpolating bad channels: %s", raw.info["bads"])
            raw.interpolate_bads(reset_bads=True, verbose=False)

        # 6. Re-reference
        if cfg.rereference:
            logger.info("Re-referencing to: %s", cfg.rereference)
            if cfg.rereference == "average":
                raw.set_eeg_reference("average", projection=False, verbose=False)
            else:
                raw.set_eeg_reference([cfg.rereference], verbose=False)

        return raw

    def epoch(
        self,
        raw: mne.io.BaseRaw,
        events: np.ndarray,
        event_id: dict[str, int],
    ) -> mne.Epochs:
        """Extract epochs from preprocessed Raw.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Preprocessed continuous EEG.
        events : np.ndarray, shape (n_events, 3)
            MNE events array.
        event_id : dict
            Mapping of trial_type name → event code integer.

        Returns
        -------
        epochs : mne.Epochs
            Baseline-corrected, amplitude-rejected epochs.
        """
        cfg = self.config
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=cfg.epoch_tmin,
            tmax=cfg.epoch_tmax,
            baseline=cfg.baseline,
            reject={"eeg": cfg.amplitude_threshold_uv * 1e-6},
            preload=True,
            verbose=False,
        )
        logger.info(
            "Epoching: %d/%d epochs retained after amplitude rejection.",
            len(epochs), len(events),
        )
        return epochs

    # ── Bad channel detection ─────────────────────────────────────────────────

    @staticmethod
    def _detect_bad_channels_correlation(
        raw: mne.io.BaseRaw,
        threshold: float = 0.4,
    ) -> list[str]:
        """Mark channels with low correlation to their neighbours as bad.

        This is a simple heuristic; RANSAC (pyprep) is more robust but
        requires an additional dependency. Set ``bad_channel_method='ransac'``
        and install ``pyprep`` for the gold standard.
        """
        data = raw.get_data(picks="eeg")  # (n_channels, n_samples)
        corr_matrix = np.corrcoef(data)   # (n_channels, n_channels)
        np.fill_diagonal(corr_matrix, 0)
        mean_corr = corr_matrix.mean(axis=1)
        bad_idx = np.where(mean_corr < threshold)[0]
        ch_names = [raw.ch_names[i] for i in raw.pick_types(eeg=True).picks
                    if i in bad_idx]
        return ch_names

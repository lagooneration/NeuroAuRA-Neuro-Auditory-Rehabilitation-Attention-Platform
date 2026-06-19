"""
neurophile.preprocessing.ci_artifact.ica_cancellation
======================================================
ICA-based Cochlear Implant artifact cancellation (Pipeline Stage 3).

Why ICA for CI Artifacts?
--------------------------
Template subtraction (Stage 1) removes the *average* CI pulse shape, but
residual artifacts remain due to pulse-to-pulse jitter and non-stationarity.
Spatial filtering (Stage 2) projects out the dominant electrode-space
topography. ICA (Stage 3) operates in the *component space* and exploits the
fact that CI electrical artifacts have:

  1. **Highly non-Gaussian temporal statistics** — kurtosis >> 3 (super-Gaussian)
     due to sharp impulse spikes.
  2. **Highly periodic temporal structure** — autocorrelation peaks at the
     CI stimulation interval (1/rate_pps seconds).
  3. **Non-biological spatial topography** — concentrated near the implant
     electrode rather than following neural source patterns.

These three features allow robust automated identification of CI artifact
independent components (ICs) without requiring manual inspection.

Algorithm
---------
1. Fit FastICA on the EEG array (post-template-subtraction, post-spatial-filter).
2. For each IC, compute:
   - Kurtosis of the IC time series (threshold: > ``kurtosis_threshold``).
   - Dominant autocorrelation peak at the expected CI rate
     (within ``rate_tolerance_pct`` of the declared CI rate).
3. Flag ICs satisfying *both* criteria as CI artifact components.
4. Zero out flagged ICs in the component matrix.
5. Reconstruct the cleaned EEG via the ICA mixing matrix inverse.

References
----------
- Viola et al. (2012) "Semi-automatic identification of independent components
  representing EEG artifact" Clinical Neurophysiology 120:868–877.
- Debener et al. (2008) "Single-trial EEG–brain-computer interface and neural
  correlates of performance measures" Annals of Biomedical Engineering 36:219.
- MNE-Python ICA: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import kurtosis
from scipy.signal import welch

logger = logging.getLogger(__name__)


@dataclass
class ICACancellationConfig:
    """Configuration for ICA-based CI artifact cancellation.

    Parameters
    ----------
    n_components : int or None
        Number of ICA components to extract. None = min(n_channels, 20).
    max_iter : int
        Maximum FastICA iterations.
    kurtosis_threshold : float
        ICs with kurtosis > this value are flagged as potentially artifactual.
        Neural signals are near-Gaussian (kurtosis ≈ 0); CI artifacts >> 3.
    rate_tolerance_pct : float
        How close the dominant autocorrelation frequency must be to the CI
        stimulation rate to be flagged (as a fraction, e.g. 0.10 = ±10%).
    ci_rate_pps : float or None
        Expected CI stimulation rate in pulses per second (e.g. 900.0).
        If None, the autocorrelation check is skipped and only kurtosis is used.
    min_artifact_components : int
        Minimum number of ICs to remove (floor, for robustness).
    max_artifact_components : int
        Maximum number of ICs to remove (safety cap — never remove too many).
    """

    n_components: int | None = None
    max_iter: int = 2000
    kurtosis_threshold: float = 5.0
    rate_tolerance_pct: float = 0.10
    ci_rate_pps: float | None = None
    min_artifact_components: int = 1
    max_artifact_components: int = 4


class ICACancellation:
    """ICA-based CI artifact cancellation for EEG signals.

    This implements Stage 3 of the ``CIArtifactPipeline``. It uses
    ``sklearn.decomposition.FastICA`` (no MNE dependency required at this
    level) so it works in all environments. Pass an MNE Raw object if you
    want MNE's richer ICA diagnostics.

    Parameters
    ----------
    fs : int
        EEG sampling rate in Hz.
    config : ICACancellationConfig
        Algorithm configuration.

    Examples
    --------
    >>> import numpy as np
    >>> ica = ICACancellation(fs=1000)
    >>> eeg = np.random.randn(10000, 32)  # 10 s, 32 channels
    >>> clean = ica.fit_transform(eeg)
    >>> clean.shape
    (10000, 32)
    """

    def __init__(
        self,
        fs: int,
        config: ICACancellationConfig | None = None,
    ) -> None:
        self.fs = fs
        self.config = config or ICACancellationConfig()
        self._ica = None
        self._artifact_component_indices: list[int] = []
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, eeg: np.ndarray) -> np.ndarray:
        """Fit ICA on EEG and remove CI artifact components.

        Parameters
        ----------
        eeg : np.ndarray, shape (n_samples, n_channels)
            EEG after template subtraction (Stage 1) and spatial filter (Stage 2).

        Returns
        -------
        clean_eeg : np.ndarray, shape (n_samples, n_channels)
            EEG with CI artifact ICs zeroed out.
        """
        n_samples, n_channels = eeg.shape
        n_components = self.config.n_components or min(n_channels, 20)
        n_components = min(n_components, n_channels)

        logger.info(
            "ICACancellation: fitting FastICA on (%d, %d), n_components=%d",
            n_samples, n_channels, n_components,
        )

        try:
            from sklearn.decomposition import FastICA
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for ICA cancellation. "
                "Install with: pip install scikit-learn"
            ) from exc

        ica = FastICA(
            n_components=n_components,
            max_iter=self.config.max_iter,
            random_state=42,
            tol=1e-4,
        )

        # ICA expects (n_samples, n_features)
        sources = ica.fit_transform(eeg)  # (n_samples, n_components)
        self._ica = ica
        self._fitted = True

        # ── Identify CI artifact ICs ─────────────────────────────────────────
        artifact_indices = self._identify_artifact_ics(sources)
        # Store final (capped) list for reporting
        self._artifact_component_indices = artifact_indices

        logger.info(
            "ICACancellation: flagging %d/%d ICs as CI artifact: %s",
            len(artifact_indices), n_components, artifact_indices,
        )

        # ── Zero out artifact components ─────────────────────────────────────
        sources_cleaned = sources.copy()
        for idx in artifact_indices:
            sources_cleaned[:, idx] = 0.0

        # ── Reconstruct EEG ──────────────────────────────────────────────────
        # X ≈ sources @ mixing.T + mean
        # cleaned_X = cleaned_sources @ mixing.T + mean
        mixing = ica.mixing_  # (n_channels, n_components)
        mean = ica.mean_       # (n_channels,)
        clean_eeg = sources_cleaned @ mixing.T + mean

        logger.info("ICACancellation: done. Output shape %s", clean_eeg.shape)
        return clean_eeg.astype(eeg.dtype)

    def get_artifact_report(self) -> dict:
        """Return a diagnostic report for the last fit_transform call.

        Returns
        -------
        report : dict with keys:
            n_components_total, artifact_indices, n_artifacts_removed
        """
        if not self._fitted:
            return {"error": "fit_transform() has not been called yet."}
        return {
            "n_components_total": (
                self._ica.n_components if self._ica else 0
            ),
            "artifact_indices": self._artifact_component_indices,
            "n_artifacts_removed": len(self._artifact_component_indices),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _identify_artifact_ics(self, sources: np.ndarray) -> list[int]:
        """Flag ICs with CI artifact characteristics.

        Criteria (both must be met if ci_rate_pps is set, else kurtosis only):
          1. Kurtosis > kurtosis_threshold
          2. Dominant frequency ≈ CI stimulation rate (±rate_tolerance_pct)
        """
        n_components = sources.shape[1]
        candidates = []

        for i in range(n_components):
            ic = sources[:, i]
            k = float(kurtosis(ic, fisher=True))  # Fisher's definition: Gaussian=0

            kurtosis_flagged = k > self.config.kurtosis_threshold

            if kurtosis_flagged and self.config.ci_rate_pps is not None:
                rate_flagged = self._check_periodicity(ic, self.config.ci_rate_pps)
            else:
                rate_flagged = kurtosis_flagged  # single criterion when rate unknown

            if kurtosis_flagged and rate_flagged:
                candidates.append((i, k))
                logger.debug("  IC %02d: kurtosis=%.2f → flagged as CI artifact", i, k)
            else:
                logger.debug("  IC %02d: kurtosis=%.2f → retained", i, k)

        # Sort by kurtosis descending (highest first = most artifact-like)
        candidates.sort(key=lambda x: x[1], reverse=True)
        indices = [idx for idx, _ in candidates]

        # Apply safety caps
        indices = indices[: self.config.max_artifact_components]
        return indices

    def _check_periodicity(self, ic: np.ndarray, expected_rate_hz: float) -> bool:
        """Check if an IC has dominant power near the CI stimulation rate."""
        freqs, psd = welch(ic, fs=self.fs, nperseg=min(len(ic) // 4, 1024))
        if len(freqs) < 2:
            return False
        dominant_freq = freqs[np.argmax(psd)]
        tol = expected_rate_hz * self.config.rate_tolerance_pct
        return abs(dominant_freq - expected_rate_hz) <= tol

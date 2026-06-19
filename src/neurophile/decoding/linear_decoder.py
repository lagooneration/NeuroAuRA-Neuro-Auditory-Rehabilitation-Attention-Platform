"""
neurophile.decoding.linear_decoder
===================================
Linear stimulus reconstruction decoder using ridge regression.

This is the workhorse AAD decoder: fit a backward model mapping EEG → envelope,
then at test time reconstruct the envelope and compare correlation to two
competing streams.

Method
------
The backward model (Crosse et al., 2016):

    s_hat(t) = Σ_τ Σ_n  w(n, τ) · eeg(t - τ, n)

    where τ ∈ [lag_min, lag_max] ms and n indexes EEG channels.

This is equivalent to a multivariate ridge regression:
    W = (X'X + λI)^-1 X'y
    where X is the Toeplitz-structured EEG lag matrix.

Ridge regularization (α) is selected by leave-one-trial-out cross-validation.

References
----------
- Crosse et al. (2016) "The Multivariate Temporal Response Function (mTRF)
  Toolbox: A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli"
  doi:10.3389/fnhum.2016.00604
- O'Sullivan et al. (2015) "Attentional Selection in a Cocktail Party Environment
  Can Be Decoded from Single-EEG Electrodes" doi:10.1371/journal.pbio.1002259
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

from neurophile.decoding.base import BaseDecoder

logger = logging.getLogger(__name__)


class LinearDecoder(BaseDecoder):
    """Ridge-regression backward model for Auditory Attention Decoding.

    Parameters
    ----------
    lag_min_ms : float
        Minimum EEG lag relative to stimulus (ms). Typically 0 ms
        (no acausal component in backward model).
    lag_max_ms : float
        Maximum EEG lag (ms). Typically 250–500 ms to capture N1/P2 range.
    fs : int
        EEG sampling rate in Hz. Set automatically during ``fit()``.
    alphas : sequence of float
        Ridge regularization strengths to cross-validate over.
    """

    name = "linear"

    def __init__(
        self,
        lag_min_ms: float = 0.0,
        lag_max_ms: float = 250.0,
        alphas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    ) -> None:
        self.lag_min_ms = lag_min_ms
        self.lag_max_ms = lag_max_ms
        self.alphas = list(alphas)
        self.fs: int = 0
        self._model: Ridge | None = None
        self._lag_samples: np.ndarray | None = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        eeg: np.ndarray,
        envelope: np.ndarray,
        fs: int,
    ) -> "LinearDecoder":
        """Fit the backward model on training EEG + attended envelope.

        Parameters
        ----------
        eeg : np.ndarray, shape (n_samples, n_channels)
        envelope : np.ndarray, shape (n_samples,)
        fs : int
        """
        self.fs = fs
        lag_min = int(np.round(self.lag_min_ms * fs / 1000))
        lag_max = int(np.round(self.lag_max_ms * fs / 1000))
        self._lag_samples = np.arange(lag_min, lag_max + 1)

        X = self._build_lag_matrix(eeg, self._lag_samples)
        y = envelope[lag_max:]   # align: remove first lag_max samples

        logger.debug(
            "LinearDecoder fitting: X=%s, y=%s, n_alphas=%d",
            X.shape, y.shape, len(self.alphas),
        )

        model = RidgeCV(alphas=self.alphas, fit_intercept=True)
        model.fit(X, y)
        self._model = Ridge(alpha=model.alpha_, fit_intercept=True)
        self._model.fit(X, y)

        logger.info(
            "LinearDecoder fitted. Alpha=%.4g, lags=[%d, %d] samples",
            model.alpha_, lag_min, lag_max,
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        """Reconstruct the attended envelope from EEG.

        Parameters
        ----------
        eeg : np.ndarray, shape (n_samples, n_channels)

        Returns
        -------
        envelope_hat : np.ndarray, shape (n_valid_samples,)
            Length is n_samples - lag_max (edge samples are dropped).
        """
        if self._model is None or self._lag_samples is None:
            raise RuntimeError("Call fit() before predict().")
        X = self._build_lag_matrix(eeg, self._lag_samples)
        return self._model.predict(X).astype(np.float32)

    # ── Lag matrix builder ────────────────────────────────────────────────────

    @staticmethod
    def _build_lag_matrix(eeg: np.ndarray, lags: np.ndarray) -> np.ndarray:
        """Construct the Toeplitz-structured EEG lag matrix.

        Parameters
        ----------
        eeg : np.ndarray, shape (n_samples, n_channels)
        lags : np.ndarray
            Integer lag values in samples (e.g. [0, 1, 2, ..., 128]).

        Returns
        -------
        X : np.ndarray, shape (n_samples - max_lag, n_channels * n_lags)
        """
        n_samples, n_channels = eeg.shape
        lag_max = int(lags.max())
        n_valid = n_samples - lag_max
        n_features = n_channels * len(lags)
        X = np.empty((n_valid, n_features), dtype=np.float64)

        for i, lag in enumerate(lags):
            start = lag_max - lag
            X[:, i * n_channels: (i + 1) * n_channels] = eeg[start: start + n_valid]

        return X

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_lags(self) -> int:
        if self._lag_samples is None:
            return 0
        return len(self._lag_samples)

    @property
    def n_weights(self) -> int:
        if self._model is None:
            return 0
        return self._model.coef_.size

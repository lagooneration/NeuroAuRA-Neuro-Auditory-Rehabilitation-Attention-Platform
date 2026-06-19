"""
neurophile.decoding.base
========================
Abstract base class for all AAD decoders.

All decoders must:
- Accept (n_samples, n_channels) EEG and (n_samples,) envelope.
- Expose a scikit-learn-compatible fit/predict interface.
- Set a ``name`` class attribute used by the CLI.
- Be importable from neurophile.decoding.

See CONTRIBUTING.md §Adding a New Decoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDecoder(ABC):
    """Abstract AAD decoder interface.

    Subclasses implement the stimulus-reconstruction paradigm:
    given EEG, reconstruct the attended audio envelope, then compare
    the reconstruction to competing envelopes to determine attention.
    """

    #: Unique identifier, used in CLI --decoder flag and result DataFrames.
    name: str = "base"

    @abstractmethod
    def fit(
        self,
        eeg: np.ndarray,
        envelope: np.ndarray,
        fs: int,
    ) -> "BaseDecoder":
        """Fit the decoder on EEG + attended envelope.

        Parameters
        ----------
        eeg : np.ndarray, shape (n_samples, n_channels)
            Preprocessed EEG at sampling rate ``fs``.
        envelope : np.ndarray, shape (n_samples,)
            Attended stream's auditory envelope at sampling rate ``fs``.
        fs : int
            Sampling rate in Hz.

        Returns
        -------
        self : fitted decoder
        """

    @abstractmethod
    def predict(self, eeg: np.ndarray) -> np.ndarray:
        """Reconstruct the attended envelope from EEG.

        Parameters
        ----------
        eeg : np.ndarray, shape (n_samples, n_channels)

        Returns
        -------
        envelope_hat : np.ndarray, shape (n_samples,)
        """

    def score(self, eeg: np.ndarray, envelope: np.ndarray) -> float:
        """Pearson correlation between predicted and true envelope.

        Parameters
        ----------
        eeg : np.ndarray, shape (n_samples, n_channels)
        envelope : np.ndarray, shape (n_samples,)

        Returns
        -------
        r : float
            Pearson correlation coefficient.
        """
        predicted = self.predict(eeg)
        r_matrix = np.corrcoef(predicted, envelope)
        return float(r_matrix[0, 1])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

"""
neurophile.models.core.base_aad_model
=====================================
Abstract base class for all deep-learning AAD models.

Design notes
------------
This is the *deep-learning* counterpart to ``neurophile.decoding.BaseDecoder``
(which uses a scikit-learn fit/predict interface for classical models).

``BaseAADModel`` inherits from ``torch.nn.Module`` so that:
  - Weight serialization uses ``state_dict()`` / ``load_state_dict()`` — the
    standard Flower (flwr) federated learning aggregation interface.
  - The training loop is a standard PyTorch backward pass.
  - Model-parallel and mixed-precision training work out of the box.

Contract
--------
Every adapter **must** accept::

    eeg_tensor           : torch.Tensor  shape (B, T, C)
                           B = batch size, T = time steps, C = EEG channels
    audio_envelope_tensor: torch.Tensor  shape (B, T, 1)
                           Low-frequency amplitude envelope of the audio stream

Every adapter **must** return::

    output : torch.Tensor  shape (B, 1)
             Probability ∈ [0, 1] that the presented stream is *attended*.
             (0 = unattended, 1 = attended)

Conventions
-----------
- ``audio_sampling_rate`` stored as an attribute is expected to equal the
  EEG sampling rate *after envelope downsampling*, not the raw audio rate.
- Subclasses set the class attribute ``name`` (used in logging and checkpoints).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# Guard: torch is an optional dependency (neurophile[dl])
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Create a sentinel base so the module is importable without torch
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "Deep-learning models require PyTorch. "
            "Install with:  pip install 'neurophile[dl]'"
        )


class BaseAADModel(ABC):
    """Abstract deep-learning AAD model (PyTorch nn.Module subtype).

    Subclasses must also inherit ``torch.nn.Module``. The split is necessary
    so this file remains importable without torch installed.

    See ``neurophile.models.adapters`` for concrete implementations.
    """

    #: Unique name used in checkpoint filenames and log lines.
    name: str = "base"

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Defer the torch check to instantiation time, not import time.

    @abstractmethod
    def forward(
        self,
        eeg_tensor: "torch.Tensor",
        audio_envelope_tensor: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute attention probability from EEG + audio envelope.

        Parameters
        ----------
        eeg_tensor : torch.Tensor, shape (B, T, C)
            Preprocessed EEG time series. B=batch, T=time steps, C=channels.
        audio_envelope_tensor : torch.Tensor, shape (B, T, 1)
            Low-frequency amplitude envelope of the candidate audio stream.

        Returns
        -------
        logits : torch.Tensor, shape (B, 1)
            Raw logit (before sigmoid). The trainer applies ``torch.sigmoid``
            to obtain the probability of attention.
        """

    def decode(
        self,
        eeg: np.ndarray,
        envelope: np.ndarray,
        device: str = "cpu",
    ) -> float:
        """Convenience: classify attention from numpy arrays.

        Parameters
        ----------
        eeg : np.ndarray, shape (T, C)
            Single-trial EEG (no batch dimension).
        envelope : np.ndarray, shape (T,) or (T, 1)
            Corresponding audio envelope.
        device : str
            Torch device string, e.g. ``"cpu"`` or ``"cuda"``.

        Returns
        -------
        p_attended : float
            Probability ∈ [0, 1] that ``envelope`` belongs to the attended stream.
        """
        _require_torch()
        self.eval()  # type: ignore[attr-defined]
        with torch.no_grad():
            eeg_t = torch.from_numpy(eeg).float().unsqueeze(0).to(device)  # (1, T, C)
            env_t = torch.from_numpy(
                envelope.reshape(-1, 1)
            ).float().unsqueeze(0).to(device)  # (1, T, 1)
            logit = self.forward(eeg_t, env_t)
            return float(torch.sigmoid(logit).item())

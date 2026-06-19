"""
neurophile.models.adapters.mesgarani_crn_adapter
================================================
Adapter wrapping the Mesgarani Lab Cortical Response Network (CRN) for AAD.

Anti-Corruption Layer (ACL)
----------------------------
The Mesgarani Lab's neural tracking models use a convolutional recurrent
architecture. Like the KULAdapter, this adapter isolates the Neurophile core
from upstream API instability in the research codebase.

How to wire the real CRN
--------------------------
1. Run ``data_pipeline/fetchers/clone_author_repos.sh``
2. Follow the ``# TODO (implementer)`` comments below.

Fallback CRN
------------
When the external Mesgarani repo is not present, ``MesgaraniAdapter`` uses a
self-contained Conv1d+GRU architecture that approximates the CRN design
described in:

  Akbari et al. (2019) "Towards reconstructing intelligible speech from the
  human auditory cortex" — the CRN maps EEG→speech using conv+recurrent layers.

References
----------
- Di Liberto et al. (2015) "Low-frequency cortical entrainment to speech
  reflects phoneme-level processing" doi:10.1016/j.cub.2015.08.030
- Mesgarani & Chang (2012) "Selective cortical representation of attended
  speaker in multi-talker speech perception" doi:10.1038/nature11020
- https://github.com/naplab/naplib-python  (lab's public Python tools)
"""

from __future__ import annotations

import logging

from neurophile.models.core.base_aad_model import BaseAADModel, _require_torch

logger = logging.getLogger(__name__)

# ── External library import (ACL shim) ────────────────────────────────────────
_MESGARANI_EXTERNAL_AVAILABLE = False
try:
    # TODO (implementer): After running clone_author_repos.sh, update this
    # import to match the Mesgarani lab's actual class. Common locations:
    #   external_libs/naplib-python/naplib/models/crn.py
    #   external_libs/mesgarani_crn/model.py
    # Example:
    #   from external_libs.naplib.models.crn import CorticalResponseNetwork as MesgaraniCRN
    from external_libs.mesgarani_crn import MesgaraniCRN  # type: ignore[import]
    _MESGARANI_EXTERNAL_AVAILABLE = True
    logger.info("MesgaraniAdapter: using real MesgaraniCRN from external_libs.")
except ImportError:
    logger.info(
        "MesgaraniAdapter: external_libs.mesgarani_crn not found — "
        "using built-in fallback Conv+GRU. Run clone_author_repos.sh to enable the real model."
    )


# ── Fallback Conv+GRU CRN ─────────────────────────────────────────────────────

def _build_fallback_mesgarani_crn(num_eeg_channels: int) -> "torch.nn.Module":
    """Build a Conv1d + GRU network approximating the Mesgarani CRN.

    Architecture:
      - EEG encoder:  3× Conv1d → LayerNorm → GeLU
      - Envelope encoder: 2× Conv1d → LayerNorm → GeLU
      - Temporal fusion: Bidirectional GRU on concatenated features
      - Classifier: Linear(hidden, 1)
    """
    _require_torch()
    import torch
    import torch.nn as nn

    class _FallbackCRN(nn.Module):
        def __init__(self, n_ch: int, hidden: int = 64) -> None:
            super().__init__()
            # EEG encoder
            self.eeg_enc = nn.Sequential(
                nn.Conv1d(n_ch, hidden, kernel_size=5, padding=2),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=4),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
            # Envelope encoder
            self.env_enc = nn.Sequential(
                nn.Conv1d(1, hidden // 2, kernel_size=5, padding=2),
                nn.LayerNorm(hidden // 2),
                nn.GELU(),
                nn.Conv1d(hidden // 2, hidden // 2, kernel_size=5, padding=4, dilation=2),
                nn.LayerNorm(hidden // 2),
                nn.GELU(),
            )
            # Temporal fusion
            self.gru = nn.GRU(
                input_size=hidden + hidden // 2,
                hidden_size=hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2,
            )
            self.classifier = nn.Linear(hidden * 2, 1)

        def forward(
            self,
            eeg: "torch.Tensor",      # (B, T, C)
            envelope: "torch.Tensor", # (B, T, 1)
        ) -> "torch.Tensor":          # (B, 1)
            # Conv1d expects (B, C, T)
            eeg_f = self.eeg_enc(eeg.permute(0, 2, 1))        # (B, H, T)
            env_f = self.env_enc(envelope.permute(0, 2, 1))   # (B, H/2, T)
            # Concatenate along channel, back to (B, T, H+H/2)
            merged = torch.cat(
                [eeg_f.permute(0, 2, 1), env_f.permute(0, 2, 1)], dim=-1
            )
            out, _ = self.gru(merged)          # (B, T, 2H)
            # Use last time step for classification
            return self.classifier(out[:, -1, :])  # (B, 1)

    return _FallbackCRN(n_ch=num_eeg_channels)


# ── MesgaraniAdapter ──────────────────────────────────────────────────────────

def _make_mesgarani_adapter_class() -> type:
    """Dynamically build MesgaraniAdapter inheriting nn.Module + BaseAADModel."""
    _require_torch()
    import torch.nn as nn

    class MesgaraniAdapter(nn.Module, BaseAADModel):
        """Neurophile adapter for the Mesgarani Lab Cortical Response Network.

        This adapter conforms to the ``BaseAADModel`` contract and is
        compatible with ``GlobalCITrainer`` and Flower federated aggregation.

        Parameters
        ----------
        num_eeg_channels : int
            Number of EEG channels in the input tensor.
        audio_sampling_rate : int
            Sampling rate of the audio envelope (post-downsampling to EEG rate).
        use_external : bool
            Force use of the real external lib (raises if absent).

        Examples
        --------
        >>> from neurophile.models.adapters import MesgaraniAdapter
        >>> model = MesgaraniAdapter(num_eeg_channels=64)
        >>> import torch
        >>> eeg = torch.randn(4, 512, 64)
        >>> env = torch.randn(4, 512, 1)
        >>> logit = model(eeg, env)  # (4, 1)
        """

        name = "mesgarani_crn"

        def __init__(
            self,
            num_eeg_channels: int = 64,
            audio_sampling_rate: int = 64,
            use_external: bool = False,
        ) -> None:
            super().__init__()
            self.num_eeg_channels = num_eeg_channels
            self.audio_sampling_rate = audio_sampling_rate
            self._using_external = False

            if use_external and not _MESGARANI_EXTERNAL_AVAILABLE:
                raise ImportError(
                    "use_external=True but external_libs.mesgarani_crn is not installed. "
                    "Run data_pipeline/fetchers/clone_author_repos.sh first."
                )

            if use_external and _MESGARANI_EXTERNAL_AVAILABLE:
                # ── Real Mesgarani CRN ────────────────────────────────────────
                # TODO (implementer): Adjust constructor arguments to match the
                # actual MesgaraniCRN signature. Common parameters:
                #   MesgaraniCRN(n_channels=num_eeg_channels, fs=audio_sampling_rate)
                self.backend_model = MesgaraniCRN(  # type: ignore[name-defined]
                    n_channels=num_eeg_channels
                )
                self._using_external = True
                logger.info("MesgaraniAdapter: backend = MesgaraniCRN (external)")
            else:
                self.backend_model = _build_fallback_mesgarani_crn(num_eeg_channels)
                logger.info("MesgaraniAdapter: backend = FallbackConvGRU")

        def forward(
            self,
            eeg_tensor: "torch.Tensor",
            audio_envelope_tensor: "torch.Tensor",
        ) -> "torch.Tensor":
            """Compute attention logit.

            Parameters
            ----------
            eeg_tensor : torch.Tensor, shape (B, T, C)
            audio_envelope_tensor : torch.Tensor, shape (B, T, 1)

            Returns
            -------
            logit : torch.Tensor, shape (B, 1)
            """
            if self._using_external:
                # TODO (implementer): Translate tensors to the Mesgarani CRN's
                # expected input format. Mesgarani models often expect separate
                # EEG and stimulus arrays as NumPy arrays or specific tensor shapes.
                # Document the exact translation here after inspecting the real code.
                return self.backend_model(eeg_tensor, audio_envelope_tensor)
            else:
                return self.backend_model(eeg_tensor, audio_envelope_tensor)

        def __repr__(self) -> str:
            backend = "MesgaraniCRN" if self._using_external else "FallbackConvGRU"
            return (
                f"MesgaraniAdapter(n_ch={self.num_eeg_channels}, "
                f"fs={self.audio_sampling_rate}, backend={backend})"
            )

    return MesgaraniAdapter


try:
    MesgaraniAdapter = _make_mesgarani_adapter_class()
except ImportError:
    class MesgaraniAdapter:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            _require_torch()

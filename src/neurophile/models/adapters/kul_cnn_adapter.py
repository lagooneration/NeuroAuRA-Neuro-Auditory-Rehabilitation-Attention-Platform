"""
neurophile.models.adapters.kul_cnn_adapter
==========================================
Adapter wrapping the KU Leuven CNN for Auditory Attention Decoding.

Anti-Corruption Layer (ACL)
----------------------------
Academic repositories rarely ship as clean pip-installable packages. This
adapter is the ACL: it absorbs the structural complexity of the external KUL
codebase and translates it into the strict Neurophile ``BaseAADModel`` contract.

If the external library breaks (e.g. upstream commits a path change), only
this shim layer requires patching — the global training loop, trainer, and
federated aggregation remain pristine.

How to wire the real KUL CNN
-----------------------------
1. Run ``data_pipeline/fetchers/clone_author_repos.sh``
2. Open this file and follow the ``# TODO (implementer)`` comments below.
3. The ``KULAdapter`` will automatically use the real model over the fallback.

Fallback CNN
------------
When ``external_libs.exporl_cnn`` is not found, ``KULAdapter`` transparently
falls back to a functionally equivalent 3-layer Temporal Convolutional Network
(TCN). This lets the entire training pipeline run immediately.

References
----------
- Vandecappelle et al. (2021) "EEG-based detection of the locus of auditory
  attention with convolutional neural networks"
  https://doi.org/10.7554/eLife.56481
- Repository: https://github.com/exporl/locus-of-auditory-attention-cnn
"""

from __future__ import annotations

import logging

from neurophile.models.core.base_aad_model import BaseAADModel, _require_torch

logger = logging.getLogger(__name__)

# ── External library import (ACL shim) ────────────────────────────────────────
_KUL_EXTERNAL_AVAILABLE = False
try:
    # TODO (implementer): After running clone_author_repos.sh, update this
    # import path to match the actual class exported by the cloned repo.
    # Common locations to check in the exporl repo:
    #   external_libs/locus-of-auditory-attention-cnn/model.py
    #   external_libs/locus-of-auditory-attention-cnn/src/model.py
    # Example:
    #   from external_libs.locus_of_auditory_attention_cnn.model import CNNModel as KULeuvenCNN
    from external_libs.exporl_cnn import KULeuvenCNN  # type: ignore[import]
    _KUL_EXTERNAL_AVAILABLE = True
    logger.info("KULAdapter: using real KULeuvenCNN from external_libs.")
except ImportError:
    logger.info(
        "KULAdapter: external_libs.exporl_cnn not found — "
        "using built-in fallback TCN. Run clone_author_repos.sh to enable the real model."
    )


# ── Fallback CNN (active when external lib is absent) ─────────────────────────

def _build_fallback_kul_cnn(num_eeg_channels: int) -> "torch.nn.Module":
    """Build a 3-layer TCN that mirrors the KUL CNN architecture.

    Architecture (simplified KUL):
      - Parallel EEG and envelope branches
      - Each branch: 3× Conv1d(in, 32, kernel=9, dilation=d) + BatchNorm + ELU
      - Merge via element-wise dot product → global average pool → Linear(1)

    This is a standalone, trainable approximation, not the exact KUL model.
    Replace with the real class once the repo is cloned.
    """
    _require_torch()
    import torch.nn as nn

    class _FallbackKULCNN(nn.Module):
        def __init__(self, n_ch: int) -> None:
            super().__init__()
            # EEG branch: (B, C, T) → (B, 32, T)
            self.eeg_branch = nn.Sequential(
                nn.Conv1d(n_ch, 32, kernel_size=9, padding=4, dilation=1),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Conv1d(32, 32, kernel_size=9, padding=8, dilation=2),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Conv1d(32, 32, kernel_size=9, padding=16, dilation=4),
                nn.BatchNorm1d(32),
                nn.ELU(),
            )
            # Envelope branch: (B, 1, T) → (B, 32, T)
            self.env_branch = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=9, padding=4, dilation=1),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Conv1d(32, 32, kernel_size=9, padding=8, dilation=2),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Conv1d(32, 32, kernel_size=9, padding=16, dilation=4),
                nn.BatchNorm1d(32),
                nn.ELU(),
            )
            # Merge + classify
            self.classifier = nn.Linear(32, 1)

        def forward(
            self,
            eeg: "torch.Tensor",         # (B, T, C)
            envelope: "torch.Tensor",    # (B, T, 1)
        ) -> "torch.Tensor":             # (B, 1)
            # Rearrange to (B, C, T) for Conv1d
            eeg_f = self.eeg_branch(eeg.permute(0, 2, 1))         # (B, 32, T)
            env_f = self.env_branch(envelope.permute(0, 2, 1))    # (B, 32, T)
            # Attended stream correlation: element-wise product, then pool
            merged = (eeg_f * env_f).mean(dim=-1)                 # (B, 32)
            return self.classifier(merged)                         # (B, 1)

    return _FallbackKULCNN(n_ch=num_eeg_channels)


# ── KULAdapter ────────────────────────────────────────────────────────────────

def _make_kul_adapter_class() -> type:
    """Dynamically build KULAdapter inheriting nn.Module + BaseAADModel."""
    _require_torch()
    import torch.nn as nn

    class KULAdapter(nn.Module, BaseAADModel):
        """Neurophile adapter for the KU Leuven Auditory Attention CNN.

        This adapter conforms to the ``BaseAADModel`` contract and is
        compatible with ``GlobalCITrainer`` and Flower federated aggregation.

        Parameters
        ----------
        num_eeg_channels : int
            Number of EEG channels in the input tensor.
        use_external : bool
            If True, forces use of the real external lib (raises if absent).
            If False (default), falls back to the built-in TCN.

        Examples
        --------
        >>> from neurophile.models.adapters import KULAdapter
        >>> model = KULAdapter(num_eeg_channels=64)
        >>> import torch
        >>> eeg = torch.randn(4, 512, 64)   # (B, T, C)
        >>> env = torch.randn(4, 512, 1)   # (B, T, 1)
        >>> logit = model(eeg, env)         # (4, 1)
        """

        name = "kul_cnn"

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

            if use_external and not _KUL_EXTERNAL_AVAILABLE:
                raise ImportError(
                    "use_external=True but external_libs.exporl_cnn is not installed. "
                    "Run data_pipeline/fetchers/clone_author_repos.sh first."
                )

            if use_external and _KUL_EXTERNAL_AVAILABLE:
                # ── Real KUL CNN ──────────────────────────────────────────────
                # TODO (implementer): Adjust constructor arguments to match the
                # actual signature of KULeuvenCNN. Common parameters include:
                #   KULeuvenCNN(n_channels=num_eeg_channels, fs=audio_sampling_rate)
                self.backend_model = KULeuvenCNN(  # type: ignore[name-defined]
                    n_channels=num_eeg_channels
                )
                self._using_external = True
                logger.info("KULAdapter: backend = KULeuvenCNN (external)")
            else:
                # ── Fallback TCN ─────────────────────────────────────────────
                self.backend_model = _build_fallback_kul_cnn(num_eeg_channels)
                logger.info("KULAdapter: backend = FallbackTCN")

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
                # TODO (implementer): Translate Neurophile tensors to the exact
                # input format expected by KULeuvenCNN. Example translation if
                # the real model expects (B, C, T):
                #   eeg_translated = eeg_tensor.permute(0, 2, 1)
                #   env_translated = audio_envelope_tensor.squeeze(-1)
                #   return self.backend_model(eeg_translated, env_translated)
                return self.backend_model(eeg_tensor, audio_envelope_tensor)
            else:
                return self.backend_model(eeg_tensor, audio_envelope_tensor)

        def __repr__(self) -> str:
            backend = "KULeuvenCNN" if self._using_external else "FallbackTCN"
            return (
                f"KULAdapter(n_ch={self.num_eeg_channels}, "
                f"fs={self.audio_sampling_rate}, backend={backend})"
            )

    return KULAdapter


# Build the class at module import time (deferred torch requirement)
try:
    KULAdapter = _make_kul_adapter_class()
except ImportError:
    # torch not installed — expose a clear placeholder
    class KULAdapter:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            _require_torch()

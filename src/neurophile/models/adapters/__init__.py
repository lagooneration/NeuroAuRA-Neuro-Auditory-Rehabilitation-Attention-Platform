"""
neurophile.models.adapters
==========================
ACL adapters wrapping external academic AAD models.

Each adapter:
  - Inherits ``torch.nn.Module`` + ``BaseAADModel``
  - Ships with a built-in fallback network (no external deps required)
  - Accepts the standard Neurophile tensors (B, T, C) EEG / (B, T, 1) envelope
  - Returns a logit (B, 1) — apply ``torch.sigmoid`` for probability
"""

from neurophile.models.adapters.kul_cnn_adapter import KULAdapter
from neurophile.models.adapters.mesgarani_crn_adapter import MesgaraniAdapter

__all__ = ["KULAdapter", "MesgaraniAdapter"]

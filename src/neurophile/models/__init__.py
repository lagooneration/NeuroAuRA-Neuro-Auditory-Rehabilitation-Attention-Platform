"""
neurophile.models
=================
Deep-learning AAD model ecosystem: abstract base, adapters, and trainer.

Architecture
------------
BaseAADModel (core)
    ├── KULAdapter          — KU Leuven CNN (fallback TCN included)
    └── MesgaraniAdapter    — Mesgarani CRN (fallback Conv+GRU included)

GlobalCITrainer — Strategy-pattern trainer (routes to PyTorch or sklearn)

Interface Contract
------------------
All adapters accept:
    eeg_tensor    : torch.Tensor  shape (B, T, C)
    envelope_tensor: torch.Tensor shape (B, T, 1)

All adapters return:
    logit         : torch.Tensor  shape (B, 1)
    (apply torch.sigmoid for probability ∈ [0, 1])
"""

from neurophile.models.core.base_aad_model import BaseAADModel
from neurophile.models.adapters.kul_cnn_adapter import KULAdapter
from neurophile.models.adapters.mesgarani_crn_adapter import MesgaraniAdapter
from neurophile.models.global_trainer import GlobalCITrainer

__all__ = [
    "BaseAADModel",
    "KULAdapter",
    "MesgaraniAdapter",
    "GlobalCITrainer",
]

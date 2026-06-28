"""
tests/unit/test_zion_golumbic_adapter.py
========================================
Unit tests for the ZionGolumbicAdapter.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — skipping deep learning tests")

from neurophile.models.adapters.zion_golumbic_adapter import ZionGolumbicAdapter
from neurophile.models.core.base_aad_model import BaseAADModel

BATCH = 4
T = 256
C = 16

@pytest.fixture
def eeg_batch() -> "torch.Tensor":
    return torch.randn(BATCH, T, C)

@pytest.fixture
def env_batch() -> "torch.Tensor":
    return torch.randn(BATCH, T, 1)

@pytest.fixture
def zion_model() -> ZionGolumbicAdapter:
    return ZionGolumbicAdapter(num_eeg_channels=C)

def test_zion_adapter_is_base_aad_model(zion_model: ZionGolumbicAdapter) -> None:
    assert isinstance(zion_model, BaseAADModel)

def test_zion_adapter_is_nn_module(zion_model: ZionGolumbicAdapter) -> None:
    import torch.nn as nn
    assert isinstance(zion_model, nn.Module)

def test_zion_forward_shape(
    zion_model: ZionGolumbicAdapter,
    eeg_batch: "torch.Tensor",
    env_batch: "torch.Tensor",
) -> None:
    out = zion_model(eeg_batch, env_batch)
    assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"

def test_zion_uses_fallback_by_default(zion_model: ZionGolumbicAdapter) -> None:
    assert zion_model._using_external is False

def test_zion_use_external_raises_without_lib() -> None:
    with pytest.raises(ImportError, match="external_libs"):
        ZionGolumbicAdapter(num_eeg_channels=C, use_external=True)

def test_zion_gradients_flow(
    zion_model: ZionGolumbicAdapter,
    eeg_batch: "torch.Tensor",
    env_batch: "torch.Tensor",
) -> None:
    import torch.nn as nn
    zion_model.train()
    out = zion_model(eeg_batch, env_batch)
    labels = torch.ones(BATCH, 1)
    loss = nn.BCEWithLogitsLoss()(out, labels)
    loss.backward()
    params_with_grad = [p for p in zion_model.parameters() if p.grad is not None]
    assert len(params_with_grad) > 0

def test_repr(zion_model: ZionGolumbicAdapter) -> None:
    assert "ZionGolumbicAdapter" in repr(zion_model)

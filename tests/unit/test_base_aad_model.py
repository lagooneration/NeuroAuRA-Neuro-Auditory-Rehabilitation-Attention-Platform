"""
tests/unit/test_base_aad_model.py
==================================
Unit tests for the BaseAADModel interface and both adapters.

Tests use the built-in fallback networks (no external libs required).
PyTorch tests are skipped automatically if torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch", reason="torch not installed — skipping deep learning tests")


from neurophile.models.core.base_aad_model import BaseAADModel, _require_torch
from neurophile.models.adapters.kul_cnn_adapter import KULAdapter
from neurophile.models.adapters.mesgarani_crn_adapter import MesgaraniAdapter
from neurophile.models.global_trainer import GlobalCITrainer


# ── Fixtures ──────────────────────────────────────────────────────────────────

BATCH = 4
T = 256    # time steps
C = 16     # EEG channels (small for fast tests)


@pytest.fixture
def eeg_batch() -> "torch.Tensor":
    return torch.randn(BATCH, T, C)


@pytest.fixture
def env_batch() -> "torch.Tensor":
    return torch.randn(BATCH, T, 1)


@pytest.fixture
def kul_model() -> KULAdapter:
    return KULAdapter(num_eeg_channels=C)


@pytest.fixture
def mesgarani_model() -> MesgaraniAdapter:
    return MesgaraniAdapter(num_eeg_channels=C)


# ── BaseAADModel contract ─────────────────────────────────────────────────────

def test_kul_adapter_is_base_aad_model(kul_model: KULAdapter) -> None:
    """KULAdapter must be an instance of BaseAADModel."""
    assert isinstance(kul_model, BaseAADModel)


def test_mesgarani_adapter_is_base_aad_model(mesgarani_model: MesgaraniAdapter) -> None:
    """MesgaraniAdapter must be an instance of BaseAADModel."""
    assert isinstance(mesgarani_model, BaseAADModel)


def test_kul_adapter_is_nn_module(kul_model: KULAdapter) -> None:
    """KULAdapter must be an nn.Module (for Flower federated aggregation)."""
    import torch.nn as nn
    assert isinstance(kul_model, nn.Module)


def test_mesgarani_adapter_is_nn_module(mesgarani_model: MesgaraniAdapter) -> None:
    import torch.nn as nn
    assert isinstance(mesgarani_model, nn.Module)


# ── Forward output shape ──────────────────────────────────────────────────────

def test_kul_forward_shape(
    kul_model: KULAdapter,
    eeg_batch: "torch.Tensor",
    env_batch: "torch.Tensor",
) -> None:
    """KULAdapter.forward() must return shape (B, 1)."""
    out = kul_model(eeg_batch, env_batch)
    assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"


def test_mesgarani_forward_shape(
    mesgarani_model: MesgaraniAdapter,
    eeg_batch: "torch.Tensor",
    env_batch: "torch.Tensor",
) -> None:
    """MesgaraniAdapter.forward() must return shape (B, 1)."""
    out = mesgarani_model(eeg_batch, env_batch)
    assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"


def test_kul_forward_dtype(
    kul_model: KULAdapter,
    eeg_batch: "torch.Tensor",
    env_batch: "torch.Tensor",
) -> None:
    """Output should be float32."""
    out = kul_model(eeg_batch, env_batch)
    assert out.dtype == torch.float32


# ── Fallback backend ──────────────────────────────────────────────────────────

def test_kul_uses_fallback_by_default(kul_model: KULAdapter) -> None:
    """Without external libs, KULAdapter should use the fallback TCN."""
    assert kul_model._using_external is False


def test_mesgarani_uses_fallback_by_default(mesgarani_model: MesgaraniAdapter) -> None:
    assert mesgarani_model._using_external is False


def test_kul_use_external_raises_without_lib() -> None:
    """use_external=True should raise ImportError when external lib is absent."""
    with pytest.raises(ImportError, match="external_libs"):
        KULAdapter(num_eeg_channels=C, use_external=True)


# ── decode() numpy convenience API ───────────────────────────────────────────

def test_kul_decode_returns_float(kul_model: KULAdapter) -> None:
    """decode() should return a float in [0, 1]."""
    eeg_np = np.random.randn(T, C).astype("float32")
    env_np = np.random.randn(T).astype("float32")
    p = kul_model.decode(eeg_np, env_np)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0, f"Expected probability in [0, 1], got {p}"


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_kul_gradients_flow(
    kul_model: KULAdapter,
    eeg_batch: "torch.Tensor",
    env_batch: "torch.Tensor",
) -> None:
    """Loss should propagate gradients to all model parameters."""
    import torch.nn as nn
    kul_model.train()
    out = kul_model(eeg_batch, env_batch)
    labels = torch.ones(BATCH, 1)
    loss = nn.BCEWithLogitsLoss()(out, labels)
    loss.backward()

    params_with_grad = [p for p in kul_model.parameters() if p.grad is not None]
    assert len(params_with_grad) > 0, "No gradients flowed to model parameters"


def test_mesgarani_gradients_flow(
    mesgarani_model: MesgaraniAdapter,
    eeg_batch: "torch.Tensor",
    env_batch: "torch.Tensor",
) -> None:
    import torch.nn as nn
    mesgarani_model.train()
    out = mesgarani_model(eeg_batch, env_batch)
    labels = torch.ones(BATCH, 1)
    loss = nn.BCEWithLogitsLoss()(out, labels)
    loss.backward()
    params_with_grad = [p for p in mesgarani_model.parameters() if p.grad is not None]
    assert len(params_with_grad) > 0


# ── State dict (Flower FL compatibility) ─────────────────────────────────────

def test_state_dict_roundtrip(kul_model: KULAdapter) -> None:
    """Model weights should survive a state_dict → load_state_dict roundtrip."""
    sd = kul_model.state_dict()
    new_model = KULAdapter(num_eeg_channels=C)
    new_model.load_state_dict(sd)
    # Check first parameter matches
    p_orig = next(kul_model.parameters()).detach().numpy()
    p_new = next(new_model.parameters()).detach().numpy()
    np.testing.assert_array_equal(p_orig, p_new)


# ── GlobalCITrainer (Strategy selection) ──────────────────────────────────────

def test_trainer_selects_pytorch_strategy(kul_model: KULAdapter) -> None:
    """Trainer should detect nn.Module and route to PyTorch strategy."""
    trainer = GlobalCITrainer(kul_model, epochs=1, batch_size=2)
    assert trainer._backend == "pytorch"


def test_trainer_smoke_train(kul_model: KULAdapter) -> None:
    """Trainer should complete 1 epoch on synthetic data without errors."""
    N = 8
    eeg = np.random.randn(N, T, C).astype("float32")
    env = np.random.randn(N, T, 1).astype("float32")
    labels = np.array([i % 2 for i in range(N)], dtype="float32")

    trainer = GlobalCITrainer(
        kul_model, epochs=1, batch_size=4,
        output_dir="/tmp/neurophile_test_ckpts"
    )
    history = trainer.train(eeg, env, labels)
    assert len(history) == 1
    assert np.isfinite(history[0])


def test_repr(kul_model: KULAdapter, mesgarani_model: MesgaraniAdapter) -> None:
    assert "KULAdapter" in repr(kul_model)
    assert "MesgaraniAdapter" in repr(mesgarani_model)

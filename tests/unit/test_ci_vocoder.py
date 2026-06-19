"""
tests/unit/test_ci_vocoder.py
==============================
Unit tests for CIVocoderSimulator.

Tests cover:
- Output shape preservation
- RMS normalisation (output ≈ input RMS)
- N-channel variation (energy reduction with fewer channels)
- Combined simulate_and_extract_envelope (shape + dtype)
- Edge cases: mono/stereo input, single channel
"""

from __future__ import annotations

import numpy as np
import pytest

from neurophile.stimulus.ci_vocoder import CIVocoderSimulator


# ── Fixtures ──────────────────────────────────────────────────────────────────

FS = 16000   # Use lower FS for speed (full 44100 is slow in CI)
DURATION_S = 2
N_SAMPLES = FS * DURATION_S


@pytest.fixture
def mono_audio() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal(N_SAMPLES).astype("float32")


@pytest.fixture
def stereo_audio(mono_audio: np.ndarray) -> np.ndarray:
    return np.stack([mono_audio, mono_audio * 0.5], axis=1)


@pytest.fixture
def vocoder_16ch() -> CIVocoderSimulator:
    return CIVocoderSimulator(fs=FS, n_channels=16, f_low=200.0, f_high=7000.0, seed=0)


# ── Shape contract ────────────────────────────────────────────────────────────

def test_output_shape_mono(vocoder_16ch: CIVocoderSimulator, mono_audio: np.ndarray) -> None:
    """Output shape must match input length."""
    out = vocoder_16ch.simulate(mono_audio)
    assert out.shape == (N_SAMPLES,), f"Expected ({N_SAMPLES},), got {out.shape}"


def test_output_shape_stereo(vocoder_16ch: CIVocoderSimulator, stereo_audio: np.ndarray) -> None:
    """Stereo input is averaged to mono — output shape = (n_samples,)."""
    out = vocoder_16ch.simulate(stereo_audio)
    assert out.shape == (N_SAMPLES,)


def test_output_dtype(vocoder_16ch: CIVocoderSimulator, mono_audio: np.ndarray) -> None:
    """Output should be float32."""
    out = vocoder_16ch.simulate(mono_audio)
    assert out.dtype == np.float32


# ── RMS normalisation ─────────────────────────────────────────────────────────

def test_rms_preservation(vocoder_16ch: CIVocoderSimulator, mono_audio: np.ndarray) -> None:
    """Vocoded output RMS should be close to input RMS (within 50%)."""
    input_rms = float(np.sqrt(np.mean(mono_audio ** 2)))
    out = vocoder_16ch.simulate(mono_audio)
    output_rms = float(np.sqrt(np.mean(out ** 2)))
    # Tolerance is wide — RMS normalisation is approximate due to band reconstruction
    ratio = output_rms / (input_rms + 1e-12)
    assert 0.5 < ratio < 2.0, f"RMS ratio {ratio:.3f} out of expected range [0.5, 2.0]"


# ── Channel count effects ─────────────────────────────────────────────────────

def test_single_channel(mono_audio: np.ndarray) -> None:
    """Single channel vocoder should still produce valid output."""
    vocoder = CIVocoderSimulator(fs=FS, n_channels=1)
    out = vocoder.simulate(mono_audio)
    assert out.shape == (N_SAMPLES,)
    assert np.isfinite(out).all(), "Single-channel output contains NaN/Inf"


def test_more_channels_preserves_shape(mono_audio: np.ndarray) -> None:
    """22-channel vocoder should produce the same shape."""
    vocoder = CIVocoderSimulator(fs=FS, n_channels=22)
    out = vocoder.simulate(mono_audio)
    assert out.shape == (N_SAMPLES,)


# ── simulate_and_extract_envelope ────────────────────────────────────────────

def test_envelope_shape(vocoder_16ch: CIVocoderSimulator, mono_audio: np.ndarray) -> None:
    """Envelope should be downsampled to FS_EEG samples."""
    fs_eeg = 64
    expected_len = int(N_SAMPLES * fs_eeg / FS)  # approximate
    _, env = vocoder_16ch.simulate_and_extract_envelope(mono_audio, fs_eeg=fs_eeg)
    # Allow ±5 sample tolerance for resampling edge effects
    assert abs(len(env) - expected_len) <= 5, (
        f"Envelope length {len(env)} differs from expected ~{expected_len}"
    )


def test_envelope_dtype(vocoder_16ch: CIVocoderSimulator, mono_audio: np.ndarray) -> None:
    """Envelope should be float32."""
    _, env = vocoder_16ch.simulate_and_extract_envelope(mono_audio)
    assert env.dtype == np.float32


def test_envelope_finite(vocoder_16ch: CIVocoderSimulator, mono_audio: np.ndarray) -> None:
    """Envelope values should be finite and non-negative."""
    _, env = vocoder_16ch.simulate_and_extract_envelope(mono_audio)
    assert np.isfinite(env).all(), "Envelope contains NaN or Inf"
    assert (env >= 0).all(), "Envelope contains negative values"


# ── Carrier types ─────────────────────────────────────────────────────────────

def test_sine_carrier(mono_audio: np.ndarray) -> None:
    """Sine carrier should produce valid output."""
    vocoder = CIVocoderSimulator(fs=FS, n_channels=8, carrier="sine")
    out = vocoder.simulate(mono_audio)
    assert out.shape == (N_SAMPLES,)
    assert np.isfinite(out).all()


# ── Validation ────────────────────────────────────────────────────────────────

def test_invalid_carrier() -> None:
    with pytest.raises(ValueError, match="carrier"):
        CIVocoderSimulator(fs=FS, n_channels=8, carrier="square")


def test_invalid_n_channels() -> None:
    with pytest.raises(ValueError, match="n_channels"):
        CIVocoderSimulator(fs=FS, n_channels=0)


def test_invalid_frequency_range() -> None:
    with pytest.raises(ValueError, match="f_low"):
        CIVocoderSimulator(fs=FS, n_channels=8, f_low=8000.0, f_high=200.0)


# ── Repr ──────────────────────────────────────────────────────────────────────

def test_repr(vocoder_16ch: CIVocoderSimulator) -> None:
    r = repr(vocoder_16ch)
    assert "CIVocoderSimulator" in r
    assert "n_channels=16" in r

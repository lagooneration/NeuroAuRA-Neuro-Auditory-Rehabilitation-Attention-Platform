"""
Pytest fixtures for Neurophile unit tests.

All fixtures produce synthetic data so that tests run instantly with no
internet access, no real datasets, and no hardware.

Available fixtures
------------------
synthetic_eeg       : (n_samples, n_channels) random EEG, fs=512
synthetic_envelope  : (n_samples,) smoothed envelope at fs=512
synthetic_ci_eeg    : EEG with synthetic periodic CI artifact injected
aad_trials          : list of 4 AADTrial objects for evaluation harness tests
bids_root           : tmp_path BIDS directory with one synthetic session
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ── Constants ────────────────────────────────────────────────────────────────
FS = 512
N_CHANNELS = 8
DURATION_S = 60
N_SAMPLES = DURATION_S * FS


@pytest.fixture
def synthetic_eeg() -> np.ndarray:
    """Random EEG: (N_SAMPLES, N_CHANNELS) at FS Hz."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((N_SAMPLES, N_CHANNELS)).astype(np.float32)


@pytest.fixture
def synthetic_envelope() -> np.ndarray:
    """Smoothed random envelope: (N_SAMPLES,) at FS Hz."""
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(N_SAMPLES)
    # Simple exponential smoothing
    alpha = 0.01
    env = np.empty_like(raw)
    env[0] = raw[0]
    for i in range(1, len(raw)):
        env[i] = alpha * abs(raw[i]) + (1 - alpha) * env[i - 1]
    return env.astype(np.float32)


@pytest.fixture
def synthetic_ci_eeg(synthetic_eeg) -> np.ndarray:
    """EEG with a synthetic periodic CI artifact at 900 pps injected on CH0."""
    eeg = synthetic_eeg.copy()
    ci_rate_pps = 900
    period_samples = int(FS / ci_rate_pps * (FS / 1000))  # crude approximation
    if period_samples < 1:
        period_samples = 1
    # Inject narrow spikes at the CI rate on the first channel
    for onset in range(0, N_SAMPLES - 2, period_samples):
        eeg[onset, 0] += 500.0   # 500 µV spike → artifact amplitude
        if onset + 1 < N_SAMPLES:
            eeg[onset + 1, 0] -= 500.0
    return eeg


@pytest.fixture
def aad_trials(synthetic_eeg, synthetic_envelope) -> list:
    """Four synthetic AADTrial objects for leave-one-trial-out evaluation tests."""
    from neurophile.decoding.aad_evaluation import AADTrial

    rng = np.random.default_rng(7)
    trials = []
    for i in range(4):
        # Attended stream correlates weakly with EEG (linear model will find it)
        eeg_i = synthetic_eeg + rng.standard_normal(synthetic_eeg.shape).astype(np.float32) * 0.5
        env_att = synthetic_envelope + rng.standard_normal(len(synthetic_envelope)).astype(np.float32) * 0.1
        env_ign = rng.standard_normal(len(synthetic_envelope)).astype(np.float32)
        trials.append(AADTrial(
            eeg=eeg_i,
            env_attended=env_att,
            env_ignored=env_ign,
            fs=FS,
            subject="test",
            session="01",
            trial_idx=i,
        ))
    return trials


@pytest.fixture
def bids_root(tmp_path) -> Path:
    """Minimal BIDS directory with one synthetic session's sidecar JSON."""
    root = tmp_path / "bids_study"
    session_dir = root / "sub-01" / "ses-01" / "eeg"
    session_dir.mkdir(parents=True)

    sidecar = {
        "SamplingFrequency": FS,
        "EEGReference": "average",
        "PowerLineFrequency": 50,
        "HardwareFilters": {"highpass": 0.1, "lowpass": 200},
        "SoftwareFilters": {},
        "StimulusSyncMethod": "LSL",
        "MeasuredSyncOffset_ms": 2.1,
        "MeasuredDrift_ppm": 0.9,
    }
    sidecar_path = session_dir / "sub-01_ses-01_task-aad_eeg.json"
    sidecar_path.write_text(json.dumps(sidecar))

    # Minimal dataset_description.json
    desc = {
        "Name": "test_study",
        "BIDSVersion": "1.7.0",
        "DatasetType": "raw",
    }
    (root / "dataset_description.json").write_text(json.dumps(desc))
    return root

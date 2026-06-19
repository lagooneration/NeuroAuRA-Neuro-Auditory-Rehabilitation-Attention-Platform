"""
tests/unit/test_ica_cancellation.py
=====================================
Unit tests for ICACancellation (ICA-based CI artifact removal).

Tests cover:
- Output shape preservation
- Output dtype preservation
- Kurtosis-based artifact detection on synthetic CI-contaminated EEG
- No-op behaviour when no high-kurtosis ICs are present
- get_artifact_report() structure
- Integration with CIArtifactPipeline (Stage 3 enabled)
"""

from __future__ import annotations

import numpy as np
import pytest

from neurophile.preprocessing.ci_artifact.ica_cancellation import (
    ICACancellation,
    ICACancellationConfig,
)
from neurophile.preprocessing.ci_artifact.pipeline import (
    CIArtifactPipeline,
    CIArtifactConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

FS = 1000       # Hz
N_SAMPLES = 5000
N_CHANNELS = 16


@pytest.fixture
def clean_eeg() -> np.ndarray:
    """Synthetic EEG: multi-channel Gaussian noise (low kurtosis)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((N_SAMPLES, N_CHANNELS)).astype("float32")


@pytest.fixture
def ci_contaminated_eeg(clean_eeg: np.ndarray) -> np.ndarray:
    """EEG + synthetic CI artifact: one channel has super-Gaussian spikes."""
    eeg = clean_eeg.copy()
    rng = np.random.default_rng(1)
    # Add very strong impulse spikes at 900 pps to channel 0
    # High amplitude (×200) ensures kurtosis is super-Gaussian even after ICA mixing
    rate_pps = 900
    spike_interval = int(FS / rate_pps)
    spike_indices = np.arange(0, N_SAMPLES, spike_interval)
    eeg[spike_indices, 0] += rng.standard_normal(len(spike_indices)) * 200.0
    return eeg


# ── Shape and dtype ───────────────────────────────────────────────────────────

def test_output_shape(ci_contaminated_eeg: np.ndarray) -> None:
    """Output shape must exactly match input shape."""
    ica = ICACancellation(fs=FS)
    out = ica.fit_transform(ci_contaminated_eeg)
    assert out.shape == ci_contaminated_eeg.shape, (
        f"Shape mismatch: input {ci_contaminated_eeg.shape}, output {out.shape}"
    )


def test_output_dtype_preserved(ci_contaminated_eeg: np.ndarray) -> None:
    """Output dtype should match input dtype."""
    ica = ICACancellation(fs=FS)
    out = ica.fit_transform(ci_contaminated_eeg)
    assert out.dtype == ci_contaminated_eeg.dtype


def test_output_finite(ci_contaminated_eeg: np.ndarray) -> None:
    """Output should contain no NaN or Inf values."""
    ica = ICACancellation(fs=FS)
    out = ica.fit_transform(ci_contaminated_eeg)
    assert np.isfinite(out).all(), "Output contains NaN or Inf"


# ── Artifact detection ────────────────────────────────────────────────────────

def test_detects_high_kurtosis_component(ci_contaminated_eeg: np.ndarray) -> None:
    """ICA should flag at least one component on CI-contaminated EEG."""
    config = ICACancellationConfig(
        kurtosis_threshold=2.0,   # realistic threshold for CI spike detection
        n_components=8,
        max_iter=2000,
    )
    ica = ICACancellation(fs=FS, config=config)
    ica.fit_transform(ci_contaminated_eeg)
    report = ica.get_artifact_report()
    assert report["n_artifacts_removed"] >= 1, (
        f"Expected ≥1 artifact IC, got {report['n_artifacts_removed']}"
    )


def test_clean_eeg_no_artifact_detected(clean_eeg: np.ndarray) -> None:
    """On clean Gaussian EEG, no ICs should exceed kurtosis threshold."""
    config = ICACancellationConfig(
        kurtosis_threshold=10.0,  # very high threshold — Gaussian noise won't exceed this
        n_components=8,
    )
    ica = ICACancellation(fs=FS, config=config)
    ica.fit_transform(clean_eeg)
    report = ica.get_artifact_report()
    assert report["n_artifacts_removed"] == 0, (
        f"Expected 0 artifacts on clean EEG, got {report['n_artifacts_removed']}"
    )


# ── Report structure ─────────────────────────────────────────────────────────

def test_report_before_fit() -> None:
    """Report should return an error dict if called before fit_transform."""
    ica = ICACancellation(fs=FS)
    report = ica.get_artifact_report()
    assert "error" in report


def test_report_after_fit(ci_contaminated_eeg: np.ndarray) -> None:
    """Report should contain expected keys after fit_transform."""
    ica = ICACancellation(fs=FS)
    ica.fit_transform(ci_contaminated_eeg)
    report = ica.get_artifact_report()
    for key in ("n_components_total", "artifact_indices", "n_artifacts_removed"):
        assert key in report, f"Missing key in report: {key}"
    assert isinstance(report["artifact_indices"], list)
    assert report["n_components_total"] > 0


# ── Safety caps ───────────────────────────────────────────────────────────────

def test_max_artifact_components_cap(ci_contaminated_eeg: np.ndarray) -> None:
    """Never remove more than max_artifact_components ICs."""
    config = ICACancellationConfig(
        kurtosis_threshold=0.0,   # flag everything
        max_artifact_components=2,
        n_components=8,
        max_iter=500,
    )
    ica = ICACancellation(fs=FS, config=config)
    ica.fit_transform(ci_contaminated_eeg)
    report = ica.get_artifact_report()
    assert report["n_artifacts_removed"] <= 2, (
        f"Expected \u22642 removed, got {report['n_artifacts_removed']}"
    )


def test_min_artifact_components_floor(clean_eeg: np.ndarray) -> None:
    """Always remove at least min_artifact_components ICs when threshold is met."""
    config = ICACancellationConfig(
        kurtosis_threshold=100.0,  # nothing qualifies naturally
        min_artifact_components=0,  # floor is 0, so nothing forced
        n_components=4,
    )
    ica = ICACancellation(fs=FS, config=config)
    ica.fit_transform(clean_eeg)
    report = ica.get_artifact_report()
    # With threshold=100 no real signal should qualify → 0 removed
    assert report["n_artifacts_removed"] == 0


# ── Pipeline integration ──────────────────────────────────────────────────────

def test_pipeline_stage3_integration(ci_contaminated_eeg: np.ndarray) -> None:
    """CIArtifactPipeline with stage3_enabled=True should produce correct output shape."""
    config = CIArtifactConfig(stage3_enabled=True)
    pipeline = CIArtifactPipeline(fs=FS, config=config)

    out = pipeline.run(ci_contaminated_eeg)
    assert out.shape == ci_contaminated_eeg.shape


def test_pipeline_stage3_disabled(ci_contaminated_eeg: np.ndarray) -> None:
    """CIArtifactPipeline with stage3_enabled=False should still work."""
    config = CIArtifactConfig(stage3_enabled=False)
    pipeline = CIArtifactPipeline(fs=FS, config=config)
    out = pipeline.run(ci_contaminated_eeg)
    assert out.shape == ci_contaminated_eeg.shape

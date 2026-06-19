"""Unit tests for CI artifact pipeline."""

import numpy as np
import pytest

from neurophile.preprocessing.ci_artifact.pipeline import CIArtifactPipeline
from neurophile.preprocessing.ci_artifact.template_subtraction import CITemplateSubtraction
from neurophile.preprocessing.ci_artifact.quality import CIArtifactQuality


FS = 1000
N_SAMPLES = FS * 5  # 5 seconds
N_CH = 8


class TestCITemplateSubtraction:

    def test_output_shape_preserved(self, synthetic_ci_eeg):
        stage1 = CITemplateSubtraction(fs=512)
        clean = stage1.fit_transform(synthetic_ci_eeg)
        assert clean.shape == synthetic_ci_eeg.shape

    def test_artifact_reduced(self, synthetic_ci_eeg):
        """High-frequency power on the contaminated channel should decrease."""
        stage1 = CITemplateSubtraction(fs=512)
        clean = stage1.fit_transform(synthetic_ci_eeg)
        raw_hf = np.std(synthetic_ci_eeg[:, 0])
        clean_hf = np.std(clean[:, 0])
        assert clean_hf < raw_hf, "HF power should decrease after artifact removal"

    def test_disabled_returns_copy(self, synthetic_eeg):
        from neurophile.preprocessing.ci_artifact.template_subtraction import TemplateSubtractionConfig
        config = TemplateSubtractionConfig(enabled=False)
        stage1 = CITemplateSubtraction(fs=512, config=config)
        clean = stage1.fit_transform(synthetic_eeg)
        np.testing.assert_array_equal(clean, synthetic_eeg)


class TestCIArtifactPipeline:

    def test_pipeline_runs(self, synthetic_ci_eeg):
        pipeline = CIArtifactPipeline(fs=512)
        clean = pipeline.run(synthetic_ci_eeg)
        assert clean.shape == synthetic_ci_eeg.shape

    def test_quality_report_keys(self, synthetic_eeg):
        pipeline = CIArtifactPipeline(fs=512)
        clean = pipeline.run(synthetic_eeg)
        metrics = pipeline.quality_report(synthetic_eeg, clean)
        assert "snr_improvement_db" in metrics
        assert "neural_band_power_ratio" in metrics
        assert "artifact_residual_corr" in metrics
        assert "topographic_asymmetry_index" in metrics

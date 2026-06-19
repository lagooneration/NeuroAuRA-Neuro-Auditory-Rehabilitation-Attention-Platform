"""Unit tests for LinearDecoder and AADEvaluator."""

import numpy as np
import pytest

from neurophile.decoding.linear_decoder import LinearDecoder
from neurophile.decoding.aad_evaluation import AADEvaluator


FS = 512
N_SAMPLES = FS * 30  # 30 seconds
N_CH = 8


def make_eeg_envelope():
    rng = np.random.default_rng(42)
    env = rng.standard_normal(N_SAMPLES).astype(np.float32)
    # Simple EEG: weighted combination of envelope + noise
    eeg = np.outer(env, np.ones(N_CH)) + rng.standard_normal((N_SAMPLES, N_CH)) * 2.0
    return eeg.astype(np.float32), env


class TestLinearDecoder:

    def test_fit_predict_shape(self):
        eeg, env = make_eeg_envelope()
        decoder = LinearDecoder(lag_min_ms=0, lag_max_ms=100)
        decoder.fit(eeg, env, FS)
        predicted = decoder.predict(eeg)
        # Output length is n_samples - lag_max
        lag_max = int(np.round(100 * FS / 1000))
        assert predicted.shape == (N_SAMPLES - lag_max,)

    def test_score_above_zero(self):
        """Decoder should find some positive correlation with the attended envelope."""
        eeg, env = make_eeg_envelope()
        decoder = LinearDecoder(lag_min_ms=0, lag_max_ms=50)
        decoder.fit(eeg, env, FS)
        r = decoder.score(eeg, env[int(50 * FS / 1000):])
        assert r > 0.0, f"Expected positive correlation, got {r}"

    def test_error_without_fit(self):
        decoder = LinearDecoder()
        with pytest.raises(RuntimeError, match="fit"):
            decoder.predict(np.zeros((100, 8)))

    def test_n_lags(self):
        decoder = LinearDecoder(lag_min_ms=0, lag_max_ms=200)
        eeg, env = make_eeg_envelope()
        decoder.fit(eeg, env, FS)
        expected = int(np.round(200 * FS / 1000)) + 1
        assert decoder.n_lags == expected


class TestAADEvaluator:

    def test_evaluate_returns_dataframe(self, aad_trials):
        evaluator = AADEvaluator(window_s=30.0, n_jobs=1)
        evaluator.register_decoder("linear", LinearDecoder(lag_max_ms=50))
        results = evaluator.evaluate(aad_trials)
        assert len(results) > 0
        required_cols = {"decoder", "subject", "trial", "window_s", "r_attended", "r_ignored", "decision"}
        assert required_cols.issubset(results.columns)

    def test_multiple_decoders(self, aad_trials):
        evaluator = AADEvaluator(window_s=30.0, n_jobs=1)
        evaluator.register_decoder("linear_fast", LinearDecoder(lag_max_ms=50))
        evaluator.register_decoder("linear_slow", LinearDecoder(lag_max_ms=100))
        results = evaluator.evaluate(aad_trials)
        assert set(results["decoder"].unique()) == {"linear_fast", "linear_slow"}

    def test_multiple_windows(self, aad_trials):
        evaluator = AADEvaluator(window_s=[10.0, 30.0], n_jobs=1)
        evaluator.register_decoder("linear", LinearDecoder(lag_max_ms=50))
        results = evaluator.evaluate(aad_trials)
        assert set(results["window_s"].unique()) == {10.0, 30.0}

    def test_no_decoders_raises(self, aad_trials):
        evaluator = AADEvaluator()
        with pytest.raises(ValueError, match="No decoders"):
            evaluator.evaluate(aad_trials)

    def test_insufficient_trials_raises(self, aad_trials):
        evaluator = AADEvaluator()
        evaluator.register_decoder("linear", LinearDecoder())
        with pytest.raises(ValueError, match="at least 2"):
            evaluator.evaluate(aad_trials[:1])

    def test_summarize(self, aad_trials):
        evaluator = AADEvaluator(window_s=30.0, n_jobs=1)
        evaluator.register_decoder("linear", LinearDecoder(lag_max_ms=50))
        results = evaluator.evaluate(aad_trials)
        summary = AADEvaluator.summarize(results)
        assert "accuracy_mean" in summary.columns
        assert "accuracy_std" in summary.columns

"""
neurophile.decoding.aad_evaluation
====================================
Parallel evaluation harness for Auditory Attention Decoding models.

This harness is decoder-agnostic: any class implementing BaseDecoder can be
registered and evaluated side-by-side. Results are returned as a tidy pandas
DataFrame, analogous to MOABB's evaluation output, but designed specifically
for the AAD regression paradigm.

Evaluation protocol
-------------------
1. For each trial: fit decoder on all OTHER trials (leave-one-trial-out).
2. Apply decoder to held-out trial EEG.
3. Compute Pearson r between predicted envelope and:
   - attended stream  (r_attended)
   - ignored stream   (r_ignored)
4. Decision: correct if r_attended > r_ignored.
5. Slide a window of length ``window_s`` across the trial for finer resolution.

Result DataFrame columns
------------------------
decoder         : decoder name (e.g. "linear")
subject         : subject ID
session         : session ID
trial           : trial index
window          : window index within trial
window_s        : decision window length in seconds
r_attended      : Pearson r with attended envelope
r_ignored       : Pearson r with ignored envelope
decision        : 1 = correct, 0 = wrong
accuracy        : mean decision accuracy across windows in trial

Adding a new decoder
--------------------
1. Implement BaseDecoder in a new module.
2. Pass an instance to AADEvaluator.register_decoder().
3. Re-run evaluate(). Results appear as a new row in the DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from neurophile.decoding.base import BaseDecoder

logger = logging.getLogger(__name__)


@dataclass
class AADTrial:
    """Container for a single AAD trial's data."""

    eeg: np.ndarray          # (n_samples, n_channels) at fs Hz
    env_attended: np.ndarray # (n_samples,) attended stream envelope
    env_ignored: np.ndarray  # (n_samples,) ignored stream envelope
    fs: int                  # EEG sampling rate
    subject: str = "unknown"
    session: str = "unknown"
    trial_idx: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class AADEvaluator:
    """Evaluate multiple AAD decoders with leave-one-trial-out cross-validation.

    Parameters
    ----------
    window_s : float or list of float
        Decision window length(s) in seconds. Multiple window sizes are evaluated
        in one pass, enabling a window-length vs. accuracy curve.
    n_jobs : int
        Number of parallel jobs for cross-validation. -1 = all CPU cores.

    Examples
    --------
    >>> from neurophile.decoding import LinearDecoder, AADEvaluator
    >>>
    >>> evaluator = AADEvaluator(window_s=[10, 30, 60])
    >>> evaluator.register_decoder("linear", LinearDecoder())
    >>>
    >>> # Load trials from BIDS dataset
    >>> trials = load_trials_from_bids("/data/my_study", subject="01", session="01")
    >>>
    >>> results = evaluator.evaluate(trials)
    >>> print(results.groupby(["decoder", "window_s"])["accuracy"].mean())
    """

    def __init__(
        self,
        window_s: float | list[float] = 60.0,
        n_jobs: int = 1,
    ) -> None:
        self.window_s = [window_s] if isinstance(window_s, (int, float)) else window_s
        self.n_jobs = n_jobs
        self._decoders: dict[str, BaseDecoder] = {}

    # ── Decoder registry ──────────────────────────────────────────────────────

    def register_decoder(self, name: str, decoder: BaseDecoder) -> None:
        """Register a decoder for evaluation.

        Parameters
        ----------
        name : str
            Human-readable name used in the results DataFrame.
        decoder : BaseDecoder
            Unfitted decoder instance.
        """
        self._decoders[name] = decoder
        logger.info("Registered decoder: %s (%s)", name, type(decoder).__name__)

    def registered_decoders(self) -> list[str]:
        return list(self._decoders.keys())

    # ── Main evaluation entry point ───────────────────────────────────────────

    def evaluate(self, trials: list[AADTrial]) -> pd.DataFrame:
        """Run leave-one-trial-out AAD evaluation for all registered decoders.

        Parameters
        ----------
        trials : list[AADTrial]
            All trials for a single subject/session. Each trial is the unit
            of cross-validation (held out one at a time).

        Returns
        -------
        results : pd.DataFrame
            Tidy results table with one row per (decoder, trial, window_s, window).
        """
        if not self._decoders:
            raise ValueError(
                "No decoders registered. Call register_decoder() first."
            )
        if len(trials) < 2:
            raise ValueError(
                f"Need at least 2 trials for leave-one-trial-out CV. Got {len(trials)}."
            )

        logger.info(
            "Evaluating %d decoders × %d trials × %d window sizes",
            len(self._decoders), len(trials), len(self.window_s),
        )

        all_rows: list[dict] = []

        for decoder_name, decoder_template in self._decoders.items():
            logger.info("  → Decoder: %s", decoder_name)
            rows = Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate_one_fold)(
                    fold_idx=i,
                    trials=trials,
                    decoder_template=decoder_template,
                    decoder_name=decoder_name,
                    window_sizes=self.window_s,
                )
                for i in range(len(trials))
            )
            for fold_rows in rows:
                all_rows.extend(fold_rows)

        df = pd.DataFrame(all_rows)
        logger.info("Evaluation complete. %d result rows.", len(df))
        return df

    # ── Per-fold evaluation ───────────────────────────────────────────────────

    @staticmethod
    def _evaluate_one_fold(
        fold_idx: int,
        trials: list[AADTrial],
        decoder_template: BaseDecoder,
        decoder_name: str,
        window_sizes: list[float],
    ) -> list[dict]:
        """Fit on N-1 trials, evaluate on the held-out trial."""
        import copy

        test_trial = trials[fold_idx]
        train_trials = [t for i, t in enumerate(trials) if i != fold_idx]
        fs = test_trial.fs

        # Concatenate all training trials
        train_eeg = np.concatenate([t.eeg for t in train_trials], axis=0)
        train_env = np.concatenate([t.env_attended for t in train_trials], axis=0)

        decoder = copy.deepcopy(decoder_template)
        try:
            decoder.fit(train_eeg, train_env, fs)
        except Exception as exc:
            logger.warning("Decoder %s failed on fold %d: %s", decoder_name, fold_idx, exc)
            return []

        rows: list[dict] = []
        for window_s in window_sizes:
            fold_rows = AADEvaluator._slide_window_decision(
                decoder=decoder,
                test_trial=test_trial,
                window_s=window_s,
                decoder_name=decoder_name,
            )
            rows.extend(fold_rows)

        return rows

    @staticmethod
    def _slide_window_decision(
        decoder: BaseDecoder,
        test_trial: AADTrial,
        window_s: float,
        decoder_name: str,
    ) -> list[dict]:
        """Slide a decision window across the test trial and collect decisions."""
        eeg = test_trial.eeg
        env_att = test_trial.env_attended
        env_ign = test_trial.env_ignored
        fs = test_trial.fs

        window_samples = int(window_s * fs)
        if window_samples > len(eeg):
            logger.debug(
                "Trial length %d < window %d samples. Using full trial.",
                len(eeg), window_samples,
            )
            window_samples = len(eeg)

        step = max(window_samples // 2, 1)   # 50% overlap
        rows: list[dict] = []

        for w_idx, start in enumerate(range(0, len(eeg) - window_samples + 1, step)):
            chunk_eeg = eeg[start: start + window_samples]
            chunk_att = env_att[start: start + window_samples]
            chunk_ign = env_ign[start: start + window_samples]

            try:
                env_hat = decoder.predict(chunk_eeg)
                # Trim envelopes to match predicted length (lag drops leading samples)
                n = len(env_hat)
                r_att = float(np.corrcoef(env_hat, chunk_att[-n:])[0, 1])
                r_ign = float(np.corrcoef(env_hat, chunk_ign[-n:])[0, 1])
            except Exception as exc:
                logger.debug("Window %d prediction failed: %s", w_idx, exc)
                continue

            # Handle NaN (can happen with constant signals)
            if np.isnan(r_att) or np.isnan(r_ign):
                continue

            correct = int(r_att > r_ign)
            rows.append({
                "decoder": decoder_name,
                "subject": test_trial.subject,
                "session": test_trial.session,
                "trial": test_trial.trial_idx,
                "window": w_idx,
                "window_s": window_s,
                "r_attended": r_att,
                "r_ignored": r_ign,
                "decision": correct,
            })

        # Append trial-level accuracy
        if rows:
            acc = float(np.mean([r["decision"] for r in rows]))
            for r in rows:
                r["accuracy"] = acc

        return rows

    # ── Convenience: aggregate results ────────────────────────────────────────

    @staticmethod
    def summarize(results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate results to decoder × window_s summary statistics.

        Returns
        -------
        summary : pd.DataFrame
            Columns: decoder, window_s, accuracy_mean, accuracy_std, n_trials.
        """
        return (
            results.groupby(["decoder", "window_s"])
            .agg(
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                r_attended_mean=("r_attended", "mean"),
                r_ignored_mean=("r_ignored", "mean"),
                n_windows=("decision", "count"),
            )
            .reset_index()
            .sort_values(["window_s", "accuracy_mean"], ascending=[True, False])
        )

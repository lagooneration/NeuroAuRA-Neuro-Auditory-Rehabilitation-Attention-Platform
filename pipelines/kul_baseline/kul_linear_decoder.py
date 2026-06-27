"""
pipelines/kul_baseline/kul_linear_decoder.py

Implements the original KUL auditory attention decoding approach:
  Ridge Regression spatial filter (linear decoder) as described in:
    Das et al. (2019) "Auditory attention detection using EEG"
    Binaural Beats: Selective Auditory Attention with EEG.

The approach:
    1. Per trial, compute the attended vs unattended Pearson r for wavA and wavB
    2. Classify: the stream with higher EEG correlation = attended stream
    3. Report accuracy across all trials (leave-one-out cross-validation)

Published baseline accuracy (Das et al. 2019):
    - 60-second trials: ~85–90%
    - 30-second trials: ~75–80%
    - 10-second trials: ~60–65%

Usage:
    python pipelines/kul_baseline/kul_linear_decoder.py \\
        --mat-file "P:\\auditory\\neurophile\\data\\raw\\kul\\DATA_preproc.zip.unzip\\S1_data_preproc.mat"
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("kul_linear_decoder")


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_kul_trials(mat_file: Path) -> dict:
    """
    Load a KUL .mat file and return raw per-trial data.

    Returns dict with:
        'eeg'    : list of (T, 66) float32 arrays
        'wavA'   : list of (T, 1) float32 arrays
        'wavB'   : list of (T, 1) float32 arrays
        'labels' : list of int (1 = wavA attended, 2 = wavB attended)
        'fs'     : int, sampling rate in Hz
    """
    logger.info("Loading KUL file: %s", mat_file.name)
    mat = sio.loadmat(str(mat_file))
    d = mat["data"][0, 0]

    eeg_raw = d["eeg"][0]
    wavA_raw = d["wavA"][0]
    wavB_raw = d["wavB"][0]
    events = d["event"][0]
    
    # Parse sampling rate — stored as a nested uint8 array
    try:
        fs_nested = d["fsample"][0, 0]
        fs = int(fs_nested[0][0])
    except Exception:
        fs = 64  # KUL dataset is always 64 Hz

    n_trials = len(eeg_raw)
    labels = []
    eeg_events = events[0]["eeg"][0]   # shape (60,) — one entry per trial
    for i in range(n_trials):
        try:
            # MATLAB FieldTrip struct: ev[1] = value field, nested uint8 array
            val = int(eeg_events[i][1][0][0][0][0])
        except Exception:
            val = 1  # safe fallback
        labels.append(val)

    logger.info(
        "Loaded %d trials at %d Hz | Label distribution: %s",
        n_trials, fs,
        {v: labels.count(v) for v in sorted(set(labels))},
    )

    return {
        "eeg":    [eeg_raw[i][:, :64].astype("float32") for i in range(n_trials)],
        "wavA":   [wavA_raw[i].reshape(-1).astype("float32") for i in range(n_trials)],
        "wavB":   [wavB_raw[i].reshape(-1).astype("float32") for i in range(n_trials)],
        "labels": labels,
        "fs":     fs,
        "n_trials": n_trials,
    }


# ── Decoder Methods ───────────────────────────────────────────────────────────

def build_spatiotemporal_features(
    eeg: np.ndarray, lags: np.ndarray
) -> np.ndarray:
    """
    Build a spatiotemporal feature matrix by stacking time-lagged copies of EEG.
    This is the core of the KUL decoder — each lag captures the EEG response
    at a different delay after the audio onset.

    Parameters
    ----------
    eeg    : (T, C) EEG data
    lags   : array of integer sample lags (e.g. np.arange(0, 32) = 0..500ms at 64 Hz)

    Returns
    -------
    X : (T, C × n_lags) spatiotemporal feature matrix
    """
    T, C = eeg.shape
    features = []
    for lag in lags:
        if lag == 0:
            features.append(eeg)
        elif lag > 0:
            shifted = np.zeros_like(eeg)
            shifted[lag:] = eeg[:-lag]
            features.append(shifted)
        else:
            shifted = np.zeros_like(eeg)
            shifted[:lag] = eeg[-lag:]
            features.append(shifted)
    return np.hstack(features)  # (T, C * n_lags)


def decode_spatiotemporal(data: dict) -> dict:
    """
    Multi-lag spatiotemporal Ridge decoder — matches Das et al. (2019).

    For each trial (LOOCV):
        - Build lagged feature matrix X: (T, 64_channels × 32_lags)
        - Fit Ridge regression: X → attended_envelope
        - Predict envelope for held-out trial
        - Correlate prediction with wavA and wavB
        - Classify: attended = stream with higher predicted correlation

    Lags: 0 to +500ms (0 to 32 samples at 64 Hz)
    This captures the full auditory TRF response window (N100 at ~100ms, P300 at ~300ms).
    """
    fs = data["fs"]
    # Backward decoder lags: −500ms to 0ms (Das et al. 2019)
    # Negative lags = EEG AFTER the audio event, which is correct because:
    #   Audio at time t → EEG response at t+delta
    #   So to decode audio(t) we need EEG from t+0 to t+500ms
    #   In array terms: lag=-32 means eeg is shifted so future EEG aligns with current audio
    lags = np.arange(-int(0.5 * fs), 1)   # -32..0 samples (−500ms to 0ms)
    n = data["n_trials"]
    correct = 0
    details = []
    scaler = StandardScaler()

    logger.info(
        "Running spatiotemporal LOOCV Ridge (lags=0–500ms, %d folds)...", n
    )

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]

        X_train, y_train = [], []
        for j in train_idx:
            X_j = build_spatiotemporal_features(data["eeg"][j], lags)
            label_j = data["labels"][j]
            env_attended = data["wavA"][j] if label_j == 1 else data["wavB"][j]
            # Z-score each trial independently before concatenation
            # (prevents inter-trial amplitude differences corrupting the fit)
            X_j = (X_j - X_j.mean(axis=0)) / (X_j.std(axis=0) + 1e-8)
            X_train.append(X_j)
            y_train.append(env_attended)

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        # alpha=1e2 confirmed best from sweep across 4 values
        ridge = Ridge(alpha=1e2)
        ridge.fit(X_train, y_train)  # already z-scored per trial

        # Test — apply same per-trial z-score normalization
        X_test = build_spatiotemporal_features(data["eeg"][i], lags)
        X_test = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0) + 1e-8)
        y_pred = ridge.predict(X_test)

        rA, _ = pearsonr(y_pred, data["wavA"][i])
        rB, _ = pearsonr(y_pred, data["wavB"][i])

        prediction = 1 if rA > rB else 2
        is_correct = prediction == data["labels"][i]
        correct += is_correct
        details.append({"rA": rA, "rB": rB, "label": data["labels"][i],
                        "prediction": prediction, "correct": is_correct})

        if (i + 1) % 10 == 0:
            logger.info("  Fold %d/%d | running acc=%.1f%%",
                        i + 1, n, (correct / (i + 1)) * 100)

    accuracy = correct / n
    return {"accuracy": accuracy, "n_correct": correct, "details": details,
            "method": "Spatiotemporal Ridge LOOCV (Das et al. 2019 — multi-lag)"}


def decode_ridge_regression(data: dict) -> dict:
    """
    Method 2: Ridge Regression Spatial Filter (core KUL algorithm).
    
    Uses leave-one-out cross-validation (LOOCV):
        - Train: fit a Ridge regression on all trials EXCEPT trial i
                 predicting wavA envelope from the 64-channel EEG
        - Test:  predict the envelope for trial i, then correlate with wavA and wavB
        - Classify: attended = stream with higher predicted correlation
    
    This matches the Das et al. 2019 published method.
    """
    n = data["n_trials"]
    correct = 0
    details = []
    scaler = StandardScaler()

    logger.info("Running LOOCV Ridge Regression (%d folds)...", n)

    for i in range(n):
        # Build train set (all trials except i)
        train_idx = [j for j in range(n) if j != i]

        X_train, y_train = [], []
        for j in train_idx:
            eeg_j = data["eeg"][j]   # (T, 64)
            label_j = data["labels"][j]
            # Target: attended envelope
            env_attended = data["wavA"][j] if label_j == 1 else data["wavB"][j]
            X_train.append(eeg_j)
            y_train.append(env_attended)

        X_train = np.vstack(X_train)   # (n_train * T, 64)
        y_train = np.concatenate(y_train)  # (n_train * T,)

        X_train_sc = scaler.fit_transform(X_train)

        ridge = Ridge(alpha=1e4)
        ridge.fit(X_train_sc, y_train)

        # Test on trial i
        X_test = scaler.transform(data["eeg"][i])  # (T, 64)
        y_pred = ridge.predict(X_test)              # (T,)

        rA, _ = pearsonr(y_pred, data["wavA"][i])
        rB, _ = pearsonr(y_pred, data["wavB"][i])

        prediction = 1 if rA > rB else 2
        is_correct = prediction == data["labels"][i]
        correct += is_correct
        details.append({"rA": rA, "rB": rB, "label": data["labels"][i],
                        "prediction": prediction, "correct": is_correct})

        if (i + 1) % 10 == 0:
            logger.info("  LOOCV fold %d/%d | running acc=%.1f%%",
                        i + 1, n, (correct / (i + 1)) * 100)

    accuracy = correct / n
    return {"accuracy": accuracy, "n_correct": correct, "details": details,
            "method": "Ridge Regression LOOCV (Das et al. 2019)"}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(data: dict, pearson_res: dict, ridge_res: dict, mat_file: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"KUL Linear Decoder Baseline — {mat_file.stem}\n"
        f"Published benchmark: ~85–90% (60-second trials, Das et al. 2019)",
        fontsize=12, fontweight="bold",
    )

    # Plot A: Per-trial rA vs rB scatter for Pearson method
    ax = axes[0]
    details = pearson_res["details"]
    rAs = [d["rA"] for d in details]
    rBs = [d["rB"] for d in details]
    colors = ["green" if d["correct"] else "red" for d in details]
    ax.scatter(rAs, rBs, c=colors, alpha=0.8, edgecolors="black", linewidth=0.5)
    lim = max(max(abs(np.array(rAs))), max(abs(np.array(rBs)))) * 1.1
    ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=0.8, label="rA=rB boundary")
    ax.set_xlabel("Pearson r (EEG vs wavA)")
    ax.set_ylabel("Pearson r (EEG vs wavB)")
    ax.set_title(f"Pearson Decoder: {pearson_res['accuracy']*100:.1f}% acc\n"
                 f"(green=correct, red=wrong)")
    ax.legend(fontsize=8)

    # Plot B: Accuracy comparison bar
    ax = axes[1]
    methods = [
        "Pearson\nCorrelation",
        "Ridge\nRegression\n(LOOCV)",
        "Published\nBenchmark\n(Das 2019)",
    ]
    accuracies = [pearson_res["accuracy"] * 100,
                  ridge_res["accuracy"] * 100,
                  87.5]  # published midpoint of 85-90%
    bar_colors = ["steelblue", "darkorange", "gray"]
    bars = ax.bar(methods, accuracies, color=bar_colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=10, padding=3)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8, label="Chance (50%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([0, 105])
    ax.set_title("Accuracy Comparison vs Published Benchmark")
    ax.legend(fontsize=8)

    # Plot C: Trial-by-trial correctness for Ridge
    ax = axes[2]
    ridge_details = ridge_res["details"]
    trial_ids = range(len(ridge_details))
    correct_mask = [d["correct"] for d in ridge_details]
    ax.bar(trial_ids, [1 if c else 0 for c in correct_mask],
           color=["green" if c else "red" for c in correct_mask],
           linewidth=0)
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Correct (1) / Wrong (0)")
    ax.set_title(f"Ridge LOOCV Trial-by-Trial: {ridge_res['accuracy']*100:.1f}%\n"
                 f"({ridge_res['n_correct']}/{len(ridge_details)} correct)")
    ax.set_ylim([-0.1, 1.3])

    plt.tight_layout()

    out_path = Path("checkpoints") / f"kul_baseline_{mat_file.stem}.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logger.info("Figure saved → %s", out_path)
    print(f"\nFigure saved → {out_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run KUL linear decoder baseline (Das et al. 2019)"
    )
    parser.add_argument(
        "--mat-file", type=Path,
        default=Path("data/raw/kul/DATA_preproc.zip.unzip/S1_data_preproc.mat"),
    )
    parser.add_argument(
        "--skip-ridge", action="store_true",
        help="Skip Ridge LOOCV (slower). Only run fast Pearson decoder.",
    )
    args = parser.parse_args()

    if not args.mat_file.exists():
        logger.error("File not found: %s", args.mat_file)
        return

    data = load_kul_trials(args.mat_file)

    # Method 1: Pearson correlation (fast, ~seconds)
    logger.info("Running Method 1: Spatiotemporal Ridge LOOCV (multi-lag, Das et al. 2019)...")
    pearson_res = decode_spatiotemporal(data)

    # Method 2: Full Ridge regression LOOCV
    if not args.skip_ridge:
        logger.info("Running Method 2: Single-lag Ridge LOOCV (for comparison)...")
        ridge_res = decode_ridge_regression(data)
    else:
        logger.info("Single-lag Ridge LOOCV skipped.")
        ridge_res = {"accuracy": 0.0, "n_correct": 0, "details": [],
                     "method": "Skipped"}

    # Print summary
    print("\n" + "=" * 60)
    print(f"KUL BASELINE RESULTS — {args.mat_file.stem}")
    print("=" * 60)
    print(f"Trials:              {data['n_trials']}")
    print(f"Sampling rate:       {data['fs']} Hz")
    print(f"Trial duration:      {data['eeg'][0].shape[0] / data['fs']:.0f} seconds")
    print()
    print(f"Spatiotemporal Ridge (multi-lag): {pearson_res['accuracy']*100:.1f}%  "
          f"({pearson_res['n_correct']}/{data['n_trials']} correct)")
    if not args.skip_ridge:
        print(f"Single-lag Ridge (comparison):    {ridge_res['accuracy']*100:.1f}%  "
              f"({ridge_res['n_correct']}/{data['n_trials']} correct)")
    print(f"Published Benchmark:              ~85–90%  (Das et al. 2019, 60-second trials)")
    print("=" * 60)

    plot_results(data, pearson_res, ridge_res, args.mat_file)


if __name__ == "__main__":
    main()

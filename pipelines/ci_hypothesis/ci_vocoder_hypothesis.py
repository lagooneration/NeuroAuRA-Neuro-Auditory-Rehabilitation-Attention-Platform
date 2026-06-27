"""
pipelines/ci_hypothesis/ci_vocoder_hypothesis.py
=================================================
CI Vocoder Hypothesis Experiment

Tests how much auditory attention decoding correlation drops when the
target audio is degraded through a modern 22-channel Cochlear Implant
simulation. Compares the classical Linear Ridge decoder against the
Deep Learning CRN (MesgaraniAdapter).

Pipeline:
  1. Load KUL Subject EEG + pre-extracted 64Hz envelopes
  2. Align each 60-second trial envelope to the raw .wav audio via
     cross-correlation to find the exact trial offset
  3. Slice the exact 60s of high-resolution (44.1kHz) audio per trial
  4. CI-vocode sliced audio through 22-channel CIVocoderSimulator
  5. Extract 64Hz CI envelope from vocoded audio
  6. Run Decoder 1: Spatiotemporal Ridge (LOOCV) - Dry vs CI envelope
  7. Run Decoder 2: MesgaraniAdapter DL (train/test split) - Dry vs CI
  8. Print comparison table

Usage:
    cd P:\\auditory\\neurophile
    python pipelines/ci_hypothesis/ci_vocoder_hypothesis.py ^
        --mat-file "data/raw/kul/DATA_preproc.zip.unzip/S1_data_preproc.mat" ^
        --stimuli-dir "stimuli/stimuli" ^
        --n-trials 10 ^
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))

from neurophile.stimulus.ci_vocoder import CIVocoderSimulator
from neurophile.models import MesgaraniAdapter, GlobalCITrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ci_hypothesis")

# ── Constants ──────────────────────────────────────────────────────────────────
FS_EEG = 64            # Hz – KUL EEG sampling rate
FS_AUDIO = 44100       # Hz – raw .wav sampling rate
CI_CHANNELS = 22       # Modern CI device channel count
CI_F_LOW = 200.0       # Hz – CI tonotopy lower edge
CI_F_HIGH = 7000.0     # Hz – CI tonotopy upper edge
N_LAGS = int(0.5 * FS_EEG)  # 500ms TRF window = 32 samples at 64Hz


# ── Step 1: Load KUL trials ────────────────────────────────────────────────────

def load_kul_trials(mat_file: Path, n_trials: int | None) -> dict:
    """Load EEG + pre-extracted envelopes from a KUL .mat file."""
    logger.info("Loading KUL file: %s", mat_file.name)
    mat = sio.loadmat(str(mat_file))
    d = mat["data"][0, 0]

    eeg_raw = d["eeg"][0]
    wavA_raw = d["wavA"][0]
    wavB_raw = d["wavB"][0]
    events = d["event"][0]
    eeg_events = events[0]["eeg"][0]

    try:
        fs_nested = d["fsample"][0, 0]
        fs = int(fs_nested[0][0])
    except Exception:
        fs = FS_EEG

    n = len(eeg_raw)
    if n_trials:
        n = min(n, n_trials)

    labels = []
    for i in range(n):
        try:
            val = int(eeg_events[i][1][0][0][0][0])
        except Exception:
            val = 1
        labels.append(val)

    return {
        "eeg":     [eeg_raw[i][:, :64].astype("float32") for i in range(n)],
        "envA":    [wavA_raw[i].reshape(-1).astype("float32") for i in range(n)],
        "envB":    [wavB_raw[i].reshape(-1).astype("float32") for i in range(n)],
        "labels":  labels,
        "fs":      fs,
        "n_trials": n,
    }


# ── Step 2: Load raw audio ────────────────────────────────────────────────────

def load_wav_audio(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load a .wav file and return (audio_float32_mono, sample_rate)."""
    try:
        import soundfile as sf
    except ImportError:
        logger.error("soundfile not installed: pip install soundfile")
        sys.exit(1)

    audio, fs = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if fs != FS_AUDIO:
        g = int(np.gcd(FS_AUDIO, fs))
        audio = resample_poly(audio, FS_AUDIO // g, fs // g).astype("float32")
        fs = FS_AUDIO

    return audio, fs


# ── Step 3-4: CI Vocoder (whole file) ─────────────────────────────────────────

def vocode_whole_wav(full_audio: np.ndarray) -> np.ndarray:
    """CI-vocode the entire .wav audio and return the 64Hz CI envelope."""
    vocoder = CIVocoderSimulator(
        fs=FS_AUDIO,
        n_channels=CI_CHANNELS,
        f_low=CI_F_LOW,
        f_high=CI_F_HIGH,
        carrier="noise",
    )
    _, ci_env = vocoder.simulate_and_extract_envelope(full_audio, fs_eeg=FS_EEG)
    return ci_env.astype("float32")


# ── Step 5: Align KUL dry envelope → CI envelope and slice ────────────────────

def align_kul_env_to_ci_env(
    trial_env: np.ndarray, ci_env_full: np.ndarray
) -> tuple[int, float]:
    """Cross-correlate the KUL trial envelope against the full CI envelope.

    Both signals share the same acoustic source but CI vocoding preserves
    the low-frequency amplitude modulations — enough for cross-correlation
    to find the correct offset.

    Returns (best_offset_in_64Hz_samples, pearson_r_at_best_offset).
    """
    t = (trial_env - trial_env.mean()) / (trial_env.std() + 1e-8)
    w = (ci_env_full - ci_env_full.mean()) / (ci_env_full.std() + 1e-8)
    T = len(t)
    W = len(w)

    # Use scipy FFT-based cross-correlation for speed
    from scipy.signal import correlate
    xcorr = correlate(w, t, mode="valid")  # length = W - T + 1
    best_offset = int(np.argmax(xcorr))

    seg = w[best_offset: best_offset + T]
    r = float(pearsonr(t, seg)[0])
    return best_offset, r


def slice_ci_envelope(
    ci_env_full: np.ndarray, offset: int, n_samples: int
) -> np.ndarray:
    """Slice and pad/trim a CI envelope to exactly n_samples."""
    sliced = ci_env_full[offset: offset + n_samples]
    if len(sliced) < n_samples:
        sliced = np.pad(sliced, (0, n_samples - len(sliced)))
    return sliced[:n_samples].astype("float32")


# ── Step 5 orchestrator ───────────────────────────────────────────────────────

def build_ci_trial_data(
    kul: dict,
    stimuli_dir: Path,
    cache_dir: Path,
) -> dict:
    """
    For each trial:
      1. CI-vocode every candidate .wav file (whole file, once each).
      2. Cross-correlate the KUL dry envelope (already correct) against the
         CI envelope to find the trial offset — CI preserves low-freq AM.
      3. Slice the CI envelope at the matched offset.

    Augments kul dict with ci_envA and ci_envB lists.
    """
    # Only use main story parts (not rep_* which are shorter repeat stimuli)
    wav_files = sorted(stimuli_dir.glob("part*_dry.wav"))
    if not wav_files:
        # Fall back to all _dry.wav if naming differs
        wav_files = sorted(stimuli_dir.glob("*_dry.wav"))
    if not wav_files:
        logger.error("No *_dry.wav files found in %s", stimuli_dir)
        sys.exit(1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ci{CI_CHANNELS}ch_{int(CI_F_LOW)}-{int(CI_F_HIGH)}hz"

    logger.info(
        "Step 3: CI-vocoding %d .wav files (22ch, %.0f\u2013%.0f Hz) "
        "[cache: %s]\u2026",
        len(wav_files), CI_F_LOW, CI_F_HIGH, cache_dir,
    )
    wav_ci_envs = {}  # filename \u2192 ci_env_64Hz array
    for wf in wav_files:
        cache_path = cache_dir / f"{wf.stem}_{tag}.npy"
        if cache_path.exists():
            ci_env = np.load(str(cache_path))
            logger.info("  Loaded from cache: %s (%d samples)", wf.name, len(ci_env))
        else:
            logger.info("  Vocoding %s (first run; will cache)\u2026", wf.name)
            audio, _ = load_wav_audio(wf)
            ci_env = vocode_whole_wav(audio)
            np.save(str(cache_path), ci_env)
            logger.info("    \u2192 %d CI samples (%.1fs) \u2014 cached to %s",
                        len(ci_env), len(ci_env) / FS_EEG, cache_path.name)
        wav_ci_envs[wf.name] = ci_env

    ci_envA_list = []
    ci_envB_list = []

    for i in range(kul["n_trials"]):
        trial_envA = kul["envA"][i]
        trial_envB = kul["envB"][i]
        T = len(trial_envA)

        # Find best-matching CI envelope for stream A
        best_r_A = -np.inf
        best_offset_A = 0
        best_wav_A = None
        for wname, ci_env_full in wav_ci_envs.items():
            if len(ci_env_full) < T:
                continue
            offset, r = align_kul_env_to_ci_env(trial_envA, ci_env_full)
            if r > best_r_A:
                best_r_A = r
                best_offset_A = offset
                best_wav_A = wname

        # Find best-matching CI envelope for stream B
        best_r_B = -np.inf
        best_offset_B = 0
        best_wav_B = None
        for wname, ci_env_full in wav_ci_envs.items():
            if len(ci_env_full) < T:
                continue
            offset, r = align_kul_env_to_ci_env(trial_envB, ci_env_full)
            if r > best_r_B:
                best_r_B = r
                best_offset_B = offset
                best_wav_B = wname

        logger.info(
            "Trial %02d | A→%s offset=%d (r=%.3f) | B→%s offset=%d (r=%.3f)",
            i, best_wav_A, best_offset_A, best_r_A,
            best_wav_B, best_offset_B, best_r_B,
        )

        ci_A = slice_ci_envelope(wav_ci_envs[best_wav_A], best_offset_A, T)
        ci_envA_list.append(ci_A)

        ci_B = slice_ci_envelope(wav_ci_envs[best_wav_B], best_offset_B, T)
        ci_envB_list.append(ci_B)

    kul["ci_envA"] = ci_envA_list
    kul["ci_envB"] = ci_envB_list
    return kul


# ── Step 6: Ridge Regression Decoder ─────────────────────────────────────────

def build_lagged_features(eeg: np.ndarray, lags: np.ndarray) -> np.ndarray:
    T, C = eeg.shape
    features = []
    for lag in lags:
        shifted = np.zeros_like(eeg)
        if lag == 0:
            shifted = eeg
        elif lag > 0:
            shifted[lag:] = eeg[:-lag]
        else:
            shifted[:lag] = eeg[-lag:]
        features.append(shifted)
    return np.hstack(features)


def run_ridge_decoder(
    kul: dict, use_ci: bool = False
) -> dict:
    """LOOCV Ridge decoder matching Das et al. (2019).

    Parameters
    ----------
    use_ci : if True, use ci_envA/ci_envB for the decoding target instead of
             the original dry envelopes.
    """
    mode = "CI-vocoded" if use_ci else "Dry"
    logger.info("Ridge LOOCV (%s envelopes)...", mode)
    lags = np.arange(-N_LAGS, 1)
    n = kul["n_trials"]
    correct = 0
    r_vals = []

    envA_key = "ci_envA" if use_ci else "envA"
    envB_key = "ci_envB" if use_ci else "envB"

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]

        X_train, y_train = [], []
        for j in train_idx:
            X_j = build_lagged_features(kul["eeg"][j], lags)
            label_j = kul["labels"][j]
            env_att = kul[envA_key][j] if label_j == 1 else kul[envB_key][j]
            # Per-trial z-score (matches Das et al. 2019 exactly)
            X_j = (X_j - X_j.mean(0)) / (X_j.std(0) + 1e-8)
            X_train.append(X_j)
            y_train.append(env_att)

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        # Fit Ridge directly on per-trial normalized data (no extra StandardScaler)
        clf = Ridge(alpha=100.0, fit_intercept=False)
        clf.fit(X_train, y_train)

        X_test = build_lagged_features(kul["eeg"][i], lags)
        X_test = (X_test - X_test.mean(0)) / (X_test.std(0) + 1e-8)
        y_pred = clf.predict(X_test)

        envA_i = kul[envA_key][i]
        envB_i = kul[envB_key][i]
        rA = pearsonr(y_pred, envA_i)[0]
        rB = pearsonr(y_pred, envB_i)[0]

        label_i = kul["labels"][i]
        r_attended = rA if label_i == 1 else rB
        r_vals.append(r_attended)

        pred_label = 1 if rA > rB else 2
        if pred_label == label_i:
            correct += 1
        logger.debug("  Fold %02d: pred=%d true=%d rA=%.3f rB=%.3f",
                     i, pred_label, label_i, rA, rB)

    accuracy = correct / n
    mean_r = float(np.mean(r_vals))
    logger.info("  Ridge [%s]: Accuracy=%.1f%%, Mean r=%.4f", mode, accuracy * 100, mean_r)
    return {"mode": mode, "decoder": "Ridge", "accuracy": accuracy, "mean_r": mean_r}


# ── Step 7: Deep Learning Decoder ─────────────────────────────────────────────

def run_dl_decoder(kul: dict, use_ci: bool, device: str) -> dict:
    """Train MesgaraniAdapter on Dry EEG, test against Dry and CI envelopes."""
    mode = "CI-vocoded" if use_ci else "Dry"
    logger.info("DL MesgaraniAdapter (%s envelopes)...", mode)

    n = kul["n_trials"]
    split = int(n * 0.8)

    # Always train on DRY envelopes (that's what the brain was listening to)
    eeg_list, env_list, lbl_list = [], [], []
    for i in range(n):
        eeg_t = kul["eeg"][i]
        # Attended vs unattended sample pairs
        label_i = kul["labels"][i]
        envA_dry = kul["envA"][i].reshape(-1, 1)
        envB_dry = kul["envB"][i].reshape(-1, 1)

        eeg_list.append(eeg_t)
        env_list.append(envA_dry)
        lbl_list.append(1.0 if label_i == 1 else 0.0)

        eeg_list.append(eeg_t)
        env_list.append(envB_dry)
        lbl_list.append(1.0 if label_i == 2 else 0.0)

    eeg_arr = np.stack(eeg_list)
    env_arr = np.stack(env_list)
    lbl_arr = np.array(lbl_list, dtype="float32")

    train_eeg, test_eeg = eeg_arr[:split*2], eeg_arr[split*2:]
    train_env, test_env = env_arr[:split*2], env_arr[split*2:]
    train_lbl, test_lbl = lbl_arr[:split*2], lbl_arr[split*2:]

    model = MesgaraniAdapter(num_eeg_channels=64, audio_sampling_rate=FS_EEG)
    trainer = GlobalCITrainer(
        model=model, epochs=30, batch_size=4,
        device=device, output_dir=Path("checkpoints"),
    )
    trainer.train(train_eeg, train_env, train_lbl,
                  val_eeg=test_eeg, val_env=test_env, val_lbl=test_lbl)

    # Now evaluate: replace the test envelopes with CI-vocoded if requested
    if use_ci:
        ci_env_list = []
        # Rebuild test env using CI envelopes for the test trials
        for i in range(split, n):
            envA_ci = kul["ci_envA"][i].reshape(-1, 1)
            envB_ci = kul["ci_envB"][i].reshape(-1, 1)
            ci_env_list.append(envA_ci)
            ci_env_list.append(envB_ci)
        test_env = np.stack(ci_env_list)

    metrics = trainer.evaluate(test_eeg, test_env, test_lbl)
    accuracy = metrics["accuracy"]
    mean_r = metrics["mean_pearson_r"]
    logger.info("  DL [%s]: Accuracy=%.1f%%, Mean r=%.4f", mode, accuracy * 100, mean_r)
    return {"mode": mode, "decoder": "DL (CRN)", "accuracy": accuracy, "mean_r": mean_r}


# ── Step 8: Print Results Table ────────────────────────────────────────────────

def print_results_table(results: list[dict]) -> None:
    """Print a comparison table of all decoder results."""
    print("\n")
    print("=" * 72)
    print(" CI VOCODER HYPOTHESIS — RESULTS")
    print(" 22-channel modern CI simulation vs Normal Hearing (Dry)")
    print("=" * 72)
    print(f"{'Decoder':<18} {'Audio':<14} {'Accuracy':>10} {'Mean r':>10}")
    print("-" * 72)
    for r in results:
        print(f"{r['decoder']:<18} {r['mode']:<14} {r['accuracy']*100:>9.1f}% {r['mean_r']:>10.4f}")
    print("=" * 72)

    # Delta summary
    if len(results) == 4:
        ridge_drop = results[0]["accuracy"] - results[1]["accuracy"]
        dl_drop = results[2]["accuracy"] - results[3]["accuracy"]
        print(f"\n📉 Ridge accuracy drop from CI degradation:  {ridge_drop*100:+.1f}%")
        print(f"🧠 DL accuracy drop from CI degradation:     {dl_drop*100:+.1f}%")
        if dl_drop < ridge_drop:
            print("✅ HYPOTHESIS SUPPORTED: DL is more resilient to CI degradation than Ridge!")
        else:
            print("❌ HYPOTHESIS NOT SUPPORTED: Ridge and DL show similar degradation.")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CI Vocoder Hypothesis: Ridge vs DL on 22ch CI audio"
    )
    parser.add_argument(
        "--mat-file", type=Path, required=True,
        help="Path to a KUL S*_data_preproc.mat file"
    )
    parser.add_argument(
        "--stimuli-dir", type=Path, required=True,
        help="Path to the folder containing the *_dry.wav files"
    )
    parser.add_argument(
        "--n-trials", type=int, default=10,
        help="Max number of trials to use (default: 10 for speed)"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="PyTorch device for DL decoder (cuda or cpu)"
    )
    parser.add_argument(
        "--skip-dl", action="store_true",
        help="Only run Ridge decoder (faster, no GPU required)"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("checkpoints/ci_cache"),
        help="Directory to cache pre-computed CI envelopes (default: checkpoints/ci_cache)"
    )
    args = parser.parse_args()

    t_start = time.time()

    # ── Load KUL EEG data
    kul = load_kul_trials(args.mat_file, args.n_trials)
    logger.info("Loaded %d trials from %s", kul["n_trials"], args.mat_file.name)

    # ── Align + CI-vocode each trial
    logger.info("Step 2-4: Aligning trials to .wav and CI-vocoding (22ch, %.0f-%.0f Hz)...",
                CI_F_LOW, CI_F_HIGH)
    kul = build_ci_trial_data(kul, args.stimuli_dir, args.cache_dir)

    results = []

    # ── Ridge: Dry
    results.append(run_ridge_decoder(kul, use_ci=False))

    # ── Ridge: CI
    results.append(run_ridge_decoder(kul, use_ci=True))

    if not args.skip_dl:
        # ── DL: Dry
        results.append(run_dl_decoder(kul, use_ci=False, device=args.device))

        # ── DL: CI
        results.append(run_dl_decoder(kul, use_ci=True, device=args.device))

    print_results_table(results)
    logger.info("Total runtime: %.1f seconds", time.time() - t_start)


if __name__ == "__main__":
    main()

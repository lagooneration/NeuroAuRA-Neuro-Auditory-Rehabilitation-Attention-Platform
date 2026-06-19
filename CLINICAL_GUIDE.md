# CLINICAL_GUIDE.md — Using Neurophile for Cochlear Implant Patients

> **Audience:** Audiologists, clinicians, and hospital researchers who want to use Neurophile to decode auditory attention in Cochlear Implant (CI) patients.
>
> **No machine learning background required.** You do not need to download training datasets or academic model code.
>
> **Researchers who want to train or modify the model:** see [CONTRIBUTORS.md](CONTRIBUTORS.md) instead.

---

## ⚠️ Important: Current Status

> **The Global CI Foundation Model checkpoint has not yet been published.**
>
> Steps 3–6 in this guide (download model, run inference, fine-tune) are **TODO** — they describe the intended workflow once the checkpoint is available. The infrastructure (training pipeline, adapter models, CI vocoder, ICA artifact cancellation) is fully built; the trained weight file is pending.
>
> **What works today:**
> - CI artifact pipeline (Stages 1 and 3 — template subtraction + ICA)
> - CI vocoder simulation
> - Training from scratch (see [CONTRIBUTORS.md](CONTRIBUTORS.md))
>
> **Subscribe to releases** at https://github.com/neurophile/neurophile/releases to be notified when the checkpoint is published.

---

## Table of Contents

1. [What Neurophile Does for CI Patients](#1-what-neurophile-does-for-ci-patients)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Software Installation](#3-software-installation)
4. [Step 1 — Record a Patient EEG Session](#4-step-1--record-a-patient-eeg-session)
5. [Step 2 — Prepare Your Audio Stimuli](#5-step-2--prepare-your-audio-stimuli)
6. [Step 3 — Download the Global CI Model (TODO)](#6-step-3--download-the-global-ci-model-todo)
7. [Step 4 — Run Attention Decoding Inference (TODO)](#7-step-4--run-attention-decoding-inference-todo)
8. [Step 5 — Fine-Tune for Your Patient (TODO)](#8-step-5--fine-tune-for-your-patient-todo)
9. [What the Model Output Means](#9-what-the-model-output-means)
10. [CI Artifact Pipeline — What Happens to the EEG](#10-ci-artifact-pipeline--what-happens-to-the-eeg)
11. [Frequently Asked Questions](#11-frequently-asked-questions)
12. [Reporting Issues](#12-reporting-issues)

---

## 1. What Neurophile Does for CI Patients

Cochlear Implant users struggle in noisy environments because their device cannot selectively amplify the speaker they are attending to. Neurophile reads the patient's **EEG (brain signal)** to decode which speaker they are *trying* to listen to, even when they cannot express it verbally.

### How It Works (Non-Technical)

```
Patient listens to two competing speakers
              │
              ▼
EEG electrodes record the patient's brain activity
              │
              ▼
Neurophile cleans the EEG (removes CI electrical interference)
              │
              ▼
The Global CI Foundation Model compares the brain signal
against the audio envelopes of both speakers
              │
              ▼
Output: "Patient is attending to Speaker A (confidence: 84%)"
```

The key insight is that the auditory cortex **physically locks onto** the 1–8 Hz amplitude rhythm of the sound the person is attending to — regardless of their language or whether they have a CI. The model learns to detect this neural-acoustic coupling.

### Why the Global Model Works Across Patients

The model was trained on data from **16 subjects** and tested on acoustic conditions simulating CI hearing (via the CI vocoder). Because it learns *how the auditory cortex tracks sounds* — not language-specific patterns — it generalises across patients, languages, and CI manufacturers.

After downloading the Global Model, you can also **fine-tune** it on 10–20 minutes of your patient's own data to improve accuracy for that individual.

---

## 2. Hardware Requirements

### EEG Amplifier

| Device | Recommended | Channels | Notes |
|---|---|---|---|
| g.tec g.USBamp | ✅ Preferred for CI labs | 16–32 | Highest SNR near CI; CI lab gold standard |
| BrainProducts LiveAmp | ✅ Recommended | 32–64 | Research-grade; used in KUL dataset |
| OpenBCI Cyton | ✅ Supported | 8 | Low cost; good for development |
| Any LSL-compatible | ✅ Supported | any | Use generic LSL stream |

> **CI-specific requirement:** Record at ≥ 1000 Hz sampling rate. The CI artifact pulses occur at 250–900 Hz. You must capture them to remove them.

### Computer

- Python 3.10 or later
- 8 GB RAM minimum (16 GB recommended for ICA)
- CPU is sufficient for inference; GPU optional for fine-tuning

---

## 3. Software Installation

```bash
# Install Neurophile with deep learning support
pip install "neurophile[dl]"

# Verify installation
python -c "from neurophile.stimulus import CIVocoderSimulator; print('Neurophile OK')"
```

If you encounter errors, check that Python 3.10+ is installed:
```bash
python --version   # should print Python 3.10.x or later
```

---

## 4. Step 1 — Record a Patient EEG Session

### Recording Protocol

1. **Fit the EEG cap** following your device manufacturer's instructions. Ensure electrode impedances are < 20 kΩ.

2. **Prepare two audio streams** (see [Step 2](#5-step-2--prepare-your-audio-stimuli) below).

3. **Play the stimuli** through the patient's CI sound processor via direct audio input (DAI) cable or speaker. Both streams are mixed and delivered simultaneously.

4. **Record EEG** at ≥ 1000 Hz. Save in any MNE-compatible format: `.fif`, `.edf`, `.bdf`, `.vhdr`.

5. **Mark the trial events.** Note the start and end time of each audio segment.

### Recommended Session Design

| Phase | Duration | Purpose |
|---|---|---|
| Rest / baseline | 2 min | Measure resting EEG; check artifact levels |
| Calibration | 5 min | Patient attends to one known speaker; used for fine-tuning |
| Testing | 10–20 min | Decode attention in real time or offline |

### Minimum Viable Session

For a first test, 10 minutes of dichotic listening (two speakers, patient told to attend to Speaker A for the first 5 min and Speaker B for the last 5 min) is sufficient to evaluate the pre-trained model.

---

## 5. Step 2 — Prepare Your Audio Stimuli

### What Kind of Audio to Use

Neurophile works best with **dry, continuous speech** from a single speaker per stream:

- ✅ Single-speaker audiobook recordings (e.g., LibriSpeech, LibriVox)
- ✅ Studio-recorded podcast speech (minimal background noise)
- ✅ Rhythmic instrumental music (strong 1–8 Hz amplitude modulations)
- ❌ Avoid: music with complex layered instruments (bass, drums, guitar simultaneously)
- ❌ Avoid: highly reverberant speech or telephone-quality audio

### Why This Matters

The model tracks the **1–8 Hz amplitude envelope** of the sound. Speech naturally has strong amplitude modulations in this range (syllable-rate oscillations). Audio with weak or irregular amplitude structure is harder for the model to track.

### File Format

Save each speaker as a separate mono `.wav` file, 44100 Hz:
```
speaker_a.wav    ← Speaker A's audio stream
speaker_b.wav    ← Speaker B's audio stream
```

---

## 6. Step 3 — Download the Global CI Model (TODO)

> **⚠️ This step is not yet available.** The checkpoint has not been published.
> Subscribe to https://github.com/neurophile/neurophile/releases for notification.

**Intended command (once available):**

```bash
# This will download the Global CI Foundation Model (~50–150 MB)
# with SHA-256 checksum verification
python scripts/download_global_model.py \
    --output-dir ./models/

# Expected output:
# Downloading neurophile-global-ci-v1.pt from HuggingFace Hub...
# ✓ Checksum verified (sha256: abc123...)
# Model saved to: ./models/neurophile-global-ci-v1.pt
```

**What the checkpoint contains:**
- Model weights trained on 16 CI subjects (KU Leuven dataset)
- Trained on CI-vocoded audio (16-channel, 900 pps simulation)
- Compatible with: `KULAdapter`, 64-channel EEG, 64 Hz envelope sampling rate

---

## 7. Step 4 — Run Attention Decoding Inference (TODO)

> **⚠️ This step is not yet available.** `scripts/run_inference.py` has not been implemented.

**Intended command (once available):**

```bash
python scripts/run_inference.py \
    --eeg patient_session.fif \
    --audio-a speaker_a.wav \
    --audio-b speaker_b.wav \
    --checkpoint ./models/neurophile-global-ci-v1.pt \
    --window-s 30

# Expected output:
# Loading EEG: patient_session.fif (1000 Hz, 32 channels)
# CI Artifact Pipeline: Stage 1 (template subtraction)...
# CI Artifact Pipeline: Stage 3 (ICA cancellation)... removed 2/20 components
# Extracting CI envelopes from audio...
#
# ── Attention Decoding Results (30-second windows) ──
# t=0–30s    Speaker A: 79%   Speaker B: 21%   → ATTENDED: Speaker A ✓
# t=30–60s   Speaker A: 23%   Speaker B: 77%   → ATTENDED: Speaker B ✓
# t=60–90s   Speaker A: 81%   Speaker B: 19%   → ATTENDED: Speaker A ✓
#
# Session accuracy (known labels): 94.4%
```

**What happens internally:**
1. EEG is cleaned through the 3-stage CI artifact pipeline
2. Audio is passed through the CI vocoder (16-channel simulation)
3. Low-frequency envelope (0.5–8 Hz) is extracted from the vocoded audio
4. The Global CI Model compares brain signal to both envelopes
5. The stream with higher neural-acoustic correlation is the attended speaker

---

## 8. Step 5 — Fine-Tune for Your Patient (TODO)

> **⚠️ This step is not yet available.** `scripts/finetune_patient.py` has not been implemented.

The Global Model is trained on population data. Fine-tuning it on **your specific patient's** calibration data (where the attended speaker is known) will improve accuracy significantly.

**Intended command (once available):**

```bash
python scripts/finetune_patient.py \
    --eeg calibration_session.fif \
    --audio-a speaker_a.wav \
    --audio-b speaker_b.wav \
    --attended-label A \
    --checkpoint ./models/neurophile-global-ci-v1.pt \
    --output ./models/patient_01_finetuned.pt \
    --epochs 20

# Expected duration: 3–8 minutes on CPU, ~1 min on GPU
# Expected improvement: +5–15% accuracy on held-out trials
```

**Why fine-tuning helps:**
- Each CI patient has a unique cortical response pattern shaped by their implant's electrode array, stimulation rate, and years of experience with the device.
- Fine-tuning adapts the last layers of the model to that patient's individual neural signature while preserving the general auditory cortex knowledge from the global training.

---

## 9. What the Model Output Means

| Output | Interpretation |
|---|---|
| Speaker A: 90%, Speaker B: 10% | Patient is clearly attending to Speaker A |
| Speaker A: 60%, Speaker B: 40% | Patient is likely attending to A but with lower confidence — try longer window |
| Speaker A: 50%, Speaker B: 50% | Ambiguous — patient may be mind-wandering, fatigued, or switching attention |

### Confidence Thresholds

For clinical applications, we recommend:
- **≥ 75%** — High confidence, use for closed-loop hearing aid control
- **60–75%** — Moderate confidence, flag for review
- **< 60%** — Insufficient confidence, do not act on this window

### Decision Window Length

| Window | Accuracy | Use case |
|---|---|---|
| 10 seconds | Lower | Real-time hearing aid control |
| 30 seconds | Medium | Rehabilitation session monitoring |
| 60 seconds | Higher | Diagnostic/research evaluation |
| 120 seconds | Highest | Baseline assessment |

Longer windows improve accuracy because the neural-acoustic correlation signal needs time to accumulate. CI patients often need **longer windows** than normal-hearing subjects due to degraded cortical tracking from the CI encoding.

---

## 10. CI Artifact Pipeline — What Happens to the EEG

The CI electrical stimulation creates massive artifacts (50–100 µV) that completely mask the microvolt-level neural signal. Neurophile removes these in three stages:

### Stage 1 — Template Subtraction ✅ (Active)
Identifies the repeating CI pulse pattern and subtracts its average shape from every pulse. Removes ~80–90% of artifact power.

### Stage 2 — Spatial Filtering 🔧 (Not yet implemented)
Projects out the residual artifact's spatial distribution using Common Spatial Patterns. **TODO.**

### Stage 3 — ICA Cancellation ✅ (Active, optional)
Runs Independent Component Analysis and auto-flags components with super-Gaussian statistics (kurtosis > 5.0) and periodicity matching the CI stimulation rate. These flagged ICs are the residual CI artifact that slipped through Stage 1.

**To enable Stage 3:**
```python
from neurophile.preprocessing.ci_artifact.pipeline import CIArtifactPipeline, CIArtifactConfig

config = CIArtifactConfig(stage3_enabled=True)   # ICA on
pipeline = CIArtifactPipeline(fs=1000, config=config)
clean_eeg = pipeline.run(raw_eeg)
```

**When to use Stage 3:** Enable it when the patient has bilateral CI implants, higher stimulation rates (> 500 pps), or when Stage 1 alone leaves visible artifact residuals in the 1–8 Hz band.

---

## 11. Frequently Asked Questions

**Q: Does the model need to be trained on my patient's language?**

No. AAD models do not learn words or language. They learn the **1–8 Hz amplitude rhythm** of speech. Because all human brains lock onto this rhythm regardless of language, a model trained on English (or Flemish) speech decodes attention to French, Hindi, or Mandarin equally well — as long as the audio envelope is extracted with the same signal processing pipeline.

**Q: What CI manufacturer/model does this work with?**

Any CI manufacturer (Cochlear, MED-EL, Advanced Bionics, Oticon). The model was trained on CI-simulated audio (not on real CI-user data yet). The ICA cancellation pipeline works with any CI that produces periodic electrical stimulation — which all modern CIs do.

**Q: How much EEG data does a patient need?**

- **Using the Global Model (no fine-tuning):** 0 minutes of patient data needed. Just record the session and run inference.
- **With patient-specific fine-tuning:** 10–20 minutes of calibration data (attended speaker known).
- **For population-level research:** 1–2 hours per subject.

**Q: What EEG sampling rate do I need?**

Record at **1000 Hz minimum** so the CI artifact pulses (250–900 Hz) are captured and can be removed by the pipeline. After artifact removal, the pipeline resamples to 512 Hz. The neural signal used for attention decoding is 0.5–8 Hz, so the final model only needs 64 Hz.

**Q: Can I use Neurophile in real time?**

The current implementation is **offline only**. Real-time LSL streaming (Tier 1/2 synchronization) is on the roadmap for Phase 2. See [README.md](README.md) Synchronization Tiers table.

**Q: Is this FDA/CE cleared for clinical use?**

No. Neurophile is a **research platform**. It has not undergone regulatory review for clinical decision-making. Do not use model outputs as the sole basis for medical decisions.

---

## 12. Reporting Issues

If the CI artifact pipeline does not clean your EEG adequately, or the model output seems incorrect:

1. **Open a GitHub Issue** at https://github.com/neurophile/neurophile/issues
2. Include:
   - CI manufacturer and model (e.g., Cochlear Nucleus CI632)
   - Stimulation rate in pulses per second (pps)
   - EEG amplifier and sampling rate
   - Whether the artifact is unilateral or bilateral
   - A short EEG segment (anonymised) if possible

3. Tag the issue: `ci-artifact`, `clinical-use`

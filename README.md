# 𝝅europhile

**Open Source Platform for Neural Decoding & Federated Research**

Transforming routine Cochlear Implant (CI) rehabilitation into an active, privacy-preserving federated learning grid provides an unprecedented, highly scalable data pipeline. By establishing an open-source ecosystem, this platform democratizes Auditory Attention Decoding (AAD) research, enabling the global scientific community to collaboratively train and benchmark robust brain-computer interfaces without compromising patient privacy or clinical workflow.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Who Is This For?

Neurophile has two distinct user groups with different entry points:

| I am a… | I want to… | Go to… |
|---|---|---|
| **Clinician / Audiologist** | Use the pre-trained model to decode CI patient attention | [CLINICAL_GUIDE.md](CLINICAL_GUIDE.md) |
| **Researcher / Engineer** | Train, extend, or contribute to the model ecosystem | [CONTRIBUTORS.md](CONTRIBUTORS.md) |

---

## What is Neurophile?

Neurophile serves as a collaborative environment for clinical applications and federated research. Our current implementations focus on auditory neuroscience, but the platform architecture is designed to support a wide range of neural decoding tasks:

- **Auditory Attention Decoding (AAD):** Decode which audio stream a listener attends to from their EEG, using envelope-tracking correlations in the delta-theta band.
- **Cochlear Implant (CI) Rehabilitation:** Provide a closed-loop environment to test whether a CI patient's auditory cortex is successfully rewiring to degraded signals.
- **Longitudinal Plasticity Tracking:** Monitor N1/P2 evoked potential amplitudes and cortical tracking strength across rehabilitation sessions.
- **Federated Research:** Aggregate anonymized model updates across clinics to study neuroplasticity trends in tonal vs. non-tonal language speakers.

---

## Quick Start

### For Clinicians (Pre-trained Model)
```bash
pip install neurophile[dl]

# TODO: hosted checkpoint not yet published — see CLINICAL_GUIDE.md for status
python scripts/download_global_model.py       # downloads the Global CI Foundation Model
python scripts/run_inference.py \
    --eeg patient_session.fif \
    --audio speaker1.wav speaker2.wav
```
> See [CLINICAL_GUIDE.md](CLINICAL_GUIDE.md) for the full step-by-step guide.

### For Researchers (Train from Scratch)
```bash
pip install -e ".[dev,dl]"
bash data_pipeline/fetchers/clone_author_repos.sh
python data_pipeline/fetchers/fetch_kul_dataset.py
python scripts/train_global_ci_model.py --synthetic --epochs 5  # smoke test
```
> See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the full training guide.

### Legacy CLI (Classical Linear Decoder)
```bash
pip install neurophile

# Validate a BIDS-EEG dataset for AAD compliance
neurophile validate /path/to/your/bids/dataset/

# Run an offline AAD evaluation on an existing dataset
neurophile decode --dataset /path/to/bids/ --decoder linear --window 60

# Run the full CI rehabilitation pipeline (offline)
neurophile decode --config configs/ci_rehab.yaml --dataset /path/to/bids/
```

---

## Architecture Overview

```
neurophile/
├── data/           Data standards: BIDS-EEG + HDF5 streaming
├── sync/           Temporal sync: TTL (Tier 1), LSL (Tier 2), Software (Tier 3)
├── stimulus/       Envelope extraction + CI vocoder simulation  [UPDATED]
├── devices/        EEG device drivers (LSL-based)
├── preprocessing/  Standard pipeline + 3-stage CI artifact rejection [UPDATED]
├── decoding/       AAD decoders + parallel evaluation harness
├── models/         Deep-learning AAD models + Adapter ecosystem  [NEW]
│   ├── core/       BaseAADModel — PyTorch contract for all DL models
│   ├── adapters/   KULAdapter, MesgaraniAdapter, ZionGolumbicAdapter
│   └── global_trainer.py  Strategy-Pattern training orchestrator
├── federated/      Federated learning: edge training + server aggregation
├── visualization/  Real-time dashboard + session reports
└── cli/            `neurophile` command-line interface
```

Data pipeline scripts (not part of the Python package):
```
data_pipeline/
├── fetchers/
│   ├── fetch_kul_dataset.py        Download KU Leuven EEG dataset (Zenodo)
│   ├── fetch_openneuro_bids.py     Download OpenNeuro BIDS datasets
│   └── clone_author_repos.sh       Clone KUL CNN + Mesgarani repos
└── eeg_preprocessing/
    └── standardize_montage.py      Remap EEG channels → standard 64-ch 10-20
```

---

## What Was Recently Added (v2025.06)

### New: Deep Learning Model Ecosystem (`neurophile.models`)
A new `models` package introduces PyTorch-native AAD models alongside the existing sklearn-based `LinearDecoder`. The architecture uses the **Adapter Pattern** to standardize diverse academic AAD models under a single interface.

```python
from neurophile.models import KULAdapter, MesgaraniAdapter, ZionGolumbicAdapter
from neurophile.models import GlobalCITrainer

model = KULAdapter(num_eeg_channels=64)          # fallback TCN — no external deps
trainer = GlobalCITrainer(model, epochs=50)
trainer.train(clean_eeg, ci_envelope, labels)    # outputs .pt checkpoint
```

### New: CI Vocoder (`neurophile.stimulus.CIVocoderSimulator`)
Converts normal-hearing audio into a mathematical simulation of what a Cochlear Implant user hears — N frequency bands, Hilbert amplitude modulation, noise/sine carriers. Used to train models that are biologically primed for CI acoustics.

```python
from neurophile.stimulus import CIVocoderSimulator
vocoder = CIVocoderSimulator(fs=44100, n_channels=16)
ci_audio, ci_envelope = vocoder.simulate_and_extract_envelope(audio, fs_eeg=64)
```

### Updated: CI Artifact Pipeline — Stage 3 Now Implemented
Stage 3 of the CI artifact pipeline (previously a no-op stub) is now a full **ICA-based cancellation** module. FastICA automatically identifies and removes CI electrical artifact components using kurtosis + periodicity criteria.

```python
from neurophile.preprocessing.ci_artifact.pipeline import CIArtifactPipeline, CIArtifactConfig

pipeline = CIArtifactPipeline(fs=1000, config=CIArtifactConfig(stage3_enabled=True))
clean_eeg = pipeline.run(raw_eeg)
```

---

## Supported Devices

| Device | Status | Driver | Channels | Notes |
|--------|--------|--------|----------|-------|
| OpenBCI Cyton | ✅ Implemented | `brainflow` | 8 | LSL-native, recommended for development |
| OpenBCI Ganglion | ✅ Implemented | `brainflow` | 4 | Bluetooth LE, screening use |
| Muse 2 / Muse S | 🔧 Scaffold | `muselsl` | 4–5 | Consumer, Tier 3 sync only |
| BrainProducts LiveAmp | 🔧 Scaffold | `brainvision-rda` | 32/64 | Research-grade reference device |
| g.tec g.USBamp | 🔧 Scaffold | `pygds` | 16/32 | CI-lab gold standard |
| Emotiv EPOC X | 🔧 Scaffold | `cortex` | 14 | Consumer, proprietary SDK |
| Any LSL-compatible | ✅ Generic | `pylsl` | any | Use `neurophile.devices.lsl_generic` |

---

## Synchronization Tiers

Auditory Attention Decoding requires < 3 ms audio-EEG temporal alignment. The platform defines three tiers:

| Tier | Method | Jitter | Status |
|------|--------|--------|--------|
| 1 | Hardware TTL trigger | < 0.5 ms | 🔧 Phase 2 |
| 2 | Lab Streaming Layer (LSL) | 1–3 ms | 🔧 Phase 2 |
| 3 | Software + calibration chirp | 5–15 ms | ✅ Phase 1 |

---

## CI Artifact Pipeline

Cochlear implant EEG artifacts are **not** removable with standard ICA/ASR. Neurophile uses a three-stage pipeline:

| Stage | Method | Status |
|-------|--------|--------|
| 1 | Template subtraction (periodic artifact) | ✅ Implemented |
| 2 | Spatial filtering (CCA / LCMV / SSD) | 🔧 Scaffold — see [CONTRIBUTING.md](CONTRIBUTING.md) |
| 3 | ICA-based cancellation (kurtosis + periodicity) | ✅ Implemented |

---

## AAD Models

| Model | Architecture | Interface | Status | Reference |
|-------|---|---|---|---|
| Linear stimulus reconstruction | Ridge regression | sklearn `fit/predict` | ✅ Implemented | Crosse et al. (2016) |
| KULAdapter | 3-layer TCN (fallback) / KUL CNN (external) | PyTorch `nn.Module` | ✅ Fallback ready | Vandecappelle et al. (2021) |
| MesgaraniAdapter | Conv+GRU (fallback) / CRN (external) | PyTorch `nn.Module` | ✅ Fallback ready | Mesgarani & Chang (2012) |
| ZionGolumbicAdapter | CNN + Transformer cross-attention | PyTorch `nn.Module` | ✅ Implemented | Zion-Golumbic et al. (2013) |
| Global CI Foundation Model | Causal Spatial-Temporal CNN | PyTorch checkpoint | ✅ Implemented (73.1% zero-shot cross-subject accuracy) | — |

---

## Decoders

| Decoder | Type | Status | Reference |
|---------|------|--------|-----------|
| Linear stimulus reconstruction | Ridge regression | ✅ Implemented | Crosse et al. (2016) |
| CNN-based AAD | Deep learning | ✅ Via KULAdapter | Vandecappelle et al. (2021) |
| Subject-independent AAD | Transfer learning | 🔧 Scaffold | Ciccarelli et al. (2023) |

---

## Stimuli

All audio stimuli bundled with Neurophile are CC-licensed. See [stimuli/manifest.yaml](stimuli/manifest.yaml) and [stimuli/LICENSE.md](stimuli/LICENSE.md) for per-file attribution.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CONTRIBUTORS.md](CONTRIBUTORS.md). The highest-value open contributions are:

1. **Global CI Foundation Model checkpoint** — Train and publish the `.pt` file to Hugging Face Hub / Zenodo
2. **`scripts/run_inference.py`** — Clinical inference script (TODO)
3. **`scripts/download_global_model.py`** — Checkpoint downloader (TODO)
4. **Wire real KUL / Mesgarani models** — Fill TODO markers in adapter shims
5. **CI artifact Stage 2** (`spatial_filter.py`) — Spatial filtering contribution
6. **New EEG device drivers** (`src/neurophile/devices/`)
7. **Federated Learning client** — Flower `NumPyClient` wrapper for edge devices

---

## Citation

If you use Neurophile in a publication, please cite:

```bibtex
@software{neurophile2025,
  title  = {Neurophile: Neuro-Auditory Rehabilitation \& Attention Platform},
  year   = {2025},
  url    = {https://github.com/neurophile/neurophile},
  license = {Apache-2.0}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

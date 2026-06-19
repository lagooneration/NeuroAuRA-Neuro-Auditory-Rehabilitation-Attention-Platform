# CONTRIBUTORS.md — Researcher & Developer Guide

> **Audience:** Researchers, engineers, and open-source contributors who want to train the Global CI Foundation Model from scratch, wire real academic AAD models, or extend the Neurophile platform.
>
> **Clinicians looking to use the pre-trained model:** see [CLINICAL_GUIDE.md](CLINICAL_GUIDE.md) instead.

---

## Table of Contents

1. [Understanding the Model Architecture](#1-understanding-the-model-architecture)
2. [Development Environment Setup](#2-development-environment-setup)
3. [Step 1 — Clone External Academic Repositories](#3-step-1--clone-external-academic-repositories)
4. [Step 2 — Download Baseline Datasets](#4-step-2--download-baseline-datasets)
5. [Step 3 — Standardize EEG Montages](#5-step-3--standardize-eeg-montages)
6. [Step 4 — Train the Global CI Foundation Model](#6-step-4--train-the-global-ci-foundation-model)
7. [Step 5 — Wire the Real Academic Models](#7-step-5--wire-the-real-academic-models)
8. [Step 6 — Publish the Checkpoint (TODO)](#8-step-6--publish-the-checkpoint-todo)
9. [Adding a New AAD Adapter](#9-adding-a-new-aad-adapter)
10. [Running the Test Suite](#10-running-the-test-suite)
11. [Code Standards](#11-code-standards)
12. [High-Priority Open Contributions](#12-high-priority-open-contributions)

---

## 1. Understanding the Model Architecture

Neurophile uses **two parallel model hierarchies**:

```
Classical (CPU, fast)                    Deep Learning (GPU-capable, Flower-FL-ready)
─────────────────────                    ─────────────────────────────────────────────
neurophile.decoding.BaseDecoder           neurophile.models.core.BaseAADModel
       │                                          │
  LinearDecoder                         ┌─────────┴──────────┐
  (Ridge regression)                KULAdapter        MesgaraniAdapter
                                  (3-layer TCN)      (Conv+GRU CRN)
                                        │                    │
                                   GlobalCITrainer ──────────┘
                                   (Strategy Pattern)
                                        │
                              ┌─────────┴──────────┐
                         PyTorchStrategy       SklearnStrategy
                         (Adam, BCE loss,      (direct fit(),
                          gradient clips)       CPU only)
```

### Why Two Hierarchies?

- `BaseDecoder` (sklearn) is for **classical, interpretable** models — fast to run, no GPU needed.
- `BaseAADModel` (PyTorch `nn.Module`) is for **deep learning** models that need gradient-based training, batch loading, and compatibility with [Flower](https://flower.ai) federated learning. The `state_dict()` produced by `GlobalCITrainer` is drop-in compatible with Flower's `get_parameters()` / `set_parameters()` API.

### Adapter Pattern (Anti-Corruption Layer)

Each external research model is wrapped in an **Adapter** that:
1. Isolates Neurophile from upstream API changes in academic repos.
2. Ships with a **built-in fallback network** so training runs immediately (no external repos needed).
3. Has clear `# TODO (implementer)` markers where the real external import goes.

---

## 2. Development Environment Setup

```bash
# Clone the Neurophile repo
git clone https://github.com/neurophile/neurophile.git
cd neurophile

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows PowerShell

# Install in editable mode with dev + deep learning extras
pip install -e ".[dev,dl]"

# Install pre-commit hooks (runs ruff + mypy on every commit)
pre-commit install

# Verify installation
python -c "from neurophile.models import KULAdapter; print('OK')"
```

**Dependencies installed by `[dl]`:**
- `torch>=2.2` (CPU build by default; install CUDA build separately for GPU)
- `tqdm>=4.66`

**For GPU training:**
```bash
# CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Step 1 — Clone External Academic Repositories

The `KULAdapter` and `MesgaraniAdapter` can run their built-in fallback networks immediately. However, to use the **real, validated** academic models, clone their repositories first.

```bash
# Run from the repo root — clones into external_libs/
bash data_pipeline/fetchers/clone_author_repos.sh

# To update already-cloned repos:
bash data_pipeline/fetchers/clone_author_repos.sh --update
```

**What gets cloned:**

| Folder | Source | Purpose |
|---|---|---|
| `external_libs/locus-of-auditory-attention-cnn/` | [exporl/locus-of-auditory-attention-cnn](https://github.com/exporl/locus-of-auditory-attention-cnn) | KU Leuven CNN for AAD |
| `external_libs/naplib-python/` | [naplab/naplib-python](https://github.com/naplab/naplib-python) | Mesgarani Lab neural-acoustic tools |
| `external_libs/auditory-eeg-challenge/` | [exporl/auditory-eeg-challenge-2023-code](https://github.com/exporl/auditory-eeg-challenge-2023-code) | EXPORL supplementary challenge code |

**After cloning**, wire the real models by following the `# TODO (implementer)` comments in:
- [`src/neurophile/models/adapters/kul_cnn_adapter.py`](src/neurophile/models/adapters/kul_cnn_adapter.py)
- [`src/neurophile/models/adapters/mesgarani_crn_adapter.py`](src/neurophile/models/adapters/mesgarani_crn_adapter.py)

Then activate:
```python
from neurophile.models import KULAdapter
model = KULAdapter(num_eeg_channels=64, use_external=True)   # uses real KUL CNN
```

---

## 4. Step 2 — Download Baseline Datasets

### KU Leuven EEG Dataset (Recommended starting point)

```bash
# Dry run — verify URLs and file sizes without downloading
python data_pipeline/fetchers/fetch_kul_dataset.py --dry-run

# Download all subjects (16 × ~42 MB ≈ 670 MB)
python data_pipeline/fetchers/fetch_kul_dataset.py \
    --output-dir ./data/raw/kul

# Download specific subjects only
python data_pipeline/fetchers/fetch_kul_dataset.py \
    --output-dir ./data/raw/kul \
    --subjects 1 2 3
```

> **TODO:** Update the SHA-256 hashes in `fetch_kul_dataset.py` from the Zenodo metadata API:
> `curl https://zenodo.org/api/records/1199011 | python -m json.tool`
> Replace `PLACEHOLDER_SHA256_REPLACE_WITH_REAL_HASH` for each subject file.

**Dataset details:**
- **DOI:** 10.5281/zenodo.1199011
- **Subjects:** 16
- **Task:** Dichotic listening — two competing Flemish speech streams
- **EEG:** 64 channels, 50 Hz (downsampled), BrainProducts actiCAP
- **Reference:** Biesmans et al. (2017) / Vandecappelle et al. (2021)

### OpenNeuro BIDS Datasets

```bash
# List all available Neurophile-compatible datasets
python data_pipeline/fetchers/fetch_openneuro_bids.py --list-datasets

# Download Zion-Golumbic dataset (requires AWS CLI — free, no login)
python data_pipeline/fetchers/fetch_openneuro_bids.py \
    --dataset ds003516 \
    --output-dir ./data/raw/openneuro

# Download one subject only (for testing the pipeline)
python data_pipeline/fetchers/fetch_openneuro_bids.py \
    --dataset ds003516 \
    --subject sub-01 \
    --output-dir ./data/raw/openneuro

# Fallback (no AWS CLI): use openneuro-py
pip install openneuro-py
python data_pipeline/fetchers/fetch_openneuro_bids.py \
    --dataset ds003516 --no-aws
```

---

## 5. Step 3 — Standardize EEG Montages

Different datasets use different channel layouts. Before training, remap all EEG to the Neurophile canonical **64-channel 10-20** montage:

```bash
# Remap a single file
python data_pipeline/eeg_preprocessing/standardize_montage.py \
    --input ./data/raw/kul/subject1.mat \
    --output ./data/processed/subject1_standardized.fif

# In Python — batch processing
from data_pipeline.eeg_preprocessing.standardize_montage import MontageStandardizer
import sys; sys.path.insert(0, "data_pipeline")

std = MontageStandardizer()

# From MNE Raw object
raw_std = std.standardize(raw_mne)

# From NumPy array (when you only have the array, not an MNE object)
eeg_std, channel_names = std.standardize_numpy(
    eeg_array,        # (n_samples, n_input_channels)
    ch_names,         # list of channel name strings
    fs=50,
)
```

---

## 6. Step 4 — Train the Global CI Foundation Model

The master training script orchestrates the full **6-step CI pipeline:**

| Step | Operation | Script component |
|---|---|---|
| 1 | Ingest baseline audio (.wav / .flac) | `step1_ingest_audio()` |
| 2 | Vocode to CI simulation (N-channel vocoder) | `step2_vocode()` via `CIVocoderSimulator` |
| 3 | Extract low-frequency CI envelope (0.5–8 Hz) | `step3_extract_envelope()` |
| 4 | Clean EEG (template subtraction + ICA Stage 3) | `step4_clean_eeg()` via `CIArtifactPipeline` |
| 5 | Feed (clean EEG, CI envelope) into adapter model | `step5_6_train()` via `GlobalCITrainer` |
| 6 | Compute loss (Pearson correlation / BCE) + backpropagate | `_PyTorchStrategy.fit()` |

### Smoke Test (no downloads needed — synthetic data)

```bash
# KULAdapter with fallback TCN, 5 epochs, synthetic data
python scripts/train_global_ci_model.py \
    --synthetic \
    --epochs 5 \
    --model kul \
    --output-dir ./checkpoints

# MesgaraniAdapter smoke test
python scripts/train_global_ci_model.py \
    --synthetic \
    --epochs 5 \
    --model mesgarani
```

**Output:**
```
checkpoints/
└── kul_cnn_global_ci.pt     ← PyTorch checkpoint
                                Contains: model_state, optimizer_state, metadata
```

### Full Training (real KUL data)

```bash
python scripts/train_global_ci_model.py \
    --eeg-dir ./data/processed/ \
    --audio-dir ./data/raw/audio/ \
    --model kul \
    --epochs 50 \
    --ci-channels 16 \
    --ci-rate 900 \
    --enable-ica \
    --device cpu \
    --output-dir ./checkpoints/kul_ci_global_v1
```

**CLI flags reference:**

| Flag | Default | Description |
|---|---|---|
| `--synthetic` | off | Use synthetic data (smoke test — no downloads needed) |
| `--eeg-dir` | None | Directory with preprocessed EEG `.npy` files |
| `--audio-dir` | None | Directory with `.wav` / `.flac` audio files |
| `--model` | `kul` | Adapter: `kul` or `mesgarani` |
| `--epochs` | 50 | Number of training epochs |
| `--n-trials` | 32 | Synthetic trial count |
| `--ci-channels` | 16 | CI vocoder channels (8–22) |
| `--ci-rate` | 900.0 | CI stimulation rate (pulses/second) |
| `--enable-ica` | off | Enable ICA-based CI artifact Stage 3 |
| `--device` | `cpu` | PyTorch device: `cpu`, `cuda`, `mps` |
| `--output-dir` | `./checkpoints` | Checkpoint save directory |

---

## 7. Step 5 — Wire the Real Academic Models

Once repos are cloned and you've located the model class inside each repo:

### KULAdapter — wire the real KUL CNN

Open [`src/neurophile/models/adapters/kul_cnn_adapter.py`](src/neurophile/models/adapters/kul_cnn_adapter.py) and find the two `# TODO (implementer)` blocks:

```python
# TODO (implementer): After clone_author_repos.sh, update this import:
# from external_libs.locus_of_auditory_attention_cnn.model import CNNModel as KULeuvenCNN

# TODO (implementer): Adjust constructor arguments:
# self.backend_model = KULeuvenCNN(n_channels=num_eeg_channels)

# TODO (implementer): Translate Neurophile tensors to KUL's expected format:
# eeg_translated = eeg_tensor.permute(0, 2, 1)  # (B,T,C) → (B,C,T) if needed
```

Then test:
```bash
python -c "
from neurophile.models import KULAdapter
import torch
m = KULAdapter(num_eeg_channels=64, use_external=True)
out = m(torch.randn(2, 512, 64), torch.randn(2, 512, 1))
print('KUL real model output shape:', out.shape)  # should be (2, 1)
"
```

### MesgaraniAdapter — wire the real CRN

Same process — open [`src/neurophile/models/adapters/mesgarani_crn_adapter.py`](src/neurophile/models/adapters/mesgarani_crn_adapter.py) and fill the `# TODO` blocks.

---

## 8. Step 6 — Publish the Checkpoint (TODO)

> **This step has not been completed yet.**

Once training converges, the checkpoint must be hosted publicly so clinicians can use `download_global_model.py`. Recommended platforms:

| Platform | Why | How |
|---|---|---|
| **Hugging Face Hub** | Standard for ML models; versioned; free; widely trusted | `huggingface-cli upload neurophile/global-ci-model kul_cnn_global_ci.pt` |
| **Zenodo** | Permanent DOI; preferred for academic citation | Upload via web UI, get DOI for citation |
| **GitHub Releases** | Simple; works for files < 2 GB | Attach `.pt` to a tagged release |

**TODO checklist for checkpoint publication:**
- [ ] Train the model on full KUL dataset with real KUL CNN (`use_external=True`)
- [ ] Evaluate: accuracy ≥ 70% on held-out subjects (subject-independent)
- [ ] Save checkpoint with full metadata (training config, dataset version, model version)
- [ ] Upload to Hugging Face Hub: `neurophile/global-ci-model`
- [ ] Compute SHA-256 of the `.pt` file
- [ ] Implement `scripts/download_global_model.py` using `pooch` with SHA-256 verification
- [ ] Implement `scripts/run_inference.py` for clinician use
- [ ] Update `CLINICAL_GUIDE.md` with the real download URL and checksum

---

## 9. Adding a New AAD Adapter

To add a new external model (e.g., the Zion-Golumbic cross-attention model):

### 1. Create the adapter file

```bash
# Create: src/neurophile/models/adapters/zion_golumbic_adapter.py
```

```python
from neurophile.models.core.base_aad_model import BaseAADModel, _require_torch

# ACL shim — guard the external import
_ZG_EXTERNAL_AVAILABLE = False
try:
    from external_libs.zion_golumbic.model import CrossAttentionAAD
    _ZG_EXTERNAL_AVAILABLE = True
except ImportError:
    pass

def _build_fallback_cross_attention(num_eeg_channels: int):
    """Fallback: multi-head attention between EEG and envelope."""
    _require_torch()
    import torch.nn as nn
    # ... implement fallback ...

def _make_zg_adapter_class():
    _require_torch()
    import torch.nn as nn

    class ZionGolumbicAdapter(nn.Module, BaseAADModel):
        name = "zion_golumbic_cross_attention"

        def __init__(self, num_eeg_channels=64, audio_sampling_rate=64, use_external=False):
            super().__init__()
            self.num_eeg_channels = num_eeg_channels
            self.audio_sampling_rate = audio_sampling_rate
            self.backend_model = _build_fallback_cross_attention(num_eeg_channels)

        def forward(self, eeg_tensor, audio_envelope_tensor):
            return self.backend_model(eeg_tensor, audio_envelope_tensor)

    return ZionGolumbicAdapter

ZionGolumbicAdapter = _make_zg_adapter_class()
```

### 2. Register in the adapters package

```python
# In src/neurophile/models/adapters/__init__.py — add:
from neurophile.models.adapters.zion_golumbic_adapter import ZionGolumbicAdapter
__all__ = ["KULAdapter", "MesgaraniAdapter", "ZionGolumbicAdapter"]
```

### 3. Add tests

```bash
# Create: tests/unit/test_zion_golumbic_adapter.py
# Follow the pattern in tests/unit/test_base_aad_model.py
```

### 4. Add to README decoder table

```markdown
| ZionGolumbicAdapter | Cross-attention Transformer | PyTorch `nn.Module` | 🔧 TODO | Zion-Golumbic et al. (2013) |
```

---

## 10. Running the Test Suite

```bash
# All tests (fast — no data download, no network)
python -m pytest tests/ -q --tb=short

# Specific new modules
python -m pytest tests/unit/test_ci_vocoder.py -v
python -m pytest tests/unit/test_ica_cancellation.py -v
python -m pytest tests/unit/test_base_aad_model.py -v       # requires: pip install torch

# Skip tests requiring hardware or network
python -m pytest tests/ -m "not slow and not network and not hardware" -q

# Coverage report
python -m pytest tests/ --cov=src/neurophile --cov-report=term-missing

# Run the training smoke test end-to-end
python scripts/train_global_ci_model.py --synthetic --epochs 2 --model kul
```

**Current test status:**

| Module | Tests | Status |
|---|---|---|
| `test_ci_vocoder.py` | 15 | ✅ All pass |
| `test_ica_cancellation.py` | 12 | ✅ All pass |
| `test_base_aad_model.py` | 14 | ✅ All pass (requires torch) |
| `test_ci_artifact.py` | pre-existing | ✅ Unchanged |
| `test_metadata.py` | pre-existing | ✅ Unchanged |
| `test_decoder.py` | pre-existing | ⚠️ 1 pre-existing off-by-one in `LinearDecoder.score()` |

---

## 11. Code Standards

| Tool | Command | When |
|---|---|---|
| Lint + format | `ruff check . && ruff format .` | Every commit (pre-commit) |
| Type check | `mypy src/neurophile/` | PRs |
| Tests | `pytest tests/unit/` | Every commit |
| Coverage | `pytest --cov` | PRs (must not drop below threshold) |

**Module docstring conventions:**
- Every new module must have a module-level docstring explaining purpose, algorithm, and references.
- All public classes and methods must have NumPy-style docstrings with `Parameters`, `Returns`, and `Examples`.

---

## 12. High-Priority Open Contributions

| Contribution | File | Effort | Impact |
|---|---|---|---|
| **Train & publish Global CI checkpoint** | `scripts/train_global_ci_model.py` + HuggingFace | 1–2 days | 🔴 Critical — enables clinical use |
| **`scripts/run_inference.py`** | New file | 3–4 hours | 🔴 Critical — clinician entry point |
| **`scripts/download_global_model.py`** | New file | 1 hour | 🔴 Critical — checkpoint distribution |
| **Wire real KUL CNN** | `kul_cnn_adapter.py` TODO blocks | 1–3 hours | 🟠 High |
| **Wire real Mesgarani CRN** | `mesgarani_crn_adapter.py` TODO blocks | 2–4 hours | 🟠 High |
| **ZionGolumbicAdapter** | New adapter file | 3–5 days | 🟠 High |
| **CI artifact Stage 2** (spatial filter) | `spatial_filter.py` | 3–5 days | 🟡 Medium |
| **Federated Learning client** | Flower `NumPyClient` wrapper | 2–3 days | 🟡 Medium |
| **Fix LinearDecoder.score() off-by-one** | `decoding/base.py:83` | 20 min | 🟡 Medium |
| **EEG device drivers** (Muse, BrainProducts) | `devices/*.py` | 1–2 days each | 🟡 Medium |
| **Real-time LSL pipeline** | `sync/ttl.py`, `sync/lsl.py` | 3–5 days | 🟡 Medium |

---

## Questions?

Open a [GitHub Issue](https://github.com/neurophile/neurophile/issues) or start a [Discussion](https://github.com/neurophile/neurophile/discussions).

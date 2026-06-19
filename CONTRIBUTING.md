# Contributing to Neurophile

Thank you for contributing to open-source auditory neuroscience tooling!

> **This file covers general contribution standards.**
>
> For a full step-by-step guide to training the model from scratch, wiring
> external academic models, and submitting new adapters, see:
> → **[CONTRIBUTORS.md](CONTRIBUTORS.md)**
>
> For clinicians using the pre-trained Global CI Foundation Model, see:
> → **[CLINICAL_GUIDE.md](CLINICAL_GUIDE.md)**

## Where to start

Check the [GitHub Issues](https://github.com/neurophile/neurophile/issues) for items tagged:
- `good-first-issue` — well-scoped tasks, good for new contributors
- `scaffold` — placeholder modules that need implementing (see map below)
- `ci-artifact` — cochlear implant artifact methods
- `device-driver` — new EEG device support

## High-Priority Scaffold Modules

These modules have detailed specs in their docstrings. Pick one, read the spec, implement it, add tests.

### CI Artifact Pipeline (Stage 2 & 3)

```
src/neurophile/preprocessing/ci_artifact/
  spatial_filter.py     ← CCA / LCMV / SSD spatial filter (Stage 2)
  adaptive_filter.py    ← LMS / RLS adaptive filter (Stage 3)
```

**Background reading:**
- Somers et al. (2019) "A generic EEG artifact removal algorithm based on the multi-channel Wiener filter" — for Stage 2
- Viola et al. (2011) "Semi-automatic identification of independent components" — for Stage 3
- Gilley et al. (2017) — template subtraction baseline

**Expected API:** See `spatial_filter.py` docstring for the exact method signatures.

### EEG Device Drivers

```
src/neurophile/devices/
  muse.py               ← Muse 2 / Muse S (muselsl)
  brainproducts.py      ← BrainProducts LiveAmp (brainvision-rda)
  gtec.py               ← g.tec g.USBamp (pygds)
  emotiv.py             ← Emotiv EPOC X (cortex SDK)
```

All device drivers must implement the `BaseDevice` protocol defined in `devices/base.py`.

### AAD Decoders

```
src/neurophile/decoding/
  cnn_decoder.py        ← CNN-based AAD (Vandecappelle et al., 2021)
  plasticity.py         ← Longitudinal N1/P2 + cortical tracking model
```

### Real-time Sync (Phase 2)

```
src/neurophile/sync/
  ttl.py                ← Hardware TTL trigger (Tier 1)
  lsl.py                ← LSL network sync (Tier 2)
  drift.py              ← Clock drift correction
```

---

## Development Setup

```bash
# Clone the repo
git clone https://github.com/neurophile/neurophile.git
cd neurophile

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in editable mode with dev extras
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/unit/           # fast, no network
pytest tests/unit/ -m "not slow"  # exclude dataset downloads
```

---

## Adding a New Decoder

1. Create `src/neurophile/decoding/my_decoder.py`.
2. Subclass `BaseDecoder` from `neurophile.decoding.base`.
3. Implement `fit(eeg, envelope, fs)` and `predict(eeg)` methods.
4. Register in `neurophile.decoding.__init__`.
5. Add unit tests in `tests/unit/test_my_decoder.py` using synthetic data from `conftest.py`.
6. Add to `README.md` decoder table with status `🔧 Community`.

**Minimal decoder skeleton:**

```python
from neurophile.decoding.base import BaseDecoder
import numpy as np

class MyDecoder(BaseDecoder):
    """One-line description.

    Parameters
    ----------
    param : type
        Description.

    References
    ----------
    Author et al. (year) "Title" DOI
    """

    name = "my_decoder"   # used in CLI --decoder flag

    def fit(self, eeg: np.ndarray, envelope: np.ndarray, fs: int) -> "MyDecoder":
        # eeg: (n_samples, n_channels)
        # envelope: (n_samples,)
        ...
        return self

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        # returns reconstructed envelope: (n_samples,)
        ...
```

---

## Adding a New EEG Device Driver

```python
from neurophile.devices.base import BaseDevice, DeviceInfo
import numpy as np

class MyDevice(BaseDevice):
    """One-line description.

    Sync tier supported: 2 (LSL) or 3 (software).
    """

    info = DeviceInfo(
        name="My EEG Device",
        manufacturer="Acme EEG",
        n_channels=8,
        sampling_rate=250,
        sync_tier=2,
        driver_package="my_eeg_sdk",
    )

    def connect(self) -> None: ...
    def stream(self) -> None: ...   # starts LSL outlet
    def disconnect(self) -> None: ...
```

---

## Code Standards

| Tool | Command | Runs on |
|------|---------|---------|
| Lint + format | `ruff check . && ruff format .` | every commit (pre-commit) |
| Type check | `mypy src/neurophile/` | PRs |
| Tests | `pytest tests/unit/` | every commit |
| Coverage | `pytest --cov` | PRs (must not drop below threshold) |

---

## Stimulus Contributions

All audio stimuli must be CC-BY or CC0. Do not add CC-BY-NC or copyrighted content.

When adding a stimulus:
1. Place the file in `stimuli/audio/<category>/`.
2. Add an entry to `stimuli/manifest.yaml` with the SHA-256 checksum, license, and source URL.
3. Compute the envelope cache: `neurophile stimulus cache stimuli/manifest.yaml`.

---

## Questions?

Open an issue or start a [Discussion](https://github.com/neurophile/neurophile/discussions).

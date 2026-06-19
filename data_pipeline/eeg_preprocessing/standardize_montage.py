"""
standardize_montage.py
=======================
Remap arbitrary EEG channel layouts to the Neurophile standard 64-channel
10-20 montage using MNE-Python.

This is the EEG preprocessing analogue of the adapter pattern: just as the
model adapters translate external model architectures to the Neurophile
tensor contract, this script translates arbitrary montages to the standard
channel layout expected by all preprocessing and decoding modules.

Montage Standard
----------------
Neurophile uses the standard 64-channel 10-20 layout (BrainProducts actiCAP,
which matches the KUL dataset and most clinical CI recording setups):

    Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8,
    TP9, CP5, CP1, CP2, CP6, TP10, P7, P3, Pz, P4, P8, PO9, O1, Oz, O2, PO10,
    ... (64 total)

Usage
-----
    python data_pipeline/eeg_preprocessing/standardize_montage.py \\
        --input ./data/raw/kul/subject1.mat \\
        --output ./data/processed/subject1_standardized.fif

    from standardize_montage import MontageStandardizer
    standardizer = MontageStandardizer()
    raw_mne = standardizer.standardize(raw_mne_object)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Neurophile canonical 64-ch channel names (10-20 standard) ─────────────────
NEUROAURA_64CH = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7",
    "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
    "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
    "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz",
    "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4",
    "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz",
    "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]
assert len(NEUROAURA_64CH) == 64, "Canonical channel list must have exactly 64 entries"


class MontageStandardizer:
    """Remap an MNE Raw object's channels to the Neurophile 64-ch standard.

    Parameters
    ----------
    target_channels : list[str] or None
        Target channel set. None = use the Neurophile 64-ch canonical layout.
    montage_name : str
        MNE standard montage to assign. Default: ``"standard_1020"``.
    handle_missing : str
        How to handle channels in ``target_channels`` not present in the input:
        ``"zero"`` = add zero-filled channels, ``"drop"`` = reduce target set.

    Examples
    --------
    >>> import mne
    >>> raw = mne.io.read_raw_fif("recording.fif", preload=True)
    >>> std = MontageStandardizer()
    >>> raw_std = std.standardize(raw)
    >>> raw_std.ch_names[:5]
    ['Fp1', 'AF7', 'AF3', 'F1', 'F3']
    """

    def __init__(
        self,
        target_channels: list[str] | None = None,
        montage_name: str = "standard_1020",
        handle_missing: str = "zero",
    ) -> None:
        self.target_channels = target_channels or NEUROAURA_64CH
        self.montage_name = montage_name
        self.handle_missing = handle_missing

    def standardize(self, raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
        """Remap channels and assign standard 10-20 montage.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Loaded MNE raw recording (any channel layout).

        Returns
        -------
        raw_std : mne.io.BaseRaw
            Copy with channels remapped to ``self.target_channels``.
        """
        try:
            import mne
        except ImportError as exc:
            raise ImportError(
                "mne is required: pip install mne"
            ) from exc

        raw = raw.copy()
        input_channels = [ch.upper() for ch in raw.ch_names]
        target_upper = [ch.upper() for ch in self.target_channels]

        # Find intersection
        present = [ch for ch in target_upper if ch in input_channels]
        missing = [ch for ch in target_upper if ch not in input_channels]

        if missing:
            logger.warning(
                "MontageStandardizer: %d target channels missing from input: %s",
                len(missing), missing[:10],
            )

        if self.handle_missing == "zero" and missing:
            raw = self._add_zero_channels(raw, missing)
        elif self.handle_missing == "drop":
            self.target_channels = [
                ch for ch in self.target_channels
                if ch.upper() in input_channels
            ]

        # Reorder to target layout
        # Map back to original case (MNE is case-sensitive)
        ch_map = {ch.upper(): ch for ch in raw.ch_names}
        picks = [ch_map[ch.upper()] for ch in self.target_channels if ch.upper() in ch_map]
        raw = raw.pick(picks)

        # Reorder channels to match canonical order
        raw = raw.reorder_channels(picks)

        # Assign standard 10-20 montage
        montage = mne.channels.make_standard_montage(self.montage_name)
        raw.set_montage(montage, on_missing="warn", verbose=False)

        logger.info(
            "MontageStandardizer: %d/%d target channels present, %d missing",
            len(present), len(self.target_channels), len(missing),
        )
        return raw

    def standardize_numpy(
        self,
        eeg: np.ndarray,
        ch_names: list[str],
        fs: int,
    ) -> tuple[np.ndarray, list[str]]:
        """Remap a raw numpy EEG array to the standard channel order.

        Parameters
        ----------
        eeg : np.ndarray, shape (n_samples, n_input_channels)
        ch_names : list[str]
            Channel names corresponding to eeg columns.
        fs : int
            Sampling rate in Hz.

        Returns
        -------
        eeg_std : np.ndarray, shape (n_samples, n_target_channels)
        out_channels : list[str]
        """
        n_samples = eeg.shape[0]
        input_upper = {ch.upper(): i for i, ch in enumerate(ch_names)}
        target = self.target_channels
        eeg_std = np.zeros((n_samples, len(target)), dtype=eeg.dtype)

        for j, ch in enumerate(target):
            idx = input_upper.get(ch.upper())
            if idx is not None:
                eeg_std[:, j] = eeg[:, idx]
            # else: stays zero (zero-fill for missing channels)

        return eeg_std, target

    @staticmethod
    def _add_zero_channels(raw: "mne.io.BaseRaw", ch_names: list[str]) -> "mne.io.BaseRaw":
        """Append zero-filled channels to a Raw object."""
        import mne
        n_times = raw.n_times
        data = np.zeros((len(ch_names), n_times))
        info = mne.create_info(ch_names=ch_names, sfreq=raw.info["sfreq"], ch_types="eeg")
        zero_raw = mne.io.RawArray(data, info, verbose=False)
        raw.add_channels([zero_raw], force_update_info=True)
        return raw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize EEG montage to Neurophile 64-ch 10-20 layout."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input EEG file (MNE-readable)")
    parser.add_argument("--output", type=Path, required=True, help="Output .fif file path")
    parser.add_argument("--n-channels", type=int, default=64, help="Target channel count")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()

    try:
        import mne
    except ImportError:
        print("Error: mne is required. pip install mne")
        raise SystemExit(1)

    logger.info("Loading %s …", args.input)
    raw = mne.io.read_raw(str(args.input), preload=True, verbose=False)
    standardizer = MontageStandardizer()
    raw_std = standardizer.standardize(raw)
    raw_std.save(str(args.output), overwrite=True)
    logger.info("Saved standardized recording → %s", args.output)

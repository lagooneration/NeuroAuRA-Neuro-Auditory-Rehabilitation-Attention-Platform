"""
neurophile.stimulus.envelope
============================
Compute the auditory envelope of a stimulus audio file.

The envelope is the key signal in AAD: the EEG's cortical tracking response
correlates with the *envelope* of the attended stream (not the waveform itself).

Method
------
1. Gammatone filterbank (4–32 bands, ERB spacing, 100 Hz–8 kHz)
   — simulates the frequency decomposition of the cochlea.
2. Hilbert transform on each band → instantaneous amplitude.
3. Sum bands → broadband envelope.
4. Downsample to EEG sampling rate.
5. Optional: power-law compression (exponent 0.3) to better match
   the neural encoding of loudness.

References
----------
- Crosse et al. (2016) "The Multivariate Temporal Response Function (mTRF)
  Toolbox" doi:10.3389/fnhum.2016.00604
- Biesmans et al. (2017) "Auditory-Inspired Speech Envelope Extraction Methods
  for Improved EEG-Based Auditory Attention Detection"
  doi:10.1109/TNSRE.2016.2571900
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, resample_poly
from math import gcd

logger = logging.getLogger(__name__)

# ── Default gammatone bank parameters ─────────────────────────────────────────
_DEFAULT_N_BANDS = 16
_DEFAULT_F_LOW = 100.0     # Hz
_DEFAULT_F_HIGH = 8000.0   # Hz
_COMPRESSION_EXPONENT = 0.3


class EnvelopeExtractor:
    """Extract the broadband auditory envelope from an audio signal.

    Parameters
    ----------
    fs_audio : int
        Sampling rate of the audio file (e.g. 44100 Hz).
    fs_eeg : int
        Target EEG sampling rate to downsample the envelope to.
    n_bands : int
        Number of gammatone filterbank bands.
    f_low, f_high : float
        Frequency range of the filterbank (Hz).
    compress : bool
        If True, apply power-law compression (exponent 0.3) after envelope.
        Recommended for matching neural encoding of loudness.

    Examples
    --------
    >>> import soundfile as sf
    >>> audio, fs = sf.read("stimulus.wav")
    >>> extractor = EnvelopeExtractor(fs_audio=fs, fs_eeg=512)
    >>> envelope = extractor.extract(audio[:, 0])   # mono or pick a channel
    >>> envelope.shape  # (n_samples_at_512Hz,)
    """

    def __init__(
        self,
        fs_audio: int = 44100,
        fs_eeg: int = 512,
        n_bands: int = _DEFAULT_N_BANDS,
        f_low: float = _DEFAULT_F_LOW,
        f_high: float = _DEFAULT_F_HIGH,
        compress: bool = True,
    ) -> None:
        self.fs_audio = fs_audio
        self.fs_eeg = fs_eeg
        self.n_bands = n_bands
        self.f_low = f_low
        self.f_high = f_high
        self.compress = compress
        self._filterbank = self._build_filterbank()

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Compute the broadband envelope of a mono audio signal.

        Parameters
        ----------
        audio : np.ndarray, shape (n_samples,)
            Mono audio waveform at ``fs_audio`` sampling rate.

        Returns
        -------
        envelope : np.ndarray, shape (n_eeg_samples,)
            Broadband envelope downsampled to ``fs_eeg``.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono
        audio = audio.astype(np.float64)

        # 1. Apply each band filter, 2. Hilbert, 3. Sum instantaneous amplitudes
        env = np.zeros(len(audio), dtype=np.float64)
        for b_low, b_high in self._filterbank:
            band = self._bandpass(audio, b_low, b_high)
            env += np.abs(hilbert(band))

        # 4. Power-law compression
        if self.compress:
            env = np.power(np.maximum(env, 1e-10), _COMPRESSION_EXPONENT)

        # 5. Downsample to EEG sampling rate
        env = self._downsample(env, self.fs_audio, self.fs_eeg)
        return env.astype(np.float32)

    def extract_from_file(self, path: str | Path) -> tuple[np.ndarray, dict]:
        """Convenience wrapper: read a WAV/FLAC file and extract its envelope.

        Parameters
        ----------
        path : str or Path
            Path to the audio file (WAV, FLAC, or OGG).

        Returns
        -------
        envelope : np.ndarray
            Envelope at ``fs_eeg`` Hz.
        meta : dict
            Audio file metadata (duration, original fs, n_channels).
        """
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError(
                "soundfile is required to read audio files. "
                "Install with: pip install soundfile"
            ) from exc

        audio, fs = sf.read(str(path), dtype="float32")
        if fs != self.fs_audio:
            logger.warning(
                "Audio file fs=%d Hz differs from extractor fs_audio=%d. "
                "Resampling audio before extraction.",
                fs, self.fs_audio,
            )
            audio = self._downsample(audio.T if audio.ndim > 1 else audio, fs, self.fs_audio)
            if audio.ndim > 1:
                audio = audio.T

        n_ch = audio.shape[1] if audio.ndim > 1 else 1
        duration_s = len(audio) / fs
        envelope = self.extract(audio)
        meta = {"duration_s": duration_s, "fs_original": fs, "n_channels": n_ch}
        return envelope, meta

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_filterbank(self) -> list[tuple[float, float]]:
        """Build a list of (f_low, f_high) band edges on an ERB scale."""
        erb_low = self._hz_to_erb(self.f_low)
        erb_high = self._hz_to_erb(self.f_high)
        erb_centers = np.linspace(erb_low, erb_high, self.n_bands)
        # Band edges: ±0.5 ERB around each center
        erb_edges = [(c - 0.5, c + 0.5) for c in erb_centers]
        # Convert back to Hz, clamp to Nyquist
        nyq = self.fs_audio / 2.0
        hz_edges = [
            (
                max(self._erb_to_hz(lo), self.f_low),
                min(self._erb_to_hz(hi), nyq * 0.99),
            )
            for lo, hi in erb_edges
        ]
        return hz_edges

    def _bandpass(self, signal: np.ndarray, f_low: float, f_high: float) -> np.ndarray:
        nyq = self.fs_audio / 2.0
        low = np.clip(f_low / nyq, 1e-6, 0.999)
        high = np.clip(f_high / nyq, 1e-6, 0.999)
        if low >= high:
            return np.zeros_like(signal)
        try:
            b, a = butter(4, [low, high], btype="band")
            return filtfilt(b, a, signal)
        except Exception:
            return np.zeros_like(signal)

    @staticmethod
    def _downsample(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
        if fs_in == fs_out:
            return signal
        g = gcd(int(fs_in), int(fs_out))
        up, down = int(fs_out) // g, int(fs_in) // g
        if signal.ndim == 1:
            return resample_poly(signal, up, down)
        # Multi-channel: resample along axis 0
        return np.stack([resample_poly(signal[:, i], up, down)
                         for i in range(signal.shape[1])], axis=1)

    @staticmethod
    def _hz_to_erb(f: float) -> float:
        """Convert frequency in Hz to ERB-rate scale."""
        return 21.4 * np.log10(4.37e-3 * f + 1)

    @staticmethod
    def _erb_to_hz(erb: float) -> float:
        """Convert ERB-rate back to Hz."""
        return (10 ** (erb / 21.4) - 1) / 4.37e-3

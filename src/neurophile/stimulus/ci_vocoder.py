"""
neurophile.stimulus.ci_vocoder
==============================
Acoustic vocoder simulating Cochlear Implant (CI) signal processing.

Purpose
-------
Standard AAD datasets use normal-hearing (NH) acoustic stimuli. This module
converts those stimuli into CI-simulated audio, allowing AAD models to be
trained and evaluated on acoustically degraded signals representative of
what a CI user hears.

How a Cochlear Implant Processes Sound
---------------------------------------
A CI's sound processor:
  1. Splits the incoming audio into N frequency bands (8–22 electrodes).
  2. Extracts the **amplitude envelope** of each band.
  3. Stimulates the corresponding cochlear electrode with a pulse train whose
     rate encodes the amplitude — not the fine spectral structure.

The listener therefore hears *amplitude modulations* across a small number of
channels, with no spectral fine structure (no pitch information for most users).

Vocoder Algorithm (Shannon et al., 1995)
-----------------------------------------
For each of the N channels:
  1. Bandpass filter the audio to the channel's frequency range
     (ERB-spaced from f_low to f_high, matching real CI tonotopy).
  2. Extract the Hilbert envelope of the filtered signal.
  3. Modulate a white noise (or sine wave) carrier with the envelope.
  4. Sum all modulated carriers → vocoded output.

The result is an acoustic simulation of CI hearing for signal processing
and model testing.

References
----------
- Shannon et al. (1995) "Speech recognition with primarily temporal cues"
  Science 270:303–304. doi:10.1126/science.270.5234.303
- Loizou (1998) "Mimicking the human ear" IEEE Signal Processing Magazine
- Friesen et al. (2001) "Speech recognition in noise as a function of the
  number of spectral channels" J. Acoust. Soc. Am. 110:1150–1163.
"""

from __future__ import annotations

import logging
from math import gcd

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, resample_poly

logger = logging.getLogger(__name__)

# ── Default CI channel configuration ──────────────────────────────────────────
_DEFAULT_N_CHANNELS = 16    # typical modern CI: 12–22 electrodes
_DEFAULT_F_LOW = 200.0      # Hz — lower edge of CI tonotopy
_DEFAULT_F_HIGH = 7000.0    # Hz — upper edge of CI tonotopy
_FILTER_ORDER = 4


class CIVocoderSimulator:
    """Acoustic vocoder simulating Cochlear Implant signal processing.

    Parameters
    ----------
    fs : int
        Sampling rate of the input audio (Hz).
    n_channels : int
        Number of simulated CI electrode channels (8–22 recommended).
        More channels → better spectral resolution simulation.
    f_low : float
        Lower frequency bound of the CI frequency range (Hz).
        Real CIs start around 200–300 Hz.
    f_high : float
        Upper frequency bound (Hz). Most CIs cover up to 7–8 kHz.
    carrier : str
        Carrier signal type: ``"noise"`` (white noise) or ``"sine"``.
        Noise carriers produce a more natural-sounding CI simulation.
    seed : int or None
        Random seed for noise carrier reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 44100
    >>> audio = np.random.randn(fs * 5)   # 5 seconds of audio
    >>> vocoder = CIVocoderSimulator(fs=fs, n_channels=16)
    >>> ci_audio = vocoder.simulate(audio)
    >>> ci_audio.shape
    (220500,)

    >>> # Use fewer channels to simulate older or less effective CIs
    >>> ci_8ch = CIVocoderSimulator(fs=fs, n_channels=8)
    >>> degraded = ci_8ch.simulate(audio)
    """

    def __init__(
        self,
        fs: int = 44100,
        n_channels: int = _DEFAULT_N_CHANNELS,
        f_low: float = _DEFAULT_F_LOW,
        f_high: float = _DEFAULT_F_HIGH,
        carrier: str = "noise",
        seed: int | None = 42,
    ) -> None:
        if n_channels < 1:
            raise ValueError(f"n_channels must be ≥ 1, got {n_channels}")
        if carrier not in ("noise", "sine"):
            raise ValueError(f"carrier must be 'noise' or 'sine', got {carrier!r}")
        if f_low >= f_high:
            raise ValueError(f"f_low ({f_low}) must be < f_high ({f_high})")

        self.fs = fs
        self.n_channels = n_channels
        self.f_low = f_low
        self.f_high = f_high
        self.carrier = carrier
        self.rng = np.random.default_rng(seed)

        # Precompute ERB-spaced band edges matching real CI tonotopy
        self._bands = self._build_erb_bands()
        logger.debug(
            "CIVocoderSimulator: fs=%d, n_channels=%d, f=[%.0f–%.0f Hz], carrier=%s",
            fs, n_channels, f_low, f_high, carrier,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def simulate(self, audio: np.ndarray) -> np.ndarray:
        """Convert normal-hearing audio to a CI-vocoded simulation.

        Parameters
        ----------
        audio : np.ndarray, shape (n_samples,) or (n_samples, n_ch)
            Input audio at ``self.fs`` Hz. Stereo is averaged to mono.

        Returns
        -------
        ci_audio : np.ndarray, shape (n_samples,)
            Vocoded audio normalised to the same RMS as the input.
        """
        # ── Normalise to mono float64 ─────────────────────────────────────────
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float64)
        input_rms = np.sqrt(np.mean(audio ** 2)) + 1e-12

        output = np.zeros_like(audio)

        for ch_idx, (f_lo, f_hi) in enumerate(self._bands):
            # 1. Bandpass filter
            filtered = self._bandpass(audio, f_lo, f_hi)

            # 2. Hilbert envelope of the band
            envelope = np.abs(hilbert(filtered))

            # 3. Generate carrier signal for this channel
            carrier_sig = self._make_carrier(len(audio), f_lo, f_hi)

            # 4. Amplitude-modulate carrier with envelope
            modulated = carrier_sig * envelope

            output += modulated
            logger.debug("  ch %02d: %.0f–%.0f Hz, env_rms=%.4f", ch_idx + 1, f_lo, f_hi, np.sqrt(np.mean(envelope**2)))

        # ── RMS normalisation ─────────────────────────────────────────────────
        output_rms = np.sqrt(np.mean(output ** 2)) + 1e-12
        output = output * (input_rms / output_rms)

        return output.astype(np.float32)

    def simulate_and_extract_envelope(
        self,
        audio: np.ndarray,
        fs_eeg: int = 64,
        compress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vocode and extract the broadband envelope at the EEG sampling rate.

        This is the combined Step 2+3 of the blueprint's training pipeline:
        vocode to CI → extract low-frequency envelope.

        Parameters
        ----------
        audio : np.ndarray, shape (n_samples,)
            Input audio at ``self.fs``.
        fs_eeg : int
            Target EEG sampling rate (envelope downsampled to this rate).
        compress : bool
            Apply power-law compression (exponent 0.3) to the envelope.

        Returns
        -------
        ci_audio : np.ndarray, shape (n_samples,)
            CI-vocoded audio at ``self.fs``.
        ci_envelope : np.ndarray, shape (n_eeg_samples,)
            Broadband envelope downsampled to ``fs_eeg``.
        """
        ci_audio = self.simulate(audio)

        # Broadband envelope of the vocoded signal
        env = np.zeros(len(ci_audio), dtype=np.float64)
        for f_lo, f_hi in self._bands:
            band = self._bandpass(ci_audio.astype(np.float64), f_lo, f_hi)
            env += np.abs(hilbert(band))

        if compress:
            env = np.power(np.maximum(env, 1e-10), 0.3)

        # Low-pass filter (AAD relevant: 0.5–8 Hz)
        env = self._lowpass(env, cutoff_hz=8.0)

        # Downsample to EEG rate
        ci_envelope = self._downsample(env, self.fs, fs_eeg)
        return ci_audio, ci_envelope.astype(np.float32)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_erb_bands(self) -> list[tuple[float, float]]:
        """Build N ERB-spaced frequency bands matching real CI tonotopy."""
        erb_lo = self._hz_to_erb(self.f_low)
        erb_hi = self._hz_to_erb(self.f_high)
        centers = np.linspace(erb_lo, erb_hi, self.n_channels)
        # Band edges ±0.5 ERB around each center
        nyq = self.fs / 2.0
        bands = []
        for c in centers:
            lo = max(self._erb_to_hz(c - 0.5), self.f_low)
            hi = min(self._erb_to_hz(c + 0.5), nyq * 0.99)
            if lo < hi:
                bands.append((lo, hi))
        return bands

    def _make_carrier(self, n: int, f_lo: float, f_hi: float) -> np.ndarray:
        """Generate the carrier signal for one CI channel."""
        if self.carrier == "noise":
            noise = self.rng.standard_normal(n)
            # Bandpass the noise to the channel's frequency range
            return self._bandpass(noise, f_lo, f_hi)
        else:  # sine
            # Use the geometric mean of the band as the sine frequency
            center_hz = np.sqrt(f_lo * f_hi)
            t = np.arange(n) / self.fs
            return np.sin(2 * np.pi * center_hz * t)

    def _bandpass(self, signal: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
        nyq = self.fs / 2.0
        lo = np.clip(f_lo / nyq, 1e-6, 0.999)
        hi = np.clip(f_hi / nyq, 1e-6, 0.999)
        if lo >= hi:
            return np.zeros_like(signal)
        try:
            b, a = butter(_FILTER_ORDER, [lo, hi], btype="band")
            return filtfilt(b, a, signal)
        except Exception:
            return np.zeros_like(signal)

    def _lowpass(self, signal: np.ndarray, cutoff_hz: float = 8.0) -> np.ndarray:
        nyq = self.fs / 2.0
        cutoff = min(cutoff_hz / nyq, 0.999)
        b, a = butter(_FILTER_ORDER, cutoff, btype="low")
        return filtfilt(b, a, signal)

    @staticmethod
    def _downsample(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
        if fs_in == fs_out:
            return signal
        g = gcd(int(fs_in), int(fs_out))
        up, down = int(fs_out) // g, int(fs_in) // g
        return resample_poly(signal, up, down)

    @staticmethod
    def _hz_to_erb(f: float) -> float:
        return 21.4 * np.log10(4.37e-3 * f + 1)

    @staticmethod
    def _erb_to_hz(erb: float) -> float:
        return (10 ** (erb / 21.4) - 1) / 4.37e-3

    def __repr__(self) -> str:
        return (
            f"CIVocoderSimulator(fs={self.fs}, n_channels={self.n_channels}, "
            f"f=[{self.f_low:.0f}–{self.f_high:.0f} Hz], carrier={self.carrier!r})"
        )

"""Numerical DSP engine for the Sampling & Aliasing Visualizer.

All routines are pure functions so they can be tested independently of the
Streamlit interface.  Frequencies are in hertz, times are in seconds.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal
from scipy.io import wavfile

Waveform = Literal["Sine", "Square", "Triangular"]
Reconstruction = Literal[
    "Whittaker-Shannon (Sinc)",
    "Linear Interpolation",
    "Zero-Order Hold (ZOH)",
]

WAVEFORMS: tuple[str, ...] = ("Sine", "Square", "Triangular")
RECONSTRUCTION_METHODS: tuple[str, ...] = (
    "Whittaker-Shannon (Sinc)",
    "Linear Interpolation",
    "Zero-Order Hold (ZOH)",
)

AUDIO_RATE = 44_100
AUDIO_DURATION = 2.0
AUDIO_CUTOFF = 20_000.0

ANALYSIS_FFT_SIZE = 32_768
SAMPLED_FFT_SIZE = 32_768
SAMPLED_SEQUENCE_LENGTH = 8_192


@dataclass(frozen=True)
class SampleSet:
    """Uniformly spaced samples of a continuous-time waveform."""

    times: np.ndarray
    values: np.ndarray
    sampling_rate: float


@dataclass(frozen=True)
class SpectrumResult:
    """Frequency-domain data for the spectrum panel."""

    original_freqs: np.ndarray
    original_magnitude: np.ndarray
    sampled_freqs: np.ndarray
    sampled_magnitude: np.ndarray
    f_max: float
    y_max: float
    nyquist: float


@dataclass(frozen=True)
class FoldResult:
    """Theoretical aliasing fold-back locus."""

    sampling_freqs: np.ndarray
    alias_freqs: np.ndarray


@dataclass(frozen=True)
class AudioResult:
    """Two in-memory 16-bit PCM WAV streams."""

    original_wav: bytes
    reconstructed_wav: bytes


def signal_waveform(t: np.ndarray, f0: float, waveform: str) -> np.ndarray:
    """Evaluate an ideal continuous waveform at time instants ``t``."""

    t = np.asarray(t, dtype=float)
    phase = 2.0 * np.pi * f0 * t

    if waveform == "Sine":
        return np.sin(phase)
    if waveform == "Square":
        return signal.square(phase)
    if waveform == "Triangular":
        return signal.sawtooth(phase, width=0.5)
    raise ValueError(f"Unknown waveform: {waveform!r}")


def bandlimited_waveform(
    t: np.ndarray,
    f0: float,
    waveform: str,
    cutoff: float,
) -> np.ndarray:
    """Ideal low-pass filtered waveform with a brick-wall cutoff.

    For square and triangular waves this is the Fourier partial sum containing
    only odd harmonics at or below ``cutoff``.  The phase conventions match
    ``scipy.signal.square`` and ``scipy.signal.sawtooth(..., width=0.5)``.
    """

    t = np.asarray(t, dtype=float)
    phase = 2.0 * np.pi * f0 * t

    if waveform == "Sine":
        if cutoff < f0:
            return np.zeros_like(t)
        return np.sin(phase)

    if waveform not in ("Square", "Triangular"):
        raise ValueError(f"Unknown waveform: {waveform!r}")

    if cutoff < f0:
        return np.zeros_like(t)

    highest = int(np.floor(cutoff / f0 + 1e-12))
    odd = np.arange(1, highest + 1, 2, dtype=float)
    if odd.size == 0:
        return np.zeros_like(t)

    harmonics = np.sin(phase[:, None] * odd[None, :])

    if waveform == "Square":
        coefficients = (4.0 / np.pi) / odd
    else:
        # Matches sawtooth(phase, width=0.5): triangle starts at -1.
        harmonics = np.cos(phase[:, None] * odd[None, :])
        coefficients = -(8.0 / np.pi**2) / odd**2

    return harmonics @ coefficients


def sample_signal(
    f0: float,
    fs: float,
    waveform: str,
    duration: float,
    *,
    prefilter: bool = False,
    start_time: float = 0.0,
) -> SampleSet:
    """Sample a waveform at ``fs`` over ``duration`` seconds."""

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if duration <= 0:
        raise ValueError("Duration must be positive")

    count = int(np.floor(duration * fs + 1e-12)) + 1
    times = start_time + np.arange(count, dtype=float) / fs

    if prefilter:
        values = bandlimited_waveform(times, f0, waveform, cutoff=fs / 2.0)
    else:
        values = signal_waveform(times, f0, waveform)

    return SampleSet(times=times, values=values, sampling_rate=float(fs))


def _sinc_reconstruct(
    t: np.ndarray,
    sample_times: np.ndarray,
    sample_values: np.ndarray,
    fs: float,
    half_width: int,
) -> np.ndarray:
    """Windowed Whittaker-Shannon interpolation.

    Only ``half_width`` samples on either side of each output instant are used,
    which keeps interactive updates fast while remaining visually identical to
    the full infinite sum inside the displayed window.
    """

    period = 1.0 / fs
    insertion = np.searchsorted(sample_times, t)
    output = np.zeros_like(t)
    count = sample_times.size

    for offset in range(-half_width, half_width + 1):
        index = insertion + offset
        valid = (index >= 0) & (index < count)
        if not np.any(valid):
            continue
        index_valid = index[valid]
        delta = t[valid] - sample_times[index_valid]
        output[valid] += sample_values[index_valid] * np.sinc(delta / period)

    return output


def reconstruct_signal(
    t: np.ndarray,
    samples: SampleSet,
    method: str,
    *,
    sinc_half_width: int = 64,
) -> np.ndarray:
    """Reconstruct a continuous waveform from uniform samples."""

    t = np.asarray(t, dtype=float)
    times = samples.times
    values = samples.values

    if times.size == 0:
        return np.zeros_like(t)

    if method == "Whittaker-Shannon (Sinc)":
        return _sinc_reconstruct(
            t, times, values, samples.sampling_rate, sinc_half_width
        )

    if method == "Linear Interpolation":
        return np.interp(t, times, values, left=0.0, right=0.0)

    if method == "Zero-Order Hold (ZOH)":
        index = np.searchsorted(times, t, side="right") - 1
        output = np.zeros_like(t)
        valid = index >= 0
        output[valid] = values[np.clip(index[valid], 0, values.size - 1)]
        return output

    raise ValueError(f"Unknown reconstruction method: {method!r}")


def apparent_frequency(f0: float, fs: float) -> float:
    """Aliased (apparent) frequency after sampling at ``fs``."""

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    return float(abs(f0 - np.round(f0 / fs) * fs))


def nyquist_state(f0: float, fs: float) -> str:
    """Return ``satisfied``, ``boundary``, or ``aliasing``."""

    tolerance = max(1e-6, 1e-4 * f0)
    if abs(fs - 2.0 * f0) <= tolerance:
        return "boundary"
    return "satisfied" if fs > 2.0 * f0 else "aliasing"


def fold_back_curve(f0: float, fs_max: float, points: int = 900) -> FoldResult:
    """Triangular fold-back locus for ``fs`` from ``0.5 f0`` to ``fs_max``."""

    fs_values = np.linspace(0.5 * f0, fs_max, points)
    alias_values = np.abs(f0 - np.round(f0 / fs_values) * fs_values)
    return FoldResult(sampling_freqs=fs_values, alias_freqs=alias_values)


def _analysis_rate(f0: float, fs: float) -> float:
    """Oversampled rate used to inspect the continuous spectrum."""

    return max(64.0 * f0, 4.0 * fs, 4_000.0)


def original_spectrum(
    f0: float,
    fs: float,
    waveform: str,
    *,
    fft_size: int = ANALYSIS_FFT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum of the ideal continuous waveform."""

    rate = _analysis_rate(f0, fs)
    t = np.arange(fft_size, dtype=float) / rate
    x = signal_waveform(t, f0, waveform)

    window = np.hanning(fft_size)
    magnitude = 2.0 * np.abs(np.fft.rfft(x * window)) / np.sum(window)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / rate)
    return freqs, magnitude


def sampled_spectrum(
    f0: float,
    fs: float,
    waveform: str,
    *,
    prefilter: bool = False,
    sequence_length: int = SAMPLED_SEQUENCE_LENGTH,
    fft_size: int = SAMPLED_FFT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseband DTFT magnitude of the discrete sequence, 0 to ``fs / 2``."""

    duration = sequence_length / fs
    samples = sample_signal(
        f0, fs, waveform, duration, prefilter=prefilter
    )
    x = samples.values[:sequence_length]

    window = np.hanning(x.size)
    magnitude = 2.0 * np.abs(
        np.fft.rfft(x * window, n=fft_size)
    ) / np.sum(window)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / fs)
    return freqs, magnitude


def _periodic_magnitude(
    base_freqs: np.ndarray,
    base_magnitude: np.ndarray,
    fs: float,
    query_freqs: np.ndarray,
) -> np.ndarray:
    """Repeat the baseband spectrum every ``fs`` hertz."""

    folded = np.abs((query_freqs + fs / 2.0) % fs - fs / 2.0)
    return np.interp(folded, base_freqs, base_magnitude, left=0.0, right=0.0)


def spectrum_curves(
    f0: float,
    fs: float,
    waveform: str,
    *,
    prefilter: bool = False,
) -> SpectrumResult:
    """Assemble original and periodic sampled spectra for plotting."""

    original_freqs, original_magnitude = original_spectrum(f0, fs, waveform)
    base_freqs, base_magnitude = sampled_spectrum(
        f0, fs, waveform, prefilter=prefilter
    )

    f_max = max(3.0 * f0, 1.15 * fs)
    query = np.linspace(0.0, f_max, 2_400)
    sampled_magnitude = _periodic_magnitude(
        base_freqs, base_magnitude, fs, query
    )

    y_max = max(
        float(np.max(original_magnitude)),
        float(np.max(sampled_magnitude)),
        1e-6,
    )

    return SpectrumResult(
        original_freqs=original_freqs,
        original_magnitude=original_magnitude,
        sampled_freqs=query,
        sampled_magnitude=sampled_magnitude,
        f_max=float(f_max),
        y_max=1.12 * y_max,
        nyquist=fs / 2.0,
    )


def _lowpass_fft(x: np.ndarray, rate: float, cutoff: float) -> np.ndarray:
    """Brick-wall low-pass filter used only for 44.1 kHz audio playback."""

    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / rate)
    spectrum[freqs > cutoff] = 0.0
    return np.fft.irfft(spectrum, n=x.size)


def _to_wav_bytes(x: np.ndarray, rate: int) -> bytes:
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, rate, pcm)
    return buffer.getvalue()


def synthesize_audio(
    f0: float,
    fs: float,
    waveform: str,
    method: str,
    *,
    prefilter: bool = False,
    duration: float = AUDIO_DURATION,
    rate: int = AUDIO_RATE,
) -> AudioResult:
    """Render the original and sampled/reconstructed signals as WAV bytes."""

    count = int(round(duration * rate))
    t = np.arange(count, dtype=float) / rate

    original = signal_waveform(t, f0, waveform)

    # A short guard interval removes reconstruction edge transients.
    guard = min(0.02, duration / 4.0)
    samples = sample_signal(
        f0,
        fs,
        waveform,
        duration + guard,
        prefilter=prefilter,
    )
    reconstructed = reconstruct_signal(
        t, samples, method, sinc_half_width=32
    )

    # The WAV container is 44.1 kHz.  Removing content above 20 kHz prevents
    # the container itself from adding a second, unrelated alias.
    original = _lowpass_fft(original, rate, AUDIO_CUTOFF)
    reconstructed = _lowpass_fft(reconstructed, rate, AUDIO_CUTOFF)

    peak = max(
        float(np.max(np.abs(original))),
        float(np.max(np.abs(reconstructed))),
        1e-9,
    )
    original = 0.9 * original / peak
    reconstructed = 0.9 * reconstructed / peak

    fade = max(1, int(0.01 * rate))
    envelope = np.ones(count)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    original *= envelope
    reconstructed *= envelope

    return AudioResult(
        original_wav=_to_wav_bytes(original, rate),
        reconstructed_wav=_to_wav_bytes(reconstructed, rate),
    )

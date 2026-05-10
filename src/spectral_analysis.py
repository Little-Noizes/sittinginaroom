"""Spectral analysis helpers for recursive acoustic generations."""

from __future__ import annotations

import librosa
import numpy as np

from .utils import validate_audio

_EPSILON = 1e-12


def compute_fft(audio: np.ndarray, sample_rate: int, n_fft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return FFT frequencies and magnitude in dBFS-like units.

    A Hann window is applied before the real FFT to reduce leakage in plots.
    """

    validate_audio(audio, label="audio")
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(max(audio.size, 2))))
    window = np.hanning(audio.size)
    spectrum = np.fft.rfft(audio * window, n=n_fft)
    magnitude = np.abs(spectrum)
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, _EPSILON))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return frequencies, magnitude_db


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return times, frequencies, and a dB-scaled magnitude spectrogram.

    Very short proof-of-concept clips use a smaller FFT window automatically so
    plotting stays quiet and useful.
    """

    validate_audio(audio, label="audio")
    effective_n_fft = min(n_fft, int(2 ** np.floor(np.log2(max(audio.size, 2)))))
    effective_hop = min(hop_length, max(1, effective_n_fft // 4))
    stft = librosa.stft(
        audio.astype(float),
        n_fft=effective_n_fft,
        hop_length=effective_hop,
        window="hann",
    )
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=effective_n_fft)
    times = librosa.frames_to_time(
        np.arange(spectrogram_db.shape[1]),
        sr=sample_rate,
        hop_length=effective_hop,
    )
    return times, frequencies, spectrogram_db

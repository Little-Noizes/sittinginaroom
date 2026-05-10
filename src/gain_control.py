"""Gain and normalisation strategies for recursive generations."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .utils import validate_audio

_EPSILON = 1e-12


def apply_fixed_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    """Apply a scalar loop gain without automatic protection or normalisation."""

    validate_audio(audio, label="audio")
    return np.asarray(audio * float(gain), dtype=np.float32)


def peak_normalise(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Scale audio so its absolute peak equals ``target_peak``."""

    validate_audio(audio, label="audio")
    peak = float(np.max(np.abs(audio)))
    if peak < _EPSILON:
        return np.asarray(audio, dtype=np.float32)
    return np.asarray(audio * (target_peak / peak), dtype=np.float32)


def rms_normalise(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Scale audio so its broadband RMS equals ``target_rms``."""

    validate_audio(audio, label="audio")
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < _EPSILON:
        return np.asarray(audio, dtype=np.float32)
    return np.asarray(audio * (target_rms / rms), dtype=np.float32)


def low_frequency_rms_normalise(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float = 20.0,
    high_hz: float = 200.0,
    target_rms: float = 0.1,
) -> np.ndarray:
    """Normalise using RMS measured in a low-frequency analysis band.

    The full-band signal is scaled by the RMS of a 20--200 Hz band-pass filtered
    copy.  This option is useful for experiments that focus on low-frequency
    recursive build-up while keeping the process room-agnostic.
    """

    validate_audio(audio, label="audio")
    nyquist = sample_rate / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError("low_hz and high_hz must define a valid band below Nyquist.")
    sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, audio)
    band_rms = float(np.sqrt(np.mean(np.square(filtered))))
    if band_rms < _EPSILON:
        return np.asarray(audio, dtype=np.float32)
    return np.asarray(audio * (target_rms / band_rms), dtype=np.float32)


def normalise_generation(
    audio: np.ndarray,
    sample_rate: int,
    method: str = "peak",
    target_level: float = 0.95,
) -> np.ndarray:
    """Apply a named normalisation method to one recursive generation.

    Supported methods are ``"none"``, ``"peak"``, ``"rms"``, and
    ``"low_frequency_rms"``.
    """

    method = method.lower()
    if method == "none":
        return np.asarray(audio, dtype=np.float32)
    if method == "peak":
        return peak_normalise(audio, target_peak=target_level)
    if method == "rms":
        return rms_normalise(audio, target_rms=target_level)
    if method in {"low_frequency_rms", "low-frequency-rms", "lf_rms"}:
        return low_frequency_rms_normalise(audio, sample_rate, target_rms=target_level)
    raise ValueError(f"Unsupported normalisation method: {method}")

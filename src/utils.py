"""Utility helpers for the recursive acoustic feedback framework.

The helpers in this module intentionally make few assumptions about the
recording space.  They only handle generic file I/O, directory creation, and
small signal-safety checks shared by the processing modules.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ArrayLike1D = np.ndarray


def ensure_directory(path: str | Path) -> Path:
    """Create *path* if needed and return it as a :class:`Path`.

    Parameters
    ----------
    path:
        Directory path to create.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_mono_wav(path: str | Path, target_sr: int | None = None) -> tuple[ArrayLike1D, int]:
    """Load a mono WAV file as ``float32`` audio.

    Parameters
    ----------
    path:
        WAV file to load.
    target_sr:
        Optional sample rate.  When provided, librosa resamples the file.

    Returns
    -------
    audio, sample_rate:
        A one-dimensional audio array and its sample rate.
    """

    audio, sample_rate = librosa.load(path, sr=target_sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    validate_audio(audio, label=str(path))
    return audio, int(sample_rate)


def validate_audio(audio: ArrayLike1D, label: str = "audio") -> None:
    """Validate that an audio vector is mono, non-empty, and finite."""

    if audio.ndim != 1:
        raise ValueError(f"{label} must be mono/one-dimensional; got shape {audio.shape}.")
    if audio.size == 0:
        raise ValueError(f"{label} is empty.")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{label} contains NaN or infinite values.")


def write_wav(path: str | Path, audio: ArrayLike1D, sample_rate: int) -> Path:
    """Write a mono WAV file without imposing integer clipping.

    The recursive fixed-gain path may exceed ``[-1, 1]`` by design, so files are
    written as 32-bit floating-point WAVs.  This preserves analysis/playback
    data while leaving final gain staging to the user.
    """

    validate_audio(audio, label="audio to write")
    output_path = Path(path)
    ensure_directory(output_path.parent)
    sf.write(output_path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="FLOAT")
    return output_path


def limit_duration(audio: ArrayLike1D, sample_rate: int, max_seconds: float | None) -> ArrayLike1D:
    """Optionally truncate ``audio`` to ``max_seconds`` seconds.

    This is a pragmatic proof-of-concept safeguard for recursive convolution,
    whose full-convolution output grows with every generation.
    """

    if max_seconds is None:
        return audio
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive when provided.")
    max_samples = int(round(max_seconds * sample_rate))
    return audio[:max_samples]

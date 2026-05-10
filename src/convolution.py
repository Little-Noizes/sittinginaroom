"""Convolution primitives for recursive acoustic feedback."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .utils import validate_audio


def convolve_signal(signal: np.ndarray, impulse_response: np.ndarray, mode: str = "full") -> np.ndarray:
    """Convolve a mono signal with a mono impulse response.

    Parameters
    ----------
    signal:
        Current recursive generation ``x_n``.
    impulse_response:
        Measured transfer function/impulse response ``h``.
    mode:
        Passed to :func:`scipy.signal.fftconvolve`; ``"full"`` preserves the
        complete response and is the default for the recursive model.
    """

    validate_audio(signal, label="signal")
    validate_audio(impulse_response, label="impulse_response")
    convolved = fftconvolve(signal, impulse_response, mode=mode)
    return np.asarray(convolved, dtype=np.float32)

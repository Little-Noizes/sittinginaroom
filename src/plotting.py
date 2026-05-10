"""Plotting utilities for recursive acoustic feedback outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .spectral_analysis import compute_fft, compute_spectrogram
from .utils import ensure_directory


def save_fft_plot(audio: np.ndarray, sample_rate: int, output_path: str | Path, title: str) -> Path:
    """Save a log-frequency FFT magnitude plot for one generation."""

    frequencies, magnitude_db = compute_fft(audio, sample_rate)
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    fig, ax = plt.subplots(figsize=(10, 5))
    nonzero = frequencies > 0
    ax.semilogx(frequencies[nonzero], magnitude_db[nonzero], linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(left=20, right=sample_rate / 2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_spectrogram_plot(audio: np.ndarray, sample_rate: int, output_path: str | Path, title: str) -> Path:
    """Save a spectrogram plot for one generation."""

    times, frequencies, spectrogram_db = compute_spectrogram(audio, sample_rate)
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(times, frequencies, spectrogram_db, shading="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, sample_rate / 2)
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB, ref=max)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

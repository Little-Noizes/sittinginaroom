"""Recursive convolution pipeline for acoustic feedback experiments.

This module provides the two Stage 1 proof-of-concept paths:

* fixed-gain recursive playback, which repeatedly convolves and applies a loop
  gain without automatic normalisation; and
* normalised recursive analysis, which normalises each generation for easier
  spectral comparison and convergence inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .convolution import convolve_signal
from .gain_control import apply_fixed_gain, normalise_generation
from .plotting import save_fft_plot, save_spectrogram_plot
from .utils import ensure_directory, limit_duration, load_mono_wav, write_wav


@dataclass(frozen=True)
class GenerationOutput:
    """File paths and basic measurements for one recursive generation."""

    generation: int
    wav_path: Path
    fft_plot_path: Path
    spectrogram_plot_path: Path
    peak: float
    rms: float
    duration_seconds: float


def _save_generation(
    audio: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    path_name: str,
    generation: int,
) -> GenerationOutput:
    """Persist WAV, FFT, and spectrogram outputs for one generation."""

    wav_dir = ensure_directory(output_dir / "wav")
    fft_dir = ensure_directory(output_dir / "fft")
    spectrogram_dir = ensure_directory(output_dir / "spectrogram")
    stem = f"{path_name}_generation_{generation:02d}"
    title = f"{path_name.replace('_', ' ').title()} — Generation {generation}"

    wav_path = write_wav(wav_dir / f"{stem}.wav", audio, sample_rate)
    fft_path = save_fft_plot(audio, sample_rate, fft_dir / f"{stem}_fft.png", title=f"FFT: {title}")
    spectrogram_path = save_spectrogram_plot(
        audio,
        sample_rate,
        spectrogram_dir / f"{stem}_spectrogram.png",
        title=f"Spectrogram: {title}",
    )

    return GenerationOutput(
        generation=generation,
        wav_path=wav_path,
        fft_plot_path=fft_path,
        spectrogram_plot_path=spectrogram_path,
        peak=float(np.max(np.abs(audio))),
        rms=float(np.sqrt(np.mean(np.square(audio)))),
        duration_seconds=float(audio.size / sample_rate),
    )


def run_recursive_feedback(
    input_wav: str | Path,
    impulse_response_wav: str | Path,
    output_dir: str | Path,
    generations: int,
    path_name: str,
    loop_gain: float = 1.0,
    normalisation: str | None = None,
    normalisation_level: float = 0.95,
    max_output_seconds: float | None = None,
) -> list[GenerationOutput]:
    """Run recursive convolution and save every generation.

    Parameters
    ----------
    input_wav:
        Mono excitation WAV, used as generation 0.
    impulse_response_wav:
        Mono measured impulse response WAV.  It is resampled to the input WAV's
        sample rate if needed.
    output_dir:
        Directory that receives ``wav/``, ``fft/``, and ``spectrogram/`` trees.
    generations:
        Number of recursive convolution steps after generation 0.
    path_name:
        Name used in output filenames, e.g. ``"fixed_gain"``.
    loop_gain:
        Scalar gain applied after each convolution.
    normalisation:
        Optional normalisation method applied after fixed gain.  Use ``None``
        for the physical fixed-gain path.
    normalisation_level:
        Target level for the selected normalisation method.
    max_output_seconds:
        Optional truncation limit to keep proof-of-concept runs small.
    """

    if generations < 0:
        raise ValueError("generations must be non-negative.")

    output_dir = ensure_directory(output_dir)
    current, sample_rate = load_mono_wav(input_wav)
    impulse_response, _ = load_mono_wav(impulse_response_wav, target_sr=sample_rate)

    outputs = [_save_generation(current, sample_rate, output_dir, path_name, generation=0)]

    for generation in range(1, generations + 1):
        # The recursive model: x_{n+1}(t) = G [x_n(t) * h(t)].
        current = convolve_signal(current, impulse_response, mode="full")
        current = apply_fixed_gain(current, loop_gain)
        if normalisation is not None:
            current = normalise_generation(
                current,
                sample_rate=sample_rate,
                method=normalisation,
                target_level=normalisation_level,
            )
        current = limit_duration(current, sample_rate, max_output_seconds)
        outputs.append(_save_generation(current, sample_rate, output_dir, path_name, generation))

    return outputs


def run_fixed_gain_recursive_playback(
    input_wav: str | Path,
    impulse_response_wav: str | Path,
    output_dir: str | Path,
    generations: int = 5,
    loop_gain: float = 0.5,
    max_output_seconds: float | None = None,
) -> list[GenerationOutput]:
    """Run Path A: fixed-gain recursive playback with no normalisation."""

    return run_recursive_feedback(
        input_wav=input_wav,
        impulse_response_wav=impulse_response_wav,
        output_dir=Path(output_dir) / "fixed_gain",
        generations=generations,
        path_name="fixed_gain",
        loop_gain=loop_gain,
        normalisation=None,
        max_output_seconds=max_output_seconds,
    )


def run_normalised_recursive_analysis(
    input_wav: str | Path,
    impulse_response_wav: str | Path,
    output_dir: str | Path,
    generations: int = 5,
    loop_gain: float = 1.0,
    normalisation: str = "peak",
    normalisation_level: float = 0.95,
    max_output_seconds: float | None = None,
) -> list[GenerationOutput]:
    """Run Path B: recursive analysis with per-generation normalisation."""

    return run_recursive_feedback(
        input_wav=input_wav,
        impulse_response_wav=impulse_response_wav,
        output_dir=Path(output_dir) / "normalised",
        generations=generations,
        path_name=f"normalised_{normalisation}",
        loop_gain=loop_gain,
        normalisation=normalisation,
        normalisation_level=normalisation_level,
        max_output_seconds=max_output_seconds,
    )

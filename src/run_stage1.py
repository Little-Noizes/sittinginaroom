"""Small command-line entry point for the Stage 1 proof-of-concept."""

from __future__ import annotations

import argparse

from .recursive_loop import run_fixed_gain_recursive_playback, run_normalised_recursive_analysis


def main() -> None:
    """Run both recursive paths from the command line."""

    parser = argparse.ArgumentParser(description="Run recursive acoustic feedback Stage 1 outputs.")
    parser.add_argument("input_wav", help="Mono input/excitation WAV file.")
    parser.add_argument("impulse_response_wav", help="Mono measured impulse response WAV file.")
    parser.add_argument("--output-dir", default="data/outputs/stage1", help="Output directory.")
    parser.add_argument("--generations", type=int, default=3, help="Number of recursive generations after generation 0.")
    parser.add_argument("--loop-gain", type=float, default=0.5, help="Fixed-gain path loop gain.")
    parser.add_argument("--normalised-loop-gain", type=float, default=1.0, help="Normalised path loop gain.")
    parser.add_argument("--normalisation", default="peak", choices=["peak", "rms", "low_frequency_rms"], help="Normalised path method.")
    parser.add_argument("--normalisation-level", type=float, default=0.95, help="Normalisation target level.")
    parser.add_argument("--max-output-seconds", type=float, default=None, help="Optional truncation limit for each generation.")
    args = parser.parse_args()

    run_fixed_gain_recursive_playback(
        args.input_wav,
        args.impulse_response_wav,
        args.output_dir,
        generations=args.generations,
        loop_gain=args.loop_gain,
        max_output_seconds=args.max_output_seconds,
    )
    run_normalised_recursive_analysis(
        args.input_wav,
        args.impulse_response_wav,
        args.output_dir,
        generations=args.generations,
        loop_gain=args.normalised_loop_gain,
        normalisation=args.normalisation,
        normalisation_level=args.normalisation_level,
        max_output_seconds=args.max_output_seconds,
    )


if __name__ == "__main__":
    main()

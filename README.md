# Recursive Acoustic Feedback

## Overview

This project investigates recursive electroacoustic playback using measured impulse responses.

The framework explores how repeated application of a room or system impulse response progressively transforms an excitation signal, potentially emphasising dominant resonant behaviour within the coupled source–room–receiver transfer function.

The project is inspired by recursive electroacoustic processes explored in experimental sound art, while approaching the phenomenon from the perspective of acoustics, signal processing, and system analysis.

The repository is intentionally designed to remain room-agnostic and modular during early development stages.

---

## Core Concept

Given an input signal:

\[
x_0(t)
\]

and a measured impulse response:

\[
h(t)
\]

recursive generations are produced through repeated convolution:

\[
x_{n+1}(t) = G \left[ x_n(t) * h(t) \right]
\]

where:

- \(x_n(t)\) is the signal at generation \(n\),
- \(h(t)\) is the measured impulse response,
- \(G\) is either:
  - a fixed loop gain, or
  - a normalisation operator.

Two parallel processing paths are implemented:

### Path A — Physical Recursive Playback
- Fixed gain.
- No automatic normalisation.
- Intended to approximate physical electroacoustic recursive playback.

### Path B — Normalised Recursive Analysis
- Signal normalisation applied each generation.
- Intended for spectral convergence and modal/emergent behaviour analysis.

---

## Current Goals

The current development phase focuses on:

- building a robust recursive convolution framework;
- implementing gain-control strategies;
- generating recursive audio generations;
- visualising spectral evolution;
- tracking dominant frequency convergence;
- comparing recursive behaviours under different processing assumptions.

The framework is intentionally generic at this stage.

Specific:
- room geometries,
- modal predictions,
- acoustic datasets,
- and experimental conditions

will be introduced later.

---

## Planned Outputs

The framework will generate:

- recursive generation WAV files;
- FFT plots;
- spectrograms;
- spectral peak tracking data;
- convergence analysis plots;
- comparative processing-path visualisations.

---

## Directory Structure

```text
recursive-acoustic-feedback/
├── AGENTS.md
├── PROJECT_SPEC.md
├── README.md
├── requirements.txt
│
├── data/
│   ├── input_signals/
│   ├── impulse_responses/
│   └── outputs/
│
├── src/
│   ├── recursive_loop.py
│   ├── convolution.py
│   ├── gain_control.py
│   ├── spectral_analysis.py
│   ├── plotting.py
│   └── utils.py
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── convergence_tests.ipynb
│
├── figures/
│
└── paper/
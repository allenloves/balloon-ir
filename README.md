---
title: Balloon IR Synthesizer
emoji: "\U0001F388"
colorFrom: yellow
colorTo: yellow
sdk: docker
app_port: 7860
---

# Balloon IR Synthesizer

Convert balloon pop recordings into clean, full-bandwidth room impulse responses.

Based on Abel, Canfield-Dafilou & Holloway (2010), [*"Estimating Room Impulse Responses from Recorded Balloon Pops"*](https://www.researchgate.net/profile/Bissera-Pentcheva/publication/277732009_Estimating_Room_Impulse_Responses_from_Recorded_Balloon_Pops/links/563a4b4408ae45b5d284a8ce/Estimating-Room-Impulse-Responses-from-Recorded-Balloon-Pops.pdf), AES Convention 129, Paper 8171.

## Demo

- **Web App**: [allenloves.github.io/balloon-ir](https://allenloves.github.io/balloon-ir/)
- **API**: [allenloves-balloon-ir.hf.space](https://allenloves-balloon-ir.hf.space/docs)

## Background

A balloon pop produces an N-wave — a pressure pulse whose spectral nulls cause comb-filtering artifacts that make the raw recording unsuitable as a room impulse response. This tool implements an analysis-resynthesis approach: it analyzes the acoustic properties of the balloon recording (echo density, spatial characteristics, spectral energy) and synthesizes a new IR that preserves the room's reverberant character while eliminating the N-wave artifacts.

## Processing Pipeline

| Stage | Description | Key Equations |
|-------|-------------|---------------|
| **0. Preprocessing** | Onset detection, normalization, optional resampling | — |
| **1. Echo Density** | NED/AED analysis, N-wave characterization, sparse-to-dense transition detection, Poisson pulse synthesis | Eq. 3–5, 8 |
| **2. Spatial** | Inter-channel cross-correlation (ICCC) analysis, stereo correlation imposition via rotation matrix | Eq. 9–11 |
| **3. Energy Shaping** | 1/3-octave filter bank, band energy imprinting, direct path equalization, energy extrapolation below noise floor | Eq. 14, 16 |
| **4. Post-processing** | Normalization, fade-out, tail trimming, WAV export | — |

## Project Structure

```
core/               DSP pipeline modules
  preprocessing.py    Stage 0: onset detection, normalization
  echo_density.py     Stage 1: NED, AED, Poisson pulse synthesis
  spatial.py          Stage 2: ICCC analysis, stereo rotation
  energy_shaping.py   Stage 3: filter bank, energy imprinting
  postprocessing.py   Stage 4: normalize, fade, trim, export
  pipeline.py         Orchestrates all stages
  visualization.py    Diagnostic plot generation
api/                FastAPI web backend (legacy — used by the hosted demo)
  main.py             App entry point + CORS
  routes.py           Endpoints (process, status, result, preview)
  tasks.py            Background job management
frontend/           Static in-browser app (Pyodide; no backend required)
  index.html          UI
  src/main.js         Glue: upload, params, progress, audio playback
  src/wav.js          WAV decode / encode
  src/pipeline.js     Loads Pyodide + numpy + scipy + matplotlib + core/*.py
  src/bridge.py       Python entry point — reuses core/ pipeline unchanged
scripts/            CLI tools
  cli.py              Command-line processing
  generate_plots.py   Batch plot generation
  validate_ir.py      Room acoustic parameter validation (RT60, EDT, C80, D50, Ts)
tests/              Unit tests (76 tests)
```

## Quick Start

### CLI

```bash
conda activate dsp
python scripts/cli.py balloon.wav -o ir.wav
```

### CLI Options

```
--target-sr          Resample to target sample rate (default: keep original)
--onset-threshold    Onset detection threshold in dB (default: -40)
--ned-window         NED estimation window in ms (default: 43)
--balloon-diameter   Balloon diameter in cm (default: auto-detect)
--ned-transition     NED threshold for sparse/dense transition (default: 0.3)
--noise-floor        Noise floor threshold in dB (default: -40)
--pulse-halo         Gain halo around early pulses in ms (default: 2.0)
--target-dbfs        Output normalization level (default: -1.0)
--bit-depth          Output bit depth: 16, 24, or 32 (default: 24)
--seed               Random seed for reproducibility
```

### Local Web App

The frontend runs entirely in the browser via [Pyodide](https://pyodide.org/) —
no backend, no API server. Just serve the project directory as static files:

```bash
python -m http.server 8000
# open http://localhost:8000/frontend/
```

First visit downloads ~40 MB (Pyodide runtime + numpy + scipy + soundfile +
matplotlib wheels) and caches it. See [`frontend/README.md`](frontend/README.md)
for details.

## Deployment

| Component | Platform | URL |
|-----------|----------|-----|
| Frontend | GitHub Pages (static) | [allenloves.github.io/balloon-ir](https://allenloves.github.io/balloon-ir/) |
| Backend (legacy) | Hugging Face Spaces (Docker) | [allenloves-balloon-ir.hf.space](https://allenloves-balloon-ir.hf.space/) |

Auto-deploy on push to `main` via GitHub Actions. The frontend now ships as a
purely static site (no build step) and includes the Python `core/` modules so
Pyodide can load them client-side; the FastAPI backend is still deployed for
anyone who needs the HTTP API.

## Requirements

- Python 3.11+
- numpy, scipy, soundfile, matplotlib
- FastAPI, uvicorn (only for the legacy HTTP backend in `api/`)

## References

Abel, J. S., Canfield-Dafilou, E. K., & Holloway, M. (2010). [Estimating room impulse responses from recorded balloon pops](https://www.researchgate.net/profile/Bissera-Pentcheva/publication/277732009_Estimating_Room_Impulse_Responses_from_Recorded_Balloon_Pops/links/563a4b4408ae45b5d284a8ce/Estimating-Room-Impulse-Responses-from-Recorded-Balloon-Pops.pdf). *Audio Engineering Society Convention 129*, Paper 8171.

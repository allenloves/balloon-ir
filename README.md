# Balloon IR Synthesizer

Turn a recording of a balloon pop into a clean room impulse response (IR) — drop it into any convolution reverb to "place" sounds inside that room.

[**Try it now →**](https://allenloves.github.io/balloon-ir/) · runs entirely in your browser, no upload

Based on Abel, Canfield-Dafilou & Holloway, [*"Estimating Room Impulse Responses from Recorded Balloon Pops"*](https://www.researchgate.net/profile/Bissera-Pentcheva/publication/277732009_Estimating_Room_Impulse_Responses_from_Recorded_Balloon_Pops/links/563a4b4408ae45b5d284a8ce/Estimating-Room-Impulse-Responses-from-Recorded-Balloon-Pops.pdf), AES Convention 129, Paper 8171 (2010).

---

## What is this?

A **room impulse response (IR)** is the acoustic fingerprint of a space — its echoes, the length and color of its reverb tail, the way sound bounces between its walls. Loaded into a convolution reverb (Altiverb, Reverberate, free plugins like Convology XT), an IR lets you take a dry recording — a voice, a guitar, anything — and make it sound as if it were performed *in that room*.

The studio way to capture an IR is a sine sweep from a calibrated speaker. But there's a much cheaper trick: **pop a balloon**. The pop is loud, broadband, instantaneous — record it on your phone in any room and you've captured something close to that room's response.

The catch: a raw balloon pop isn't directly usable. The pop itself is an "N-wave" — a sharp pressure pulse with characteristic notches in its frequency content. Convolving with the raw recording would smear those notches across whatever you reverb, producing a hollow, comb-filtered sound.

**This tool fixes that.** It analyzes what your recording reveals about the room (when the early reflections arrive, how the energy decays in each frequency band, how stereo width evolves over time) and synthesizes a fresh IR with the same acoustic character but no N-wave artifacts.

## How to use it

1. **Record** a balloon pop in the room you want to capture. Any phone or recorder works. Position the mic where you'd want the listener to be, and the balloon roughly where you'd want a sound source. A stereo pair gives you a stereo IR.
2. **Drag** the WAV file onto [the website](https://allenloves.github.io/balloon-ir/) (or click to choose).
3. **Click "Synthesize IR"** and wait a few seconds.
4. **Play** the result, **Download** the WAV (24-bit), or scroll down to inspect the diagnostic plots.
5. **Load** the downloaded IR into your favorite convolution reverb.

The first visit to the site downloads ~40 MB of Python runtime + scientific libraries (numpy, scipy, matplotlib) that run inside your browser. Subsequent visits are instant. Your audio file never leaves your machine — there is no server.

## How it works

The pipeline runs in five stages, all in your browser:

| Stage | What it does |
|-------|--------------|
| **0. Preprocessing** | Find where the pop starts, normalize the level, trim the silence before it. |
| **1. Echo density** | Measure how reflections accumulate over time. Detect the moment they transition from discrete *early reflections* to a dense *diffuse tail*. Synthesize a Poisson-distributed pulse train as the carrier for the diffuse part. |
| **2. Spatial** *(stereo only)* | Measure how correlated the two channels are over time. Imprint that same correlation profile onto the synthesized IR's two channels. |
| **3. Energy shaping** | Split the recording into 1/3-octave frequency bands. Measure how energy decays in each band over time. Imprint the same decay profile onto the synthesized IR — band by band. |
| **4. Post-processing** | Normalize to the target output level, apply a smooth fade-out, trim the silent tail, export as WAV. |

The output preserves the room's actual early reflections (which carry most of its perceptual character) but replaces the messy diffuse tail with a synthesized one that has no N-wave artifacts.

## Parameters

The defaults work for typical clean balloon recordings. Open the **Parameters** panel only if you want to fine-tune.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| **onset_threshold_db** | −40 | How sensitive the onset detector is, in decibels relative to the recording's peak. **More negative** = catches the pop earlier (and is more easily fooled by background noise). Lower this if your recording has a soft attack or a long approach. |
| **ned_window_ms** | 43 | The analysis window length, in milliseconds, used to measure echo density. The paper's default. **Larger** = smoother density curve. **Smaller** = more responsive to local changes, but noisier. |
| **ned_transition_threshold** | 0.3 | The echo-density value at which the pipeline switches from preserving distinct early reflections to synthesizing a fully diffuse tail. **Higher** (e.g. 0.5) = preserve more of your recording's early reflections verbatim. **Lower** (e.g. 0.1) = switch to synthesized diffuse reverb sooner. |
| **pulse_halo_ms** | 2.0 | Width, in milliseconds, of an energy "halo" around each early reflection. Inside the halo, the IR keeps the original transient shape; outside, it gets shaped noise. **Increase** if the early reflections sound smeared in the result. |
| **noise_floor_db** | −40 | The level below which the recording is considered noise rather than reverb. The IR's energy decay is *extrapolated* below this floor, so the tail keeps decaying naturally instead of trailing off into recorded noise. **Lower** if your recording has a very long, very quiet tail you trust. |
| **gain_smoothing_ms** | 0 | Time-smoothing applied to the per-band gain function. Default 0 keeps fast transient detail. **Increase** (try 5–10) if you hear chirps or pumping artifacts. |
| **target_dbfs** | −1 | Output peak level in dBFS. −1 dBFS leaves a small headroom; set to −3 or −6 if you want more. |

## What the plots mean

After synthesis, six plots stream in (seven for stereo). Toggle off **"Generate diagnostic plots"** before processing if you don't need them.

- **Summary** — All other plots compressed onto one figure. The big-picture overview at a glance.

- **NED profile** — *Normalized Echo Density* over time. The curve rises from 0 (sparse, distinct echoes) toward 1 (fully diffuse, Gaussian-statistical reverb tail). The marker shows where the pipeline switched from preserving early reflections to synthesizing diffuse reverb. A curve that *stays low* indicates a "lively" room with lots of distinct reflections; a curve that *quickly hits 1* indicates a heavily diffuse room.

- **Waveform comparison** — The original balloon recording (top) vs the synthesized IR (bottom). The early portion looks similar by design (it's preserved). The tail looks different in detail but matches in envelope, and decays smoothly to silence with no comb-filter ringing.

- **Spectrogram comparison** — Frequency content over time, side by side. The original recording shows **horizontal dark notches** — those are the N-wave's spectral nulls, the artifact this tool exists to remove. The synthesized IR has a smooth, broadband decay across all frequencies, with no nulls.

- **Band energy decay** — Per-frequency-band energy curves over time, shown for both the recording and the IR. Each curve is one 1/3-octave band; together they describe how the room loses energy at each frequency (closely related to RT60). The IR's curves should track the original's, just smoother and extended below the noise floor. **This is the plot to inspect if the synthesized reverb sounds tonally wrong.**

- **Echo sequence** — The Poisson-distributed pulse train used as the carrier signal for the diffuse tail. Pulse density grows with time, matching the natural transition from sparse early reflections to dense reverberation.

- **ICCC profile** *(stereo only)* — *Inter-Channel Cross-Correlation* over time. **1.0** = both channels are identical (mono-like); **0** = fully de-correlated (wide stereo). The curve typically starts high (the direct sound arrives similarly at both ears) and decreases as the diffuse tail builds (different reflections reaching each ear). The synthesized stereo IR mimics this profile.

## Privacy

Everything runs in your browser via [Pyodide](https://pyodide.org/) (CPython compiled to WebAssembly). **Your audio never leaves your machine** — no uploads, no servers, no analytics, no cookies.

---

## For developers

### Project structure

```
core/                Python DSP pipeline (used by both the web app and the CLI)
  preprocessing.py     Stage 0
  echo_density.py      Stage 1: NED, AED, Poisson pulse synthesis
  spatial.py           Stage 2: ICCC analysis, stereo rotation
  energy_shaping.py    Stage 3: filter bank, energy imprinting
  postprocessing.py    Stage 4: normalize, fade, trim, export
  pipeline.py          Orchestrates all stages
  visualization.py     Diagnostic plot generation
frontend/            Static in-browser app (no build step)
  index.html
  src/main.js          UI: upload, params, progress, audio playback
  src/wav.js           WAV decode/encode (PCM 16/24/32, float)
  src/pipeline.js      Loads Pyodide + numpy + scipy + matplotlib + core/*.py
  src/bridge.py        Adapts core/ to take Float32 arrays from JS
scripts/             CLI tools
  cli.py               Command-line processing
  generate_plots.py    Batch plot generation
  validate_ir.py       Room acoustic params (RT60, EDT, C80, D50, Ts)
tests/               Unit tests (76 tests)
```

### CLI

```bash
conda activate dsp
python scripts/cli.py balloon.wav -o ir.wav
```

Common options (`python scripts/cli.py --help` for the full list):

```
--onset-threshold    Onset detection threshold in dB (default: -40)
--ned-window         NED estimation window in ms (default: 43)
--ned-transition     NED threshold for sparse/dense transition (default: 0.3)
--noise-floor        Noise floor threshold in dB (default: -40)
--pulse-halo         Gain halo around early pulses in ms (default: 2.0)
--target-dbfs        Output normalization level (default: -1.0)
--bit-depth          Output bit depth: 16, 24, or 32 (default: 24)
--target-sr          Resample to target sample rate (default: keep original)
--seed               Random seed for reproducibility
```

### Running the web frontend locally

```bash
python -m http.server 8000
# open http://localhost:8000/frontend/
```

See [`frontend/README.md`](frontend/README.md) for details.

### Deployment

GitHub Pages — auto-deploys on push to `main` via [`.github/workflows/deploy-frontend.yml`](.github/workflows/deploy-frontend.yml). The site is purely static (no build step) and bundles the Python `core/` modules so Pyodide can load them client-side.

### Requirements

- Python 3.11+ (for the CLI and tests; the browser frontend ships its own runtime via Pyodide)
- numpy, scipy, soundfile, matplotlib

## Reference

Abel, J. S., Canfield-Dafilou, E. K., & Holloway, M. (2010). [Estimating room impulse responses from recorded balloon pops](https://www.researchgate.net/profile/Bissera-Pentcheva/publication/277732009_Estimating_Room_Impulse_Responses_from_Recorded_Balloon_Pops/links/563a4b4408ae45b5d284a8ce/Estimating-Room-Impulse-Responses-from-Recorded-Balloon-Pops.pdf). *Audio Engineering Society Convention 129*, Paper 8171.

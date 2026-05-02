# Balloon IR Synthesizer — standalone (no API server)

This is a 100% client-side build of the balloon-pop → room IR pipeline.
The same Python code under [`core/`](../core) runs in the browser via
[Pyodide](https://pyodide.org/) (CPython compiled to WebAssembly).

There is no FastAPI server, no job queue, no upload — the WAV file
never leaves your machine.

## How to run locally

Pyodide and the `core/` modules are loaded over HTTP, so you need a
**static file server** rooted at the project directory (the parent of
this folder). For example:

```bash
cd "/path/to/balloon"
python -m http.server 8000
# then open http://localhost:8000/frontend-standalone/
```

First visit downloads ~30 MB (Pyodide runtime + numpy + scipy wheels)
and caches it. Subsequent visits are instant.

## How it works

```
WAV file (browser)
   │
   │  decodeWav()  ──  Float32Array channels, sample rate
   ▼
Pyodide  ◄── core/preprocessing.py, echo_density.py, … (mounted in virtual FS)
   │      bridge.process_array(left, right, sr, **params)
   ▼
Float32Array IR  ──  encodeWav()  ──  download as 24-bit WAV
```

The bridge ([`src/bridge.py`](src/bridge.py)) mirrors
`core.pipeline.process_balloon` but takes raw arrays instead of a file
path, so the WAV decode/encode lives in JavaScript ([`wav.js`](src/wav.js))
rather than round-tripping through Pyodide's virtual filesystem.

## Files

| Path             | What it does                                           |
| ---------------- | ------------------------------------------------------ |
| `index.html`     | UI                                                     |
| `src/main.js`    | Glue: file picker, params, progress, audio playback    |
| `src/wav.js`     | WAV decode / encode (PCM 16/24/32, float 32/64)        |
| `src/pipeline.js`| Loads Pyodide + numpy + scipy + `core/*.py`            |
| `src/bridge.py`  | Python entry point that takes Float32Arrays            |

## Limitations vs. the React + FastAPI version

- No server-side caching of jobs (you re-run from scratch each time).
- No matplotlib previews. Plots could be added later (matplotlib runs
  in Pyodide), but the analysis-only previews aren't wired up here.
- First-run download is large (~30 MB).

## Deploying as a fully static site

Copy `core/` and `frontend-standalone/` to any static host (GitHub
Pages, Netlify, Cloudflare Pages, S3, …) so that `core/*.py` are
reachable at `../core/<name>.py` relative to `index.html`. No build
step required.

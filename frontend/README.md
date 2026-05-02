# Balloon IR Synthesizer — frontend

100% client-side build of the balloon-pop → room IR pipeline. The same
Python code under [`core/`](../core) runs in the browser via
[Pyodide](https://pyodide.org/) (CPython compiled to WebAssembly).

There is no API server, no job queue, no upload — the WAV file never
leaves your machine.

## How to run locally

Pyodide and the `core/` modules are loaded over HTTP, so you need a
**static file server** rooted at the project directory (the parent of
this folder). For example:

```bash
cd "/path/to/balloon"
python -m http.server 8000
# then open http://localhost:8000/frontend/
```

First visit downloads ~40 MB (Pyodide runtime + numpy + scipy + soundfile +
matplotlib wheels) and caches it. Subsequent visits are instant.

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
| `src/pipeline.js`| Loads Pyodide + numpy + scipy + matplotlib + `core/*.py` |
| `src/bridge.py`  | Python entry point that takes Float32Arrays            |

## Limitations

- First-run download is large (~40 MB) for the Pyodide runtime + wheels.
- Plot rendering reuses `core/visualization.py` and runs in the browser
  via matplotlib (Agg backend). Toggle off "Generate diagnostic plots"
  to skip — audio synthesis itself takes only a few seconds.

## Deploying as a fully static site

The site needs `core/*.py` reachable at `../core/<name>.py` relative
to `index.html`. The included `.github/workflows/deploy-frontend.yml`
stages this layout and publishes to GitHub Pages on every push to
`main`. To deploy elsewhere (Netlify, Cloudflare Pages, S3, …), copy
both `frontend/` and `core/` preserving the parent layout. No build
step required.

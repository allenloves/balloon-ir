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

## Upgrading Pyodide

The Pyodide version is pinned in
[`src/pipeline.js`](src/pipeline.js) as
`PYODIDE_VERSION`. Bumping it also swaps in the numpy / scipy /
matplotlib / soundfile wheels that Pyodide bundles for that release,
so the upgrade can shift behavior in subtle ways. Validation
checklist before you commit a version bump:

1. Edit `PYODIDE_VERSION` in `src/pipeline.js`.
2. Hard-reload the page (`Cmd+Shift+R`) so the new runtime + wheels
   are fetched. Watch the network tab to confirm everything is 200.
3. **Mono path**: load a mono WAV → Synthesize → **Play** should
   produce sound → **Download** should give a non-silent file.
4. **Stereo path**: same with a stereo WAV. Confirm the ICCC plot
   appears in addition to the other six.
5. **All seven plots render** — `summary`, `ned_profile`,
   `waveform_comparison`, `spectrogram_comparison`, `band_energy`,
   `echo_sequence`, `iccc_profile` (stereo only). The spectrogram
   especially exercises the `_patch_specgram` workaround in
   `src/bridge.py`; if matplotlib's internals change, that patch may
   need to be re-aligned.
6. Audio remains audible **after** plot rendering completes — this
   exercises the WASM-heap-detach defense in `bytesToF32`
   (`src/pipeline.js`). If Play goes silent only after plots have
   rendered, suspect the bytes-to-Float32Array path.

Skim the [Pyodide changelog](https://pyodide.org/en/stable/project/changelog.html)
for the version range you're crossing — the *Type conversions* and
*FFI* sections are the ones most likely to affect this app.

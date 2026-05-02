// Pyodide loader + bridge invocation for the balloon-IR pipeline.
//
// Loads core/*.py from the project root (served as static files) and
// bridge.py from this directory, then exposes runPipeline().

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.js";

const CORE_FILES = [
  "preprocessing.py",
  "filterbank.py",
  "echo_density.py",
  "spatial.py",
  "energy_shaping.py",
  "postprocessing.py",
  "pipeline.py",
  "visualization.py",
];

let _pyodidePromise = null;

export function loadPipeline(onLog = () => {}) {
  if (!_pyodidePromise) _pyodidePromise = init(onLog);
  return _pyodidePromise;
}

async function init(onLog) {
  onLog("Loading Pyodide runtime…");
  if (!window.loadPyodide) {
    await loadScript(PYODIDE_URL);
  }
  const pyodide = await window.loadPyodide();

  onLog("Loading numpy + scipy + soundfile + matplotlib (heavy: ~40 MB first time)…");
  await pyodide.loadPackage(["numpy", "scipy", "soundfile", "matplotlib"]);

  onLog("Mounting Python modules…");
  pyodide.FS.mkdirTree("core");
  pyodide.FS.writeFile("core/__init__.py", "");
  await Promise.all(
    CORE_FILES.map(async (name) => {
      const text = await fetchText(`../core/${name}`);
      pyodide.FS.writeFile(`core/${name}`, text);
    }),
  );
  const bridgeSrc = await fetchText("./src/bridge.py");
  pyodide.FS.writeFile("bridge.py", bridgeSrc);

  await pyodide.runPythonAsync("import bridge");
  onLog("Ready.");
  return pyodide;
}

// channels: array of Float32Array (1 = mono, 2 = stereo).
// params: plain object passed as kwargs to bridge.process_array.
// onProgress: (pct, msg) callback during the run.
export async function runPipeline(pyodide, channels, sampleRate, params, onProgress) {
  const left = channels[0];
  const right = channels.length > 1 ? channels[1] : null;

  pyodide.globals.set("_js_left_bytes", asU8(left));
  pyodide.globals.set("_js_right_bytes", right ? asU8(right) : null);
  pyodide.globals.set("_js_sr", sampleRate);
  pyodide.globals.set("_js_params", pyodide.toPy(params));
  pyodide.globals.set("_js_progress", onProgress || null);

  const result = await pyodide.runPythonAsync(`
import numpy as np
import bridge

def _decode(buf):
    if buf is None:
        return None
    # buf is a JsProxy for a Uint8Array; bytes() forces a real Python copy.
    return np.frombuffer(bytes(buf.to_py()), dtype=np.float32).astype(np.float64)

_left  = _decode(_js_left_bytes)
_right = _decode(_js_right_bytes)

_kwargs = dict(_js_params)
if _js_progress is not None:
    _kwargs['progress'] = _js_progress

bridge.process_array(_left, _right, int(_js_sr), **_kwargs)
`);

  // Convert PyProxy dict → JS object. The 'left'/'right' values are bytes
  // on the Python side, which Pyodide hands over as Uint8Array — reinterpret
  // those buffers as Float32Array so the rest of the app sees regular samples.
  const obj = result.toJs({ dict_converter: Object.fromEntries });
  result.destroy();

  // plot_names arrives as a JS Array via toJs.
  const plotNames = obj.plot_names ? Array.from(obj.plot_names) : [];

  return {
    sr: obj.sr,
    channels: obj.channels,
    left: bytesToF32(obj.left),
    right: obj.right ? bytesToF32(obj.right) : null,
    plotNames,
  };
}

// Render a single plot from the most recent process_array() run.
// Returns a base64-encoded PNG string. Each call yields control back to
// the event loop, so audio playback / UI updates stay responsive between
// plots.
export async function renderPlot(pyodide, name, dpi = 100) {
  pyodide.globals.set("_js_plot_name", name);
  pyodide.globals.set("_js_plot_dpi", dpi);
  const b64 = await pyodide.runPythonAsync(
    "bridge.render_plot(str(_js_plot_name), dpi=int(_js_plot_dpi))",
  );
  return b64;
}

function asU8(f32) {
  return new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
}

function bytesToF32(u8) {
  // Always copy into a fresh ArrayBuffer. The source Uint8Array can be a
  // view onto Pyodide's WASM heap, which gets detached whenever the heap
  // grows later (e.g. during matplotlib plot rendering). A view onto a
  // detached buffer reads as zero — silent audio. Copying here makes the
  // Float32Array independent of anything Pyodide does afterwards.
  const copy = new ArrayBuffer(u8.byteLength);
  new Uint8Array(copy).set(u8);
  return new Float32Array(copy);
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.text();
}

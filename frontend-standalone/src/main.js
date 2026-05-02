import { decodeWav, encodeWav } from "./wav.js";
import { loadPipeline, runPipeline } from "./pipeline.js";

const $ = (sel) => document.querySelector(sel);

const fileInput = $("#file");
const fileLabel = $("#file-label");
const runBtn = $("#run");
const statusEl = $("#status");
const progressEl = $("#progress");
const progressBar = $("#progress-bar");
const resultEl = $("#result");
const playBtn = $("#play");
const stopBtn = $("#stop");
const downloadBtn = $("#download");
const summaryEl = $("#summary");
const paramsForm = $("#params");

let pyodide = null;
let inputAudio = null; // { channels: [Float32Array], sampleRate }
let resultAudio = null; // { channels: [Float32Array], sampleRate }
let audioCtx = null;
let activeSource = null;

function log(msg) {
  statusEl.textContent = msg;
}

function setProgress(pct, msg) {
  progressEl.hidden = false;
  progressBar.style.width = `${pct}%`;
  if (msg) log(msg);
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileLabel.textContent = file.name;
  runBtn.disabled = true;
  log(`Decoding ${file.name}…`);
  try {
    const buf = await file.arrayBuffer();
    inputAudio = decodeWav(buf);
    const ch = inputAudio.channels.length;
    const sec = (inputAudio.channels[0].length / inputAudio.sampleRate).toFixed(2);
    log(
      `Loaded: ${ch === 1 ? "mono" : "stereo"}, ${inputAudio.sampleRate} Hz, ` +
      `${inputAudio.bitsPerSample}-bit, ${sec}s`,
    );
    runBtn.disabled = false;
  } catch (e) {
    log(`WAV decode failed: ${e.message}`);
    inputAudio = null;
  }
});

runBtn.addEventListener("click", async () => {
  if (!inputAudio) return;
  runBtn.disabled = true;
  resultEl.hidden = true;
  setProgress(0, "Loading Python runtime (first run takes a moment)…");

  try {
    if (!pyodide) {
      pyodide = await loadPipeline((m) => log(m));
    }

    const params = readParams();
    setProgress(2, "Starting pipeline…");

    resultAudio = await runPipeline(
      pyodide,
      inputAudio.channels,
      inputAudio.sampleRate,
      params,
      (pct, msg) => setProgress(pct, msg),
    );

    setProgress(100, "Done.");
    showResult();
  } catch (e) {
    console.error(e);
    log(`Error: ${e.message}`);
  } finally {
    runBtn.disabled = false;
  }
});

function readParams() {
  const data = new FormData(paramsForm);
  const num = (k) => {
    const v = data.get(k);
    return v === "" || v == null ? null : Number(v);
  };
  return {
    onset_threshold_db: num("onset_threshold_db"),
    ned_window_ms: num("ned_window_ms"),
    ned_transition_threshold: num("ned_transition_threshold"),
    pulse_halo_ms: num("pulse_halo_ms"),
    noise_floor_db: num("noise_floor_db"),
    gain_smoothing_ms: num("gain_smoothing_ms"),
    target_dbfs: num("target_dbfs"),
  };
}

function showResult() {
  const left = resultAudio.left;
  const right = resultAudio.right;
  const channels = right ? [left, right] : [left];
  const sr = resultAudio.sr;
  const sec = (left.length / sr).toFixed(2);
  summaryEl.textContent =
    `${channels.length === 1 ? "Mono" : "Stereo"} IR · ${sr} Hz · ${sec}s · ${left.length} samples`;
  resultEl.hidden = false;
  resultAudio._channels = channels;
}

playBtn.addEventListener("click", async () => {
  if (!resultAudio) return;
  stopPlayback();
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const channels = resultAudio._channels;
  const buf = audioCtx.createBuffer(channels.length, channels[0].length, resultAudio.sr);
  for (let c = 0; c < channels.length; c++) buf.copyToChannel(channels[c], c);
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  src.onended = () => {
    if (activeSource === src) activeSource = null;
  };
  src.start();
  activeSource = src;
});

stopBtn.addEventListener("click", stopPlayback);
function stopPlayback() {
  if (activeSource) {
    try { activeSource.stop(); } catch {}
    activeSource = null;
  }
}

downloadBtn.addEventListener("click", () => {
  if (!resultAudio) return;
  const blob = encodeWav(resultAudio._channels, resultAudio.sr, 24);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "balloon_ir.wav";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

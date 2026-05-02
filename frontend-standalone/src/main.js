import { decodeWav, encodeWav } from "./wav.js";
import { loadPipeline, runPipeline, renderPlot } from "./pipeline.js";

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
const plotsEl = $("#plots");
const plotsToggle = $("#generate-plots");

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

    // Audio is ready — show it now so the user can play / download
    // immediately, then stream the plots in afterwards.
    showResult();

    if (plotsToggle.checked && resultAudio.plotNames?.length) {
      await streamPlots(resultAudio.plotNames);
    } else {
      log("Done.");
    }
  } catch (e) {
    console.error(e);
    log(`Error: ${e.message}`);
  } finally {
    runBtn.disabled = false;
  }
});

async function streamPlots(names) {
  // Render plots one by one, awaiting between each so the audio playback
  // and other UI events can run on the main thread between renders.
  plotsEl.innerHTML = "";
  // Pre-create skeleton tiles so the user sees what's coming.
  const tiles = {};
  for (const key of PLOT_ORDER) {
    if (!names.includes(key)) continue;
    const fig = document.createElement("figure");
    fig.className = "plot pending";
    const img = document.createElement("img");
    img.alt = PLOT_LABELS[key] || key;
    const cap = document.createElement("figcaption");
    cap.textContent = `${PLOT_LABELS[key] || key} — pending…`;
    fig.appendChild(img);
    fig.appendChild(cap);
    plotsEl.appendChild(fig);
    tiles[key] = { fig, img, cap };
  }

  const total = Object.keys(tiles).length;
  let i = 0;
  for (const key of PLOT_ORDER) {
    if (!tiles[key]) continue;
    i += 1;
    log(`Rendering plot ${i}/${total}: ${PLOT_LABELS[key] || key}…`);
    // Yield to the event loop before each render so any pending audio /
    // playback events get a chance to run.
    await new Promise((r) => requestAnimationFrame(r));
    try {
      const b64 = await renderPlot(pyodide, key);
      tiles[key].img.src = `data:image/png;base64,${b64}`;
      tiles[key].cap.textContent = PLOT_LABELS[key] || key;
      tiles[key].fig.classList.remove("pending");
    } catch (e) {
      console.error(`Plot ${key} failed:`, e);
      tiles[key].cap.textContent = `${PLOT_LABELS[key] || key} — failed`;
    }
  }
  log("All plots rendered.");
}

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

const PLOT_LABELS = {
  summary: "Summary",
  ned_profile: "NED profile",
  waveform_comparison: "Waveform comparison",
  spectrogram_comparison: "Spectrogram comparison",
  band_energy: "Band energy decay",
  echo_sequence: "Echo sequence",
  iccc_profile: "ICCC profile (stereo)",
};
const PLOT_ORDER = [
  "summary", "ned_profile", "waveform_comparison",
  "spectrogram_comparison", "band_energy", "echo_sequence", "iccc_profile",
];

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
  plotsEl.innerHTML = "";
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

"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  ws: null,
  reconnectDelay: 1000,
  frameCount: 0,
  lastFpsTime: Date.now(),
  fps: 0,
  freqs: [],
  useWebFFT: true,
  webfft: null,
  webfftSize: 0,
};

// ---------------------------------------------------------------------------
// Plasma colormap (256 RGB entries, matches server-side palette)
// ---------------------------------------------------------------------------
const PLASMA_LUT = (() => {
  // Compact representation — same values as Python PLASMA array in sdr_worker.py
  // Re-use server's waterfall_row directly for waterfall canvas;
  // this LUT is used only for the spectrum gradient legend (not strictly needed).
  return null; // Not used in JS — we receive pre-mapped rows from server
})();

// ---------------------------------------------------------------------------
// Canvas: Waterfall
// ---------------------------------------------------------------------------
const waterfallCanvas = document.getElementById("waterfallCanvas");
const waterfallCtx    = waterfallCanvas.getContext("2d");
let   waterfallImg    = null;

function waterfallInit() {
  waterfallCanvas.width  = waterfallCanvas.offsetWidth  || 800;
  waterfallCanvas.height = waterfallCanvas.offsetHeight || 200;
  waterfallImg = waterfallCtx.createImageData(waterfallCanvas.width, 1);
}

function waterfallDrawRow(colorRow) {
  const W = waterfallCanvas.width;
  const H = waterfallCanvas.height;
  // Shift existing pixels down by 1 row
  const existing = waterfallCtx.getImageData(0, 0, W, H - 1);
  waterfallCtx.putImageData(existing, 0, 1);

  // Draw new row at top by resampling colorRow to canvas width
  const imgData = waterfallCtx.createImageData(W, 1);
  const n = colorRow.length;
  for (let x = 0; x < W; x++) {
    const srcIdx = Math.floor((x / W) * n);
    const [r, g, b] = colorRow[Math.min(srcIdx, n - 1)];
    const p = x * 4;
    imgData.data[p]     = r;
    imgData.data[p + 1] = g;
    imgData.data[p + 2] = b;
    imgData.data[p + 3] = 255;
  }
  waterfallCtx.putImageData(imgData, 0, 0);
}

// ---------------------------------------------------------------------------
// Canvas: IQ Constellation
// ---------------------------------------------------------------------------
const constellCanvas = document.getElementById("constellationCanvas");
const constellCtx    = constellCanvas.getContext("2d");

function constellInit() {
  const size = Math.min(constellCanvas.parentElement.offsetWidth, 300);
  constellCanvas.width  = size;
  constellCanvas.height = size;
}

function constellDraw(iqInterleaved) {
  const W = constellCanvas.width;
  const H = constellCanvas.height;
  constellCtx.fillStyle = "#0d1117";
  constellCtx.fillRect(0, 0, W, H);

  // Axes
  constellCtx.strokeStyle = "#30363d";
  constellCtx.lineWidth = 1;
  constellCtx.beginPath();
  constellCtx.moveTo(W / 2, 0); constellCtx.lineTo(W / 2, H);
  constellCtx.moveTo(0, H / 2); constellCtx.lineTo(W, H / 2);
  constellCtx.stroke();

  const cx = W / 2, cy = H / 2;
  const scale = W / 2.4;

  constellCtx.fillStyle = "rgba(56, 139, 253, 0.6)";
  for (let i = 0; i < iqInterleaved.length - 1; i += 2) {
    const px = cx + iqInterleaved[i]     * scale;
    const py = cy - iqInterleaved[i + 1] * scale;
    constellCtx.fillRect(px - 1, py - 1, 2, 2);
  }
}

// ---------------------------------------------------------------------------
// Canvas 2D: Spectrum (replaces Chart.js to avoid axis drift / fill artefacts)
// ---------------------------------------------------------------------------
const spectrumCanvas = document.getElementById("spectrumCanvas");
const spectrumCtx    = spectrumCanvas.getContext("2d");

const DB_MIN = -80;
const DB_MAX = -10;

function spectrumInit() {
  spectrumCanvas.width  = spectrumCanvas.offsetWidth  || 1200;
  spectrumCanvas.height = spectrumCanvas.offsetHeight || 180;
}

function spectrumUpdate(freqs, powerDb, centerFreq) {
  const W = spectrumCanvas.width;
  const H = spectrumCanvas.height;
  const PAD_L = 42, PAD_B = 20, PAD_T = 8;
  const plotW = W - PAD_L;
  const plotH = H - PAD_B - PAD_T;

  spectrumCtx.fillStyle = "#0d1117";
  spectrumCtx.fillRect(0, 0, W, H);

  const n = powerDb.length;
  if (n < 2) return;

  // Grid lines & Y labels
  spectrumCtx.strokeStyle = "#21262d";
  spectrumCtx.lineWidth   = 1;
  spectrumCtx.fillStyle   = "#8b949e";
  spectrumCtx.font        = "11px monospace";
  spectrumCtx.textAlign   = "right";
  const ySteps = [-80, -70, -60, -50, -40, -30, -20, -10];
  for (const db of ySteps) {
    if (db < DB_MIN || db > DB_MAX) continue;
    const y = PAD_T + plotH * (1 - (db - DB_MIN) / (DB_MAX - DB_MIN));
    spectrumCtx.beginPath();
    spectrumCtx.moveTo(PAD_L, y);
    spectrumCtx.lineTo(W, y);
    spectrumCtx.stroke();
    spectrumCtx.fillText(db + " dB", PAD_L - 4, y + 4);
  }

  // X tick labels (frequency)
  spectrumCtx.textAlign = "center";
  spectrumCtx.fillStyle = "#8b949e";
  const xTicks = 8;
  for (let t = 0; t <= xTicks; t++) {
    const idx = Math.floor((t / xTicks) * (n - 1));
    const x   = PAD_L + (idx / (n - 1)) * plotW;
    const hz  = freqs[idx];
    const absHz = (centerFreq || 0) + hz;
    spectrumCtx.fillText((absHz / 1e6).toFixed(2) + "M", x, H - 4);
  }

  // Spectrum line + fill
  spectrumCtx.beginPath();
  for (let i = 0; i < n; i++) {
    const x  = PAD_L + (i / (n - 1)) * plotW;
    const db = Math.max(DB_MIN, Math.min(DB_MAX, powerDb[i]));
    const y  = PAD_T + plotH * (1 - (db - DB_MIN) / (DB_MAX - DB_MIN));
    if (i === 0) spectrumCtx.moveTo(x, y);
    else         spectrumCtx.lineTo(x, y);
  }
  // Fill down to bottom of plot
  spectrumCtx.lineTo(PAD_L + plotW, PAD_T + plotH);
  spectrumCtx.lineTo(PAD_L,         PAD_T + plotH);
  spectrumCtx.closePath();
  spectrumCtx.fillStyle   = "rgba(56,139,253,0.12)";
  spectrumCtx.fill();
  spectrumCtx.strokeStyle = "#388bfd";
  spectrumCtx.lineWidth   = 1.5;
  spectrumCtx.stroke();
}

// ---------------------------------------------------------------------------
// WebFFT initialisation
// ---------------------------------------------------------------------------
async function initWebFFT(size) {
  if (!window.WebFFT) return;
  if (state.webfft && state.webfftSize === size) return;
  try {
    if (state.webfft) state.webfft.dispose();
    state.webfft = new window.WebFFT(size);
    await state.webfft.profile();
    state.webfftSize = size;
    console.log("WebFFT ready, size =", size);
  } catch (e) {
    console.warn("WebFFT init failed:", e);
    state.webfft = null;
  }
}

function webfftCompute(iqInterleaved) {
  if (!state.webfft) return null;
  try {
    const input = new Float32Array(iqInterleaved);
    return state.webfft.fft(input);  // returns Float32Array interleaved real/imag
  } catch (e) {
    return null;
  }
}

function fftOutputToDb(fftOut, n, normalize) {
  const scale = normalize || 1;
  const powerDb = new Array(n);
  for (let i = 0; i < n; i++) {
    const re = fftOut[2 * i];
    const im = fftOut[2 * i + 1];
    // Normalize by FFT size (same as server) so dB range is consistent
    powerDb[i] = 20 * Math.log10(Math.sqrt(re * re + im * im) / scale + 1e-10);
  }
  // fftshift: move DC from bin 0 to center
  const half = n >> 1;
  return [...powerDb.slice(half), ...powerDb.slice(0, half)];
}

function makeFreqs(n, sampleRate) {
  const df = sampleRate / n;
  const freqs = [];
  const half = n >> 1;
  for (let i = -half; i < n - half; i++) freqs.push(i * df);
  return freqs;
}

// ---------------------------------------------------------------------------
// Frame handler
// ---------------------------------------------------------------------------
async function handleFrame(frame) {
  const useWebFFT = state.useWebFFT && window.WebFFT;
  const iqRaw = frame.iq_raw;

  let freqs    = frame.freqs;
  let spectrum = frame.spectrum;

  if (useWebFFT && iqRaw && iqRaw.length >= 2) {
    // iq_raw is interleaved float32: length = 2 * n_iq_pairs
    // WebFFT constructor takes FFT size = number of IQ pairs
    const n = iqRaw.length >> 1;   // number of IQ pairs
    await initWebFFT(n);           // WebFFT(fftSize) — NOT n*2

    const fftOut = webfftCompute(iqRaw);
    if (fftOut) {
      spectrum = fftOutputToDb(fftOut, n, n);
      freqs    = makeFreqs(n, frame.sample_rate);
    }
  }

  spectrumUpdate(freqs, spectrum, frame.center_freq);
  waterfallDrawRow(frame.waterfall_row);
  constellDraw(iqRaw || []);

  // FPS counter
  state.frameCount++;
  const now = Date.now();
  if (now - state.lastFpsTime >= 1000) {
    state.fps = state.frameCount;
    state.frameCount = 0;
    state.lastFpsTime = now;
    const mode = useWebFFT ? " | WebFFT ✓" : " | Server FFT";
    document.getElementById("statusBar").textContent =
      `${state.fps} fps | ${(frame.center_freq / 1e6).toFixed(3)} MHz | ${(frame.sample_rate / 1e6).toFixed(3)} Msps${mode}`;
  }
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
function connect() {
  const wsUrl = `ws://${location.host}/ws`;
  const ws = new WebSocket(wsUrl);
  state.ws = ws;

  ws.onopen = () => {
    console.log("WebSocket connected");
    document.getElementById("statusBadge").textContent  = "Connected";
    document.getElementById("statusBadge").className    = "badge badge-connected px-2 py-1";
    state.reconnectDelay = 1000;
  };

  ws.onmessage = (evt) => {
    let frame;
    try { frame = JSON.parse(evt.data); } catch { return; }
    handleFrame(frame);
  };

  ws.onclose = () => {
    console.log("WebSocket disconnected, retrying in", state.reconnectDelay, "ms");
    document.getElementById("statusBadge").textContent = "Disconnected";
    document.getElementById("statusBadge").className   = "badge badge-disconnected px-2 py-1";
    setTimeout(connect, state.reconnectDelay);
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, 10000);
  };

  ws.onerror = (e) => { console.error("WebSocket error:", e); ws.close(); };
}

// ---------------------------------------------------------------------------
// Config controls
// ---------------------------------------------------------------------------
function applyConfig() {
  const cfg = {
    center_freq: parseFloat(document.getElementById("cfgFreq").value) * 1e6,
    sample_rate: parseFloat(document.getElementById("cfgRate").value) * 1e6,
    lna_gain:    parseInt(document.getElementById("cfgLna").value, 10),
    vga_gain:    parseInt(document.getElementById("cfgVga").value, 10),
    fft_size:    parseInt(document.getElementById("cfgFft").value, 10),
  };
  fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  }).catch(console.error);
}

document.getElementById("btnApply").addEventListener("click", applyConfig);
document.getElementById("toggleWebFFT").addEventListener("change", (e) => {
  state.useWebFFT = e.target.checked;
});

// Populate controls from server config on load
fetch("/api/config").then(r => r.json()).then(cfg => {
  if (cfg.center_freq) document.getElementById("cfgFreq").value = (cfg.center_freq / 1e6).toFixed(3);
  if (cfg.sample_rate) document.getElementById("cfgRate").value = (cfg.sample_rate / 1e6).toFixed(3);
  if (cfg.lna_gain !== undefined) document.getElementById("cfgLna").value = cfg.lna_gain;
  if (cfg.vga_gain !== undefined) document.getElementById("cfgVga").value = cfg.vga_gain;
  if (cfg.fft_size)  document.getElementById("cfgFft").value  = cfg.fft_size;
}).catch(() => {});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function initAll() {
  waterfallInit();
  constellInit();
  spectrumInit();
}

window.addEventListener("load", () => {
  initAll();
  connect();
});

window.addEventListener("resize", () => {
  initAll();
});

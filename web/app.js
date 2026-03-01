/* ================================================================
   app.js – Dual-mode MNIST (digits) / EMNIST (characters) inference
   ================================================================ */

// --------------- visualization constants -------------------------
const MAX_DISPLAY_NODES = 12; // max visible nodes per layer; rest shown as "..."
const NODE_GAP = 20;          // px between node centers
const BASE_NODE_RADIUS = 6;

// --------------- mode definitions --------------------------------
const MODES = {
  digits: {
    key: "digits",
    defaultModelUrl: "./model/model.onnx",
    classCount: 10,
    label: (i) => String(i),                       // "0".."9"
    outputHeading: "Output Layer (0-9)",
    idleStatusMsg: "Model auto-loaded. Draw a digit to predict.",
    loadingMsg: "Auto-loading digit model...",
    transpose: false,
  },
  characters: {
    key: "characters",
    defaultModelUrl: "./model/emnist_model.onnx",
    classCount: 26,
    label: (i) => String.fromCharCode(65 + i),     // "A".."Z"
    outputHeading: "Output Layer (A-Z)",
    idleStatusMsg: "Model loaded. Draw a letter to predict.",
    loadingMsg: "Auto-loading character model...",
    transpose: true,  // EMNIST images are transposed
  },
};

// --------------- per-mode runtime state --------------------------
function createModeState(modeKey) {
  const m = MODES[modeKey];
  const suffix = modeKey;

  const drawCanvas = document.getElementById(`draw-canvas-${suffix}`);
  const gridCanvas = document.getElementById(`grid-canvas-${suffix}`);
  const networkCanvas = document.getElementById(`network-canvas-${suffix}`);

  // Add CSS classes so styles work (class-based selectors)
  drawCanvas.classList.add("draw-canvas");
  gridCanvas.classList.add("grid-canvas");
  networkCanvas.classList.add("network-canvas");

  // offscreen canvas for 28x28 downscale
  const offscreen = document.createElement("canvas");
  offscreen.width = 28;
  offscreen.height = 28;
  const offscreenCtx = offscreen.getContext("2d", { willReadFrequently: true });
  offscreenCtx.imageSmoothingEnabled = true;
  offscreenCtx.imageSmoothingQuality = "medium";

  return {
    config: m,

    // DOM refs
    drawCanvas,
    gridCanvas,
    networkCanvas,
    clearBtn: document.getElementById(`clear-btn-${suffix}`),
    loadBtn: document.getElementById(`load-btn-${suffix}`),
    downloadBtn: document.getElementById(`download-btn-${suffix}`),
    modelUrlInput: document.getElementById(`model-url-${suffix}`),
    statusEl: document.getElementById(`status-${suffix}`),
    predictionEl: document.getElementById(`prediction-${suffix}`),
    outputGrid: document.getElementById(`output-grid-${suffix}`),
    gridDimsLabel: document.getElementById(`grid-dims-label-${suffix}`),

    // canvas contexts
    drawCtx: drawCanvas.getContext("2d", { willReadFrequently: true }),
    gridCtx: gridCanvas.getContext("2d"),
    networkCtx: networkCanvas.getContext("2d"),

    // offscreen for 28x28 downscale (cached)
    offscreenCanvas: offscreen,
    offscreenCtx,

    // MNIST-style normalization scratch canvas
    mnistPrepCanvas: (() => { const c = document.createElement("canvas"); c.width = 28; c.height = 28; return c; })(),

    // ONNX state
    onnxSession: null,
    onnxInputName: null,
    onnxOutputName: null,
    onnxInputDims: null,
    onnxOutputDims: null,
    loadedModelUrl: null,

    // drawing
    drawing: false,
    predictionFrame: 0,

    // output items (populated in setupOutputGrid)
    outputItems: [],

    // network visualization
    layerSpecs: [
      { name: "Input", count: 784 },
      { name: "Hidden", count: 128 },
      { name: "Output", count: m.classCount },
    ],
  };
}

const INPUT_GRID_ROWS = 28;
const INPUT_GRID_COLS = 28;

// --------------- EMNIST transpose utility ------------------------
function transposeGrid(arr, rows, cols) {
  const result = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      result[c * rows + r] = arr[r * cols + c];
    }
  }
  return result;
}

// Create state objects
const modes = {
  digits: createModeState("digits"),
  characters: createModeState("characters"),
};

let currentMode = modes.digits;

// --------------- initialise both modes ---------------------------
Object.values(modes).forEach((mode) => {
  setupOutputGrid(mode);
  resetDrawCanvas(mode);
  drawInputGrid(mode);
  setIdlePredictionState(mode);
  updateGridDimsLabel(mode);
  wireDrawingEvents(mode);
  wireModeButtons(mode);
});

// auto-load digits model on startup
autoLoadModel(modes.digits);

// --------------- tab switching -----------------------------------
const tabBtns = document.querySelectorAll(".tab-btn");
const tabPanes = document.querySelectorAll(".tab-pane");
const tabPill = document.querySelector(".tab-pill");

function positionPill(activeBtn) {
  if (!tabPill || !activeBtn) return;
  const bar = activeBtn.parentElement;
  const barRect = bar.getBoundingClientRect();
  const btnRect = activeBtn.getBoundingClientRect();
  const offsetX = btnRect.left - barRect.left - bar.clientLeft;
  tabPill.style.width = `${btnRect.width}px`;
  tabPill.style.transform = `translateX(${offsetX - 4}px)`;
}

// position pill on first tab on load
requestAnimationFrame(() => positionPill(document.querySelector(".tab-btn.active")));

tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const tabKey = btn.dataset.tab;
    const targetMode = modes[tabKey];
    if (!targetMode) return;

    // toggle active classes
    tabBtns.forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-selected", "false"); });
    tabPanes.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    document.getElementById(`pane-${tabKey}`).classList.add("active");

    // slide the pill
    positionPill(btn);

    currentMode = targetMode;

    // lazy-load model on first switch
    if (!targetMode.onnxSession) {
      autoLoadModel(targetMode);
    }

    // defer redraw until layout has settled (pane was display:none)
    requestAnimationFrame(() => {
      drawInputGrid(targetMode);
      drawNetwork(targetMode, zeroActivations(targetMode));
    });
  });
});

// --------------- resize handler ----------------------------------
window.addEventListener("resize", () => {
  drawInputGrid(currentMode);
  drawNetwork(currentMode, zeroActivations(currentMode));
  positionPill(document.querySelector(".tab-btn.active"));
});

// --------------- per-mode button wiring --------------------------
function wireModeButtons(mode) {
  mode.clearBtn.addEventListener("click", () => {
    resetDrawCanvas(mode);
    setIdlePredictionState(mode);
  });

  mode.loadBtn.addEventListener("click", async () => {
    const url = mode.modelUrlInput.value.trim();
    if (!url) { setStatus(mode, "Enter a model URL first."); return; }
    try {
      setStatus(mode, "Loading model...", true);
      await loadOnnxModel(mode, url);
      setIdlePredictionState(mode);
      setStatus(mode, mode.config.idleStatusMsg);
    } catch (err) {
      setStatus(mode, `Model load failed: ${err.message}`);
    }
  });

  mode.downloadBtn.addEventListener("click", async () => {
    if (!mode.onnxSession || !mode.loadedModelUrl) { setStatus(mode, "Load a model first."); return; }
    try {
      const response = await fetch(mode.loadedModelUrl);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      const link = document.createElement("a");
      const fileName = mode.loadedModelUrl.split("/").pop() || "model.onnx";
      link.href = URL.createObjectURL(blob);
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
      setStatus(mode, "Downloaded ONNX model file.");
    } catch (err) {
      setStatus(mode, `Download failed: ${err.message}`);
    }
  });
}

// --------------- status helpers ----------------------------------
function setStatus(mode, text, loading = false) {
  mode.statusEl.textContent = text;
  mode.statusEl.classList.toggle("loading", loading);
}

// --------------- model loading -----------------------------------
async function autoLoadModel(mode) {
  mode.modelUrlInput.value = mode.config.defaultModelUrl;
  try {
    setStatus(mode, mode.config.loadingMsg, true);
    await loadOnnxModel(mode, mode.config.defaultModelUrl);
    // redraw network after model load updates layerSpecs
    requestAnimationFrame(() => {
      drawInputGrid(mode);
      setIdlePredictionState(mode);
    });
    setStatus(mode, mode.config.idleStatusMsg);
  } catch (err) {
    setStatus(mode, `Auto-load failed: ${err.message}`);
  }
}

async function loadOnnxModel(mode, url) {
  if (typeof ort === "undefined") throw new Error("onnxruntime-web failed to load.");

  mode.loadedModelUrl = url;
  mode.onnxSession = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
  mode.onnxInputName = mode.onnxSession.inputNames[0];
  if (!mode.onnxInputName) throw new Error("ONNX model has no inputs.");

  const inputMetadataMap = mode.onnxSession.inputMetadata || null;
  const outputMetadataMap = mode.onnxSession.outputMetadata || null;
  const inputMetadata = inputMetadataMap ? inputMetadataMap[mode.onnxInputName] : null;

  mode.onnxOutputName = (mode.onnxSession.outputNames || [])[0] || null;
  if (!mode.onnxOutputName && outputMetadataMap) {
    const keys = Object.keys(outputMetadataMap);
    mode.onnxOutputName = keys.length > 0 ? keys[0] : null;
  }

  const outputMetadata = mode.onnxOutputName && outputMetadataMap ? outputMetadataMap[mode.onnxOutputName] : null;

  mode.onnxInputDims = inputMetadata ? inputMetadata.dimensions : null;
  mode.onnxOutputDims = outputMetadata ? outputMetadata.dimensions : null;
  drawInputGrid(mode);
  mode.layerSpecs = buildOnnxLayerSpecs(mode.onnxInputDims, mode.onnxOutputDims, mode.config.classCount);

  setStatus(mode,
    `Loaded ONNX model: input ${mode.onnxInputName} ${formatDims(mode.onnxInputDims)} | grid ${INPUT_GRID_COLS}x${INPUT_GRID_ROWS}`
  );
}

// --------------- layer spec builder ------------------------------
function buildOnnxLayerSpecs(inputDims, outputDims, expectedClassCount) {
  const inputFeatureCount = inferInputFeatureCount(inputDims);
  const outputNodes = expectedClassCount || inferOutputClassCount(outputDims);

  if (outputNodes > 10) {
    // deeper architecture for character model
    return [
      { name: "Input", count: inputFeatureCount },
      { name: "Hidden 1", count: 256 },
      { name: "Hidden 2", count: 128 },
      { name: "Output", count: outputNodes },
    ];
  }

  return [
    { name: "Input", count: inputFeatureCount },
    { name: "Hidden", count: 128 },
    { name: "Output", count: outputNodes },
  ];
}

function inferInputFeatureCount(dims) {
  if (!Array.isArray(dims) || dims.length === 0) return 784;
  const numericDims = dims.slice(1).filter((v) => typeof v === "number" && v > 0);
  if (numericDims.length === 0) return 784;
  const product = numericDims.reduce((a, b) => a * b, 1);
  return product > 0 ? product : 784;
}

function inferOutputClassCount(dims) {
  if (!Array.isArray(dims) || dims.length === 0) return 10;
  for (let i = dims.length - 1; i >= 0; i -= 1) {
    const v = dims[i];
    if (typeof v === "number" && v > 0) return clamp(v, 2, 100);
  }
  return 10;
}

function formatDims(dims) {
  if (!Array.isArray(dims) || dims.length === 0) return "(unknown shape)";
  return `[${dims.map((d) => (d == null ? "?" : d)).join(", ")}]`;
}

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

// --------------- output grid setup -------------------------------
function setupOutputGrid(mode) {
  mode.outputItems = [];
  mode.outputGrid.innerHTML = "";
  for (let i = 0; i < mode.config.classCount; i += 1) {
    const card = document.createElement("div");
    card.className = `digit-card${mode.config.key === "characters" ? " char-mode" : ""}`;
    card.innerHTML = `
      <div class="digit">${mode.config.label(i)}</div>
      <div class="prob-chart"><div class="prob-fill"></div></div>
      <div class="prob-value">0.0%</div>
    `;
    mode.outputGrid.appendChild(card);
    mode.outputItems.push({
      card,
      fill: card.querySelector(".prob-fill"),
      value: card.querySelector(".prob-value"),
    });
  }
}

// --------------- idle / prediction state -------------------------
function zeroActivations(mode) {
  return mode.layerSpecs.map((layer) => new Array(Math.min(layer.count, 50)).fill(0));
}

function setIdlePredictionState(mode) {
  updateOutputs(mode, new Array(mode.config.classCount).fill(0));
  mode.predictionEl.textContent = "Prediction: -";
  drawNetwork(mode, zeroActivations(mode));
}

function updateGridDimsLabel(mode) {
  if (!mode.gridDimsLabel) return;
  mode.gridDimsLabel.textContent = `${INPUT_GRID_COLS} x ${INPUT_GRID_ROWS}`;
}

// --------------- draw grid (28x28 overlay) -----------------------
function drawInputGrid(mode) {
  const dpr = window.devicePixelRatio || 1;
  const width = mode.gridCanvas.clientWidth;
  const height = mode.gridCanvas.clientHeight;

  mode.gridCanvas.width = Math.floor(width * dpr);
  mode.gridCanvas.height = Math.floor(height * dpr);
  mode.gridCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  mode.gridCtx.clearRect(0, 0, width, height);

  const rows = INPUT_GRID_ROWS;
  const cols = INPUT_GRID_COLS;
  const cellW = width / cols;
  const cellH = height / rows;

  for (let c = 1; c < cols; c += 1) {
    const major = cols % 4 === 0 && c % (cols / 4) === 0;
    mode.gridCtx.strokeStyle = major ? "rgba(255, 255, 255, 0.22)" : "rgba(255, 255, 255, 0.08)";
    mode.gridCtx.lineWidth = 1;
    const x = c * cellW;
    mode.gridCtx.beginPath();
    mode.gridCtx.moveTo(x, 0);
    mode.gridCtx.lineTo(x, height);
    mode.gridCtx.stroke();
  }

  for (let r = 1; r < rows; r += 1) {
    const major = rows % 4 === 0 && r % (rows / 4) === 0;
    mode.gridCtx.strokeStyle = major ? "rgba(95, 104, 122, 0.28)" : "rgba(95, 104, 122, 0.12)";
    mode.gridCtx.lineWidth = 1;
    const y = r * cellH;
    mode.gridCtx.beginPath();
    mode.gridCtx.moveTo(0, y);
    mode.gridCtx.lineTo(width, y);
    mode.gridCtx.stroke();
  }
}

// --------------- canvas drawing ----------------------------------
function resetDrawCanvas(mode) {
  mode.drawCtx.fillStyle = "#ffffff";
  mode.drawCtx.fillRect(0, 0, mode.drawCanvas.width, mode.drawCanvas.height);
  mode.drawCtx.lineCap = "round";
  mode.drawCtx.lineJoin = "round";
  mode.drawCtx.strokeStyle = "#000000";
  mode.drawCtx.lineWidth = 22;
}

function wireDrawingEvents(mode) {
  mode.drawCanvas.addEventListener("pointerdown", (event) => {
    mode.drawing = true;
    mode.drawCanvas.setPointerCapture(event.pointerId);
    const { x, y } = getCanvasPoint(mode, event);
    mode.drawCtx.beginPath();
    mode.drawCtx.moveTo(x, y);
    mode.drawCtx.lineTo(x + 0.01, y + 0.01);
    mode.drawCtx.stroke();
    schedulePrediction(mode);
  });

  mode.drawCanvas.addEventListener("pointermove", (event) => {
    if (!mode.drawing) return;
    drawAt(mode, event);
    schedulePrediction(mode);
  });

  const stopDrawing = () => {
    if (!mode.drawing) return;
    mode.drawing = false;
    mode.drawCtx.beginPath();
    schedulePrediction(mode);
  };

  mode.drawCanvas.addEventListener("pointerup", stopDrawing);
  mode.drawCanvas.addEventListener("pointercancel", stopDrawing);
}

function drawAt(mode, event) {
  const { x, y } = getCanvasPoint(mode, event);
  mode.drawCtx.lineTo(x, y);
  mode.drawCtx.stroke();
  mode.drawCtx.beginPath();
  mode.drawCtx.moveTo(x, y);
}

function getCanvasPoint(mode, event) {
  const rect = mode.drawCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * mode.drawCanvas.width,
    y: ((event.clientY - rect.top) / rect.height) * mode.drawCanvas.height,
  };
}

// --------------- prediction pipeline ----------------------------
function schedulePrediction(mode) {
  if (!mode.onnxSession) return;
  if (mode.predictionFrame) return;
  mode.predictionFrame = requestAnimationFrame(async () => {
    mode.predictionFrame = 0;
    await runPrediction(mode);
  });
}

async function runPrediction(mode) {
  if (!mode.onnxSession) return;
  try {
    const { inputArray, inputSummary, inkSum } = getInvertedInputArrayAndSummary(mode);
    if (inkSum < 2.0) { setIdlePredictionState(mode); return; }

    // EMNIST characters need transposed input (rows <-> columns)
    const finalInput = mode.config.transpose
      ? transposeGrid(inputArray, 28, 28)
      : inputArray;

    const candidateDims = buildOnnxInputDimCandidates(finalInput.length, mode.onnxInputDims);

    let results = null;
    let lastError = null;

    for (const dims of candidateDims) {
      try {
        const feeds = {};
        feeds[mode.onnxInputName] = new ort.Tensor("float32", finalInput, dims);
        results = await mode.onnxSession.run(feeds);
        break;
      } catch (err) { lastError = err; }
    }

    if (!results) {
      throw new Error(
        `ONNX inference failed for tested input shapes ${candidateDims
          .map((d) => `[${d.join(", ")}]`).join(", ")}. Last error: ${lastError ? lastError.message : "unknown"}`
      );
    }

    const outputName = mode.onnxOutputName || Object.keys(results)[0];
    const outputTensor = outputName ? results[outputName] : null;
    if (!outputTensor) throw new Error("ONNX output tensor missing.");

    const classCount = mode.config.classCount;
    const rawOutput = Array.from(outputTensor.data).slice(-classCount);
    const probs = normalizeAsProbabilities(rawOutput);

    const hidden = buildHiddenActivations(finalInput, probs, mode.layerSpecs[1].count);

    // build layer activations: for 4-layer models, include a second hidden layer
    let layerActivations;
    if (mode.layerSpecs.length === 4) {
      const hidden2 = buildHiddenActivations(finalInput, probs, mode.layerSpecs[2].count);
      layerActivations = [inputSummary, hidden, hidden2, probs];
    } else {
      layerActivations = [inputSummary, hidden, probs];
    }

    applyPredictionToUi(mode, probs, layerActivations);
  } catch (err) {
    setStatus(mode, `Predict failed: ${err.message}`);
  }
}

// --------------- input preprocessing ----------------------------
function getInvertedInputArrayAndSummary(mode) {
  mode.offscreenCtx.drawImage(mode.drawCanvas, 0, 0, 28, 28);
  const imageData = mode.offscreenCtx.getImageData(0, 0, 28, 28);
  const pixels = imageData.data;
  const grayscale = new Float32Array(28 * 28);

  for (let i = 0, p = 0; i < pixels.length; i += 4, p += 1) {
    const gray = pixels[i] / 255;
    grayscale[p] = 1 - gray;
  }

  const mnistReady = normalizeToMnistStyle(mode, grayscale);

  let inkSum = 0;
  for (let i = 0; i < mnistReady.length; i += 1) inkSum += mnistReady[i];

  return {
    inputArray: mnistReady,
    inputSummary: summarizeArray(mnistReady, mode.layerSpecs[0].count),
    inkSum,
  };
}

function normalizeToMnistStyle(mode, src) {
  const threshold = 0.08;
  let minX = 28, minY = 28, maxX = -1, maxY = -1;

  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      if (src[y * 28 + x] > threshold) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < minX || maxY < minY) return src;

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const targetSize = 20;
  const scale = targetSize / Math.max(boxW, boxH);
  const drawW = Math.max(1, Math.round(boxW * scale));
  const drawH = Math.max(1, Math.round(boxH * scale));
  const offsetX = Math.floor((28 - drawW) / 2);
  const offsetY = Math.floor((28 - drawH) / 2);

  const prepCtx = mode.mnistPrepCanvas.getContext("2d", { willReadFrequently: true });
  prepCtx.clearRect(0, 0, 28, 28);
  prepCtx.fillStyle = "#000";
  for (let y = 0; y < drawH; y += 1) {
    for (let x = 0; x < drawW; x += 1) {
      const srcX = minX + Math.min(boxW - 1, Math.floor(x / scale));
      const srcY = minY + Math.min(boxH - 1, Math.floor(y / scale));
      const value = src[srcY * 28 + srcX];
      const alpha = clamp(value, 0, 1);
      if (alpha > 0) {
        prepCtx.fillStyle = `rgba(0,0,0,${alpha})`;
        prepCtx.fillRect(offsetX + x, offsetY + y, 1, 1);
      }
    }
  }

  const out = new Float32Array(28 * 28);
  const centered = prepCtx.getImageData(0, 0, 28, 28).data;
  for (let i = 0, p = 0; i < centered.length; i += 4, p += 1) out[p] = centered[i + 3] / 255;
  return out;
}

function buildOnnxInputDimCandidates(flatCount, declaredDims) {
  const candidates = [];
  const addCandidate = (dims) => {
    if (!Array.isArray(dims) || dims.length === 0) return;
    const product = dims.reduce((a, b) => a * b, 1);
    if (product !== flatCount) return;
    const key = dims.join("x");
    if (!candidates.some((e) => e.join("x") === key)) candidates.push(dims);
  };

  if (Array.isArray(declaredDims) && declaredDims.length > 0)
    addCandidate(declaredDims.map((d) => (typeof d === "number" && d > 0 ? d : 1)));

  addCandidate([1, 1, 28, 28]);
  addCandidate([1, 28, 28]);
  addCandidate([1, 28, 28, 1]);
  addCandidate([1, 784]);
  return candidates;
}

// --------------- probability helpers ----------------------------
function buildHiddenActivations(inputArray, probabilities, hiddenCount) {
  const bins = new Array(hiddenCount).fill(0);
  const step = Math.floor(inputArray.length / hiddenCount) || 1;
  const confidence = Math.max(...probabilities);

  for (let i = 0; i < hiddenCount; i += 1) {
    const start = i * step;
    const end = i === hiddenCount - 1 ? inputArray.length : (i + 1) * step;
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += inputArray[j];
    const localActivation = sum / Math.max(1, end - start);
    const bias = (i % 3) * 0.05;
    bins[i] = clamp(localActivation * 1.8 + confidence * 0.2 + bias, 0, 1);
  }
  return bins;
}

function applyPredictionToUi(mode, probabilities, layerActivations) {
  if (!Array.isArray(probabilities) || probabilities.length !== mode.config.classCount) return;
  const bestIdx = probabilities.indexOf(Math.max(...probabilities));
  const confidence = (probabilities[bestIdx] * 100).toFixed(1);
  mode.predictionEl.textContent = `Prediction: ${mode.config.label(bestIdx)} (${confidence}%)`;
  updateOutputs(mode, probabilities, bestIdx);
  drawNetwork(mode, layerActivations);
}

function summarizeArray(values, count) {
  const positive = values.map((v) => Math.max(0, v));
  const max = Math.max(...positive, 1e-6);
  const normalized = positive.map((v) => v / max);
  const sampled = new Array(count).fill(0);
  if (normalized.length === 0) return sampled;
  for (let i = 0; i < count; i += 1) {
    const idx = Math.floor((i / count) * normalized.length);
    sampled[i] = normalized[Math.min(idx, normalized.length - 1)];
  }
  return sampled;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((v) => v / sum);
}

function normalizeAsProbabilities(values) {
  const nonNegative = values.every((v) => v >= 0);
  const max = Math.max(...values);
  const sum = values.reduce((a, b) => a + b, 0);
  if (nonNegative && max <= 1.0001 && sum > 0.99 && sum < 1.01) return values;
  return softmax(values);
}

function updateOutputs(mode, probabilities, winnerIdx = -1) {
  probabilities.forEach((probability, index) => {
    if (!mode.outputItems[index]) return;
    const pct = clamp(probability * 100, 0, 100);
    mode.outputItems[index].fill.style.height = `${pct}%`;
    mode.outputItems[index].value.textContent = `${pct.toFixed(1)}%`;
    mode.outputItems[index].card.classList.toggle("winner", index === winnerIdx);
  });
}

// --------------- network visualization --------------------------

function drawNetwork(mode, activations) {
  const dpr = window.devicePixelRatio || 1;
  const width = mode.networkCanvas.clientWidth;
  const height = mode.networkCanvas.clientHeight;

  if (width < 1 || height < 1) {
    requestAnimationFrame(() => drawNetwork(mode, activations));
    return;
  }

  mode.networkCanvas.width = Math.floor(width * dpr);
  mode.networkCanvas.height = Math.floor(height * dpr);
  const ctx = mode.networkCtx;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(0, 0, 0, 0.25)";
  ctx.fillRect(0, 0, width, height);

  const specs = mode.layerSpecs;
  const layerCount = specs.length;
  const LABEL_H = 22;  // space for label below each layer

  // --- compute each layer's bounding box (horizontal row or grid) ---
  const layers = specs.map((spec, li) => {
    const n = spec.count;
    const acts = activations[li] || [];

    if (li === 0 && n > 50) {
      // input layer → 2D grid (it represents the 28×28 image)
      const cols = Math.ceil(Math.sqrt(n));
      const rows = Math.ceil(n / cols);
      const maxGridW = width * 0.7;
      const dotSize = Math.max(1.5, Math.min(4, maxGridW / (cols * 1.5)));
      const gap = dotSize + Math.max(0.5, dotSize * 0.2);
      return {
        spec, n, acts, type: "grid", cols, rows, dotSize, gap,
        w: cols * gap, h: rows * gap
      };
    } else {
      // all other layers → 1D horizontal row (they are 1D vectors)
      const maxRowW = width * 0.95;
      const spacing = Math.min(18, maxRowW / Math.max(n - 1, 1));
      const r = Math.max(0.8, Math.min(7, spacing * 0.4));
      return {
        spec, n, acts, type: "row", r, spacing, visCount: n, truncated: false,
        w: spacing * (n - 1) + r * 2, h: r * 2
      };
    }
  });

  // --- distribute layers vertically, evenly spaced ---
  const padTop = 14;
  const padBottom = 8;
  const totalLayerH = layers.reduce((s, l) => s + l.h + LABEL_H, 0);
  const freeSpace = height - padTop - padBottom - totalLayerH;
  const gapBetween = Math.max(16, freeSpace / Math.max(layerCount - 1, 1));

  let curY = padTop;
  layers.forEach((layer) => {
    layer.y = curY;
    layer.cx = width / 2;
    curY += layer.h + LABEL_H + gapBetween;
  });

  // --- compute node center positions for connection drawing ---
  layers.forEach((layer) => {
    layer.nodeCenters = [];
    if (layer.type === "grid") {
      const startX = layer.cx - layer.w / 2;
      const startY = layer.y;
      for (let i = 0; i < layer.n; i++) {
        const r = Math.floor(i / layer.cols);
        const c = i % layer.cols;
        layer.nodeCenters.push({
          x: startX + c * layer.gap + layer.dotSize / 2,
          y: startY + r * layer.gap + layer.dotSize / 2,
        });
      }
    } else {
      const startX = layer.cx - layer.w / 2 + layer.r;
      for (let i = 0; i < layer.visCount; i++) {
        layer.nodeCenters.push({ x: startX + i * layer.spacing, y: layer.y + layer.r });
      }
    }
  });

  // --- draw connections (sample N representative source → target lines) ---
  for (let i = 0; i < layers.length - 1; i++) {
    const src = layers[i];
    const dst = layers[i + 1];
    const srcActs = src.acts;
    const dstActs = dst.acts;

    // sample ~20 source nodes evenly spread among visible nodes
    const srcLen = src.nodeCenters.length;
    const dstLen = dst.nodeCenters.length;
    const sampleCount = Math.min(20, srcLen);
    const srcStep = srcLen / sampleCount;
    // each sampled source connects to 2 target nodes
    const dstTargets = 2;

    for (let si = 0; si < sampleCount; si++) {
      const srcIdx = Math.floor(si * srcStep);
      const sNode = src.nodeCenters[srcIdx];
      if (!sNode) continue;

      for (let k = 0; k < dstTargets; k++) {
        const dstIdx = (srcIdx * 3 + k * 7) % dstLen;
        const dNode = dst.nodeCenters[dstIdx];
        if (!dNode) continue;

        const a = srcActs[srcIdx] || 0;
        const b = dstActs[dstIdx] || 0;
        const s = clamp((a + b) * 0.5, 0, 1);

        ctx.strokeStyle = `rgba(255, 255, 255, ${0.05 + s * 0.2})`;
        ctx.lineWidth = 0.6 + s * 0.4;
        ctx.beginPath();
        ctx.moveTo(sNode.x, sNode.y + (src.type === "grid" ? src.dotSize : src.r));
        ctx.lineTo(dNode.x, dNode.y - (dst.type === "grid" ? dst.dotSize : dst.r));
        ctx.stroke();
      }
    }
  }

  // --- draw nodes ---
  layers.forEach((layer) => {
    if (layer.type === "grid") {
      const startX = layer.cx - layer.w / 2;
      const startY = layer.y;
      const ds = layer.dotSize;

      for (let i = 0; i < layer.n; i++) {
        const r = Math.floor(i / layer.cols);
        const c = i % layer.cols;
        const act = clamp(layer.acts[i] || 0, 0, 1);
        const shade = Math.round(240 - act * 190);
        ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
        ctx.fillRect(startX + c * layer.gap, startY + r * layer.gap, ds, ds);
      }
      // subtle border around the grid
      ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
      ctx.lineWidth = 1;
      ctx.strokeRect(startX - 1, startY - 1, layer.w + 2, layer.h + 2);

    } else {
      const startX = layer.cx - layer.w / 2 + layer.r;
      layer.nodeCenters.forEach((node, ni) => {
        const act = clamp(layer.acts[ni] || 0, 0, 1);
        const shade = Math.round(245 - act * 170);
        ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
        ctx.lineWidth = layer.r < 2 ? 0.5 : 1;
        ctx.beginPath();
        ctx.arc(node.x, node.y, layer.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    }

    // label below layer
    const labelY = layer.y + layer.h + 16;
    ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
    ctx.font = "11px Manrope";
    ctx.textAlign = "center";
    ctx.fillText(`${layer.spec.name} (${layer.n})`, layer.cx, labelY);
  });
}

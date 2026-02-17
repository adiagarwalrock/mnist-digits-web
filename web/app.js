const drawCanvas = document.getElementById("draw-canvas");
const gridCanvas = document.getElementById("grid-canvas");
const networkCanvas = document.getElementById("network-canvas");
const clearBtn = document.getElementById("clear-btn");
const loadBtn = document.getElementById("load-btn");
const downloadBtn = document.getElementById("download-btn");
const modelUrlInput = document.getElementById("model-url");
const statusEl = document.getElementById("status");
const predictionEl = document.getElementById("prediction");
const outputGrid = document.getElementById("output-grid");
const gridDimsLabel = document.getElementById("grid-dims-label");

const DEFAULT_MODEL_URL = "./model/model.onnx";

const drawCtx = drawCanvas.getContext("2d");
const gridCtx = gridCanvas.getContext("2d");
const networkCtx = networkCanvas.getContext("2d");
const offscreenCanvas = document.createElement("canvas");
offscreenCanvas.width = 28;
offscreenCanvas.height = 28;
const offscreenCtx = offscreenCanvas.getContext("2d");

const digitItems = [];
let onnxSession = null;
let onnxInputName = null;
let onnxOutputName = null;
let onnxInputDims = null;
let onnxOutputDims = null;
let loadedModelUrl = null;
let drawing = false;
let predictionFrame = 0;
const INPUT_GRID_ROWS = 28;
const INPUT_GRID_COLS = 28;
const mnistPrepCanvas = document.createElement("canvas");
mnistPrepCanvas.width = 28;
mnistPrepCanvas.height = 28;
const mnistPrepCtx = mnistPrepCanvas.getContext("2d");
let layerSpecs = [
  { name: "Input", count: 28 },
  { name: "Hidden", count: 16 },
  { name: "Output", count: 10 },
];

setupOutputGrid();
resetDrawCanvas();
drawInputGrid();
setIdlePredictionState();
updateGridDimsLabel();
wireDrawingEvents();
autoLoadDefaultModel();

clearBtn.addEventListener("click", () => {
  resetDrawCanvas();
  setIdlePredictionState();
});

loadBtn.addEventListener("click", async () => {
  const url = modelUrlInput.value.trim();
  if (!url) {
    setStatus("Enter a model URL first.");
    return;
  }

  try {
    setStatus("Loading model...");
    await loadOnnxModel(url);
    setIdlePredictionState();
    setStatus("Model loaded. Draw a digit to predict.");
  } catch (err) {
    setStatus(`Model load failed: ${err.message}`);
  }
});

downloadBtn.addEventListener("click", async () => {
  if (!onnxSession || !loadedModelUrl) {
    setStatus("Load a model first.");
    return;
  }

  try {
    const response = await fetch(loadedModelUrl);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const link = document.createElement("a");
    const fileName = loadedModelUrl.split("/").pop() || "model.onnx";
    link.href = URL.createObjectURL(blob);
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
    setStatus("Downloaded ONNX model file.");
  } catch (err) {
    setStatus(`Download failed: ${err.message}`);
  }
});

window.addEventListener("resize", () => {
  drawInputGrid();
  drawNetwork(layerSpecs.map((layer) => new Array(layer.count).fill(0)));
});

function setStatus(text) {
  statusEl.textContent = text;
}

async function autoLoadDefaultModel() {
  modelUrlInput.value = DEFAULT_MODEL_URL;
  try {
    setStatus(`Auto-loading ${DEFAULT_MODEL_URL}...`);
    await loadOnnxModel(DEFAULT_MODEL_URL);
    setIdlePredictionState();
    setStatus("Model auto-loaded. Draw a digit to predict.");
  } catch (err) {
    setStatus(`Auto-load failed: ${err.message}`);
  }
}

async function loadOnnxModel(url) {
  if (typeof ort === "undefined") {
    throw new Error("onnxruntime-web failed to load.");
  }

  loadedModelUrl = url;
  onnxSession = await ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
  });
  onnxInputName = onnxSession.inputNames[0];
  if (!onnxInputName) {
    throw new Error("ONNX model has no inputs.");
  }

  const inputMetadataMap = onnxSession.inputMetadata || null;
  const outputMetadataMap = onnxSession.outputMetadata || null;
  const inputMetadata = inputMetadataMap ? inputMetadataMap[onnxInputName] : null;

  onnxOutputName = (onnxSession.outputNames || [])[0] || null;
  if (!onnxOutputName && outputMetadataMap) {
    const outputKeys = Object.keys(outputMetadataMap);
    onnxOutputName = outputKeys.length > 0 ? outputKeys[0] : null;
  }

  const outputMetadata = onnxOutputName && outputMetadataMap
    ? outputMetadataMap[onnxOutputName]
    : null;

  onnxInputDims = inputMetadata ? inputMetadata.dimensions : null;
  onnxOutputDims = outputMetadata ? outputMetadata.dimensions : null;
  drawInputGrid();
  layerSpecs = buildOnnxLayerSpecs(onnxInputDims, onnxOutputDims);

  setStatus(
    `Loaded ONNX model: input ${onnxInputName} ${formatDims(onnxInputDims)} | grid ${INPUT_GRID_COLS}x${INPUT_GRID_ROWS}`
  );
}

function buildOnnxLayerSpecs(inputDims, outputDims) {
  const inputFeatureCount = inferInputFeatureCount(inputDims);
  let inputNodes = clamp(Math.round(Math.sqrt(inputFeatureCount)), 20, 28);
  let hiddenNodes = clamp(Math.round(inputNodes * 0.58), 12, 18);
  if (hiddenNodes >= inputNodes) hiddenNodes = Math.max(12, inputNodes - 8);
  const outputNodes = inferOutputClassCount(outputDims);
  if (outputNodes === hiddenNodes) hiddenNodes = Math.max(12, hiddenNodes - 2);
  return [
    { name: "Input", count: inputNodes },
    { name: "Hidden", count: hiddenNodes },
    { name: "Output", count: outputNodes },
  ];
}

function inferInputFeatureCount(dims) {
  if (!Array.isArray(dims) || dims.length === 0) return 784;
  const numericDims = dims
    .slice(1)
    .filter((value) => typeof value === "number" && value > 0);
  if (numericDims.length === 0) return 784;
  const product = numericDims.reduce((acc, value) => acc * value, 1);
  return product > 0 ? product : 784;
}

function inferOutputClassCount(dims) {
  if (!Array.isArray(dims) || dims.length === 0) return 10;
  for (let i = dims.length - 1; i >= 0; i -= 1) {
    const value = dims[i];
    if (typeof value === "number" && value > 0) {
      return clamp(value, 2, 10);
    }
  }
  return 10;
}

function formatDims(dims) {
  if (!Array.isArray(dims) || dims.length === 0) return "(unknown shape)";
  return `[${dims.map((d) => (d == null ? "?" : d)).join(", ")}]`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function zeroActivations() {
  return layerSpecs.map((layer) => new Array(layer.count).fill(0));
}

function setIdlePredictionState() {
  updateOutputs(new Array(10).fill(0));
  predictionEl.textContent = "Prediction: -";
  drawNetwork(zeroActivations());
}

function updateGridDimsLabel() {
  if (!gridDimsLabel) return;
  gridDimsLabel.textContent = `${INPUT_GRID_COLS} x ${INPUT_GRID_ROWS}`;
}

function drawInputGrid() {
  const dpr = window.devicePixelRatio || 1;
  const width = gridCanvas.clientWidth;
  const height = gridCanvas.clientHeight;

  gridCanvas.width = Math.floor(width * dpr);
  gridCanvas.height = Math.floor(height * dpr);
  gridCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  gridCtx.clearRect(0, 0, width, height);

  const rows = INPUT_GRID_ROWS;
  const cols = INPUT_GRID_COLS;
  const cellW = width / cols;
  const cellH = height / rows;

  for (let c = 1; c < cols; c += 1) {
    const major = cols % 4 === 0 && c % (cols / 4) === 0;
    gridCtx.strokeStyle = major ? "rgba(95, 104, 122, 0.28)" : "rgba(95, 104, 122, 0.15)";
    gridCtx.lineWidth = 1;
    const x = c * cellW;
    gridCtx.beginPath();
    gridCtx.moveTo(x, 0);
    gridCtx.lineTo(x, height);
    gridCtx.stroke();
  }

  for (let r = 1; r < rows; r += 1) {
    const major = rows % 4 === 0 && r % (rows / 4) === 0;
    gridCtx.strokeStyle = major ? "rgba(95, 104, 122, 0.28)" : "rgba(95, 104, 122, 0.15)";
    gridCtx.lineWidth = 1;
    const y = r * cellH;
    gridCtx.beginPath();
    gridCtx.moveTo(0, y);
    gridCtx.lineTo(width, y);
    gridCtx.stroke();
  }
}

function setupOutputGrid() {
  for (let digit = 0; digit < 10; digit += 1) {
    const card = document.createElement("div");
    card.className = "digit-card";
    card.innerHTML = `
      <div class="digit">${digit}</div>
      <div class="prob-chart"><div class="prob-fill"></div></div>
      <div class="prob-value">0.0%</div>
    `;
    outputGrid.appendChild(card);
    digitItems.push({
      fill: card.querySelector(".prob-fill"),
      value: card.querySelector(".prob-value"),
    });
  }
}

function resetDrawCanvas() {
  drawCtx.fillStyle = "#ffffff";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.strokeStyle = "#000000";
  drawCtx.lineWidth = 22;
}

function wireDrawingEvents() {
  drawCanvas.addEventListener("pointerdown", (event) => {
    drawing = true;
    drawCanvas.setPointerCapture(event.pointerId);
    const { x, y } = getCanvasPoint(event);
    drawCtx.beginPath();
    drawCtx.moveTo(x, y);
    drawCtx.lineTo(x + 0.01, y + 0.01);
    drawCtx.stroke();
    schedulePrediction();
  });

  drawCanvas.addEventListener("pointermove", (event) => {
    if (!drawing) return;
    drawAt(event);
    schedulePrediction();
  });

  const stopDrawing = () => {
    if (!drawing) return;
    drawing = false;
    drawCtx.beginPath();
    schedulePrediction();
  };

  drawCanvas.addEventListener("pointerup", stopDrawing);
  drawCanvas.addEventListener("pointercancel", stopDrawing);
}

function drawAt(event) {
  const { x, y } = getCanvasPoint(event);
  drawCtx.lineTo(x, y);
  drawCtx.stroke();
  drawCtx.beginPath();
  drawCtx.moveTo(x, y);
}

function getCanvasPoint(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * drawCanvas.width,
    y: ((event.clientY - rect.top) / rect.height) * drawCanvas.height,
  };
}

function schedulePrediction() {
  if (!onnxSession) return;
  if (predictionFrame) return;
  predictionFrame = requestAnimationFrame(async () => {
    predictionFrame = 0;
    await runPrediction();
  });
}

async function runPrediction() {
  if (!onnxSession) return;
  try {
    const { inputArray, inputSummary, inkSum } = getInvertedInputArrayAndSummary();
    if (inkSum < 2.0) {
      setIdlePredictionState();
      return;
    }

    const candidateDims = buildOnnxInputDimCandidates(
      inputArray.length,
      onnxInputDims
    );

    let results = null;
    let lastError = null;

    for (const dims of candidateDims) {
      try {
        const feeds = {};
        feeds[onnxInputName] = new ort.Tensor("float32", inputArray, dims);
        results = await onnxSession.run(feeds);
        break;
      } catch (err) {
        lastError = err;
      }
    }

    if (!results) {
      throw new Error(
        `ONNX inference failed for tested input shapes ${candidateDims
          .map((d) => `[${d.join(", ")}]`)
          .join(", ")}. Last error: ${lastError ? lastError.message : "unknown"}`
      );
    }

    const outputName = onnxOutputName || Object.keys(results)[0];
    const outputTensor = outputName ? results[outputName] : null;
    if (!outputTensor) {
      throw new Error("ONNX output tensor missing.");
    }

    const classCount = inferOutputClassCount(onnxOutputDims);
    const rawOutput = Array.from(outputTensor.data).slice(-classCount);
    const probs = normalizeAsProbabilities(rawOutput);

    if (probs.length !== 10) {
      throw new Error(
        `Expected 10 outputs for MNIST, got ${probs.length}. Check model output classes.`
      );
    }

    const hidden = buildHiddenActivations(inputArray, probs, layerSpecs[1].count);
    applyPredictionToUi(probs, [inputSummary, hidden, probs]);
  } catch (err) {
    setStatus(`Predict failed: ${err.message}`);
  }
}

function getInvertedInputArrayAndSummary() {
  offscreenCtx.drawImage(drawCanvas, 0, 0, 28, 28);
  const imageData = offscreenCtx.getImageData(0, 0, 28, 28);
  const pixels = imageData.data;
  const grayscale = new Float32Array(28 * 28);

  for (let i = 0, p = 0; i < pixels.length; i += 4, p += 1) {
    const gray = pixels[i] / 255;
    grayscale[p] = 1 - gray;
  }

  const mnistReady = normalizeToMnistStyle(grayscale);

  let inkSum = 0;
  for (let i = 0; i < mnistReady.length; i += 1) {
    inkSum += mnistReady[i];
  }

  return {
    inputArray: mnistReady,
    inputSummary: summarizeArray(mnistReady, layerSpecs[0].count),
    inkSum,
  };
}

function normalizeToMnistStyle(src) {
  const threshold = 0.08;
  let minX = 28;
  let minY = 28;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      const value = src[y * 28 + x];
      if (value > threshold) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < minX || maxY < minY) {
    return src;
  }

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const targetSize = 20;
  const scale = targetSize / Math.max(boxW, boxH);
  const drawW = Math.max(1, Math.round(boxW * scale));
  const drawH = Math.max(1, Math.round(boxH * scale));
  const offsetX = Math.floor((28 - drawW) / 2);
  const offsetY = Math.floor((28 - drawH) / 2);

  mnistPrepCtx.clearRect(0, 0, 28, 28);
  mnistPrepCtx.fillStyle = "#000";
  for (let y = 0; y < drawH; y += 1) {
    for (let x = 0; x < drawW; x += 1) {
      const srcX = minX + Math.min(boxW - 1, Math.floor(x / scale));
      const srcY = minY + Math.min(boxH - 1, Math.floor(y / scale));
      const value = src[srcY * 28 + srcX];
      const alpha = clamp(value, 0, 1);
      if (alpha > 0) {
        mnistPrepCtx.fillStyle = `rgba(0,0,0,${alpha})`;
        mnistPrepCtx.fillRect(offsetX + x, offsetY + y, 1, 1);
      }
    }
  }

  const out = new Float32Array(28 * 28);
  const centered = mnistPrepCtx.getImageData(0, 0, 28, 28).data;
  for (let i = 0, p = 0; i < centered.length; i += 4, p += 1) {
    out[p] = centered[i + 3] / 255;
  }

  return out;
}

function buildOnnxInputDimCandidates(flatCount, declaredDims) {
  const candidates = [];

  const addCandidate = (dims) => {
    if (!Array.isArray(dims) || dims.length === 0) return;
    const product = dims.reduce((acc, value) => acc * value, 1);
    if (product !== flatCount) return;
    const key = dims.join("x");
    if (!candidates.some((existing) => existing.join("x") === key)) {
      candidates.push(dims);
    }
  };

  if (Array.isArray(declaredDims) && declaredDims.length > 0) {
    addCandidate(declaredDims.map((d) => (typeof d === "number" && d > 0 ? d : 1)));
  }

  addCandidate([1, 1, 28, 28]);
  addCandidate([1, 28, 28]);
  addCandidate([1, 28, 28, 1]);
  addCandidate([1, 784]);

  return candidates;
}

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

function applyPredictionToUi(probabilities, layerActivations) {
  if (!Array.isArray(probabilities) || probabilities.length !== 10) return;
  updateOutputs(probabilities);
  const bestDigit = probabilities.indexOf(Math.max(...probabilities));
  predictionEl.textContent = `Prediction: ${bestDigit}`;
  drawNetwork(layerActivations);
}

function summarizeArray(values, count) {
  const positive = values.map((value) => Math.max(0, value));
  const max = Math.max(...positive, 1e-6);
  const normalized = positive.map((value) => value / max);
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
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((value) => value / sum);
}

function normalizeAsProbabilities(values) {
  const nonNegative = values.every((value) => value >= 0);
  const max = Math.max(...values);
  const sum = values.reduce((a, b) => a + b, 0);
  if (nonNegative && max <= 1.0001 && sum > 0.99 && sum < 1.01) return values;
  return softmax(values);
}

function updateOutputs(probabilities) {
  probabilities.forEach((probability, index) => {
    const pct = clamp(probability * 100, 0, 100);
    digitItems[index].fill.style.height = `${pct}%`;
    digitItems[index].value.textContent = `${pct.toFixed(1)}%`;
  });
}

function drawNetwork(activations) {
  const dpr = window.devicePixelRatio || 1;
  const width = networkCanvas.clientWidth;
  const height = networkCanvas.clientHeight;
  networkCanvas.width = Math.floor(width * dpr);
  networkCanvas.height = Math.floor(height * dpr);
  networkCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  networkCtx.clearRect(0, 0, width, height);
  networkCtx.fillStyle = "#ffffff";
  networkCtx.fillRect(0, 0, width, height);

  const left = 38;
  const right = width - 38;
  const top = 26;
  const bottom = height - 30;
  const layerGap =
    layerSpecs.length > 1 ? (right - left) / (layerSpecs.length - 1) : 0;
  const nodePositions = [];

  layerSpecs.forEach((layer, layerIndex) => {
    const x = left + layerGap * layerIndex;
    const count = layer.count;
    const span = bottom - top;
    const nodeGap = count > 1 ? span / (count - 1) : 0;
    const points = [];
    for (let i = 0; i < count; i += 1) {
      points.push({ x, y: top + nodeGap * i });
    }
    nodePositions.push(points);
  });

  for (let i = 0; i < nodePositions.length - 1; i += 1) {
    const fromNodes = nodePositions[i];
    const toNodes = nodePositions[i + 1];
    const fromActs = activations[i] || [];
    const toActs = activations[i + 1] || [];
    const linkSpread = Math.max(2, Math.floor(toNodes.length / 4));

    fromNodes.forEach((fromNode, fromIndex) => {
      for (let k = 0; k < linkSpread; k += 1) {
        const toIndex = (fromIndex * 3 + k * 2) % toNodes.length;
        const toNode = toNodes[toIndex];
        const a = fromActs[fromIndex] || 0;
        const b = toActs[toIndex] || 0;
        const strength = clamp((a + b) * 0.5, 0, 1);

        networkCtx.strokeStyle = `rgba(35, 43, 58, ${0.05 + strength * 0.35})`;
        networkCtx.lineWidth = 1.2;
        networkCtx.beginPath();
        networkCtx.moveTo(fromNode.x + 8, fromNode.y);
        networkCtx.lineTo(toNode.x - 8, toNode.y);
        networkCtx.stroke();

        networkCtx.strokeStyle = `rgba(185, 196, 215, ${0.04 + (1 - strength) * 0.3})`;
        networkCtx.lineWidth = 0.8;
        networkCtx.beginPath();
        networkCtx.moveTo(fromNode.x + 7, fromNode.y);
        networkCtx.lineTo(toNode.x - 7, toNode.y);
        networkCtx.stroke();
      }
    });
  }

  nodePositions.forEach((nodes, layerIndex) => {
    const acts = activations[layerIndex] || [];
    nodes.forEach((node, nodeIndex) => {
      const act = clamp(acts[nodeIndex] || 0, 0, 1);
      const shade = Math.round(245 - act * 170);
      networkCtx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
      networkCtx.strokeStyle = "#2c374a";
      networkCtx.lineWidth = 1;
      networkCtx.beginPath();
      networkCtx.arc(node.x, node.y, 7, 0, Math.PI * 2);
      networkCtx.fill();
      networkCtx.stroke();
    });

    networkCtx.fillStyle = "#5f6572";
    networkCtx.font = "11px Manrope";
    networkCtx.textAlign = "center";
    networkCtx.fillText(layerSpecs[layerIndex].name, nodes[0].x, height - 10);
  });
}

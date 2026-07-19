/**
 * app.js
 * メインエントリーポイント — honda/app.py の UI ロジックを JS に移植
 *
 * 機能:
 *   - 画像アップロード (ファイル選択 / D&D)
 *   - モデル選択・ロード
 *   - 自動マスク生成の実行・進捗表示・キャンセル
 *   - スライダーによるフィルタリング (明度・面積・信頼度)
 *   - キャンバスクリックによる個別除外/復元
 *   - 全除外リセット
 */

import { SAM2 }                           from './sam2.js';
import { decodeMasks, applyPostFilters }  from './automask.js';
import {
  detectServerBackend,
  getAvailableServerModels,
  getServerModelProvider,
  segmentOnServer,
} from './server_api.js';
import {
  DetectedObject,
  DetectionResult,
  computeDerivedProperties,
  computeBbox,
} from './detection.js';
import { drawResults, COLORS } from './visualize.js';

// ---- グローバル状態 ----
const sam2             = new SAM2();
let   currentResult    = null;    // DetectionResult | null
let   rawMasks         = null;    // decodeMasks() の生結果 (再フィルタ用)
let   currentModelName = '';
let   inferenceBackend = 'local'; // 'local' | 'server'
let   serverInfo       = null;
let   manuallyExcluded = new Set();
let   cancelToken      = null;
const objectNotes      = new Map(); // objId → note文字列 (再構築をまたいで保持)

// ---- DOM 参照 ----
const $  = id => document.getElementById(id);
const fileInput        = $('file-input');
const uploadArea       = $('upload-area');
const uploadPlaceholder= $('upload-placeholder');
const inputPreview     = $('input-preview');
const runBtn           = $('run-btn');
const cancelBtn        = $('cancel-btn');
const progressArea     = $('progress-area');
const progressFill     = $('progress-fill');
const progressText     = $('progress-text');
const outputCanvas     = $('output-canvas');
const canvasPlaceholder= $('canvas-placeholder');
const summaryText      = $('summary-text');
const modelStatus      = $('model-status');
const gpuStatus        = $('gpu-status');
const restoreBtn       = $('restore-btn');
const modelOptions     = $('model-options');

// 検出設定
const pointsPerSideSlider = $('points-per-side');
const pointsPerSideVal    = $('points-per-side-val');
const pointsPerSideField  = $('points-per-side-field');
const iouThreshSlider     = $('iou-thresh');
const iouThreshVal        = $('iou-thresh-val');
const minMaskAreaSlider   = $('min-mask-area');
const minMaskAreaVal      = $('min-mask-area-val');
const maxMaskRatioSlider  = $('max-mask-ratio');
const maxMaskRatioVal     = $('max-mask-ratio-val');
const contourWidthSlider  = $('contour-width');
const contourWidthVal     = $('contour-width-val');

// フィルタスライダー
const filterSliders = {
  minBrightness: $('min-brightness'),
  maxBrightness: $('max-brightness'),
  minArea:       $('min-area'),
  maxArea:       $('max-area'),
  minConfidence: $('min-confidence'),
};
const filterVals = {
  minBrightness: $('min-brightness-val'),
  maxBrightness: $('max-brightness-val'),
  minArea:       $('min-area-val'),
  maxArea:       $('max-area-val'),
  minConfidence: $('min-confidence-val'),
};

// ---- 初期化 ----
(async () => {
  const server = await detectServerBackend();
  if (server.available) {
    inferenceBackend = 'server';
    serverInfo = server.info;
    configureModelOptions();
    const provider = serverInfo.provider ?? 'server';
    gpuStatus.textContent = provider.includes('CUDA') ? 'サーバ GPU' : 'サーバ推論';
    gpuStatus.className = 'status-badge server';
    updateModelControls();
  } else {
    configureModelOptions();
    const provider = await sam2.detectExecutionProvider();
    gpuStatus.textContent = provider === 'webgpu' ? 'WebGPU 有効' : 'WASM (CPU)';
    gpuStatus.className = `status-badge ${provider}`;
    if (server.info?.mode === 'server' && server.info.ready === false) {
      modelStatus.textContent = `サーバ推論は未準備: ${server.info.error ?? '依存関係を確認してください'}`;
    }
  }

  // サンプル画像をデフォルトでロード
  loadImageFromUrl('./assets/sample.png');
})();

modelOptions.addEventListener('change', event => {
  if (event.target.matches('input[name="model"]')) updateModelControls();
});

function configureModelOptions() {
  const availableModels = new Set(getAvailableServerModels(serverInfo));
  const labels = [...modelOptions.querySelectorAll('[data-model-key]')];

  for (const label of labels) {
    const serverOnly = label.dataset.serverOnly === 'true';
    label.hidden = inferenceBackend === 'server'
      ? !availableModels.has(label.dataset.modelKey)
      : serverOnly;
  }

  const selected = modelOptions.querySelector('input[name="model"]:checked');
  if (!selected || selected.closest('.radio-label').hidden) {
    const firstAvailable = labels.find(label => !label.hidden)?.querySelector('input[name="model"]');
    if (firstAvailable) firstAvailable.checked = true;
  }
}

function updateModelControls() {
  const modelType = selectedModelType();
  if (!modelType) return;

  pointsPerSideField.hidden = modelType === 'cellpose-cpsam-v2';
  if (inferenceBackend === 'server') {
    const provider = getServerModelProvider(serverInfo, modelType);
    modelStatus.textContent = `${modelDisplayName(modelType)} / サーバ推論 (${provider})`;
  }
}

function selectedModelType() {
  return modelOptions.querySelector('input[name="model"]:checked')?.value ?? null;
}

function modelDisplayName(modelType) {
  const label = modelOptions.querySelector(`[data-model-key="${modelType}"] strong`);
  return label?.textContent ?? modelType;
}

// ---- 画像アップロード ----
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover',  e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', ()=> { uploadArea.classList.remove('drag-over'); });
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file?.type.startsWith('image/')) loadFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

function loadFile(file) {
  loadImageFromUrl(URL.createObjectURL(file));
}

function loadImageFromUrl(url) {
  inputPreview.onload = () => {
    inputPreview.hidden = false;
    uploadPlaceholder.hidden = true;
    runBtn.disabled = false;
    resetResult();
  };
  inputPreview.src = url;
}

function resetResult() {
  currentResult    = null;
  rawMasks         = null;
  currentModelName = '';
  manuallyExcluded = new Set();
  objectNotes.clear();
  outputCanvas.style.display = 'none';
  canvasPlaceholder.style.display = '';
  summaryText.innerHTML = '<span style="color:#555">—</span>';
}

// ---- 検出設定スライダー ----
const rebuildDebounced = debounce(rebuildFromRawMasks, 200);

// range ↔ number input の双方向同期ヘルパー
function linkSlider(range, num, { toNum = v => v, fromNum = v => v, onChange = null } = {}) {
  range.addEventListener('input', () => {
    num.value = toNum(range.value);
    onChange?.();
  });
  num.addEventListener('input', () => {
    const v = parseFloat(num.value);
    if (isNaN(v)) return;
    const clamped = Math.min(parseFloat(num.max), Math.max(parseFloat(num.min), v));
    range.value = fromNum(clamped);
    num.value   = clamped;
    onChange?.();
  });
}

linkSlider(pointsPerSideSlider, pointsPerSideVal);
linkSlider(iouThreshSlider,     iouThreshVal,   { onChange: rebuildDebounced });
linkSlider(minMaskAreaSlider,   minMaskAreaVal, { onChange: rebuildDebounced });
// max-mask-ratio: range=0.001〜0.99、number input=0.1〜99 (%, 0.1刻み)
linkSlider(maxMaskRatioSlider, maxMaskRatioVal, {
  toNum:    v => Math.round(parseFloat(v) * 1000) / 10,  // 0.001 → 0.1%
  fromNum:  v => (v / 100).toFixed(3),                   // 4.0% → 0.040
  onChange: rebuildDebounced,
});
linkSlider(contourWidthSlider, contourWidthVal, { onChange: () => render() });

// ---- 検出実行 ----
runBtn.addEventListener('click', async () => {
  if (!inputPreview.src || inputPreview.hidden) return;

  const modelType = selectedModelType();
  if (!modelType) return;

  // 必要ならモデルをロード
  if (inferenceBackend === 'local' && (sam2.currentModel !== modelType || !sam2.encoderSession)) {
    runBtn.disabled = true;
    modelStatus.textContent = 'モデルをダウンロード中...';
    try {
      await sam2.load(modelType, ({ message }) => {
        modelStatus.textContent = message;
      });
      modelStatus.textContent = `${modelDisplayName(modelType)} 準備完了 (${sam2.executionProvider})`;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error('Model load error:', err);
      modelStatus.textContent = `ロードエラー: ${msg}`;
      runBtn.disabled = false;
      return;
    }
  } else if (inferenceBackend === 'server') {
    const provider = getServerModelProvider(serverInfo, modelType);
    modelStatus.textContent = `${modelDisplayName(modelType)} / サーバ推論 (${provider})`;
  }

  // 状態リセット
  cancelToken      = { cancelled: false };
  manuallyExcluded = new Set();
  currentResult    = null;
  currentModelName = '';

  runBtn.disabled           = true;
  progressArea.hidden       = false;
  progressFill.style.width  = '0%';
  canvasPlaceholder.style.display = 'none';
  outputCanvas.style.display = 'block';

  const decodeOpts = {
    pointsPerSide: parseInt(pointsPerSideSlider.value),
  };

  try {
    let decoded = null;
    if (inferenceBackend === 'server') {
      const serverResult = await segmentOnServer(
        inputPreview,
        { modelType, pointsPerSide: decodeOpts.pointsPerSide },
        updateProgress,
        cancelToken,
      );
      if (serverResult !== null) {
        decoded = serverResult.rawMasks;
        currentModelName = serverResult.modelName;
      }
    } else {
      decoded = await decodeMasks(
        sam2,
        inputPreview,
        decodeOpts,
        updateProgress,
        cancelToken,
      );
      currentModelName = modelDisplayName(modelType);
    }

    if (decoded === null) {
      progressText.textContent = 'キャンセルされました';
      outputCanvas.style.display = 'none';
      canvasPlaceholder.style.display = '';
      return;
    }

    // 生マスクを保持 (IoU/面積/比率スライダー変更時に再利用)
    rawMasks = decoded;

    // フィルタ適用 → DetectionResult 構築
    await rebuildFromRawMasks();

    progressFill.style.width = '100%';
    progressText.textContent = `完了 — ${currentResult?.objects.length ?? 0} 個を検出`;

  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('Detection error:', err);
    progressText.textContent = `エラー: ${msg}`;
  } finally {
    runBtn.disabled  = false;
    cancelToken      = null;
    // 少し待ってプログレスを隠す
    setTimeout(() => { progressArea.hidden = true; }, 2000);
  }
});

cancelBtn.addEventListener('click', () => {
  if (cancelToken) cancelToken.cancelled = true;
});

// ---- フィルタスライダー ----
for (const [key, slider] of Object.entries(filterSliders)) {
  linkSlider(slider, filterVals[key], { onChange: applyFiltersAndRender });
}

// ---- 全除外リセット ----
restoreBtn.addEventListener('click', () => {
  if (!currentResult) return;
  manuallyExcluded = new Set();
  for (const obj of currentResult.objects) obj.excluded = false;
  render(`全除外をリセットしました\n`);
});

// ---- キャンバスクリック → 個別除外/復元 ----
outputCanvas.addEventListener('click', e => {
  if (!currentResult) return;

  const rect   = outputCanvas.getBoundingClientRect();
  const scaleX = currentResult.imageWidth  / rect.width;
  const scaleY = currentResult.imageHeight / rect.height;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top)  * scaleY);

  const hit = currentResult.toggleAtPoint(x, y);
  if (!hit) {
    render(`(${x}, ${y}) に物体なし\n`);
    return;
  }

  // 手動除外セットを現在の excluded 状態に同期 (honda と同じ)
  manuallyExcluded = new Set(currentResult.objects.filter(o => o.excluded).map(o => o.objId));

  const msg = hit.action === 'excluded'
    ? `#${hit.obj.objId} を除外しました\n`
    : `#${hit.obj.objId} を復元しました\n`;
  render(msg);
});

// ---- ヘルパー ----

/**
 * 生マスクにポストフィルタを適用して DetectionResult を再構築する。
 * IoU・最小面積・最大比率スライダー変更時に再検出なしで呼ぶ。
 */
async function rebuildFromRawMasks() {
  if (!rawMasks) return;

  const modelType = selectedModelType();
  const modelName = currentModelName || modelDisplayName(modelType);
  const filtered  = applyPostFilters(rawMasks, {
    predIouThresh: parseFloat(iouThreshSlider.value),
    nmsThresh:     0.70,
    minMaskArea:   parseInt(minMaskAreaSlider.value),
    maxMaskRatio:  parseFloat(maxMaskRatioSlider.value),
  });

  if (filtered.length === 0) {
    summaryText.textContent = '条件に合うオブジェクトがありません。設定を調整してください。';
    outputCanvas.style.display = 'none';
    canvasPlaceholder.style.display = '';
    currentResult = null;
    return;
  }

  const imageData = getImageData(inputPreview);
  const W = imageData.width;
  const H = imageData.height;

  const objects = filtered.map((m, i) => {
    const obj = new DetectedObject({
      objId:      i + 1,
      mask:       m.mask,
      bbox:       computeBbox(m.mask, W, H),
      confidence: m.iou,
    });
    computeDerivedProperties(obj, imageData);
    return obj;
  });

  currentResult = new DetectionResult({
    objects,
    imageData,
    modelName,
  });

  manuallyExcluded = new Set();
  autoAdjustSliders(objects);
  outputCanvas.style.display = 'block';
  canvasPlaceholder.style.display = 'none';
  render();
}

function applyFiltersAndRender() {
  if (!currentResult) return;
  currentResult.applyFilters({
    minBrightness: parseFloat(filterSliders.minBrightness.value),
    maxBrightness: parseFloat(filterSliders.maxBrightness.value),
    minArea:       parseInt(filterSliders.minArea.value),
    maxArea:       parseInt(filterSliders.maxArea.value),
    minConfidence: parseFloat(filterSliders.minConfidence.value),
    manuallyExcluded,
  });
  render();
}

function render(prefix = '') {
  if (!currentResult) return;
  drawResults(outputCanvas, currentResult, true, parseFloat(contourWidthSlider.value));
  buildSummaryDOM(prefix);
}

function buildSummaryDOM(prefix = '') {
  summaryText.innerHTML = '';

  if (!currentResult) {
    summaryText.innerHTML = '<span style="color:#555">—</span>';
    return;
  }

  const active = currentResult.activeObjects.length;
  const total  = currentResult.objects.length;

  // ヘッダー
  const header = document.createElement('div');
  header.className = 'summary-header';
  header.textContent = `${prefix}${currentResult.modelName} | ${active}/${total} 個表示中 (除外: ${total - active})`;
  summaryText.appendChild(header);

  // 表
  const table = document.createElement('table');
  table.className = 'summary-table';

  const thead = table.createTHead();
  const hrow  = thead.insertRow();
  for (const text of ['', '#', 'Conf', 'Area (px)', 'Circ', 'Bright', 'Note']) {
    const th = document.createElement('th');
    th.textContent = text;
    hrow.appendChild(th);
  }

  const tbody = table.createTBody();
  for (const obj of currentResult.objects) {
    const color    = obj.excluded ? [140,140,140] : COLORS[obj.objId % COLORS.length];
    const colorStr = `rgb(${color[0]},${color[1]},${color[2]})`;

    const tr = tbody.insertRow();
    tr.className = 'obj-row' + (obj.excluded ? ' excluded' : '');

    // チェックボックスセル
    const cbCell = tr.insertCell();
    const cb = document.createElement('input');
    cb.type    = 'checkbox';
    cb.checked = !obj.excluded;
    cb.addEventListener('change', () => {
      obj.excluded = !cb.checked;
      obj.excluded ? manuallyExcluded.add(obj.objId) : manuallyExcluded.delete(obj.objId);
      render();
    });
    cbCell.appendChild(cb);

    // カラードット + ID
    const idCell = tr.insertCell();
    const dot = document.createElement('span');
    dot.className        = 'obj-color';
    dot.style.background = colorStr;
    dot.style.display    = 'inline-block';
    dot.style.marginRight = '4px';
    dot.style.verticalAlign = 'middle';
    idCell.appendChild(dot);
    idCell.appendChild(document.createTextNode(`#${obj.objId}`));

    // メトリクス列
    tr.insertCell().textContent = obj.confidence.toFixed(2);
    tr.insertCell().textContent = obj.areaPixels.toLocaleString();
    tr.insertCell().textContent = obj.circularity.toFixed(2);
    tr.insertCell().textContent = Math.round(obj.meanBrightness);

    // Note 列
    const noteCell  = tr.insertCell();
    const noteInput = document.createElement('input');
    noteInput.type        = 'text';
    noteInput.className   = 'note-input';
    noteInput.placeholder = '—';
    noteInput.value       = objectNotes.get(obj.objId) ?? '';
    noteInput.addEventListener('input', () => {
      objectNotes.set(obj.objId, noteInput.value);
    });
    noteInput.addEventListener('click', e => e.stopPropagation()); // 行クリックと干渉しない
    noteCell.appendChild(noteInput);

    // 行クリックでもトグル (チェックボックス・Note以外の部分)
    tr.addEventListener('click', e => {
      if (e.target === cb || e.target === noteInput) return;
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event('change'));
    });
  }

  summaryText.appendChild(table);

  // 凡例
  const legend = document.createElement('div');
  legend.className = 'summary-legend';
  legend.textContent =
    'Conf: SAM2デコーダの信頼度スコア (0〜1) / ' +
    'Area: マスク面積 (ピクセル数) / ' +
    'Circ: 円形度 — 1.0が完全な円、値が低いほど複雑な形状 / ' +
    'Bright: マスク領域の平均輝度 (0〜255)';
  summaryText.appendChild(legend);
}

/**
 * 検出結果からスライダーの min/max/value/step を動的更新する。
 * honda の run_segmentation() と同じロジック。
 */
function autoAdjustSliders(objects) {
  if (!objects.length) return;

  const areas   = objects.map(o => o.areaPixels);
  const brights = objects.map(o => o.meanBrightness);
  const confs   = objects.map(o => o.confidence);

  const areaMax   = Math.max(...areas);
  const brightMax = Math.min(255, Math.max(...brights) + 10);
  const confMin   = Math.min(...confs);
  const confMax   = Math.max(...confs);
  const confMargin = Math.max((confMax - confMin) * 0.05, 0.01);
  const areaUpper  = Math.round(areaMax * 1.1) + 100;
  const areaStep   = Math.max(Math.round(areaUpper / 500), 1);

  setSlider('minBrightness', 0,          brightMax,                   0,                          1);
  setSlider('maxBrightness', 0,          brightMax,                   brightMax,                  1);
  setSlider('minArea',       0,          areaUpper,                   0,                          areaStep);
  setSlider('maxArea',       0,          areaUpper,                   areaUpper,                  areaStep);
  setSlider('minConfidence', Math.max(0, confMin - confMargin),
                             Math.min(1, confMax + confMargin),
                             Math.max(0, confMin - confMargin),       0.01);
}

function setSlider(name, min, max, value, step) {
  const slider = filterSliders[name];
  const numEl  = filterVals[name];
  slider.min = numEl.min = min;
  slider.max = numEl.max = max;
  slider.step = numEl.step = step;
  slider.value = numEl.value = value;
}

/** img 要素から ImageData を取得する。 */
function getImageData(imgEl) {
  const w = imgEl.naturalWidth;
  const h = imgEl.naturalHeight;
  const canvas = Object.assign(document.createElement('canvas'), { width: w, height: h });
  canvas.getContext('2d').drawImage(imgEl, 0, 0);
  return canvas.getContext('2d').getImageData(0, 0, w, h);
}

function yieldToUI() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

function updateProgress({ done, total, message }) {
  progressText.textContent = message;
  if (total > 0) progressFill.style.width = `${(done / total) * 100}%`;
}

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

// ---- パネルリサイズ ----
(function initResizers() {
  const mainLayout   = document.querySelector('.main-layout');
  const controlPanel = document.querySelector('.control-panel');
  const resultPanel  = document.querySelector('.result-panel');
  const canvasContainer = document.querySelector('.canvas-container');
  const summaryArea  = document.querySelector('.summary-area');
  const sepH = $('sep-h'); // 左右セパレーター
  const sepV = $('sep-v'); // 上下セパレーター

  // 左右ドラッグ
  sepH.addEventListener('mousedown', e => {
    e.preventDefault();
    sepH.classList.add('dragging');
    document.body.classList.add('resizing');
    const startX     = e.clientX;
    const startWidth = controlPanel.getBoundingClientRect().width;

    const onMove = e => {
      const newWidth = Math.max(180, Math.min(520, startWidth + e.clientX - startX));
      mainLayout.style.gridTemplateColumns = `${newWidth}px 5px 1fr`;
    };
    const onUp = () => {
      sepH.classList.remove('dragging');
      document.body.classList.remove('resizing');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });

  // 上下ドラッグ
  sepV.addEventListener('mousedown', e => {
    e.preventDefault();
    sepV.classList.add('dragging');
    document.body.classList.add('resizing-v');
    const startY      = e.clientY;
    const startHeight = summaryArea.getBoundingClientRect().height;

    const onMove = e => {
      const delta     = startY - e.clientY; // 上にドラッグ → サマリ拡大
      const newHeight = Math.max(60, Math.min(600, startHeight + delta));
      summaryArea.style.height = `${newHeight}px`;
    };
    const onUp = () => {
      sepV.classList.remove('dragging');
      document.body.classList.remove('resizing-v');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
})();

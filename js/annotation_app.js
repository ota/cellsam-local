import {
  DRAFT_KIND,
  GROUND_TRUTH_KIND,
  annotationFileStem,
  computeMaskBbox,
  countMask,
  documentFromEditable,
  editableObjectsFromDocument,
  validateAnnotationDocument,
} from './annotations.js';

const COLORS = [
  [83, 154, 255], [65, 196, 132], [238, 142, 76], [178, 116, 235],
  [58, 190, 194], [230, 98, 139], [141, 189, 72], [224, 185, 58],
  [102, 133, 220], [210, 104, 88], [77, 174, 115], [193, 130, 194],
];
const STATUS_COLORS = {
  candidate: [242, 190, 73],
  rejected: [212, 93, 93],
};
const STATUS_LABELS = {
  accepted: '承認',
  candidate: '候補',
  rejected: '却下',
};
const MAX_HISTORY = 30;

const $ = id => document.getElementById(id);
const canvas = $('annotation-canvas');
const canvasShell = $('canvas-shell');
const canvasEmpty = $('canvas-empty');
const jsonFileInput = $('json-file-input');
const imageFileInput = $('image-file-input');
const openJsonBtn = $('open-json-btn');
const openImageBtn = $('open-image-btn');
const documentState = $('document-state');
const imageName = $('image-name');
const imageSize = $('image-size');
const sourceModel = $('source-model');
const blindReview = $('blind-review');
const toolSelector = $('tool-selector');
const brushSize = $('brush-size');
const brushSizeValue = $('brush-size-value');
const maskOpacity = $('mask-opacity');
const maskOpacityValue = $('mask-opacity-value');
const showRejected = $('show-rejected');
const undoBtn = $('undo-btn');
const redoBtn = $('redo-btn');
const newObjectBtn = $('new-object-btn');
const acceptBtn = $('accept-btn');
const rejectBtn = $('reject-btn');
const zoomOutBtn = $('zoom-out-btn');
const zoomInBtn = $('zoom-in-btn');
const fitBtn = $('fit-btn');
const zoomValue = $('zoom-value');
const polygonActions = $('polygon-actions');
const finishPolygonBtn = $('finish-polygon-btn');
const cancelPolygonBtn = $('cancel-polygon-btn');
const pointerPosition = $('pointer-position');
const statusMessage = $('status-message');
const objectList = $('object-list');
const objectCount = $('object-count');
const acceptedCount = $('accepted-count');
const candidateCount = $('candidate-count');
const rejectedCount = $('rejected-count');
const objectNotes = $('object-notes');
const saveDraftBtn = $('save-draft-btn');
const saveGroundTruthBtn = $('save-ground-truth-btn');
const saveLabelBtn = $('save-label-btn');

let baseDocument = null;
let objects = [];
let sourceImage = null;
let sourceImageCanvas = null;
let overlayCanvas = null;
let selectedId = null;
let activeTool = 'select';
let dirty = false;
let fitMode = true;
let zoom = 1;
let panX = 0;
let panY = 0;
let pointerImage = null;
let pointerCss = null;
let isPanning = false;
let panStart = null;
let isPainting = false;
let lastPaintPoint = null;
let activeChanges = null;
let activeStatusBefore = null;
let liveStroke = [];
let polygonPoints = [];
let undoStack = [];
let redoStack = [];
let noteBeforeEdit = '';
let currentObjectUrl = null;

openJsonBtn.addEventListener('click', () => jsonFileInput.click());
openImageBtn.addEventListener('click', () => imageFileInput.click());
jsonFileInput.addEventListener('change', () => {
  const file = jsonFileInput.files[0];
  if (file) loadAnnotationFile(file);
  jsonFileInput.value = '';
});
imageFileInput.addEventListener('change', () => {
  const file = imageFileInput.files[0];
  if (file) loadReplacementImage(file);
  imageFileInput.value = '';
});

toolSelector.addEventListener('click', event => {
  const button = event.target.closest('[data-tool]');
  if (!button) return;
  setTool(button.dataset.tool);
});
brushSize.addEventListener('input', () => {
  brushSizeValue.value = `${brushSize.value} px`;
  renderCanvas();
});
maskOpacity.addEventListener('input', () => {
  maskOpacityValue.value = `${Math.round(Number(maskOpacity.value) * 100)}%`;
  rebuildOverlay();
});
showRejected.addEventListener('change', () => {
  rebuildOverlay();
  renderObjectList();
});
blindReview.addEventListener('change', refreshSourceModel);

undoBtn.addEventListener('click', undo);
redoBtn.addEventListener('click', redo);
newObjectBtn.addEventListener('click', addNewObject);
acceptBtn.addEventListener('click', () => setSelectedStatus('accepted'));
rejectBtn.addEventListener('click', () => setSelectedStatus('rejected'));
zoomOutBtn.addEventListener('click', () => zoomAtCenter(zoom / 1.25));
zoomInBtn.addEventListener('click', () => zoomAtCenter(zoom * 1.25));
fitBtn.addEventListener('click', fitImage);
finishPolygonBtn.addEventListener('click', finishPolygon);
cancelPolygonBtn.addEventListener('click', cancelPolygon);
saveDraftBtn.addEventListener('click', () => saveAnnotation(DRAFT_KIND));
saveGroundTruthBtn.addEventListener('click', () => saveAnnotation(GROUND_TRUTH_KIND));
saveLabelBtn.addEventListener('click', saveLabelMask);

objectNotes.addEventListener('focus', () => {
  noteBeforeEdit = selectedObject()?.notes || '';
});
objectNotes.addEventListener('input', () => {
  const object = selectedObject();
  if (object) object.notes = objectNotes.value;
});
objectNotes.addEventListener('change', () => {
  const object = selectedObject();
  if (!object || object.notes === noteBeforeEdit) return;
  recordAction({ type: 'note', id: object.id, before: noteBeforeEdit, after: object.notes });
});

canvas.addEventListener('pointerdown', onPointerDown);
canvas.addEventListener('pointermove', onPointerMove);
canvas.addEventListener('pointerup', onPointerUp);
canvas.addEventListener('pointercancel', onPointerUp);
canvas.addEventListener('pointerleave', () => {
  pointerImage = null;
  pointerCss = null;
  pointerPosition.textContent = 'x: -, y: -';
  renderCanvas();
});
canvas.addEventListener('wheel', onWheel, { passive: false });
canvas.addEventListener('dblclick', event => {
  if (activeTool === 'polygon' && polygonPoints.length >= 3) {
    event.preventDefault();
    finishPolygon();
  }
});

document.addEventListener('keydown', event => {
  const editable = ['INPUT', 'TEXTAREA'].includes(event.target.tagName);
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z' && !editable) {
    event.preventDefault();
    event.shiftKey ? redo() : undo();
  } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y' && !editable) {
    event.preventDefault();
    redo();
  } else if (event.key === 'Escape') {
    cancelPolygon();
  } else if (event.key === 'Enter' && activeTool === 'polygon' && !editable) {
    finishPolygon();
  }
});

window.addEventListener('beforeunload', event => {
  if (!dirty) return;
  event.preventDefault();
  event.returnValue = '';
});

new ResizeObserver(() => {
  resizeCanvas();
  if (fitMode && sourceImage) fitImage();
  else renderCanvas();
}).observe(canvasShell);

async function loadAnnotationFile(file) {
  setMessage('JSONを読込中...');
  try {
    const documentData = JSON.parse(await file.text());
    validateAnnotationDocument(documentData);
    baseDocument = documentData;
    objects = editableObjectsFromDocument(documentData);
    selectedId = objects.find(object => object.status !== 'rejected')?.id ?? objects[0]?.id ?? null;
    undoStack = [];
    redoStack = [];
    dirty = false;
    polygonPoints = [];
    if (documentData.image.embeddedDataUrl) {
      await loadImageSource(documentData.image.embeddedDataUrl, false);
    } else {
      clearSourceImage();
      setMessage('元画像を選択してください');
    }
    rebuildOverlay();
    refreshUI();
    setMessage(`${objects.length} objects`);
  } catch (error) {
    setMessage(error instanceof Error ? error.message : String(error), true);
  }
}

async function loadReplacementImage(file) {
  if (!baseDocument) {
    setMessage('先にannotation JSONを開いてください', true);
    return;
  }
  if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
  currentObjectUrl = URL.createObjectURL(file);
  try {
    await loadImageSource(currentObjectUrl, true);
    baseDocument.image.fileName = file.name;
    refreshUI();
    markDirty();
    setMessage('画像を置き換えました');
  } catch (error) {
    setMessage(error instanceof Error ? error.message : String(error), true);
  }
}

async function loadImageSource(src, updateDocument) {
  const image = new Image();
  image.decoding = 'async';
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error('画像を読み込めませんでした'));
    image.src = src;
  });

  const { width, height } = baseDocument.image;
  if (image.naturalWidth !== width || image.naturalHeight !== height) {
    throw new Error(`画像サイズが一致しません: expected ${width}x${height}`);
  }
  sourceImageCanvas = Object.assign(document.createElement('canvas'), { width, height });
  const context = sourceImageCanvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  const pixels = context.getImageData(0, 0, width, height);
  const digest = await hashImagePixels(pixels);
  if (baseDocument.image.pixelSha256 && digest && digest !== baseDocument.image.pixelSha256) {
    throw new Error('画像のpixel SHA-256がannotation JSONと一致しません');
  }
  if (updateDocument) {
    baseDocument.image.embeddedDataUrl = sourceImageCanvas.toDataURL('image/png');
    baseDocument.image.pixelSha256 = digest;
    baseDocument.image.hashEncoding = digest ? 'sha256-rgba' : null;
  }
  sourceImage = image;
  canvasEmpty.hidden = true;
  resizeCanvas();
  fitImage();
}

function clearSourceImage() {
  sourceImage = null;
  sourceImageCanvas = null;
  overlayCanvas = null;
  canvasEmpty.hidden = false;
  renderCanvas();
}

function resizeCanvas() {
  const rect = canvasShell.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function fitImage() {
  if (!sourceImage) return;
  const rect = canvas.getBoundingClientRect();
  zoom = Math.min(rect.width / sourceImage.naturalWidth, rect.height / sourceImage.naturalHeight) * 0.94;
  panX = (rect.width - sourceImage.naturalWidth * zoom) / 2;
  panY = (rect.height - sourceImage.naturalHeight * zoom) / 2;
  fitMode = true;
  updateZoomValue();
  renderCanvas();
}

function zoomAtCenter(nextZoom) {
  const rect = canvas.getBoundingClientRect();
  setZoomAt(nextZoom, rect.width / 2, rect.height / 2);
}

function setZoomAt(nextZoom, cssX, cssY) {
  if (!sourceImage) return;
  nextZoom = Math.max(0.03, Math.min(16, nextZoom));
  const imageX = (cssX - panX) / zoom;
  const imageY = (cssY - panY) / zoom;
  panX = cssX - imageX * nextZoom;
  panY = cssY - imageY * nextZoom;
  zoom = nextZoom;
  fitMode = false;
  updateZoomValue();
  renderCanvas();
}

function updateZoomValue() {
  zoomValue.value = `${Math.round(zoom * 100)}%`;
}

function rebuildOverlay() {
  if (!baseDocument || !sourceImage) {
    overlayCanvas = null;
    renderCanvas();
    return;
  }
  const { width, height } = baseDocument.image;
  overlayCanvas = Object.assign(document.createElement('canvas'), { width, height });
  const context = overlayCanvas.getContext('2d');
  const imageData = context.createImageData(width, height);
  const alpha = Math.round(Number(maskOpacity.value) * 255);
  for (const object of objects) {
    if (object.status === 'rejected' && !showRejected.checked) continue;
    const color = colorForObject(object);
    const objectAlpha = object.status === 'rejected' ? Math.min(alpha, 90) : alpha;
    for (let index = 0; index < object.mask.length; index++) {
      if (!object.mask[index]) continue;
      const offset = index * 4;
      imageData.data[offset] = color[0];
      imageData.data[offset + 1] = color[1];
      imageData.data[offset + 2] = color[2];
      imageData.data[offset + 3] = objectAlpha;
    }
  }
  context.putImageData(imageData, 0, 0);
  renderCanvas();
}

function renderCanvas() {
  const context = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  if (!sourceImage) return;

  context.save();
  context.translate(panX, panY);
  context.scale(zoom, zoom);
  context.imageSmoothingEnabled = true;
  context.drawImage(sourceImageCanvas, 0, 0);
  if (overlayCanvas) {
    context.imageSmoothingEnabled = false;
    context.drawImage(overlayCanvas, 0, 0);
  }

  const selected = selectedObject();
  if (selected?.bbox && (selected.status !== 'rejected' || showRejected.checked)) {
    const { x1, y1, x2, y2 } = selected.bbox;
    context.strokeStyle = '#ffffff';
    context.lineWidth = 2 / zoom;
    context.setLineDash([7 / zoom, 5 / zoom]);
    context.strokeRect(x1, y1, Math.max(1, x2 - x1 + 1), Math.max(1, y2 - y1 + 1));
    context.setLineDash([]);
  }

  if (polygonPoints.length) {
    context.strokeStyle = '#ffffff';
    context.fillStyle = 'rgba(255,255,255,0.75)';
    context.lineWidth = 2 / zoom;
    context.beginPath();
    context.moveTo(polygonPoints[0].x, polygonPoints[0].y);
    for (const point of polygonPoints.slice(1)) context.lineTo(point.x, point.y);
    if (pointerImage && activeTool === 'polygon') context.lineTo(pointerImage.x, pointerImage.y);
    context.stroke();
    for (const point of polygonPoints) {
      context.beginPath();
      context.arc(point.x, point.y, 3.5 / zoom, 0, Math.PI * 2);
      context.fill();
    }
  }

  if (liveStroke.length) {
    context.strokeStyle = activeTool === 'erase' ? 'rgba(255,90,90,0.85)' : 'rgba(100,230,160,0.85)';
    context.lineWidth = Number(brushSize.value);
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.beginPath();
    context.moveTo(liveStroke[0].x, liveStroke[0].y);
    for (const point of liveStroke.slice(1)) context.lineTo(point.x, point.y);
    context.stroke();
  }

  if (pointerImage && ['brush', 'erase'].includes(activeTool)) {
    context.strokeStyle = activeTool === 'erase' ? '#ff7676' : '#77e3a9';
    context.lineWidth = 1.5 / zoom;
    context.beginPath();
    context.arc(pointerImage.x, pointerImage.y, Number(brushSize.value) / 2, 0, Math.PI * 2);
    context.stroke();
  }
  context.restore();
}

function onPointerDown(event) {
  if (!sourceImage) return;
  const point = eventPoint(event);
  pointerCss = point.css;
  pointerImage = point.image;

  if (activeTool === 'pan' || event.button === 1) {
    event.preventDefault();
    isPanning = true;
    panStart = { x: event.clientX, y: event.clientY, panX, panY };
    canvas.classList.add('panning');
    canvas.setPointerCapture(event.pointerId);
    return;
  }
  if (!insideImage(point.image)) return;

  if (activeTool === 'select') {
    selectObjectAt(point.image.x, point.image.y);
  } else if (activeTool === 'polygon') {
    polygonPoints.push(clampImagePoint(point.image));
    refreshPolygonControls();
    renderCanvas();
  } else if (['brush', 'erase'].includes(activeTool)) {
    const object = selectedObject();
    if (!object) {
      setMessage('編集するオブジェクトを選択してください', true);
      return;
    }
    event.preventDefault();
    isPainting = true;
    activeChanges = new Map();
    activeStatusBefore = object.status;
    lastPaintPoint = clampImagePoint(point.image);
    liveStroke = [lastPaintPoint];
    paintLine(object, lastPaintPoint, lastPaintPoint, activeTool === 'brush' ? 1 : 0);
    canvas.setPointerCapture(event.pointerId);
    renderCanvas();
  }
}

function onPointerMove(event) {
  const point = eventPoint(event);
  pointerCss = point.css;
  pointerImage = point.image;
  if (insideImage(point.image)) {
    pointerPosition.textContent = `x: ${Math.round(point.image.x)}, y: ${Math.round(point.image.y)}`;
  } else {
    pointerPosition.textContent = 'x: -, y: -';
  }

  if (isPanning && panStart) {
    panX = panStart.panX + event.clientX - panStart.x;
    panY = panStart.panY + event.clientY - panStart.y;
    fitMode = false;
  } else if (isPainting && lastPaintPoint) {
    const object = selectedObject();
    const next = clampImagePoint(point.image);
    paintLine(object, lastPaintPoint, next, activeTool === 'brush' ? 1 : 0);
    lastPaintPoint = next;
    liveStroke.push(next);
  }
  renderCanvas();
}

function onPointerUp(event) {
  if (isPanning) {
    isPanning = false;
    panStart = null;
    canvas.classList.remove('panning');
  }
  if (isPainting) finishStroke();
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
}

function onWheel(event) {
  if (!sourceImage) return;
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const factor = Math.exp(-event.deltaY * 0.0015);
  setZoomAt(zoom * factor, event.clientX - rect.left, event.clientY - rect.top);
}

function eventPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const css = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  return {
    css,
    image: { x: (css.x - panX) / zoom, y: (css.y - panY) / zoom },
  };
}

function insideImage(point) {
  return Boolean(
    baseDocument
    && point.x >= 0
    && point.y >= 0
    && point.x < baseDocument.image.width
    && point.y < baseDocument.image.height
  );
}

function clampImagePoint(point) {
  return {
    x: Math.max(0, Math.min(baseDocument.image.width - 1, point.x)),
    y: Math.max(0, Math.min(baseDocument.image.height - 1, point.y)),
  };
}

function selectObjectAt(x, y) {
  const { width } = baseDocument.image;
  const index = Math.floor(y) * width + Math.floor(x);
  const matches = objects.filter(object => {
    if (object.status === 'rejected' && !showRejected.checked) return false;
    return object.mask[index];
  });
  if (!matches.length) {
    selectedId = null;
  } else {
    selectedId = matches.reduce((smallest, object) => (
      object.areaPixels < smallest.areaPixels ? object : smallest
    )).id;
  }
  refreshUI();
  renderCanvas();
}

function paintLine(object, from, to, value) {
  if (!object) return;
  const radius = Number(brushSize.value) / 2;
  const distance = Math.hypot(to.x - from.x, to.y - from.y);
  const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius * 0.45)));
  for (let step = 0; step <= steps; step++) {
    const ratio = step / steps;
    paintCircle(
      object,
      from.x + (to.x - from.x) * ratio,
      from.y + (to.y - from.y) * ratio,
      radius,
      value,
    );
  }
}

function paintCircle(object, cx, cy, radius, value) {
  const { width, height } = baseDocument.image;
  const x0 = Math.max(0, Math.floor(cx - radius));
  const x1 = Math.min(width - 1, Math.ceil(cx + radius));
  const y0 = Math.max(0, Math.floor(cy - radius));
  const y1 = Math.min(height - 1, Math.ceil(cy + radius));
  const radiusSq = radius * radius;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 > radiusSq) continue;
      const index = y * width + x;
      if (object.mask[index] === value) continue;
      if (!activeChanges.has(index)) activeChanges.set(index, object.mask[index]);
      object.mask[index] = value;
    }
  }
}

function finishStroke() {
  const object = selectedObject();
  isPainting = false;
  lastPaintPoint = null;
  liveStroke = [];
  if (!object || !activeChanges?.size) {
    activeChanges = null;
    renderCanvas();
    return;
  }
  const indices = Uint32Array.from(activeChanges.keys());
  const before = Uint8Array.from(activeChanges.values());
  const after = Uint8Array.from(indices, index => object.mask[index]);
  const statusAfter = 'accepted';
  object.status = statusAfter;
  updateObjectGeometry(object);
  recordAction({
    type: 'mask',
    id: object.id,
    indices,
    before,
    after,
    statusBefore: activeStatusBefore,
    statusAfter,
  });
  activeChanges = null;
  rebuildOverlay();
  refreshUI();
}

function finishPolygon() {
  if (!baseDocument || polygonPoints.length < 3) return;
  let object = selectedObject();
  let created = false;
  if (!object || object.status === 'rejected') {
    object = createEmptyObject();
    objects.push(object);
    selectedId = object.id;
    created = true;
  }

  const statusBefore = object.status;
  const changes = rasterizePolygon(object, polygonPoints);
  polygonPoints = [];
  refreshPolygonControls();
  if (!changes.size) {
    if (created) objects = objects.filter(item => item.id !== object.id);
    refreshUI();
    renderCanvas();
    return;
  }
  object.status = 'accepted';
  updateObjectGeometry(object);
  if (created) {
    recordAction({ type: 'add', object: cloneObject(object) });
  } else {
    const indices = Uint32Array.from(changes.keys());
    recordAction({
      type: 'mask',
      id: object.id,
      indices,
      before: Uint8Array.from(changes.values()),
      after: Uint8Array.from(indices, index => object.mask[index]),
      statusBefore,
      statusAfter: 'accepted',
    });
  }
  rebuildOverlay();
  refreshUI();
}

function rasterizePolygon(object, points) {
  const { width, height } = baseDocument.image;
  const x0 = Math.max(0, Math.floor(Math.min(...points.map(point => point.x))));
  const y0 = Math.max(0, Math.floor(Math.min(...points.map(point => point.y))));
  const x1 = Math.min(width - 1, Math.ceil(Math.max(...points.map(point => point.x))));
  const y1 = Math.min(height - 1, Math.ceil(Math.max(...points.map(point => point.y))));
  const temp = Object.assign(document.createElement('canvas'), {
    width: Math.max(1, x1 - x0 + 1),
    height: Math.max(1, y1 - y0 + 1),
  });
  const context = temp.getContext('2d');
  context.fillStyle = '#fff';
  context.beginPath();
  context.moveTo(points[0].x - x0, points[0].y - y0);
  for (const point of points.slice(1)) context.lineTo(point.x - x0, point.y - y0);
  context.closePath();
  context.fill();
  const pixels = context.getImageData(0, 0, temp.width, temp.height).data;
  const changes = new Map();
  for (let localY = 0; localY < temp.height; localY++) {
    for (let localX = 0; localX < temp.width; localX++) {
      if (!pixels[(localY * temp.width + localX) * 4 + 3]) continue;
      const index = (y0 + localY) * width + x0 + localX;
      if (object.mask[index]) continue;
      changes.set(index, object.mask[index]);
      object.mask[index] = 1;
    }
  }
  return changes;
}

function cancelPolygon() {
  if (!polygonPoints.length) return;
  polygonPoints = [];
  refreshPolygonControls();
  renderCanvas();
}

function refreshPolygonControls() {
  polygonActions.hidden = activeTool !== 'polygon';
  finishPolygonBtn.disabled = polygonPoints.length < 3;
}

function setTool(tool) {
  activeTool = tool;
  canvas.dataset.tool = tool;
  for (const button of toolSelector.querySelectorAll('[data-tool]')) {
    button.classList.toggle('active', button.dataset.tool === tool);
  }
  if (tool !== 'polygon') cancelPolygon();
  refreshPolygonControls();
  renderCanvas();
}

function addNewObject() {
  if (!baseDocument) return;
  const object = createEmptyObject();
  objects.push(object);
  selectedId = object.id;
  recordAction({ type: 'add', object: cloneObject(object) });
  setTool('brush');
  rebuildOverlay();
  refreshUI();
}

function createEmptyObject() {
  return {
    id: Math.max(0, ...objects.map(object => object.id)) + 1,
    status: 'accepted',
    confidence: null,
    notes: '',
    bbox: null,
    areaPixels: 0,
    mask: new Uint8Array(baseDocument.image.width * baseDocument.image.height),
  };
}

function setSelectedStatus(status) {
  const object = selectedObject();
  if (!object || object.status === status) return;
  const before = object.status;
  object.status = status;
  recordAction({ type: 'status', id: object.id, before, after: status });
  rebuildOverlay();
  refreshUI();
}

function recordAction(action) {
  undoStack.push(action);
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  redoStack = [];
  markDirty();
  refreshHistoryButtons();
}

function undo() {
  const action = undoStack.pop();
  if (!action) return;
  applyAction(action, true);
  redoStack.push(action);
  markDirty();
  rebuildOverlay();
  refreshUI();
}

function redo() {
  const action = redoStack.pop();
  if (!action) return;
  applyAction(action, false);
  undoStack.push(action);
  markDirty();
  rebuildOverlay();
  refreshUI();
}

function applyAction(action, reverse) {
  if (action.type === 'add') {
    if (reverse) {
      objects = objects.filter(object => object.id !== action.object.id);
      if (selectedId === action.object.id) selectedId = null;
    } else {
      objects.push(cloneObject(action.object));
      selectedId = action.object.id;
    }
    return;
  }
  const object = objects.find(item => item.id === action.id);
  if (!object) return;
  if (action.type === 'status') {
    object.status = reverse ? action.before : action.after;
  } else if (action.type === 'note') {
    object.notes = reverse ? action.before : action.after;
  } else if (action.type === 'mask') {
    const values = reverse ? action.before : action.after;
    for (let index = 0; index < action.indices.length; index++) {
      object.mask[action.indices[index]] = values[index];
    }
    object.status = reverse ? action.statusBefore : action.statusAfter;
    updateObjectGeometry(object);
  }
}

function updateObjectGeometry(object) {
  object.areaPixels = countMask(object.mask);
  object.bbox = computeMaskBbox(
    object.mask,
    baseDocument.image.width,
    baseDocument.image.height,
  );
}

function selectedObject() {
  return objects.find(object => object.id === selectedId) || null;
}

function refreshUI() {
  const counts = countStatuses();
  imageName.textContent = baseDocument?.image.fileName || '-';
  imageSize.textContent = baseDocument ? `${baseDocument.image.width} x ${baseDocument.image.height}` : '-';
  refreshSourceModel();
  objectCount.textContent = String(objects.length);
  acceptedCount.textContent = `${counts.accepted} 承認`;
  candidateCount.textContent = `${counts.candidate} 候補`;
  rejectedCount.textContent = `${counts.rejected} 却下`;
  renderObjectList();
  refreshObjectDetail();
  refreshHistoryButtons();
  const selected = selectedObject();
  acceptBtn.disabled = !selected || selected.status === 'accepted';
  rejectBtn.disabled = !selected || selected.status === 'rejected';
  newObjectBtn.disabled = !baseDocument;
  saveDraftBtn.disabled = !baseDocument || !sourceImage;
  const finalReady = canFinalize();
  saveGroundTruthBtn.disabled = !finalReady;
  saveLabelBtn.disabled = !finalReady;
  refreshDocumentState();
  refreshPolygonControls();
}

function renderObjectList() {
  objectList.innerHTML = '';
  const fragment = document.createDocumentFragment();
  for (const object of objects) {
    if (object.status === 'rejected' && !showRejected.checked) continue;
    const row = document.createElement('button');
    row.type = 'button';
    row.className = `object-row ${object.status}${object.id === selectedId ? ' selected' : ''}`;
    row.addEventListener('click', () => {
      selectedId = object.id;
      refreshUI();
      renderCanvas();
    });
    const swatch = document.createElement('span');
    swatch.className = 'object-swatch';
    swatch.style.background = rgb(colorForObject(object));
    const id = document.createElement('span');
    id.className = 'object-id';
    id.textContent = `#${object.id}`;
    const status = document.createElement('span');
    status.className = `object-status ${object.status}`;
    status.textContent = STATUS_LABELS[object.status];
    const area = document.createElement('span');
    area.className = 'object-area';
    area.textContent = `${object.areaPixels.toLocaleString()} px`;
    row.append(swatch, id, status, area);
    fragment.appendChild(row);
  }
  objectList.appendChild(fragment);
}

function refreshObjectDetail() {
  const object = selectedObject();
  objectNotes.disabled = !object;
  objectNotes.value = object?.notes || '';
}

function refreshHistoryButtons() {
  undoBtn.disabled = undoStack.length === 0;
  redoBtn.disabled = redoStack.length === 0;
}

function refreshSourceModel() {
  if (!baseDocument) {
    sourceModel.textContent = '-';
  } else if (blindReview.checked) {
    sourceModel.textContent = '非表示';
  } else {
    sourceModel.textContent = baseDocument.source?.modelName || baseDocument.source?.modelKey || '-';
  }
}

function refreshDocumentState() {
  documentState.className = 'document-state';
  if (!baseDocument) {
    documentState.textContent = '未読込';
  } else if (dirty) {
    documentState.textContent = '未保存';
    documentState.classList.add('dirty');
  } else {
    documentState.textContent = baseDocument.kind === GROUND_TRUTH_KIND ? '正解読込' : '下書き読込';
    documentState.classList.add('ready');
  }
}

function countStatuses() {
  return objects.reduce((counts, object) => {
    counts[object.status]++;
    return counts;
  }, { accepted: 0, candidate: 0, rejected: 0 });
}

function canFinalize() {
  if (!baseDocument || !sourceImage) return false;
  const counts = countStatuses();
  return counts.candidate === 0
    && counts.accepted > 0
    && objects.filter(object => object.status === 'accepted').every(object => object.areaPixels > 0);
}

function markDirty() {
  dirty = true;
  refreshDocumentState();
}

function colorForObject(object) {
  return STATUS_COLORS[object.status] || COLORS[(object.id - 1) % COLORS.length];
}

function rgb(color) {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

async function saveAnnotation(kind) {
  if (!baseDocument || !sourceImage) return;
  if (kind === GROUND_TRUTH_KIND) {
    const error = finalizationError();
    if (error) {
      setMessage(error, true);
      return;
    }
  }
  try {
    const documentData = documentFromEditable({ baseDocument, objects, kind });
    assignLabelValues(documentData);
    const suffix = kind === GROUND_TRUTH_KIND ? 'ground-truth' : 'annotation-draft';
    downloadBlob(
      JSON.stringify(documentData),
      `${annotationFileStem(documentData.image.fileName)}.${suffix}.json`,
      'application/json',
    );
    baseDocument = documentData;
    dirty = false;
    refreshUI();
    setMessage(kind === GROUND_TRUTH_KIND ? '正解JSONを保存しました' : '下書きJSONを保存しました');
  } catch (error) {
    setMessage(error instanceof Error ? error.message : String(error), true);
  }
}

function assignLabelValues(documentData) {
  let labelValue = 1;
  for (const object of documentData.objects) {
    object.labelValue = object.status === 'accepted' ? labelValue++ : null;
  }
}

function finalizationError() {
  const counts = countStatuses();
  if (counts.candidate) return `${counts.candidate}件の候補が未確認です`;
  const accepted = objects.filter(object => object.status === 'accepted');
  if (!accepted.length) return '承認済みオブジェクトがありません';
  if (accepted.some(object => object.areaPixels === 0)) return '空の承認マスクがあります';
  if (hasAcceptedOverlap(accepted)) return '承認マスク同士が重なっています';
  if (accepted.length > 255) return 'Label PNGは255 objectsまでです';
  return null;
}

function hasAcceptedOverlap(accepted) {
  const occupied = new Uint8Array(baseDocument.image.width * baseDocument.image.height);
  for (const object of accepted) {
    for (let index = 0; index < object.mask.length; index++) {
      if (!object.mask[index]) continue;
      if (occupied[index]) return true;
      occupied[index] = 1;
    }
  }
  return false;
}

function saveLabelMask() {
  const error = finalizationError();
  if (error) {
    setMessage(error, true);
    return;
  }
  const { width, height, fileName } = baseDocument.image;
  const labelCanvas = Object.assign(document.createElement('canvas'), { width, height });
  const context = labelCanvas.getContext('2d');
  const imageData = context.createImageData(width, height);
  for (let index = 0; index < width * height; index++) imageData.data[index * 4 + 3] = 255;
  const accepted = objects.filter(object => object.status === 'accepted');
  accepted.forEach((object, objectIndex) => {
    const label = objectIndex + 1;
    for (let index = 0; index < object.mask.length; index++) {
      if (!object.mask[index]) continue;
      const offset = index * 4;
      imageData.data[offset] = label;
      imageData.data[offset + 1] = label;
      imageData.data[offset + 2] = label;
    }
  });
  context.putImageData(imageData, 0, 0);
  labelCanvas.toBlob(blob => {
    if (!blob) {
      setMessage('Label PNGを生成できませんでした', true);
      return;
    }
    downloadBlob(blob, `${annotationFileStem(fileName)}.labels.png`, 'image/png');
    setMessage('Label PNGを保存しました');
  }, 'image/png');
}

function cloneObject(object) {
  return {
    ...object,
    bbox: object.bbox ? { ...object.bbox } : null,
    mask: object.mask.slice(),
  };
}

async function hashImagePixels(imageData) {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await globalThis.crypto.subtle.digest('SHA-256', imageData.data);
  return [...new Uint8Array(digest)]
    .map(byte => byte.toString(16).padStart(2, '0'))
    .join('');
}

function downloadBlob(content, fileName, contentType) {
  const blob = content instanceof Blob ? content : new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const anchor = Object.assign(document.createElement('a'), { href: url, download: fileName });
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setMessage(message, error = false) {
  statusMessage.textContent = message;
  statusMessage.className = error ? 'error' : '';
}

resizeCanvas();
setTool('select');
refreshUI();

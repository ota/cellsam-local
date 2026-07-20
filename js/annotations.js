/**
 * Versioned annotation interchange format shared by detection and review UIs.
 */

export const ANNOTATION_SCHEMA_VERSION = 1;
export const DRAFT_KIND = 'cellsam-annotation-draft';
export const GROUND_TRUTH_KIND = 'cellsam-ground-truth';
export const OBJECT_STATUSES = new Set(['candidate', 'accepted', 'rejected']);

export function encodeMaskRle(mask) {
  if (!(mask instanceof Uint8Array) && !(mask instanceof Uint8ClampedArray)) {
    throw new TypeError('mask must be a Uint8Array');
  }

  const counts = [];
  let start = -1;
  for (let index = 0; index < mask.length; index++) {
    if (mask[index]) {
      if (start < 0) start = index;
    } else if (start >= 0) {
      counts.push(start, index - start);
      start = -1;
    }
  }
  if (start >= 0) counts.push(start, mask.length - start);
  return { encoding: 'start-length-row-major', counts };
}

export function decodeMaskRle(rle, width, height) {
  const size = validateDimensions(width, height);
  const counts = rle?.counts;
  if (!Array.isArray(counts) || counts.length % 2 !== 0) {
    throw new Error('Annotation mask RLE counts must contain start-length pairs');
  }
  if (rle.encoding && !['start-length', 'start-length-row-major'].includes(rle.encoding)) {
    throw new Error(`Unsupported annotation mask encoding: ${rle.encoding}`);
  }

  const mask = new Uint8Array(size);
  let previousEnd = 0;
  for (let index = 0; index < counts.length; index += 2) {
    const start = Number(counts[index]);
    const length = Number(counts[index + 1]);
    if (!Number.isInteger(start) || !Number.isInteger(length) || start < 0 || length <= 0) {
      throw new Error('Annotation mask RLE contains an invalid span');
    }
    const end = start + length;
    if (start < previousEnd || end > size) {
      throw new Error('Annotation mask RLE spans overlap or exceed image dimensions');
    }
    mask.fill(1, start, end);
    previousEnd = end;
  }
  return mask;
}

export function createDraftAnnotation({
  image,
  source,
  objects,
  createdAt = new Date().toISOString(),
}) {
  const width = Number(image?.width);
  const height = Number(image?.height);
  validateDimensions(width, height);
  if (!Array.isArray(objects)) throw new Error('Annotation objects must be an array');

  const document = {
    schemaVersion: ANNOTATION_SCHEMA_VERSION,
    kind: DRAFT_KIND,
    image: {
      fileName: String(image.fileName || 'image.png'),
      width,
      height,
      pixelSha256: image.pixelSha256 || null,
      hashEncoding: image.pixelSha256 ? 'sha256-rgba' : null,
      embeddedDataUrl: image.embeddedDataUrl || null,
    },
    source: {
      modelKey: source?.modelKey || null,
      modelName: source?.modelName || null,
      backend: source?.backend || null,
      provider: source?.provider || null,
      settings: source?.settings || {},
      createdAt,
    },
    review: {
      status: 'unreviewed',
      updatedAt: createdAt,
      editCount: 0,
    },
    objects: objects.map((object, index) => ({
      id: normalizeObjectId(object.id ?? object.objId, index + 1),
      status: object.status || (object.excluded ? 'rejected' : 'candidate'),
      confidence: finiteOrNull(object.confidence),
      bbox: normalizeBbox(object.bbox),
      areaPixels: Number.isInteger(object.areaPixels) ? object.areaPixels : countMask(object.mask),
      notes: String(object.notes || ''),
      rle: encodeMaskRle(object.mask),
    })),
  };
  validateAnnotationDocument(document);
  return document;
}

export function validateAnnotationDocument(document) {
  if (!document || typeof document !== 'object') {
    throw new Error('Annotation JSON must contain an object');
  }
  if (document.schemaVersion !== ANNOTATION_SCHEMA_VERSION) {
    throw new Error(`Unsupported annotation schema version: ${document.schemaVersion}`);
  }
  if (![DRAFT_KIND, GROUND_TRUTH_KIND].includes(document.kind)) {
    throw new Error(`Unsupported annotation document kind: ${document.kind}`);
  }

  const width = Number(document.image?.width);
  const height = Number(document.image?.height);
  validateDimensions(width, height);
  if (typeof document.image?.fileName !== 'string' || !document.image.fileName) {
    throw new Error('Annotation image fileName is required');
  }
  if (document.image.pixelSha256 != null && !/^[a-f0-9]{64}$/i.test(document.image.pixelSha256)) {
    throw new Error('Annotation image pixelSha256 must be a hexadecimal SHA-256 digest');
  }
  if (
    document.image.embeddedDataUrl != null
    && !/^data:image\/(png|jpeg|webp);base64,/i.test(document.image.embeddedDataUrl)
  ) {
    throw new Error('Annotation embedded image must be a PNG, JPEG, or WebP data URL');
  }
  if (!Array.isArray(document.objects)) throw new Error('Annotation objects must be an array');

  const objectIds = new Set();
  for (const object of document.objects) {
    if (!Number.isInteger(object.id) || object.id <= 0 || objectIds.has(object.id)) {
      throw new Error('Annotation object IDs must be unique positive integers');
    }
    objectIds.add(object.id);
    if (!OBJECT_STATUSES.has(object.status)) {
      throw new Error(`Unsupported annotation object status: ${object.status}`);
    }
    decodeMaskRle(object.rle, width, height);
  }
  return document;
}

export function editableObjectsFromDocument(document) {
  validateAnnotationDocument(document);
  const { width, height } = document.image;
  return document.objects.map(object => ({
    id: object.id,
    status: object.status,
    confidence: finiteOrNull(object.confidence),
    notes: String(object.notes || ''),
    bbox: object.bbox ? { ...object.bbox } : null,
    areaPixels: Number.isInteger(object.areaPixels) ? object.areaPixels : 0,
    mask: decodeMaskRle(object.rle, width, height),
  }));
}

export function documentFromEditable({ baseDocument, objects, kind = DRAFT_KIND, updatedAt = new Date().toISOString() }) {
  validateAnnotationDocument(baseDocument);
  if (![DRAFT_KIND, GROUND_TRUTH_KIND].includes(kind)) {
    throw new Error(`Unsupported annotation document kind: ${kind}`);
  }

  const document = {
    ...baseDocument,
    kind,
    image: { ...baseDocument.image },
    source: { ...baseDocument.source, settings: { ...(baseDocument.source?.settings || {}) } },
    review: {
      ...(baseDocument.review || {}),
      status: kind === GROUND_TRUTH_KIND ? 'approved' : 'in-progress',
      updatedAt,
      editCount: Number(baseDocument.review?.editCount || 0) + 1,
    },
    objects: objects.map(object => ({
      id: object.id,
      status: object.status,
      confidence: finiteOrNull(object.confidence),
      bbox: computeMaskBbox(object.mask, baseDocument.image.width, baseDocument.image.height),
      areaPixels: countMask(object.mask),
      notes: String(object.notes || ''),
      rle: encodeMaskRle(object.mask),
    })),
  };
  validateAnnotationDocument(document);
  return document;
}

export function countMask(mask) {
  let count = 0;
  for (const value of mask || []) count += value ? 1 : 0;
  return count;
}

export function computeMaskBbox(mask, width, height) {
  validateDimensions(width, height);
  let x1 = width;
  let y1 = height;
  let x2 = -1;
  let y2 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask[y * width + x]) continue;
      x1 = Math.min(x1, x);
      y1 = Math.min(y1, y);
      x2 = Math.max(x2, x);
      y2 = Math.max(y2, y);
    }
  }
  return x2 < 0 ? null : { x1, y1, x2, y2 };
}

export function annotationFileStem(fileName) {
  const base = String(fileName || 'image').replace(/\.[^.]+$/, '');
  return base.replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'image';
}

function validateDimensions(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error('Annotation image dimensions must be positive integers');
  }
  const size = width * height;
  if (!Number.isSafeInteger(size)) throw new Error('Annotation image dimensions are too large');
  return size;
}

function normalizeObjectId(value, fallback) {
  const id = Number(value ?? fallback);
  return Number.isInteger(id) && id > 0 ? id : fallback;
}

function normalizeBbox(bbox) {
  if (!bbox) return null;
  const values = ['x1', 'y1', 'x2', 'y2'].map(key => Number(bbox[key]));
  return values.every(Number.isFinite)
    ? { x1: values[0], y1: values[1], x2: values[2], y2: values[3] }
    : null;
}

function finiteOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

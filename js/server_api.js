/**
 * server_api.js
 * Same-origin server inference client for LAN GPU deployments.
 */

const HEALTH_TIMEOUT_MS = 800;

export async function detectServerBackend() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);

  try {
    const resp = await fetch('/api/health', {
      cache: 'no-store',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
    if (!resp.ok) return { available: false };

    const info = await resp.json();
    const isServer = info?.mode === 'server';
    return {
      available: isServer && info.ready !== false,
      info: isServer ? info : null,
    };
  } catch (err) {
    return { available: false, error: err };
  } finally {
    clearTimeout(timer);
  }
}

export function getAvailableServerModels(info) {
  if (!Array.isArray(info?.models)) return [];
  return [...new Set(info.models.filter(model => typeof model === 'string' && model.length > 0))];
}

export function getServerModelProvider(info, model) {
  const experimental = Object.values(info?.experimentalModels ?? {});
  const backend = experimental.find(item => Array.isArray(item?.models) && item.models.includes(model));
  return backend?.provider ?? info?.provider ?? 'remote';
}

export async function segmentOnServer(imgEl, opts = {}, onProgress = null, cancelToken = null) {
  const {
    modelType = 'tiny',
    pointsPerSide = 8,
  } = opts;

  const controller = new AbortController();
  const cancelPoll = setInterval(() => {
    if (cancelToken?.cancelled) controller.abort();
  }, 100);

  try {
    onProgress?.({ done: 0, total: 3, stage: 'encode', message: '送信用画像を準備中...' });
    const imageBlob = await imageElementToBlob(imgEl);
    if (cancelToken?.cancelled) return null;

    const form = new FormData();
    form.append('image', imageBlob, 'image.png');
    form.append('model', modelType);
    form.append('points_per_side', String(pointsPerSide));

    onProgress?.({ done: 1, total: 3, stage: 'upload', message: 'サーバへ画像を送信中...' });
    const resp = await fetch('/api/segment', {
      method: 'POST',
      body: form,
      signal: controller.signal,
    });

    if (cancelToken?.cancelled) return null;
    onProgress?.({ done: 2, total: 3, stage: 'download', message: 'サーバ推論結果を受信中...' });

    const data = await readJson(resp);
    if (!resp.ok) {
      const detail = data?.detail ?? data?.error ?? `Server inference failed (${resp.status})`;
      throw new Error(detail);
    }

    const rawMasks = decodeServerMasks(data);
    onProgress?.({
      done: 3,
      total: 3,
      stage: 'done',
      message: `サーバ推論完了 (${rawMasks.length} masks)`,
    });

    return {
      rawMasks,
      modelName: data.modelName ?? data.model_name ?? `${modelType} (server)`,
    };
  } catch (err) {
    if (err?.name === 'AbortError' || cancelToken?.cancelled) return null;
    throw err;
  } finally {
    clearInterval(cancelPoll);
  }
}

async function readJson(resp) {
  try {
    return await resp.json();
  } catch (_) {
    return null;
  }
}

function imageElementToBlob(imgEl) {
  const canvas = Object.assign(document.createElement('canvas'), {
    width: imgEl.naturalWidth,
    height: imgEl.naturalHeight,
  });
  canvas.getContext('2d').drawImage(imgEl, 0, 0);

  return new Promise((resolve, reject) => {
    canvas.toBlob(blob => {
      if (blob) resolve(blob);
      else reject(new Error('Failed to encode image for server inference'));
    }, 'image/png');
  });
}

export function decodeServerMasks(data) {
  const width = Number(data.width);
  const height = Number(data.height);
  const masks = data.rawMasks ?? data.raw_masks ?? data.masks ?? [];

  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error('Server response is missing image dimensions');
  }
  if (!Array.isArray(masks)) {
    throw new Error('Server response masks must be an array');
  }

  return masks.map(item => ({
    mask: decodeRle(item.rle, width, height),
    iou: Number(item.iou ?? item.confidence ?? 0),
    width,
    height,
  }));
}

function decodeRle(rle, width, height) {
  const counts = rle?.counts;
  if (!Array.isArray(counts)) throw new Error('Server mask is missing RLE counts');

  const mask = new Uint8Array(width * height);
  for (let i = 0; i < counts.length; i += 2) {
    const start = Number(counts[i]);
    const len = Number(counts[i + 1]);
    if (!Number.isFinite(start) || !Number.isFinite(len) || start < 0 || len <= 0) continue;
    if (start >= mask.length) continue;
    mask.fill(1, start, Math.min(start + len, mask.length));
  }
  return mask;
}

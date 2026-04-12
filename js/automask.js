/**
 * automask.js
 * SAM2 自動マスク生成 — SAM2AutomaticMaskGenerator の JS 実装
 *
 * アルゴリズム:
 *   1. 画像をエンコード (1回)
 *   2. pointsPerSide × pointsPerSide のグリッド点を生成
 *   3. 各点でデコーダを実行し IoU スコアが閾値以上のマスクを収集
 *   4. 最小面積フィルタ
 *   5. マスク IoU ベースの NMS で重複除去
 *
 * 速度目安 (WebGPU, SAM2.1-tiny):
 *   エンコード ~2-3s、デコード ~100-200ms/点
 *   16×16=256点 → 合計 ~30-50s
 *   8×8=64点   → 合計  ~8-15s
 */

/**
 * Step 1〜3 のみ実行: 全デコード結果 (生マスク) を返す。
 * IoU しきい値フィルタは後から applyPostFilters() で適用できる。
 *
 * @param {import('./sam2.js').SAM2} sam2
 * @param {HTMLImageElement} imgEl
 * @param {object} [opts]
 * @param {number} [opts.pointsPerSide=16]
 * @param {(info: ProgressInfo) => void} [onProgress]
 * @param {{ cancelled: boolean }} [cancelToken]
 * @returns {Promise<MaskResult[] | null>}  null = キャンセル
 */
export async function decodeMasks(sam2, imgEl, opts = {}, onProgress = null, cancelToken = null) {
  const { pointsPerSide = 16 } = opts;

  const origW = imgEl.naturalWidth;
  const origH = imgEl.naturalHeight;

  onProgress?.({ done: 0, total: 0, stage: 'encode', message: '画像をエンコード中...' });
  const enc = await sam2.encodeImage(imgEl);
  if (cancelToken?.cancelled) return null;

  const points = makeGrid(pointsPerSide, origW, origH);
  const total  = points.length;
  const rawResults = [];

  for (let i = 0; i < total; i++) {
    if (cancelToken?.cancelled) return null;

    const [x, y] = points[i];
    onProgress?.({
      done: i + 1, total, stage: 'decode',
      message: `デコード中 ${i + 1} / ${total} 点...`,
    });

    try {
      const { mask, iou } = await sam2.decodePoint(enc, x, y);
      rawResults.push({ mask, iou, width: origW, height: origH });
    } catch (err) {
      console.warn(`Decoder failed at (${x}, ${y}):`, err);
    }

    if ((i & 7) === 7) await yieldToUI();
  }

  if (cancelToken?.cancelled) return null;
  return rawResults;
}

/**
 * 生マスクに IoU・面積・NMS フィルタを適用して最終マスクを返す。
 * 再検出なしで即時呼び出せる。
 *
 * @param {MaskResult[]} rawMasks
 * @param {object} [opts]
 * @param {number} [opts.predIouThresh=0.70]
 * @param {number} [opts.nmsThresh=0.70]
 * @param {number} [opts.minMaskArea=100]
 * @param {number} [opts.maxMaskRatio=0.30]
 * @returns {MaskResult[]}
 */
export function applyPostFilters(rawMasks, opts = {}) {
  const {
    predIouThresh = 0.96,
    nmsThresh     = 0.70,
    minMaskArea   = 100,
    maxMaskRatio  = 0.04,
  } = opts;

  if (!rawMasks.length) return [];

  const { width: W, height: H } = rawMasks[0];
  const imageArea   = W * H;
  const maxMaskArea = imageArea * maxMaskRatio;

  const filtered = rawMasks.filter(r => {
    if (r.iou < predIouThresh) return false;
    const area = countSet(r.mask);
    return area >= minMaskArea && area <= maxMaskArea;
  });

  return nms(filtered, nmsThresh);
}

// ---- ヘルパー ----

/**
 * pointsPerSide × pointsPerSide の均等グリッド点を生成する。
 * @returns {[number, number][]}
 */
function makeGrid(pointsPerSide, width, height) {
  const pts = [];
  for (let j = 0; j < pointsPerSide; j++) {
    for (let i = 0; i < pointsPerSide; i++) {
      pts.push([
        Math.round((i + 0.5) / pointsPerSide * width),
        Math.round((j + 0.5) / pointsPerSide * height),
      ]);
    }
  }
  return pts;
}

/** Uint8Array のセット数 (=マスク面積 px) を返す。 */
function countSet(mask) {
  let n = 0;
  for (let i = 0; i < mask.length; i++) { if (mask[i]) n++; }
  return n;
}

/**
 * マスク IoU を計算する。
 * @param {Uint8Array} a
 * @param {Uint8Array} b
 * @returns {number} 0..1
 */
function maskIou(a, b) {
  let inter = 0, union = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i] ? 1 : 0;
    const bv = b[i] ? 1 : 0;
    inter += av & bv;
    union += av | bv;
  }
  return union === 0 ? 0 : inter / union;
}

/**
 * Greedy NMS: IoU スコア降順でソートし、重複マスクを除去する。
 * @param {MaskResult[]} results
 * @param {number} iouThresh
 * @returns {MaskResult[]}
 */
function nms(results, iouThresh) {
  const sorted     = [...results].sort((a, b) => b.iou - a.iou);
  const keep       = [];
  const suppressed = new Uint8Array(sorted.length); // 0=keep, 1=suppress

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed[i]) continue;
    keep.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed[j]) continue;
      if (maskIou(sorted[i].mask, sorted[j].mask) > iouThresh) {
        suppressed[j] = 1;
      }
    }
  }
  return keep;
}

/** 次のマイクロタスクまで制御を返す。 */
function yieldToUI() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

/**
 * @typedef {{ mask: Uint8Array, iou: number, width: number, height: number }} MaskResult
 * @typedef {{ done: number, total: number, stage: string, message: string }} ProgressInfo
 */

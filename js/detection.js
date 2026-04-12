/**
 * detection.js
 * データ構造とフィルタリングロジック — honda/core/schema.py の JS 移植
 */

export class DetectedObject {
  /**
   * @param {object} p
   * @param {number}     p.objId       - 1-based ID
   * @param {Uint8Array} p.mask        - binary mask, length = W * H (1=foreground)
   * @param {{x1,y1,x2,y2}} p.bbox
   * @param {number}     p.confidence  - IoU score from SAM2 decoder
   */
  constructor({ objId, mask, bbox, confidence }) {
    this.objId      = objId;
    this.mask       = mask;
    this.bbox       = bbox;
    this.confidence = confidence;

    // 派生プロパティ (computeDerivedProperties で算出)
    this.areaPixels    = 0;
    this.perimeter     = 0;
    this.circularity   = 0;
    this.meanBrightness = 0;

    // フィルタリング状態
    this.excluded = false;
  }
}

export class DetectionResult {
  /**
   * @param {object} p
   * @param {DetectedObject[]} p.objects
   * @param {ImageData}        p.imageData
   * @param {string}           p.modelName
   */
  constructor({ objects = [], imageData = null, modelName = '' }) {
    this.objects   = objects;
    this.imageData = imageData;  // RGBA ImageData
    this.modelName = modelName;
  }

  get imageWidth()  { return this.imageData?.width  ?? 0; }
  get imageHeight() { return this.imageData?.height ?? 0; }

  get activeObjects() {
    return this.objects.filter(o => !o.excluded);
  }

  /**
   * (x, y) にある最小面積オブジェクトの除外状態をトグルする。
   * 非除外物体を優先(除外)、なければ除外済みを復元。
   * @returns {{ action: 'excluded'|'restored', obj: DetectedObject } | null}
   */
  toggleAtPoint(x, y) {
    const W = this.imageWidth;
    const H = this.imageHeight;
    if (x < 0 || x >= W || y < 0 || y >= H) return null;
    const idx = y * W + x;

    const activeHits = this.objects.filter(o => !o.excluded && o.mask[idx]);
    if (activeHits.length > 0) {
      const target = activeHits.reduce((a, b) => a.areaPixels < b.areaPixels ? a : b);
      target.excluded = true;
      return { action: 'excluded', obj: target };
    }

    const excludedHits = this.objects.filter(o => o.excluded && o.mask[idx]);
    if (excludedHits.length > 0) {
      const target = excludedHits.reduce((a, b) => a.areaPixels < b.areaPixels ? a : b);
      target.excluded = false;
      return { action: 'restored', obj: target };
    }
    return null;
  }

  /**
   * スライダー閾値に基づいて excluded フラグを更新する。
   * manuallyExcluded に含まれる objId は常に除外。
   */
  applyFilters({ minBrightness, maxBrightness, minArea, maxArea, minConfidence, manuallyExcluded = new Set() }) {
    for (const obj of this.objects) {
      if (manuallyExcluded.has(obj.objId)) {
        obj.excluded = true;
        continue;
      }
      obj.excluded = (
        obj.meanBrightness < minBrightness ||
        obj.meanBrightness > maxBrightness ||
        obj.areaPixels     < minArea       ||
        obj.areaPixels     > maxArea       ||
        obj.confidence     < minConfidence
      );
    }
  }
}

/**
 * mask と ImageData から面積・明度・円形度を算出して obj に格納。
 * honda/core/schema.py の compute_derived_properties 相当。
 *
 * @param {DetectedObject} obj
 * @param {ImageData}      imageData - RGBA
 */
export function computeDerivedProperties(obj, imageData) {
  const { data, width: W, height: H } = imageData;
  const mask = obj.mask;

  let area     = 0;
  let sumBright = 0;

  for (let i = 0, n = W * H; i < n; i++) {
    if (!mask[i]) continue;
    area++;
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    sumBright += (r + g + b) / 3;
  }

  obj.areaPixels    = area;
  obj.meanBrightness = area > 0 ? sumBright / area : 0;

  // 周囲長: 4近傍に背景ピクセルを持つマスクピクセルをカウント
  let border = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!mask[i]) continue;
      if (
        x === 0      || !mask[i - 1] ||
        x === W - 1  || !mask[i + 1] ||
        y === 0      || !mask[i - W] ||
        y === H - 1  || !mask[i + W]
      ) {
        border++;
      }
    }
  }
  obj.perimeter   = border;
  obj.circularity = border > 0 ? (4 * Math.PI * area) / (border * border) : 0;
}

/**
 * マスクのバウンディングボックスを計算する。
 * @param {Uint8Array} mask
 * @param {number} W
 * @param {number} H
 * @returns {{x1, y1, x2, y2}}
 */
export function computeBbox(mask, W, H) {
  let x1 = W, y1 = H, x2 = 0, y2 = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (!mask[y * W + x]) continue;
      if (x < x1) x1 = x;
      if (x > x2) x2 = x;
      if (y < y1) y1 = y;
      if (y > y2) y2 = y;
    }
  }
  return { x1, y1, x2, y2 };
}

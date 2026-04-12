/**
 * detection.test.js
 * Node.js 組み込みテストランナー (node:test) を使用
 * 実行: node --test test/detection.test.js
 *
 * テスト対象: detection.js のフィルタリング・クリック除外ロジック
 * ブラウザ API 不要 — ImageData の最小スタブのみ使用
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

// ---- ImageData スタブ (ブラウザ API の代替) ----
class ImageData {
  constructor(width, height) {
    this.width  = width;
    this.height = height;
    this.data   = new Uint8ClampedArray(width * height * 4).fill(128); // グレー画像
  }
}
global.ImageData = ImageData;

// detection.js は ESM なので dynamic import で読み込む
const {
  DetectedObject,
  DetectionResult,
  computeDerivedProperties,
  computeBbox,
} = await import('../js/detection.js');

// ---- テスト用ヘルパー ----

/** W×H の真っ黒マスクを作り、指定矩形内だけ 1 にする */
function makeRectMask(W, H, rx1, ry1, rx2, ry2) {
  const mask = new Uint8Array(W * H);
  for (let y = ry1; y < ry2; y++)
    for (let x = rx1; x < rx2; x++)
      mask[y * W + x] = 1;
  return mask;
}

/** テスト用 DetectionResult を組み立てる */
function makeResult(W, H, objectDefs) {
  const imageData = new ImageData(W, H);
  const objects = objectDefs.map(({ id, rx1, ry1, rx2, ry2, confidence = 0.9 }) => {
    const mask = makeRectMask(W, H, rx1, ry1, rx2, ry2);
    const obj  = new DetectedObject({
      objId: id,
      mask,
      bbox: { x1: rx1, y1: ry1, x2: rx2, y2: ry2 },
      confidence,
    });
    computeDerivedProperties(obj, imageData);
    return obj;
  });
  return new DetectionResult({ objects, imageData, modelName: 'test' });
}

// ============================================================
describe('computeBbox', () => {
  test('矩形マスクの bbox が正しい', () => {
    const W = 100, H = 100;
    const mask = makeRectMask(W, H, 10, 20, 40, 60);
    const bbox = computeBbox(mask, W, H);
    assert.equal(bbox.x1, 10);
    assert.equal(bbox.y1, 20);
    assert.equal(bbox.x2, 39);
    assert.equal(bbox.y2, 59);
  });
});

// ============================================================
describe('computeDerivedProperties', () => {
  test('面積が正しく計算される', () => {
    const W = 100, H = 100;
    const imageData = new ImageData(W, H);
    const mask = makeRectMask(W, H, 10, 10, 30, 30); // 20×20 = 400px
    const obj  = new DetectedObject({ objId: 1, mask, bbox: { x1:10,y1:10,x2:30,y2:30 }, confidence: 0.9 });
    computeDerivedProperties(obj, imageData);
    assert.equal(obj.areaPixels, 400);
  });

  test('円形度: 正方形は 1 未満になる', () => {
    const W = 200, H = 200;
    const imageData = new ImageData(W, H);
    const mask = makeRectMask(W, H, 50, 50, 150, 150); // 100×100 正方形
    const obj  = new DetectedObject({ objId: 1, mask, bbox: {x1:50,y1:50,x2:150,y2:150}, confidence: 0.9 });
    computeDerivedProperties(obj, imageData);
    // 正方形の理論値: 4π*10000/(400^2) ≈ 0.785
    assert.ok(obj.circularity > 0 && obj.circularity <= 1.0,
      `circularity should be (0, 1], got ${obj.circularity}`);
  });
});

// ============================================================
describe('DetectionResult.applyFilters', () => {
  test('面積フィルタが機能する', () => {
    const result = makeResult(200, 200, [
      { id: 1, rx1:  0, ry1:  0, rx2: 10, ry2: 10 }, // 100px — 小さい
      { id: 2, rx1: 50, ry1: 50, rx2: 100, ry2: 100 }, // 2500px — 大きい
    ]);

    result.applyFilters({ minBrightness:0, maxBrightness:255, minArea:500, maxArea:9999, minConfidence:0 });

    assert.equal(result.objects[0].excluded, true,  '#1 (100px) は除外されるべき');
    assert.equal(result.objects[1].excluded, false, '#2 (2500px) は残るべき');
  });

  test('明度フィルタが機能する', () => {
    // ImageData.data はデフォルト 128 (グレー) → meanBrightness ≈ 128
    const result = makeResult(200, 200, [
      { id: 1, rx1: 10, ry1: 10, rx2: 50, ry2: 50 },
    ]);
    result.applyFilters({ minBrightness:150, maxBrightness:255, minArea:0, maxArea:999999, minConfidence:0 });
    assert.equal(result.objects[0].excluded, true, '明度 128 は 150 以上フィルタで除外されるべき');
  });

  test('信頼度フィルタが機能する', () => {
    const result = makeResult(200, 200, [
      { id: 1, rx1: 10, ry1: 10, rx2: 50, ry2: 50, confidence: 0.5 },
    ]);
    result.applyFilters({ minBrightness:0, maxBrightness:255, minArea:0, maxArea:999999, minConfidence:0.8 });
    assert.equal(result.objects[0].excluded, true, 'conf=0.5 は 0.8 フィルタで除外されるべき');
  });

  test('手動除外は常に除外される', () => {
    const result = makeResult(200, 200, [
      { id: 1, rx1: 10, ry1: 10, rx2: 50, ry2: 50 },
    ]);
    // フィルタ条件は全て通過するが、手動除外セットに入っている
    result.applyFilters({
      minBrightness:0, maxBrightness:255, minArea:0, maxArea:999999, minConfidence:0,
      manuallyExcluded: new Set([1]),
    });
    assert.equal(result.objects[0].excluded, true, '手動除外は常に除外されるべき');
  });
});

// ============================================================
describe('DetectionResult.toggleAtPoint', () => {
  test('オブジェクト上をクリックすると除外される', () => {
    const result = makeResult(100, 100, [
      { id: 1, rx1: 10, ry1: 10, rx2: 50, ry2: 50 },
    ]);
    const hit = result.toggleAtPoint(20, 20); // マスク内
    assert.ok(hit, 'ヒットするべき');
    assert.equal(hit.action, 'excluded');
    assert.equal(result.objects[0].excluded, true);
  });

  test('除外済みオブジェクト上をクリックすると復元される', () => {
    const result = makeResult(100, 100, [
      { id: 1, rx1: 10, ry1: 10, rx2: 50, ry2: 50 },
    ]);
    result.objects[0].excluded = true;
    const hit = result.toggleAtPoint(20, 20);
    assert.equal(hit.action, 'restored');
    assert.equal(result.objects[0].excluded, false);
  });

  test('マスク外クリックは null を返す', () => {
    const result = makeResult(100, 100, [
      { id: 1, rx1: 10, ry1: 10, rx2: 50, ry2: 50 },
    ]);
    const hit = result.toggleAtPoint(80, 80); // マスク外
    assert.equal(hit, null);
  });

  test('重なったオブジェクトでは面積最小が優先される', () => {
    const result = makeResult(200, 200, [
      { id: 1, rx1:  0, ry1:  0, rx2: 100, ry2: 100 }, // 大きい
      { id: 2, rx1: 10, ry1: 10, rx2:  30, ry2:  30 }, // 小さい (内側に重なる)
    ]);
    const hit = result.toggleAtPoint(20, 20); // 両マスクが重なる座標
    assert.equal(hit.obj.objId, 2, '面積最小 (#2) が選ばれるべき');
  });
});

// ============================================================
describe('DetectionResult.activeObjects', () => {
  test('excluded=false のオブジェクトだけ返す', () => {
    const result = makeResult(100, 100, [
      { id: 1, rx1: 0,  ry1: 0,  rx2: 10, ry2: 10 },
      { id: 2, rx1: 20, ry1: 20, rx2: 30, ry2: 30 },
    ]);
    result.objects[0].excluded = true;
    assert.equal(result.activeObjects.length, 1);
    assert.equal(result.activeObjects[0].objId, 2);
  });
});

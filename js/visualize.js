/**
 * visualize.js
 * Canvas への検出結果描画
 *
 * 描画内容:
 *   - 元画像
 *   - マスク輪郭線 (塗りつぶしなし / 除外物体は破線)
 *   - ラベル (#ID / area + circularity)
 */

export const COLORS = [
  [255,  80,  80], [ 80, 200,  80], [ 80,  80, 255], [230, 200,  40],
  [220,  60, 220], [ 40, 210, 210], [180,  60,  60], [ 60, 150,  60],
  [ 60,  60, 180], [170, 160,  40], [160,  50, 160], [ 50, 160, 160],
  [255, 150,  40], [255,  50, 150], [140, 230,  40], [ 40, 230, 140],
  [ 40, 150, 255], [150,  40, 255], [ 90, 210,  40], [210,  90,  40],
];
const EXCLUDED_COLOR = [140, 140, 140];

/**
 * 検出結果を canvas に描画する。
 * @param {HTMLCanvasElement} canvas
 * @param {import('./detection.js').DetectionResult} result
 * @param {boolean} showExcluded
 */
export function drawResults(canvas, result, showExcluded = true, lineWidth = 2.5) {
  const { imageData, imageWidth: W, imageHeight: H, objects } = result;
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  ctx.putImageData(imageData, 0, 0);

  const toRender = showExcluded ? objects : objects.filter(o => !o.excluded);
  if (toRender.length === 0) return;

  for (const obj of toRender) {
    const color    = obj.excluded ? EXCLUDED_COLOR : COLORS[obj.objId % COLORS.length];
    const colorStr = `rgb(${color[0]},${color[1]},${color[2]})`;
    drawContour(ctx, obj.mask, W, H, colorStr, obj.excluded, lineWidth);
    drawLabel(ctx, obj, color, W, H);
  }
}

// ---- 輪郭抽出・スムージング ----

/**
 * Moore 近傍境界トレース → 輪郭点列を返す。
 * 連結成分ごとに独立したリストを返す。
 * @returns {Array<{x,y}[]>}
 */
function traceContours(mask, W, H) {
  // 境界ピクセル = マスク内かつ4近傍に背景ピクセルを持つ
  const isBorder = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!mask[i]) continue;
      if (
        x === 0      || !mask[i - 1] ||
        x === W - 1  || !mask[i + 1] ||
        y === 0      || !mask[i - W] ||
        y === H - 1  || !mask[i + W]
      ) isBorder[i] = 1;
    }
  }

  // 8近傍の走査順 (時計回り)
  const dx8 = [ 1, 1, 0,-1,-1,-1, 0, 1];
  const dy8 = [ 0, 1, 1, 1, 0,-1,-1,-1];

  const visited = new Uint8Array(W * H);
  const contours = [];

  for (let sy = 0; sy < H; sy++) {
    for (let sx = 0; sx < W; sx++) {
      const si = sy * W + sx;
      if (!isBorder[si] || visited[si]) continue;

      // この境界ピクセルを起点にトレース
      const pts = [];
      let cx = sx, cy = sy;
      let dir = 0; // 前回の進行方向 (逆から探索再開)

      do {
        visited[cy * W + cx] = 1;
        pts.push({ x: cx, y: cy });

        // 前回方向の逆から時計回りに次の境界ピクセルを探す
        const startDir = (dir + 5) % 8;
        let found = false;
        for (let d = 0; d < 8; d++) {
          const nd = (startDir + d) % 8;
          const nx = cx + dx8[nd];
          const ny = cy + dy8[nd];
          if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
          const ni = ny * W + nx;
          if (isBorder[ni] && !visited[ni]) {
            dir = nd;
            cx  = nx;
            cy  = ny;
            found = true;
            break;
          }
        }
        if (!found) break;
      } while (cx !== sx || cy !== sy);

      if (pts.length >= 4) contours.push(pts);
    }
  }
  return contours;
}

/**
 * 点列を Catmull-Rom スプラインで補間した Canvas Path を描く。
 * 点が少ない場合は直線 polyline にフォールバック。
 */
function drawSmoothedPath(ctx, pts) {
  if (pts.length < 2) return;
  if (pts.length < 6) {
    // 点が少ない場合は直線
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    return;
  }

  // Catmull-Rom → Bezier 変換 (tension=0.5)
  const n   = pts.length;
  const ten = 0.5;
  ctx.moveTo(pts[0].x, pts[0].y);

  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n];
    const p1 = pts[i];
    const p2 = pts[(i + 1) % n];
    const p3 = pts[(i + 2) % n];

    const cp1x = p1.x + (p2.x - p0.x) * ten / 3;
    const cp1y = p1.y + (p2.y - p0.y) * ten / 3;
    const cp2x = p2.x - (p3.x - p1.x) * ten / 3;
    const cp2y = p2.y - (p3.y - p1.y) * ten / 3;

    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
  }
  ctx.closePath();
}

/**
 * 点列を間引く (描画負荷を下げつつ形状を保持)。
 * nth 点に1つだけ残す。
 */
function thinPoints(pts, nth) {
  if (nth <= 1) return pts;
  return pts.filter((_, i) => i % nth === 0);
}

/**
 * マスクの輪郭をスムーズな曲線として描画する。
 */
function drawContour(ctx, mask, W, H, colorStr, dashed, lineWidth = 2.5) {
  const contours = traceContours(mask, W, H);
  if (contours.length === 0) return;

  ctx.save();
  ctx.strokeStyle = colorStr;
  ctx.lineWidth   = lineWidth;
  ctx.lineJoin    = 'round';
  ctx.lineCap     = 'round';
  ctx.setLineDash(dashed ? [6, 4] : []);
  ctx.beginPath();

  for (const pts of contours) {
    // 点数が多い輪郭は間引いて滑らかさと速度を両立
    const step    = Math.max(1, Math.floor(pts.length / 300));
    const thinned = thinPoints(pts, step);
    drawSmoothedPath(ctx, thinned);
  }

  ctx.stroke();
  ctx.restore();
}

// ---- ラベル描画 ----

/**
 * 輝度に基づいてテキスト色 (黒 or 白) を返す。
 * @param {number[]} rgb
 * @returns {'#000'|'#fff'}
 */
function contrastColor(rgb) {
  // 相対輝度 (ITU-R BT.709)
  const L = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  return L > 160 ? '#000' : '#fff';
}

/**
 * バウンディングボックス付近にラベルを描画する。
 * 背景: オブジェクト色 (半透明)
 * テキスト: 輝度に応じて黒 / 白を自動選択
 */
function drawLabel(ctx, obj, colorRgb, W, H) {
  const { x1, y1 } = obj.bbox;
  const line1 = `#${obj.objId}`;
  const line2 = `A=${obj.areaPixels} C=${obj.circularity.toFixed(2)}`;

  const fontSize = 11;
  ctx.font = `bold ${fontSize}px monospace`;

  const tw = Math.max(ctx.measureText(line1).width, ctx.measureText(line2).width);
  const lh = fontSize + 3;
  const bh = lh * 2 + 6;
  const bw = tw + 8;

  const lx = Math.max(0, Math.min(x1, W - bw - 1));
  const ly = Math.max(0, y1 > bh ? y1 - bh : y1 + 2);

  // 背景: オブジェクト色 + 不透明度 0.85
  ctx.globalAlpha = 0.85;
  ctx.fillStyle = `rgb(${colorRgb[0]},${colorRgb[1]},${colorRgb[2]})`;
  ctx.fillRect(lx, ly, bw, bh);
  ctx.globalAlpha = 1.0;

  // テキスト: 輝度自動判定
  ctx.fillStyle = contrastColor(colorRgb);
  ctx.fillText(line1, lx + 4, ly + lh);
  ctx.fillText(line2, lx + 4, ly + lh * 2 + 2);
}

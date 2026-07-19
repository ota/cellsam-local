/**
 * sam2.js
 * SAM2.1 ONNX Runtime Web ラッパー
 *
 * 使用モデル (SharpAI / HuggingFace):
 *   tiny  : SharpAI/sam2-hiera-tiny-onnx  (~155 MB total)
 *   small : SharpAI/sam2-hiera-small-onnx (~184 MB total)
 *
 * エンコーダ入力:
 *   image : float32 [1, 3, 1024, 1024]  — ImageNet 正規化済み NCHW
 *
 * エンコーダ出力:
 *   image_embed      : float32 [1, 256, 64, 64]
 *   high_res_feats_0 : float32 [1, 32, 256, 256]
 *   high_res_feats_1 : float32 [1, 64, 128, 128]
 *
 * デコーダ入力:
 *   image_embed, high_res_feats_0, high_res_feats_1
 *   point_coords    : float32 [1, N, 2]  — 1024px スケールでの座標
 *   point_labels    : float32 [1, N]     — 1=前景, 0=背景, -1=パディング
 *   mask_input      : float32 [1, 1, 256, 256]  — 前回マスク (なければ zeros)
 *   has_mask_input  : float32 [1]               — 0=なし, 1=あり
 *
 * デコーダ出力:
 *   masks           : float32 [1, 3, 256, 256]  — ロジット (sigmoid 前)
 *   iou_predictions : float32 [1, 3]             — 各マスクの IoU スコア
 */

const HF_BASE = 'https://huggingface.co';
const ORT_VERSION = '1.19.2';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
let ortRuntime = globalThis.ort ?? null;
let ortLoadPromise = null;

// WebGPU 用: .ort (pre-optimized)、WASM 用: .onnx
const MODEL_URLS = {
  tiny: {
    encoderWebgpu: `${HF_BASE}/SharpAI/sam2-hiera-tiny-onnx/resolve/main/encoder.with_runtime_opt.ort`,
    encoderWasm:   `${HF_BASE}/SharpAI/sam2-hiera-tiny-onnx/resolve/main/encoder.onnx`,
    decoder:       `${HF_BASE}/SharpAI/sam2-hiera-tiny-onnx/resolve/main/decoder.onnx`,
  },
  small: {
    encoderWebgpu: `${HF_BASE}/SharpAI/sam2-hiera-small-onnx/resolve/main/encoder.with_runtime_opt.ort`,
    encoderWasm:   `${HF_BASE}/SharpAI/sam2-hiera-small-onnx/resolve/main/encoder.onnx`,
    decoder:       `${HF_BASE}/SharpAI/sam2-hiera-small-onnx/resolve/main/decoder.onnx`,
  },
  'base-plus': {
    encoderWebgpu: `${HF_BASE}/SharpAI/sam2-hiera-base-plus-onnx/resolve/main/encoder.with_runtime_opt.ort`,
    encoderWasm:   `${HF_BASE}/SharpAI/sam2-hiera-base-plus-onnx/resolve/main/encoder.onnx`,
    decoder:       `${HF_BASE}/SharpAI/sam2-hiera-base-plus-onnx/resolve/main/decoder.onnx`,
  },
  large: {
    encoderWebgpu: `${HF_BASE}/SharpAI/sam2-hiera-large-onnx/resolve/main/encoder.with_runtime_opt.ort`,
    encoderWasm:   `${HF_BASE}/SharpAI/sam2-hiera-large-onnx/resolve/main/encoder.onnx`,
    decoder:       `${HF_BASE}/SharpAI/sam2-hiera-large-onnx/resolve/main/decoder.onnx`,
  },
};

// ImageNet 正規化定数 (RGB)
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

// SAM2 の固定入力サイズ
const SAM2_SIZE = 1024;

// マスクデコーダの低解像度出力サイズ
const MASK_LOWRES = 256;

export class SAM2 {
  constructor() {
    this.encoderSession = null;
    this.decoderSession = null;
    this.currentModel   = null;
    this.executionProvider = 'wasm';
  }

  /** WebGPU が使えるか確認し、executionProvider を設定する。 */
  async detectExecutionProvider() {
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          this.executionProvider = 'webgpu';
          return 'webgpu';
        }
      } catch (_) { /* fall through */ }
    }
    this.executionProvider = 'wasm';
    return 'wasm';
  }

  /**
   * モデルをロードする。既にロード済みなら何もしない。
   * @param {'tiny'|'small'} modelType
   * @param {(info: {message: string, progress: number}) => void} [onProgress]
   */
  async load(modelType = 'tiny', onProgress = null) {
    if (this.currentModel === modelType && this.encoderSession && this.decoderSession) return;

    const urls = MODEL_URLS[modelType];
    if (!urls) throw new Error(`Unknown model type: ${modelType}`);
    await ensureOnnxRuntime();

    const useWebgpu = this.executionProvider === 'webgpu';

    // WebGPU: .ort ファイル + graphOptimizationLevel disabled (必須)
    // WASM  : .onnx ファイル
    const encoderUrl = useWebgpu ? urls.encoderWebgpu : urls.encoderWasm;

    onProgress?.({ message: 'エンコーダをダウンロード中...', progress: 0 });
    const encBuf = await fetchWithProgress(encoderUrl, p =>
      onProgress?.({ message: `エンコーダ ${(p * 100).toFixed(0)}%`, progress: p * 0.5 })
    );

    onProgress?.({ message: 'デコーダをダウンロード中...', progress: 0.5 });
    const decBuf = await fetchWithProgress(urls.decoder, p =>
      onProgress?.({ message: `デコーダ ${(p * 100).toFixed(0)}%`, progress: 0.5 + p * 0.4 })
    );

    onProgress?.({ message: 'モデルを初期化中...', progress: 0.9 });

    // ArrayBuffer は ORT 内部で転送(detach)される場合があるため slice でコピー
    const encBytes = () => new Uint8Array(encBuf.slice(0));
    const decBytes = () => new Uint8Array(decBuf.slice(0));

    if (useWebgpu) {
      const gpuOpts = {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'disabled',  // .ort ファイルには必須
        enableCpuMemArena: false,
        enableMemPattern: false,
      };
      try {
        this.encoderSession = await ortRuntime.InferenceSession.create(encBytes(), gpuOpts);
        this.decoderSession = await ortRuntime.InferenceSession.create(decBytes(), gpuOpts);
      } catch (e) {
        console.warn('WebGPU session failed, falling back to wasm:', e);
        // WASM フォールバック時は .onnx を再ダウンロード
        onProgress?.({ message: 'WASM でリトライ中...', progress: 0.9 });
        const encBufWasm = await fetchWithProgress(urls.encoderWasm, () => {});
        const wasmOpts = { executionProviders: ['wasm'] };
        this.encoderSession = await ortRuntime.InferenceSession.create(new Uint8Array(encBufWasm), wasmOpts);
        this.decoderSession = await ortRuntime.InferenceSession.create(decBytes(), wasmOpts);
        this.executionProvider = 'wasm';
      }
    } else {
      const wasmOpts = { executionProviders: ['wasm'] };
      this.encoderSession = await ortRuntime.InferenceSession.create(encBytes(), wasmOpts);
      this.decoderSession = await ortRuntime.InferenceSession.create(decBytes(), wasmOpts);
    }

    this.currentModel = modelType;

    onProgress?.({ message: 'ロード完了', progress: 1.0 });
  }

  /**
   * 画像をエンコードして埋め込みを返す。
   * @param {HTMLImageElement} imgEl - 完全にロード済みの img 要素
   * @returns {EncoderResult}
   */
  async encodeImage(imgEl) {
    const prep = preprocessImage(imgEl);
    const output = await this.encoderSession.run({ image: prep.tensor });
    return { ...prep, output };
  }

  /**
   * 1点プロンプトでマスクをデコードする。
   * @param {EncoderResult} enc - encodeImage の戻り値
   * @param {number} x - 元画像座標 (px)
   * @param {number} y - 元画像座標 (px)
   * @returns {{ mask: Uint8Array, iou: number }}
   *          mask: origW * origH の binary 配列 (1=前景)
   */
  async decodePoint(enc, x, y) {
    const { output, scale, padX, padY, origW, origH } = enc;

    // 元画像座標 → SAM2 入力空間 (1024px) 座標に変換
    const sx = x * scale + padX;
    const sy = y * scale + padY;

    // SAM2 デコーダはパディング点 (label=-1, coords=(0,0)) を必ず要求する。
    // これがないと全画像を単一マスクとして出力する。
    const feeds = {
      ...buildDecoderFeeds(output),
      point_coords:   new ortRuntime.Tensor('float32', new Float32Array([sx, sy, 0, 0]), [1, 2, 2]),
      point_labels:   new ortRuntime.Tensor('float32', new Float32Array([1, -1]),        [1, 2]),
      mask_input:     new ortRuntime.Tensor('float32', new Float32Array(MASK_LOWRES * MASK_LOWRES), [1, 1, MASK_LOWRES, MASK_LOWRES]),
      has_mask_input: new ortRuntime.Tensor('float32', new Float32Array([0]),            [1]),
    };

    const result = await this.decoderSession.run(feeds);

    // 出力テンソル名の揺れに対応
    const iouData   = (result.iou_predictions ?? result.iou_pred).data;
    const maskData  = (result.masks ?? result.low_res_masks).data;
    const numMasks  = iouData.length; // 通常 3

    // 最も IoU スコアが高いマスクを選択
    let bestIdx = 0;
    for (let i = 1; i < numMasks; i++) {
      if (iouData[i] > iouData[bestIdx]) bestIdx = i;
    }
    const bestIou = iouData[bestIdx];

    // 低解像度マスク (256x256 ロジット) → 元画像サイズ (origW x origH) に変換
    const offset   = bestIdx * MASK_LOWRES * MASK_LOWRES;
    const rawMask  = maskData.slice(offset, offset + MASK_LOWRES * MASK_LOWRES);
    const fullMask = upsampleAndCrop(rawMask, scale, padX, padY, origW, origH);

    return { mask: fullMask, iou: bestIou };
  }
}

async function ensureOnnxRuntime() {
  if (ortRuntime) return ortRuntime;
  if (ortLoadPromise) return ortLoadPromise;
  if (typeof document === 'undefined') {
    throw new Error('ONNX Runtime Web can only be loaded in a browser');
  }

  ortLoadPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = `${ORT_BASE}ort.all.min.js`;
    script.async = true;
    script.onload = () => {
      ortRuntime = globalThis.ort ?? null;
      if (!ortRuntime) {
        reject(new Error('ONNX Runtime Web loaded without exposing its runtime'));
        return;
      }
      ortRuntime.env.wasm.wasmPaths = ORT_BASE;
      resolve(ortRuntime);
    };
    script.onerror = () => reject(new Error('Failed to load ONNX Runtime Web'));
    document.head.appendChild(script);
  });

  try {
    return await ortLoadPromise;
  } catch (error) {
    ortLoadPromise = null;
    throw error;
  }
}

// ---- 内部ヘルパー ----

/**
 * 進捗コールバック付きフェッチ。ArrayBuffer を返す。
 * @param {string} url
 * @param {(progress: number) => void} onProgress - 0..1
 * @returns {Promise<ArrayBuffer>}
 */
async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Download failed (${resp.status}): ${url}`);

  const total    = parseInt(resp.headers.get('Content-Length') ?? '0', 10);
  const reader   = resp.body.getReader();
  const chunks   = [];
  let received   = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) onProgress(received / total);
  }

  // チャンクを結合
  const merged = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.length; }
  return merged.buffer;
}

/**
 * img 要素を SAM2 入力テンソル (1, 3, 1024, 1024) に変換する。
 * アスペクト比を保ちながら短辺を 1024 に fit し、黒でパディングする。
 *
 * @param {HTMLImageElement} imgEl
 * @returns {{ tensor, scale, padX, padY, origW, origH }}
 */
function preprocessImage(imgEl) {
  const origW = imgEl.naturalWidth;
  const origH = imgEl.naturalHeight;

  const scale  = Math.min(SAM2_SIZE / origW, SAM2_SIZE / origH);
  const scaledW = Math.round(origW * scale);
  const scaledH = Math.round(origH * scale);
  const padX   = Math.floor((SAM2_SIZE - scaledW) / 2);
  const padY   = Math.floor((SAM2_SIZE - scaledH) / 2);

  // 一時 canvas で RGBA ピクセルを取得
  const canvas = Object.assign(document.createElement('canvas'), { width: SAM2_SIZE, height: SAM2_SIZE });
  const ctx    = canvas.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, SAM2_SIZE, SAM2_SIZE);
  ctx.drawImage(imgEl, padX, padY, scaledW, scaledH);
  const { data } = ctx.getImageData(0, 0, SAM2_SIZE, SAM2_SIZE);

  // NHWC RGBA → NCHW float32 + ImageNet 正規化
  const n      = SAM2_SIZE * SAM2_SIZE;
  const tensor = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    tensor[i]         = (data[i * 4]     / 255 - MEAN[0]) / STD[0]; // R
    tensor[n + i]     = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1]; // G
    tensor[2 * n + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2]; // B
  }

  return {
    tensor: new ortRuntime.Tensor('float32', tensor, [1, 3, SAM2_SIZE, SAM2_SIZE]),
    scale, padX, padY, origW, origH,
  };
}

/**
 * エンコーダ出力をデコーダの入力フィールドにマッピングする。
 * SharpAI の出力名に合わせている。
 */
function buildDecoderFeeds(encoderOutput) {
  return {
    image_embed:      encoderOutput.image_embed      ?? encoderOutput.image_embeddings,
    high_res_feats_0: encoderOutput.high_res_feats_0,
    high_res_feats_1: encoderOutput.high_res_feats_1,
  };
}

/**
 * 低解像度マスク (MASK_LOWRES x MASK_LOWRES、ロジット) を
 * 元画像サイズ (origW x origH) の binary Uint8Array に変換する。
 *
 * 処理:
 *   1. ロジットをバイリニア補間しながら元画像サイズへ逆マッピング
 *      (PyTorch の bilinear upsample 相当 → 滑らかな境界)
 *   2. パディング領域を除去して元画像サイズにクロップ
 *   3. logit > 0 で 2値化 (sigmoid > 0.5 と等価)
 *
 * @param {Float32Array} raw  - MASK_LOWRES * MASK_LOWRES ロジット
 * @returns {Uint8Array}      - origW * origH binary (0/1)
 */
function upsampleAndCrop(raw, scale, padX, padY, origW, origH) {
  const result = new Uint8Array(origW * origH);

  // ロジットマップ上での 1px あたりの移動量
  const stepX = MASK_LOWRES / SAM2_SIZE;
  const stepY = MASK_LOWRES / SAM2_SIZE;

  for (let y = 0; y < origH; y++) {
    for (let x = 0; x < origW; x++) {
      // 元画像座標 → SAM2 入力空間 (1024px) 座標
      const sx = x * scale + padX;
      const sy = y * scale + padY;
      if (sx < 0 || sx >= SAM2_SIZE || sy < 0 || sy >= SAM2_SIZE) continue;

      // SAM2 入力座標 → ロジットマップ上の連続座標
      const mx = sx * stepX;
      const my = sy * stepY;

      // バイリニア補間: 4近傍ロジット値を双線形補間
      const mx0 = Math.floor(mx);
      const my0 = Math.floor(my);
      const mx1 = Math.min(mx0 + 1, MASK_LOWRES - 1);
      const my1 = Math.min(my0 + 1, MASK_LOWRES - 1);
      const fx  = mx - mx0;
      const fy  = my - my0;

      const logit =
        raw[my0 * MASK_LOWRES + mx0] * (1 - fx) * (1 - fy) +
        raw[my0 * MASK_LOWRES + mx1] *      fx  * (1 - fy) +
        raw[my1 * MASK_LOWRES + mx0] * (1 - fx) *      fy  +
        raw[my1 * MASK_LOWRES + mx1] *      fx  *      fy;

      result[y * origW + x] = logit > 0 ? 1 : 0;
    }
  }
  return result;
}

/**
 * @typedef {{ output: Object, scale: number, padX: number, padY: number, origW: number, origH: number }} EncoderResult
 */

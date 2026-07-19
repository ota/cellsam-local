from __future__ import annotations

import os
import tempfile
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path


HF_BASE = "https://huggingface.co"

MODEL_URLS = {
  "tiny": {
    "encoder": f"{HF_BASE}/SharpAI/sam2-hiera-tiny-onnx/resolve/main/encoder.onnx",
    "decoder": f"{HF_BASE}/SharpAI/sam2-hiera-tiny-onnx/resolve/main/decoder.onnx",
  },
  "small": {
    "encoder": f"{HF_BASE}/SharpAI/sam2-hiera-small-onnx/resolve/main/encoder.onnx",
    "decoder": f"{HF_BASE}/SharpAI/sam2-hiera-small-onnx/resolve/main/decoder.onnx",
  },
  "base-plus": {
    "encoder": f"{HF_BASE}/SharpAI/sam2-hiera-base-plus-onnx/resolve/main/encoder.onnx",
    "decoder": f"{HF_BASE}/SharpAI/sam2-hiera-base-plus-onnx/resolve/main/decoder.onnx",
  },
  "large": {
    "encoder": f"{HF_BASE}/SharpAI/sam2-hiera-large-onnx/resolve/main/encoder.onnx",
    "decoder": f"{HF_BASE}/SharpAI/sam2-hiera-large-onnx/resolve/main/decoder.onnx",
  },
}

MODEL_LABELS = {
  "tiny": "SAM2.1-Tiny",
  "small": "SAM2.1-Small",
  "base-plus": "SAM2.1-Base+",
  "large": "SAM2.1-Large",
}

SAM2_SIZE = 1024
MASK_LOWRES = 256
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass
class ModelSessions:
  encoder: object
  decoder: object
  provider: str


class Sam2OnnxSegmenter:
  def __init__(self, cache_dir: str | Path | None = None):
    default_cache = Path.home() / ".cache" / "cellsam-local" / "models"
    self.cache_dir = Path(cache_dir or os.getenv("CELLSAM_MODEL_CACHE", default_cache))
    self.sessions: dict[str, ModelSessions] = {}
    self._ort = None
    self._np = None
    self._image_cls = None

  def health(self) -> dict:
    ort, _, _ = self._deps()
    providers = ort.get_available_providers()
    provider = _preferred_providers(providers)[0]
    return {
      "provider": provider,
      "availableProviders": providers,
      "models": list(MODEL_URLS.keys()),
    }

  def segment_image_bytes(self, content: bytes, model: str, points_per_side: int) -> dict:
    if not content:
      raise ValueError("Uploaded image is empty")

    ort, np, image_cls = self._deps()
    sessions = self._load_model(model, ort)

    image = image_cls.open(BytesIO(content)).convert("RGB")
    prep = self._preprocess_image(image, np)
    encoder_output = _named_outputs(sessions.encoder, sessions.encoder.run(None, {"image": prep["tensor"]}))

    raw_masks = []
    failures = 0
    last_error = None
    for x, y in _make_grid(points_per_side, prep["orig_w"], prep["orig_h"]):
      try:
        mask, iou = self._decode_point(sessions.decoder, encoder_output, prep, x, y, np, image_cls)
      except Exception as exc:
        failures += 1
        last_error = exc
        continue
      raw_masks.append({
        "iou": float(iou),
        "rle": _encode_rle(mask, np),
      })

    if not raw_masks and failures > 0:
      raise RuntimeError(f"Decoder failed for all prompt points: {last_error}")

    return {
      "modelName": f"{MODEL_LABELS[model]} (server {sessions.provider})",
      "width": prep["orig_w"],
      "height": prep["orig_h"],
      "rawMasks": raw_masks,
      "provider": sessions.provider,
    }

  def _deps(self):
    if self._ort is not None:
      return self._ort, self._np, self._image_cls

    try:
      import numpy as np
      import onnxruntime as ort
      from PIL import Image
    except ImportError as exc:
      raise RuntimeError(
        "Server inference dependencies are not installed. "
        "Install server/requirements.txt on the GPU server."
      ) from exc

    self._ort = ort
    self._np = np
    self._image_cls = Image
    return self._ort, self._np, self._image_cls

  def _load_model(self, model: str, ort) -> ModelSessions:
    if model in self.sessions:
      return self.sessions[model]

    paths = self._ensure_model_files(model)
    available = ort.get_available_providers()
    providers = _preferred_providers(available)
    opts = ort.SessionOptions()
    encoder = ort.InferenceSession(str(paths["encoder"]), sess_options=opts, providers=providers)
    decoder = ort.InferenceSession(str(paths["decoder"]), sess_options=opts, providers=providers)
    provider = encoder.get_providers()[0]

    sessions = ModelSessions(encoder=encoder, decoder=decoder, provider=provider)
    self.sessions[model] = sessions
    return sessions

  def _ensure_model_files(self, model: str) -> dict[str, Path]:
    model_dir = self.cache_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    paths = {
      "encoder": model_dir / "encoder.onnx",
      "decoder": model_dir / "decoder.onnx",
    }
    for key, path in paths.items():
      if not path.exists():
        _download_atomic(MODEL_URLS[model][key], path)
    return paths

  def _preprocess_image(self, image, np) -> dict:
    orig_w, orig_h = image.size
    scale = min(SAM2_SIZE / orig_w, SAM2_SIZE / orig_h)
    scaled_w = round(orig_w * scale)
    scaled_h = round(orig_h * scale)
    pad_x = (SAM2_SIZE - scaled_w) // 2
    pad_y = (SAM2_SIZE - scaled_h) // 2

    canvas = self._image_cls.new("RGB", (SAM2_SIZE, SAM2_SIZE), (0, 0, 0))
    canvas.paste(image.resize((scaled_w, scaled_h), self._image_cls.Resampling.BILINEAR), (pad_x, pad_y))

    arr = np.asarray(canvas).astype("float32") / 255.0
    mean = np.asarray(MEAN, dtype="float32").reshape(1, 1, 3)
    std = np.asarray(STD, dtype="float32").reshape(1, 1, 3)
    arr = (arr - mean) / std
    tensor = np.transpose(arr, (2, 0, 1))[None, :, :, :].astype("float32")

    return {
      "tensor": tensor,
      "scale": scale,
      "pad_x": pad_x,
      "pad_y": pad_y,
      "scaled_w": scaled_w,
      "scaled_h": scaled_h,
      "orig_w": orig_w,
      "orig_h": orig_h,
    }

  def _decode_point(self, decoder, encoder_output, prep, x: int, y: int, np, image_cls):
    sx = x * prep["scale"] + prep["pad_x"]
    sy = y * prep["scale"] + prep["pad_y"]
    image_embed = encoder_output.get("image_embed")
    if image_embed is None:
      image_embed = encoder_output.get("image_embeddings")

    feeds = {
      "image_embed": image_embed,
      "high_res_feats_0": encoder_output["high_res_feats_0"],
      "high_res_feats_1": encoder_output["high_res_feats_1"],
      "point_coords": np.asarray([[[sx, sy], [0, 0]]], dtype="float32"),
      "point_labels": np.asarray([[1, -1]], dtype="float32"),
      "mask_input": np.zeros((1, 1, MASK_LOWRES, MASK_LOWRES), dtype="float32"),
      "has_mask_input": np.asarray([0], dtype="float32"),
    }

    output = _named_outputs(decoder, decoder.run(None, feeds))
    iou_data = output.get("iou_predictions")
    if iou_data is None:
      iou_data = output.get("iou_pred")
    mask_data = output.get("masks")
    if mask_data is None:
      mask_data = output.get("low_res_masks")

    iou_flat = iou_data.reshape(-1)
    best_idx = int(iou_flat.argmax())
    raw = mask_data.reshape(-1, MASK_LOWRES, MASK_LOWRES)[best_idx]
    mask = _upsample_and_crop(raw, prep, np, image_cls)
    return mask, float(iou_flat[best_idx])


def _preferred_providers(available: list[str]) -> list[str]:
  if "CUDAExecutionProvider" in available:
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]
  return ["CPUExecutionProvider"]


def _download_atomic(url: str, dest: Path):
  dest.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent) as tmp:
    tmp_path = Path(tmp.name)
  try:
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(dest)
  finally:
    if tmp_path.exists():
      tmp_path.unlink()


def _named_outputs(session, values) -> dict:
  return {meta.name: value for meta, value in zip(session.get_outputs(), values)}


def _make_grid(points_per_side: int, width: int, height: int) -> list[tuple[int, int]]:
  points = []
  for j in range(points_per_side):
    for i in range(points_per_side):
      points.append((
        round((i + 0.5) / points_per_side * width),
        round((j + 0.5) / points_per_side * height),
      ))
  return points


def _upsample_and_crop(raw, prep: dict, np, image_cls):
  logit = image_cls.fromarray(raw.astype("float32"), mode="F")
  full = logit.resize((SAM2_SIZE, SAM2_SIZE), image_cls.Resampling.BILINEAR)
  crop = full.crop((
    prep["pad_x"],
    prep["pad_y"],
    prep["pad_x"] + prep["scaled_w"],
    prep["pad_y"] + prep["scaled_h"],
  ))
  restored = crop.resize((prep["orig_w"], prep["orig_h"]), image_cls.Resampling.BILINEAR)
  return np.asarray(restored) > 0


def _encode_rle(mask, np) -> dict:
  flat = np.asarray(mask, dtype="uint8").reshape(-1)
  indices = np.flatnonzero(flat)
  if len(indices) == 0:
    return {"counts": [], "encoding": "start-length"}

  counts = []
  start = int(indices[0])
  prev = start
  for idx in indices[1:]:
    idx = int(idx)
    if idx == prev + 1:
      prev = idx
      continue
    counts.extend([start, prev - start + 1])
    start = prev = idx
  counts.extend([start, prev - start + 1])
  return {"counts": counts, "encoding": "start-length"}

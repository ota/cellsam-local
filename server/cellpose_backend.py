from __future__ import annotations

import importlib.util
import os

from .image_utils import decode_rgb_image


CELLPOSE_MODELS = {
  "cellpose-cpsam-v2": {
    "pretrained": "cpsam_v2",
    "label": "Cellpose-SAM v2",
  },
}


def cellpose_health() -> dict:
  cellpose_available = importlib.util.find_spec("cellpose") is not None
  torch_available = importlib.util.find_spec("torch") is not None
  provider = "unavailable"
  error = None
  if torch_available:
    try:
      import torch
      provider = "PyTorch CUDA" if torch.cuda.is_available() else "PyTorch CPU"
    except Exception as exc:
      torch_available = False
      error = f"torch import failed: {exc}"
      provider = f"torch import failed: {exc}"

  result = {
    "available": cellpose_available and torch_available,
    "torchAvailable": torch_available,
    "provider": provider,
    "models": list(CELLPOSE_MODELS.keys()),
    "modelCache": os.getenv("CELLPOSE_LOCAL_MODELS_PATH"),
    "license": (
      "Cellpose code and mouseland/cellpose-sam model card are BSD-3-Clause; "
      "upstream README notes CC-BY-NC training data."
    ),
    "install": (
      "uv venv .venv-cellpose && "
      "uv pip install --python .venv-cellpose/bin/python -r server/requirements-cellpose.txt"
    ),
  }
  if error:
    result["error"] = error
  return result


class CellposeBackend:
  def __init__(self):
    self._models = {}

  def segment_image_bytes(self, content: bytes, model_key: str) -> dict:
    if not content:
      raise ValueError("Uploaded image is empty")
    if model_key not in CELLPOSE_MODELS:
      raise ValueError(f"Unknown Cellpose model: {model_key}")

    np, image_cls, image_ops = self._deps()
    image = decode_rgb_image(content, image_cls, image_ops)
    arr = np.asarray(image)

    model = self._load_model(model_key)
    masks = first_eval_output(model.eval(arr, **self._eval_kwargs()))
    label_mask = normalize_label_mask(masks, np)

    raw_masks = []
    for label in np.unique(label_mask):
      label = int(label)
      if label == 0:
        continue
      raw_masks.append({
        "iou": 1.0,
        "rle": encode_rle(label_mask == label, np),
      })

    return {
      "modelName": f"{CELLPOSE_MODELS[model_key]['label']} (server {self._provider()})",
      "width": image.width,
      "height": image.height,
      "rawMasks": raw_masks,
      "provider": self._provider(),
    }

  def _load_model(self, model_key: str):
    if model_key in self._models:
      return self._models[model_key]

    if importlib.util.find_spec("cellpose") is None:
      raise RuntimeError(
        "Cellpose backend is not installed. "
        "Install it with: uv venv .venv-cellpose && "
        "uv pip install --python .venv-cellpose/bin/python -r server/requirements-cellpose.txt"
      )

    from cellpose import models

    model = models.CellposeModel(
      gpu=self._torch_cuda_available(),
      pretrained_model=CELLPOSE_MODELS[model_key]["pretrained"],
    )
    self._models[model_key] = model
    return model

  def _eval_kwargs(self) -> dict:
    kwargs = {
      "batch_size": int(os.getenv("CELLPOSE_BATCH_SIZE", "1")),
      "channel_axis": -1,
      "flow_threshold": float(os.getenv("CELLPOSE_FLOW_THRESHOLD", "0.4")),
      "cellprob_threshold": float(os.getenv("CELLPOSE_CELLPROB_THRESHOLD", "0.0")),
    }
    diameter = os.getenv("CELLPOSE_DIAMETER")
    if diameter:
      kwargs["diameter"] = float(diameter)
    return kwargs

  def _provider(self) -> str:
    return "PyTorch CUDA" if self._torch_cuda_available() else "PyTorch CPU"

  def _torch_cuda_available(self) -> bool:
    if importlib.util.find_spec("torch") is None:
      return False
    try:
      import torch
      return bool(torch.cuda.is_available())
    except Exception:
      return False

  def _deps(self):
    try:
      import numpy as np
      from PIL import Image, ImageOps
    except ImportError as exc:
      raise RuntimeError(
        "Cellpose backend requires numpy and Pillow from server/requirements-cellpose.txt"
      ) from exc
    return np, Image, ImageOps


def first_eval_output(result):
  if isinstance(result, tuple):
    result = result[0]
  if isinstance(result, list):
    return result[0]
  return result


def normalize_label_mask(mask, np):
  arr = np.asarray(mask)
  arr = np.squeeze(arr)
  if arr.ndim != 2:
    raise RuntimeError(f"Cellpose returned an unsupported mask shape: {arr.shape}")
  if arr.dtype.kind not in {"u", "i"}:
    arr = arr.astype("uint32")
  return arr


def encode_rle(mask, np) -> dict:
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

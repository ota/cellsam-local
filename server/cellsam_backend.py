from __future__ import annotations

import importlib.util
import os
from io import BytesIO
from pathlib import Path


def cellsam_health() -> dict:
  cell_sam_available = importlib.util.find_spec("cellSAM") is not None
  torch_available = importlib.util.find_spec("torch") is not None
  provider = "unavailable"
  if torch_available:
    try:
      import torch
      provider = "PyTorch CUDA" if torch.cuda.is_available() else "PyTorch CPU"
    except Exception as exc:
      provider = f"torch import failed: {exc}"

  return {
    "available": cell_sam_available,
    "torchAvailable": torch_available,
    "provider": provider,
    "accessTokenPresent": bool(os.getenv("DEEPCELL_ACCESS_TOKEN")),
    "modelPathPresent": bool(os.getenv("CELLSAM_MODEL_PATH")),
    "license": "CellSAM code is Apache-2.0; official pretrained weights are modified Apache for non-commercial academic use only.",
    "install": "uv pip install --python .venv/bin/python -r server/requirements-cellsam.txt",
  }


class CellSamBackend:
  def __init__(self):
    self._pipeline = None

  def segment_image_bytes(self, content: bytes) -> dict:
    if not content:
      raise ValueError("Uploaded image is empty")

    pipeline = self._load_pipeline()
    np, image_cls = self._deps()

    image = image_cls.open(BytesIO(content)).convert("RGB")
    arr = np.asarray(image)
    kwargs = self._pipeline_kwargs()
    mask = pipeline(arr, **kwargs)
    label_mask = normalize_label_mask(mask, np)

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
      "modelName": f"CellSAM (server {self._provider()})",
      "width": image.width,
      "height": image.height,
      "rawMasks": raw_masks,
      "provider": self._provider(),
    }

  def _load_pipeline(self):
    if self._pipeline is not None:
      return self._pipeline

    if importlib.util.find_spec("cellSAM") is None:
      raise RuntimeError(
        "CellSAM backend is not installed. "
        "Install it with: uv pip install --python .venv/bin/python -r server/requirements-cellsam.txt"
      )
    if not os.getenv("CELLSAM_MODEL_PATH") and not os.getenv("DEEPCELL_ACCESS_TOKEN"):
      raise RuntimeError(
        "CellSAM official weights require DEEPCELL_ACCESS_TOKEN from users.deepcell.org. "
        "Alternatively set CELLSAM_MODEL_PATH to a local weights file."
      )

    from cellSAM import cellsam_pipeline

    self._pipeline = cellsam_pipeline
    return self._pipeline

  def _pipeline_kwargs(self) -> dict:
    kwargs = {
      "use_wsi": env_bool("CELLSAM_USE_WSI", False),
      "low_contrast_enhancement": env_bool("CELLSAM_LOW_CONTRAST_ENHANCEMENT", False),
      "gauge_cell_size": env_bool("CELLSAM_GAUGE_CELL_SIZE", False),
      "bbox_threshold": float(os.getenv("CELLSAM_BBOX_THRESHOLD", "0.4")),
    }
    model_path = os.getenv("CELLSAM_MODEL_PATH")
    if model_path:
      kwargs["model_path"] = Path(model_path)
    return kwargs

  def _provider(self) -> str:
    if importlib.util.find_spec("torch") is None:
      return "PyTorch unavailable"
    try:
      import torch
      return "PyTorch CUDA" if torch.cuda.is_available() else "PyTorch CPU"
    except Exception as exc:
      return f"PyTorch unavailable: {exc}"

  def _deps(self):
    try:
      import numpy as np
      from PIL import Image
    except ImportError as exc:
      raise RuntimeError("CellSAM backend requires numpy and Pillow from server/requirements.txt") from exc
    return np, Image


def normalize_label_mask(mask, np):
  arr = np.asarray(mask)
  arr = np.squeeze(arr)
  if arr.ndim != 2:
    raise RuntimeError(f"CellSAM returned an unsupported mask shape: {arr.shape}")
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


def env_bool(name: str, default: bool) -> bool:
  value = os.getenv(name)
  if value is None:
    return default
  return value.strip().lower() in {"1", "true", "yes", "on"}

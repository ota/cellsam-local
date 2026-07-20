from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import threading
import urllib.request
from pathlib import Path

from .image_utils import decode_rgb_image
from .well_detection import choose_prompt_mask, detect_wells, find_spheroid_prompt


MICROSAM_MODELS = {
  "microsam-vit-b-lm": {
    "architecture": "vit_b",
    "label": "MicroSAM ViT-B LM",
    "url": "https://zenodo.org/records/10524791/files/vit_b_lm.pth?download=1",
    "filename": "vit_b_lm.pth",
    "md5": "3b5db6086051b24d80a4b41e42d4c82c",
  },
}


def microsam_health() -> dict:
  required = ("torch", "segment_anything", "cv2", "numpy", "PIL")
  missing = [name for name in required if importlib.util.find_spec(name) is None]
  provider = "unavailable"
  error = None
  if not missing:
    try:
      import torch
      provider = "PyTorch CUDA" if torch.cuda.is_available() else "PyTorch CPU"
    except Exception as exc:
      error = f"torch import failed: {exc}"

  result = {
    "available": not missing and error is None,
    "provider": provider,
    "models": list(MICROSAM_MODELS.keys()),
    "modelCache": str(_model_cache()),
    "license": (
      "micro-sam code is MIT; the MicroSAM-LM-Generalist-ViT-B checkpoint "
      "is CC-BY-4.0 on Zenodo; Segment Anything code is Apache-2.0."
    ),
    "install": (
      "uv venv .venv-microsam && "
      "uv pip install --python .venv-microsam/bin/python -r server/requirements-microsam.txt"
    ),
  }
  if missing:
    result["error"] = f"Missing dependencies: {', '.join(missing)}"
  elif error:
    result["error"] = error
  return result


class MicroSamBackend:
  def __init__(self):
    self._predictors = {}
    self._inference_lock = threading.Lock()

  def segment_image_bytes(self, content: bytes, model_key: str) -> dict:
    if model_key not in MICROSAM_MODELS:
      raise ValueError(f"Unknown micro-sam model: {model_key}")

    torch, np, image_cls, image_ops = self._deps()
    image = decode_rgb_image(content, image_cls, image_ops)
    image_rgb = np.asarray(image)
    wells = detect_wells(image_rgb)
    prompts = [find_spheroid_prompt(image_rgb, well) for well in wells]
    active = [(well, prompt) for well, prompt in zip(wells, prompts) if prompt is not None]

    raw_masks = []
    if active:
      with self._inference_lock:
        predictor = self._load_predictor(model_key, torch)
        with torch.inference_mode():
          predictor.set_image(image_rgb)
          for well, prompt in active:
            masks, scores, _ = predictor.predict(
              box=np.asarray(prompt.box, dtype="float32"),
              point_coords=np.asarray([prompt.center], dtype="float32"),
              point_labels=np.asarray([1], dtype="int32"),
              multimask_output=True,
            )
            selected = choose_prompt_mask(masks, scores, well, prompt)
            if selected is None:
              continue
            score, mask = selected
            raw_masks.append({
              "iou": float(score),
              "rle": encode_rle(mask, np),
            })

    provider = self._provider(torch)
    return {
      "modelName": f"{MICROSAM_MODELS[model_key]['label']} (well-aware, server {provider})",
      "width": image.width,
      "height": image.height,
      "rawMasks": raw_masks,
      "provider": provider,
      "wellCount": len(wells),
      "promptCount": len(active),
      "wells": [well.as_dict() for well in wells],
    }

  def _load_predictor(self, model_key: str, torch):
    if model_key in self._predictors:
      return self._predictors[model_key]

    try:
      from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
      raise RuntimeError(
        "micro-sam backend dependencies are not installed. Install "
        "server/requirements-microsam.txt in .venv-microsam."
      ) from exc

    config = MICROSAM_MODELS[model_key]
    checkpoint = _ensure_checkpoint(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[config["architecture"]](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    self._predictors[model_key] = predictor
    return predictor

  def _provider(self, torch) -> str:
    return "PyTorch CUDA" if torch.cuda.is_available() else "PyTorch CPU"

  def _deps(self):
    try:
      import numpy as np
      import torch
      from PIL import Image, ImageOps
    except ImportError as exc:
      raise RuntimeError(
        "micro-sam backend requires PyTorch, numpy, Pillow, OpenCV, and segment-anything"
      ) from exc
    return torch, np, Image, ImageOps


def encode_rle(mask, np) -> dict:
  flat = np.asarray(mask, dtype="uint8").reshape(-1)
  indices = np.flatnonzero(flat)
  if len(indices) == 0:
    return {"counts": [], "encoding": "start-length"}

  counts = []
  start = int(indices[0])
  previous = start
  for index in indices[1:]:
    index = int(index)
    if index == previous + 1:
      previous = index
      continue
    counts.extend([start, previous - start + 1])
    start = previous = index
  counts.extend([start, previous - start + 1])
  return {"counts": counts, "encoding": "start-length"}


def _model_cache() -> Path:
  default = Path(__file__).resolve().parents[1] / ".cache" / "microsam" / "models"
  return Path(os.getenv("MICROSAM_MODEL_CACHE", default))


def _ensure_checkpoint(config: dict) -> Path:
  destination = _model_cache() / config["filename"]
  if destination.exists() and _md5(destination) == config["md5"]:
    return destination
  if destination.exists():
    destination.unlink()

  destination.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as temp:
    temporary = Path(temp.name)
  try:
    urllib.request.urlretrieve(config["url"], temporary)
    digest = _md5(temporary)
    if digest != config["md5"]:
      raise RuntimeError(
        f"micro-sam checkpoint checksum mismatch: expected {config['md5']}, got {digest}"
      )
    temporary.replace(destination)
  finally:
    if temporary.exists():
      temporary.unlink()
  return destination


def _md5(path: Path) -> str:
  digest = hashlib.md5()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()

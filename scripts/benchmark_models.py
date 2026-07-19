from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from server.segmenter import MODEL_LABELS, MODEL_URLS, Sam2OnnxSegmenter


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Benchmark server-side segmentation models on local validation images.",
  )
  parser.add_argument(
    "--images",
    type=Path,
    default=ROOT / "assets" / "validation",
    help="Validation image file or directory. Default: assets/validation",
  )
  parser.add_argument(
    "--models",
    nargs="+",
    default=["tiny"],
    help=f"Model keys to run. Available: {', '.join(MODEL_URLS)}",
  )
  parser.add_argument(
    "--points-per-side",
    type=int,
    default=4,
    help="Prompt grid density. 4 runs 16 decoder prompts per image.",
  )
  parser.add_argument("--limit", type=int, default=None, help="Limit number of images.")
  parser.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=0.96,
    help="IoU threshold used for the post-filter count.",
  )
  parser.add_argument(
    "--nms-thresh",
    type=float,
    default=0.70,
    help="Mask-IoU NMS threshold used for the kept count.",
  )
  parser.add_argument(
    "--min-mask-area",
    type=int,
    default=100,
    help="Minimum mask area used for the post-filter count.",
  )
  parser.add_argument(
    "--max-mask-ratio",
    type=float,
    default=0.04,
    help="Maximum mask area ratio used for the post-filter count.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="JSON output path. Default: reports/benchmark-<timestamp>.json",
  )
  parser.add_argument(
    "--write-overlays",
    action="store_true",
    help="Write PNG previews with kept masks overlaid under reports/overlays.",
  )
  parser.add_argument(
    "--overlay-dir",
    type=Path,
    default=None,
    help="Overlay output directory. Default: reports/overlays/<benchmark-name>",
  )
  parser.add_argument(
    "--overlay-max-edge",
    type=int,
    default=1600,
    help="Maximum width or height for overlay preview PNGs.",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  invalid_models = [model for model in args.models if model not in MODEL_URLS]
  if invalid_models:
    print(f"Unknown model(s): {', '.join(invalid_models)}", file=sys.stderr)
    return 2
  if args.points_per_side < 1:
    print("--points-per-side must be >= 1", file=sys.stderr)
    return 2

  images = discover_images(args.images)
  if args.limit is not None:
    images = images[:args.limit]
  if not images:
    print(f"No validation images found at {args.images}", file=sys.stderr)
    return 2

  started_at = datetime.now(timezone.utc)
  output_path = args.output or ROOT / "reports" / f"benchmark-{started_at:%Y%m%d-%H%M%S}.json"
  overlay_dir = None
  if args.write_overlays:
    overlay_dir = args.overlay_dir or ROOT / "reports" / "overlays" / output_path.stem

  segmenter = Sam2OnnxSegmenter()
  health = safe_health(segmenter)
  records = []

  for model in args.models:
    for image_path in images:
      records.append(run_one(segmenter, model, image_path, args, overlay_dir))

  payload = {
    "generatedAt": started_at.isoformat(),
    "config": {
      "images": str(args.images),
      "models": args.models,
      "pointsPerSide": args.points_per_side,
      "predIouThresh": args.pred_iou_thresh,
      "nmsThresh": args.nms_thresh,
      "minMaskArea": args.min_mask_area,
      "maxMaskRatio": args.max_mask_ratio,
      "writeOverlays": args.write_overlays,
      "overlayDir": str(overlay_dir) if overlay_dir else None,
      "overlayMaxEdge": args.overlay_max_edge,
    },
    "environment": {
      "python": sys.version.split()[0],
      "serverHealth": health,
    },
    "summary": summarize(records),
    "results": records,
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

  print_summary(payload, output_path)
  return 0 if all(record.get("ok") for record in records) else 1


def safe_health(segmenter: Sam2OnnxSegmenter) -> dict:
  try:
    return segmenter.health()
  except Exception as exc:
    return {"error": str(exc)}


def discover_images(path: Path) -> list[Path]:
  if path.is_file():
    return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
  if not path.is_dir():
    return []
  return sorted(
    child for child in path.iterdir()
    if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
  )


def run_one(
  segmenter: Sam2OnnxSegmenter,
  model: str,
  image_path: Path,
  args: argparse.Namespace,
  overlay_dir: Path | None,
) -> dict:
  started = time.perf_counter()
  try:
    result = segmenter.segment_image_bytes(
      image_path.read_bytes(),
      model=model,
      points_per_side=args.points_per_side,
    )
  except Exception as exc:
    return {
      "ok": False,
      "model": model,
      "modelLabel": MODEL_LABELS.get(model, model),
      "image": str(image_path),
      "elapsedSec": round(time.perf_counter() - started, 4),
      "error": str(exc),
    }

  elapsed = time.perf_counter() - started
  width = int(result["width"])
  height = int(result["height"])
  image_area = width * height
  max_mask_area = image_area * args.max_mask_ratio

  masks = [
    {
      "iou": float(raw["iou"]),
      "area": rle_area(raw["rle"]),
      "spans": rle_spans(raw["rle"]),
    }
    for raw in result["rawMasks"]
  ]
  filtered = [
    mask for mask in masks
    if (
      mask["iou"] >= args.pred_iou_thresh
      and mask["area"] >= args.min_mask_area
      and mask["area"] <= max_mask_area
    )
  ]
  kept = nms(filtered, args.nms_thresh)

  ious = [mask["iou"] for mask in masks]
  areas = [mask["area"] for mask in masks]
  filtered_areas = [mask["area"] for mask in kept]

  record = {
    "ok": True,
    "model": model,
    "modelLabel": MODEL_LABELS.get(model, model),
    "runtimeModelName": result["modelName"],
    "provider": result.get("provider"),
    "image": str(image_path),
    "width": width,
    "height": height,
    "elapsedSec": round(elapsed, 4),
    "rawMaskCount": len(masks),
    "filteredMaskCount": len(filtered),
    "keptMaskCount": len(kept),
    "rawIou": stats(ious),
    "rawArea": stats(areas),
    "keptArea": stats(filtered_areas),
  }
  if overlay_dir is not None:
    record["overlay"] = str(write_overlay(image_path, model, width, height, kept, overlay_dir, args.overlay_max_edge))
  return record


def rle_area(rle: dict) -> int:
  counts = rle.get("counts", [])
  return int(sum(counts[1::2]))


def rle_spans(rle: dict) -> list[tuple[int, int]]:
  counts = rle.get("counts", [])
  return [
    (int(start), int(start + length))
    for start, length in zip(counts[0::2], counts[1::2])
    if length > 0
  ]


def span_iou(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
  area_a = span_area(a)
  area_b = span_area(b)
  if area_a == 0 or area_b == 0:
    return 0.0

  ia = 0
  ib = 0
  inter = 0
  while ia < len(a) and ib < len(b):
    start = max(a[ia][0], b[ib][0])
    end = min(a[ia][1], b[ib][1])
    if end > start:
      inter += end - start
    if a[ia][1] < b[ib][1]:
      ia += 1
    else:
      ib += 1

  union = area_a + area_b - inter
  return inter / union if union else 0.0


def span_area(spans: Iterable[tuple[int, int]]) -> int:
  return sum(end - start for start, end in spans)


def nms(masks: list[dict], threshold: float) -> list[dict]:
  keep = []
  for mask in sorted(masks, key=lambda item: item["iou"], reverse=True):
    if all(span_iou(mask["spans"], kept["spans"]) <= threshold for kept in keep):
      keep.append(mask)
  return keep


def write_overlay(
  image_path: Path,
  model: str,
  width: int,
  height: int,
  masks: list[dict],
  output_dir: Path,
  max_edge: int,
) -> Path:
  np, image_cls = overlay_deps()

  image = image_cls.open(image_path).convert("RGB")
  scale = min(1.0, max_edge / max(width, height)) if max_edge > 0 else 1.0
  out_size = (max(1, round(width * scale)), max(1, round(height * scale)))
  if image.size != out_size:
    image = image.resize(out_size, image_cls.Resampling.BILINEAR)
  base = image.convert("RGBA")

  for idx, mask in enumerate(masks):
    mask_arr = mask_from_spans(mask["spans"], width, height, np)
    if out_size != (width, height):
      mask_img = image_cls.fromarray((mask_arr * 255).astype("uint8"), mode="L")
      mask_arr = np.asarray(mask_img.resize(out_size, image_cls.Resampling.NEAREST)) > 0

    color = overlay_color(idx)
    fill = image_cls.new("RGBA", out_size, color + (0,))
    fill.putalpha(image_cls.fromarray((mask_arr * 70).astype("uint8"), mode="L"))
    base = image_cls.alpha_composite(base, fill)

    border_arr = mask_border(mask_arr, np)
    border = image_cls.new("RGBA", out_size, color + (0,))
    border.putalpha(image_cls.fromarray((border_arr * 240).astype("uint8"), mode="L"))
    base = image_cls.alpha_composite(base, border)

  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / f"{safe_stem(image_path)}__{model}.png"
  base.convert("RGB").save(output_path)
  return output_path


def overlay_deps():
  try:
    import numpy as np
    from PIL import Image
  except ImportError as exc:
    raise RuntimeError("Overlay output requires numpy and Pillow from server/requirements.txt") from exc
  return np, Image


def mask_from_spans(spans: list[tuple[int, int]], width: int, height: int, np):
  mask = np.zeros(width * height, dtype=bool)
  for start, end in spans:
    mask[start:end] = True
  return mask.reshape((height, width))


def mask_border(mask, np):
  eroded = mask.copy()
  eroded[1:, :] &= mask[:-1, :]
  eroded[:-1, :] &= mask[1:, :]
  eroded[:, 1:] &= mask[:, :-1]
  eroded[:, :-1] &= mask[:, 1:]
  eroded[0, :] = False
  eroded[-1, :] = False
  eroded[:, 0] = False
  eroded[:, -1] = False
  return mask & ~eroded


def overlay_color(index: int) -> tuple[int, int, int]:
  colors = [
    (0, 168, 150),
    (238, 108, 77),
    (64, 145, 255),
    (255, 190, 46),
    (174, 93, 219),
    (83, 184, 72),
  ]
  return colors[index % len(colors)]


def safe_stem(path: Path) -> str:
  return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_") or "image"


def stats(values: list[float | int]) -> dict:
  if not values:
    return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
  return {
    "count": len(values),
    "min": round(min(values), 4),
    "median": round(statistics.median(values), 4),
    "mean": round(statistics.fmean(values), 4),
    "max": round(max(values), 4),
  }


def summarize(records: list[dict]) -> dict:
  summary = {}
  models = sorted({record["model"] for record in records})
  for model in models:
    model_records = [record for record in records if record["model"] == model]
    ok_records = [record for record in model_records if record.get("ok")]
    summary[model] = {
      "label": MODEL_LABELS.get(model, model),
      "runs": len(model_records),
      "ok": len(ok_records),
      "errors": len(model_records) - len(ok_records),
      "elapsedSec": stats([record["elapsedSec"] for record in ok_records]),
      "rawMaskCount": stats([record["rawMaskCount"] for record in ok_records]),
      "filteredMaskCount": stats([record["filteredMaskCount"] for record in ok_records]),
      "keptMaskCount": stats([record["keptMaskCount"] for record in ok_records]),
    }
  return summary


def print_summary(payload: dict, output_path: Path) -> None:
  print(f"Wrote {output_path}")
  for model, item in payload["summary"].items():
    elapsed = item["elapsedSec"]
    kept = item["keptMaskCount"]
    print(
      f"{model}: ok={item['ok']}/{item['runs']} "
      f"elapsed_mean={elapsed['mean']}s kept_mean={kept['mean']}"
    )


if __name__ == "__main__":
  raise SystemExit(main())

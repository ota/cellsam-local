from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Well:
  x: int
  y: int
  radius: int
  inferred: bool = False
  deep: bool = False

  def as_dict(self) -> dict:
    return {
      "x": self.x,
      "y": self.y,
      "radius": self.radius,
      "inferred": self.inferred,
      "deep": self.deep,
    }


@dataclass(frozen=True)
class SpheroidPrompt:
  box: tuple[int, int, int, int]
  center: tuple[int, int]
  area: int
  contrast: float

  def as_dict(self) -> dict:
    return {
      "box": list(self.box),
      "center": list(self.center),
      "area": self.area,
      "contrast": round(self.contrast, 4),
    }


def detect_wells(image_rgb, max_edge: int = 1200) -> list[Well]:
  cv2, np = _deps()
  if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
    raise ValueError("Well detection expects an RGB image")

  height, width = image_rgb.shape[:2]
  scale = min(1.0, max_edge / max(width, height)) if max_edge > 0 else 1.0
  if scale < 1.0:
    work = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
  else:
    work = image_rgb

  gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
  gray = cv2.GaussianBlur(gray, (9, 9), 2)
  short_edge = min(gray.shape[:2])
  circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=max(24, round(short_edge * 0.20)),
    param1=60,
    param2=25,
    minRadius=max(12, round(short_edge * 0.10)),
    maxRadius=max(24, round(short_edge * 0.25)),
  )
  if circles is None:
    return []

  detected = [
    Well(
      x=round(float(x) / scale),
      y=round(float(y) / scale),
      radius=round(float(radius) / scale),
    )
    for x, y, radius in circles[0]
  ]
  detected = _filter_radius_outliers(detected)
  completed = complete_well_rows(detected, width, height)
  deep_plate = _is_deep_well_plate(image_rgb, completed, cv2, np)
  return [
    Well(well.x, well.y, well.radius, inferred=well.inferred, deep=deep_plate)
    for well in completed
  ]


def complete_well_rows(wells: list[Well], width: int, height: int) -> list[Well]:
  if len(wells) < 4:
    return _sort_wells(wells)

  median_radius = statistics.median(well.radius for well in wells)
  rows = _cluster_rows(wells, tolerance=max(8.0, median_radius * 0.48))
  spacing = _horizontal_spacing(rows, median_radius)
  if spacing is None:
    return _sort_wells(wells)

  completed = []
  for row in rows:
    completed.extend(_complete_row(row, spacing, median_radius, width))
  return _sort_wells(_deduplicate_wells(completed, median_radius * 0.35))


def find_spheroid_prompt(image_rgb, well: Well) -> SpheroidPrompt | None:
  cv2, np = _deps()
  height, width = image_rgb.shape[:2]
  radius = max(12, well.radius)
  crop_radius = round(radius * 0.68)
  left = max(0, well.x - crop_radius)
  top = max(0, well.y - crop_radius)
  right = min(width, well.x + crop_radius + 1)
  bottom = min(height, well.y + crop_radius + 1)
  if right - left < 16 or bottom - top < 16:
    return None

  crop = image_rgb[top:bottom, left:right]
  gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
  sigma = max(1.0, radius * 0.012)
  smooth = cv2.GaussianBlur(gray, (0, 0), sigma)

  yy, xx = np.ogrid[:smooth.shape[0], :smooth.shape[1]]
  local_x = well.x - left
  local_y = well.y - top
  distance_sq = (xx - local_x) ** 2 + (yy - local_y) ** 2
  deep_well = well.deep
  search_radius = radius * (0.30 if deep_well else 0.48)
  search_mask = distance_sq <= search_radius ** 2
  if not np.any(search_mask):
    return None

  search_values = smooth[search_mask]
  background = float(np.percentile(search_values, 75))
  dark_percentile = float(np.percentile(search_values, 18 if deep_well else 28))
  threshold = min(dark_percentile, background - 5.0)
  if background - float(np.percentile(search_values, 10)) < 7.0:
    return None

  binary = ((smooth <= threshold) & search_mask).astype("uint8") * 255
  close_size = _odd_size(radius * 0.035, minimum=5, maximum=21)
  open_size = _odd_size(radius * 0.012, minimum=3, maximum=9)
  binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
  )
  binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
  )

  count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
  well_area = math.pi * radius * radius
  candidates = []
  for label in range(1, count):
    x, y, component_width, component_height, area = [int(value) for value in stats[label]]
    max_area_ratio = 0.12 if deep_well else 0.32
    if area < well_area * 0.0025 or area > well_area * max_area_ratio:
      continue
    if component_width < radius * 0.06 or component_height < radius * 0.06:
      continue
    cx, cy = [float(value) for value in centroids[label]]
    center_distance = math.hypot(cx - local_x, cy - local_y)
    max_center_distance = radius * (0.18 if deep_well else 0.42)
    if center_distance > max_center_distance:
      continue
    component = labels == label
    component_mean = float(np.mean(smooth[component]))
    contrast = background - component_mean
    if contrast < (9.0 if deep_well else 5.0):
      continue
    area_ratio = area / well_area
    target_area_ratio = 0.015 if deep_well else 0.035
    score = (
      center_distance / radius * 0.45
      + abs(math.log(max(area_ratio, 1e-6) / target_area_ratio)) * 0.18
      - contrast * 0.002
    )
    candidates.append((score, x, y, component_width, component_height, area, cx, cy, contrast))

  gray_float = gray.astype("float32")
  detail_sigma = max(2.0, radius * 0.008)
  detail = gray_float - cv2.GaussianBlur(gray_float, (0, 0), detail_sigma)
  texture = cv2.GaussianBlur(detail * detail, (0, 0), max(3.0, radius * 0.025))
  texture_radius = radius * (0.28 if deep_well else 0.35)
  texture_mask = distance_sq <= texture_radius ** 2
  texture_values = texture[texture_mask]
  texture_floor = float(np.percentile(texture_values, 80))
  texture_weights = np.where(texture_mask, np.maximum(texture - texture_floor, 0.0), 0.0)
  texture_total = float(texture_weights.sum())
  if texture_total > 0:
    x_axis = np.arange(texture.shape[1], dtype="float32")[None, :]
    y_axis = np.arange(texture.shape[0], dtype="float32")[:, None]
    texture_x = float((texture_weights * x_axis).sum() / texture_total)
    texture_y = float((texture_weights * y_axis).sum() / texture_total)
  else:
    texture_x, texture_y = float(local_x), float(local_y)

  texture_median = float(np.median(texture_values))
  texture_concentration = float(texture_values.max()) / max(1.0, texture_median)
  if candidates:
    _, x, y, component_width, component_height, area, cx, cy, contrast = min(candidates)
  elif not deep_well and texture_concentration > 8.0:
    cx, cy = texture_x, texture_y
    area = round(well_area * 0.03)
    contrast = 0.0
  else:
    return None

  if deep_well:
    texture_x, texture_y = cx, cy
    box_ratio = 0.30
  else:
    box_ratio = 0.50 if texture_concentration > 12.0 else 0.72
  box_width = round(radius * box_ratio)
  box_height = round(radius * box_ratio)
  global_cx = left + texture_x
  global_cy = top + texture_y
  x0 = max(0, round(global_cx - box_width / 2))
  y0 = max(0, round(global_cy - box_height / 2))
  x1 = min(width - 1, round(global_cx + box_width / 2))
  y1 = min(height - 1, round(global_cy + box_height / 2))
  return SpheroidPrompt(
    box=(x0, y0, x1, y1),
    center=(round(global_cx), round(global_cy)),
    area=area,
    contrast=contrast,
  )


def choose_prompt_mask(masks, scores, well: Well, prompt: SpheroidPrompt):
  _, np = _deps()
  well_area = math.pi * well.radius * well.radius
  x0, y0, x1, y1 = prompt.box
  box_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
  ranked = []
  for mask, raw_score in zip(masks, scores):
    prompt_x, prompt_y = prompt.center
    if not mask[prompt_y, prompt_x]:
      continue
    ys, xs = np.nonzero(mask)
    area = len(xs)
    max_area = min(well_area * 0.35, box_area * 1.65)
    if area < well_area * 0.002 or area > max_area:
      continue
    cx = float(xs.mean())
    cy = float(ys.mean())
    well_distance = math.hypot(cx - well.x, cy - well.y)
    prompt_distance = math.hypot(cx - prompt_x, cy - prompt_y)
    if well_distance > well.radius * 0.58 or prompt_distance > well.radius * 0.35:
      continue
    score = (
      float(raw_score)
      - 0.14 * well_distance / well.radius
      - 0.18 * prompt_distance / well.radius
      - 0.08 * abs(math.log(area / max(prompt.area, 1)))
    )
    ranked.append((score, float(raw_score), mask))
  if not ranked:
    return None
  _, raw_score, mask = max(ranked, key=lambda item: item[0])
  return raw_score, mask


def _filter_radius_outliers(wells: list[Well]) -> list[Well]:
  if len(wells) < 3:
    return wells
  median_radius = statistics.median(well.radius for well in wells)
  return [
    well for well in wells
    if median_radius * 0.60 <= well.radius <= median_radius * 1.50
  ]


def _is_deep_well_plate(image_rgb, wells: list[Well], cv2, np) -> bool:
  if len(wells) < 3:
    return False
  gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
  contrasts = []
  for well in wells:
    radius = well.radius
    crop_radius = round(radius * 0.58)
    left = max(0, well.x - crop_radius)
    top = max(0, well.y - crop_radius)
    right = min(gray.shape[1], well.x + crop_radius + 1)
    bottom = min(gray.shape[0], well.y + crop_radius + 1)
    crop = gray[top:bottom, left:right]
    if crop.size == 0:
      continue
    yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
    local_x = well.x - left
    local_y = well.y - top
    distance_sq = (xx - local_x) ** 2 + (yy - local_y) ** 2
    inner = distance_sq <= (radius * 0.28) ** 2
    outer = (distance_sq >= (radius * 0.38) ** 2) & (distance_sq <= (radius * 0.55) ** 2)
    if np.any(inner) and np.any(outer):
      contrasts.append(float(np.median(crop[inner])) - float(np.median(crop[outer])))
  return bool(contrasts and statistics.median(contrasts) > 50.0)


def _cluster_rows(wells: list[Well], tolerance: float) -> list[list[Well]]:
  rows = []
  for well in sorted(wells, key=lambda item: (item.y, item.x)):
    matching = [
      row for row in rows
      if abs(well.y - statistics.fmean(item.y for item in row)) <= tolerance
    ]
    if matching:
      min(matching, key=lambda row: abs(well.y - statistics.fmean(item.y for item in row))).append(well)
    else:
      rows.append([well])
  return [sorted(row, key=lambda item: item.x) for row in rows]


def _horizontal_spacing(rows: list[list[Well]], median_radius: float) -> float | None:
  differences = []
  for row in rows:
    differences.extend(
      right.x - left.x
      for left, right in zip(row, row[1:])
      if median_radius * 1.6 <= right.x - left.x <= median_radius * 6.0
    )
  if not differences:
    return None
  differences.sort()
  lower_count = max(1, math.ceil(len(differences) * 0.60))
  spacing = statistics.median(differences[:lower_count])
  return spacing if median_radius * 1.8 <= spacing <= median_radius * 5.5 else None


def _complete_row(row: list[Well], spacing: float, median_radius: float, width: int) -> list[Well]:
  anchor = min(row, key=lambda item: item.x)
  positions = []
  index = math.floor((0 - anchor.x) / spacing)
  while True:
    x = anchor.x + index * spacing
    if x > width - 1:
      break
    if x >= 0:
      nearest = min(row, key=lambda item: abs(item.x - x))
      if abs(nearest.x - x) <= spacing * 0.28:
        positions.append(nearest)
      else:
        positions.append(Well(
          x=round(x),
          y=round(statistics.fmean(item.y for item in row)),
          radius=round(median_radius),
          inferred=True,
        ))
    index += 1
  return positions


def _deduplicate_wells(wells: list[Well], distance: float) -> list[Well]:
  kept = []
  for well in sorted(wells, key=lambda item: (item.inferred, item.y, item.x)):
    if all(math.hypot(well.x - other.x, well.y - other.y) > distance for other in kept):
      kept.append(well)
  return kept


def _sort_wells(wells: list[Well]) -> list[Well]:
  return sorted(wells, key=lambda item: (item.y, item.x))


def _odd_size(value: float, minimum: int, maximum: int) -> int:
  size = max(minimum, min(maximum, round(value)))
  return size if size % 2 == 1 else size + 1


def _deps():
  try:
    import cv2
    import numpy as np
  except ImportError as exc:
    raise RuntimeError(
      "Well-aware segmentation requires numpy and opencv-python-headless"
    ) from exc
  return cv2, np

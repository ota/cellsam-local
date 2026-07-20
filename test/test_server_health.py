from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image, ImageOps

from server.image_utils import decode_rgb_image
from server.segmenter import MODEL_KEYS, Sam2OnnxSegmenter
from server.well_detection import Well, complete_well_rows


class ServerHealthTest(unittest.TestCase):
  def test_cellpose_only_environment_is_ready(self):
    segmenter = Sam2OnnxSegmenter()
    segmenter._onnx_health = lambda: {
      "available": False,
      "provider": None,
      "availableProviders": [],
    }
    cellpose = {
      "available": True,
      "provider": "PyTorch CUDA",
      "models": ["cellpose-cpsam-v2"],
    }

    with (
      patch("server.segmenter._cellpose_health", return_value=cellpose),
      patch("server.segmenter._microsam_health", return_value={"available": False}),
    ):
      health = segmenter.health()

    self.assertTrue(health["ready"])
    self.assertEqual(health["provider"], "PyTorch CUDA")
    self.assertEqual(health["models"], ["cellpose-cpsam-v2"])
    self.assertEqual(health["supportedModels"], MODEL_KEYS)

  def test_unavailable_backends_do_not_advertise_models(self):
    segmenter = Sam2OnnxSegmenter()
    segmenter._onnx_health = lambda: {
      "available": False,
      "provider": None,
      "availableProviders": [],
    }
    cellpose = {
      "available": False,
      "provider": "unavailable",
      "models": ["cellpose-cpsam-v2"],
    }

    with (
      patch("server.segmenter._cellpose_health", return_value=cellpose),
      patch("server.segmenter._microsam_health", return_value={"available": False}),
    ):
      health = segmenter.health()

    self.assertFalse(health["ready"])
    self.assertIsNone(health["provider"])
    self.assertEqual(health["models"], [])

  def test_microsam_only_environment_is_ready(self):
    segmenter = Sam2OnnxSegmenter()
    segmenter._onnx_health = lambda: {
      "available": False,
      "provider": None,
      "availableProviders": [],
    }
    microsam = {
      "available": True,
      "provider": "PyTorch CUDA",
      "models": ["microsam-vit-b-lm"],
    }

    with (
      patch("server.segmenter._cellpose_health", return_value={"available": False}),
      patch("server.segmenter._microsam_health", return_value=microsam),
    ):
      health = segmenter.health()

    self.assertTrue(health["ready"])
    self.assertEqual(health["models"], ["microsam-vit-b-lm"])
    self.assertEqual(health["provider"], "PyTorch CUDA")

  def test_staggered_rows_are_completed(self):
    detected = [
      Well(1280, 220, 380),
      Well(1780, 1110, 380),
      Well(2820, 1110, 380),
      Well(1270, 2010, 380),
      Well(2310, 2010, 380),
      Well(3350, 2010, 380),
    ]

    completed = complete_well_rows(detected, width=3568, height=2368)

    self.assertEqual(len(completed), 11)
    self.assertEqual(sum(well.inferred for well in completed), 5)

  def test_image_decode_applies_exif_orientation(self):
    source = Image.new("RGB", (4, 2), "white")
    exif = Image.Exif()
    exif[274] = 6
    content = BytesIO()
    source.save(content, format="JPEG", exif=exif)

    decoded = decode_rgb_image(content.getvalue(), Image, ImageOps)

    self.assertEqual(decoded.size, (2, 4))


if __name__ == "__main__":
  unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

from server.segmenter import MODEL_KEYS, Sam2OnnxSegmenter


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

    with patch("server.segmenter._cellpose_health", return_value=cellpose):
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

    with patch("server.segmenter._cellpose_health", return_value=cellpose):
      health = segmenter.health()

    self.assertFalse(health["ready"])
    self.assertIsNone(health["provider"])
    self.assertEqual(health["models"], [])


if __name__ == "__main__":
  unittest.main()

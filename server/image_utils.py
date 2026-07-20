from __future__ import annotations

from io import BytesIO


def decode_rgb_image(content: bytes, image_cls, image_ops):
  if not content:
    raise ValueError("Uploaded image is empty")
  image = image_cls.open(BytesIO(content))
  return image_ops.exif_transpose(image).convert("RGB")

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .segmenter import MODEL_URLS, Sam2OnnxSegmenter


ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="CellSAM Local Server", version="0.1.0")
segmenter = Sam2OnnxSegmenter()


@app.get("/api/health")
def health():
  try:
    info = segmenter.health()
    return {
      "mode": "server",
      "ready": True,
      **info,
    }
  except RuntimeError as exc:
    return {
      "mode": "server",
      "ready": False,
      "error": str(exc),
    }


@app.post("/api/segment")
async def segment(
  image: UploadFile = File(...),
  model: str = Form("tiny"),
  points_per_side: int = Form(8),
):
  if model not in MODEL_URLS:
    raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
  if points_per_side < 4 or points_per_side > 32:
    raise HTTPException(status_code=400, detail="points_per_side must be between 4 and 32")

  content = await image.read()
  try:
    return segmenter.segment_image_bytes(
      content,
      model=model,
      points_per_side=points_per_side,
    )
  except RuntimeError as exc:
    raise HTTPException(status_code=503, detail=str(exc)) from exc
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/")
def index():
  return FileResponse(ROOT / "index.html")


@app.get("/index.html")
def index_html():
  return FileResponse(ROOT / "index.html")


app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
app.mount("/css", StaticFiles(directory=ROOT / "css"), name="css")
app.mount("/js", StaticFiles(directory=ROOT / "js"), name="js")


def main():
  parser = argparse.ArgumentParser(description="Run the CellSAM Local LAN inference server.")
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", default=8080, type=int)
  parser.add_argument("--reload", action="store_true")
  args = parser.parse_args()

  import uvicorn

  uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
  main()

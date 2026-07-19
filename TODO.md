# TODO

## Decisions

- Only adopt models that can be downloaded without user registration or API tokens.
- Research-use-compatible models may be evaluated, but document each adopted
  model's source, license, execution mode, and practical restrictions in both
  README files.
- Keep validation images, reports, overlays, model caches, and Python virtual
  environments out of git.
- Keep framework-specific biology models server-only until there is a clear
  reason and a practical format for local browser execution.

## Completed

- Added a LAN server mode using FastAPI and ONNX Runtime GPU.
- Added a benchmark harness and ignored overlay previews for validation images.
- Evaluated MobileSAM as a smaller general segmentation candidate.
- Excluded CellSAM because its official pretrained weights require DeepCell
  registration and an API token.
- Added Cellpose-SAM v2 as a no-registration, server-only research/evaluation
  backend in an isolated `.venv-cellpose` environment.
- Added backend-specific health discovery. The browser UI lists only models
  available in the active server environment.
- Added Cellpose-SAM v2 to the browser model selector when a Cellpose-capable
  server is running.
- Made ONNX Runtime Web load only when local browser inference is actually used,
  so Cellpose LAN mode does not depend on the ONNX CDN.

## Validation

- Python modules compile with `py_compile`.
- JavaScript and Python unit tests pass with `npm test`.
- SAM2.1-Tiny server smoke test: `ok=1/1`, `CUDAExecutionProvider`.
- Cellpose-SAM v2 full validation run: `ok=15/15`, mean elapsed `7.7662s`,
  mean kept masks `20.2667`.
- Cellpose-SAM v2 post-UI-change smoke test: `ok=1/1`, `PyTorch CUDA`.
- Cellpose-only server health: `ready=true`, available model
  `cellpose-cpsam-v2`, provider `PyTorch CUDA`.
- End-to-end HTTP segmentation of `assets/sample.png`: 12 raw masks returned.
- Headless Chrome confirmed that Cellpose-only server mode hides unavailable
  SAM2 models, selects Cellpose-SAM v2, and hides `points_per_side`.

Generated reports and overlays remain ignored under `reports/`.

## Next Steps

1. Compare Cellpose-SAM v2 overlays against SAM2.1-Tiny and MobileSAM on the
   same validation images, then tune model-specific post-filters.
2. Investigate a smaller no-registration cell-specific model such as Cellpose
   cyto3 ONNX, but verify its redistribution and training-data terms first.
3. Keep `cellpose-cpdino-vitb` out of the implemented model list for now. It
   requires DINOv3 and failed the initial smoke test with missing DINOv3 symbols.
4. Consider a combined server environment only after confirming a compatible
   CUDA dependency set for ONNX Runtime GPU and PyTorch.

## Commands

Normal ONNX GPU server:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r server/requirements.txt
npm run serve:gpu
```

Cellpose-SAM v2 server:

```bash
uv venv .venv-cellpose
uv pip install --python .venv-cellpose/bin/python -r server/requirements-cellpose.txt
npm run serve:gpu:cellpose
```

Benchmark and verify:

```bash
npm run benchmark:cellpose -- --limit 1 --write-overlays
python3 -m py_compile server/app.py server/segmenter.py server/cellpose_backend.py scripts/benchmark_models.py
npm test
git diff --check
```

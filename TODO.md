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
- Added shared EXIF orientation correction to the server backends and benchmark
  overlay writer.
- Added staggered multiwell detection and per-well spheroid prompt generation.
- Added MicroSAM ViT-B LM as a no-registration, server-only research/evaluation
  backend in an isolated `.venv-microsam` environment.
- Added conditional MicroSAM browser UI discovery and a model-specific initial
  IoU threshold.
- Added versioned, self-contained annotation draft JSON export to the detection
  UI, including embedded source pixels, optional SHA-256, model provenance,
  settings, object states, notes, and RLE masks.
- Added a separate human annotation workspace with blind review, accept/reject,
  brush, eraser, polygon, pan/zoom, undo/redo, and per-object notes.
- Added approved ground-truth JSON and sequential label-mask PNG export.

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
- MicroSAM ViT-B LM full validation run: `ok=15/15`, mean elapsed `0.8178s`,
  mean kept masks `8.6`, mean detected wells `9.8667`, `PyTorch CUDA`.
- End-to-end MicroSAM HTTP segmentation of `assets/sample.png`: 11 wells,
  11 prompts, and 11 raw masks returned through `PyTorch CUDA`.
- Headless Chrome confirmed that a MicroSAM-only server selects ViT-B LM, hides
  unavailable models and `points_per_side`, and initializes IoU to `0.70`.
- Headless Chrome completed the full annotation workflow from an 11-mask
  MicroSAM result to a self-contained draft, an approved 11-object ground-truth
  JSON, and a label PNG.
- Headless Chrome verified candidate approval, new-object brush editing,
  undo/redo, desktop layout, and 390 px mobile layout.
- Visual overlay review found clean contours on dense, translucent, and sparse
  brightfield spheroid images. Deep-well recall is intentionally conservative,
  especially for faint or partial objects near a well edge.

Generated reports and overlays remain ignored under `reports/`.

## Next Steps

1. Use the annotation workspace to review a representative subset, then add
   object-level precision/recall plus mask-IoU evaluation to the benchmark
   harness.
2. Tune deep-well candidate selection against those annotations, especially for
   faint and partially cropped edge spheroids.
3. Implement and compare the public, no-registration MicroSAM ViT-T LM
   checkpoint as the next lower-memory microscopy candidate.
4. Evaluate SpheroScan in an isolated GPL environment on per-well crops as a
   domain-specific benchmark.
5. Keep `cellpose-cpdino-vitb` out of the implemented model list for now. It
   requires DINOv3 and failed the initial smoke test with missing DINOv3 symbols.
6. Consider a combined server environment only after confirming a compatible
   CUDA dependency set and a clear operational benefit.

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

MicroSAM ViT-B LM server:

```bash
uv venv .venv-microsam
uv pip install --python .venv-microsam/bin/python -r server/requirements-microsam.txt
npm run serve:gpu:microsam
```

Benchmark and verify:

```bash
npm run benchmark:cellpose -- --limit 1 --write-overlays
npm run benchmark:microsam -- --limit 1 --write-overlays
python3 -m py_compile server/app.py server/segmenter.py server/cellpose_backend.py server/microsam_backend.py server/well_detection.py server/image_utils.py scripts/benchmark_models.py
npm test
git diff --check
```

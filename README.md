# CellSAM Local

[English](README.md) | [日本語](README.ja.md)

CellSAM Local is a browser-based cell image segmentation tool that runs
Meta's SAM 2.1 models with ONNX Runtime Web. It is designed for local image
analysis: uploaded images stay in the browser, while model files are downloaded
from Hugging Face when first used.

For LAN use with low-spec client PCs, the same UI can also run in server mode:
the browser opens the UI from a GPU server, and segmentation runs through the
same-origin `/api/segment` endpoint on that server.

The UI is currently Japanese.

## Features

- Browser-only segmentation for microscopy and cell-like images
- SAM 2.1 automatic mask generation using ONNX Runtime Web
- WebGPU acceleration with WASM fallback
- Optional LAN GPU server mode with same-origin API inference
- Optional well-aware MicroSAM backend for spheroid and organoid images
- Drag-and-drop image loading
- Built-in sample image at `assets/sample.png`
- Adjustable prompt grid density, IoU threshold, mask size filters, and contour width
- Post-detection filtering by brightness, area, and confidence without rerunning inference
- Click-to-exclude and restore individual masks on the output canvas
- Per-object summary table with confidence, area, circularity, brightness, and notes
- Self-contained annotation draft JSON export from segmentation results
- Separate human-review UI with mask approval, brush/eraser, polygon, and undo/redo
- Ground-truth JSON and label-mask PNG export for quantitative validation
- Static frontend with no build step

## Inference Modes

| Mode | Command | Where inference runs | Image handling |
| --- | --- | --- | --- |
| Local browser mode | `npm run serve` | User's browser through WebGPU or WASM | Images stay in the browser |
| LAN GPU server mode | `npm run serve:gpu` | GPU server through `/api/segment` | Images are sent to the LAN server |

The frontend automatically checks `/api/health`. If a ready server backend is
available from the same origin, it uses server inference. Otherwise, it falls
back to the existing browser inference path.

## How It Works

The app ports the core behavior of SAM 2 automatic mask generation into
JavaScript:

1. Resize and pad the input image to SAM 2's `1024x1024` input size.
2. Run the selected SAM 2 encoder once.
3. Generate a `pointsPerSide x pointsPerSide` grid of prompt points.
4. Decode one mask candidate for each prompt point.
5. Filter masks by decoder IoU score and area.
6. Apply mask-IoU non-maximum suppression to remove duplicates.
7. Compute derived object metrics such as area, mean brightness, perimeter, and circularity.
8. Render mask outlines and the object summary table in the browser.

In server mode, steps 1-4 run on the Python server with ONNX Runtime GPU when
CUDA is available. The server returns RLE-compressed raw masks, and the browser
still handles post-filters, display, click exclusion, notes, and the summary
table.

The optional MicroSAM backend follows a different path for multiwell images. It
detects wells, creates one microscopy-oriented box and point prompt per well,
and computes one image embedding with the light-microscopy-fine-tuned ViT-B
checkpoint through PyTorch CUDA. This avoids the dense automatic prompt grid
and reduces false positives from well rims.

## Models

The app uses ONNX SAM 2.1 models hosted by
[SharpAI on Hugging Face](https://huggingface.co/SharpAI).

| UI option | Hugging Face model | Approx. size | Notes |
| --- | --- | ---: | --- |
| SAM2.1-Tiny | `SharpAI/sam2-hiera-tiny-onnx` | 155 MB | Default, fastest option |
| SAM2.1-Small | `SharpAI/sam2-hiera-small-onnx` | 184 MB | Balanced option |
| SAM2.1-Base+ | `SharpAI/sam2-hiera-base-plus-onnx` | 667 MB | Higher accuracy, GPU recommended |
| SAM2.1-Large | `SharpAI/sam2-hiera-large-onnx` | 910 MB | Highest accuracy, strong GPU recommended |

Experimental server-side model:

| Model key | Source | Approx. size | License note | Status |
| --- | --- | ---: | --- | --- |
| `mobile-sam` | [`Heliosoph/sam-onnx`](https://huggingface.co/Heliosoph/sam-onnx) | 43 MB | Model card lists Apache-2.0. It bundles MobileSAM's ViT-T encoder with a SAM mask decoder. | Server benchmark/API candidate; not yet exposed in the browser UI |
| `cellpose-cpsam-v2` | [Cellpose](https://cellpose.readthedocs.io/en/latest/models.html), [`mouseland/cellpose-sam`](https://huggingface.co/mouseland/cellpose-sam) | About 1.2 GB on first use | Cellpose code and model card list BSD-3-Clause; upstream README notes CC-BY-NC training data. No DeepCell-style account token is required. | Optional server-only research/evaluation backend; shown in the UI when available |
| `microsam-vit-b-lm` | [micro-sam](https://github.com/computational-cell-analytics/micro-sam), [ViT-B LM checkpoint](https://zenodo.org/records/10524791) | 375 MB / 358 MiB | micro-sam code is MIT; the checkpoint is CC-BY-4.0; the Segment Anything runtime is Apache-2.0. No account or token is required. | Optional server-only research/evaluation backend; shown in the UI when available |

Adopted model license notes:

- The SharpAI ONNX model cards used by this project list the conversions as
  Apache-2.0.
- The experimental MobileSAM ONNX bundle is also listed as Apache-2.0 on its
  Hugging Face model card.
- The optional Cellpose backend is kept server-only because it adds PyTorch and
  Cellpose dependencies. Cellpose does not require user registration or an API
  token, but it should stay in the research/evaluation lane unless its upstream
  training-data terms are acceptable for the intended workflow.
- The optional MicroSAM backend uses the CC-BY-4.0 light-microscopy checkpoint,
  so redistributed copies or adaptations of the checkpoint must retain
  appropriate attribution. Its minimal headless environment uses Meta's Apache-2.0
  Segment Anything runtime rather than micro-sam's desktop GUI dependencies.
- The SharpAI models are converted from Meta SAM 2.1 models, so check the upstream
  [SAM 2 repository](https://github.com/facebookresearch/sam2) and model terms
  for the intended use.
- Research-use-compatible candidate models can be benchmarked locally. Before a
  candidate is exposed through the UI or server API, its source, license, mode,
  and restrictions must be documented here and in `README.ja.md`.

Candidate models and license notes are tracked in
[docs/model_candidates.md](docs/model_candidates.md).

When WebGPU is available, the encoder uses the pre-optimized `.ort` model.
Otherwise, the app uses the `.onnx` model with the WASM backend.

## Requirements

For local browser mode:

- A modern browser with ES module support
- Chrome or Edge is recommended for WebGPU
- Network access for the initial model download
- A local HTTP server
- Node.js for running the test script

For LAN GPU server mode:

- A GPU server reachable from the client PCs on the LAN
- Python 3.10+
- `uv` for creating and syncing the local Python environment
- NVIDIA CUDA-compatible ONNX Runtime environment for GPU acceleration
- Dependencies from `server/requirements.txt`
- Network access from the server for the initial model download
- Optional Cellpose backend dependencies from `server/requirements-cellpose.txt`
- Optional MicroSAM backend dependencies from `server/requirements-microsam.txt`

Because the app uses ES modules, opening `index.html` directly with `file://`
will not work reliably. Serve the directory over HTTP instead.

## Run Locally

From the project root:

```bash
npm run serve
```

Then open:

```text
http://localhost:8080
```

No package installation is required for the app itself. Runtime libraries are
loaded from CDN, and model weights are downloaded by the browser when a model is
selected for the first time.

## Run on a LAN GPU Server

On the GPU server:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r server/requirements.txt
npm run serve:gpu
```

Then open the server from a client PC:

```text
http://<gpu-server-hostname-or-ip>:8080
```

The server provides the UI and API from the same origin, so client PCs do not
need to enter an API endpoint. The first request for a model downloads ONNX
files into `~/.cache/cellsam-local/models` on the server. Override that path
with `CELLSAM_MODEL_CACHE` if needed.

Server mode exposes:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Reports server mode, dependency readiness, provider, and available models |
| `POST /api/segment` | Accepts an image, model name, and `points_per_side`; returns RLE raw masks |

Optional Cellpose backend setup:

```bash
uv venv .venv-cellpose
uv pip install --python .venv-cellpose/bin/python -r server/requirements-cellpose.txt
npm run benchmark:cellpose -- --limit 1 --write-overlays
```

Cellpose downloads its built-in pretrained models on first use without a user
registration flow. Use the separate `.venv-cellpose` environment because
Cellpose/PyTorch and ONNX Runtime GPU may install incompatible CUDA wheel stacks
inside one virtual environment. Cellpose model files are cached under
`.cache/cellpose/models` by default when using the provided npm scripts.
Cellpose ignores `points_per_side` because it runs its own cell segmentation
pipeline and returns labeled instances. Start the LAN server with:

```bash
npm run serve:gpu:cellpose
```

The browser UI reads `/api/health`, lists only models available in that server
environment, and exposes Cellpose-SAM v2 automatically. Open
`http://<server-ip>:8080` from another machine on the LAN.

Optional MicroSAM light-microscopy backend setup:

```bash
uv venv .venv-microsam
uv pip install --python .venv-microsam/bin/python -r server/requirements-microsam.txt
npm run benchmark:microsam -- --limit 1 --write-overlays
npm run serve:gpu:microsam
```

The first inference downloads the public ViT-B LM checkpoint to
`.cache/microsam/models` without an account or token. The backend detects the
multiwell layout and submits one prompt per candidate well, so
`points_per_side` is ignored. The UI selects an initial IoU threshold of `0.70`
for this model. Current deep-well handling is intentionally conservative:
central, complete spheroids are preferred, while faint, partial objects near a
well edge can be omitted. Validate and tune it against task-specific annotations
before using the counts as quantitative results.

## Basic Usage

1. Open the app in the browser.
2. Use the default sample image or upload an image by clicking or dragging it into the input area.
3. Select one of the models available in the current inference mode.
4. Adjust detection settings if needed.
5. Click `検出実行` to run segmentation.
6. Use the filter sliders to refine the displayed objects.
7. Click masks or table rows to exclude or restore individual detections.

## Ground Truth Annotation

After segmentation, click `下書きJSON保存` in the detection settings. The draft
contains the displayed image as a PNG data URL, its dimensions and pixel
SHA-256 when Web Crypto is available, model provenance, settings, notes, and
all current masks in row-major start-length RLE. Active masks are exported as
unreviewed `candidate` objects; excluded masks are retained as `rejected`.

Open `annotate.html` from the header and load the draft JSON. The annotation UI
provides:

- Candidate approval and rejection with blind review enabled by default
- Object selection, brush, eraser, polygon addition, zoom, and pan
- Up to 30 mask/status edit actions through undo and redo
- Per-object notes and optional display of rejected masks
- In-progress draft JSON, approved ground-truth JSON, and label-mask PNG export

Ground-truth export is available after every candidate has been reviewed and at
least one non-empty object is accepted. Export rejects overlapping accepted
masks. Accepted objects remain separate RLE masks in JSON. In the PNG, the
background is RGB value `0`, and accepted objects are encoded sequentially as
equal RGB channel values `1..255`; `labelValue` in the ground-truth JSON records
the mapping.

Files are downloaded by the browser and are not written to the project
automatically. Store reviewed files under an ignored directory such as
`assets/validation/annotations/` when they should remain outside git. Because
the JSON embeds the source image, treat it with the same privacy controls as the
original image.

## Recommended Settings for Low-Spec PCs

For PCs without a strong GPU, start with the smallest model and a sparse prompt
grid:

| Setting | Recommendation |
| --- | --- |
| Model | `SAM2.1-Tiny` |
| `points/side` | `8` |
| IoU threshold | Keep the default first, then lower only if too few objects are found |
| Minimum mask area | Increase this when many tiny false positives appear |
| Larger models | Avoid `Base+` and `Large` unless WebGPU is available and memory is sufficient |

`points/side` has the largest impact on runtime because the decoder runs once
for each grid point. For example, `8` means 64 decoder runs, while `16` means
256 decoder runs. On WASM/CPU, the difference is usually substantial.

## Detection Settings

| Setting | Description |
| --- | --- |
| `points/side` | Prompt grid density. Higher values can find more objects but increase runtime. |
| IoU threshold | Minimum SAM decoder confidence score. |
| Minimum mask area | Removes masks smaller than the selected pixel area. |
| Maximum mask ratio | Removes masks that are too large relative to the image. |
| Contour width | Controls the rendered outline width on the output canvas. |

After inference, the app keeps raw masks in memory so IoU, area, brightness, and
confidence filters can be adjusted without decoding every prompt again.

## Tests

Run the unit tests with:

```bash
npm test
```

Watch mode:

```bash
npm run test:watch
```

The tests use Node's built-in `node:test` runner and cover the detection data
model, derived metrics, filtering, click-toggle behavior, server RLE mask
decoding, annotation RLE round-trips, and annotation document validation.

For server-side validation on local, ignored images:

```bash
npm run benchmark:server -- --limit 2
npm run benchmark:server -- --models tiny mobile-sam --limit 2
npm run benchmark:server -- --models tiny mobile-sam --limit 2 --write-overlays
npm run benchmark:microsam -- --limit 2 --write-overlays
```

The benchmark reads images from `assets/validation/`, runs the server
segmenter directly, and writes JSON reports under `reports/`. Both directories
are ignored by git so validation inputs and outputs stay separate from commits.
With `--write-overlays`, preview PNGs with kept masks are written under
`reports/overlays/`.

## Project Structure

```text
.
├── index.html              # Japanese UI layout and ONNX Runtime Web loader
├── annotate.html           # Human ground-truth review and mask editing UI
├── css/
│   ├── style.css           # Detection application styling
│   └── annotation.css      # Annotation workspace styling
├── js/
│   ├── annotation_app.js   # Annotation editing state, canvas, and exports
│   ├── annotations.js      # Versioned schema, RLE, and validation helpers
│   ├── app.js              # UI state, events, model loading, rendering flow
│   ├── automask.js         # Grid prompts, mask decoding, post-filters, NMS
│   ├── detection.js        # DetectionResult/DetectedObject and metrics
│   ├── sam2.js             # SAM 2.1 ONNX Runtime Web wrapper
│   ├── server_api.js       # Same-origin server inference client
│   └── visualize.js        # Canvas drawing helpers
├── docs/
│   └── model_candidates.md # Candidate model and license notes
├── server/
│   ├── app.py              # FastAPI UI/API server for LAN deployments
│   ├── run_gpu_python.sh   # Shared GPU Python environment launcher
│   ├── run_gpu_server.sh   # LAN server launcher
│   ├── cellpose_backend.py # Optional Cellpose server-only backend
│   ├── microsam_backend.py # Optional well-aware MicroSAM backend
│   ├── well_detection.py   # Multiwell and per-well prompt detection
│   ├── image_utils.py      # Shared image decoding and EXIF orientation
│   ├── segmenter.py        # Server-side SAM 2.1 ONNX Runtime inference
│   ├── requirements.txt    # GPU server Python dependencies
│   ├── requirements-cellpose.txt # Optional Cellpose dependencies
│   └── requirements-microsam.txt # Optional headless MicroSAM dependencies
├── scripts/
│   └── benchmark_models.py # Server model benchmark harness
├── assets/
│   └── sample.png          # Default sample image
├── test/
│   ├── annotations.test.js # Annotation schema and RLE tests
│   ├── detection.test.js   # Detection model tests
│   └── server_api.test.js  # Server response decoding tests
└── package.json            # Scripts
```

## Privacy Notes

In local browser mode, image files are loaded into the browser and are not
uploaded by this project. The browser does make network requests to download
ONNX Runtime Web assets from jsDelivr and SAM 2.1 model files from Hugging Face.

In LAN GPU server mode, images are sent to the server running this project.
Use that mode only on a trusted LAN or add authentication and upload limits
before exposing it beyond the LAN.

Annotation draft and ground-truth JSON files embed the full source image so
they can be reopened without a separate image file. Handle those JSON files as
image data, even though their extension is `.json`.

## Credits

- [SAM 2.1](https://github.com/facebookresearch/sam2) by Meta AI
- [Segment Anything](https://github.com/facebookresearch/segment-anything) by Meta AI
- [micro-sam](https://github.com/computational-cell-analytics/micro-sam) and its
  [ViT-B LM checkpoint](https://zenodo.org/records/10524791)
- [Cellpose](https://github.com/MouseLand/cellpose)
- ONNX model conversions from [SharpAI](https://huggingface.co/SharpAI)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)

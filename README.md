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
- Drag-and-drop image loading
- Built-in sample image at `assets/sample.png`
- Adjustable prompt grid density, IoU threshold, mask size filters, and contour width
- Post-detection filtering by brightness, area, and confidence without rerunning inference
- Click-to-exclude and restore individual masks on the output canvas
- Per-object summary table with confidence, area, circularity, brightness, and notes
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

## Models

The app uses ONNX SAM 2.1 models hosted by
[SharpAI on Hugging Face](https://huggingface.co/SharpAI).

| UI option | Hugging Face model | Approx. size | Notes |
| --- | --- | ---: | --- |
| SAM2.1-Tiny | `SharpAI/sam2-hiera-tiny-onnx` | 155 MB | Default, fastest option |
| SAM2.1-Small | `SharpAI/sam2-hiera-small-onnx` | 184 MB | Balanced option |
| SAM2.1-Base+ | `SharpAI/sam2-hiera-base-plus-onnx` | 667 MB | Higher accuracy, GPU recommended |
| SAM2.1-Large | `SharpAI/sam2-hiera-large-onnx` | 910 MB | Highest accuracy, strong GPU recommended |

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

## Basic Usage

1. Open the app in the browser.
2. Use the default sample image or upload an image by clicking or dragging it into the input area.
3. Select a SAM 2.1 model.
4. Adjust detection settings if needed.
5. Click `検出実行` to run segmentation.
6. Use the filter sliders to refine the displayed objects.
7. Click masks or table rows to exclude or restore individual detections.

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
model, derived metrics, filtering, click-toggle behavior, and server RLE mask
decoding.

## Project Structure

```text
.
├── index.html              # Japanese UI layout and ONNX Runtime Web loader
├── css/
│   └── style.css           # Application styling
├── js/
│   ├── app.js              # UI state, events, model loading, rendering flow
│   ├── automask.js         # Grid prompts, mask decoding, post-filters, NMS
│   ├── detection.js        # DetectionResult/DetectedObject and metrics
│   ├── sam2.js             # SAM 2.1 ONNX Runtime Web wrapper
│   ├── server_api.js       # Same-origin server inference client
│   └── visualize.js        # Canvas drawing helpers
├── server/
│   ├── app.py              # FastAPI UI/API server for LAN deployments
│   ├── segmenter.py        # Server-side SAM 2.1 ONNX Runtime inference
│   └── requirements.txt    # GPU server Python dependencies
├── assets/
│   └── sample.png          # Default sample image
├── test/
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

## Credits

- [SAM 2.1](https://github.com/facebookresearch/sam2) by Meta AI
- ONNX model conversions from [SharpAI](https://huggingface.co/SharpAI)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)

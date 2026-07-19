# CellSAM Local

[English](README.md) | [日本語](README.ja.md)

CellSAM Local is a browser-based cell image segmentation tool that runs
Meta's SAM 2.1 models with ONNX Runtime Web. It is designed for local image
analysis: uploaded images stay in the browser, while model files are downloaded
from Hugging Face when first used.

The UI is currently Japanese.

## Features

- Browser-only segmentation for microscopy and cell-like images
- SAM 2.1 automatic mask generation using ONNX Runtime Web
- WebGPU acceleration with WASM fallback
- Drag-and-drop image loading
- Built-in sample image at `assets/sample.png`
- Adjustable prompt grid density, IoU threshold, mask size filters, and contour width
- Post-detection filtering by brightness, area, and confidence without rerunning inference
- Click-to-exclude and restore individual masks on the output canvas
- Per-object summary table with confidence, area, circularity, brightness, and notes
- Static frontend with no build step

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

- A modern browser with ES module support
- Chrome or Edge is recommended for WebGPU
- Network access for the initial model download
- A local HTTP server
- Node.js for running the test script

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

## Basic Usage

1. Open the app in the browser.
2. Use the default sample image or upload an image by clicking or dragging it into the input area.
3. Select a SAM 2.1 model.
4. Adjust detection settings if needed.
5. Click `検出実行` to run segmentation.
6. Use the filter sliders to refine the displayed objects.
7. Click masks or table rows to exclude or restore individual detections.

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
model, derived metrics, filtering, and click-toggle behavior.

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
│   └── visualize.js        # Canvas drawing helpers
├── assets/
│   └── sample.png          # Default sample image
├── test/
│   └── detection.test.js   # Node unit tests
└── package.json            # Scripts
```

## Privacy Notes

Image files are loaded into the browser and are not uploaded by this project.
The browser does make network requests to download ONNX Runtime Web assets from
jsDelivr and SAM 2.1 model files from Hugging Face.

## Credits

- [SAM 2.1](https://github.com/facebookresearch/sam2) by Meta AI
- ONNX model conversions from [SharpAI](https://huggingface.co/SharpAI)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)

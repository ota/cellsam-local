# Model Candidates and License Notes

This project can evaluate research-use models, but any model exposed in the UI
or server API should have its license and practical restrictions documented in
the README before it is treated as adopted.

## Adoption Policy

- Research-use-compatible models can be added to the benchmark queue.
- Adopted models must list source, license, execution mode, and known use
  restrictions in `README.md` and `README.ja.md`.
- Browser models need ONNX Runtime Web-compatible encoder and decoder files.
- Large or framework-specific biology models should start as server-only
  backends to avoid complicating the local browser mode.
- Models with non-commercial, unclear, or redistribution-limited terms can be
  benchmarked locally, but should stay clearly marked as research/evaluation
  candidates unless the project owner approves broader use.

## Implemented Models

| Model family | Source | License note | Mode | Status |
| --- | --- | --- | --- | --- |
| SAM 2.1 ONNX Tiny/Small/Base+/Large | [SharpAI SAM2 ONNX models](https://huggingface.co/SharpAI) | Hugging Face model cards mark the ONNX conversions as Apache-2.0. The models are converted from Meta SAM 2.1, so Meta SAM 2.1 terms should also be checked for the intended use. | Browser and server | Adopted |
| MobileSAM ONNX | [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [Heliosoph/sam-onnx](https://huggingface.co/Heliosoph/sam-onnx) | Official repo and ONNX bundle card list Apache-2.0. | Server benchmark/API candidate | Experimental; not yet exposed in the browser UI |

Initial local smoke test on one ignored validation image with `pointsPerSide=4`
showed MobileSAM returning valid masks through CUDA, but slower than
SAM2.1-Tiny in this server path and with fewer kept masks after the current
post-filters. Keep it experimental until it has been tested across the full
validation set and, if browser use is still desired, with an ONNX Runtime Web
wrapper.

## Candidate Queue

| Priority | Candidate | Source | License note | Likely mode | Reason to evaluate |
| ---: | --- | --- | --- | --- | --- |
| 1 | Cellpose cyto3 ONNX | [kmlyyll/cellpose-cyto3-onnx](https://huggingface.co/kmlyyll/cellpose-cyto3-onnx) | Model card lists a custom redistribution-with-permission license. Verify terms before adoption. | Server first | Small cell-specific model; useful comparison against SAM-style masks. |
| 2 | Cellpose-SAM / CPSAM | [Cellpose docs](https://cellpose.readthedocs.io/en/latest/), [mouseland/cellpose-sam](https://huggingface.co/mouseland/cellpose-sam) | Cellpose-SAM model card lists BSD-3-Clause. Some ONNX cards warn about non-commercial training data, so adoption needs explicit README warning. | Server first | Strong cell-specific baseline, but heavier than the current local target. |
| 3 | CellSAM | [cellSAM docs](https://vanvalenlab.github.io/cellSAM/), [cellSAM repo](https://github.com/vanvalenlab/cellSAM) | Official repo is Apache-2.0. Some third-party ONNX exports add academic/non-commercial warnings; use the official PyTorch path first. | Server only | Biology-specific segmentation for diverse cell images; good GPU-server candidate. |
| 4 | micro-sam | [micro-sam repo](https://github.com/computational-cell-analytics/micro-sam) | MIT license. | Server only | Microscopy-oriented SAM workflows, including fine-tuned models and interactive review. |
| 5 | EfficientSAM | [EfficientSAM repo](https://github.com/yformer/EfficientSAM) | Apache-2.0. | Browser candidate after ONNX compatibility check | Smaller general segmentation baseline. |

## Caution / Lower Priority

| Candidate | Source | License note | Reason |
| --- | --- | --- | --- |
| FastSAM | [FastSAM repo](https://github.com/CASIA-LMC-Lab/FastSAM), [Ultralytics FastSAM docs](https://docs.ultralytics.com/models/fast-sam/) | AGPL-3.0 in the official repo. | Speed is attractive, but AGPL obligations and general-image training make it less suitable as the first adopted browser model. |
| EdgeSAM | [EdgeSAM repo](https://github.com/chongzhou96/EdgeSAM) | License must be rechecked before adoption. | Promising edge-device performance, but it is a secondary low-spec candidate until terms and ONNX runtime fit are confirmed. |
| sam2-cells-seg | [DnaRnaProteins/sam2-cells-seg](https://huggingface.co/DnaRnaProteins/sam2-cells-seg) | Apache-2.0 model card. | Good fluorescence-cell candidate, but currently best treated as a server-side PyTorch integration unless converted and verified for ONNX. |

## Next Implementation Order

1. Keep `scripts/benchmark_models.py` as the baseline measurement harness.
2. Benchmark the adopted SAM2.1-Tiny server backend on ignored validation
   images.
3. Compare `tiny` and `mobile-sam` with the same benchmark settings, then decide
   whether MobileSAM should be promoted to the browser UI.
4. Add one cell-specific server backend, preferably Cellpose cyto3 or official
   CellSAM, with README license notes before exposing it in the UI.

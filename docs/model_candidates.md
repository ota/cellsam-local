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
- Prefer models that can be downloaded without a user registration flow or API
  token.
- Models with non-commercial, unclear, or redistribution-limited terms can be
  benchmarked locally, but should stay clearly marked as research/evaluation
  candidates unless the project owner approves broader use.

## Implemented Models

| Model family | Source | License note | Mode | Status |
| --- | --- | --- | --- | --- |
| SAM 2.1 ONNX Tiny/Small/Base+/Large | [SharpAI SAM2 ONNX models](https://huggingface.co/SharpAI) | Hugging Face model cards mark the ONNX conversions as Apache-2.0. The models are converted from Meta SAM 2.1, so Meta SAM 2.1 terms should also be checked for the intended use. | Browser and server | Adopted |
| MobileSAM ONNX | [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [Heliosoph/sam-onnx](https://huggingface.co/Heliosoph/sam-onnx) | Official repo and ONNX bundle card list Apache-2.0. | Server benchmark/API candidate | Experimental; not yet exposed in the browser UI |
| Cellpose-SAM v2 | [Cellpose docs](https://cellpose.readthedocs.io/en/latest/models.html), [mouseland/cellpose-sam](https://huggingface.co/mouseland/cellpose-sam) | Cellpose code and model card list BSD-3-Clause. Upstream README notes CC-BY-NC training data. Built-in model download does not require user registration. | Server-only optional backend and conditional browser UI option | Experimental research/evaluation backend; requires isolated `.venv-cellpose` with `server/requirements-cellpose.txt` |
| MicroSAM ViT-B LM | [micro-sam](https://github.com/computational-cell-analytics/micro-sam), [ViT-B LM checkpoint](https://zenodo.org/records/10524791) | micro-sam code is MIT, the checkpoint is CC-BY-4.0, and the Segment Anything runtime is Apache-2.0. Public download does not require registration. | Server-only optional backend and conditional browser UI option | Experimental well-aware research/evaluation backend; requires isolated `.venv-microsam` with `server/requirements-microsam.txt` |

Initial local smoke test on one ignored validation image with `pointsPerSide=4`
showed MobileSAM returning valid masks through CUDA, but slower than
SAM2.1-Tiny in this server path and with fewer kept masks after the current
post-filters. Keep it experimental until it has been tested across the full
validation set and, if browser use is still desired, with an ONNX Runtime Web
wrapper.

Cellpose-SAM v2 was benchmarked through the isolated `.venv-cellpose` PyTorch
CUDA path on 15 ignored validation images. It completed all runs with
`ok=15/15`, mean elapsed time `7.7662s`, and mean kept masks `20.2667`.
Generated reports and overlay PNGs stay ignored under `reports/`.
When the Cellpose server is running, health discovery advertises only the models
installed in that environment and the browser UI exposes Cellpose-SAM v2.

MicroSAM ViT-B LM was benchmarked through the isolated `.venv-microsam`
PyTorch CUDA path on the same 15 ignored validation images. It completed all
runs with `ok=15/15`, mean elapsed time `0.8178s`, mean kept masks `8.6`, and
mean detected wells `9.8667`. The custom preprocessing detects the staggered
multiwell layout and submits one box/point prompt per candidate well. Visual
review was substantially cleaner than Cellpose-SAM v2 on the dense, translucent,
and sparse brightfield spheroid examples. Deep-well detection is deliberately
conservative and can omit faint or partial edge objects, so these measurements
are engineering benchmarks rather than accuracy scores.

## Candidate Queue

| Priority | Candidate | Source | License note | Likely mode | Reason to evaluate |
| ---: | --- | --- | --- | --- | --- |
| 1 | MicroSAM ViT-T LM | [ViT-T LM checkpoint](https://zenodo.org/records/11111329), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | Checkpoint is CC-BY-4.0; MobileSAM runtime is Apache-2.0. Public download does not require registration. | Server first, then browser feasibility | Roughly 41 MB microscopy-tuned encoder candidate for lower-memory systems; reuse the current well prompts and compare accuracy with ViT-B LM. |
| 2 | SpheroScan | [SpheroScan](https://github.com/FunctionalUrology/SpheroScan) | GPL-3.0. Public source and weights do not require registration. | Isolated server benchmark | Domain-specific spheroid baseline. It expects one spheroid per image, so evaluation would use crops from the current well detector; copyleft and its separate environment make direct integration undesirable. |
| 3 | Cellpose cyto3 ONNX | [kmlyyll/cellpose-cyto3-onnx](https://huggingface.co/kmlyyll/cellpose-cyto3-onnx) | Model card lists a custom redistribution-with-permission license. Verify terms before adoption. | Server first | Small cell-specific model; useful comparison against SAM-style masks. |
| 4 | CellposeDINO-ViTB | [Cellpose docs](https://cellpose.readthedocs.io/en/latest/), [mouseland/cellpose-sam](https://huggingface.co/mouseland/cellpose-sam) | Cellpose model card lists BSD-3-Clause. Upstream README notes CC-BY-NC training data. Requires the DINOv3 dependency before use. | Server first | Smaller Cellpose family candidate, but it needs another dependency and failed the initial smoke test with missing DINOv3 symbols. |
| 5 | EfficientSAM | [EfficientSAM repo](https://github.com/yformer/EfficientSAM) | Apache-2.0. | Browser candidate after ONNX compatibility check | Smaller general segmentation baseline. |

## Caution / Lower Priority

| Candidate | Source | License note | Reason |
| --- | --- | --- | --- |
| FastSAM | [FastSAM repo](https://github.com/CASIA-LMC-Lab/FastSAM), [Ultralytics FastSAM docs](https://docs.ultralytics.com/models/fast-sam/) | AGPL-3.0 in the official repo. | Speed is attractive, but AGPL obligations and general-image training make it less suitable as the first adopted browser model. |
| EdgeSAM | [EdgeSAM repo](https://github.com/chongzhou96/EdgeSAM) | License must be rechecked before adoption. | Promising edge-device performance, but it is a secondary low-spec candidate until terms and ONNX runtime fit are confirmed. |
| CellSAM | [cellSAM docs](https://vanvalenlab.github.io/cellSAM/), [cellSAM repo](https://github.com/vanvalenlab/cellSAM) | Code is Apache-2.0, but official pretrained weights require DeepCell access and are licensed under a modified Apache license for non-commercial academic use only. | Requires user registration/API token, so it is out of scope for the current no-registration direction. |
| sam2-cells-seg | [DnaRnaProteins/sam2-cells-seg](https://huggingface.co/DnaRnaProteins/sam2-cells-seg) | Apache-2.0 model card. | Good fluorescence-cell candidate, but currently best treated as a server-side PyTorch integration unless converted and verified for ONNX. |
| OrgaSegment / OrgaSeg2 | [OrgaSegment](https://github.com/kleelab-bch/OrganoSeg), [OrgaSeg2](https://github.com/yu-lab-vt/OrgaSeg2) | OrgaSegment is MIT but uses an old TensorFlow 1 stack; OrgaSeg2 needs an explicit upstream license before adoption. | Organoid-specific approaches are relevant, but runtime age or unclear terms make them lower priority than the current prompt-based backend. |

## Next Implementation Order

1. Create a small manually annotated validation subset and report object-level
   precision/recall and mask IoU instead of relying only on visual overlays.
2. Tune deep-well candidate selection against those annotations, especially for
   faint and partially cropped edge spheroids.
3. Add MicroSAM ViT-T LM in the isolated MicroSAM environment and compare its
   memory use, runtime, and accuracy with ViT-B LM.
4. Run SpheroScan as an isolated GPL benchmark on per-well crops; do not import
   it into the main server process.
5. Revisit a combined ONNX/PyTorch server environment only after model accuracy
   is measured and the operational benefit is clear.

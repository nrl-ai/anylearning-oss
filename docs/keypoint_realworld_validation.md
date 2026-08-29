# Real-world RF-DETR keypoint validation

## Dataset

The validation dataset is [Spondylolisthesis Vertebral Landmark, version
1](https://doi.org/10.17632/5jdfdgp762.1), published by Karla Reyes on Mendeley
Data. It contains 698 annotated sagittal lumbar-spine X-rays in its training
and validation splits. Every visible vertebra is a separate instance with four
corner landmarks, so it exercises multi-instance grouping and an arbitrary
non-human skeleton rather than repeating RF-DETR's original COCO-person task.

The dataset page assigns the archive **CC BY 4.0**. That permits commercial use
and adaptation with attribution, a license link, and an indication of changes.
The archive is not bundled with AnyLearning: the preparation tool downloads
the fixed DOI version directly, verifies its SHA-256, and embeds an attribution
file in each derived archive.

There is still a provenance caveat for anyone considering redistribution. The
publisher says 208 images came from a proprietary Honduran source and 508 from
BUU-LSPINE. Mendeley applies CC BY 4.0 to the published dataset, but its license
notice also says separately identified third-party content may need additional
permission. Direct local download for engineering validation is the intended
use here; shipping the pixels in an installer or marketing asset needs a fresh
rights review.

This is an engineering benchmark, not a clinical validation dataset, and no
result from it makes AnyLearning or its models suitable for medical diagnosis.

## Preparing it

From the repository root:

```bash
python prepare_vertebral_keypoints.py \
  --download \
  --output-dir /path/to/vertebral-keypoints
```

This writes `vertebral-keypoints-train.zip` and
`vertebral-keypoints-valid.zip`, ready for the Keypoint Detection project's
train and validation upload slots. The complete converted dataset contains:

| split      | images | vertebra instances |
| ---------- | -----: | -----------------: |
| train      |    494 |              3,151 |
| validation |    204 |              1,449 |

The source archive has an undocumented inconsistency: its training labels are
339 Keypoint R-CNN JSON files plus 155 YOLO-pose text files; validation is 145
JSON plus 59 text. The two encodings also order the four corners differently.
The converter reads both, assigns the corners geometrically, and emits the one
schema `top_left`, `top_right`, `bottom_left`, `bottom_right`. Binary source
visibility `1` becomes COCO's visible value `2`.

For a quick mechanical run rather than an accuracy experiment:

```bash
python prepare_vertebral_keypoints.py \
  --archive /path/to/spondylolisthesis-vertebral-landmark-v1.zip \
  --output-dir /path/to/vertebral-keypoints-small \
  --train-limit 32 \
  --valid-limit 8
```

## Exercising a packaged build

`realworld_keypoint_test.py` drives the actual packaged application over HTTP:
project creation, both COCO imports, RF-DETR training, ONNX-gated model
registration, and held-out inference.

```bash
python realworld_keypoint_test.py \
  ./twin-out/app.dist/app.bin \
  /path/to/vertebral-keypoints-small/vertebral-keypoints-train.zip \
  /path/to/vertebral-keypoints-small/vertebral-keypoints-valid.zip \
  --epochs 3 --batch-size 2 --image-size 256 --device cpu \
  --preview /path/to/keypoint-preview.png
```

The bounded 32/8-image, three-epoch CPU run on 2026-08-18 imported every image,
trained, exported ONNX, and registered a model. Validation loss decreased from
25.43 to 19.59, while box and keypoint AP remained 0.0. That is a useful
integration result and an intentionally meaningless accuracy result: 32
images, 256-pixel input, and three epochs are far below a defensible training
regime for adapting COCO-person weights to radiographs.

The run also found a frozen-build defect that the synthetic test could not:
RF-DETR imports `supervision` only during prediction, which reaches PyAV, whose
Cython extensions dynamically import `av.utils`. Nuitka had omitted that
hidden module. `build_app.sh` now includes it explicitly, and held-out packaged
inference through this harness was the acceptance gate for the fix.

The fresh twin at `d092163` did: Nuitka included `av.utils` as an extension
module, the build and smoke test passed, `feature_test.py` passed 19/19 checks
with its one intentional browser-only skip, and held-out X-ray inference
returned HTTP 200 with structured results and a visualization. The deliberately
undertrained model returned no detections at its normal confidence threshold,
which is consistent with its measured 0 AP and is not reported as an accuracy
success.

## What a model-quality experiment requires

Use the complete 494/204 split, the model's native 576-pixel resolution, a GPU,
and a realistic fine-tuning schedule. Report keypoint AP/OKS and box AP on the
held-out validation split, plus visual failures. Do not infer model quality
from successful export, falling loss, or a few attractive overlays.

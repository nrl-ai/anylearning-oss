# Keypoint detection examples

AnyLearning 0.26.2 treats keypoints as named landmarks grouped into object
instances. These examples cover a quick first project and a real-world packaged
application test.

## Quick generated project

Generate two COCO archives without downloading external data:

```bash
python examples/keypoint_detection/generate_dataset.py \
  --output-dir /tmp/anylearning-keypoints
```

In AnyLearning:

1. Create a **Keypoint Detection** project.
2. Upload `stick-figures-train.zip` to Train and
   `stick-figures-valid.zip` to Validation.
3. Confirm that the labels are `head`, `left_hand`, `right_hand`, `left_foot`,
   and `right_foot` in that order.
4. Open an image. Each image has two instance numbers; some right-hand points
   are marked occluded to demonstrate COCO's visibility state.
5. Start training with **RF-DETR-Keypoint-Preview**. The normal learning-rate
   default is `0.0001`.
6. When the run finishes, open Models, try the model on a validation image, and
   export its ONNX file.

The figures are intentionally simple and generated under CC0. They demonstrate
the workflow; they are not an accuracy benchmark.

## Real-world vertebral landmarks

For a non-human, multi-instance test on real X-rays, follow
[the real-world validation guide](../../docs/keypoint_realworld_validation.md).
The repository provides:

- `prepare_vertebral_keypoints.py`, which downloads the fixed dataset version,
  verifies its SHA-256, normalizes both source annotation encodings, and emits
  import-ready COCO archives; and
- `realworld_keypoint_test.py`, which creates a project in a packaged app,
  imports train and validation splits, trains, verifies ONNX-gated model
  registration, runs held-out inference, and saves a visual preview.

The source dataset is not bundled. Its dataset-level license is CC BY 4.0, and
the guide records the third-party provenance caveat that limits this workflow
to engineering validation unless the pixels receive a separate rights review.

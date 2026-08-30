# Real-model inference validation

Real model files are intentionally not committed. To run a licensed local ONNX
artifact through the same inference contract used by applications and servers,
create a JSON manifest and set `ANYLEARNING_REAL_MODEL_MANIFEST` to its path:

```json
{
  "name": "local-yolox-check",
  "backend": "yolo_onnx",
  "provenance": {
    "source_url": "https://example.org/official-model-repository",
    "artifact_url": "https://example.org/immutable-model.onnx",
    "source_revision": "immutable-revision-or-release",
    "code_license": "SPDX-identifier",
    "artifact_license": "license verified for this model artifact",
    "license_url": "https://example.org/official-model-license"
  },
  "config": {
    "name": "local-yolox",
    "model_path": "models/yolox_tiny.onnx",
    "sha256": "<64-character SHA-256>",
    "task": "detection",
    "format": "yolox",
    "class_names": ["class-0", "class-1"]
  },
  "runs": 3,
  "lifecycle_cycles": 3,
  "maximum_steady_state_rss_growth_bytes": 67108864,
  "images": [
    {
      "path": "images/example.jpg",
      "request_parameters": { "confidence": 0.3, "iou": 0.45 },
      "expected": {
        "minimum_shapes": 1,
        "maximum_shapes": 20,
        "minimum_label_counts": { "class-0": 1 },
        "detections": [
          {
            "label": "class-0",
            "box": [10, 20, 100, 120],
            "minimum_iou": 0.7,
            "minimum_score": 0.3
          }
        ]
      }
    }
  ]
}
```

Paths are relative to the manifest. Run:

```bash
python scripts/validate_real_model.py path/to/manifest.json
ANYLEARNING_REAL_MODEL_MANIFEST=path/to/manifest.json pytest -q tests/e2e/test_real_onnx_models_e2e.py
```

Each run creates a new directory under `validation-results/` with `summary.json`,
all repeated contract results, annotated PNGs, timing and memory metadata, model
and image SHA-256 values, external-file sizes/digests when present, and a local
`index.html` visual report. The directory is git-ignored so multi-gigabyte model
assets and local evidence cannot be committed accidentally.

The steady-state RSS limit is calibrated per model from retained results on all
hosted operating systems. It includes native runtime and platform allocator
caches, not only live tensors. Raising a limit therefore requires an exact
before/after report and visual/consistency review; keep only bounded headroom
above the highest valid platform result so a later memory regression still
fails the gate.

The hosted real-model workflow also derives an external-data form of the same
official ONNX model using `scripts/prepare_external_onnx_validation.py`. It runs
both forms through identical golden detections and lifecycle cycles on Linux,
Windows, and macOS, and uploads the visual/machine-readable reports per OS.

Promptable cases add `prompts` and `output_shape` to each image. Point prompts
use `{ "type": "point", "point": [x, y], "foreground": true }`; box prompts
use `{ "type": "box", "box": [x1, y1, x2, y2] }`. Split models configure
`encoder_model_path`, `decoder_model_path`, and independent graph digests.
Reports draw the prompt over the predicted polygon so coordinate-scaling or
channel-order defects are visible during review. The committed
`efficient_sam_ti_official.json` is a complete real point/box example.

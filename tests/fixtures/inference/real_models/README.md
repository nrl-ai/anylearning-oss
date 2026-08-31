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
    "license_url": "https://example.org/official-model-license",
    "weight_license_confirmation": "https://example.org/optional-maintainer-confirmation"
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

For a composed pipeline, keep the primary artifact in `provenance` and record
every additional model in `component_provenance`. The retained report preserves
both records and recursively hashes every nested graph, rather than presenting
only one child model as evidence. The committed
`detector_sam_yolox_mobile_sam_official.json` fixture demonstrates this with
the checksum-pinned official YOLOX-S graph and MobileSAM encoder/decoder pair.

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
hosted operating systems. With three or more lifecycle cycles, the first
repeated load/unload establishes the warmed native-runtime and platform-
allocator baseline; the report records its one-time retention separately as
`warmup_retained_rss_growth_bytes`. The leak gate then measures later growth
from that warmed baseline. Two-cycle manifests conservatively keep the cold
baseline because they lack a second post-warm-up sample. Raising a limit
therefore requires an exact before/after report and visual/consistency review;
keep only bounded headroom above the highest valid platform result so a later
memory regression still fails the gate.

The hosted real-model workflow also derives an external-data form of the same
official ONNX model using `scripts/prepare_external_onnx_validation.py`. It runs
both forms through identical golden detections and lifecycle cycles on Linux,
Windows, and macOS, and uploads the visual/machine-readable reports per OS.

Promptable cases add `prompts` and `output_shape` to each image. Point prompts
use `{ "type": "point", "point": [x, y], "foreground": true }`; box prompts
use `{ "type": "box", "box": [x1, y1, x2, y2] }`; bounded text prompts use
`{ "type": "text", "text": "concept" }`. Split models configure
`encoder_model_path`, `decoder_model_path`, and independent graph digests.
SAM3 manifests configure image encoder, language encoder, and decoder paths,
independent graph digests, and exact external-data maps for each graph.
Reports draw the prompt over the predicted polygon so coordinate-scaling or
channel-order defects are visible during review. The committed
`efficient_sam_ti_official.json` is a complete real point/box example;
`efficient_sam_s_official.json` and `sam_vit_b_official.json` exercise the
larger scheduled graph variants. `sam2_hiera_small_official.json`,
`sam2_hiera_base_plus_official.json`, and `sam2_hiera_large_official.json`
exercise every larger prepared SAM2 encoder against the unchanged decoder
contract. The corresponding `sam2_1_hiera_*_official.json` manifests exercise
every SAM 2.1 size from immutable, checksum-pinned prepared archives. Each
archive pins the encoder, decoder, configuration, and bundled license. Tiny remains a
cross-platform pull-request gate; Small, Base+, and Large run in the scheduled
or manually dispatched resource-qualified matrix.
`efficientvit_sam_l0_official.json` downloads the immutable official encoder
and decoder, checksum-gates a deterministic ONNX-only decoder output transform,
and validates point/box prompts in landscape and portrait orientations both
directly and through password-authenticated HTTP. L0 is the per-pull-request
graph-contract gate; larger EfficientViT-SAM variants are scheduled/manual
artifact-size cases after their exact transformed digests are published. A
downstream matrix verifier compares direct/server prediction digests and decoded
annotated pixels across Linux, Windows, and macOS; per-run success alone is not
sufficient.
`sam3_vit_h_official.json` covers text, text+point, and box inference. These
larger cases run only in the scheduled/manual resource-qualified workflow so
per-PR checks do not repeatedly download hundreds of megabytes or several
gigabytes.

`rfdetr_nano_detection_official.json` and
`rfdetr_nano_segmentation_official.json` use the immutable, license-complete
RF-DETR 1.9.4 archives. They validate sparse COCO class IDs, float32 resizing,
sigmoid multiclass top-k, detection boxes, editable instance polygons,
landscape/portrait behavior, repeated lifecycle growth, authenticated-server
transport, and decoded-pixel identity across Linux, Windows, and macOS.

`dfine_n_coco_official.json` uses the immutable, license-complete COCO-N
archive. It validates the native direct-stretch preprocessing contract,
contiguous COCO-80 labels, embedded top-k boxes/scores, landscape/portrait
behavior, repeated lifecycle growth, authenticated-server transport, and exact
cross-platform prediction identity. Direct/server pixels must match exactly on
each platform; the matrix permits at most two cross-platform decoded source
pixels to differ by one channel value, covering the measured macOS JPEG decoder
variance without masking prediction drift. Objects365-derived weights are
excluded from the artifact and test path.

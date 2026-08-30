# Headless inference

`anylearning.inference` is the shared, UI-free inference boundary designed for
the desktop application and authenticated inference server. Backends receive a
decoded `uint8` RGB image and an `InferenceRequest`, and return versioned,
identity-preserving editable shapes.

Importing the package root does not import ONNX Runtime, OpenCV, PyTorch,
FastAPI, or a desktop framework. Model runtimes are loaded only when a registry
backend is selected.

## Promptable ONNX segmentation

`segment_anything` runs SAM, MobileSAM, SAM 2, and SAM 2.1 split ONNX pairs.
The decoder graph selects the compatible SAM generation, or production
manifests can pin `family` to `sam` or `sam2`. `efficient_sam` is a distinct
backend for the official EfficientSAM-Ti/S split graph contract; it is not the
similarly named EfficientViT-SAM architecture. Both backends accept point and
box prompts, cache image embeddings by model revision and source identity, and
select the highest-IoU candidate mask.

`sam3` is a separate three-graph backend for text-driven and geometrically
guided segmentation. It accepts one bounded `TextPrompt`, plus point or box
geometry up to the decoder graph's fixed exported capacity. Text-only requests
discover matching instances; text plus geometry narrows the concept; a
geometry-only request uses the graph's generic `visual` token. The image,
language, and decoder graphs and both external tensor files have independent
SHA-256 manifests and contribute to one triplet-bound model revision.

SAM3 filtering stays bounded before full-resolution postprocessing: raw query
count, NMS candidates, retained instances, mask elements, contours, shapes, and
polygon points each have independent limits. Native mask-IoU NMS operates on
bit-packed samples, and only retained masks are resized. The default one-item
image-feature cache is intentionally small because a single ViT-H embedding is
roughly 223 MB. On Linux/glibc, unload also returns released multi-gigabyte CPU
allocations to the operating system; set
`release_cpu_memory_on_unload=false` only after measuring a long-lived worker.

SAM3 weights remain under Meta's separate SAM License. They are never bundled
into the Apache-2.0 Python package; deployments must acquire them separately,
retain the model license, and provide exact graph/external-data digests.

Every graph is independently bounded and optionally digest-verified before
ONNX Runtime sees it. Large pairs use separate
`encoder_external_data_sha256` and `decoder_external_data_sha256` maps. SAM and
MobileSAM apply the official longest-side resize and aspect-ratio-aware
low-resolution mask crop; EfficientSAM preserves its dynamic native image size.
All image boundaries are explicitly `uint8` RGB.

Promptable pairs disable ONNX Runtime's per-session CPU memory arena by default.
This keeps repeated model load/unload cycles bounded on desktop and server hosts
without changing warm prompt decoding. A long-lived, throughput-oriented worker
that has measured sufficient memory headroom may set
`enable_cpu_mem_arena=true`; real-model validation should be rerun for that
deployment profile.

## User-supplied YOLO ONNX models

The `yolo_onnx` backend implements neutral output layouts for YOLOv5, YOLOv8,
YOLOv9, YOLOv10, YOLO11, YOLO12, and YOLO26 detection and instance
segmentation. It does not include model implementation code, configuration, or
weights. The user is responsible for the rights to each supplied artifact.

```python
from anylearning.inference import InferenceRequest, get_default_registry

config = {
    "name": "local-detector",
    "model_path": "/models/detector.onnx",
    "sha256": "<verified 64-character digest>",
    "task": "detection",
    "format": "yolov8",
    "class_names": ["person", "vehicle"],
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "max_detections": 300,
}

session = get_default_registry().create_session("yolo_onnx", config)
session.load()
request = InferenceRequest(
    request_id="request-1",
    source_id="image-sha256:<decoded-pixel-digest>",
    model_id=session.capabilities.model_id,
    model_revision=session.capabilities.model_revision,
    parameters={
        "confidence": 0.35,
        "iou": 0.45,
        "class_names": ("person",),
    },
)
result = session.predict(request, rgb_image)
session.unload()
```

Models whose tensors are stored outside the graph add an exact digest map. Each
key is the relative `location` recorded by the graph; the map must cover every
referenced file and may not contain extras:

```python
config.update(
    {
        "model_path": "/models/large-detector/model.onnx",
        "sha256": "<graph SHA-256>",
        "external_data_sha256": {
            "weights-0001.bin": "<file SHA-256>",
            "weights-0002.bin": "<file SHA-256>",
        },
        "max_external_data_bytes": 100 * 1024**3,
    }
)
```

External files are read-only memory mapped and supplied to ONNX Runtime through
its external-initializer buffer API. This avoids a second multi-gigabyte disk
copy and avoids reading the complete bundle into Python memory. The graph and
all external digests contribute to the model revision, so embedding and result
caches cannot alias two bundles with different tensor data.

`format="auto"` accepts a raw tensor only when its channel count and orientation
identify one layout unambiguously. For actionable errors and stable deployment,
production configuration should set the exact model family explicitly. The
supported layouts are:

- YOLOv5: `[x_center, y_center, width, height, objectness, classes..., masks...]`
- YOLOv8/v9/v10/v11/v12/26 raw: `[x_center, y_center, width, height, classes..., masks...]`
- YOLOv10/26 end-to-end: `[x1, y1, x2, y2, confidence, class_id, masks...]`
- Either `batch × predictions × channels` or `batch × channels × predictions`
- Segmentation prototypes: `batch × mask_channels × height × width`

YOLOv10 and YOLO26 default to the end-to-end profile, whose graph has already
selected detections. Set `end_to_end=false` only for an explicitly verified raw
export. Conversely, set `end_to_end=true` for another family exported with the
same end-to-end contract. Request-level IoU and class-agnostic NMS settings are
ignored for end-to-end outputs and reported in result warnings.

The `yolox` format implements the official P5 grid/stride ONNX profile and has
an opt-in P6 mode. It performs the profile's top-left padding, RGB-contract to
BGR conversion, unnormalized byte-range input, objectness/class composition,
and bounded host NMS without importing its native framework. Its default NMS is
class-agnostic, matching the official ONNX Runtime reference; requests may set
`agnostic_nms=false` explicitly when preserving overlapping cross-class boxes is
more important than reference parity.

## Real-model validation reports

`scripts/validate_real_model.py` runs a manifest-defined local model and image
at least twice through the public inference contract. It fails expectations or
non-repeatable shapes, and writes `summary.json`, every contract result,
per-stage timings, model/image SHA-256 values, annotated PNGs, and a local HTML
visual report beneath `validation-results/`. External bundles and SAM3 graph
triplets additionally log each graph role and every tensor file's relative
location, bytes, and verified digest. See
`tests/fixtures/inference/real_models/README.md` for the manifest and opt-in
pytest workflow. Weights remain outside the repository.

Dynamic image inputs require an explicit bounded `input_size`. Multiple
rank-2/rank-3 prediction outputs or rank-4 prototype outputs require
`prediction_output` or `prototype_output` by name.

Outputs must have static dimensions by default so their allocation can be
bounded before execution. `allow_dynamic_outputs=true` exists for trusted local
artifacts only; a future remote server must additionally isolate that execution
in a deadline- and memory-bounded worker.

## Resource and artifact safety

The backend validates configuration and model size before parsing, verifies a
configured SHA-256, and binds graph parsing plus runtime loading to the same
opened artifact. On platforms without stable descriptor paths, it uses a private
temporary snapshot and removes it immediately after session construction. It
accepts external-data tensor references only with exact SHA-256 coverage and
ONNX Runtime 1.29 or newer. Relative-path containment, metadata whitelisting,
offset/length bounds, regular-file checks, symlink/hardlink rejection, secure
post-open identity checks, and a total byte/file ceiling are enforced before
runtime construction. Graph messages, nesting, inputs, outputs, nodes, decoded
image pixels, runtime output elements, box-coordinate magnitude, mask polygon
complexity, raw predictions, pre-NMS candidates, and final detections are also
bounded. Provider fallback is explicit in result warnings and session
capabilities.

ONNX execution is cooperatively cancellable before and after the runtime call.
An in-progress provider call is not preemptible in-process; the authenticated
server therefore applies queue, image, result, concurrency, and deadline bounds
around its per-model session worker.

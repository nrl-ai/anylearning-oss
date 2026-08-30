# Headless inference

`anylearning.inference` is the shared, UI-free inference boundary designed for
the desktop application and authenticated inference server. Backends receive a
decoded `uint8` RGB image and an `InferenceRequest`, and return versioned,
identity-preserving editable shapes.

Importing the package root does not import ONNX Runtime, OpenCV, PyTorch,
FastAPI, or a desktop framework. Model runtimes are loaded only when a registry
backend is selected.

## User-supplied YOLO ONNX models

The `yolo_onnx` backend implements neutral output layouts for YOLOv5, YOLOv8,
and YOLO11 detection and instance segmentation. It does not include model
implementation code, configuration, or weights. The user is responsible for
the rights to each supplied artifact.

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

`format="auto"` accepts a tensor only when its channel count and orientation
identify one layout unambiguously. For actionable errors and stable deployment,
production configuration should set `yolov5`, `yolov8`, or `yolo11` explicitly.
The supported raw layouts are:

- YOLOv5: `[x_center, y_center, width, height, objectness, classes..., masks...]`
- YOLOv8/11: `[x_center, y_center, width, height, classes..., masks...]`
- Either `batch × predictions × channels` or `batch × channels × predictions`
- Segmentation prototypes: `batch × mask_channels × height × width`

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
rejects external-data tensor references at every nesting level and limits graph
messages, nesting, inputs, outputs, nodes, decoded image pixels, runtime output
elements, box-coordinate magnitude, mask polygon complexity, raw predictions,
pre-NMS candidates, and final detections. Provider fallback is explicit in
result warnings and session capabilities.

ONNX execution is cooperatively cancellable before and after the runtime call.
An in-progress provider call is not preemptible in-process; the server profile
must add a deadline and worker-isolation boundary before exposing this backend.

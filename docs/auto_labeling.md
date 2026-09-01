# Desktop auto-labeling API

The desktop labeling workspace exposes prompted segmentation and automatic
detection/instance-segmentation through project-scoped endpoints. Preprocessing,
ONNX Runtime execution, postprocessing, and resource limits live in the shared
`anylearning.inference` backends documented in [`inference.md`](inference.md).

Supported project types are Object Detection, Image Segmentation, and Instance
Segmentation. Each model declares its compatible project types, tasks,
interaction mode, and output shapes; the API rejects incompatible combinations
instead of converting them silently.

## Discover and load models

`GET /api/projects/{project_id}/auto_labeling/models` returns only public model
metadata. Catalog models, imported models, and compatible ONNX exports from
project-trained RF-DETR runs appear in the same response. Important fields are:

- `name` and `display_name`
- `tasks`: `promptable_segmentation`, `detection`, or `instance_segmentation`
- `interaction_mode`: `prompted` or `automatic`
- `output_modes`: `polygon`, `rectangle`, or both
- `project_types`
- `has_downloaded` and `archive_size_bytes`
- `is_project_model` and `is_custom_model`

Load one selected model before inference:

```http
POST /api/projects/42/auto_labeling/load_model
Content-Type: application/json

{"model_name":"sam2_hiera_small_20240803"}
```

The load operation downloads a missing catalog bundle only on explicit
selection, verifies its pinned archive/member hashes and sizes, then creates the
bounded ONNX session. A `409` means another model operation is still active or
the model has not become ready; clients can inspect
`GET /api/projects/{project_id}/auto_labeling/status`.

## Prompted request

```http
POST /api/projects/42/auto_labeling/inference
Content-Type: application/json

{
  "model_name": "sam2_hiera_small_20240803",
  "data_item_id": 1261,
  "marks": [
    {"data": [78, 53], "label": 1, "type": "point"},
    {"data": [115, 56], "label": 0, "type": "point"},
    {"data": [178, 50, 222, 102], "label": 1, "type": "rectangle"}
  ],
  "output_shape": "polygon",
  "preload_data_item_ids": [1262, 1263]
}
```

- A point label of `1` includes an area and `0` excludes it.
- Rectangle coordinates are `[x1, y1, x2, y2]` and define an include box.
- `preload_data_item_ids` is bounded to seven images. Promptable backends may
  cache their embeddings; automatic models ignore preloading.
- `output_shape` is constrained by the selected model and project. Object
  Detection always produces rectangles.

The current desktop keeps prompt marks on the canvas while a model changes, so
the route deliberately drops them when the selected model is automatic.

## Automatic request

```http
POST /api/projects/42/auto_labeling/inference
Content-Type: application/json

{
  "model_name": "rfdetr_nano_detection",
  "data_item_id": 1261,
  "marks": [],
  "output_shape": "rectangle",
  "parameters": {}
}
```

Catalog detectors and segmenters use immutable COCO label spaces. The adapter
passes only class IDs whose names exactly match labels in the project; if there
are no matches, inference fails with a clear `400` rather than assigning the
wrong class. Project-trained and imported models carry their own explicit class
order.

The response wraps `AutoLabelingResult` in
`{"status":"success","result":...}`. Shapes retain subpixel points, class,
score, instance/group identity, model revision, warnings, and timing metadata.
The frontend converts the result into editable canvas shapes. The inference
request itself never persists annotations; manual save or the labeling
workspace's auto-save setting controls persistence. The annotation endpoint
separately enforces rectangle-only geometry for Object Detection projects.

## Imported models

The native desktop bridge can import a single-file YOLO-family ONNX graph up to
20 GiB. Supported profiles are YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12,
YOLO26, and YOLOX detection. The user must select the task and provide the exact
class order (or use the known COCO 80 order). External-data ONNX bundles are not
accepted by the desktop import dialog yet.

Imported graphs are copied into the managed application data directory and
registered through the safety-checked `yolo_onnx` backend. See
[`inference.md`](inference.md#user-supplied-yolo-onnx-models) for supported
tensor layouts and bounds, and [`onnx_model_sources.md`](onnx_model_sources.md)
for catalog provenance and licensing.

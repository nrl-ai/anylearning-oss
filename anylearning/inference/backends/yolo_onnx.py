"""License-neutral YOLO tensor decoding for user-supplied ONNX artifacts.

This module implements documented tensor layouts and does not bundle model code,
configuration, or weights from another project.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..contracts import (
    InferenceRequest,
    InferenceResult,
    InferenceShape,
    ModelCapabilities,
    ModelTask,
    Point,
    ShapeType,
)
from ..runtime import BaseInferenceSession, CancellationToken, InferenceBackend
from .onnx_safety import (
    local_artifact_revision,
    resolve_model_path,
    select_providers,
    stable_onnx_artifact,
    validate_onnx_artifact,
)

YoloFormat = Literal[
    "auto",
    "yolov5",
    "yolov8",
    "yolov9",
    "yolov10",
    "yolo11",
    "yolo12",
    "yolo26",
    "yolox",
]

_MAX_MODEL_BYTES = 20 * 1024**3
_MAX_IMAGE_PIXELS = 100_000_000
_MAX_OUTPUT_ELEMENTS = 25_000_000
_MAX_RAW_PREDICTIONS = 1_000_000
_MAX_NMS_CANDIDATES = 30_000
_MAX_COORDINATE_MAGNITUDE = 1_000_000.0
_MAX_POLYGON_POINTS = 4_096


def _output_selection_error(kind: str, candidates: Sequence[str]) -> str:
    shown = [name[:80] for name in candidates[:3]]
    suffix = "" if len(candidates) <= 3 else f", plus {len(candidates) - 3} more"
    return (
        f"Could not select one YOLO {kind} output; set {kind}_output. "
        f"Found {len(candidates)} candidate(s): {shown}{suffix}"
    )


def _validate_graph_output_budget(model: Any, config: YoloOnnxConfig) -> None:
    """Reject unsafe declared output shapes before ONNX Runtime parses the graph."""
    total_elements = 0
    for output in model.graph.output:
        if not output.type.HasField("tensor_type"):
            raise ValueError("YOLO ONNX graph outputs must be tensors")
        dimensions: list[int] = []
        for dimension in output.type.tensor_type.shape.dim:
            if not dimension.HasField("dim_value"):
                if config.allow_dynamic_outputs:
                    dimensions = []
                    break
                raise ValueError(
                    "YOLO ONNX outputs must have static dimensions; symbolic "
                    "outputs require allow_dynamic_outputs=true and must only be "
                    "used with a trusted model in an isolated worker"
                )
            if dimension.dim_value <= 0:
                raise ValueError(
                    "YOLO ONNX graph declares a non-positive output dimension"
                )
            dimensions.append(dimension.dim_value)
        if dimensions:
            total_elements += math.prod(dimensions)
            if total_elements > config.max_output_elements:
                raise ValueError(
                    f"Declared ONNX outputs contain {total_elements} elements; "
                    f"configured limit is {config.max_output_elements}"
                )


class YoloOnnxConfig(BaseModel):
    """Strict configuration for the neutral YOLO ONNX backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=512)
    model_path: Path
    config_file: Path | None = None
    model_revision: str | None = Field(default=None, min_length=1, max_length=512)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    task: Literal["detection", "instance_segmentation"] = "detection"
    format: YoloFormat = "auto"
    end_to_end: bool | None = None
    class_names: tuple[str, ...] = Field(min_length=1, max_length=10_000)
    input_size: tuple[int, int] | None = None
    prediction_output: str | None = Field(default=None, min_length=1, max_length=512)
    prototype_output: str | None = Field(default=None, min_length=1, max_length=512)
    confidence: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    iou: float = Field(default=0.45, ge=0, le=1, allow_inf_nan=False)
    mask_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    max_detections: int = Field(default=300, ge=1, le=1_000)
    max_model_bytes: int = Field(default=_MAX_MODEL_BYTES, ge=1, le=40 * 1024**3)
    max_image_pixels: int = Field(default=_MAX_IMAGE_PIXELS, ge=1, le=400_000_000)
    max_output_elements: int = Field(default=_MAX_OUTPUT_ELEMENTS, ge=1, le=100_000_000)
    max_raw_predictions: int = Field(default=_MAX_RAW_PREDICTIONS, ge=1, le=5_000_000)
    max_nms_candidates: int = Field(default=_MAX_NMS_CANDIDATES, ge=1, le=100_000)
    max_coordinate_magnitude: float = Field(
        default=_MAX_COORDINATE_MAGNITUDE,
        ge=1_024,
        le=1_000_000_000,
        allow_inf_nan=False,
    )
    max_polygon_points: int = Field(default=_MAX_POLYGON_POINTS, ge=3, le=100_000)
    allow_dynamic_outputs: bool = False
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    allow_cpu_fallback: bool = True
    intra_op_threads: int = Field(default=0, ge=0, le=256)
    inter_op_threads: int = Field(default=0, ge=0, le=256)
    yolox_p6: bool = False

    @field_validator("model_path", "config_file", mode="before")
    @classmethod
    def reject_empty_paths(cls, value: Any) -> Any:
        if value is not None and (not isinstance(value, (str, Path)) or not str(value)):
            raise ValueError("Model paths must be non-empty strings or paths")
        return value

    @field_validator("input_size", mode="before")
    @classmethod
    def normalize_input_size(cls, value: Any) -> Any:
        if isinstance(value, int):
            return (value, value)
        return value

    @field_validator("input_size")
    @classmethod
    def validate_input_size(
        cls, value: tuple[int, int] | None
    ) -> tuple[int, int] | None:
        if value is not None and (
            value[0] < 16 or value[1] < 16 or value[0] > 16_384 or value[1] > 16_384
        ):
            raise ValueError("input_size dimensions must be between 16 and 16384")
        return value

    @field_validator("class_names")
    @classmethod
    def validate_class_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name or len(name) > 1024 for name in value):
            raise ValueError("Class names must contain 1 to 1024 characters")
        if len(set(value)) != len(value):
            raise ValueError("Class names must be unique")
        return value

    @field_validator("providers")
    @classmethod
    def validate_provider_count(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 16:
            raise ValueError("providers must contain between 1 and 16 names")
        return value

    @model_validator(mode="after")
    def validate_outputs(self) -> Self:
        if self.task == "detection" and self.prototype_output is not None:
            raise ValueError("prototype_output is only valid for instance segmentation")
        if self.format == "yolox" and self.task != "detection":
            raise ValueError("YOLOX currently supports detection only")
        if self.yolox_p6 and self.format != "yolox":
            raise ValueError("yolox_p6 is only valid when format='yolox'")
        if self.format == "yolox" and self.end_to_end:
            raise ValueError("YOLOX uses its raw grid output, not end-to-end output")
        return self

    @property
    def resolved_model_path(self) -> Path:
        return resolve_model_path(self.model_path, config_file=self.config_file)

    @property
    def revision(self) -> str:
        return local_artifact_revision(
            self.resolved_model_path,
            explicit_revision=self.model_revision,
            sha256=self.sha256,
        )

    @property
    def uses_end_to_end_output(self) -> bool:
        if self.end_to_end is not None:
            return self.end_to_end
        return self.format in {"yolov10", "yolo26"}


@dataclass(frozen=True)
class ImageTransform:
    original_height: int
    original_width: int
    input_height: int
    input_width: int
    scale: float
    pad_x: float
    pad_y: float

    def box_to_original(self, box: np.ndarray) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(item) for item in box)
        x1 = (x1 - self.pad_x) / self.scale
        x2 = (x2 - self.pad_x) / self.scale
        y1 = (y1 - self.pad_y) / self.scale
        y2 = (y2 - self.pad_y) / self.scale
        return (
            min(max(x1, 0.0), float(self.original_width)),
            min(max(y1, 0.0), float(self.original_height)),
            min(max(x2, 0.0), float(self.original_width)),
            min(max(y2, 0.0), float(self.original_height)),
        )


@dataclass(frozen=True)
class DecodedDetection:
    box: tuple[float, float, float, float]
    confidence: float
    class_id: int
    source_index: int
    mask_coefficients: np.ndarray | None = None


def _layout_candidates(
    dimensions: tuple[int, int], class_count: int, mask_dim: int
) -> list[tuple[str, bool]]:
    candidates: list[tuple[str, bool]] = []
    for layout, channel_count in (
        ("yolov5", 5 + class_count + mask_dim),
        ("yolov8", 4 + class_count + mask_dim),
    ):
        if dimensions[1] == channel_count:
            candidates.append((layout, False))
        if dimensions[0] == channel_count:
            candidates.append((layout, True))
    return candidates


def normalize_yolo_tensor(
    output: Any,
    *,
    class_count: int,
    mask_dim: int = 0,
    layout: YoloFormat = "auto",
    max_output_elements: int = _MAX_OUTPUT_ELEMENTS,
    max_raw_predictions: int = _MAX_RAW_PREDICTIONS,
) -> tuple[np.ndarray, Literal["yolov5", "yolov8"]]:
    """Return an ``N x channels`` tensor and an unambiguous layout."""
    array = np.asarray(output)
    if array.size > max_output_elements:
        raise ValueError(
            f"YOLO output has {array.size} elements; limit is {max_output_elements}"
        )
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("YOLO backend currently requires an output batch of 1")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(
            f"YOLO prediction output must have rank 2 or 3; received {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"YOLO prediction output must be numeric; received {array.dtype}"
        )

    requested = (
        "yolov8"
        if layout in {"yolov9", "yolov10", "yolo11", "yolo12", "yolo26"}
        else layout
    )
    if requested == "yolox":
        raise ValueError("YOLOX tensors require the YOLOX grid decoder")
    candidates = _layout_candidates(tuple(array.shape), class_count, mask_dim)
    if requested != "auto":
        candidates = [item for item in candidates if item[0] == requested]
    if len(candidates) != 1:
        expected_v5 = 5 + class_count + mask_dim
        expected_v8 = 4 + class_count + mask_dim
        detail = "no matching layout" if not candidates else "ambiguous orientation"
        raise ValueError(
            f"YOLO tensor layout has {detail}: shape={tuple(array.shape)}, "
            f"classes={class_count}, mask_channels={mask_dim}, expected channel "
            f"counts v5={expected_v5} or v8+={expected_v8}; set format explicitly "
            "and verify the exported output"
        )
    resolved, transpose = candidates[0]
    rows = array.T if transpose else array
    if rows.shape[0] > max_raw_predictions:
        raise ValueError(
            f"YOLO output has {rows.shape[0]} predictions; limit is "
            f"{max_raw_predictions}"
        )
    rows = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(rows).all():
        raise ValueError("YOLO output contains NaN or infinity")
    return rows, resolved  # type: ignore[return-value]


def decode_yolo_tensor(
    output: Any,
    *,
    class_count: int,
    confidence: float,
    class_ids: frozenset[int] | None = None,
    mask_dim: int = 0,
    layout: YoloFormat = "auto",
    max_output_elements: int = _MAX_OUTPUT_ELEMENTS,
    max_raw_predictions: int = _MAX_RAW_PREDICTIONS,
    max_candidates: int = _MAX_NMS_CANDIDATES,
    max_coordinate_magnitude: float = _MAX_COORDINATE_MAGNITUDE,
) -> list[DecodedDetection]:
    """Decode v5 or v8+ raw rows before non-maximum suppression."""
    rows, resolved = normalize_yolo_tensor(
        output,
        class_count=class_count,
        mask_dim=mask_dim,
        layout=layout,
        max_output_elements=max_output_elements,
        max_raw_predictions=max_raw_predictions,
    )
    class_start = 5 if resolved == "yolov5" else 4
    class_end = class_start + class_count
    class_scores = rows[:, class_start:class_end]
    if class_scores.size and (
        float(class_scores.min()) < 0 or float(class_scores.max()) > 1
    ):
        raise ValueError("YOLO class scores must be probabilities in the range [0, 1]")
    class_indices = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(rows.shape[0]), class_indices]
    if resolved == "yolov5":
        objectness = rows[:, 4]
        if objectness.size and (
            float(objectness.min()) < 0 or float(objectness.max()) > 1
        ):
            raise ValueError("YOLO objectness scores must be in the range [0, 1]")
        scores = scores * objectness

    selected = scores >= confidence
    if class_ids is not None:
        selected &= np.fromiter(
            (int(item) in class_ids for item in class_indices),
            dtype=bool,
            count=len(class_indices),
        )
    selected_indices = np.flatnonzero(selected)
    if len(selected_indices) > max_candidates:
        order = np.argsort(-scores[selected_indices], kind="stable")
        selected_indices = selected_indices[order[:max_candidates]]
    detections: list[DecodedDetection] = []
    for index in selected_indices:
        center_x, center_y, width, height = rows[index, :4]
        if np.max(np.abs(rows[index, :4])) > max_coordinate_magnitude:
            raise ValueError(
                "YOLO box coordinates exceed the configured magnitude limit"
            )
        if width <= 0 or height <= 0:
            continue
        coefficients = (
            rows[index, class_end : class_end + mask_dim].copy() if mask_dim else None
        )
        detections.append(
            DecodedDetection(
                box=(
                    float(center_x - width / 2),
                    float(center_y - height / 2),
                    float(center_x + width / 2),
                    float(center_y + height / 2),
                ),
                confidence=float(scores[index]),
                class_id=int(class_indices[index]),
                source_index=int(index),
                mask_coefficients=coefficients,
            )
        )
    return detections


def decode_end_to_end_yolo_tensor(
    output: Any,
    *,
    class_count: int,
    confidence: float,
    class_ids: frozenset[int] | None = None,
    mask_dim: int = 0,
    max_output_elements: int = _MAX_OUTPUT_ELEMENTS,
    max_raw_predictions: int = _MAX_RAW_PREDICTIONS,
    max_candidates: int = _MAX_NMS_CANDIDATES,
    max_coordinate_magnitude: float = _MAX_COORDINATE_MAGNITUDE,
) -> list[DecodedDetection]:
    """Decode NMS-free ``xyxy, confidence, class_id[, masks...]`` outputs."""
    array = np.asarray(output)
    if array.size > max_output_elements:
        raise ValueError(
            f"YOLO output has {array.size} elements; limit is {max_output_elements}"
        )
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("YOLO backend currently requires an output batch of 1")
        array = array[0]
    expected_channels = 6 + mask_dim
    if array.ndim != 2 or array.shape[1] != expected_channels:
        raise ValueError(
            "End-to-end YOLO output must have shape "
            f"[batch,predictions,{expected_channels}]; received {array.shape}"
        )
    if array.shape[0] > max_raw_predictions:
        raise ValueError(
            f"YOLO output has {array.shape[0]} predictions; limit is "
            f"{max_raw_predictions}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"YOLO prediction output must be numeric; received {array.dtype}"
        )
    rows = np.asarray(array, dtype=np.float32)
    if not np.isfinite(rows).all():
        raise ValueError("YOLO output contains NaN or infinity")
    if rows.size and np.max(np.abs(rows[:, :4])) > max_coordinate_magnitude:
        raise ValueError("YOLO box coordinates exceed the configured magnitude limit")
    scores = rows[:, 4]
    if scores.size and (float(scores.min()) < 0 or float(scores.max()) > 1):
        raise ValueError("YOLO confidence scores must be in the range [0, 1]")
    raw_class_ids = rows[:, 5]
    rounded_class_ids = np.rint(raw_class_ids)
    if not np.allclose(raw_class_ids, rounded_class_ids, rtol=0, atol=1e-4):
        raise ValueError("YOLO end-to-end class IDs must be integers")
    resolved_class_ids = rounded_class_ids.astype(np.int64)
    if resolved_class_ids.size and (
        int(resolved_class_ids.min()) < 0
        or int(resolved_class_ids.max()) >= class_count
    ):
        raise ValueError("YOLO end-to-end class ID is outside configured classes")

    selected = scores >= confidence
    if class_ids is not None:
        selected &= np.isin(resolved_class_ids, tuple(class_ids))
    selected_indices = np.flatnonzero(selected)
    order = np.argsort(-scores[selected_indices], kind="stable")
    selected_indices = selected_indices[order[:max_candidates]]
    detections: list[DecodedDetection] = []
    for index in selected_indices:
        x1, y1, x2, y2 = rows[index, :4]
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            DecodedDetection(
                box=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(scores[index]),
                class_id=int(resolved_class_ids[index]),
                source_index=int(index),
                mask_coefficients=(
                    rows[index, 6 : 6 + mask_dim].copy() if mask_dim else None
                ),
            )
        )
    return detections


def decode_yolox_tensor(
    output: Any,
    *,
    class_count: int,
    input_height: int,
    input_width: int,
    confidence: float,
    class_ids: frozenset[int] | None = None,
    p6: bool = False,
    max_output_elements: int = _MAX_OUTPUT_ELEMENTS,
    max_raw_predictions: int = _MAX_RAW_PREDICTIONS,
    max_candidates: int = _MAX_NMS_CANDIDATES,
    max_coordinate_magnitude: float = _MAX_COORDINATE_MAGNITUDE,
) -> list[DecodedDetection]:
    """Decode the documented YOLOX grid/stride ONNX output contract."""
    array = np.asarray(output)
    if array.size > max_output_elements:
        raise ValueError(
            f"YOLOX output has {array.size} elements; limit is {max_output_elements}"
        )
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("YOLOX backend currently requires an output batch of 1")
        array = array[0]
    expected_channels = 5 + class_count
    if array.ndim != 2 or array.shape[1] != expected_channels:
        raise ValueError(
            "YOLOX output must have shape "
            f"[batch,predictions,{expected_channels}]; received {array.shape}"
        )
    if array.shape[0] > max_raw_predictions:
        raise ValueError(
            f"YOLOX output has {array.shape[0]} predictions; limit is "
            f"{max_raw_predictions}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"YOLOX prediction output must be numeric; received {array.dtype}"
        )
    rows = np.asarray(array, dtype=np.float32)
    if not np.isfinite(rows).all():
        raise ValueError("YOLOX output contains NaN or infinity")

    strides = (8, 16, 32, 64) if p6 else (8, 16, 32)
    grids: list[np.ndarray] = []
    expanded_strides: list[np.ndarray] = []
    for stride in strides:
        grid_height = input_height // stride
        grid_width = input_width // stride
        x_coordinates, y_coordinates = np.meshgrid(
            np.arange(grid_width, dtype=np.float32),
            np.arange(grid_height, dtype=np.float32),
        )
        grid = np.stack((x_coordinates, y_coordinates), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((grid.shape[0], 1), stride, dtype=np.float32))
    grid = np.concatenate(grids, axis=0)
    stride_values = np.concatenate(expanded_strides, axis=0)
    if rows.shape[0] != grid.shape[0]:
        raise ValueError(
            f"YOLOX output has {rows.shape[0]} predictions but input "
            f"{input_width}x{input_height} with strides {strides} requires "
            f"{grid.shape[0]}"
        )

    objectness = rows[:, 4]
    class_scores = rows[:, 5:]
    if objectness.size and (
        float(objectness.min()) < 0
        or float(objectness.max()) > 1
        or float(class_scores.min()) < 0
        or float(class_scores.max()) > 1
    ):
        raise ValueError(
            "YOLOX objectness and class scores must be in the range [0, 1]"
        )
    largest_safe_log = math.log(max_coordinate_magnitude / min(strides))
    if rows.size and float(rows[:, 2:4].max()) > largest_safe_log:
        raise ValueError("YOLOX box dimensions exceed the configured magnitude limit")

    centers = (rows[:, :2] + grid) * stride_values
    dimensions = np.exp(rows[:, 2:4]) * stride_values
    boxes = np.concatenate((centers - dimensions / 2, centers + dimensions / 2), axis=1)
    if boxes.size and np.max(np.abs(boxes)) > max_coordinate_magnitude:
        raise ValueError("YOLOX box coordinates exceed the configured magnitude limit")
    resolved_class_ids = np.argmax(class_scores, axis=1)
    scores = (
        objectness * class_scores[np.arange(class_scores.shape[0]), resolved_class_ids]
    )
    selected = scores >= confidence
    if class_ids is not None:
        selected &= np.isin(resolved_class_ids, tuple(class_ids))
    selected_indices = np.flatnonzero(selected)
    order = np.argsort(-scores[selected_indices], kind="stable")
    selected_indices = selected_indices[order[:max_candidates]]
    detections: list[DecodedDetection] = []
    for index in selected_indices:
        x1, y1, x2, y2 = boxes[index]
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            DecodedDetection(
                box=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(scores[index]),
                class_id=int(resolved_class_ids[index]),
                source_index=int(index),
            )
        )
    return detections


def non_maximum_suppression(
    detections: Sequence[DecodedDetection],
    *,
    iou_threshold: float,
    max_detections: int,
    class_agnostic: bool = False,
) -> list[DecodedDetection]:
    """Deterministic, bounded NMS with stable tie-breaking."""
    ordered = sorted(
        detections,
        key=lambda item: (-item.confidence, item.class_id, item.source_index),
    )
    if not ordered:
        return []
    boxes = np.asarray([item.box for item in ordered], dtype=np.float32)
    classes = np.asarray([item.class_id for item in ordered], dtype=np.int64)
    remaining = np.arange(len(ordered))
    kept: list[DecodedDetection] = []
    while remaining.size and len(kept) < max_detections:
        chosen_index = int(remaining[0])
        kept.append(ordered[chosen_index])
        if remaining.size == 1:
            break
        others = remaining[1:]
        chosen_box = boxes[chosen_index]
        other_boxes = boxes[others]
        intersection_width = np.maximum(
            0.0,
            np.minimum(chosen_box[2], other_boxes[:, 2])
            - np.maximum(chosen_box[0], other_boxes[:, 0]),
        )
        intersection_height = np.maximum(
            0.0,
            np.minimum(chosen_box[3], other_boxes[:, 3])
            - np.maximum(chosen_box[1], other_boxes[:, 1]),
        )
        intersection = intersection_width * intersection_height
        chosen_area = max(0.0, chosen_box[2] - chosen_box[0]) * max(
            0.0, chosen_box[3] - chosen_box[1]
        )
        other_areas = np.maximum(
            0.0, other_boxes[:, 2] - other_boxes[:, 0]
        ) * np.maximum(0.0, other_boxes[:, 3] - other_boxes[:, 1])
        union = chosen_area + other_areas - intersection
        ious = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        suppressible = (
            np.ones_like(ious, dtype=bool)
            if class_agnostic
            else classes[others] == classes[chosen_index]
        )
        remaining = others[~(suppressible & (ious > iou_threshold))]
    return kept


def _request_options(
    request: InferenceRequest, config: YoloOnnxConfig
) -> tuple[float, float, int, bool, frozenset[int] | None]:
    supported = {
        "confidence",
        "iou",
        "max_detections",
        "agnostic_nms",
        "class_ids",
        "class_names",
    }
    unknown = set(request.parameters) - supported
    if unknown:
        raise ValueError("Unsupported YOLO parameters: " + ", ".join(sorted(unknown)))

    confidence = request.parameters.get("confidence", config.confidence)
    iou = request.parameters.get("iou", config.iou)
    maximum = request.parameters.get("max_detections", config.max_detections)
    # The official YOLOX ONNX reference defaults to class-agnostic NMS. Other
    # profiles retain the conventional class-aware default.
    agnostic = request.parameters.get("agnostic_nms", config.format == "yolox")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    if isinstance(iou, bool) or not isinstance(iou, (int, float)):
        raise ValueError("iou must be a number")
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise ValueError("max_detections must be an integer")
    if not isinstance(agnostic, bool):
        raise ValueError("agnostic_nms must be a boolean")
    if not 0 <= float(confidence) <= 1 or not 0 <= float(iou) <= 1:
        raise ValueError("confidence and iou must be in the range [0, 1]")
    if maximum < 1 or maximum > config.max_detections:
        raise ValueError(
            f"max_detections must be between 1 and configured limit "
            f"{config.max_detections}"
        )

    ids_value = request.parameters.get("class_ids")
    names_value = request.parameters.get("class_names")
    if ids_value is not None and names_value is not None:
        raise ValueError("Set class_ids or class_names, not both")
    class_ids: frozenset[int] | None = None
    if ids_value is not None:
        values = ids_value if isinstance(ids_value, tuple) else (ids_value,)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
            raise ValueError("class_ids must contain integers")
        class_ids = frozenset(int(item) for item in values)
    elif names_value is not None:
        values = names_value if isinstance(names_value, tuple) else (names_value,)
        if any(not isinstance(item, str) for item in values):
            raise ValueError("class_names must contain strings")
        by_name = {name: index for index, name in enumerate(config.class_names)}
        missing = sorted(set(values) - set(by_name))
        if missing:
            raise ValueError("Unknown class names: " + ", ".join(missing))
        class_ids = frozenset(by_name[item] for item in values)
    if class_ids is not None and any(
        item < 0 or item >= len(config.class_names) for item in class_ids
    ):
        raise ValueError("class_ids contains an index outside configured class names")
    return float(confidence), float(iou), maximum, agnostic, class_ids


def _prepare_image(
    image: Any,
    *,
    input_height: int,
    input_width: int,
    max_image_pixels: int,
    dtype: np.dtype[Any],
) -> tuple[np.ndarray, ImageTransform]:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("YOLO inference expects an H x W x 3 uint8 RGB image")
    height, width = array.shape[:2]
    if height <= 0 or width <= 0 or height * width > max_image_pixels:
        raise ValueError(
            f"Image dimensions {width}x{height} exceed the configured pixel limit"
        )
    scale = min(input_width / width, input_height / height)
    resized_width = max(1, min(input_width, round(width * scale)))
    resized_height = max(1, min(input_height, round(height * scale)))
    resized = cv2.resize(
        array,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=dtype)
    tensor /= np.asarray(255.0, dtype=dtype)
    return tensor, ImageTransform(
        original_height=height,
        original_width=width,
        input_height=input_height,
        input_width=input_width,
        scale=scale,
        pad_x=float(pad_x),
        pad_y=float(pad_y),
    )


def _prepare_yolox_image(
    image: Any,
    *,
    input_height: int,
    input_width: int,
    max_image_pixels: int,
    dtype: np.dtype[Any],
) -> tuple[np.ndarray, ImageTransform]:
    """Prepare contract RGB input for the official YOLOX ONNX profile."""
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("YOLOX inference expects an H x W x 3 uint8 RGB image")
    height, width = array.shape[:2]
    if height <= 0 or width <= 0 or height * width > max_image_pixels:
        raise ValueError(
            f"Image dimensions {width}x{height} exceed the configured pixel limit"
        )
    scale = min(input_width / width, input_height / height)
    resized_width = max(1, min(input_width, int(width * scale)))
    resized_height = max(1, min(input_height, int(height * scale)))
    resized = cv2.resize(
        array,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    # The published ONNX profile consumes the BGR byte range used by OpenCV,
    # without 1/255 normalization.
    tensor = np.ascontiguousarray(
        canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=dtype
    )
    return tensor, ImageTransform(
        original_height=height,
        original_width=width,
        input_height=input_height,
        input_width=input_width,
        scale=scale,
        pad_x=0.0,
        pad_y=0.0,
    )


def _mask_shape(
    detection: DecodedDetection,
    prototypes: np.ndarray,
    transform: ImageTransform,
    *,
    threshold: float,
    label: str,
    max_polygon_points: int,
) -> tuple[InferenceShape | None, int]:
    coefficients = detection.mask_coefficients
    if coefficients is None:
        return None, 0
    channels, proto_height, proto_width = prototypes.shape
    if coefficients.shape != (channels,):
        raise ValueError(
            f"Mask coefficient count {coefficients.size} does not match prototype "
            f"channels {channels}"
        )
    logits = coefficients @ prototypes.reshape(channels, -1)
    logits = np.clip(logits, -80, 80)
    mask = (1.0 / (1.0 + np.exp(-logits))).reshape(proto_height, proto_width)
    mask = cv2.resize(
        mask,
        (transform.input_width, transform.input_height),
        interpolation=cv2.INTER_LINEAR,
    )
    left = max(0, int(math.floor(transform.pad_x)))
    top = max(0, int(math.floor(transform.pad_y)))
    right = min(
        transform.input_width,
        int(math.ceil(transform.pad_x + transform.original_width * transform.scale)),
    )
    bottom = min(
        transform.input_height,
        int(math.ceil(transform.pad_y + transform.original_height * transform.scale)),
    )
    mask = mask[top:bottom, left:right]
    if mask.size == 0:
        return None, 0
    mask = cv2.resize(
        mask,
        (transform.original_width, transform.original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    binary = np.asarray(mask > threshold, dtype=np.uint8)
    x1, y1, x2, y2 = transform.box_to_original(np.asarray(detection.box))
    crop = np.zeros_like(binary)
    ix1, iy1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    ix2 = min(transform.original_width, int(math.ceil(x2)))
    iy2 = min(transform.original_height, int(math.ceil(y2)))
    crop[iy1:iy2, ix1:ix2] = binary[iy1:iy2, ix1:ix2]
    contours, hierarchy = cv2.findContours(
        crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, 0
    parents = (
        hierarchy[0, :, 3] if hierarchy is not None else np.full(len(contours), -1)
    )
    exterior = [
        contour for contour, parent in zip(contours, parents, strict=True) if parent < 0
    ]
    if not exterior:
        return None, len(contours)
    contour = max(exterior, key=cv2.contourArea)
    epsilon = max(0.5, 0.001 * cv2.arcLength(contour, True))
    points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    for _ in range(8):
        if len(points) <= max_polygon_points:
            break
        epsilon *= 2
        points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(points) > max_polygon_points:
        raise ValueError(
            f"Mask polygon exceeds the configured {max_polygon_points}-point limit"
        )
    if len(points) < 3:
        return None, len(contours)
    topology_loss = max(0, len(contours) - 1)
    attributes = {"class_id": detection.class_id}
    if topology_loss:
        attributes["discarded_mask_components"] = topology_loss
    return (
        InferenceShape(
            type=ShapeType.POLYGON,
            points=tuple(Point(x=float(x), y=float(y)) for x, y in points),
            label=label,
            score=detection.confidence,
            attributes=attributes,
        ),
        topology_loss,
    )


class YoloOnnxSession(BaseInferenceSession):
    """One bounded ONNX Runtime session for detection or instance masks."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = YoloOnnxConfig.model_validate(config)
        self._session: Any | None = None
        self._input_name = ""
        self._prediction_output = ""
        self._prototype_output: str | None = None
        self._input_height = 0
        self._input_width = 0
        self._input_dtype = np.dtype(np.float32)
        self._provider_warnings: tuple[str, ...] = ()
        super().__init__(YoloOnnxBackend().capabilities(config))

    def _load(self, cancellation: CancellationToken) -> None:
        path = self.config.resolved_model_path
        import onnxruntime

        providers, warnings = select_providers(
            self.config.providers,
            onnxruntime.get_available_providers(),
            allow_cpu_fallback=self.config.allow_cpu_fallback,
        )
        options = onnxruntime.SessionOptions()
        if self.config.intra_op_threads:
            options.intra_op_num_threads = self.config.intra_op_threads
        if self.config.inter_op_threads:
            options.inter_op_num_threads = self.config.inter_op_threads
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        cancellation.raise_if_cancelled()
        with stable_onnx_artifact(
            path,
            max_bytes=self.config.max_model_bytes,
            expected_sha256=self.config.sha256,
            cancellation=cancellation,
        ) as (artifact, runtime_path, _digest):
            model = validate_onnx_artifact(
                artifact, max_bytes=self.config.max_model_bytes
            )
            _validate_graph_output_budget(model, self.config)
            cancellation.raise_if_cancelled()
            session = onnxruntime.InferenceSession(
                runtime_path, sess_options=options, providers=list(providers)
            )
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"YOLO ONNX graph must expose exactly one runtime input; "
                f"received {len(inputs)}"
            )
        model_input = inputs[0]
        if len(model_input.shape) != 4:
            raise ValueError(
                f"YOLO input must be rank-4 NCHW; received {model_input.shape}"
            )
        batch, channels, graph_height, graph_width = model_input.shape
        if isinstance(batch, int) and batch != 1:
            raise ValueError("YOLO backend currently requires model batch size 1")
        if isinstance(channels, int) and channels != 3:
            raise ValueError("YOLO input must have three image channels")
        if model_input.type == "tensor(float)":
            dtype = np.dtype(np.float32)
        elif model_input.type == "tensor(float16)":
            dtype = np.dtype(np.float16)
        else:
            raise ValueError(
                f"YOLO input must be float32 or float16; received {model_input.type}"
            )
        configured_size = self.config.input_size
        if isinstance(graph_height, int) and isinstance(graph_width, int):
            graph_size = (graph_height, graph_width)
            if configured_size is not None and configured_size != graph_size:
                raise ValueError(
                    f"Configured input_size {configured_size} does not match graph "
                    f"{graph_size}"
                )
            input_size = graph_size
        elif configured_size is None:
            raise ValueError("Dynamic YOLO inputs require an explicit input_size")
        else:
            input_size = configured_size
        if any(dimension < 16 or dimension > 16_384 for dimension in input_size):
            raise ValueError(
                f"YOLO graph input dimensions must be between 16 and 16384; "
                f"received {input_size}"
            )
        if input_size[0] * input_size[1] > self.config.max_image_pixels:
            raise ValueError("Configured model input exceeds max_image_pixels")

        outputs = {item.name: item for item in session.get_outputs()}
        if len(outputs) > 128:
            raise ValueError("YOLO ONNX graph exposes more than 128 outputs")
        prediction_name = self.config.prediction_output
        if prediction_name is None:
            candidates = [
                item.name for item in outputs.values() if len(item.shape) in (2, 3)
            ]
            if len(candidates) != 1:
                raise ValueError(_output_selection_error("prediction", candidates))
            prediction_name = candidates[0]
        if prediction_name not in outputs:
            raise ValueError(f"Unknown prediction_output {prediction_name!r}")

        prototype_name: str | None = None
        if self.config.task == "instance_segmentation":
            prototype_name = self.config.prototype_output
            if prototype_name is None:
                candidates = [
                    item.name for item in outputs.values() if len(item.shape) == 4
                ]
                if len(candidates) != 1:
                    raise ValueError(_output_selection_error("prototype", candidates))
                prototype_name = candidates[0]
            if prototype_name not in outputs:
                raise ValueError(f"Unknown prototype_output {prototype_name!r}")
            if prototype_name == prediction_name:
                raise ValueError("Prediction and prototype outputs must be different")

        selected_output_names = [prediction_name]
        if prototype_name is not None:
            selected_output_names.append(prototype_name)
        declared_elements = 0
        fully_static = True
        for output_name in selected_output_names:
            output_shape = outputs[output_name].shape
            if not all(isinstance(dimension, int) for dimension in output_shape):
                fully_static = False
                continue
            if any(dimension <= 0 for dimension in output_shape):
                raise ValueError(
                    f"ONNX output {output_name!r} has a non-positive dimension"
                )
            declared_elements += math.prod(output_shape)
        if not fully_static and not self.config.allow_dynamic_outputs:
            raise ValueError(
                "YOLO ONNX outputs must have static dimensions; symbolic outputs "
                "require allow_dynamic_outputs=true and must only be used with a "
                "trusted model in an isolated worker"
            )
        if fully_static and declared_elements > self.config.max_output_elements:
            raise ValueError(
                f"Declared ONNX outputs contain {declared_elements} elements; "
                f"configured limit is {self.config.max_output_elements}"
            )

        del model
        self._session = session
        self._input_name = model_input.name
        self._prediction_output = prediction_name
        self._prototype_output = prototype_name
        self._input_height, self._input_width = input_size
        self._input_dtype = dtype
        self._provider_warnings = warnings
        self._capabilities = self._capabilities.model_copy(
            update={
                "metadata": {
                    **self._capabilities.metadata,
                    "providers": ",".join(session.get_providers()),
                }
            }
        )

    def _predict(
        self,
        request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken,
    ) -> InferenceResult:
        if self._session is None:
            raise RuntimeError("YOLO ONNX session is not loaded")
        confidence, iou, maximum, agnostic, class_ids = _request_options(
            request, self.config
        )
        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        prepare = (
            _prepare_yolox_image if self.config.format == "yolox" else _prepare_image
        )
        tensor, transform = prepare(
            image,
            input_height=self._input_height,
            input_width=self._input_width,
            max_image_pixels=self.config.max_image_pixels,
            dtype=self._input_dtype,
        )
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        cancellation.raise_if_cancelled()
        output_names = [self._prediction_output]
        if self._prototype_output is not None:
            output_names.append(self._prototype_output)
        inference_started = time.perf_counter()
        outputs = self._session.run(output_names, {self._input_name: tensor})
        inference_ms = (time.perf_counter() - inference_started) * 1000
        cancellation.raise_if_cancelled()
        total_elements = sum(np.asarray(item).size for item in outputs)
        if total_elements > self.config.max_output_elements:
            raise ValueError(
                f"ONNX outputs contain {total_elements} elements; configured limit "
                f"is {self.config.max_output_elements}"
            )

        prototypes: np.ndarray | None = None
        mask_dim = 0
        if self._prototype_output is not None:
            prototypes = np.asarray(outputs[1], dtype=np.float32)
            if prototypes.ndim != 4 or prototypes.shape[0] != 1:
                raise ValueError(
                    f"Mask prototypes must have shape [1,C,H,W]; received "
                    f"{prototypes.shape}"
                )
            prototypes = prototypes[0]
            if any(dimension <= 0 for dimension in prototypes.shape):
                raise ValueError("Mask prototype dimensions must be positive")
            if not np.isfinite(prototypes).all():
                raise ValueError("Mask prototypes contain NaN or infinity")
            mask_dim = int(prototypes.shape[0])

        postprocess_started = time.perf_counter()
        if self.config.format == "yolox":
            decoded = decode_yolox_tensor(
                outputs[0],
                class_count=len(self.config.class_names),
                input_height=self._input_height,
                input_width=self._input_width,
                confidence=confidence,
                class_ids=class_ids,
                p6=self.config.yolox_p6,
                max_output_elements=self.config.max_output_elements,
                max_raw_predictions=self.config.max_raw_predictions,
                max_candidates=self.config.max_nms_candidates,
                max_coordinate_magnitude=self.config.max_coordinate_magnitude,
            )
            detections = non_maximum_suppression(
                decoded,
                iou_threshold=iou,
                max_detections=maximum,
                class_agnostic=agnostic,
            )
        elif self.config.uses_end_to_end_output:
            decoded = decode_end_to_end_yolo_tensor(
                outputs[0],
                class_count=len(self.config.class_names),
                confidence=confidence,
                class_ids=class_ids,
                mask_dim=mask_dim,
                max_output_elements=self.config.max_output_elements,
                max_raw_predictions=self.config.max_raw_predictions,
                max_candidates=self.config.max_nms_candidates,
                max_coordinate_magnitude=self.config.max_coordinate_magnitude,
            )
            detections = decoded[:maximum]
        else:
            decoded = decode_yolo_tensor(
                outputs[0],
                class_count=len(self.config.class_names),
                confidence=confidence,
                class_ids=class_ids,
                mask_dim=mask_dim,
                layout=self.config.format,
                max_output_elements=self.config.max_output_elements,
                max_raw_predictions=self.config.max_raw_predictions,
                max_candidates=self.config.max_nms_candidates,
                max_coordinate_magnitude=self.config.max_coordinate_magnitude,
            )
            detections = non_maximum_suppression(
                decoded,
                iou_threshold=iou,
                max_detections=maximum,
                class_agnostic=agnostic,
            )
        shapes: list[InferenceShape] = []
        warnings = list(self._provider_warnings)
        if self.config.uses_end_to_end_output and (
            "iou" in request.parameters or "agnostic_nms" in request.parameters
        ):
            warnings.append(
                "Ignored request NMS settings because the ONNX graph returns "
                "end-to-end NMS-free detections"
            )
        empty_masks = 0
        discarded_mask_components = 0
        for detection in detections:
            label = self.config.class_names[detection.class_id]
            if prototypes is not None:
                shape, topology_loss = _mask_shape(
                    detection,
                    prototypes,
                    transform,
                    threshold=self.config.mask_threshold,
                    label=label,
                    max_polygon_points=self.config.max_polygon_points,
                )
                discarded_mask_components += topology_loss
                if shape is None:
                    empty_masks += 1
                    continue
                shapes.append(shape)
            else:
                x1, y1, x2, y2 = transform.box_to_original(np.asarray(detection.box))
                if x2 <= x1 or y2 <= y1:
                    continue
                shapes.append(
                    InferenceShape(
                        type=ShapeType.RECTANGLE,
                        points=(Point(x=x1, y=y1), Point(x=x2, y=y2)),
                        label=label,
                        score=detection.confidence,
                        attributes={"class_id": detection.class_id},
                    )
                )
        if empty_masks:
            warnings.append(f"Discarded {empty_masks} detections with empty masks")
        if discarded_mask_components:
            warnings.append(
                f"Reduced {discarded_mask_components} mask holes or disconnected "
                "components to editable exterior polygons"
            )
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        return InferenceResult(
            request_id=request.request_id,
            source_id=request.source_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
            shapes=tuple(shapes),
            warnings=tuple(warnings),
            timings_ms={
                "preprocess": preprocess_ms,
                "inference": inference_ms,
                "postprocess": postprocess_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        )

    def _unload(self) -> None:
        self._session = None
        self._provider_warnings = ()


class YoloOnnxBackend(InferenceBackend):
    backend_id = "yolo_onnx"

    def capabilities(self, config: Mapping[str, Any]) -> ModelCapabilities:
        parsed = YoloOnnxConfig.model_validate(config)
        task = (
            ModelTask.DETECTION
            if parsed.task == "detection"
            else ModelTask.INSTANCE_SEGMENTATION
        )
        return ModelCapabilities(
            model_id=parsed.name,
            model_revision=parsed.revision,
            tasks=(task,),
            supports_cancellation=True,
            metadata={
                "backend": self.backend_id,
                "format": parsed.format,
                "end_to_end": parsed.uses_end_to_end_output,
                "class_count": len(parsed.class_names),
                "artifact_policy": "user-supplied",
                "requested_providers": ",".join(parsed.providers),
            },
        )

    def create_session(self, config: Mapping[str, Any]) -> YoloOnnxSession:
        return YoloOnnxSession(config)

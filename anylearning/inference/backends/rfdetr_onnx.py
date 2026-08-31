"""Bounded ONNX inference for RF-DETR detection and instance segmentation.

The tensor contract and preprocessing flow are derived from RF-DETR 1.9.4 at
commit 9b009fa928d6218320439803d1da01869a85c072. RF-DETR code and the Nano through
Large model tier are Apache-2.0. The inference process itself depends only on
the shared AnyLearning ONNX boundary, NumPy, and OpenCV; it never imports or
deserializes the RF-DETR training runtime.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
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
from .onnx_safety import local_onnx_bundle_revision, resolve_model_path
from .onnx_session import create_checked_onnx_session, release_unused_cpu_memory

_MAX_MODEL_BYTES = 20 * 1024**3
_MAX_EXTERNAL_DATA_BYTES = 100 * 1024**3
_MAX_IMAGE_PIXELS = 100_000_000
_MAX_MODEL_INPUT_PIXELS = 16_777_216
_MAX_OUTPUT_ELEMENTS = 25_000_000
_MAX_QUERIES = 10_000
_MAX_CLASSES = 10_000
_MAX_MASK_COMPONENTS = 32
_MAX_POLYGON_POINTS = 4_096
_MAX_TOTAL_SHAPE_POINTS = 100_000
_FLOAT_TENSOR_TYPE = 1
_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class _FrozenStringMap(dict[str, str]):
    """Serializable dict whose validated integrity entries cannot be changed."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Validated ONNX integrity metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class RfDetrOnnxConfig(BaseModel):
    """Strict configuration for an official RF-DETR ONNX export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=512)
    model_path: Path
    config_file: Path | None = None
    model_revision: str | None = Field(default=None, min_length=1, max_length=512)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    external_data_sha256: dict[str, str] = Field(default_factory=dict)
    task: Literal["detection", "instance_segmentation"] = "detection"
    class_names: tuple[str | None, ...] = Field(min_length=1, max_length=_MAX_CLASSES)
    background_class_id: int | None = -1
    input_name: str = Field(default="input", min_length=1, max_length=512)
    boxes_output: str = Field(default="dets", min_length=1, max_length=512)
    logits_output: str = Field(default="labels", min_length=1, max_length=512)
    masks_output: str = Field(default="masks", min_length=1, max_length=512)
    supported_opsets: tuple[int, ...] = (17,)
    confidence: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)
    mask_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    max_detections: int = Field(default=300, ge=1, le=1_000)
    max_model_bytes: int = Field(default=_MAX_MODEL_BYTES, ge=1, le=40 * 1024**3)
    max_external_data_bytes: int = Field(
        default=_MAX_EXTERNAL_DATA_BYTES,
        ge=1,
        le=1024 * 1024**3,
    )
    max_image_pixels: int = Field(default=_MAX_IMAGE_PIXELS, ge=1, le=400_000_000)
    max_model_input_pixels: int = Field(
        default=_MAX_MODEL_INPUT_PIXELS,
        ge=1,
        le=100_000_000,
    )
    max_output_elements: int = Field(
        default=_MAX_OUTPUT_ELEMENTS,
        ge=1,
        le=100_000_000,
    )
    max_queries: int = Field(default=_MAX_QUERIES, ge=1, le=100_000)
    max_classes: int = Field(default=_MAX_CLASSES, ge=1, le=100_000)
    max_mask_components: int = Field(default=_MAX_MASK_COMPONENTS, ge=1, le=1_000)
    max_shapes: int = Field(default=1_000, ge=1, le=10_000)
    max_polygon_points: int = Field(
        default=_MAX_POLYGON_POINTS,
        ge=3,
        le=100_000,
    )
    max_total_shape_points: int = Field(
        default=_MAX_TOTAL_SHAPE_POINTS,
        ge=3,
        le=1_000_000,
    )
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    allow_cpu_fallback: bool = True
    enable_cpu_mem_arena: bool = False
    enable_mem_pattern: bool = False
    release_cpu_memory_on_unload: bool = True
    intra_op_threads: int = Field(default=0, ge=0, le=256)
    inter_op_threads: int = Field(default=0, ge=0, le=256)

    @field_validator("model_path", "config_file", mode="before")
    @classmethod
    def reject_empty_paths(cls, value: Any) -> Any:
        if value is not None and (not isinstance(value, (str, Path)) or not str(value)):
            raise ValueError("Model paths must be non-empty strings or paths")
        return value

    @field_validator("class_names")
    @classmethod
    def validate_class_names(
        cls, value: tuple[str | None, ...]
    ) -> tuple[str | None, ...]:
        visible = [name for name in value if name is not None]
        if not visible:
            raise ValueError("class_names must contain at least one foreground label")
        if any(not name or len(name) > 1_024 for name in visible):
            raise ValueError("Class names must contain 1 to 1024 characters")
        if len(visible) != len(set(visible)):
            raise ValueError("Foreground class names must be unique")
        return value

    @field_validator("background_class_id")
    @classmethod
    def reject_boolean_background(cls, value: int | None) -> int | None:
        if isinstance(value, bool):
            raise ValueError("background_class_id must be an integer or null")
        if value is not None and not -_MAX_CLASSES <= value < _MAX_CLASSES:
            raise ValueError(
                "background_class_id is outside the configured class bound"
            )
        return value

    @field_validator("supported_opsets")
    @classmethod
    def validate_supported_opsets(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or len(value) > 8 or len(value) != len(set(value)):
            raise ValueError("supported_opsets must contain 1 to 8 unique versions")
        if any(
            isinstance(version, bool) or version < 13 or version > 30
            for version in value
        ):
            raise ValueError("supported_opsets must be integer versions from 13 to 30")
        return value

    @field_validator("providers")
    @classmethod
    def validate_providers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 16 or any(not item for item in value):
            raise ValueError("providers must contain between 1 and 16 names")
        return value

    @field_validator("external_data_sha256")
    @classmethod
    def validate_external_data_sha256(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 1_024:
            raise ValueError("external_data_sha256 may contain at most 1024 files")
        for location, digest in value.items():
            if not location or len(location.encode("utf-8")) > 4_096:
                raise ValueError("external_data_sha256 contains an invalid path")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdefABCDEF" for character in digest
                )
            ):
                raise ValueError("external_data_sha256 values must be SHA-256")
        return _FrozenStringMap(value)

    @model_validator(mode="after")
    def validate_output_names(self) -> Self:
        selected = (self.input_name, self.boxes_output, self.logits_output)
        if len(set(selected)) != len(selected):
            raise ValueError("RF-DETR input and output names must be distinct")
        if self.task == "instance_segmentation" and self.masks_output in selected:
            raise ValueError("RF-DETR masks_output must be distinct")
        return self

    @property
    def resolved_model_path(self) -> Path:
        return resolve_model_path(self.model_path, config_file=self.config_file)

    @property
    def revision(self) -> str:
        return local_onnx_bundle_revision(
            self.resolved_model_path,
            explicit_revision=self.model_revision,
            sha256=self.sha256,
            external_data_sha256=self.external_data_sha256,
        )


def _static_tensor_shape(value: Any, *, role: str) -> tuple[int, ...]:
    if not value.type.HasField("tensor_type"):
        raise ValueError(f"RF-DETR {role} must be a tensor")
    if value.type.tensor_type.elem_type != _FLOAT_TENSOR_TYPE:
        raise ValueError(f"RF-DETR {role} must use float32 tensors")
    dimensions: list[int] = []
    for dimension in value.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError(f"RF-DETR {role} must have a positive static shape")
        dimensions.append(dimension.dim_value)
    return tuple(dimensions)


def _validate_graph_contract(model: Any, config: RfDetrOnnxConfig) -> dict[str, int]:
    """Validate resource bounds and the official exported tensor profile."""
    if any(item.domain not in {"", "ai.onnx"} for item in model.graph.node):
        raise ValueError("RF-DETR graphs must not contain custom operator domains")
    default_opsets = [
        item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}
    ]
    custom_opsets = [
        item.domain for item in model.opset_import if item.domain not in {"", "ai.onnx"}
    ]
    if custom_opsets or len(default_opsets) != 1:
        raise ValueError("RF-DETR graphs must declare exactly one standard ONNX opset")
    if default_opsets[0] not in config.supported_opsets:
        raise ValueError(
            f"RF-DETR graph opset {default_opsets[0]} is not in supported_opsets "
            f"{config.supported_opsets}"
        )

    inputs = {item.name: item for item in model.graph.input}
    if set(inputs) != {config.input_name}:
        raise ValueError(
            f"RF-DETR graph inputs must be exactly {[config.input_name]!r}; "
            f"received {sorted(inputs)!r}"
        )
    input_shape = _static_tensor_shape(inputs[config.input_name], role="input")
    if len(input_shape) != 4 or input_shape[:2] != (1, 3):
        raise ValueError(
            f"RF-DETR input must have shape [1,3,H,W]; received {input_shape}"
        )
    input_height, input_width = input_shape[2:]
    if not 32 <= input_height <= 16_384 or not 32 <= input_width <= 16_384:
        raise ValueError("RF-DETR input dimensions must be between 32 and 16384")
    if input_height * input_width > config.max_model_input_pixels:
        raise ValueError("RF-DETR graph input exceeds max_model_input_pixels")

    expected_outputs = {config.boxes_output, config.logits_output}
    if config.task == "instance_segmentation":
        expected_outputs.add(config.masks_output)
    outputs = {item.name: item for item in model.graph.output}
    if set(outputs) != expected_outputs:
        raise ValueError(
            f"RF-DETR graph outputs must be exactly {sorted(expected_outputs)!r}; "
            f"received {sorted(outputs)!r}"
        )
    boxes_shape = _static_tensor_shape(
        outputs[config.boxes_output], role="boxes output"
    )
    logits_shape = _static_tensor_shape(
        outputs[config.logits_output], role="logits output"
    )
    if len(boxes_shape) != 3 or boxes_shape[0] != 1 or boxes_shape[2] != 4:
        raise ValueError(
            f"RF-DETR boxes must have shape [1,Q,4]; received {boxes_shape}"
        )
    queries = boxes_shape[1]
    if queries > config.max_queries:
        raise ValueError(f"RF-DETR query count {queries} exceeds max_queries")
    if len(logits_shape) != 3 or logits_shape[:2] != (1, queries):
        raise ValueError(
            f"RF-DETR logits must have shape [1,{queries},C]; received {logits_shape}"
        )
    classes = logits_shape[2]
    if classes > config.max_classes:
        raise ValueError(f"RF-DETR class count {classes} exceeds max_classes")

    mask_height = mask_width = 0
    if config.task == "instance_segmentation":
        mask_shape = _static_tensor_shape(
            outputs[config.masks_output], role="masks output"
        )
        if len(mask_shape) != 4 or mask_shape[:2] != (1, queries):
            raise ValueError(
                f"RF-DETR masks must have shape [1,{queries},H,W]; received {mask_shape}"
            )
        mask_height, mask_width = mask_shape[2:]
        if mask_height > input_height or mask_width > input_width:
            raise ValueError(
                "RF-DETR mask outputs cannot exceed the model input dimensions"
            )

    output_elements = queries * 4 + queries * classes
    if config.task == "instance_segmentation":
        output_elements += queries * mask_height * mask_width
    if output_elements > config.max_output_elements:
        raise ValueError(
            f"RF-DETR graph declares {output_elements} output elements; configured "
            f"limit is {config.max_output_elements}"
        )
    return {
        "input_height": input_height,
        "input_width": input_width,
        "queries": queries,
        "classes": classes,
        "mask_height": mask_height,
        "mask_width": mask_width,
        "output_elements": output_elements,
        "opset": default_opsets[0],
    }


def _labels_by_exported_slot(
    config: RfDetrOnnxConfig, class_count: int
) -> tuple[str | None, ...]:
    labels = list(config.class_names)
    background = config.background_class_id
    if len(labels) == class_count:
        if background is not None:
            index = background % class_count
            if labels[index] is not None:
                raise ValueError(
                    "class_names with one entry per exported slot must use null at "
                    "background_class_id"
                )
        return tuple(labels)
    if background is None or len(labels) != class_count - 1:
        raise ValueError(
            f"RF-DETR class_names has {len(labels)} entries for {class_count} exported "
            "class slots; provide one name per slot (null for gaps), or one fewer "
            "name with background_class_id"
        )
    index = background % class_count
    labels.insert(index, None)
    return tuple(labels)


def _prepare_image(
    image: Any,
    *,
    input_height: int,
    input_width: int,
    max_image_pixels: int,
) -> tuple[np.ndarray, int, int]:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("RF-DETR expects an H x W x 3 uint8 RGB image")
    original_height, original_width = int(array.shape[0]), int(array.shape[1])
    pixels = original_height * original_width
    if pixels <= 0 or pixels > max_image_pixels:
        raise ValueError(
            f"RF-DETR image has {pixels} pixels; configured limit is {max_image_pixels}"
        )
    # OpenCV's float32 INTER_LINEAR path uses the same half-pixel, non-antialiased
    # resize convention as the official torch exporter preprocessing. Resizing
    # uint8 would quantize interpolated pixels and materially shift scores.
    resized = cv2.resize(
        array.astype(np.float32) / np.float32(255.0),
        (input_width, input_height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized -= _IMAGENET_MEAN
    resized /= _IMAGENET_STD
    tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None])
    return tensor, original_height, original_width


def _top_indices(flat_scores: np.ndarray, maximum: int) -> np.ndarray:
    """Return deterministic top indices without sorting the whole score grid."""
    if maximum <= 0 or flat_scores.size == 0:
        return np.empty(0, dtype=np.int64)
    maximum = min(maximum, flat_scores.size)
    if maximum == flat_scores.size:
        selected = np.arange(flat_scores.size, dtype=np.int64)
    else:
        cutoff_index = flat_scores.size - maximum
        cutoff = np.partition(flat_scores, cutoff_index)[cutoff_index]
        above = np.flatnonzero(flat_scores > cutoff)
        equal = np.flatnonzero(flat_scores == cutoff)
        selected = np.concatenate((above, equal[: maximum - above.size]))
    order = np.lexsort((selected, -flat_scores[selected]))
    return selected[order]


def _request_options(
    request: InferenceRequest,
    config: RfDetrOnnxConfig,
    labels_by_slot: tuple[str | None, ...],
) -> tuple[float, float, int, frozenset[int] | None]:
    unknown = set(request.parameters) - {
        "confidence",
        "mask_threshold",
        "max_detections",
        "class_ids",
    }
    if unknown:
        raise ValueError(f"Unsupported RF-DETR request parameters: {sorted(unknown)}")

    confidence = request.parameters.get("confidence", config.confidence)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("RF-DETR confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("RF-DETR confidence must be finite and between 0 and 1")

    mask_threshold = request.parameters.get("mask_threshold", config.mask_threshold)
    if isinstance(mask_threshold, bool) or not isinstance(mask_threshold, (int, float)):
        raise ValueError("RF-DETR mask_threshold must be numeric")
    mask_threshold = float(mask_threshold)
    if not math.isfinite(mask_threshold) or not 0 <= mask_threshold <= 1:
        raise ValueError("RF-DETR mask_threshold must be finite and between 0 and 1")

    maximum = request.parameters.get("max_detections", config.max_detections)
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise ValueError("RF-DETR max_detections must be an integer")
    if not 1 <= maximum <= config.max_detections:
        raise ValueError(
            f"RF-DETR max_detections must be between 1 and {config.max_detections}"
        )

    raw_class_ids = request.parameters.get("class_ids")
    class_ids: frozenset[int] | None = None
    if raw_class_ids is not None:
        if not isinstance(raw_class_ids, tuple) or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in raw_class_ids
        ):
            raise ValueError("RF-DETR class_ids must be a list of integers")
        if not raw_class_ids or len(raw_class_ids) != len(set(raw_class_ids)):
            raise ValueError("RF-DETR class_ids must be non-empty and unique")
        class_ids = frozenset(raw_class_ids)
        if any(
            item < 0 or item >= len(labels_by_slot) or labels_by_slot[item] is None
            for item in class_ids
        ):
            raise ValueError("RF-DETR class_ids contains an unknown foreground slot")
    return confidence, mask_threshold, maximum, class_ids


def _mask_logit_threshold(probability: float) -> float:
    if probability <= 0:
        return -math.inf
    if probability >= 1:
        return math.inf
    return math.log(probability / (1.0 - probability))


def _mask_shapes(
    mask_logits: np.ndarray,
    *,
    original_height: int,
    original_width: int,
    threshold: float,
    label: str,
    score: float,
    class_id: int,
    group_id: int,
    config: RfDetrOnnxConfig,
) -> tuple[list[InferenceShape], int, int]:
    resized = cv2.resize(
        mask_logits,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    binary = np.asarray(resized > _mask_logit_threshold(threshold), dtype=np.uint8)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return [], 0, 0
    parents = hierarchy[0, :, 3]
    holes = int(np.count_nonzero(parents >= 0))
    exteriors = [
        contour
        for contour, parent in zip(contours, parents, strict=True)
        if parent < 0 and cv2.contourArea(contour) > 0
    ]
    exteriors.sort(key=cv2.contourArea, reverse=True)
    discarded_components = max(0, len(exteriors) - config.max_mask_components)
    shapes: list[InferenceShape] = []
    for contour in exteriors[: config.max_mask_components]:
        epsilon = max(0.5, 0.001 * cv2.arcLength(contour, True))
        points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        for _ in range(12):
            if len(points) <= config.max_polygon_points:
                break
            epsilon *= 2
            points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(points) > config.max_polygon_points:
            raise ValueError(
                f"RF-DETR mask polygon exceeds the configured "
                f"{config.max_polygon_points}-point limit"
            )
        if len(points) < 3:
            continue
        shapes.append(
            InferenceShape(
                type=ShapeType.POLYGON,
                points=tuple(Point(x=float(x), y=float(y)) for x, y in points),
                label=label,
                score=score,
                group_id=group_id,
                attributes={"class_id": class_id},
            )
        )
    return shapes, holes, discarded_components


class RfDetrOnnxSession(BaseInferenceSession):
    """One bounded RF-DETR ONNX Runtime session."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = RfDetrOnnxConfig.model_validate(config)
        self._session: Any | None = None
        self._profile: dict[str, int] = {}
        self._labels_by_slot: tuple[str | None, ...] = ()
        self._provider_warnings: tuple[str, ...] = ()
        super().__init__(RfDetrOnnxBackend().capabilities(config))

    def _load(self, cancellation: CancellationToken) -> None:
        profile: dict[str, int] = {}

        def validate_graph(model: Any) -> None:
            profile.update(_validate_graph_contract(model, self.config))

        session, _graph, warnings = create_checked_onnx_session(
            self.config.resolved_model_path,
            providers=self.config.providers,
            allow_cpu_fallback=self.config.allow_cpu_fallback,
            max_model_bytes=self.config.max_model_bytes,
            expected_sha256=self.config.sha256,
            external_data_sha256=self.config.external_data_sha256,
            max_external_data_bytes=self.config.max_external_data_bytes,
            enable_cpu_mem_arena=self.config.enable_cpu_mem_arena,
            enable_mem_pattern=self.config.enable_mem_pattern,
            intra_op_threads=self.config.intra_op_threads,
            inter_op_threads=self.config.inter_op_threads,
            cancellation=cancellation,
            graph_validator=validate_graph,
        )
        inputs = {item.name: item for item in session.get_inputs()}
        outputs = {item.name: item for item in session.get_outputs()}
        expected_outputs = {self.config.boxes_output, self.config.logits_output}
        if self.config.task == "instance_segmentation":
            expected_outputs.add(self.config.masks_output)
        if set(inputs) != {self.config.input_name} or set(outputs) != expected_outputs:
            raise ValueError(
                "RF-DETR runtime tensor names changed after graph validation"
            )
        runtime_input_shape = tuple(inputs[self.config.input_name].shape)
        if runtime_input_shape != (
            1,
            3,
            profile["input_height"],
            profile["input_width"],
        ):
            raise ValueError(
                "RF-DETR runtime input shape changed after graph validation"
            )
        labels = _labels_by_exported_slot(self.config, profile["classes"])
        cancellation.raise_if_cancelled()
        self._session = session
        self._profile = profile
        self._labels_by_slot = labels
        self._provider_warnings = warnings
        self._capabilities = self._capabilities.model_copy(
            update={
                "metadata": {
                    **self._capabilities.metadata,
                    "providers": ",".join(session.get_providers()),
                    "input_size": f"{profile['input_height']}x{profile['input_width']}",
                    "query_count": profile["queries"],
                    "exported_class_slots": profile["classes"],
                    "opset": profile["opset"],
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
            raise RuntimeError("RF-DETR ONNX session is not loaded")
        if request.prompts:
            raise ValueError("RF-DETR does not accept inference prompts")
        if self.config.task == "detection":
            if request.output_shape not in {None, ShapeType.RECTANGLE}:
                raise ValueError("RF-DETR detection output_shape must be rectangle")
            output_shape = ShapeType.RECTANGLE
        else:
            output_shape = request.output_shape or ShapeType.POLYGON
            if output_shape not in {ShapeType.POLYGON, ShapeType.RECTANGLE}:
                raise ValueError(
                    "RF-DETR instance segmentation output_shape must be polygon or rectangle"
                )
        confidence, mask_threshold, maximum, class_filter = _request_options(
            request,
            self.config,
            self._labels_by_slot,
        )
        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        tensor, original_height, original_width = _prepare_image(
            image,
            input_height=self._profile["input_height"],
            input_width=self._profile["input_width"],
            max_image_pixels=self.config.max_image_pixels,
        )
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        cancellation.raise_if_cancelled()
        output_names = [self.config.boxes_output, self.config.logits_output]
        if self.config.task == "instance_segmentation":
            output_names.append(self.config.masks_output)
        inference_started = time.perf_counter()
        raw_outputs = self._session.run(output_names, {self.config.input_name: tensor})
        inference_ms = (time.perf_counter() - inference_started) * 1000
        cancellation.raise_if_cancelled()
        arrays = [np.asarray(value) for value in raw_outputs]
        if sum(value.size for value in arrays) > self.config.max_output_elements:
            raise ValueError("RF-DETR runtime outputs exceed max_output_elements")
        expected_shapes = [
            (1, self._profile["queries"], 4),
            (1, self._profile["queries"], self._profile["classes"]),
        ]
        if self.config.task == "instance_segmentation":
            expected_shapes.append(
                (
                    1,
                    self._profile["queries"],
                    self._profile["mask_height"],
                    self._profile["mask_width"],
                )
            )
        for value, expected in zip(arrays, expected_shapes, strict=True):
            if value.shape != expected or value.dtype != np.dtype(np.float32):
                raise ValueError(
                    f"RF-DETR runtime output must have shape {expected} and float32 "
                    f"dtype; received {value.shape} {value.dtype}"
                )
            if not np.isfinite(value).all():
                raise ValueError("RF-DETR runtime outputs contain NaN or infinity")

        postprocess_started = time.perf_counter()
        boxes = np.asarray(arrays[0][0], dtype=np.float32)
        logits = np.asarray(arrays[1][0], dtype=np.float32)
        clipped = np.clip(logits, -88, 88)
        scores_all = np.reciprocal(np.float32(1.0) + np.exp(-clipped))
        eligible_slots = np.asarray(
            [
                index
                for index, label in enumerate(self._labels_by_slot)
                if label is not None and (class_filter is None or index in class_filter)
            ],
            dtype=np.int64,
        )
        eligible_scores = scores_all[:, eligible_slots]
        selected = _top_indices(eligible_scores.reshape(-1), self._profile["queries"])
        selected_scores = eligible_scores.reshape(-1)[selected]
        keep = selected_scores > confidence
        selected = selected[keep][:maximum]
        selected_scores = selected_scores[keep][:maximum]
        selected_queries = selected // eligible_slots.size
        selected_slots = eligible_slots[selected % eligible_slots.size]

        shapes: list[InferenceShape] = []
        warnings = list(self._provider_warnings)
        empty_masks = 0
        discarded_holes = 0
        discarded_components = 0
        total_points = 0
        masks = arrays[2][0] if self.config.task == "instance_segmentation" else None
        for rank, (query, class_id, score) in enumerate(
            zip(selected_queries, selected_slots, selected_scores, strict=True)
        ):
            class_id_int = int(class_id)
            label = self._labels_by_slot[class_id_int]
            if label is None:
                raise RuntimeError("RF-DETR selected a non-foreground class slot")
            cx, cy, width, height = (float(item) for item in boxes[int(query)])
            x1 = min(max((cx - width / 2) * original_width, 0.0), float(original_width))
            y1 = min(
                max((cy - height / 2) * original_height, 0.0), float(original_height)
            )
            x2 = min(max((cx + width / 2) * original_width, 0.0), float(original_width))
            y2 = min(
                max((cy + height / 2) * original_height, 0.0), float(original_height)
            )
            if x2 <= x1 or y2 <= y1:
                continue
            score_float = float(score)
            if output_shape is ShapeType.RECTANGLE:
                if len(shapes) >= self.config.max_shapes:
                    raise ValueError("RF-DETR results exceed max_shapes")
                shapes.append(
                    InferenceShape(
                        type=ShapeType.RECTANGLE,
                        points=(Point(x=x1, y=y1), Point(x=x2, y=y2)),
                        label=label,
                        score=score_float,
                        group_id=rank
                        if self.config.task == "instance_segmentation"
                        else None,
                        attributes={"class_id": class_id_int},
                    )
                )
                continue
            if masks is None:
                raise RuntimeError("RF-DETR segmentation masks are unavailable")
            mask_shapes, holes, components = _mask_shapes(
                np.asarray(masks[int(query)], dtype=np.float32),
                original_height=original_height,
                original_width=original_width,
                threshold=mask_threshold,
                label=label,
                score=score_float,
                class_id=class_id_int,
                group_id=rank,
                config=self.config,
            )
            if not mask_shapes:
                empty_masks += 1
                continue
            discarded_holes += holes
            discarded_components += components
            for shape in mask_shapes:
                total_points += len(shape.points)
                if total_points > self.config.max_total_shape_points:
                    raise ValueError("RF-DETR results exceed max_total_shape_points")
                if len(shapes) >= self.config.max_shapes:
                    raise ValueError("RF-DETR results exceed max_shapes")
                shapes.append(shape)
        if empty_masks:
            warnings.append(f"Discarded {empty_masks} detections with empty masks")
        if discarded_holes:
            warnings.append(
                f"Discarded {discarded_holes} mask holes that editable polygons cannot represent"
            )
        if discarded_components:
            warnings.append(
                f"Discarded {discarded_components} mask components above the per-instance limit"
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
        self._profile = {}
        self._labels_by_slot = ()
        self._provider_warnings = ()
        if self.config.release_cpu_memory_on_unload:
            release_unused_cpu_memory()


class RfDetrOnnxBackend(InferenceBackend):
    backend_id = "rfdetr_onnx"

    def capabilities(self, config: Mapping[str, Any]) -> ModelCapabilities:
        parsed = RfDetrOnnxConfig.model_validate(config)
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
                "task": parsed.task,
                "artifact_policy": "verified-onnx",
                "requested_providers": ",".join(parsed.providers),
            },
        )

    def create_session(self, config: Mapping[str, Any]) -> RfDetrOnnxSession:
        return RfDetrOnnxSession(config)


__all__ = ["RfDetrOnnxBackend", "RfDetrOnnxConfig", "RfDetrOnnxSession"]

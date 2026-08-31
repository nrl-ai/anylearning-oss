"""Bounded ONNX inference for D-FINE object detection.

The graph contract and processing flow are derived from the official D-FINE
exporter at source revision 956d1709314c2c6a4df6f34de232054578a7449f.
Inference accepts only an integrity-checked ONNX graph and never imports the
training framework or deserializes native checkpoints.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
from PIL import Image
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
_MAX_OUTPUT_ELEMENTS = 1_000_000
_MAX_QUERIES = 10_000
_MAX_CLASSES = 10_000
_FLOAT_TENSOR_TYPE = 1
_INT64_TENSOR_TYPE = 7
_OUTPUT_COORDINATE_DECIMALS = 3
_OUTPUT_SCORE_DECIMALS = 4


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


class DFineOnnxConfig(BaseModel):
    """Strict configuration for a static official D-FINE ONNX export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=512)
    model_path: Path
    config_file: Path | None = None
    model_revision: str | None = Field(default=None, min_length=1, max_length=512)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    external_data_sha256: dict[str, str] = Field(default_factory=dict)
    class_names: tuple[str, ...] = Field(min_length=1, max_length=_MAX_CLASSES)
    images_input: str = Field(default="images", min_length=1, max_length=512)
    original_sizes_input: str = Field(
        default="orig_target_sizes", min_length=1, max_length=512
    )
    labels_output: str = Field(default="labels", min_length=1, max_length=512)
    boxes_output: str = Field(default="boxes", min_length=1, max_length=512)
    scores_output: str = Field(default="scores", min_length=1, max_length=512)
    supported_opsets: tuple[int, ...] = (16,)
    confidence: float = Field(default=0.4, ge=0, le=1, allow_inf_nan=False)
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
    max_shapes: int = Field(default=1_000, ge=1, le=10_000)
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
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
    def validate_class_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name or len(name) > 1_024 for name in value):
            raise ValueError("Class names must contain 1 to 1024 characters")
        if len(value) != len(set(value)):
            raise ValueError("Class names must be unique")
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
    def require_verified_cpu_provider(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != ("CPUExecutionProvider",):
            raise ValueError(
                "D-FINE currently supports only the verified CPUExecutionProvider"
            )
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
    def validate_tensor_names_and_bounds(self) -> Self:
        names = (
            self.images_input,
            self.original_sizes_input,
            self.labels_output,
            self.boxes_output,
            self.scores_output,
        )
        if len(names) != len(set(names)):
            raise ValueError("D-FINE input and output names must be distinct")
        if len(self.class_names) > self.max_classes:
            raise ValueError("D-FINE class_names exceeds max_classes")
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


def _static_tensor_shape(
    value: Any, *, role: str, element_type: int
) -> tuple[int, ...]:
    if not value.type.HasField("tensor_type"):
        raise ValueError(f"D-FINE {role} must be a tensor")
    if value.type.tensor_type.elem_type != element_type:
        dtype = "float32" if element_type == _FLOAT_TENSOR_TYPE else "int64"
        raise ValueError(f"D-FINE {role} must use {dtype} tensors")
    dimensions: list[int] = []
    for dimension in value.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError(f"D-FINE {role} must have a positive static shape")
        dimensions.append(dimension.dim_value)
    return tuple(dimensions)


def _validate_graph_contract(model: Any, config: DFineOnnxConfig) -> dict[str, int]:
    """Validate the official bounded D-FINE deployment profile before ORT."""
    if any(item.domain not in {"", "ai.onnx"} for item in model.graph.node):
        raise ValueError("D-FINE graphs must not contain custom operator domains")
    default_opsets = [
        item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}
    ]
    custom_opsets = [
        item.domain for item in model.opset_import if item.domain not in {"", "ai.onnx"}
    ]
    if custom_opsets or len(default_opsets) != 1:
        raise ValueError("D-FINE graphs must declare exactly one standard ONNX opset")
    if default_opsets[0] not in config.supported_opsets:
        raise ValueError(
            f"D-FINE graph opset {default_opsets[0]} is not in supported_opsets "
            f"{config.supported_opsets}"
        )

    inputs = {item.name: item for item in model.graph.input}
    expected_inputs = {config.images_input, config.original_sizes_input}
    if set(inputs) != expected_inputs:
        raise ValueError(
            f"D-FINE graph inputs must be exactly {sorted(expected_inputs)!r}; "
            f"received {sorted(inputs)!r}"
        )
    image_shape = _static_tensor_shape(
        inputs[config.images_input],
        role="images input",
        element_type=_FLOAT_TENSOR_TYPE,
    )
    if len(image_shape) != 4 or image_shape[:2] != (1, 3):
        raise ValueError(
            f"D-FINE images input must have shape [1,3,H,W]; received {image_shape}"
        )
    input_height, input_width = image_shape[2:]
    if not 32 <= input_height <= 16_384 or not 32 <= input_width <= 16_384:
        raise ValueError("D-FINE input dimensions must be between 32 and 16384")
    if input_height * input_width > config.max_model_input_pixels:
        raise ValueError("D-FINE graph input exceeds max_model_input_pixels")
    size_shape = _static_tensor_shape(
        inputs[config.original_sizes_input],
        role="original sizes input",
        element_type=_INT64_TENSOR_TYPE,
    )
    if size_shape != (1, 2):
        raise ValueError(
            "D-FINE original sizes input must have static shape [1,2]; "
            f"received {size_shape}"
        )

    expected_outputs = {
        config.labels_output,
        config.boxes_output,
        config.scores_output,
    }
    outputs = {item.name: item for item in model.graph.output}
    if set(outputs) != expected_outputs:
        raise ValueError(
            f"D-FINE graph outputs must be exactly {sorted(expected_outputs)!r}; "
            f"received {sorted(outputs)!r}"
        )
    labels_shape = _static_tensor_shape(
        outputs[config.labels_output],
        role="labels output",
        element_type=_INT64_TENSOR_TYPE,
    )
    scores_shape = _static_tensor_shape(
        outputs[config.scores_output],
        role="scores output",
        element_type=_FLOAT_TENSOR_TYPE,
    )
    boxes_shape = _static_tensor_shape(
        outputs[config.boxes_output],
        role="boxes output",
        element_type=_FLOAT_TENSOR_TYPE,
    )
    if len(labels_shape) != 2 or labels_shape[0] != 1:
        raise ValueError(
            f"D-FINE labels must have shape [1,Q]; received {labels_shape}"
        )
    queries = labels_shape[1]
    if queries > config.max_queries:
        raise ValueError(f"D-FINE query count {queries} exceeds max_queries")
    if scores_shape != (1, queries):
        raise ValueError(
            f"D-FINE scores must have shape [1,{queries}]; received {scores_shape}"
        )
    if boxes_shape != (1, queries, 4):
        raise ValueError(
            f"D-FINE boxes must have shape [1,{queries},4]; received {boxes_shape}"
        )
    output_elements = queries * 6
    if output_elements > config.max_output_elements:
        raise ValueError(
            f"D-FINE graph declares {output_elements} output elements; configured "
            f"limit is {config.max_output_elements}"
        )
    return {
        "input_height": input_height,
        "input_width": input_width,
        "queries": queries,
        "output_elements": output_elements,
        "opset": default_opsets[0],
    }


def _prepare_image(
    image: Any,
    *,
    input_height: int,
    input_width: int,
    max_image_pixels: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("D-FINE expects an H x W x 3 uint8 RGB image")
    original_height, original_width = int(array.shape[0]), int(array.shape[1])
    pixels = original_height * original_width
    if pixels <= 0 or pixels > max_image_pixels:
        raise ValueError(
            f"D-FINE image has {pixels} pixels; configured limit is {max_image_pixels}"
        )
    # The training and native evaluation pipeline stretches directly to the
    # configured shape.  Letterboxing is intentionally not used here because it
    # changes the input distribution and has documented accuracy regressions.
    resized = Image.fromarray(array).resize(
        (input_width, input_height), Image.Resampling.BILINEAR
    )
    normalized = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
    original_sizes = np.asarray([[original_width, original_height]], dtype=np.int64)
    return tensor, original_sizes, original_height, original_width


def _request_options(
    request: InferenceRequest,
    config: DFineOnnxConfig,
) -> tuple[float, int, frozenset[int] | None]:
    unknown = set(request.parameters) - {
        "confidence",
        "max_detections",
        "class_ids",
    }
    if unknown:
        raise ValueError(f"Unsupported D-FINE request parameters: {sorted(unknown)}")

    confidence = request.parameters.get("confidence", config.confidence)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("D-FINE confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("D-FINE confidence must be finite and between 0 and 1")

    maximum = request.parameters.get("max_detections", config.max_detections)
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise ValueError("D-FINE max_detections must be an integer")
    if not 1 <= maximum <= config.max_detections:
        raise ValueError(
            f"D-FINE max_detections must be between 1 and {config.max_detections}"
        )

    raw_class_ids = request.parameters.get("class_ids")
    class_ids: frozenset[int] | None = None
    if raw_class_ids is not None:
        if not isinstance(raw_class_ids, tuple) or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in raw_class_ids
        ):
            raise ValueError("D-FINE class_ids must be a list of integers")
        if not raw_class_ids or len(raw_class_ids) != len(set(raw_class_ids)):
            raise ValueError("D-FINE class_ids must be non-empty and unique")
        if any(item < 0 or item >= len(config.class_names) for item in raw_class_ids):
            raise ValueError("D-FINE class_ids contains an unknown class")
        class_ids = frozenset(raw_class_ids)
    return confidence, maximum, class_ids


class DFineOnnxSession(BaseInferenceSession):
    """One bounded D-FINE ONNX Runtime session."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = DFineOnnxConfig.model_validate(config)
        self._session: Any | None = None
        self._profile: dict[str, int] = {}
        self._provider_warnings: tuple[str, ...] = ()
        super().__init__(DFineOnnxBackend().capabilities(config))

    def _load(self, cancellation: CancellationToken) -> None:
        profile: dict[str, int] = {}

        def validate_graph(model: Any) -> None:
            profile.update(_validate_graph_contract(model, self.config))

        session, _graph, warnings = create_checked_onnx_session(
            self.config.resolved_model_path,
            providers=self.config.providers,
            allow_cpu_fallback=False,
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
        if set(inputs) != {self.config.images_input, self.config.original_sizes_input}:
            raise ValueError(
                "D-FINE runtime input names changed after graph validation"
            )
        if set(outputs) != {
            self.config.labels_output,
            self.config.boxes_output,
            self.config.scores_output,
        }:
            raise ValueError(
                "D-FINE runtime output names changed after graph validation"
            )
        if tuple(inputs[self.config.images_input].shape) != (
            1,
            3,
            profile["input_height"],
            profile["input_width"],
        ) or tuple(inputs[self.config.original_sizes_input].shape) != (1, 2):
            raise ValueError(
                "D-FINE runtime input shape changed after graph validation"
            )
        cancellation.raise_if_cancelled()
        self._session = session
        self._profile = profile
        self._provider_warnings = warnings
        self._capabilities = self._capabilities.model_copy(
            update={
                "metadata": {
                    **self._capabilities.metadata,
                    "providers": ",".join(session.get_providers()),
                    "input_size": f"{profile['input_height']}x{profile['input_width']}",
                    "query_count": profile["queries"],
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
            raise RuntimeError("D-FINE ONNX session is not loaded")
        if request.prompts:
            raise ValueError("D-FINE does not accept inference prompts")
        if request.output_shape not in {None, ShapeType.RECTANGLE}:
            raise ValueError("D-FINE output_shape must be rectangle")
        confidence, maximum, class_filter = _request_options(request, self.config)

        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        tensor, original_sizes, original_height, original_width = _prepare_image(
            image,
            input_height=self._profile["input_height"],
            input_width=self._profile["input_width"],
            max_image_pixels=self.config.max_image_pixels,
        )
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        cancellation.raise_if_cancelled()

        output_names = [
            self.config.labels_output,
            self.config.boxes_output,
            self.config.scores_output,
        ]
        inference_started = time.perf_counter()
        raw_outputs = self._session.run(
            output_names,
            {
                self.config.images_input: tensor,
                self.config.original_sizes_input: original_sizes,
            },
        )
        inference_ms = (time.perf_counter() - inference_started) * 1000
        cancellation.raise_if_cancelled()

        labels, boxes, scores = (np.asarray(value) for value in raw_outputs)
        queries = self._profile["queries"]
        expected = (
            (labels, (1, queries), np.dtype(np.int64), "labels"),
            (boxes, (1, queries, 4), np.dtype(np.float32), "boxes"),
            (scores, (1, queries), np.dtype(np.float32), "scores"),
        )
        if (
            sum(value.size for value, *_rest in expected)
            > self.config.max_output_elements
        ):
            raise ValueError("D-FINE runtime outputs exceed max_output_elements")
        for value, shape, dtype, role in expected:
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(
                    f"D-FINE runtime {role} must have shape {shape} and {dtype} "
                    f"dtype; received {value.shape} {value.dtype}"
                )
            if role != "labels" and not np.isfinite(value).all():
                raise ValueError(f"D-FINE runtime {role} contain NaN or infinity")

        postprocess_started = time.perf_counter()
        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]
        if np.any(labels < 0) or np.any(labels >= len(self.config.class_names)):
            raise ValueError("D-FINE runtime labels contain an unknown class index")
        order = np.lexsort((np.arange(queries, dtype=np.int64), -scores))
        shapes: list[InferenceShape] = []
        for index in order:
            score = float(scores[index])
            if score <= confidence:
                continue
            class_id = int(labels[index])
            if class_filter is not None and class_id not in class_filter:
                continue
            x1, y1, x2, y2 = (float(item) for item in boxes[index])
            x1 = min(max(x1, 0.0), float(original_width))
            y1 = min(max(y1, 0.0), float(original_height))
            x2 = min(max(x2, 0.0), float(original_width))
            y2 = min(max(y2, 0.0), float(original_height))
            if x2 <= x1 or y2 <= y1:
                continue
            if len(shapes) >= self.config.max_shapes:
                raise ValueError("D-FINE results exceed max_shapes")
            shapes.append(
                InferenceShape(
                    type=ShapeType.RECTANGLE,
                    points=(
                        Point(
                            x=round(x1, _OUTPUT_COORDINATE_DECIMALS),
                            y=round(y1, _OUTPUT_COORDINATE_DECIMALS),
                        ),
                        Point(
                            x=round(x2, _OUTPUT_COORDINATE_DECIMALS),
                            y=round(y2, _OUTPUT_COORDINATE_DECIMALS),
                        ),
                    ),
                    label=self.config.class_names[class_id],
                    score=round(score, _OUTPUT_SCORE_DECIMALS),
                    attributes={"class_id": class_id},
                )
            )
            if len(shapes) >= maximum:
                break
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        return InferenceResult(
            request_id=request.request_id,
            source_id=request.source_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
            shapes=tuple(shapes),
            warnings=self._provider_warnings,
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
        self._provider_warnings = ()
        if self.config.release_cpu_memory_on_unload:
            release_unused_cpu_memory()


class DFineOnnxBackend(InferenceBackend):
    backend_id = "dfine_onnx"

    def capabilities(self, config: Mapping[str, Any]) -> ModelCapabilities:
        parsed = DFineOnnxConfig.model_validate(config)
        return ModelCapabilities(
            model_id=parsed.name,
            model_revision=parsed.revision,
            tasks=(ModelTask.DETECTION,),
            supports_cancellation=True,
            metadata={
                "backend": self.backend_id,
                "artifact_policy": "verified-onnx",
                "requested_providers": ",".join(parsed.providers),
                "preprocessing": "rgb-stretch",
            },
        )

    def create_session(self, config: Mapping[str, Any]) -> DFineOnnxSession:
        return DFineOnnxSession(config)


__all__ = ["DFineOnnxBackend", "DFineOnnxConfig", "DFineOnnxSession"]

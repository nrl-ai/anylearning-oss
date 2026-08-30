"""SAM3 text and geometric prompt inference behind the shared ONNX boundary."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..cache import LRUCache
from ..contracts import (
    BoxPrompt,
    InferenceRequest,
    InferenceResult,
    InferenceShape,
    ModelCapabilities,
    ModelTask,
    Point,
    PointPrompt,
    ShapeType,
    TextPrompt,
)
from ..runtime import (
    BaseInferenceSession,
    CancellationToken,
    InferenceBackend,
    SessionLifecycleError,
    SessionState,
)
from .onnx_safety import local_onnx_bundle_revision
from .onnx_session import create_checked_onnx_session, release_unused_cpu_memory
from .sam import _FrozenStringMap, _resolved_model_path, mask_contours
from .sam3_onnx import Sam3Detections, Sam3OnnxPipeline

logger = logging.getLogger(__name__)

_MAX_MODEL_BYTES = 20 * 1024**3
_MAX_EXTERNAL_DATA_BYTES = 20 * 1024**3
_MAX_IMAGE_PIXELS = 16_000_000
_MAX_FEATURE_ELEMENTS = 100_000_000
_MAX_OUTPUT_ELEMENTS = 300_000_000
_MAX_RAW_QUERIES = 256
_MAX_NMS_CANDIDATES = 128
_MAX_INSTANCES = 64
_MAX_MASK_CONTOURS = 4096
_MAX_SHAPES = 512
_MAX_POLYGON_POINTS = 4096
_MAX_TOTAL_SHAPE_POINTS = 50_000


class Sam3Config(BaseModel):
    """Integrity-addressed configuration for a SAM3 ONNX graph triplet."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    name: str = Field(min_length=1, max_length=512)
    image_encoder_model_path: Path = Field(
        validation_alias=AliasChoices("image_encoder_model_path", "encoder_model_path")
    )
    language_encoder_model_path: Path = Field(
        validation_alias=AliasChoices(
            "language_encoder_model_path", "language_encoder_path"
        )
    )
    decoder_model_path: Path
    config_file: Path | None = None
    model_revision: str | None = Field(default=None, min_length=1, max_length=512)
    archive_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    image_encoder_sha256: str | None = Field(
        default=None,
        validation_alias=AliasChoices("image_encoder_sha256", "encoder_sha256"),
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    language_encoder_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-fA-F]{64}$"
    )
    decoder_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    image_encoder_external_data_sha256: dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "image_encoder_external_data_sha256", "encoder_external_data_sha256"
        ),
    )
    language_encoder_external_data_sha256: dict[str, str] = Field(default_factory=dict)
    decoder_external_data_sha256: dict[str, str] = Field(default_factory=dict)
    max_model_bytes: int = Field(default=_MAX_MODEL_BYTES, ge=1, le=40 * 1024**3)
    max_external_data_bytes: int = Field(
        default=_MAX_EXTERNAL_DATA_BYTES, ge=1, le=1024 * 1024**3
    )
    max_image_pixels: int = Field(default=_MAX_IMAGE_PIXELS, ge=1, le=100_000_000)
    max_feature_elements: int = Field(
        default=_MAX_FEATURE_ELEMENTS, ge=1, le=1_000_000_000
    )
    max_output_elements: int = Field(
        default=_MAX_OUTPUT_ELEMENTS, ge=1, le=2_000_000_000
    )
    max_raw_queries: int = Field(default=_MAX_RAW_QUERIES, ge=1, le=4096)
    max_nms_candidates: int = Field(default=_MAX_NMS_CANDIDATES, ge=1, le=1024)
    max_instances: int = Field(default=_MAX_INSTANCES, ge=1, le=512)
    max_mask_contours: int = Field(default=_MAX_MASK_CONTOURS, ge=1, le=100_000)
    max_shapes: int = Field(default=_MAX_SHAPES, ge=1, le=10_000)
    max_polygon_points: int = Field(default=_MAX_POLYGON_POINTS, ge=3, le=100_000)
    max_total_shape_points: int = Field(
        default=_MAX_TOTAL_SHAPE_POINTS, ge=3, le=1_000_000
    )
    max_text_bytes: int = Field(default=4096, ge=1, le=65_536)
    image_cache_items: int = Field(default=1, ge=1, le=4)
    text_cache_items: int = Field(default=32, ge=1, le=256)
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    nms_threshold: float = Field(default=0.7, ge=0, le=1)
    processed_graph_confidence_floor: float = Field(default=0.5, ge=0, le=1)
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    allow_cpu_fallback: bool = True
    enable_cpu_mem_arena: bool = False
    enable_mem_pattern: bool = False
    release_cpu_memory_on_unload: bool = True
    intra_op_threads: int = Field(default=0, ge=0, le=256)
    inter_op_threads: int = Field(default=0, ge=0, le=256)

    @field_validator(
        "image_encoder_model_path",
        "language_encoder_model_path",
        "decoder_model_path",
        "config_file",
        mode="before",
    )
    @classmethod
    def reject_empty_paths(cls, value: Any) -> Any:
        if value is not None and (not isinstance(value, (str, Path)) or not str(value)):
            raise ValueError("SAM3 model paths must be non-empty strings or paths")
        return value

    @field_validator("providers")
    @classmethod
    def validate_providers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 16:
            raise ValueError("providers must contain between 1 and 16 names")
        return value

    @field_validator(
        "image_encoder_external_data_sha256",
        "language_encoder_external_data_sha256",
        "decoder_external_data_sha256",
    )
    @classmethod
    def validate_external_manifest(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 1024:
            raise ValueError("External-data manifest may contain at most 1024 files")
        for location, digest in value.items():
            if not location or len(location.encode("utf-8")) > 4096:
                raise ValueError("External-data manifest contains an invalid path")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdefABCDEF" for character in digest
                )
            ):
                raise ValueError("External-data manifest values must be SHA-256")
        return _FrozenStringMap(value)

    @model_validator(mode="after")
    def validate_limits_and_graphs(self) -> Self:
        paths = {
            self.image_encoder_model_path,
            self.language_encoder_model_path,
            self.decoder_model_path,
        }
        if len(paths) != 3:
            raise ValueError("SAM3 requires three distinct ONNX graph files")
        if self.max_instances > self.max_nms_candidates:
            raise ValueError("max_instances cannot exceed max_nms_candidates")
        if self.max_nms_candidates > self.max_raw_queries:
            raise ValueError("max_nms_candidates cannot exceed max_raw_queries")
        return self

    def resolved_path(self, field: str) -> Path:
        return _resolved_model_path(self.model_dump(), field)

    @property
    def revision(self) -> str:
        if self.model_revision is not None:
            return self.model_revision
        identity: dict[str, Any] = {
            "name": self.name,
            "archive_sha256": self.archive_sha256,
        }
        for role in ("image_encoder", "language_encoder", "decoder"):
            identity[role] = local_onnx_bundle_revision(
                self.resolved_path(f"{role}_model_path"),
                sha256=getattr(self, f"{role}_sha256"),
                external_data_sha256=getattr(self, f"{role}_external_data_sha256"),
            )
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return f"onnx-triplet-sha256:{hashlib.sha256(payload).hexdigest()}"


def _geometric_prompts(
    request: InferenceRequest,
) -> tuple[str, list[dict[str, Any]]]:
    text_prompts = [
        prompt for prompt in request.prompts if isinstance(prompt, TextPrompt)
    ]
    if len(text_prompts) > 1:
        raise ValueError("SAM3 accepts at most one text prompt per request")
    text = text_prompts[0].text if text_prompts else "visual"
    geometric: list[dict[str, Any]] = []
    for prompt in request.prompts:
        if isinstance(prompt, PointPrompt):
            geometric.append(
                {
                    "type": "point",
                    "data": [prompt.point.x, prompt.point.y],
                    "label": 1 if prompt.foreground else 0,
                }
            )
        elif isinstance(prompt, BoxPrompt):
            geometric.append(
                {
                    "type": "rectangle",
                    "data": [
                        prompt.top_left.x,
                        prompt.top_left.y,
                        prompt.bottom_right.x,
                        prompt.bottom_right.y,
                    ],
                }
            )
    if not text_prompts and not geometric:
        raise ValueError("SAM3 requires a text, point, or box prompt")
    return text, geometric


def _numeric_parameter(
    parameters: Mapping[str, Any],
    name: str,
    default: float,
) -> float:
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"SAM3 parameter {name!r} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"SAM3 parameter {name!r} must be finite")
    return result


def _request_options(
    request: InferenceRequest,
    config: Sam3Config,
    *,
    output_profile: str,
) -> tuple[float, float, int]:
    unknown = set(request.parameters) - {"confidence", "iou", "max_instances"}
    if unknown:
        raise ValueError(f"Unsupported SAM3 request parameter: {sorted(unknown)[0]}")
    confidence = _numeric_parameter(
        request.parameters, "confidence", config.confidence_threshold
    )
    iou = _numeric_parameter(request.parameters, "iou", config.nms_threshold)
    raw_instances = request.parameters.get("max_instances", config.max_instances)
    if isinstance(raw_instances, bool) or not isinstance(raw_instances, int):
        raise ValueError("SAM3 parameter 'max_instances' must be an integer")
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        raise ValueError("SAM3 confidence and IoU parameters must be between 0 and 1")
    if not 1 <= raw_instances <= config.max_instances:
        raise ValueError(
            f"SAM3 max_instances must be between 1 and {config.max_instances}"
        )
    if (
        output_profile == "processed"
        and confidence < config.processed_graph_confidence_floor
    ):
        raise ValueError(
            "SAM3 confidence is below the processed graph's configured floor"
        )
    return confidence, iou, raw_instances


def sam3_shapes(
    detections: Sam3Detections,
    *,
    output_shape: ShapeType,
    label: str,
    config: Sam3Config,
) -> tuple[InferenceShape, ...]:
    if output_shape not in {ShapeType.POLYGON, ShapeType.RECTANGLE}:
        raise ValueError("SAM3 output_shape must be polygon or rectangle")
    shapes: list[InferenceShape] = []
    total_points = 0
    for instance, (mask, score, box) in enumerate(
        zip(detections.masks[:, 0], detections.scores, detections.boxes, strict=True)
    ):
        if output_shape is ShapeType.RECTANGLE:
            x1, y1, x2, y2 = (float(value) for value in box)
            if x2 <= x1 or y2 <= y1:
                continue
            shapes.append(
                InferenceShape(
                    type=ShapeType.RECTANGLE,
                    points=(Point(x=x1, y=y1), Point(x=x2, y=y2)),
                    label=label,
                    score=float(score),
                    group_id=instance,
                )
            )
            total_points += 2
        else:
            contours = mask_contours(
                mask,
                max_mask_contours=config.max_mask_contours,
                max_shapes=config.max_shapes,
                max_polygon_points=config.max_polygon_points,
            )
            for contour in contours:
                coordinates = contour.reshape(-1, 2).astype(int).tolist()
                coordinates.append(coordinates[0])
                total_points += len(coordinates)
                if total_points > config.max_total_shape_points:
                    raise ValueError(
                        "SAM3 polygons exceed the configured total point limit"
                    )
                shapes.append(
                    InferenceShape(
                        type=ShapeType.POLYGON,
                        points=tuple(Point(x=x, y=y) for x, y in coordinates),
                        label=label,
                        score=float(score),
                        group_id=instance,
                    )
                )
        if len(shapes) > config.max_shapes:
            raise ValueError("SAM3 results exceed the configured editable shape limit")
    return tuple(shapes)


class Sam3Session(BaseInferenceSession):
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = Sam3Config.model_validate(config)
        self._model: Sam3OnnxPipeline | None = None
        self._image_cache: LRUCache[tuple[str, str], dict[str, np.ndarray]] = LRUCache(
            self.config.image_cache_items
        )
        self._text_cache: LRUCache[str, dict[str, np.ndarray]] = LRUCache(
            self.config.text_cache_items
        )
        self._provider_warnings: tuple[str, ...] = ()
        super().__init__(Sam3Backend().capabilities(config))

    def _load_graph(
        self,
        role: str,
        cancellation: CancellationToken,
    ) -> tuple[Any, tuple[str, ...]]:
        session, _graph, warnings = create_checked_onnx_session(
            self.config.resolved_path(f"{role}_model_path"),
            providers=self.config.providers,
            allow_cpu_fallback=self.config.allow_cpu_fallback,
            max_model_bytes=self.config.max_model_bytes,
            expected_sha256=getattr(self.config, f"{role}_sha256"),
            external_data_sha256=getattr(self.config, f"{role}_external_data_sha256"),
            max_external_data_bytes=self.config.max_external_data_bytes,
            enable_cpu_mem_arena=self.config.enable_cpu_mem_arena,
            enable_mem_pattern=self.config.enable_mem_pattern,
            intra_op_threads=self.config.intra_op_threads,
            inter_op_threads=self.config.inter_op_threads,
            cancellation=cancellation,
        )
        return session, warnings

    def _load(self, cancellation: CancellationToken) -> None:
        image_session, image_warnings = self._load_graph("image_encoder", cancellation)
        language_session, language_warnings = self._load_graph(
            "language_encoder", cancellation
        )
        decoder_session, decoder_warnings = self._load_graph("decoder", cancellation)
        cancellation.raise_if_cancelled()
        self._model = Sam3OnnxPipeline(
            image_session,
            language_session,
            decoder_session,
            max_text_bytes=self.config.max_text_bytes,
            max_raw_queries=self.config.max_raw_queries,
            max_output_elements=self.config.max_output_elements,
            max_nms_candidates=self.config.max_nms_candidates,
            max_feature_elements=self.config.max_feature_elements,
        )
        self._provider_warnings = tuple(
            dict.fromkeys((*image_warnings, *language_warnings, *decoder_warnings))
        )
        for warning in self._provider_warnings:
            logger.warning("SAM3 ONNX provider selection: %s", warning)

    def _validated_image(self, image: Any) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("SAM3 expects an H x W x 3 uint8 RGB image")
        pixels = int(array.shape[0]) * int(array.shape[1])
        if pixels <= 0 or pixels > self.config.max_image_pixels:
            raise ValueError(
                f"SAM3 image has {pixels} pixels; configured limit is "
                f"{self.config.max_image_pixels}"
            )
        if self._model is not None:
            multiplier = (
                self.config.max_raw_queries
                if self._model.decoder.output_profile == "processed"
                else self.config.max_instances
            )
            if pixels * multiplier > self.config.max_output_elements:
                raise ValueError(
                    "SAM3 image and query limits exceed the configured output bound"
                )
        return array

    def _predict(
        self,
        request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken,
    ) -> InferenceResult:
        model = self._model
        if model is None:
            raise SessionLifecycleError("SAM3 runtime is not loaded")
        image_array = self._validated_image(image)
        has_text_prompt = any(
            isinstance(prompt, TextPrompt) for prompt in request.prompts
        )
        text, geometric = _geometric_prompts(request)
        if len(geometric) > model.decoder.geometric_prompt_capacity:
            raise ValueError(
                "SAM3 request exceeds the decoder's geometric prompt capacity"
            )
        confidence, iou, max_instances = _request_options(
            request, self.config, output_profile=model.decoder.output_profile
        )
        output_shape = request.output_shape or ShapeType.POLYGON
        started = time.perf_counter()

        image_key = (request.model_revision, request.source_id)
        image_features = self._image_cache.get(image_key)
        image_encode_ms = 0.0
        if image_features is None:
            encode_started = time.perf_counter()
            image_features = model.encode_image(image_array)
            image_encode_ms = (time.perf_counter() - encode_started) * 1000
            cancellation.raise_if_cancelled()
            self._image_cache.put(image_key, image_features)

        text_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        language_features = self._text_cache.get(text_key)
        text_encode_ms = 0.0
        if language_features is None:
            text_started = time.perf_counter()
            language_features = model.encode_text(text)
            text_encode_ms = (time.perf_counter() - text_started) * 1000
            cancellation.raise_if_cancelled()
            self._text_cache.put(text_key, language_features)

        decode_started = time.perf_counter()
        detections = model.predict(
            image_features=image_features,
            language_features=language_features,
            geometric_prompts=geometric,
            original_size=image_array.shape[:2],
            confidence_threshold=confidence,
            nms_threshold=iou,
            max_instances=max_instances,
        )
        decode_ms = (time.perf_counter() - decode_started) * 1000
        cancellation.raise_if_cancelled()
        label = text.strip() if has_text_prompt else "AUTOLABEL_OBJECT"
        shapes = sam3_shapes(
            detections,
            output_shape=output_shape,
            label=label,
            config=self.config,
        )
        return InferenceResult(
            request_id=request.request_id,
            source_id=request.source_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
            shapes=shapes,
            warnings=self._provider_warnings,
            timings_ms={
                "image_encode": image_encode_ms,
                "text_encode": text_encode_ms,
                "decode": decode_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        )

    def preload(
        self,
        image: Any,
        source_id: str,
        cancellation: CancellationToken | None = None,
    ) -> None:
        token = CancellationToken.linked(cancellation, self._shutdown)
        try:
            with self._operation_lock:
                if self.state is not SessionState.READY:
                    raise SessionLifecycleError(
                        f"Cannot preload with a session in state {self.state.value!r}"
                    )
                token.raise_if_cancelled()
                key = (self.capabilities.model_revision, source_id)
                if key not in self._image_cache:
                    if self._model is None:
                        raise SessionLifecycleError("SAM3 runtime is not loaded")
                    features = self._model.encode_image(self._validated_image(image))
                    token.raise_if_cancelled()
                    self._image_cache.put(key, features)
        finally:
            token.close()

    def _unload(self) -> None:
        self._image_cache.clear()
        self._text_cache.clear()
        self._model = None
        self._provider_warnings = ()
        if self.config.release_cpu_memory_on_unload:
            release_unused_cpu_memory()


class Sam3Backend(InferenceBackend):
    backend_id = "sam3"

    def capabilities(self, config: Mapping[str, Any]) -> ModelCapabilities:
        validated = Sam3Config.model_validate(config)
        return ModelCapabilities(
            model_id=validated.name,
            model_revision=validated.revision,
            tasks=(ModelTask.PROMPTABLE_SEGMENTATION,),
            supports_cancellation=True,
            metadata={
                "backend": self.backend_id,
                "prompt_types": "text,point,box",
                "image_cache_items": validated.image_cache_items,
                "max_instances": validated.max_instances,
                "artifact_license": "SAM-License",
            },
        )

    def create_session(self, config: Mapping[str, Any]) -> Sam3Session:
        return Sam3Session(config)


__all__ = ["Sam3Backend", "Sam3Config", "Sam3Session", "sam3_shapes"]

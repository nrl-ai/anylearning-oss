"""Promptable SAM and SAM2 inference behind the shared runtime boundary."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
from .sam2_onnx import SegmentAnything2ONNX
from .sam_onnx import SegmentAnythingONNX

logger = logging.getLogger(__name__)

_MAX_MODEL_BYTES = 20 * 1024**3
_MAX_EXTERNAL_DATA_BYTES = 100 * 1024**3
_MAX_IMAGE_PIXELS = 100_000_000
_MAX_MASK_CONTOURS = 4_096
_MAX_SHAPES = 256
_MAX_POLYGON_POINTS = 4_096
_MAX_TOTAL_SHAPE_POINTS = 10_000


def _default_data_root() -> Path:
    """Resolve the legacy model store without importing desktop configuration.

    The inference package is intentionally installable without the desktop
    application's YAML and logging dependencies. Keep this fallback compatible
    with the application while avoiding its import-time directory creation.
    """
    configured = os.environ.get("ANYLEARNING_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "anylearning-data"


class _FrozenStringMap(dict[str, str]):
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


class SamOnnxConfig(BaseModel):
    """Compatible but bounded configuration for paired SAM ONNX graphs."""

    model_config = ConfigDict(extra="allow", frozen=True)

    name: str = Field(min_length=1, max_length=512)
    encoder_model_path: Path
    decoder_model_path: Path
    config_file: Path | None = None
    model_revision: str | None = Field(default=None, min_length=1, max_length=512)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    encoder_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    decoder_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    encoder_external_data_sha256: dict[str, str] = Field(default_factory=dict)
    decoder_external_data_sha256: dict[str, str] = Field(default_factory=dict)
    family: Literal["auto", "sam", "sam2"] = "auto"
    max_model_bytes: int = Field(default=_MAX_MODEL_BYTES, ge=1, le=40 * 1024**3)
    max_external_data_bytes: int = Field(
        default=_MAX_EXTERNAL_DATA_BYTES,
        ge=1,
        le=1024 * 1024**3,
    )
    max_image_pixels: int = Field(default=_MAX_IMAGE_PIXELS, ge=1, le=400_000_000)
    max_mask_contours: int = Field(default=_MAX_MASK_CONTOURS, ge=1, le=100_000)
    max_shapes: int = Field(default=_MAX_SHAPES, ge=1, le=10_000)
    max_polygon_points: int = Field(default=_MAX_POLYGON_POINTS, ge=3, le=100_000)
    max_total_shape_points: int = Field(
        default=_MAX_TOTAL_SHAPE_POINTS, ge=3, le=1_000_000
    )
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    allow_cpu_fallback: bool = True
    enable_cpu_mem_arena: bool = False
    release_cpu_memory_on_unload: bool = True
    intra_op_threads: int = Field(default=0, ge=0, le=256)
    inter_op_threads: int = Field(default=0, ge=0, le=256)

    @field_validator(
        "encoder_model_path", "decoder_model_path", "config_file", mode="before"
    )
    @classmethod
    def reject_empty_paths(cls, value: Any) -> Any:
        if value is not None and (not isinstance(value, (str, Path)) or not str(value)):
            raise ValueError("Model paths must be non-empty strings or paths")
        return value

    @field_validator("providers")
    @classmethod
    def validate_providers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 16:
            raise ValueError("providers must contain between 1 and 16 names")
        return value

    @field_validator("encoder_external_data_sha256", "decoder_external_data_sha256")
    @classmethod
    def validate_external_manifest(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 1_024:
            raise ValueError("External-data manifest may contain at most 1024 files")
        for location, digest in value.items():
            if not location or len(location.encode("utf-8")) > 4_096:
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
    def validate_distinct_graphs(self) -> Self:
        if self.encoder_model_path == self.decoder_model_path:
            raise ValueError("SAM encoder and decoder must be different graph files")
        return self

    def resolved_path(self, field: str) -> Path:
        return _resolved_model_path(self.model_dump(), field)

    @property
    def revision(self) -> str:
        return _model_revision(self.model_dump())


def _resolved_model_path(config: Mapping[str, Any], field: str) -> Path:
    value = config.get(field)
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"Missing model configuration field: {field}")
    configured = Path(value).expanduser()
    config_file = config.get("config_file")
    if configured.is_absolute():
        candidate = configured
    elif isinstance(config_file, (str, Path)) and str(config_file):
        candidate = Path(config_file).expanduser().resolve().parent / configured
    else:
        candidate = configured
    if candidate.is_file():
        return candidate.resolve()

    model_name = config.get("name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Missing model configuration field: name")
    model_root = (_default_data_root() / "models" / model_name).resolve()
    relative = Path(configured.name) if configured.is_absolute() else configured
    fallback = (model_root / relative).resolve()
    try:
        fallback.relative_to(model_root)
    except ValueError as error:
        raise ValueError(f"Model path escapes its model directory: {value}") from error
    return fallback


def _model_revision(config: Mapping[str, Any]) -> str:
    explicit = config.get("model_revision")
    if isinstance(explicit, str) and explicit:
        return explicit
    archive_digest = config.get("sha256")
    graph_hashes_configured = any(
        config.get(field)
        for field in (
            "encoder_sha256",
            "decoder_sha256",
            "encoder_external_data_sha256",
            "decoder_external_data_sha256",
        )
    )
    if (
        isinstance(archive_digest, str)
        and len(archive_digest) == 64
        and not graph_hashes_configured
    ):
        return f"sha256:{archive_digest.lower()}"

    identity = {
        "name": config.get("name"),
        "archive_sha256": archive_digest,
    }
    for role, path_field, digest_field, external_field in (
        (
            "encoder",
            "encoder_model_path",
            "encoder_sha256",
            "encoder_external_data_sha256",
        ),
        (
            "decoder",
            "decoder_model_path",
            "decoder_sha256",
            "decoder_external_data_sha256",
        ),
    ):
        digest = config.get(digest_field)
        external = config.get(external_field)
        identity[role] = local_onnx_bundle_revision(
            _resolved_model_path(config, path_field),
            sha256=digest if isinstance(digest, str) else None,
            external_data_sha256=external if isinstance(external, Mapping) else None,
        )
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return f"onnx-pair-sha256:{hashlib.sha256(payload).hexdigest()}"


def image_source_id(image: Any, filename: str | Path | None = None) -> str:
    """Hash the decoded pixels that the model sees, independent of encoding."""
    del filename  # Compatibility with callers that also know a source path.
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("Image must be a two- or three-dimensional array")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return f"image-sha256:{digest.hexdigest()}"


def _legacy_prompts(request: InferenceRequest) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for prompt in request.prompts:
        if isinstance(prompt, PointPrompt):
            prompts.append(
                {
                    "type": "point",
                    "data": [prompt.point.x, prompt.point.y],
                    "label": 1 if prompt.foreground else 0,
                }
            )
        elif isinstance(prompt, BoxPrompt):
            prompts.append(
                {
                    "type": "rectangle",
                    "data": [
                        prompt.top_left.x,
                        prompt.top_left.y,
                        prompt.bottom_right.x,
                        prompt.bottom_right.y,
                    ],
                    "label": 1,
                }
            )
        elif isinstance(prompt, TextPrompt):
            raise ValueError("This SAM backend supports only point and box prompts")
    if not prompts:
        raise ValueError("Promptable segmentation requires at least one prompt")
    return prompts


def mask_contours(
    mask: np.ndarray,
    *,
    max_mask_contours: int = _MAX_MASK_CONTOURS,
    max_shapes: int = _MAX_SHAPES,
    max_polygon_points: int = _MAX_POLYGON_POINTS,
) -> list[np.ndarray]:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) > max_mask_contours:
        raise ValueError(
            f"SAM mask has {len(contours)} contours; configured limit is "
            f"{max_mask_contours}"
        )
    approximated = []
    for contour in contours:
        epsilon = max(0.5, 0.001 * cv2.arcLength(contour, True))
        points = cv2.approxPolyDP(contour, epsilon, True)
        for _ in range(12):
            if len(points) <= max_polygon_points:
                break
            epsilon *= 2
            points = cv2.approxPolyDP(contour, epsilon, True)
        if len(points) > max_polygon_points:
            raise ValueError("SAM contour exceeds the configured polygon point limit")
        approximated.append(points)
    approximated = [contour for contour in approximated if len(contour) >= 3]
    if len(approximated) <= 1:
        return approximated

    image_area = binary.shape[0] * binary.shape[1]
    without_background = [
        contour
        for contour in approximated
        if cv2.contourArea(contour) < image_area * 0.9
    ]
    if without_background:
        approximated = without_background

    areas = np.asarray([cv2.contourArea(item) for item in approximated])
    threshold = float(areas.mean()) * 0.2
    kept = [
        contour
        for contour, area in zip(approximated, areas, strict=True)
        if area > threshold
    ]
    if len(kept) > max_shapes:
        raise ValueError(
            f"SAM mask has {len(kept)} editable shapes; configured limit is "
            f"{max_shapes}"
        )
    return kept


def mask_shapes(
    mask: np.ndarray,
    output_shape: ShapeType,
    *,
    max_mask_contours: int = _MAX_MASK_CONTOURS,
    max_shapes: int = _MAX_SHAPES,
    max_polygon_points: int = _MAX_POLYGON_POINTS,
    max_total_shape_points: int = _MAX_TOTAL_SHAPE_POINTS,
) -> tuple[InferenceShape, ...]:
    contours = mask_contours(
        mask,
        max_mask_contours=max_mask_contours,
        max_shapes=max_shapes,
        max_polygon_points=max_polygon_points,
    )
    if output_shape is ShapeType.POLYGON:
        shapes: list[InferenceShape] = []
        total_points = 0
        for contour in contours:
            coordinates = contour.reshape(-1, 2).astype(int).tolist()
            coordinates.append(coordinates[0])
            total_points += len(coordinates)
            if total_points > max_total_shape_points:
                raise ValueError("SAM polygons exceed the configured total point limit")
            shapes.append(
                InferenceShape(
                    type=ShapeType.POLYGON,
                    points=tuple(Point(x=x, y=y) for x, y in coordinates),
                    label="AUTOLABEL_OBJECT",
                )
            )
        return tuple(shapes)

    if output_shape is ShapeType.RECTANGLE and contours:
        if sum(len(item) for item in contours) > max_total_shape_points:
            raise ValueError("SAM contours exceed the configured total point limit")
        points = np.concatenate([item.reshape(-1, 2) for item in contours])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        return (
            InferenceShape(
                type=ShapeType.RECTANGLE,
                points=(Point(x=x_min, y=y_min), Point(x=x_max, y=y_max)),
                label="AUTOLABEL_OBJECT",
            ),
        )
    if output_shape is not ShapeType.RECTANGLE:
        raise ValueError("SAM output_shape must be polygon or rectangle")
    return ()


class SegmentAnythingSession(BaseInferenceSession):
    """One loaded SAM encoder/decoder pair with a revision-aware cache."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = SamOnnxConfig.model_validate(config)
        self._model: SegmentAnythingONNX | SegmentAnything2ONNX | None = None
        self._embedding_cache: LRUCache[tuple[str, str], dict[str, Any]] = LRUCache(10)
        self._provider_warnings: tuple[str, ...] = ()
        super().__init__(SegmentAnythingBackend().capabilities(config))

    def _load(self, cancellation: CancellationToken) -> None:
        encoder = self.config.resolved_path("encoder_model_path")
        decoder = self.config.resolved_path("decoder_model_path")
        encoder_session, _encoder_graph, encoder_warnings = create_checked_onnx_session(
            encoder,
            providers=self.config.providers,
            allow_cpu_fallback=self.config.allow_cpu_fallback,
            max_model_bytes=self.config.max_model_bytes,
            expected_sha256=self.config.encoder_sha256,
            external_data_sha256=self.config.encoder_external_data_sha256,
            max_external_data_bytes=self.config.max_external_data_bytes,
            enable_cpu_mem_arena=self.config.enable_cpu_mem_arena,
            intra_op_threads=self.config.intra_op_threads,
            inter_op_threads=self.config.inter_op_threads,
            cancellation=cancellation,
        )
        decoder_session, decoder_graph, decoder_warnings = create_checked_onnx_session(
            decoder,
            providers=self.config.providers,
            allow_cpu_fallback=self.config.allow_cpu_fallback,
            max_model_bytes=self.config.max_model_bytes,
            expected_sha256=self.config.decoder_sha256,
            external_data_sha256=self.config.decoder_external_data_sha256,
            max_external_data_bytes=self.config.max_external_data_bytes,
            enable_cpu_mem_arena=self.config.enable_cpu_mem_arena,
            intra_op_threads=self.config.intra_op_threads,
            inter_op_threads=self.config.inter_op_threads,
            cancellation=cancellation,
        )
        is_sam2 = any(
            item.name == "high_res_feats_0" for item in decoder_graph.graph.input
        )
        detected_family = "sam2" if is_sam2 else "sam"
        if self.config.family != "auto" and self.config.family != detected_family:
            raise ValueError(
                f"Configured SAM family {self.config.family!r} does not match "
                f"decoder graph family {detected_family!r}"
            )
        adapter = SegmentAnything2ONNX if is_sam2 else SegmentAnythingONNX
        cancellation.raise_if_cancelled()
        self._model = adapter(encoder_session, decoder_session)
        self._provider_warnings = tuple(
            dict.fromkeys((*encoder_warnings, *decoder_warnings))
        )
        for warning in self._provider_warnings:
            logger.warning("SAM ONNX provider selection: %s", warning)

    def _validated_image(self, image: Any) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("SAM expects an H x W x 3 uint8 RGB image")
        pixels = int(array.shape[0]) * int(array.shape[1])
        if pixels <= 0 or pixels > self.config.max_image_pixels:
            raise ValueError(
                f"SAM image has {pixels} pixels; configured limit is "
                f"{self.config.max_image_pixels}"
            )
        return array

    def _predict(
        self,
        request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken,
    ) -> InferenceResult:
        if self._model is None:
            raise SessionLifecycleError("SAM runtime is not loaded")
        image_array = self._validated_image(image)
        prompts = _legacy_prompts(request)
        output_shape = request.output_shape or ShapeType.POLYGON
        cache_key = (request.model_revision, request.source_id)
        started = time.perf_counter()
        embedding = self._embedding_cache.get(cache_key)
        encode_ms = 0.0
        if embedding is None:
            encode_started = time.perf_counter()
            embedding = self._model.encode(image_array)
            encode_ms = (time.perf_counter() - encode_started) * 1000
            cancellation.raise_if_cancelled()
            self._embedding_cache.put(cache_key, embedding)
        decode_started = time.perf_counter()
        masks = np.asarray(self._model.predict_masks(embedding, prompts))
        decode_ms = (time.perf_counter() - decode_started) * 1000
        cancellation.raise_if_cancelled()
        mask = masks[0, 0] if masks.ndim == 4 else masks[0]
        shapes = mask_shapes(
            mask,
            output_shape,
            max_mask_contours=self.config.max_mask_contours,
            max_shapes=self.config.max_shapes,
            max_polygon_points=self.config.max_polygon_points,
            max_total_shape_points=self.config.max_total_shape_points,
        )
        return InferenceResult(
            request_id=request.request_id,
            source_id=request.source_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
            shapes=shapes,
            timings_ms={
                "encode": encode_ms,
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
                cache_key = (self.capabilities.model_revision, source_id)
                if cache_key not in self._embedding_cache:
                    if self._model is None:
                        raise SessionLifecycleError("SAM runtime is not loaded")
                    embedding = self._model.encode(self._validated_image(image))
                    token.raise_if_cancelled()
                    self._embedding_cache.put(cache_key, embedding)
        finally:
            token.close()

    def _unload(self) -> None:
        self._embedding_cache.clear()
        self._model = None
        self._provider_warnings = ()
        if self.config.release_cpu_memory_on_unload:
            release_unused_cpu_memory()


class SegmentAnythingBackend(InferenceBackend):
    backend_id = "segment_anything"

    def capabilities(self, config: Mapping[str, Any]) -> ModelCapabilities:
        validated = SamOnnxConfig.model_validate(config)
        return ModelCapabilities(
            model_id=validated.name,
            model_revision=validated.revision,
            tasks=(ModelTask.PROMPTABLE_SEGMENTATION,),
            supports_cancellation=True,
            metadata={"backend": self.backend_id, "embedding_cache_items": 10},
        )

    def create_session(self, config: Mapping[str, Any]) -> SegmentAnythingSession:
        return SegmentAnythingSession(config)


__all__ = [
    "SamOnnxConfig",
    "SegmentAnythingBackend",
    "SegmentAnythingSession",
    "image_source_id",
    "mask_contours",
    "mask_shapes",
]

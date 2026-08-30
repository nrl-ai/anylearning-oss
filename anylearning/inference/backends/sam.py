"""Promptable SAM and SAM2 inference behind the shared runtime boundary."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx

from anylearning.config import DATA_ROOT

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
)
from ..runtime import (
    BaseInferenceSession,
    CancellationToken,
    InferenceBackend,
    SessionLifecycleError,
    SessionState,
)
from .sam2_onnx import SegmentAnything2ONNX
from .sam_onnx import SegmentAnythingONNX

logger = logging.getLogger(__name__)


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
    model_root = (Path(DATA_ROOT) / "models" / model_name).resolve()
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
    if isinstance(archive_digest, str) and len(archive_digest) == 64:
        return f"sha256:{archive_digest.lower()}"

    identity = {
        "name": config.get("name"),
        "encoder": str(config.get("encoder_model_path", "")),
        "decoder": str(config.get("decoder_model_path", "")),
    }
    for role, field in (
        ("encoder_stat", "encoder_model_path"),
        ("decoder_stat", "decoder_model_path"),
    ):
        try:
            stat = _resolved_model_path(config, field).stat()
            identity[role] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        except OSError:
            identity[role] = None
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return f"config-sha256:{hashlib.sha256(payload).hexdigest()}"


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
    if not prompts:
        raise ValueError("Promptable segmentation requires at least one prompt")
    return prompts


def mask_contours(mask: np.ndarray) -> list[np.ndarray]:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    approximated = [
        cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
        for contour in contours
    ]
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
    return [contour for contour, area in zip(approximated, areas) if area > threshold]


def mask_shapes(
    mask: np.ndarray, output_shape: ShapeType
) -> tuple[InferenceShape, ...]:
    contours = mask_contours(mask)
    if output_shape is ShapeType.POLYGON:
        shapes: list[InferenceShape] = []
        for contour in contours:
            coordinates = contour.reshape(-1, 2).astype(int).tolist()
            coordinates.append(coordinates[0])
            shapes.append(
                InferenceShape(
                    type=ShapeType.POLYGON,
                    points=tuple(Point(x=x, y=y) for x, y in coordinates),
                    label="AUTOLABEL_OBJECT",
                )
            )
        return tuple(shapes)

    if output_shape is ShapeType.RECTANGLE and contours:
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
        self._config = dict(config)
        self._model: SegmentAnythingONNX | SegmentAnything2ONNX | None = None
        self._embedding_cache: LRUCache[tuple[str, str], dict[str, Any]] = LRUCache(10)
        super().__init__(SegmentAnythingBackend().capabilities(config))

    def _load(self, cancellation: CancellationToken) -> None:
        encoder = _resolved_model_path(self._config, "encoder_model_path")
        decoder = _resolved_model_path(self._config, "decoder_model_path")
        for role, path in (("encoder", encoder), ("decoder", decoder)):
            if not path.is_file():
                raise FileNotFoundError(f"Segment Anything {role} not found: {path}")
        cancellation.raise_if_cancelled()
        graph = onnx.load_model(str(decoder), load_external_data=False).graph
        adapter = (
            SegmentAnything2ONNX
            if any(item.name == "high_res_feats_0" for item in graph.input)
            else SegmentAnythingONNX
        )
        cancellation.raise_if_cancelled()
        self._model = adapter(str(encoder), str(decoder))

    def _predict(
        self,
        request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken,
    ) -> InferenceResult:
        if self._model is None:
            raise SessionLifecycleError("SAM runtime is not loaded")
        prompts = _legacy_prompts(request)
        output_shape = request.output_shape or ShapeType.POLYGON
        cache_key = (request.model_revision, request.source_id)
        started = time.perf_counter()
        embedding = self._embedding_cache.get(cache_key)
        encode_ms = 0.0
        if embedding is None:
            encode_started = time.perf_counter()
            embedding = self._model.encode(np.asarray(image))
            encode_ms = (time.perf_counter() - encode_started) * 1000
            cancellation.raise_if_cancelled()
            self._embedding_cache.put(cache_key, embedding)
        decode_started = time.perf_counter()
        masks = np.asarray(self._model.predict_masks(embedding, prompts))
        decode_ms = (time.perf_counter() - decode_started) * 1000
        cancellation.raise_if_cancelled()
        mask = masks[0, 0] if masks.ndim == 4 else masks[0]
        shapes = mask_shapes(mask, output_shape)
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
                    embedding = self._model.encode(np.asarray(image))
                    token.raise_if_cancelled()
                    self._embedding_cache.put(cache_key, embedding)
        finally:
            token.close()

    def _unload(self) -> None:
        self._embedding_cache.clear()
        self._model = None


class SegmentAnythingBackend(InferenceBackend):
    backend_id = "segment_anything"

    def capabilities(self, config: Mapping[str, Any]) -> ModelCapabilities:
        model_id = config.get("name")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Missing model configuration field: name")
        return ModelCapabilities(
            model_id=model_id,
            model_revision=_model_revision(config),
            tasks=(ModelTask.PROMPTABLE_SEGMENTATION,),
            supports_cancellation=True,
            metadata={"backend": self.backend_id, "embedding_cache_items": 10},
        )

    def create_session(self, config: Mapping[str, Any]) -> SegmentAnythingSession:
        return SegmentAnythingSession(config)

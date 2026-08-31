"""Bounded detector-box to promptable-segmentation composition.

This backend orchestrates existing AnyLearning inference sessions. It does not
load model formats itself: the detector and segmenter remain responsible for
their own integrity-checked ONNX boundaries. The processing order follows the
official SAM predictor contract by reusing one image embedding while decoding
one independent box prompt per detected object.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..contracts import (
    BoxPrompt,
    InferenceRequest,
    InferenceResult,
    InferenceShape,
    ModelCapabilities,
    ModelTask,
    Point,
    ShapeType,
)
from ..runtime import (
    BaseInferenceSession,
    CancellationToken,
    InferenceBackend,
    InferenceSession,
    ModelRegistry,
    SessionState,
)

_BACKEND_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_MAX_IMAGE_PIXELS = 100_000_000
_MAX_REFINEMENTS = 1_000
_MAX_SHAPES = 10_000
_MAX_TOTAL_POINTS = 1_000_000
_MAX_WARNINGS = 128
_COMPOSITE_BACKEND_ID = "detector_sam"


class PipelineModelConfig(BaseModel):
    """One nested model selected from the existing lazy registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str = Field(min_length=1, max_length=128)
    config: dict[str, Any] = Field(max_length=2_048)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        if not _BACKEND_ID.fullmatch(value):
            raise ValueError("pipeline backend identifier is invalid")
        if value == _COMPOSITE_BACKEND_ID:
            raise ValueError("detector_sam pipelines cannot contain themselves")
        return value

    @field_validator("config")
    @classmethod
    def validate_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_config_value(value)
        return value


class DetectorSamConfig(BaseModel):
    """Strict resource and child-session configuration for refinement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=512)
    model_revision: str | None = Field(default=None, min_length=1, max_length=512)
    config_file: Path | None = None
    detector: PipelineModelConfig
    segmenter: PipelineModelConfig
    max_image_pixels: int = Field(default=_MAX_IMAGE_PIXELS, ge=1, le=400_000_000)
    max_refinements: int = Field(default=100, ge=1, le=_MAX_REFINEMENTS)
    max_shapes: int = Field(default=1_000, ge=1, le=_MAX_SHAPES)
    max_total_points: int = Field(
        default=100_000,
        ge=3,
        le=_MAX_TOTAL_POINTS,
    )
    box_padding_pixels: float = Field(
        default=0,
        ge=0,
        le=10_000,
        allow_inf_nan=False,
    )
    box_prompt_grid_pixels: float = Field(
        default=1,
        ge=0.25,
        le=1_024,
        allow_inf_nan=False,
    )
    output_score_decimals: int = Field(default=3, ge=0, le=8)
    minimum_box_area_pixels: float = Field(
        default=1,
        ge=0,
        le=160_000_000_000_000_000,
        allow_inf_nan=False,
    )
    fallback_to_box: bool = True

    @field_validator("config_file", mode="before")
    @classmethod
    def reject_empty_config_path(cls, value: Any) -> Any:
        if value is not None and (not isinstance(value, (str, Path)) or not str(value)):
            raise ValueError("config_file must be a non-empty string or path")
        return value

    @model_validator(mode="after")
    def validate_child_roles(self) -> Self:
        if self.detector == self.segmenter:
            raise ValueError("detector and segmenter configurations must be different")
        return self

    def child_config(self, child: PipelineModelConfig) -> dict[str, Any]:
        values = dict(child.config)
        if self.config_file is not None:
            # A server startup manifest is the trusted anchor for every nested
            # relative artifact path. Nested values cannot replace that anchor.
            values["config_file"] = self.config_file
        return values


def _validate_config_value(value: Any, *, depth: int = 0) -> None:
    if depth > 8:
        raise ValueError("pipeline model configuration nesting exceeds 8 levels")
    if isinstance(value, dict):
        if len(value) > 2_048:
            raise ValueError("pipeline model configuration mapping is too large")
        for key, child in value.items():
            if not isinstance(key, str) or not key or len(key) > 512:
                raise ValueError("pipeline model configuration keys are invalid")
            _validate_config_value(child, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 100_000:
            raise ValueError("pipeline model configuration sequence is too large")
        for child in value:
            _validate_config_value(child, depth=depth + 1)
        return
    if value is not None and not isinstance(value, (str, Path, int, float, bool)):
        raise ValueError("pipeline model configuration contains an unsupported value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("pipeline model configuration numbers must be finite")
    if isinstance(value, (str, Path)) and len(str(value)) > 8_192:
        raise ValueError("pipeline model configuration string is too large")


def _pipeline_capabilities(
    config: DetectorSamConfig,
    registry: ModelRegistry,
) -> tuple[ModelCapabilities, ModelCapabilities, ModelCapabilities]:
    detector_config = config.child_config(config.detector)
    segmenter_config = config.child_config(config.segmenter)
    detector = registry.get(config.detector.backend).capabilities(detector_config)
    segmenter = registry.get(config.segmenter.backend).capabilities(segmenter_config)
    capabilities = _composite_capabilities(config, detector, segmenter)
    return capabilities, detector, segmenter


def _composite_capabilities(
    config: DetectorSamConfig,
    detector: ModelCapabilities,
    segmenter: ModelCapabilities,
) -> ModelCapabilities:
    if ModelTask.DETECTION not in detector.tasks:
        raise ValueError("detector_sam detector must advertise detection capability")
    if ModelTask.PROMPTABLE_SEGMENTATION not in segmenter.tasks:
        raise ValueError(
            "detector_sam segmenter must advertise promptable segmentation capability"
        )
    identity = {
        "box_padding_pixels": config.box_padding_pixels,
        "box_prompt_grid_pixels": config.box_prompt_grid_pixels,
        "configured_model_revision": config.model_revision,
        "detector_backend": config.detector.backend,
        "detector_model_id": detector.model_id,
        "detector_revision": detector.model_revision,
        "fallback_to_box": config.fallback_to_box,
        "max_image_pixels": config.max_image_pixels,
        "max_refinements": config.max_refinements,
        "max_shapes": config.max_shapes,
        "max_total_points": config.max_total_points,
        "minimum_box_area_pixels": config.minimum_box_area_pixels,
        "output_score_decimals": config.output_score_decimals,
        "schema": 1,
        "segmenter_backend": config.segmenter.backend,
        "segmenter_model_id": segmenter.model_id,
        "segmenter_revision": segmenter.model_revision,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    revision = f"detector-sam-sha256:{hashlib.sha256(encoded).hexdigest()}"
    return ModelCapabilities(
        model_id=config.name,
        model_revision=revision,
        tasks=(ModelTask.INSTANCE_SEGMENTATION,),
        supports_cancellation=True,
        metadata={
            "backend": _COMPOSITE_BACKEND_ID,
            "detector_backend": config.detector.backend,
            "detector_model_id": detector.model_id,
            "detector_revision": detector.model_revision,
            "segmenter_backend": config.segmenter.backend,
            "segmenter_model_id": segmenter.model_id,
            "segmenter_revision": segmenter.model_revision,
            "processing": "encode-once-box-per-object",
        },
    )


def _validated_image(image: Any, *, max_image_pixels: int) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("detector_sam expects an H x W x 3 uint8 RGB image")
    pixels = int(array.shape[0]) * int(array.shape[1])
    if pixels <= 0 or pixels > max_image_pixels:
        raise ValueError(
            f"detector_sam image has {pixels} pixels; configured limit is "
            f"{max_image_pixels}"
        )
    return array


def _box_prompt(
    shape: InferenceShape,
    *,
    image_height: int,
    image_width: int,
    padding: float,
    minimum_area: float,
    grid_pixels: float,
) -> BoxPrompt | None:
    if shape.type is not ShapeType.RECTANGLE:
        raise ValueError("detector_sam detector results must contain rectangles")
    first, second = shape.points
    x1 = max(0.0, min(first.x, second.x) - padding)
    y1 = max(0.0, min(first.y, second.y) - padding)
    x2 = min(float(image_width), max(first.x, second.x) + padding)
    y2 = min(float(image_height), max(first.y, second.y) + padding)
    if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < minimum_area:
        return None
    x1 = max(0.0, math.floor(x1 / grid_pixels) * grid_pixels)
    y1 = max(0.0, math.floor(y1 / grid_pixels) * grid_pixels)
    x2 = min(float(image_width), math.ceil(x2 / grid_pixels) * grid_pixels)
    y2 = min(float(image_height), math.ceil(y2 / grid_pixels) * grid_pixels)
    return BoxPrompt(
        top_left=Point(x=x1, y=y1),
        bottom_right=Point(x=x2, y=y2),
    )


def _internal_request_id(request_id: str, role: str, index: int | None = None) -> str:
    digest = hashlib.sha256(
        request_id.encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:24]
    suffix = role if index is None else f"{role}-{index}"
    return f"pipeline-{digest}-{suffix}"


def _append_warning(warnings: list[str], value: str) -> None:
    bounded = value[:2_048]
    if bounded not in warnings and len(warnings) < _MAX_WARNINGS:
        warnings.append(bounded)


def _refined_shape(
    mask: InferenceShape,
    detection: InferenceShape,
    *,
    detection_index: int,
    score_decimals: int,
) -> InferenceShape:
    if mask.type is not ShapeType.POLYGON:
        raise ValueError("detector_sam segmenter results must contain polygons")
    attributes = dict(detection.attributes)
    if (
        mask.score is not None
        and "segmenter_score" not in attributes
        and len(attributes) < 128
    ):
        attributes["segmenter_score"] = round(mask.score, score_decimals)
    return InferenceShape(
        type=ShapeType.POLYGON,
        points=mask.points,
        label=detection.label,
        score=(
            None if detection.score is None else round(detection.score, score_decimals)
        ),
        group_id=(
            detection.group_id if detection.group_id is not None else detection_index
        ),
        attributes=attributes,
    )


class DetectorSamSession(BaseInferenceSession):
    """One serialized detector and promptable-segmenter session pair."""

    def __init__(self, config: Mapping[str, Any], registry: ModelRegistry) -> None:
        self.config = DetectorSamConfig.model_validate(config)
        self._registry = registry
        self._detector, self._segmenter = self._create_child_sessions()
        capabilities = _composite_capabilities(
            self.config,
            self._detector.capabilities,
            self._segmenter.capabilities,
        )
        super().__init__(capabilities)

    def _create_child_sessions(self) -> tuple[InferenceSession, InferenceSession]:
        detector = self._registry.create_session(
            self.config.detector.backend,
            self.config.child_config(self.config.detector),
        )
        try:
            segmenter = self._registry.create_session(
                self.config.segmenter.backend,
                self.config.child_config(self.config.segmenter),
            )
        except Exception:
            try:
                detector.unload()
            except Exception:
                pass
            raise
        return detector, segmenter

    def _load(self, cancellation: CancellationToken) -> None:
        if (
            self._detector.state is SessionState.CLOSED
            or self._segmenter.state is SessionState.CLOSED
        ):
            detector, segmenter = self._create_child_sessions()
            capabilities = _composite_capabilities(
                self.config,
                detector.capabilities,
                segmenter.capabilities,
            )
            if capabilities != self.capabilities:
                segmenter.unload()
                detector.unload()
                raise RuntimeError("detector_sam child model identity changed on retry")
            self._detector, self._segmenter = detector, segmenter
        self._detector.load(cancellation)
        cancellation.raise_if_cancelled()
        self._segmenter.load(cancellation)

    def _predict(
        self,
        request: InferenceRequest,
        image: Any,
        cancellation: CancellationToken,
    ) -> InferenceResult:
        if request.prompts:
            raise ValueError("detector_sam does not accept external prompts")
        if request.output_shape not in {None, ShapeType.POLYGON}:
            raise ValueError("detector_sam output_shape must be polygon")
        image_array = _validated_image(
            image,
            max_image_pixels=self.config.max_image_pixels,
        )
        image_height, image_width = image_array.shape[:2]
        started = time.perf_counter()
        detector_started = time.perf_counter()
        detector_request = InferenceRequest(
            request_id=_internal_request_id(request.request_id, "detector"),
            source_id=request.source_id,
            model_id=self._detector.capabilities.model_id,
            model_revision=self._detector.capabilities.model_revision,
            parameters=request.parameters,
        )
        detections = self._detector.predict(
            detector_request,
            image_array,
            cancellation,
        )
        detector_ms = (time.perf_counter() - detector_started) * 1000
        cancellation.raise_if_cancelled()

        warnings: list[str] = []
        for warning in detections.warnings:
            _append_warning(warnings, f"detector: {warning}")
        candidates = detections.shapes
        if len(candidates) > self.config.max_refinements:
            _append_warning(
                warnings,
                f"detector returned {len(candidates)} shapes; refined the first "
                f"{self.config.max_refinements}",
            )
            candidates = candidates[: self.config.max_refinements]

        segmenter_ms = 0.0
        segmenter_encode_ms = 0.0
        segmenter_decode_ms = 0.0
        shapes: list[InferenceShape] = []
        total_points = 0
        for index, detection in enumerate(candidates):
            cancellation.raise_if_cancelled()
            prompt = _box_prompt(
                detection,
                image_height=image_height,
                image_width=image_width,
                padding=self.config.box_padding_pixels,
                minimum_area=self.config.minimum_box_area_pixels,
                grid_pixels=self.config.box_prompt_grid_pixels,
            )
            if prompt is None:
                _append_warning(
                    warnings,
                    f"detector shape {index} was empty or below the minimum box area",
                )
                continue
            segmenter_request = InferenceRequest(
                request_id=_internal_request_id(request.request_id, "segmenter", index),
                source_id=request.source_id,
                model_id=self._segmenter.capabilities.model_id,
                model_revision=self._segmenter.capabilities.model_revision,
                prompts=(prompt,),
                output_shape=ShapeType.POLYGON,
            )
            segmenter_started = time.perf_counter()
            masks = self._segmenter.predict(
                segmenter_request,
                image_array,
                cancellation,
            )
            segmenter_ms += (time.perf_counter() - segmenter_started) * 1000
            segmenter_encode_ms += masks.timings_ms.get("encode", 0.0)
            segmenter_decode_ms += masks.timings_ms.get("decode", 0.0)
            for warning in masks.warnings:
                _append_warning(warnings, f"segmenter[{index}]: {warning}")
            refined = [
                _refined_shape(
                    mask,
                    detection,
                    detection_index=index,
                    score_decimals=self.config.output_score_decimals,
                )
                for mask in masks.shapes
            ]
            if not refined and self.config.fallback_to_box:
                refined = [
                    detection.model_copy(
                        update={
                            "score": (
                                None
                                if detection.score is None
                                else round(
                                    detection.score,
                                    self.config.output_score_decimals,
                                )
                            ),
                            "group_id": (
                                detection.group_id
                                if detection.group_id is not None
                                else index
                            ),
                        }
                    )
                ]
                _append_warning(
                    warnings,
                    f"segmenter returned no mask for shape {index}; kept detector box",
                )
            elif not refined:
                _append_warning(
                    warnings,
                    f"segmenter returned no mask for shape {index}; dropped detection",
                )
            new_points = sum(len(shape.points) for shape in refined)
            if len(shapes) + len(refined) > self.config.max_shapes:
                raise ValueError("detector_sam results exceed max_shapes")
            if total_points + new_points > self.config.max_total_points:
                raise ValueError("detector_sam results exceed max_total_points")
            shapes.extend(refined)
            total_points += new_points

        return InferenceResult(
            request_id=request.request_id,
            source_id=request.source_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
            shapes=tuple(shapes),
            warnings=tuple(warnings),
            timings_ms={
                "detector": detector_ms,
                "segmenter": segmenter_ms,
                "segmenter_encode": segmenter_encode_ms,
                "segmenter_decode": segmenter_decode_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        )

    def _unload(self) -> None:
        errors: list[Exception] = []
        for session in (self._segmenter, self._detector):
            try:
                session.unload()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError("detector_sam child session unload failed") from errors[
                0
            ]


class DetectorSamBackend(InferenceBackend):
    backend_id = _COMPOSITE_BACKEND_ID

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        if registry is None:
            from ..defaults import create_default_registry

            registry = create_default_registry()
        self._registry = registry

    def capabilities(self, config: Mapping[str, Any]) -> ModelCapabilities:
        validated = DetectorSamConfig.model_validate(config)
        capabilities, _detector, _segmenter = _pipeline_capabilities(
            validated,
            self._registry,
        )
        return capabilities

    def create_session(self, config: Mapping[str, Any]) -> DetectorSamSession:
        return DetectorSamSession(config, self._registry)


__all__ = [
    "DetectorSamBackend",
    "DetectorSamConfig",
    "DetectorSamSession",
    "PipelineModelConfig",
]

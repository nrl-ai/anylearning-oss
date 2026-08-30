"""Versioned, transport-neutral contracts shared by inference consumers."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

CURRENT_PROTOCOL_VERSION = "1.0"
SUPPORTED_PROTOCOL_VERSIONS = (CURRENT_PROTOCOL_VERSION,)


class ContractModel(BaseModel):
    """Strict base model for data that can cross a process boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


FiniteCoordinate = Annotated[float, Field(allow_inf_nan=False)]
Milliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
MetadataKey = Annotated[str, Field(min_length=1, max_length=128)]
MetadataText = Annotated[str, Field(max_length=2048)]
MetadataInteger = Annotated[int, Field(ge=-(2**63), le=2**63 - 1)]
MetadataFloat = Annotated[float, Field(allow_inf_nan=False)]
MetadataValue = MetadataText | bool | MetadataInteger | MetadataFloat | None
ParameterTextList = Annotated[tuple[MetadataText, ...], Field(max_length=256)]
ParameterIntegerList = Annotated[tuple[MetadataInteger, ...], Field(max_length=256)]
ParameterValue = MetadataValue | ParameterTextList | ParameterIntegerList
WarningText = Annotated[str, Field(max_length=2048)]


class Point(ContractModel):
    """An image-space point in pixels."""

    x: FiniteCoordinate
    y: FiniteCoordinate


class PointPrompt(ContractModel):
    """A positive or negative point used by an interactive model."""

    type: Literal["point"] = "point"
    point: Point
    foreground: bool = True


class BoxPrompt(ContractModel):
    """An axis-aligned box used to constrain an interactive model."""

    type: Literal["box"] = "box"
    top_left: Point
    bottom_right: Point

    @model_validator(mode="after")
    def validate_ordered_corners(self) -> Self:
        if (
            self.bottom_right.x <= self.top_left.x
            or self.bottom_right.y <= self.top_left.y
        ):
            raise ValueError("Box prompt corners must have positive width and height")
        return self


InferencePrompt = Annotated[PointPrompt | BoxPrompt, Field(discriminator="type")]


class ShapeType(str, Enum):
    """Geometry representations supported by the first protocol version."""

    POINT = "point"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    ROTATED_RECTANGLE = "rotated_rectangle"


_REQUIRED_POINT_COUNTS: dict[ShapeType, tuple[int, str]] = {
    ShapeType.POINT: (1, "exactly 1 point"),
    ShapeType.RECTANGLE: (2, "exactly 2 opposite-corner points"),
    ShapeType.POLYGON: (3, "at least 3 points"),
    ShapeType.ROTATED_RECTANGLE: (4, "exactly 4 ordered corner points"),
}


class InferenceShape(ContractModel):
    """An editable, model-neutral shape returned by inference."""

    type: ShapeType
    points: tuple[Point, ...] = Field(max_length=100_000)
    label: str | None = Field(default=None, max_length=1024)
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    group_id: MetadataText | MetadataInteger | None = None
    attributes: dict[MetadataKey, MetadataValue] = Field(
        default_factory=dict, max_length=128
    )

    @model_validator(mode="after")
    def validate_point_count(self) -> Self:
        required, description = _REQUIRED_POINT_COUNTS[self.type]
        count = len(self.points)
        valid = (
            count >= required if self.type is ShapeType.POLYGON else count == required
        )
        if not valid:
            raise ValueError(
                f"{self.type.value} requires {description}; received {count}"
            )
        return self


class ModelTask(str, Enum):
    """Inference tasks a backend can advertise without prescribing UI."""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"
    PROMPTABLE_SEGMENTATION = "promptable_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


class ModelCapabilities(ContractModel):
    """Stable model discovery information for local and remote clients."""

    protocol_version: str = CURRENT_PROTOCOL_VERSION
    model_id: str = Field(min_length=1, max_length=512)
    model_revision: str = Field(min_length=1, max_length=512)
    tasks: tuple[ModelTask, ...] = Field(min_length=1, max_length=16)
    supports_batch: bool = False
    supports_cancellation: bool = False
    max_batch_size: int = Field(default=1, ge=1)
    metadata: dict[MetadataKey, MetadataValue] = Field(
        default_factory=dict, max_length=128
    )

    @field_validator("protocol_version")
    @classmethod
    def validate_protocol_version(cls, value: str) -> str:
        return _validate_protocol_version(value)

    @field_validator("tasks")
    @classmethod
    def validate_unique_tasks(
        cls, value: tuple[ModelTask, ...]
    ) -> tuple[ModelTask, ...]:
        if len(value) != len(set(value)):
            raise ValueError("Model capability tasks must be unique")
        return value

    @model_validator(mode="after")
    def validate_batch_capability(self) -> Self:
        if not self.supports_batch and self.max_batch_size != 1:
            raise ValueError(
                "max_batch_size must be 1 when batch inference is unsupported"
            )
        return self


class InferenceRequest(ContractModel):
    """A model-neutral prediction request, excluding the image transport."""

    protocol_version: str = CURRENT_PROTOCOL_VERSION
    request_id: str = Field(min_length=1, max_length=512)
    source_id: str = Field(min_length=1, max_length=2048)
    model_id: str = Field(min_length=1, max_length=512)
    model_revision: str = Field(min_length=1, max_length=512)
    prompts: tuple[InferencePrompt, ...] = Field(default=(), max_length=10_000)
    output_shape: ShapeType | None = None
    parameters: dict[MetadataKey, ParameterValue] = Field(
        default_factory=dict, max_length=128
    )

    @field_validator("protocol_version")
    @classmethod
    def validate_protocol_version(cls, value: str) -> str:
        return _validate_protocol_version(value)


class InferenceResult(ContractModel):
    """A versioned prediction result with stale-result protection metadata."""

    protocol_version: str = CURRENT_PROTOCOL_VERSION
    request_id: str = Field(min_length=1, max_length=512)
    source_id: str = Field(min_length=1, max_length=2048)
    model_id: str = Field(min_length=1, max_length=512)
    model_revision: str = Field(min_length=1, max_length=512)
    shapes: tuple[InferenceShape, ...] = Field(default=(), max_length=10_000)
    warnings: tuple[WarningText, ...] = Field(default=(), max_length=128)
    timings_ms: dict[MetadataKey, Milliseconds] = Field(
        default_factory=dict, max_length=64
    )

    @field_validator("protocol_version")
    @classmethod
    def validate_protocol_version(cls, value: str) -> str:
        return _validate_protocol_version(value)


def _validate_protocol_version(value: str) -> str:
    if value not in SUPPORTED_PROTOCOL_VERSIONS:
        supported = ", ".join(SUPPORTED_PROTOCOL_VERSIONS)
        raise ValueError(
            f"Unsupported protocol version {value!r}; supported: {supported}"
        )
    return value

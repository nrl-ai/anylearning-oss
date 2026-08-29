"""Shared value objects used by auto-labeling models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Sequence


@dataclass(slots=True)
class AutoLabelingResult:
    """Shapes produced by a model and how they should update the canvas."""

    shapes: Sequence[Any]
    replace: bool = True


@dataclass(frozen=True, slots=True)
class AutoLabelingMode:
    """Canvas interaction mode used to collect model prompts."""

    OBJECT: ClassVar[str] = "AUTOLABEL_OBJECT"
    ADD: ClassVar[str] = "AUTOLABEL_ADD"
    REMOVE: ClassVar[str] = "AUTOLABEL_REMOVE"
    POINT: ClassVar[str] = "point"
    RECTANGLE: ClassVar[str] = "rectangle"

    edit_mode: str | None
    shape_type: str | None

    @classmethod
    def get_default_mode(cls) -> "AutoLabelingMode":
        return cls(cls.ADD, cls.POINT)


AutoLabelingMode.NONE = AutoLabelingMode(None, None)

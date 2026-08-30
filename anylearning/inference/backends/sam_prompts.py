"""Validation and ONNX conversion for SAM segmentation prompts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def prompt_arrays(prompt: Iterable[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert point and rectangle marks to the coordinate/label ONNX format."""
    points: list[list[float]] = []
    labels: list[float] = []
    for mark in prompt:
        kind = mark.get("type")
        data = mark.get("data")
        if kind == "point":
            if not isinstance(data, (list, tuple)) or len(data) != 2:
                raise ValueError("Point prompts require [x, y]")
            points.append([float(data[0]), float(data[1])])
            labels.append(float(mark.get("label", 1)))
        elif kind == "rectangle":
            if not isinstance(data, (list, tuple)) or len(data) != 4:
                raise ValueError("Rectangle prompts require [x1, y1, x2, y2]")
            points.extend(
                ([float(data[0]), float(data[1])], [float(data[2]), float(data[3])])
            )
            labels.extend((2.0, 3.0))
        else:
            raise ValueError(f"Unsupported prompt type: {kind!r}")
    if not points:
        raise ValueError("At least one prompt is required")
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.float32)

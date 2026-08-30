"""Interactive Segment Anything model integration."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from anylearning.inference import (
    BoxPrompt,
    InferenceRequest,
    PointPrompt,
    ShapeType,
    get_default_registry,
)
from anylearning.inference import (
    Point as InferencePoint,
)
from anylearning.inference.backends.sam import (
    SegmentAnythingSession,
    image_source_id,
    mask_contours,
    mask_shapes,
)

from .model import Model
from .types import AutoLabelingResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Point:
    x: float
    y: float


@dataclass(slots=True)
class Shape:
    points: list[Point]
    shape_type: str
    closed: bool = True
    fill_color: str = "#000000"
    line_color: str = "#000000"
    line_width: int = 1
    label: str = "AUTOLABEL_OBJECT"
    selected: bool = False
    flags: dict[str, Any] = field(default_factory=dict)

    def add_point(self, point: Point) -> None:
        self.points.append(point)


class SegmentAnything(Model):
    """Create polygons or rectangles from SAM point/box prompts."""

    class Meta:
        required_config_names = (
            "type",
            "name",
            "display_name",
            "encoder_model_path",
            "decoder_model_path",
        )
        widgets = [
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]
        output_modes = {"polygon": "Polygon", "rectangle": "Rectangle"}
        default_output_mode = "polygon"

    def __init__(self, config_path: Any, on_message: Any) -> None:
        super().__init__(config_path, on_message)
        session = get_default_registry().create_session("segment_anything", self.config)
        if not isinstance(session, SegmentAnythingSession):
            raise TypeError("Segment Anything backend returned an incompatible session")
        session.load()
        self.session = session
        self.marks: list[dict[str, Any]] = []
        self.preloaded_size = 7
        self.pre_inference_thread: threading.Thread | None = None
        self.stop_inference = False

    @staticmethod
    def detect_model_variant(decoder_path: str | Path) -> str:
        graph = onnx.load_model(str(decoder_path), load_external_data=False).graph
        return (
            "sam2"
            if any(item.name == "high_res_feats_0" for item in graph.input)
            else "sam"
        )

    def set_auto_labeling_marks(self, marks: list[dict[str, Any]]) -> None:
        self.marks = list(marks)

    @staticmethod
    def _contours(mask: np.ndarray) -> list[np.ndarray]:
        return mask_contours(mask)

    def post_process(self, mask: np.ndarray) -> list[Shape]:
        output_shape = ShapeType(self.output_mode)
        return self._legacy_shapes(mask_shapes(mask, output_shape))

    @staticmethod
    def _legacy_shapes(shapes: Any) -> list[Shape]:
        return [
            Shape(
                points=[Point(int(point.x), int(point.y)) for point in shape.points],
                shape_type=shape.type.value,
                label=shape.label or "AUTOLABEL_OBJECT",
            )
            for shape in shapes
        ]

    @staticmethod
    def _prompts(marks: list[dict[str, Any]]) -> tuple[PointPrompt | BoxPrompt, ...]:
        prompts: list[PointPrompt | BoxPrompt] = []
        for mark in marks:
            kind = mark.get("type")
            data = mark.get("data")
            if kind == "point" and isinstance(data, (list, tuple)) and len(data) == 2:
                prompts.append(
                    PointPrompt(
                        point=InferencePoint(x=data[0], y=data[1]),
                        foreground=bool(mark.get("label", 1)),
                    )
                )
            elif (
                kind == "rectangle"
                and isinstance(data, (list, tuple))
                and len(data) == 4
            ):
                x1, y1, x2, y2 = (float(value) for value in data)
                prompts.append(
                    BoxPrompt(
                        top_left=InferencePoint(x=min(x1, x2), y=min(y1, y2)),
                        bottom_right=InferencePoint(x=max(x1, x2), y=max(y1, y2)),
                    )
                )
            else:
                raise ValueError(f"Unsupported prompt type: {kind!r}")
        return tuple(prompts)

    def predict_shapes(
        self,
        image: Any,
        filename: str | None = None,
        preload_paths: list[str] | None = None,
    ) -> AutoLabelingResult:
        if isinstance(image, str):
            filename = image
            image = self.load_image_from_filename(image)
        if image is None or not self.marks or self.stop_inference:
            return AutoLabelingResult([], replace=False)

        try:
            capabilities = self.session.capabilities
            request = InferenceRequest(
                request_id=str(uuid.uuid4()),
                source_id=image_source_id(image, filename),
                model_id=capabilities.model_id,
                model_revision=capabilities.model_revision,
                prompts=self._prompts(self.marks),
                output_shape=ShapeType(self.output_mode),
            )
            result = self.session.predict(request, image)
            return AutoLabelingResult(self._legacy_shapes(result.shapes), replace=False)
        except Exception:
            logger.exception("Segment Anything inference failed")
            return AutoLabelingResult([], replace=False)

    def unload(self) -> None:
        self.stop_inference = True
        if self.pre_inference_thread and self.pre_inference_thread.is_alive():
            self.pre_inference_thread.join(timeout=5)
        self.session.unload()

    def preload_worker(self, files: list[str]) -> None:
        for filename in files[: self.preloaded_size]:
            if self.stop_inference:
                return
            image = self.load_image_from_filename(filename)
            if image is not None:
                try:
                    self.session.preload(
                        image,
                        image_source_id(image, filename),
                    )
                except Exception:
                    logger.exception("Could not preload %s", filename)

    def on_next_files_changed(self, next_files: list[str]) -> None:
        if self.pre_inference_thread and self.pre_inference_thread.is_alive():
            return
        self.pre_inference_thread = threading.Thread(
            target=self.preload_worker,
            args=(list(next_files),),
            daemon=True,
        )
        self.pre_inference_thread.start()

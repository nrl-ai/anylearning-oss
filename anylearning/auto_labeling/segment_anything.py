"""Interactive Segment Anything model integration."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx

from .lru_cache import LRUCache
from .model import Model
from .sam2_onnx import SegmentAnything2ONNX
from .sam_onnx import SegmentAnythingONNX
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
        encoder_path = Path(self.get_model_abs_path(self.config, "encoder_model_path"))
        decoder_path = Path(self.get_model_abs_path(self.config, "decoder_model_path"))
        for role, path in (("encoder", encoder_path), ("decoder", decoder_path)):
            if not path.is_file():
                raise FileNotFoundError(f"Segment Anything {role} not found: {path}")

        adapter = (
            SegmentAnything2ONNX
            if self.detect_model_variant(decoder_path) == "sam2"
            else SegmentAnythingONNX
        )
        self.model = adapter(str(encoder_path), str(decoder_path))
        self.marks: list[dict[str, Any]] = []
        self.image_embedding_cache: LRUCache[Any, dict[str, Any]] = LRUCache(10)
        self.preloaded_size = 7
        self.pre_inference_thread: threading.Thread | None = None
        self.stop_inference = False

    @staticmethod
    def detect_model_variant(decoder_path: str | Path) -> str:
        graph = onnx.load(str(decoder_path)).graph
        return (
            "sam2"
            if any(item.name == "high_res_feats_0" for item in graph.input)
            else "sam"
        )

    def set_auto_labeling_marks(self, marks: list[dict[str, Any]]) -> None:
        self.marks = list(marks)

    @staticmethod
    def _contours(mask: np.ndarray) -> list[np.ndarray]:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
        return [
            contour for contour, area in zip(approximated, areas) if area > threshold
        ]

    def post_process(self, mask: np.ndarray) -> list[Shape]:
        contours = self._contours(mask)
        if self.output_mode == "polygon":
            shapes: list[Shape] = []
            for contour in contours:
                coordinates = contour.reshape(-1, 2).astype(int).tolist()
                coordinates.append(coordinates[0])
                shapes.append(
                    Shape(
                        points=[Point(x, y) for x, y in coordinates],
                        shape_type="polygon",
                    )
                )
            return shapes

        if self.output_mode == "rectangle" and contours:
            points = np.concatenate([item.reshape(-1, 2) for item in contours])
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            return [
                Shape(
                    points=[
                        Point(int(x_min), int(y_min)),
                        Point(int(x_max), int(y_max)),
                    ],
                    shape_type="rectangle",
                )
            ]
        return []

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

        cache_key = filename if filename is not None else id(image)
        try:
            embedding = self.image_embedding_cache.get(cache_key)
            if embedding is None:
                embedding = self.model.encode(np.asarray(image))
                self.image_embedding_cache.put(cache_key, embedding)
            if self.stop_inference:
                return AutoLabelingResult([], replace=False)
            masks = np.asarray(self.model.predict_masks(embedding, self.marks))
            mask = masks[0, 0] if masks.ndim == 4 else masks[0]
            return AutoLabelingResult(self.post_process(mask), replace=False)
        except Exception:
            logger.exception("Segment Anything inference failed")
            return AutoLabelingResult([], replace=False)

    def unload(self) -> None:
        self.stop_inference = True
        if self.pre_inference_thread and self.pre_inference_thread.is_alive():
            self.pre_inference_thread.join(timeout=5)
        self.image_embedding_cache.clear()

    def preload_worker(self, files: list[str]) -> None:
        for filename in files[: self.preloaded_size]:
            if self.stop_inference:
                return
            if filename in self.image_embedding_cache:
                continue
            image = self.load_image_from_filename(filename)
            if image is not None:
                try:
                    self.image_embedding_cache.put(
                        filename, self.model.encode(np.asarray(image))
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

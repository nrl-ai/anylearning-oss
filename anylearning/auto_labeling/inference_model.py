"""Desktop adapter for models registered in :mod:`anylearning.inference`.

This is deliberately thin: preprocessing, ONNX execution, postprocessing, and
resource bounds stay in the shared inference backends used by desktop and
server.  The adapter only translates canvas prompts and editable shapes.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from anylearning.inference import (
    BoxPrompt,
    InferenceRequest,
    ModelTask,
    PointPrompt,
    ShapeType,
    get_default_registry,
)
from anylearning.inference import (
    Point as InferencePoint,
)
from anylearning.inference.backends.sam import image_source_id

from .label_spaces import resolve_label_space
from .model import Model
from .segment_anything import Point, Shape
from .types import AutoLabelingResult

logger = logging.getLogger(__name__)


def inference_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the strict backend config from a trusted catalog entry."""
    raw = model_config.get("inference_config")
    if not isinstance(raw, Mapping):
        raise ValueError("Inference auto-labeling model requires inference_config")
    config = dict(raw)
    label_space = config.pop("label_space", None)
    if label_space is not None:
        if not isinstance(label_space, str):
            raise ValueError("Inference label_space must be a string")
        config["class_names"] = resolve_label_space(label_space)
    config["config_file"] = str(Path(str(model_config["config_file"])).resolve())
    return config


class InferenceModel(Model):
    """Run any compatible shared inference backend in the desktop workflow."""

    class Meta:
        required_config_names = (
            "type",
            "name",
            "display_name",
            "backend",
            "inference_config",
        )
        widgets = ["button_run"]
        output_modes = {"polygon": "Polygon", "rectangle": "Rectangle"}
        default_output_mode = "polygon"

    def __init__(self, config_path: Any, on_message: Any) -> None:
        super().__init__(config_path, on_message)
        backend = self.config.get("backend")
        if not isinstance(backend, str) or not backend:
            raise ValueError("Inference auto-labeling model requires a backend")
        self.backend = backend
        self.runtime_config = inference_config(self.config)
        self.session = get_default_registry().create_session(
            backend, self.runtime_config
        )
        self.session.load()
        self.marks: list[dict[str, Any]] = []
        self.preloaded_size = 7
        self.pre_inference_thread: threading.Thread | None = None
        self.stop_inference = False

    @property
    def promptable(self) -> bool:
        return ModelTask.PROMPTABLE_SEGMENTATION in self.session.capabilities.tasks

    def set_auto_labeling_marks(self, marks: list[dict[str, Any]]) -> None:
        # Prompt marks can remain on the canvas while the user switches from a
        # prompted segmenter to a one-click detector. They belong to the old
        # interaction mode and must not make an automatic model fail. The
        # router also omits them for automatic catalog entries; keeping this
        # guard at the session adapter makes custom clients safe as well.
        self.marks = list(marks) if self.promptable else []

    @staticmethod
    def _prompts(
        marks: Sequence[Mapping[str, Any]],
    ) -> tuple[PointPrompt | BoxPrompt, ...]:
        prompts: list[PointPrompt | BoxPrompt] = []
        for mark in marks:
            kind = mark.get("type")
            data = mark.get("data")
            if kind == "point" and isinstance(data, (list, tuple)) and len(data) == 2:
                label = mark.get("label", 1)
                if not isinstance(label, (bool, int)) or label not in (0, 1):
                    raise ValueError("Point prompt label must be 0 or 1")
                prompts.append(
                    PointPrompt(
                        point=InferencePoint(x=data[0], y=data[1]),
                        foreground=bool(label),
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

    def _class_ids(
        self, allowed_labels: Sequence[str] | None
    ) -> tuple[int, ...] | None:
        class_names = self.runtime_config.get("class_names")
        if class_names is None or allowed_labels is None:
            return None
        allowed = set(allowed_labels)
        class_ids = tuple(
            index
            for index, class_name in enumerate(class_names)
            if class_name is not None and class_name in allowed
        )
        if not class_ids:
            raise ValueError(
                "This model has no classes matching the project's labels. "
                "Add a supported label or choose a promptable model."
            )
        return class_ids

    def predict_shapes(
        self,
        image: Any,
        filename: str | None = None,
        preload_paths: list[str] | None = None,
        *,
        allowed_labels: Sequence[str] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> AutoLabelingResult:
        if isinstance(image, str):
            filename = image
            image = self.load_image_from_filename(image)
        if image is None or self.stop_inference:
            return AutoLabelingResult([], replace=False)
        prompts = self._prompts(self.marks)
        if self.promptable and not prompts:
            return AutoLabelingResult([], replace=False)
        if not self.promptable and prompts:
            raise ValueError(f"{self.config['display_name']} does not accept prompts")

        request_parameters = dict(parameters or {})
        class_ids = self._class_ids(allowed_labels)
        if class_ids is not None:
            request_parameters["class_ids"] = class_ids
        capabilities = self.session.capabilities
        output_shape = ShapeType(self.output_mode)
        if ModelTask.DETECTION in capabilities.tasks:
            output_shape = ShapeType.RECTANGLE
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            source_id=image_source_id(image, filename),
            model_id=capabilities.model_id,
            model_revision=capabilities.model_revision,
            prompts=prompts,
            output_shape=output_shape,
            parameters=request_parameters,
        )
        result = self.session.predict(request, image)
        shapes = [
            Shape(
                points=[
                    # Keep subpixel detector coordinates for the frontend; it
                    # rounds only when annotations are serialized.
                    Point(point.x, point.y)
                    for point in shape.points
                ],
                shape_type=shape.type.value,
                label=shape.label or "AUTOLABEL_OBJECT",
                score=shape.score,
                group_id=shape.group_id,
                attributes=dict(shape.attributes),
            )
            for shape in result.shapes
        ]
        return AutoLabelingResult(
            shapes,
            replace=False,
            protocol_version=result.protocol_version,
            request_id=result.request_id,
            source_id=result.source_id,
            model_id=result.model_id,
            model_revision=result.model_revision,
            warnings=result.warnings,
            timings_ms=dict(result.timings_ms),
        )

    def unload(self) -> None:
        self.stop_inference = True
        if self.pre_inference_thread and self.pre_inference_thread.is_alive():
            self.pre_inference_thread.join(timeout=5)
        self.session.unload()

    def preload_worker(self, files: list[str]) -> None:
        preload = getattr(self.session, "preload", None)
        if not callable(preload):
            return
        for filename in files[: self.preloaded_size]:
            if self.stop_inference:
                return
            image = self.load_image_from_filename(filename)
            if image is not None:
                try:
                    preload(image, image_source_id(image, filename))
                except Exception:
                    logger.exception("Could not preload %s", filename)

    def on_next_files_changed(self, next_files: list[str]) -> None:
        if not self.promptable:
            return
        if self.pre_inference_thread and self.pre_inference_thread.is_alive():
            return
        self.pre_inference_thread = threading.Thread(
            target=self.preload_worker,
            args=(list(next_files),),
            daemon=True,
        )
        self.pre_inference_thread.start()


__all__ = ["InferenceModel", "inference_config"]

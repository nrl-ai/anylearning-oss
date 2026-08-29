"""Base class and file-loading helpers for auto-labeling models."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml
from PIL import Image, UnidentifiedImageError

from anylearning.config import DATA_ROOT

from .types import AutoLabelingResult

logger = logging.getLogger(__name__)


class Model(ABC):
    """Interface implemented by each interactive auto-labeling model."""

    class Meta:
        required_config_names: tuple[str, ...] = ()
        widgets = ["button_run"]
        output_modes = {"rectangle": "Rectangle"}
        default_output_mode = "rectangle"

    def __init__(
        self,
        model_config: str | Path | Mapping[str, Any],
        on_message: Callable[[str], None] | None,
    ) -> None:
        self.on_message = on_message or (lambda _message: None)
        self.config = self._load_config(model_config)
        self.check_missing_config(self.Meta.required_config_names, self.config)
        self.output_mode = self.Meta.default_output_mode

    @staticmethod
    def _load_config(
        model_config: str | Path | Mapping[str, Any],
    ) -> dict[str, Any]:
        if isinstance(model_config, Mapping):
            return dict(model_config)
        if not isinstance(model_config, (str, Path)):
            raise TypeError(f"Unsupported config type: {type(model_config).__name__}")

        config_path = Path(model_config).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError(f"Config must contain a mapping: {config_path}")
        config.setdefault("config_file", str(config_path))
        return config

    def get_required_widgets(self) -> list[str]:
        return list(self.Meta.widgets)

    @staticmethod
    def get_model_abs_path(
        model_config: Mapping[str, Any], model_path_field_name: str
    ) -> str:
        config_file = Path(str(model_config["config_file"])).expanduser().resolve()
        configured_path = Path(str(model_config[model_path_field_name])).expanduser()
        local_path = (
            configured_path
            if configured_path.is_absolute()
            else config_file.parent / configured_path
        )
        if local_path.is_file():
            return str(local_path.resolve())
        model_root = (Path(DATA_ROOT) / "models" / str(model_config["name"])).resolve()
        relative_path = (
            Path(configured_path.name)
            if configured_path.is_absolute()
            else configured_path
        )
        fallback = (model_root / relative_path).resolve()
        try:
            fallback.relative_to(model_root)
        except ValueError as error:
            raise ValueError(
                f"Model path escapes its model directory: {configured_path}"
            ) from error
        return str(fallback)

    @staticmethod
    def check_missing_config(
        config_names: Iterable[str], config: Mapping[str, Any]
    ) -> None:
        missing = [name for name in config_names if name not in config]
        if missing:
            raise ValueError(f"Missing config field(s): {', '.join(missing)}")

    @abstractmethod
    def predict_shapes(
        self, image: Any, filename: str | None = None
    ) -> AutoLabelingResult:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError

    @staticmethod
    def load_image_from_filename(filename: str | Path) -> Image.Image | None:
        image_path = Path(filename)
        label_path = image_path.with_suffix(".json")
        try:
            image_data = image_path.read_bytes()
            if label_path.is_file():
                label_data = json.loads(label_path.read_text(encoding="utf-8"))
                embedded = label_data.get("imageData")
                if embedded:
                    image_data = base64.b64decode(embedded, validate=True)
            with Image.open(BytesIO(image_data)) as image:
                return image.copy()
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            binascii.Error,
            UnidentifiedImageError,
        ) as error:
            logger.warning("Could not load image %s: %s", image_path, error)
            return None

    def on_next_files_changed(self, next_files: list[str]) -> None:
        """Allow models to preload files. The default implementation is a no-op."""

    def set_output_mode(self, mode: str) -> None:
        if mode not in self.Meta.output_modes:
            raise ValueError(f"Unsupported output mode: {mode}")
        self.output_mode = mode

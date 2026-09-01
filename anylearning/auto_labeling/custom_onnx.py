"""Install user-selected YOLO-family ONNX models as desktop auto-labelers.

Only a data graph selected through the native file dialog is accepted. The
model is copied into AnyLearning's data root, parsed with the same bounded ONNX
validator as the shared inference runtime, hashed, and described by a strict
``yolo_onnx`` configuration. No exporter package or model-specific Python code
is imported.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import stat
import tempfile
from collections.abc import Mapping
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from anylearning.auto_labeling.label_spaces import resolve_label_space
from anylearning.config import DATA_ROOT
from anylearning.inference.backends.onnx_safety import validate_onnx_artifact
from anylearning.inference.backends.yolo_onnx import (
    YoloFormat,
    YoloOnnxBackend,
    YoloOnnxConfig,
)

MAX_CUSTOM_ONNX_BYTES = 20 * 1024**3
COPY_CHUNK_BYTES = 8 * 1024 * 1024


class CustomYoloOptions(BaseModel):
    """Small, UI-facing subset of the strict shared backend configuration."""

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=256)
    format: YoloFormat = "auto"
    task: Literal["detection", "instance_segmentation"] = "detection"
    label_space: Literal["coco80"] | None = None
    class_names: tuple[str, ...] = Field(default=(), max_length=10_000)
    input_size: tuple[int, int] | None = None

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(character) < 32 for character in value):
            raise ValueError("Model name contains unsupported characters")
        return value

    @field_validator("class_names")
    @classmethod
    def validate_class_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(name.strip() for name in value)
        if any(not name or len(name) > 1_024 for name in normalized):
            raise ValueError("Class names must contain 1 to 1024 characters")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Class names must be unique")
        return normalized

    @field_validator("input_size", mode="before")
    @classmethod
    def normalize_input_size(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        if isinstance(value, int):
            return (value, value)
        return value

    @model_validator(mode="after")
    def require_one_label_source(self):
        if bool(self.label_space) == bool(self.class_names):
            raise ValueError("Choose either a label preset or an explicit class list")
        if self.format == "yolox" and self.task != "detection":
            raise ValueError("YOLOX supports detection only")
        return self


def _copy_and_hash(source: pathlib.Path, destination: pathlib.Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("The selected ONNX model is not a regular file")
        if not 0 < opened.st_size <= MAX_CUSTOM_ONNX_BYTES:
            raise ValueError("The selected ONNX model is empty or exceeds 20 GiB")
        digest = hashlib.sha256()
        consumed = 0
        with (
            os.fdopen(descriptor, "rb", closefd=False) as input_stream,
            destination.open("xb") as output_stream,
        ):
            while chunk := input_stream.read(COPY_CHUNK_BYTES):
                consumed += len(chunk)
                if consumed > opened.st_size or consumed > MAX_CUSTOM_ONNX_BYTES:
                    raise ValueError("The selected ONNX model changed while importing")
                output_stream.write(chunk)
                digest.update(chunk)
        current = os.fstat(descriptor)
        if (
            consumed != opened.st_size
            or current.st_size != opened.st_size
            or current.st_mtime_ns != opened.st_mtime_ns
        ):
            raise ValueError("The selected ONNX model changed while importing")
        return consumed, digest.hexdigest()
    finally:
        os.close(descriptor)


def _write_yaml_atomic(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    """Replace a config only after its complete bytes reach a sibling file."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(value, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_plain_directory(path: pathlib.Path) -> None:
    """Reject a pre-existing install path that can redirect writes elsewhere."""
    metadata = path.lstat()
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    is_junction = getattr(path, "is_junction", lambda: False)()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or is_junction
        or bool(file_attributes & reparse_flag)
    ):
        raise ValueError("A conflicting custom model installation exists")


def install_custom_yolo_onnx(
    source_path: str | os.PathLike[str],
    options: Mapping[str, Any],
    *,
    models_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Copy, validate and describe one user-selected single-file ONNX graph."""
    parsed = CustomYoloOptions.model_validate(options)
    source = pathlib.Path(source_path).expanduser()
    if source.suffix.lower() != ".onnx":
        raise ValueError("Choose a file with the .onnx extension")

    root = pathlib.Path(models_root or pathlib.Path(DATA_ROOT) / "models" / "custom")
    root.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = pathlib.Path(
        tempfile.mkdtemp(prefix=".import-", dir=root)
    )
    try:
        assert temporary is not None
        staged_model = temporary / "model.onnx"
        size, digest = _copy_and_hash(source, staged_model)
        # Parsing is data-only and does not ask ONNX Runtime to execute the
        # graph. External tensor references are deliberately rejected here;
        # a future bundle importer must copy and hash every referenced file.
        validate_onnx_artifact(
            staged_model,
            max_bytes=MAX_CUSTOM_ONNX_BYTES,
            allow_external_data=False,
        )

        class_names = (
            resolve_label_space(parsed.label_space)
            if parsed.label_space
            else parsed.class_names
        )
        identity_payload = yaml.safe_dump(
            {
                "sha256": digest,
                "format": parsed.format,
                "task": parsed.task,
                "class_names": list(class_names),
                "input_size": parsed.input_size,
            },
            sort_keys=True,
        ).encode()
        identity = hashlib.sha256(identity_payload).hexdigest()[:20]
        name = f"custom-yolo-{identity}"
        final_directory = root / name
        runtime_config: dict[str, Any] = {
            "name": name,
            "model_path": "model.onnx",
            "model_revision": f"user-sha256:{digest}",
            "sha256": digest,
            "task": parsed.task,
            "format": parsed.format,
            "class_names": list(class_names),
            "max_model_bytes": MAX_CUSTOM_ONNX_BYTES,
            "providers": ["CPUExecutionProvider"],
            "intra_op_threads": 1,
            "inter_op_threads": 1,
        }
        if parsed.input_size is not None:
            runtime_config["input_size"] = list(parsed.input_size)
        # Re-run the backend's exact schema before anything is persisted.
        YoloOnnxConfig.model_validate(
            {**runtime_config, "config_file": temporary / "config.yaml"}
        )
        YoloOnnxBackend().capabilities(
            {**runtime_config, "config_file": temporary / "config.yaml"}
        )

        task_name = parsed.task
        config = {
            "name": name,
            "display_name": parsed.display_name,
            "type": "inference",
            "backend": "yolo_onnx",
            "tasks": [task_name],
            "interaction_mode": "automatic",
            "output_modes": (
                ["polygon", "rectangle"]
                if task_name == "instance_segmentation"
                else ["rectangle"]
            ),
            "project_types": (
                ["Object Detection", "Image Segmentation", "Instance Segmentation"]
                if task_name == "instance_segmentation"
                else ["Object Detection"]
            ),
            "archive_size_bytes": size,
            "has_downloaded": True,
            "is_custom_model": True,
            "inference_config": runtime_config,
        }
        _write_yaml_atomic(temporary / "config.yaml", config)

        if final_directory.exists():
            # Identical graph and tensor contract: keep the already verified
            # model bytes, but allow the user-facing name to be refreshed.
            _require_plain_directory(final_directory)
            shutil.rmtree(temporary)
            temporary = None
            config_path = final_directory / "config.yaml"
            existing = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict) or existing.get("name") != name:
                raise ValueError("A conflicting custom model installation exists")
            existing["display_name"] = parsed.display_name
            _write_yaml_atomic(config_path, existing)
            config = existing
        else:
            temporary.replace(final_directory)
            temporary = None

        config["config_file"] = str((final_directory / "config.yaml").resolve())
        return config
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = ["CustomYoloOptions", "install_custom_yolo_onnx"]

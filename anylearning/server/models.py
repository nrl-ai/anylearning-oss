"""Strict, local-only model configuration for the public inference service."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_MAX_MANIFEST_BYTES = 1_048_576
_MAX_MODELS = 64
_BACKEND_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SENSITIVE_KEY = re.compile(r"(?:password|passwd|secret|token|api.?key)", re.I)

# Server-side inference is deliberately ONNX-only. Additions to this set require
# their own model-format, resource-bound, and license review.
_SERVER_BACKENDS = frozenset(
    {
        "efficient_sam",
        "efficientvit_sam",
        "sam3",
        "segment_anything",
        "yolo_onnx",
    }
)


class ServerModelDefinition(BaseModel):
    """One preconfigured model; clients cannot alter backend configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str = Field(min_length=1, max_length=128)
    config: dict[str, Any] = Field(max_length=256)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        if not _BACKEND_ID.fullmatch(value):
            raise ValueError("model backend identifier is invalid")
        if value not in _SERVER_BACKENDS:
            raise ValueError("model backend is not approved for public ONNX serving")
        return value

    @field_validator("config")
    @classmethod
    def reject_sensitive_configuration(cls, value: dict[str, Any]) -> dict[str, Any]:
        _check_configuration(value)
        return value


class ServerModelManifest(BaseModel):
    """Versioned server model manifest loaded only at process startup."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = 1
    models: tuple[ServerModelDefinition, ...] = Field(
        min_length=1,
        max_length=_MAX_MODELS,
    )

    @model_validator(mode="after")
    def validate_version(self) -> Self:
        if self.version != 1:
            raise ValueError("unsupported server model manifest version")
        return self


def load_server_model_manifest(path: Path) -> tuple[ServerModelDefinition, ...]:
    """Load a bounded regular JSON file and bind relative paths to that file."""
    configured = Path(path).expanduser()
    file_stat = os.lstat(configured)
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("server model manifest must be a regular non-link file")
    resolved = configured.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(configured, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened_stat.st_mode) or (
            opened_stat.st_dev,
            opened_stat.st_ino,
        ) != (file_stat.st_dev, file_stat.st_ino):
            raise ValueError("server model manifest changed before it was opened")
        if opened_stat.st_size <= 0 or opened_stat.st_size > _MAX_MANIFEST_BYTES:
            raise ValueError(
                f"server model manifest must contain 1 to {_MAX_MANIFEST_BYTES} bytes"
            )
        payload = stream.read(_MAX_MANIFEST_BYTES + 1)
        final_stat = os.fstat(stream.fileno())
    if len(payload) != opened_stat.st_size or _stat_identity(
        final_stat
    ) != _stat_identity(opened_stat):
        raise ValueError("server model manifest changed while it was being read")
    try:
        manifest = ServerModelManifest.model_validate_json(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("server model manifest is not valid JSON") from error

    definitions: list[ServerModelDefinition] = []
    for definition in manifest.models:
        config = dict(definition.config)
        # Every backend resolves relative artifact paths against this trusted,
        # immutable startup input rather than the process working directory.
        config["config_file"] = str(resolved)
        definitions.append(
            ServerModelDefinition(backend=definition.backend, config=config)
        )
    return tuple(definitions)


def _check_configuration(value: Any, *, depth: int = 0) -> None:
    if depth > 8:
        raise ValueError("model configuration nesting exceeds the server limit")
    if isinstance(value, dict):
        if len(value) > 256:
            raise ValueError("model configuration mapping is too large")
        for key, child in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise ValueError("model configuration keys are invalid")
            if _SENSITIVE_KEY.search(key):
                raise ValueError("model configuration must not contain credentials")
            _check_configuration(child, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 10_000:
            raise ValueError("model configuration list is too large")
        for child in value:
            _check_configuration(child, depth=depth + 1)
        return
    if value is not None and not isinstance(value, (str, Path, int, float, bool)):
        raise ValueError("model configuration contains an unsupported value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("model configuration numbers must be finite")
    if isinstance(value, (str, Path)) and len(str(value)) > 8_192:
        raise ValueError("model configuration string is too large")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


__all__ = [
    "ServerModelDefinition",
    "ServerModelManifest",
    "load_server_model_manifest",
]

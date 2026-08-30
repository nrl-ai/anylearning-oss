"""Shared validation and provider selection for untrusted ONNX artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import onnx
from google.protobuf.message import Message

_PROVIDER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}ExecutionProvider$")
_MAX_PROTO_MESSAGES = 1_000_000
_MAX_PROTO_DEPTH = 128
_MAX_GRAPH_INPUTS = 128
_MAX_GRAPH_OUTPUTS = 128
_MAX_GRAPH_NODES = 100_000


class OnnxArtifactError(ValueError):
    """Raised before ONNX Runtime is allowed to load an unsafe artifact."""


def resolve_model_path(
    value: str | Path,
    *,
    config_file: str | Path | None = None,
) -> Path:
    """Resolve a configured artifact path without silently changing its root."""
    configured = Path(value).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    if config_file is not None:
        return (Path(config_file).expanduser().resolve().parent / configured).resolve()
    return configured.resolve()


def local_artifact_revision(
    path: Path,
    *,
    explicit_revision: str | None = None,
    sha256: str | None = None,
) -> str:
    """Return a configured digest or a cheap identity for a local file.

    Downloaded artifacts should supply their verified SHA-256. Hashing a
    user-selected multi-gigabyte file merely to populate discovery would make
    startup unusable, so the fallback binds its path and portable stat fields.
    """
    if explicit_revision:
        return explicit_revision
    if sha256:
        return f"sha256:{sha256.lower()}"
    try:
        stat = path.stat()
        identity = (
            f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\0{stat.st_ctime_ns}"
        ).encode()
    except OSError:
        identity = f"{path}\0missing".encode()
    return f"local-stat-sha256:{hashlib.sha256(identity).hexdigest()}"


def validate_onnx_artifact(path: str | Path | BinaryIO, *, max_bytes: int) -> Any:
    """Inspect an ONNX graph without resolving external tensor references."""
    display_path = Path(path) if isinstance(path, (str, Path)) else None
    if display_path is not None and not display_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {display_path}")
    if display_path is not None:
        size = display_path.stat().st_size
    else:
        start = path.tell()
        path.seek(0, os.SEEK_END)
        size = path.tell()
        path.seek(start)
    if size <= 0:
        raise OnnxArtifactError("ONNX model is empty")
    if size > max_bytes:
        raise OnnxArtifactError(
            f"ONNX model is {size} bytes; configured limit is {max_bytes} bytes"
        )
    try:
        model = onnx.load_model(path, load_external_data=False)
    except Exception as error:
        raise OnnxArtifactError("ONNX graph could not be parsed") from error

    external_count = 0
    for tensor in _iter_tensor_protos(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            external_count += 1
    if external_count:
        raise OnnxArtifactError(
            "External-data ONNX models are not accepted; package tensors in one "
            f"integrity-checkable file ({external_count} external tensor reference(s))"
        )
    if not model.graph.input:
        raise OnnxArtifactError("ONNX graph has no inputs")
    if not model.graph.output:
        raise OnnxArtifactError("ONNX graph has no outputs")
    if len(model.graph.input) > _MAX_GRAPH_INPUTS:
        raise OnnxArtifactError(
            f"ONNX graph exceeds the {_MAX_GRAPH_INPUTS}-input limit"
        )
    if len(model.graph.output) > _MAX_GRAPH_OUTPUTS:
        raise OnnxArtifactError(
            f"ONNX graph exceeds the {_MAX_GRAPH_OUTPUTS}-output limit"
        )
    if len(model.graph.node) > _MAX_GRAPH_NODES:
        raise OnnxArtifactError(f"ONNX graph exceeds the {_MAX_GRAPH_NODES}-node limit")
    return model


def _iter_tensor_protos(
    message: Message,
    *,
    max_messages: int = _MAX_PROTO_MESSAGES,
    max_depth: int = _MAX_PROTO_DEPTH,
) -> Iterable[onnx.TensorProto]:
    """Walk initializers, tensor attributes, and nested subgraphs."""
    pending: list[tuple[Message, int]] = [(message, 0)]
    visited = 0
    while pending:
        current, depth = pending.pop()
        visited += 1
        if visited > max_messages:
            raise OnnxArtifactError(
                f"ONNX graph exceeds the protobuf message limit of {max_messages}"
            )
        if depth > max_depth:
            raise OnnxArtifactError(
                f"ONNX graph exceeds the nesting-depth limit of {max_depth}"
            )
        if isinstance(current, onnx.TensorProto):
            yield current
            continue
        children: list[Message] = []
        for field, value in current.ListFields():
            if field.type != field.TYPE_MESSAGE:
                continue
            if field.is_repeated:
                children.extend(value)
            else:
                children.append(value)
        pending.extend((child, depth + 1) for child in reversed(children))


def _hash_stream(stream: BinaryIO, *, cancellation: Any = None) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    while block := stream.read(8 * 1024 * 1024):
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


@contextmanager
def stable_onnx_artifact(
    path: Path,
    *,
    max_bytes: int,
    expected_sha256: str | None = None,
    cancellation: Any = None,
) -> Iterator[tuple[BinaryIO, str, str | None]]:
    """Yield one stable artifact for hashing, parsing, and runtime loading.

    POSIX runtimes load through the already-open descriptor, so replacement of
    the configured path cannot change the graph between verification and load.
    Platforms without a descriptor path receive a private temporary snapshot.
    The temporary snapshot is removed as soon as the runtime has constructed its
    own session.
    """
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise OnnxArtifactError(
            f"ONNX model size {size} is outside the configured limit of "
            f"{max_bytes} bytes"
        )

    temporary_path: Path | None = None
    with path.open("rb") as source:
        initial_identity = _stream_identity(source)
        if initial_identity[2] <= 0 or initial_identity[2] > max_bytes:
            raise OnnxArtifactError(
                f"Opened ONNX model size {initial_identity[2]} is outside the "
                f"configured limit of {max_bytes} bytes"
            )
        runtime_path = _descriptor_path(source)
        stream: BinaryIO = source
        snapshot: BinaryIO | None = None
        digest: str | None = None
        try:
            if runtime_path is None:
                handle, temporary_name = tempfile.mkstemp(
                    prefix="anylearning-onnx-", suffix=".onnx"
                )
                os.close(handle)
                temporary_path = Path(temporary_name)
                hasher = hashlib.sha256() if expected_sha256 else None
                with temporary_path.open("wb") as destination:
                    source.seek(0)
                    while block := source.read(8 * 1024 * 1024):
                        if cancellation is not None:
                            cancellation.raise_if_cancelled()
                        destination.write(block)
                        if hasher is not None:
                            hasher.update(block)
                snapshot = temporary_path.open("rb")
                stream = snapshot
                runtime_path = str(temporary_path)
                digest = hasher.hexdigest() if hasher is not None else None

            if expected_sha256 and digest is None:
                digest = _hash_stream(stream, cancellation=cancellation)
            if _stream_identity(source) != initial_identity:
                raise OnnxArtifactError("ONNX model changed while it was being loaded")
            if expected_sha256 and (
                digest is None or digest.lower() != expected_sha256.lower()
            ):
                raise OnnxArtifactError(
                    "ONNX model SHA-256 does not match configuration"
                )
            yield stream, runtime_path, digest
            if runtime_path == _descriptor_path(source):
                if _stream_identity(source) != initial_identity:
                    raise OnnxArtifactError(
                        "ONNX model changed while it was being loaded"
                    )
        finally:
            if snapshot is not None:
                snapshot.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _descriptor_path(stream: BinaryIO) -> str | None:
    descriptor = stream.fileno()
    for root in (Path("/proc/self/fd"), Path("/dev/fd")):
        candidate = root / str(descriptor)
        if root.is_dir() and candidate.exists():
            return str(candidate)
    return None


def _stream_identity(stream: BinaryIO) -> tuple[int, int, int, int]:
    stat = os.fstat(stream.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def select_providers(
    requested: Iterable[str],
    available: Iterable[str],
    *,
    allow_cpu_fallback: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select available providers in caller order and report safe fallbacks."""
    requested_names = tuple(requested)
    available_names = tuple(available)
    if not requested_names:
        raise ValueError("At least one ONNX Runtime provider must be requested")
    invalid = [name for name in requested_names if not _PROVIDER_NAME.fullmatch(name)]
    if invalid:
        raise ValueError(f"Invalid ONNX Runtime provider name: {invalid[0]!r}")

    selected = tuple(
        dict.fromkeys(name for name in requested_names if name in available_names)
    )
    warnings: list[str] = []
    unavailable = tuple(name for name in requested_names if name not in available_names)
    if unavailable:
        warnings.append(
            "Unavailable ONNX Runtime providers were skipped: " + ", ".join(unavailable)
        )
    if (
        not selected
        and allow_cpu_fallback
        and "CPUExecutionProvider" in available_names
    ):
        selected = ("CPUExecutionProvider",)
        warnings.append("Fell back to CPUExecutionProvider")
    elif (
        allow_cpu_fallback
        and "CPUExecutionProvider" in available_names
        and "CPUExecutionProvider" not in selected
    ):
        selected = (*selected, "CPUExecutionProvider")
    if not selected:
        raise RuntimeError(
            "None of the requested ONNX Runtime providers is available; available: "
            + ", ".join(available_names)
        )
    return selected, tuple(warnings)

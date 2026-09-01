"""Shared validation and provider selection for untrusted ONNX artifacts."""

from __future__ import annotations

import hashlib
import mmap
import os
import re
import stat
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Iterator

import onnx
from google.protobuf.message import Message

_PROVIDER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}ExecutionProvider$")
_MAX_PROTO_MESSAGES = 1_000_000
_MAX_PROTO_DEPTH = 128
_MAX_GRAPH_INPUTS = 128
_MAX_GRAPH_OUTPUTS = 128
_MAX_GRAPH_NODES = 100_000
_MAX_EXTERNAL_FILES = 1_024
_MAX_HYDRATED_EXTERNAL_BYTES = 64 * 1024 * 1024
_MAX_HYDRATED_TENSOR_BYTES = 64 * 1024
_EXTERNAL_DATA_KEYS = frozenset({"location", "offset", "length", "checksum"})
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_SHA1 = re.compile(r"^[0-9a-fA-F]{40}$")
_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)$")


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


def local_onnx_bundle_revision(
    path: Path,
    *,
    explicit_revision: str | None = None,
    sha256: str | None = None,
    external_data_sha256: Mapping[str, str] | None = None,
) -> str:
    """Bind cache identity to the graph and every configured external file."""
    if explicit_revision:
        return explicit_revision
    graph_revision = local_artifact_revision(path, sha256=sha256)
    if not external_data_sha256:
        return graph_revision
    digest = hashlib.sha256()
    digest.update(graph_revision.encode("utf-8"))
    for location, file_sha256 in sorted(external_data_sha256.items()):
        digest.update(b"\0")
        digest.update(location.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.lower().encode("ascii"))
    return f"onnx-bundle-sha256:{digest.hexdigest()}"


def validate_onnx_artifact(
    path: str | Path | BinaryIO,
    *,
    max_bytes: int,
    allow_external_data: bool = False,
) -> Any:
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
    if external_count and not allow_external_data:
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


@dataclass(frozen=True)
class ExternalDataFiles:
    """Verified buffers keyed by the relative locations stored in an ONNX graph."""

    locations: tuple[str, ...]
    buffers: tuple[mmap.mmap, ...]
    lengths: tuple[int, ...]
    total_bytes: int

    def add_to_session_options(
        self, options: Any, model: Any | None = None
    ) -> bytes | None:
        """Register mapped files and return a shape-inference-safe graph.

        ONNX Runtime resolves some small constant inputs while it constructs the
        graph, before its file-buffer overrides are consulted. Hydrate only
        bounded small tensors into the graph so shape inference works without
        copying multi-gigabyte weights into Python memory.
        """
        add_files = getattr(
            options, "add_external_initializers_from_files_in_memory", None
        )
        if add_files is None:
            raise RuntimeError(
                "External-data ONNX models require ONNX Runtime 1.29 or newer"
            )
        add_files(list(self.locations), list(self.buffers), list(self.lengths))
        if model is None:
            return None

        by_location = dict(zip(self.locations, self.buffers))
        hydrated_bytes = 0
        for tensor in _iter_tensor_protos(model):
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            reference = _external_reference(tensor)
            expected_length = _tensor_raw_data_size(tensor)
            length = expected_length if reference.length is None else reference.length
            if length != expected_length:
                raise OnnxArtifactError(
                    "ONNX external-data length does not match tensor dimensions"
                )
            if length > _MAX_HYDRATED_TENSOR_BYTES:
                continue
            if hydrated_bytes + length > _MAX_HYDRATED_EXTERNAL_BYTES:
                raise OnnxArtifactError(
                    "ONNX graph requires more than the bounded small-tensor "
                    "hydration budget"
                )
            buffer = by_location[reference.location]
            tensor.raw_data = buffer[reference.offset : reference.offset + length]
            tensor.ClearField("external_data")
            tensor.data_location = onnx.TensorProto.DEFAULT
            hydrated_bytes += length
        return model.SerializeToString()


@dataclass(frozen=True)
class _ExternalReference:
    location: str
    offset: int
    length: int | None
    checksum: str | None


def _external_references(model: Any) -> tuple[_ExternalReference, ...]:
    references: list[_ExternalReference] = []
    for tensor in _iter_tensor_protos(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        references.append(_external_reference(tensor))
    return tuple(references)


def _external_reference(tensor: Any) -> _ExternalReference:
    """Parse one TensorProto external_data record without resolving its path."""
    entries: dict[str, str] = {}
    for entry in tensor.external_data:
        if entry.key not in _EXTERNAL_DATA_KEYS:
            raise OnnxArtifactError(
                "ONNX external_data contains an unsupported metadata key"
            )
        if entry.key in entries:
            raise OnnxArtifactError(
                "ONNX external_data contains a duplicate metadata key"
            )
        entries[entry.key] = entry.value
    location = _normalize_external_location(entries.get("location", ""))
    offset = _parse_external_integer(entries.get("offset", "0"), name="offset")
    length = (
        _parse_external_integer(entries["length"], name="length")
        if "length" in entries
        else None
    )
    checksum = entries.get("checksum")
    if checksum is not None and not _SHA1.fullmatch(checksum):
        raise OnnxArtifactError("ONNX external_data checksum must be SHA-1")
    return _ExternalReference(
        location=location,
        offset=offset,
        length=length,
        checksum=checksum.lower() if checksum else None,
    )


def _tensor_raw_data_size(tensor: Any) -> int:
    """Return the packed ONNX raw-data size for a fixed-width tensor."""
    element_count = 1
    for dimension in tensor.dims:
        if dimension < 0:
            raise OnnxArtifactError("ONNX external tensor has a negative dimension")
        element_count *= dimension
        if element_count > 2**63 - 1:
            raise OnnxArtifactError("ONNX external tensor dimensions exceed the limit")
    byte_widths = {
        onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.INT8: 1,
        onnx.TensorProto.UINT16: 2,
        onnx.TensorProto.INT16: 2,
        onnx.TensorProto.INT32: 4,
        onnx.TensorProto.INT64: 8,
        onnx.TensorProto.BOOL: 1,
        onnx.TensorProto.FLOAT16: 2,
        onnx.TensorProto.DOUBLE: 8,
        onnx.TensorProto.UINT32: 4,
        onnx.TensorProto.UINT64: 8,
        onnx.TensorProto.COMPLEX64: 8,
        onnx.TensorProto.COMPLEX128: 16,
        onnx.TensorProto.BFLOAT16: 2,
    }
    for name in (
        "FLOAT8E4M3FN",
        "FLOAT8E4M3FNUZ",
        "FLOAT8E5M2",
        "FLOAT8E5M2FNUZ",
    ):
        value = getattr(onnx.TensorProto, name, None)
        if value is not None:
            byte_widths[value] = 1
    packed_types = {
        value
        for name in ("UINT4", "INT4", "FLOAT4E2M1")
        if (value := getattr(onnx.TensorProto, name, None)) is not None
    }
    if tensor.data_type in packed_types:
        return (element_count + 1) // 2
    byte_width = byte_widths.get(tensor.data_type)
    if byte_width is None:
        raise OnnxArtifactError(
            "ONNX external tensor uses a variable-width or unsupported data type"
        )
    return element_count * byte_width


def _normalize_external_location(value: str) -> str:
    if not value or "\x00" in value or len(value.encode("utf-8")) > 4_096:
        raise OnnxArtifactError("ONNX external_data location is invalid")
    normalized = value.replace("\\", "/")
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(normalized)
    raw_parts = normalized.split("/")
    if (
        windows_path.drive
        or windows_path.is_absolute()
        or posix_path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise OnnxArtifactError(
            "ONNX external_data location must be a contained relative path"
        )
    return posix_path.as_posix()


def _parse_external_integer(value: str, *, name: str) -> int:
    if not _DECIMAL.fullmatch(value):
        raise OnnxArtifactError(
            f"ONNX external_data {name} must be a non-negative decimal integer"
        )
    parsed = int(value)
    if parsed > 2**63 - 1:
        raise OnnxArtifactError(f"ONNX external_data {name} exceeds the limit")
    return parsed


def _normalize_external_manifest(
    values: Mapping[str, str] | None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_location, digest in (values or {}).items():
        location = _normalize_external_location(raw_location)
        if location in normalized:
            raise OnnxArtifactError(
                "External-data SHA-256 manifest has duplicate normalized paths"
            )
        if not _SHA256.fullmatch(digest):
            raise OnnxArtifactError(
                "External-data SHA-256 manifest contains an invalid digest"
            )
        normalized[location] = digest.lower()
    return normalized


def _path_components_are_plain(
    root: Path, location: str
) -> tuple[Path, os.stat_result]:
    current = root
    metadata: os.stat_result | None = None
    for component in PurePosixPath(location).parts:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as error:
            raise OnnxArtifactError("ONNX external-data file is unavailable") from error
        file_attributes = getattr(metadata, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        is_junction = getattr(current, "is_junction", lambda: False)()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or is_junction
            or bool(file_attributes & reparse_flag)
        ):
            raise OnnxArtifactError(
                "ONNX external-data paths may not use links or reparse points"
            )
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise OnnxArtifactError(
            "ONNX external-data path escapes the model directory"
        ) from error
    if metadata is None:
        raise OnnxArtifactError("ONNX external-data location is empty")
    return resolved, metadata


def _open_external_file(path: Path, *, expected: os.stat_result) -> BinaryIO:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OnnxArtifactError(
            "ONNX external-data file could not be opened"
        ) from error
    try:
        opened = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode):
            raise OnnxArtifactError("ONNX external data must be a regular file")
        if opened.st_nlink != 1:
            raise OnnxArtifactError("ONNX external-data hardlinks are not accepted")
        if _stat_file_id(opened) != _stat_file_id(expected):
            raise OnnxArtifactError("ONNX external-data path changed before opening")
        if _stat_file_id(opened) != _stat_file_id(current):
            raise OnnxArtifactError("ONNX external-data path changed while opening")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _stat_file_id(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, value.st_nlink


def _file_identity(stream: BinaryIO) -> tuple[int, int, int, int, int, int]:
    value = os.fstat(stream.fileno())
    return (
        value.st_dev,
        value.st_ino,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_mapped_file(
    buffer: mmap.mmap,
    *,
    cancellation: Any = None,
    include_sha1: bool = False,
) -> tuple[str, str | None]:
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1() if include_sha1 else None  # noqa: S324 - ONNX field format
    view = memoryview(buffer)
    try:
        for start in range(0, len(view), 8 * 1024 * 1024):
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            block = view[start : start + 8 * 1024 * 1024]
            try:
                sha256.update(block)
                if sha1 is not None:
                    sha1.update(block)
            finally:
                block.release()
    finally:
        view.release()
    return sha256.hexdigest(), sha1.hexdigest() if sha1 is not None else None


@contextmanager
def stable_external_data_files(
    model: Any,
    *,
    model_path: Path,
    expected_sha256: Mapping[str, str] | None,
    max_bytes: int,
    max_files: int = _MAX_EXTERNAL_FILES,
    cancellation: Any = None,
) -> Iterator[ExternalDataFiles]:
    """Open, bound, hash, and map every external file referenced by a graph."""
    references = _external_references(model)
    locations = tuple(dict.fromkeys(item.location for item in references))
    if not locations:
        if _normalize_external_manifest(expected_sha256):
            raise OnnxArtifactError(
                "External-data SHA-256 manifest has entries not used by the graph"
            )
        yield ExternalDataFiles((), (), (), 0)
        return
    if len(locations) > max_files:
        raise OnnxArtifactError(
            f"ONNX graph references {len(locations)} external files; limit is {max_files}"
        )
    manifest = _normalize_external_manifest(expected_sha256)
    referenced = set(locations)
    if set(manifest) != referenced:
        missing = len(referenced - set(manifest))
        extra = len(set(manifest) - referenced)
        raise OnnxArtifactError(
            "External-data SHA-256 manifest must exactly cover graph locations "
            f"(missing={missing}, extra={extra})"
        )

    root = model_path.parent.resolve(strict=True)
    streams: list[BinaryIO] = []
    buffers: list[mmap.mmap] = []
    identities: list[tuple[int, int, int, int, int, int]] = []
    paths: list[Path] = []
    total_bytes = 0
    try:
        for location in locations:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            path, expected = _path_components_are_plain(root, location)
            stream = _open_external_file(path, expected=expected)
            streams.append(stream)
            identity = _file_identity(stream)
            size = identity[3]
            if size <= 0:
                raise OnnxArtifactError("ONNX external-data file is empty")
            total_bytes += size
            if total_bytes > max_bytes:
                raise OnnxArtifactError(
                    f"ONNX external data exceeds the configured {max_bytes}-byte limit"
                )
            buffer = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            buffers.append(buffer)
            identities.append(identity)
            paths.append(path)
            needs_sha1 = any(
                item.checksum is not None
                for item in references
                if item.location == location
            )
            digest, sha1 = _hash_mapped_file(
                buffer, cancellation=cancellation, include_sha1=needs_sha1
            )
            if digest != manifest[location]:
                raise OnnxArtifactError(
                    "ONNX external-data SHA-256 does not match configuration"
                )
            declared_sha1 = {
                item.checksum
                for item in references
                if item.location == location and item.checksum is not None
            }
            if len(declared_sha1) > 1 or (declared_sha1 and sha1 not in declared_sha1):
                raise OnnxArtifactError(
                    "ONNX external-data checksum does not match the referenced file"
                )

        sizes = dict(zip(locations, (identity[3] for identity in identities)))
        for reference in references:
            file_size = sizes[reference.location]
            end = (
                file_size
                if reference.length is None
                else reference.offset + reference.length
            )
            if reference.offset > file_size or end > file_size:
                raise OnnxArtifactError(
                    "ONNX external-data offset/length exceeds the referenced file"
                )

        yield ExternalDataFiles(
            locations=locations,
            buffers=tuple(buffers),
            lengths=tuple(identity[3] for identity in identities),
            total_bytes=total_bytes,
        )
        for stream, path, identity in zip(streams, paths, identities):
            if _file_identity(stream) != identity:
                raise OnnxArtifactError(
                    "ONNX external data changed while it was being loaded"
                )
            try:
                current = path.stat(follow_symlinks=False)
            except OSError as error:
                raise OnnxArtifactError(
                    "ONNX external-data path changed while loading"
                ) from error
            if _stat_file_id(current) != identity[:3]:
                raise OnnxArtifactError("ONNX external-data path changed while loading")
    finally:
        for buffer in reversed(buffers):
            buffer.close()
        for stream in reversed(streams):
            stream.close()


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
    # Opening /proc/self/fd/N creates an independently seekable file
    # description on Linux. BSD/macOS /dev/fd/N duplicates the current stream
    # offset instead; after graph validation ONNX Runtime then sees an empty
    # protobuf. Those platforms intentionally take the private-snapshot path.
    root = Path("/proc/self/fd")
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

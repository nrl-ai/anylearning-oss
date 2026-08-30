#!/usr/bin/env python3
"""Extract an exact, already integrity-verified, flat ZIP bundle."""

from __future__ import annotations

import argparse
import stat
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

_COPY_CHUNK_BYTES = 1024 * 1024


def _safe_flat_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in {".", ".."}
        or ":" in name
    ):
        raise ValueError(f"Unsafe ZIP member name: {name!r}")
    return name


def extract_exact_zip(
    archive_path: Path,
    destination: Path,
    expected_sizes: Mapping[str, int],
    *,
    remove_archive: bool = False,
) -> tuple[Path, ...]:
    """Extract only an exact set of regular files with exact uncompressed sizes.

    The caller must verify the archive's cryptographic digest before invoking this
    function. Existing destination files are never overwritten, and partially
    extracted files are removed if any member fails validation or decompression.
    """
    archive_path = archive_path.resolve(strict=True)
    destination = destination.resolve(strict=True)
    normalized = {_safe_flat_name(name): size for name, size in expected_sizes.items()}
    if not normalized:
        raise ValueError("At least one expected ZIP member is required")
    if len(normalized) != len(expected_sizes):
        raise ValueError("Expected ZIP member names must be unique")
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in normalized.values()
    ):
        raise ValueError("Expected ZIP member sizes must be positive integers")

    created: list[Path] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.infolist()
            by_name = {entry.filename: entry for entry in entries}
            if len(by_name) != len(entries):
                raise ValueError("ZIP archive contains duplicate member names")
            if set(by_name) != set(normalized):
                raise ValueError("ZIP archive members do not match the manifest")

            for name, expected_size in normalized.items():
                entry = by_name[name]
                file_type = stat.S_IFMT(entry.external_attr >> 16)
                if entry.is_dir() or file_type not in (0, stat.S_IFREG):
                    raise ValueError(f"ZIP member is not a regular file: {name!r}")
                if entry.flag_bits & 1:
                    raise ValueError(f"Encrypted ZIP member is not allowed: {name!r}")
                if entry.file_size != expected_size or entry.compress_size <= 0:
                    raise ValueError(f"ZIP member size does not match: {name!r}")

            for name, expected_size in normalized.items():
                output_path = destination / name
                written = 0
                with (
                    archive.open(by_name[name]) as source,
                    output_path.open("xb") as output,
                ):
                    created.append(output_path)
                    while chunk := source.read(_COPY_CHUNK_BYTES):
                        written += len(chunk)
                        if written > expected_size:
                            raise ValueError(
                                f"ZIP member expanded past its limit: {name!r}"
                            )
                        output.write(chunk)
                if written != expected_size:
                    raise ValueError(f"Extracted ZIP member size changed: {name!r}")
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise

    if remove_archive:
        archive_path.unlink()
    return tuple(created)


def _member(value: str) -> tuple[str, int]:
    name, separator, raw_size = value.rpartition(":")
    if not separator:
        raise argparse.ArgumentTypeError("member must use NAME:SIZE syntax")
    try:
        size = int(raw_size)
        return _safe_flat_name(name), size
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--member",
        action="append",
        type=_member,
        required=True,
        metavar="NAME:SIZE",
        help="exact flat archive member and uncompressed byte size",
    )
    parser.add_argument("--remove-archive", action="store_true")
    args = parser.parse_args(argv)
    members = dict(args.member)
    if len(members) != len(args.member):
        parser.error("--member names must be unique")
    extract_exact_zip(
        args.archive,
        args.destination,
        members,
        remove_archive=args.remove_archive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download one immutable HTTPS asset with a byte limit and SHA-256 gate."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse


def download_verified_file(
    url: str,
    output: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
    timeout_seconds: int = 600,
) -> None:
    if urlparse(url).scheme.lower() != "https":
        raise ValueError("Download URL must use HTTPS")
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in expected_sha256
    ):
        raise ValueError("Expected digest must be SHA-256")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if timeout_seconds <= 0 or timeout_seconds > 3_600:
        raise ValueError("timeout_seconds must be between 1 and 3600")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".part", dir=output.parent
    )
    total = 0
    digest = hashlib.sha256()
    try:
        os.close(descriptor)
        descriptor = -1
        subprocess.run(
            [
                "curl",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--retry-all-errors",
                "--retry-delay",
                "1",
                "--retry-max-time",
                str(timeout_seconds),
                "--connect-timeout",
                "30",
                "--max-time",
                str(timeout_seconds),
                "--max-filesize",
                str(max_bytes),
                "--user-agent",
                "AnyLearning-model-validation/1",
                "--output",
                temporary_name,
                url,
            ],
            check=True,
            timeout=timeout_seconds + 30,
        )
        with Path(temporary_name).open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Download exceeds the configured byte limit")
                digest.update(chunk)
        if total <= 0:
            raise ValueError("Downloaded file is empty")
        if digest.hexdigest() != expected_sha256.lower():
            raise ValueError("Downloaded file SHA-256 does not match")
        os.replace(temporary_name, output)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--max-bytes", required=True, type=int)
    parser.add_argument("--timeout-seconds", default=600, type=int)
    arguments = parser.parse_args()
    download_verified_file(
        arguments.url,
        arguments.output,
        expected_sha256=arguments.sha256,
        max_bytes=arguments.max_bytes,
        timeout_seconds=arguments.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

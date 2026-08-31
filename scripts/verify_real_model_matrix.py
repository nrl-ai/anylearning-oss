#!/usr/bin/env python3
"""Verify transport and cross-platform identity in retained real-model reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

_MAX_REPORT_BYTES = 8 * 1024 * 1024
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_IMAGE_PIXELS = 64_000_000


def _safe_file(path: Path, root: Path, *, max_bytes: int) -> Path:
    root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Matrix artifact escapes its root: {path}") from error
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"Matrix artifact must be a regular non-symlink file: {path}")
    size = resolved.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"Matrix artifact has an invalid size: {path}")
    return resolved


def _load_report(path: Path, root: Path) -> dict[str, Any]:
    path = _safe_file(path, root, max_bytes=_MAX_REPORT_BYTES)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"Matrix report must be a JSON object: {path}")
    return report


def _image_path(summary_path: Path, filename: object, root: Path) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("Annotated image name must be one relative filename")
    return _safe_file(summary_path.parent / filename, root, max_bytes=_MAX_IMAGE_BYTES)


def _pixel_digest(path: Path) -> str:
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS
    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            digest = hashlib.sha256()
            digest.update(f"{rgb.width}x{rgb.height}:RGB\n".encode())
            digest.update(rgb.tobytes())
            return digest.hexdigest()
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def _pixel_drift(
    reference: Path,
    candidate: Path,
    *,
    max_differing_pixels: int,
    max_channel_delta: int,
) -> tuple[int, int]:
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS
    try:
        with (
            Image.open(reference) as reference_image,
            Image.open(candidate) as candidate_image,
        ):
            reference_image.load()
            candidate_image.load()
            reference_rgb = reference_image.convert("RGB")
            candidate_rgb = candidate_image.convert("RGB")
            if reference_rgb.size != candidate_rgb.size:
                raise ValueError("Cross-platform annotated image dimensions changed")
            reference_bytes = reference_rgb.tobytes()
            candidate_bytes = candidate_rgb.tobytes()
            differing_pixels = 0
            maximum_channel_delta = 0
            for offset in range(0, len(reference_bytes), 3):
                deltas = tuple(
                    abs(
                        reference_bytes[offset + channel]
                        - candidate_bytes[offset + channel]
                    )
                    for channel in range(3)
                )
                if any(deltas):
                    differing_pixels += 1
                    maximum_channel_delta = max(maximum_channel_delta, *deltas)
                    if (
                        differing_pixels > max_differing_pixels
                        or maximum_channel_delta > max_channel_delta
                    ):
                        return differing_pixels, maximum_channel_delta
            return differing_pixels, maximum_channel_delta
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def _validated_images(
    report: dict[str, Any],
    summary_path: Path,
    root: Path,
    *,
    expected_cases: int,
    direct: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[Path, ...]]:
    images = report.get("images")
    if not isinstance(images, list) or len(images) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} image cases in matrix report: {summary_path}"
        )
    predictions: list[str] = []
    pixels: list[str] = []
    paths: list[Path] = []
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(f"Image result {index} is not an object: {summary_path}")
        if direct and image.get("passed") is not True:
            raise ValueError(
                f"Direct image result {index} did not pass: {summary_path}"
            )
        if image.get("consistent_runs") is not True or image.get("failures") != []:
            raise ValueError(
                f"Image result {index} is not repeatable and failure-free: {summary_path}"
            )
        prediction = image.get("prediction_digest")
        if not isinstance(prediction, str) or len(prediction) != 64:
            raise ValueError(
                f"Image result {index} has no SHA-256 digest: {summary_path}"
            )
        predictions.append(prediction)
        path = _image_path(summary_path, image.get("annotated_image"), root)
        pixels.append(_pixel_digest(path))
        paths.append(path)
    return tuple(predictions), tuple(pixels), tuple(paths)


def verify_matrix(
    root: Path,
    *,
    artifact_prefix: str,
    variants: tuple[str, ...],
    platforms: tuple[str, ...],
    expected_cases: int = 4,
    max_cross_platform_differing_pixels: int = 0,
    max_cross_platform_channel_delta: int = 0,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Matrix root must be a regular non-symlink directory")
    if not artifact_prefix or not variants or not platforms or expected_cases < 1:
        raise ValueError("Matrix expectations must not be empty")
    if not 0 <= max_cross_platform_differing_pixels <= _MAX_IMAGE_PIXELS:
        raise ValueError(
            "Cross-platform differing-pixel limit must be between 0 and "
            f"{_MAX_IMAGE_PIXELS}"
        )
    if not 0 <= max_cross_platform_channel_delta <= 255:
        raise ValueError("Cross-platform channel-delta limit must be between 0 and 255")

    verified: dict[str, Any] = {}
    for variant in variants:
        baseline_predictions: tuple[str, ...] | None = None
        baseline_pixels: tuple[str, ...] | None = None
        baseline_pixel_paths: tuple[Path, ...] | None = None
        platform_reports: dict[str, Any] = {}
        for platform in platforms:
            artifact = root / (
                f"{artifact_prefix}-{variant}-real-model-validation-{platform}"
            )
            if artifact.is_symlink() or not artifact.is_dir():
                raise ValueError(f"Missing matrix artifact directory: {artifact.name}")
            summaries = sorted(artifact.rglob("summary.json"))
            if len(summaries) != 2:
                raise ValueError(
                    f"Expected direct and server reports in {artifact.name}; "
                    f"found {len(summaries)}"
                )
            loaded = [(path, _load_report(path, artifact)) for path in summaries]
            direct_matches = [
                (path, report) for path, report in loaded if "backend" in report
            ]
            server_matches = [
                (path, report) for path, report in loaded if "manifest" in report
            ]
            if len(direct_matches) != 1 or len(server_matches) != 1:
                raise ValueError(
                    f"Could not identify direct/server reports in {artifact.name}"
                )
            direct_path, direct = direct_matches[0]
            server_path, server = server_matches[0]
            if direct.get("passed") is not True or direct.get("failures") != []:
                raise ValueError(f"Direct matrix report failed: {direct_path}")
            if server.get("passed") is not True or server.get("failures") != []:
                raise ValueError(f"Server matrix report failed: {server_path}")
            direct_predictions, direct_pixels, direct_pixel_paths = _validated_images(
                direct,
                direct_path,
                artifact,
                expected_cases=expected_cases,
                direct=True,
            )
            server_predictions, server_pixels, _server_pixel_paths = _validated_images(
                server,
                server_path,
                artifact,
                expected_cases=expected_cases,
                direct=False,
            )
            if direct_predictions != server_predictions:
                raise ValueError(
                    f"Transport prediction mismatch for {variant}/{platform}"
                )
            if direct_pixels != server_pixels:
                raise ValueError(f"Transport pixel mismatch for {variant}/{platform}")
            if baseline_predictions is None:
                baseline_predictions = direct_predictions
                baseline_pixels = direct_pixels
                baseline_pixel_paths = direct_pixel_paths
                pixel_drift = [
                    {"case": index, "differing_pixels": 0, "maximum_channel_delta": 0}
                    for index in range(expected_cases)
                ]
            else:
                if direct_predictions != baseline_predictions:
                    raise ValueError(
                        f"Cross-platform prediction mismatch for {variant}/{platform}"
                    )
                assert baseline_pixels is not None and baseline_pixel_paths is not None
                pixel_drift = []
                for index, (baseline_path, candidate_path) in enumerate(
                    zip(baseline_pixel_paths, direct_pixel_paths, strict=True)
                ):
                    if direct_pixels[index] == baseline_pixels[index]:
                        differing_pixels, maximum_channel_delta = 0, 0
                    else:
                        differing_pixels, maximum_channel_delta = _pixel_drift(
                            baseline_path,
                            candidate_path,
                            max_differing_pixels=max_cross_platform_differing_pixels,
                            max_channel_delta=max_cross_platform_channel_delta,
                        )
                    if (
                        differing_pixels > max_cross_platform_differing_pixels
                        or maximum_channel_delta > max_cross_platform_channel_delta
                    ):
                        raise ValueError(
                            f"Cross-platform pixel mismatch for {variant}/{platform} "
                            f"case {index}: {differing_pixels} pixels, maximum channel "
                            f"delta {maximum_channel_delta}"
                        )
                    pixel_drift.append(
                        {
                            "case": index,
                            "differing_pixels": differing_pixels,
                            "maximum_channel_delta": maximum_channel_delta,
                        }
                    )
            platform_reports[platform] = {
                "cross_platform_pixel_drift": pixel_drift,
                "direct_peak_rss_bytes": direct.get("peak_observed_rss_bytes"),
                "direct_steady_state_rss_growth_bytes": direct.get(
                    "steady_state_rss_growth_bytes"
                ),
                "server_peak_rss_bytes": server.get("peak_observed_rss_bytes"),
            }
        assert baseline_predictions is not None and baseline_pixels is not None
        verified[variant] = {
            "annotated_pixel_digests": list(baseline_pixels),
            "platforms": platform_reports,
            "prediction_digests": list(baseline_predictions),
        }
    return {
        "artifact_prefix": artifact_prefix,
        "expected_cases": expected_cases,
        "passed": True,
        "platforms": list(platforms),
        "schema_version": 1,
        "variants": verified,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    if path.is_symlink():
        raise ValueError("Matrix verification output may not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--variant", action="append", required=True, dest="variants")
    parser.add_argument("--platform", action="append", required=True, dest="platforms")
    parser.add_argument("--expected-cases", type=int, default=4)
    parser.add_argument("--max-cross-platform-differing-pixels", type=int, default=0)
    parser.add_argument("--max-cross-platform-channel-delta", type=int, default=0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = verify_matrix(
        arguments.root,
        artifact_prefix=arguments.artifact_prefix,
        variants=tuple(arguments.variants),
        platforms=tuple(arguments.platforms),
        expected_cases=arguments.expected_cases,
        max_cross_platform_differing_pixels=(
            arguments.max_cross_platform_differing_pixels
        ),
        max_cross_platform_channel_delta=arguments.max_cross_platform_channel_delta,
    )
    if arguments.output is not None:
        _write_report(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Matrix verification failed: {error}") from None

"""Reproducible real-model inference validation and visual reports."""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np
import psutil
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import InferenceRequest, InferenceResult, ShapeType
from .defaults import get_default_registry

_MAX_MANIFEST_BYTES = 1_048_576
_MAX_IMAGE_FILE_BYTES = 512 * 1024**2
_SENSITIVE_KEY = re.compile(r"(?:password|passwd|secret|token|api.?key)", re.I)


class ExpectedDetection(BaseModel):
    """One detection that must have a sufficiently overlapping match."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1, max_length=1024)
    box: tuple[float, float, float, float]
    minimum_iou: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    minimum_score: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_box(self) -> Self:
        x1, y1, x2, y2 = self.box
        if not all(np.isfinite(self.box)) or x2 <= x1 or y2 <= y1:
            raise ValueError("Expected boxes must contain finite ordered xyxy values")
        return self


class ImageExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_shapes: int = Field(default=1, ge=0, le=10_000)
    maximum_shapes: int = Field(default=10_000, ge=0, le=10_000)
    minimum_label_counts: dict[str, int] = Field(default_factory=dict, max_length=1_000)
    detections: tuple[ExpectedDetection, ...] = Field(default=(), max_length=1_000)

    @field_validator("minimum_label_counts")
    @classmethod
    def validate_label_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not label or len(label) > 1024 for label in value):
            raise ValueError("Expected labels must contain 1 to 1024 characters")
        if any(
            isinstance(count, bool) or count < 1 or count > 10_000
            for count in value.values()
        ):
            raise ValueError("Expected label counts must be between 1 and 10000")
        return value

    @model_validator(mode="after")
    def validate_shape_range(self) -> Self:
        if self.maximum_shapes < self.minimum_shapes:
            raise ValueError("maximum_shapes must be at least minimum_shapes")
        return self


class ValidationImage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    request_parameters: dict[str, Any] = Field(default_factory=dict, max_length=128)
    expected: ImageExpectations = Field(default_factory=ImageExpectations)


class ModelProvenance(BaseModel):
    """Human-auditable origin and license record for a real test artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_url: str = Field(min_length=8, max_length=2048, pattern=r"^https://")
    artifact_url: str = Field(min_length=8, max_length=2048, pattern=r"^https://")
    source_revision: str = Field(min_length=1, max_length=128)
    code_license: str = Field(min_length=1, max_length=128)
    artifact_license: str = Field(min_length=1, max_length=256)
    license_url: str = Field(min_length=8, max_length=2048, pattern=r"^https://")


class RealModelValidationManifest(BaseModel):
    """Bounded input manifest for an opt-in real model test."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    backend: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9_]+$")
    provenance: ModelProvenance
    config: dict[str, Any] = Field(max_length=256)
    runs: int = Field(default=3, ge=2, le=20)
    lifecycle_cycles: int = Field(default=3, ge=2, le=10)
    maximum_steady_state_rss_growth_bytes: int = Field(
        default=64 * 1024**2, ge=0, le=2 * 1024**3
    )
    images: tuple[ValidationImage, ...] = Field(min_length=1, max_length=100)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-")
    return cleaned[:100] or "validation"


def _redact(value: Any, *, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return "<redacted>"
    if key.endswith("_path") and isinstance(value, (str, Path)):
        return Path(value).name
    if isinstance(value, dict):
        return {
            str(item): _redact(child, key=str(item)) for item, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(encoded + "\n", encoding="utf-8")


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and len(revision) == 40 else None


def _load_rgb(path: Path) -> np.ndarray:
    size = path.stat().st_size
    if size <= 0 or size > _MAX_IMAGE_FILE_BYTES:
        raise ValueError(
            f"Image {path.name!r} has {size} bytes; limit is {_MAX_IMAGE_FILE_BYTES}"
        )
    encoded = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image {path.name!r}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _result_payload(result: InferenceResult, *, timings: bool) -> dict[str, Any]:
    excluded = set() if timings else {"timings_ms"}
    return result.model_dump(mode="json", exclude=excluded)


def _result_digest(result: InferenceResult) -> str:
    encoded = json.dumps(
        _result_payload(result, timings=False),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rectangle(shape: Any) -> tuple[float, float, float, float] | None:
    if shape.type is not ShapeType.RECTANGLE:
        return None
    return (
        shape.points[0].x,
        shape.points[0].y,
        shape.points[1].x,
        shape.points[1].y,
    )


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _check_expectations(
    result: InferenceResult, expected: ImageExpectations
) -> list[str]:
    failures: list[str] = []
    if not expected.minimum_shapes <= len(result.shapes) <= expected.maximum_shapes:
        failures.append(
            f"shape count {len(result.shapes)} is outside "
            f"[{expected.minimum_shapes}, {expected.maximum_shapes}]"
        )
    counts = Counter(shape.label for shape in result.shapes if shape.label is not None)
    for label, minimum in expected.minimum_label_counts.items():
        if counts[label] < minimum:
            failures.append(
                f"label {label!r} count {counts[label]} is below required {minimum}"
            )
    for item in expected.detections:
        candidates = [
            (_box_iou(box, item.box), shape)
            for shape in result.shapes
            if shape.label == item.label
            and shape.score is not None
            and shape.score >= item.minimum_score
            and (box := _rectangle(shape)) is not None
        ]
        best = max((candidate[0] for candidate in candidates), default=0.0)
        if best < item.minimum_iou:
            failures.append(
                f"expected {item.label!r} box best IoU {best:.4f} is below "
                f"{item.minimum_iou:.4f}"
            )
    return failures


def _color(label: str | None) -> tuple[int, int, int]:
    digest = hashlib.sha256((label or "shape").encode()).digest()
    return tuple(int(80 + channel % 176) for channel in digest[:3])


def _annotate(rgb: np.ndarray, result: InferenceResult) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for shape in result.shapes:
        color = _color(shape.label)
        points = np.asarray(
            [[round(point.x), round(point.y)] for point in shape.points],
            dtype=np.int32,
        )
        if shape.type is ShapeType.RECTANGLE:
            cv2.rectangle(canvas, points[0], points[1], color, 3, cv2.LINE_AA)
            anchor = points[0]
        elif shape.type is ShapeType.POINT:
            cv2.circle(canvas, points[0], 5, color, -1, cv2.LINE_AA)
            anchor = points[0]
        else:
            cv2.polylines(canvas, [points], True, color, 3, cv2.LINE_AA)
            anchor = points[0]
        caption = shape.label or shape.type.value
        if shape.score is not None:
            caption += f" {shape.score:.3f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_x = min(max(2, int(anchor[0])), max(2, canvas.shape[1] - text_width - 2))
        text_y = min(
            max(text_height + baseline + 2, int(anchor[1]) - 7),
            canvas.shape[0] - baseline - 2,
        )
        text_origin = (text_x, text_y)
        cv2.putText(
            canvas,
            caption,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            caption,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def _write_html(output_dir: Path, summary: dict[str, Any]) -> None:
    rows = []
    for item in summary["images"]:
        failures = "<br>".join(html.escape(value) for value in item["failures"])
        status = "PASS" if item["passed"] else "FAIL"
        rows.append(
            "<section>"
            f"<h2>{html.escape(item['image'])} — {status}</h2>"
            f'<img src="{html.escape(item["annotated_image"])}" '
            'loading="lazy" alt="annotated inference output">'
            f"<p>Shapes: {item['shape_count']} · "
            f"Consistency: {html.escape(item['consistency_digest'])}</p>"
            f'<p class="fail">{failures}</p>'
            "</section>"
        )
    global_failures = "<br>".join(html.escape(value) for value in summary["failures"])
    document = (
        """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AnyLearning real-model validation</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}
img{max-width:100%;height:auto;border:1px solid #555}section{margin:2rem 0}.fail{color:#ff8a8a;white-space:pre-wrap}
</style>
<h1>AnyLearning real-model validation</h1>
"""
        + f'<p class="fail">{global_failures}</p>\n'
        + "\n".join(rows)
    )
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def load_validation_manifest(path: Path) -> RealModelValidationManifest:
    path = path.resolve(strict=True)
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError(f"Validation manifest exceeds {_MAX_MANIFEST_BYTES} bytes")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RealModelValidationManifest.model_validate(payload)


def _rss_bytes() -> int:
    """Return current process RSS portably for the report."""
    return int(psutil.Process().memory_info().rss)


def run_real_model_validation(
    manifest_path: Path,
    *,
    output_root: Path = Path("validation-results"),
) -> Path:
    """Run repeated real inference and return the immutable report directory."""
    manifest_path = manifest_path.resolve(strict=True)
    manifest = load_validation_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{stamp}-{_safe_name(manifest.name)}-", dir=output_root
        )
    )

    config = dict(manifest.config)
    if manifest.backend == "yolo_onnx" and "config_file" not in config:
        config["config_file"] = str(manifest_path)
    registry = get_default_registry()
    session = registry.create_session(manifest.backend, config)
    rss_samples = {"before_load": _rss_bytes()}
    load_started = time.perf_counter()
    session.load()
    load_ms = (time.perf_counter() - load_started) * 1000
    rss_samples["after_load"] = _rss_bytes()
    peak_rss_bytes = max(rss_samples.values())
    image_summaries: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    cached_images: list[tuple[np.ndarray, str, str]] = []
    baseline_digests: list[str] = []
    try:
        for index, image_case in enumerate(manifest.images):
            image_path = image_case.path
            if not image_path.is_absolute():
                image_path = manifest_dir / image_path
            image_path = image_path.resolve(strict=True)
            rgb = _load_rgb(image_path)
            image_digest = _sha256(image_path)
            results: list[InferenceResult] = []
            for run in range(manifest.runs):
                request = InferenceRequest(
                    request_id=f"{manifest.name}-{index}-{run}",
                    source_id=f"image-sha256:{image_digest}",
                    model_id=session.capabilities.model_id,
                    model_revision=session.capabilities.model_revision,
                    parameters=image_case.request_parameters,
                )
                results.append(session.predict(request, rgb))
                peak_rss_bytes = max(peak_rss_bytes, _rss_bytes())
            canonical = [
                result.model_copy(update={"request_id": f"{manifest.name}-{index}"})
                for result in results
            ]
            digests = [_result_digest(result) for result in canonical]
            failures = _check_expectations(results[0], image_case.expected)
            if len(set(digests)) != 1:
                failures.append("repeated inference results were not byte-consistent")
            case_name = f"{index:03d}-{_safe_name(image_path.stem)}"
            annotated_name = case_name + "-annotated.png"
            if not cv2.imwrite(
                str(output_dir / annotated_name), _annotate(rgb, results[0])
            ):
                raise OSError(f"Could not write {annotated_name}")
            all_results.append(
                {
                    "image": image_path.name,
                    "image_sha256": image_digest,
                    "runs": [
                        _result_payload(result, timings=True) for result in results
                    ],
                    "lifecycle_runs": [],
                }
            )
            cached_images.append((rgb, image_digest, image_path.name))
            baseline_digests.append(digests[0])
            image_summaries.append(
                {
                    "image": image_path.name,
                    "image_sha256": image_digest,
                    "annotated_image": annotated_name,
                    "shape_count": len(results[0].shapes),
                    "labels": dict(
                        Counter(
                            shape.label
                            for shape in results[0].shapes
                            if shape.label is not None
                        )
                    ),
                    "consistency_digest": digests[0],
                    "consistent_runs": len(set(digests)) == 1,
                    "failures": failures,
                    "passed": not failures,
                    "timings_ms": [result.timings_ms for result in results],
                }
            )
    finally:
        rss_samples["before_unload"] = _rss_bytes()
        peak_rss_bytes = max(peak_rss_bytes, rss_samples["before_unload"])
        unload_started = time.perf_counter()
        session.unload()
        unload_ms = (time.perf_counter() - unload_started) * 1000
        rss_samples["after_unload"] = _rss_bytes()

    lifecycle_metrics: list[dict[str, Any]] = []
    global_failures: list[str] = []
    steady_state_rss = rss_samples["after_unload"]
    for cycle in range(1, manifest.lifecycle_cycles):
        lifecycle_session = registry.create_session(manifest.backend, config)
        cycle_load_started = time.perf_counter()
        lifecycle_session.load()
        cycle_load_ms = (time.perf_counter() - cycle_load_started) * 1000
        cycle_inference_ms: list[float] = []
        try:
            for index, (rgb, image_digest, _image_name) in enumerate(cached_images):
                image_case = manifest.images[index]
                request = InferenceRequest(
                    request_id=f"{manifest.name}-{index}-lifecycle-{cycle}",
                    source_id=f"image-sha256:{image_digest}",
                    model_id=lifecycle_session.capabilities.model_id,
                    model_revision=lifecycle_session.capabilities.model_revision,
                    parameters=image_case.request_parameters,
                )
                result = lifecycle_session.predict(request, rgb)
                cycle_inference_ms.append(result.timings_ms["total"])
                all_results[index]["lifecycle_runs"].append(
                    _result_payload(result, timings=True)
                )
                canonical = result.model_copy(
                    update={"request_id": f"{manifest.name}-{index}"}
                )
                if _result_digest(canonical) != baseline_digests[index]:
                    global_failures.append(
                        f"lifecycle cycle {cycle} changed results for "
                        f"{cached_images[index][2]!r}"
                    )
                expectation_failures = _check_expectations(result, image_case.expected)
                global_failures.extend(
                    f"lifecycle cycle {cycle}, {cached_images[index][2]}: {failure}"
                    for failure in expectation_failures
                )
                peak_rss_bytes = max(peak_rss_bytes, _rss_bytes())
        finally:
            cycle_unload_started = time.perf_counter()
            lifecycle_session.unload()
            cycle_unload_ms = (time.perf_counter() - cycle_unload_started) * 1000
        after_unload_rss = _rss_bytes()
        peak_rss_bytes = max(peak_rss_bytes, after_unload_rss)
        lifecycle_metrics.append(
            {
                "cycle": cycle + 1,
                "load_ms": cycle_load_ms,
                "inference_ms": cycle_inference_ms,
                "unload_ms": cycle_unload_ms,
                "rss_bytes_after_unload": after_unload_rss,
            }
        )
    steady_state_growth = max(0, _rss_bytes() - steady_state_rss)
    if steady_state_growth > manifest.maximum_steady_state_rss_growth_bytes:
        global_failures.append(
            f"steady-state RSS grew {steady_state_growth} bytes; limit is "
            f"{manifest.maximum_steady_state_rss_growth_bytes}"
        )

    model_path_value = config.get("model_path")
    model_details: dict[str, Any] = {}
    if isinstance(model_path_value, (str, Path)):
        model_path = Path(model_path_value)
        if not model_path.is_absolute():
            model_path = manifest_dir / model_path
        model_path = model_path.resolve(strict=True)
        model_details = {
            "filename": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": _sha256(model_path),
        }
        external_manifest = config.get("external_data_sha256")
        if isinstance(external_manifest, dict):
            external_files = []
            total_external_bytes = 0
            for location, expected_digest in sorted(external_manifest.items()):
                external_path = (model_path.parent / location).resolve(strict=True)
                external_path.relative_to(model_path.parent)
                size = external_path.stat().st_size
                digest = _sha256(external_path)
                if digest != expected_digest.lower():
                    raise ValueError(
                        "External-data report digest changed after inference"
                    )
                total_external_bytes += size
                external_files.append(
                    {"location": location, "bytes": size, "sha256": digest}
                )
            model_details["external_files"] = external_files
            model_details["external_bytes"] = total_external_bytes
    summary = {
        "schema_version": 1,
        "name": manifest.name,
        "backend": manifest.backend,
        "provenance": manifest.provenance.model_dump(mode="json"),
        "passed": not global_failures
        and all(item["passed"] for item in image_summaries),
        "failures": global_failures,
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "model": model_details,
        "config": _redact(manifest.config),
        "runs_per_image": manifest.runs,
        "lifecycle_cycles": manifest.lifecycle_cycles,
        "lifecycle": lifecycle_metrics,
        "steady_state_rss_growth_bytes": steady_state_growth,
        "maximum_steady_state_rss_growth_bytes": (
            manifest.maximum_steady_state_rss_growth_bytes
        ),
        "load_ms": load_ms,
        "unload_ms": unload_ms,
        "peak_observed_rss_bytes": peak_rss_bytes,
        "rss_bytes": rss_samples,
        "platform": sys.platform,
        "images": image_summaries,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "results.json", all_results)
    _write_html(output_dir, summary)
    return output_dir

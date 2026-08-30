#!/usr/bin/env python3
"""Exercise auth, HTTP transport, queueing, and results with a real ONNX model."""

from __future__ import annotations

import argparse
import html
import json
import secrets
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import psutil
from fastapi.testclient import TestClient

from anylearning.inference import InferenceRequest, InferenceResult
from anylearning.inference.validation import (
    _annotate,
    _check_expectations,
    _load_rgb,
    _result_digest,
    load_validation_manifest,
)
from anylearning.server import (
    ServerModelDefinition,
    ServerSettings,
    create_server_app,
    encode_request_header,
    hash_password,
)
from anylearning.server.transport import encoded_image_source_id

_PASSWORD = "real-model-validation-password"
_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_server_validation(
    manifest_path: Path,
    output_root: Path,
    *,
    model_path_override: Path | None = None,
    image_path_override: Path | None = None,
) -> Path:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = load_validation_manifest(manifest_path)
    if manifest.backend != "yolo_onnx":
        raise ValueError("public real-server validation currently requires yolo_onnx")
    config = dict(manifest.config)
    config["config_file"] = str(manifest_path)
    if model_path_override is not None:
        config["model_path"] = str(model_path_override.resolve(strict=True))
    definition = ServerModelDefinition(backend=manifest.backend, config=config)
    settings = ServerSettings(
        password_hash=hash_password(_PASSWORD),
        token_secret=secrets.token_bytes(32),
        token_ttl_seconds=300,
        prediction_timeout_seconds=120,
        prediction_result_ttl_seconds=300,
    )
    app = create_server_app(settings, model_definitions=(definition,))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / f"{stamp}-real-server-{secrets.token_hex(4)}"
    output_dir.mkdir(mode=0o700)

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    image_summaries: list[dict[str, object]] = []
    retained_results: list[dict[str, object]] = []
    global_failures: list[str] = []
    started = time.perf_counter()
    with TestClient(app) as client:
        login = client.post("/v1/auth/token", json={"password": _PASSWORD})
        if login.status_code != 200:
            raise RuntimeError("real-server login failed")
        authorization = {"Authorization": f"Bearer {login.json()['access_token']}"}
        discovered = client.get("/v1/models", headers=authorization)
        if discovered.status_code != 200 or len(discovered.json()["models"]) != 1:
            raise RuntimeError("real-server model discovery failed")
        capabilities = discovered.json()["models"][0]

        for index, image_case in enumerate(manifest.images):
            image_path = (
                image_path_override
                if image_path_override is not None
                else image_case.path
            )
            if image_path_override is not None and len(manifest.images) != 1:
                raise ValueError("image path override requires a one-image manifest")
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            image_path = image_path.resolve(strict=True)
            media_type = _MEDIA_TYPES.get(image_path.suffix.lower())
            if media_type is None:
                raise ValueError("real-server validation image type is unsupported")
            encoded = image_path.read_bytes()
            rgb = _load_rgb(image_path)
            results: list[InferenceResult] = []
            round_trip_ms: list[float] = []
            for run in range(manifest.runs):
                request = InferenceRequest(
                    request_id=f"server-{manifest.name}-{index}-{run}",
                    source_id=encoded_image_source_id(encoded),
                    model_id=capabilities["model_id"],
                    model_revision=capabilities["model_revision"],
                    parameters=image_case.request_parameters,
                )
                run_started = time.perf_counter()
                submitted = client.post(
                    "/v1/predictions",
                    content=encoded,
                    headers={
                        **authorization,
                        "Content-Type": media_type,
                        "X-AnyLearning-Request": encode_request_header(request),
                    },
                )
                if submitted.status_code != 202:
                    raise RuntimeError(
                        f"real-server submission failed: {submitted.status_code}"
                    )
                job_id = submitted.json()["job_id"]
                deadline = time.monotonic() + settings.prediction_timeout_seconds
                while True:
                    polled = client.get(
                        f"/v1/predictions/{job_id}", headers=authorization
                    )
                    if polled.status_code != 200:
                        raise RuntimeError("real-server polling failed")
                    state = polled.json()["state"]
                    if state == "succeeded":
                        result = InferenceResult.model_validate(polled.json()["result"])
                        break
                    if state not in {"queued", "running"}:
                        raise RuntimeError(f"real-server prediction ended as {state}")
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "real-server client polling deadline expired"
                        )
                    time.sleep(0.01)
                round_trip_ms.append((time.perf_counter() - run_started) * 1000)
                results.append(result)
                client.delete(f"/v1/predictions/{job_id}", headers=authorization)
                peak_rss = max(peak_rss, process.memory_info().rss)

            canonical = [
                result.model_copy(update={"request_id": "canonical-server-request"})
                for result in results
            ]
            digests = [_result_digest(result) for result in canonical]
            failures = _check_expectations(results[0], image_case.expected)
            if len(set(digests)) != 1:
                failures.append("server results changed across identical requests")
            global_failures.extend(f"{image_path.name}: {item}" for item in failures)
            annotated_name = f"{index:03d}-{image_path.stem}-server.png"
            if not cv2.imwrite(
                str(output_dir / annotated_name), _annotate(rgb, results[0])
            ):
                raise OSError("could not write real-server annotated image")
            image_summaries.append(
                {
                    "image": image_path.name,
                    "annotated_image": annotated_name,
                    "shape_count": len(results[0].shapes),
                    "consistent_runs": len(set(digests)) == 1,
                    "consistency_digest": digests[0],
                    "round_trip_ms": round_trip_ms,
                    "failures": failures,
                }
            )
            retained_results.append(
                {
                    "image": image_path.name,
                    "runs": [result.model_dump(mode="json") for result in results],
                }
            )

    summary = {
        "schema_version": 1,
        "passed": not global_failures,
        "created_at": datetime.now(UTC).isoformat(),
        "manifest": manifest_path.name,
        "provenance": manifest.provenance.model_dump(mode="json"),
        "model": capabilities,
        "runs_per_image": manifest.runs,
        "total_round_trip_ms": (time.perf_counter() - started) * 1000,
        "peak_observed_rss_bytes": peak_rss,
        "failures": global_failures,
        "images": image_summaries,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "results.json", retained_results)
    rows = "\n".join(
        f"<h2>{html.escape(str(item['image']))}</h2>"
        f'<img src="{html.escape(str(item["annotated_image"]))}" '
        'style="max-width:100%;height:auto">'
        for item in image_summaries
    )
    (output_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>AnyLearning real server validation</title>"
        f"<h1>Real authenticated ONNX server: {'PASS' if summary['passed'] else 'FAIL'}</h1>{rows}",
        encoding="utf-8",
    )
    if global_failures:
        raise AssertionError("; ".join(global_failures))
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("validation-results"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--image-path", type=Path)
    arguments = parser.parse_args()
    print(
        run_server_validation(
            arguments.manifest,
            arguments.output_root,
            model_path_override=arguments.model_path,
            image_path_override=arguments.image_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

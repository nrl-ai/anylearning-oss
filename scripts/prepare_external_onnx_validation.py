#!/usr/bin/env python3
"""Derive a validation-only external-data ONNX bundle and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import onnx

# Make the checkout importable when invoked as `python scripts/...` without an
# editable install. Installed entry points do not need this path adjustment.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anylearning.inference.backends.onnx_safety import validate_onnx_artifact

_MAX_SOURCE_MODEL_BYTES = 2 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_external_validation_bundle(
    source_model: Path,
    source_manifest: Path,
    output_dir: Path,
) -> Path:
    source_model = source_model.resolve(strict=True)
    source_manifest = source_manifest.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if source_manifest.stat().st_size > 1_048_576:
        raise ValueError("Source manifest exceeds 1 MiB")
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Source validation manifest is invalid")
    expected_source_sha256 = payload["config"].get("sha256")
    if (
        not isinstance(expected_source_sha256, str)
        or len(expected_source_sha256) != 64
        or _sha256(source_model) != expected_source_sha256.lower()
    ):
        raise ValueError("Source model SHA-256 does not match its manifest")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        graph_path = staging / "model.onnx"
        data_name = "weights.bin"
        data_path = staging / data_name
        model = validate_onnx_artifact(
            source_model,
            max_bytes=_MAX_SOURCE_MODEL_BYTES,
            allow_external_data=False,
        )
        onnx.save_model(
            model,
            graph_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=0,
            # Tensor-valued attributes may be required before runtime file
            # overrides are applied. They are usually tiny, so retain them in
            # the graph while large initializers remain external.
            convert_attribute=False,
        )
        if not data_path.is_file() or data_path.stat().st_size <= 0:
            raise RuntimeError("ONNX conversion did not create external tensor data")

        config = dict(payload["config"])
        config.update(
            {
                "model_path": graph_path.name,
                "sha256": _sha256(graph_path),
                "external_data_sha256": {data_name: _sha256(data_path)},
            }
        )
        payload["config"] = config
        payload["name"] = f"{payload.get('name', 'real-model')}-external-data"
        images = payload.get("images")
        if not isinstance(images, list):
            raise ValueError("Source validation manifest has no image cases")
        for image in images:
            if not isinstance(image, dict) or not isinstance(image.get("path"), str):
                raise ValueError("Source validation image entry is invalid")
            source_image = (source_manifest.parent / image["path"]).resolve(strict=True)
            image["path"] = os.path.relpath(source_image, staging).replace(os.sep, "/")
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
        return output_dir / manifest_path.name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_model", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    manifest = prepare_external_validation_bundle(
        arguments.source_model,
        arguments.source_manifest,
        arguments.output_dir,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

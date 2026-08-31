#!/usr/bin/env python3
"""Expose the four masks hidden by official EfficientViT-SAM ONNX decoders.

The official downloadable graphs select one mask inside the graph.  The native
evaluation path instead scores the three multimask candidates and keeps the
best one.  This deterministic, data-only transform exposes the four tensors
immediately before the lossy selection so the inference backend can reproduce
the native policy without loading a checkpoint or importing PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper

_MAX_SOURCE_BYTES = 128 * 1024**2
_EXPECTED_INPUTS = ("image_embeddings", "point_coords", "point_labels")
_EXPECTED_OUTPUTS = ("masks", "iou_predictions")
_RAW_MASKS = "/Reshape_5_output_0"
_RAW_SCORES = "/iou_prediction_head/layers.2/Gemm_output_0"
_UNUSED_MASKS = "selected_masks_unused"
_UNUSED_SCORES = "selected_iou_predictions_unused"
_TRANSFORM_VERSION = "efficientvit-sam-multimask-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: str, label: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be exactly 64 hexadecimal characters")
    return value.lower()


def _validated_source(path: Path, expected_sha256: str) -> Path:
    expected_sha256 = _validated_sha256(expected_sha256, "Source decoder SHA-256")
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Source decoder must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_SOURCE_BYTES:
        raise ValueError(
            f"Source decoder has {size} bytes; maximum is {_MAX_SOURCE_BYTES}"
        )
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Source decoder SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return path.resolve()


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _reject_symlink_components(path: Path) -> None:
    """Reject existing symlinks without resolving away the evidence."""

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Output path may not traverse a symlink: {current}")


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {name: node for node in model.graph.node for name in node.output}


def _rename_tensor(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for index, name in enumerate(node.output):
            if name == old:
                node.output[index] = new
    for collection in (
        model.graph.input,
        model.graph.output,
        model.graph.value_info,
        model.graph.initializer,
    ):
        for value in collection:
            if value.name == old:
                value.name = new
    for value in model.graph.sparse_initializer:
        if value.values.name == old:
            value.values.name = new
        if value.indices.name == old:
            value.indices.name = new


def prepare_decoder(
    source: Path,
    output: Path,
    *,
    source_sha256: str,
    expected_output_sha256: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    source_sha256 = _validated_sha256(source_sha256, "Source decoder SHA-256")
    if expected_output_sha256 is not None:
        expected_output_sha256 = _validated_sha256(
            expected_output_sha256, "Prepared decoder SHA-256"
        )
    source = _validated_source(source, source_sha256)
    output = _absolute_without_symlink_resolution(output)
    if output == source:
        raise ValueError("Output decoder must differ from the source decoder")
    _reject_symlink_components(output)
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output}")
    if output.exists() and not output.is_file():
        raise ValueError("Output decoder must be a regular file")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output)

    model = onnx.load_model(source, load_external_data=False)
    onnx.checker.check_model(model)
    if any(
        tensor.data_location == TensorProto.EXTERNAL
        for tensor in model.graph.initializer
    ):
        raise ValueError("Official decoder transform does not accept external data")
    if tuple(value.name for value in model.graph.input) != _EXPECTED_INPUTS:
        raise ValueError("Unexpected EfficientViT-SAM decoder input contract")
    if tuple(value.name for value in model.graph.output) != _EXPECTED_OUTPUTS:
        raise ValueError("Unexpected EfficientViT-SAM decoder output contract")
    if [(item.domain, item.version) for item in model.opset_import] != [("", 17)]:
        raise ValueError("EfficientViT-SAM source decoder must use ONNX opset 17")

    producers = _producer_map(model)
    expected_nodes = {
        _RAW_MASKS: "Reshape",
        _RAW_SCORES: "Gemm",
        "masks": "Unsqueeze",
        "iou_predictions": "Unsqueeze",
    }
    for tensor_name, operation in expected_nodes.items():
        producer = producers.get(tensor_name)
        if producer is None or producer.op_type != operation:
            raise ValueError(
                f"Decoder tensor {tensor_name!r} is not produced by {operation}"
            )

    # Free the public output names before assigning them to the full tensors.
    _rename_tensor(model, "masks", _UNUSED_MASKS)
    _rename_tensor(model, "iou_predictions", _UNUSED_SCORES)
    _rename_tensor(model, _RAW_MASKS, "masks")
    _rename_tensor(model, _RAW_SCORES, "iou_predictions")
    del model.graph.output[:]
    model.graph.output.extend(
        (
            helper.make_tensor_value_info(
                "masks", TensorProto.FLOAT, ["batch_size", 4, 256, 256]
            ),
            helper.make_tensor_value_info(
                "iou_predictions", TensorProto.FLOAT, ["batch_size", 4]
            ),
        )
    )
    model.producer_name = "AnyLearning"
    model.producer_version = _TRANSFORM_VERSION
    helper.set_model_props(
        model,
        {
            "anylearning.transform": _TRANSFORM_VERSION,
            "anylearning.source_sha256": source_sha256.lower(),
        },
    )
    onnx.checker.check_model(model)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        onnx.save_model(model, temporary)
        output_sha256 = _sha256(temporary)
        if (
            expected_output_sha256 is not None
            and output_sha256 != expected_output_sha256
        ):
            raise ValueError(
                "Prepared decoder SHA-256 mismatch: expected "
                f"{expected_output_sha256}, got {output_sha256}"
            )
        os.replace(temporary, output)
        output.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": output_sha256,
        "source": str(source),
        "source_sha256": source_sha256,
        "transform": _TRANSFORM_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    report = prepare_decoder(
        arguments.source,
        arguments.output,
        source_sha256=arguments.source_sha256,
        expected_output_sha256=arguments.expected_output_sha256,
        force=arguments.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

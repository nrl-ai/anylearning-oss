#!/usr/bin/env python3
"""Repair known exporter metadata defects in checksum-pinned SAM2 encoders.

The supported SAM2 and SAM2.1 encoder graphs are already numerically correct,
but their exported metadata makes ONNX Runtime fall back to lenient shape
merging. Some SAM2 sizes also retain initializers that no node can reach. This
deterministic, data-only transform repairs those exact defects, removes only
unreferenced initializers, and requires strict ONNX shape inference before the
result becomes visible.
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
from onnx import AttributeProto, TensorProto, helper

_MAX_SOURCE_BYTES = 1536 * 1024**2
_EXPECTED_INPUTS = ("image",)
_EXPECTED_OUTPUTS = ("high_res_feats_0", "high_res_feats_1", "image_embed")
_SAM2_STALE_VALUES = frozenset(("/conv_s0/Conv_output_0", "/conv_s1/Conv_output_0"))
_SAM21_IF_NODE = "/image_encoder/trunk/If"
_SAM21_THEN_OUTPUT = "/image_encoder/trunk/Concat_3_output_0"
_TRANSFORM_VERSION = "sam2-encoder-metadata-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_SUPPORTED_FAMILIES = frozenset(("sam2", "sam2_1"))


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
    expected_sha256 = _validated_sha256(expected_sha256, "Source encoder SHA-256")
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Source encoder must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_SOURCE_BYTES:
        raise ValueError(
            f"Source encoder has {size} bytes; maximum is {_MAX_SOURCE_BYTES}"
        )
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Source encoder SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return path.resolve()


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Output path may not traverse a symlink: {current}")


def _dimensions(value: onnx.ValueInfoProto) -> tuple[int | str | None, ...]:
    dimensions: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return tuple(dimensions)


def _subgraphs(node: onnx.NodeProto) -> tuple[onnx.GraphProto, ...]:
    graphs: list[onnx.GraphProto] = []
    for attribute in node.attribute:
        if attribute.type == AttributeProto.GRAPH:
            graphs.append(attribute.g)
        elif attribute.type == AttributeProto.GRAPHS:
            graphs.extend(attribute.graphs)
    return tuple(graphs)


def _all_graphs(graph: onnx.GraphProto) -> tuple[onnx.GraphProto, ...]:
    graphs = [graph]
    for node in graph.node:
        for child in _subgraphs(node):
            graphs.extend(_all_graphs(child))
    return tuple(graphs)


def _validate_contract(model: onnx.ModelProto, family: str) -> None:
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported SAM2 encoder family: {family!r}")
    expected_opset = 17 if family == "sam2" else 18
    if [(item.domain, item.version) for item in model.opset_import] != [
        ("", expected_opset)
    ]:
        raise ValueError(f"{family} encoder must use ONNX opset {expected_opset}")
    if tuple(value.name for value in model.graph.input) != _EXPECTED_INPUTS:
        raise ValueError("Unexpected SAM2 encoder input contract")
    if tuple(value.name for value in model.graph.output) != _EXPECTED_OUTPUTS:
        raise ValueError("Unexpected SAM2 encoder output contract")
    image = model.graph.input[0]
    if image.type.tensor_type.elem_type != TensorProto.FLOAT or _dimensions(image) != (
        1,
        3,
        1024,
        1024,
    ):
        raise ValueError("SAM2 encoder image input must be float32 1x3x1024x1024")
    if any(
        value.type.tensor_type.elem_type != TensorProto.FLOAT
        or len(_dimensions(value)) != 4
        for value in model.graph.output
    ):
        raise ValueError("SAM2 encoder outputs must be rank-4 float32 tensors")
    for graph in _all_graphs(model.graph):
        if any(
            tensor.data_location == TensorProto.EXTERNAL for tensor in graph.initializer
        ):
            raise ValueError("SAM2 encoder transform does not accept external data")


def _producer_map(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {name: node for node in graph.node for name in node.output}


def _repair_sam2_metadata(model: onnx.ModelProto) -> list[str]:
    producers = _producer_map(model.graph)
    stale = {
        value.name: value
        for value in model.graph.value_info
        if value.name in _SAM2_STALE_VALUES
    }
    if set(stale) != _SAM2_STALE_VALUES:
        raise ValueError(
            "SAM2 encoder does not contain the expected stale Conv metadata"
        )
    for name, value in stale.items():
        producer = producers.get(name)
        if (
            producer is None
            or producer.op_type != "Conv"
            or value.type.tensor_type.elem_type != TensorProto.FLOAT
            or _dimensions(value) != ()
        ):
            raise ValueError(f"Unexpected stale SAM2 tensor contract: {name}")
    retained = [value for value in model.graph.value_info if value.name not in stale]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)
    return sorted(stale)


def _repair_sam21_metadata(model: onnx.ModelProto) -> list[str]:
    matches = [node for node in model.graph.node if node.name == _SAM21_IF_NODE]
    if len(matches) != 1 or matches[0].op_type != "If":
        raise ValueError("SAM2.1 encoder does not contain the expected If node")
    attributes = {item.name: item for item in matches[0].attribute}
    if set(attributes) != {"else_branch", "then_branch"}:
        raise ValueError("Unexpected SAM2.1 If branch contract")
    then_graph = attributes["then_branch"].g
    else_graph = attributes["else_branch"].g
    if len(then_graph.output) != 1 or len(else_graph.output) != 1:
        raise ValueError("SAM2.1 If branches must each expose one output")
    then_output = then_graph.output[0]
    else_output = else_graph.output[0]
    producer = _producer_map(then_graph).get(_SAM21_THEN_OUTPUT)
    if (
        then_output.name != _SAM21_THEN_OUTPUT
        or _dimensions(then_output) != (5,)
        or else_output.type.tensor_type.elem_type != TensorProto.INT64
        or _dimensions(else_output) != (4,)
        or then_output.type.tensor_type.elem_type != TensorProto.INT64
        or producer is None
        or producer.op_type != "Concat"
        or tuple(producer.output) != (_SAM21_THEN_OUTPUT,)
        or len(producer.input) != 2
    ):
        raise ValueError("Unexpected stale SAM2.1 branch-output contract")
    del then_output.type.tensor_type.shape.dim[:]
    then_output.type.tensor_type.shape.dim.add().dim_value = 4
    return [_SAM21_THEN_OUTPUT]


def _referenced_names(graph: onnx.GraphProto) -> set[str]:
    referenced = {name for node in graph.node for name in node.input if name}
    referenced.update(value.name for value in graph.output)
    for node in graph.node:
        for child in _subgraphs(node):
            referenced.update(_referenced_names(child))
    return referenced


def _remove_unused_initializers(graph: onnx.GraphProto) -> list[str]:
    referenced = _referenced_names(graph)
    removed = [
        tensor.name for tensor in graph.initializer if tensor.name not in referenced
    ]
    retained = [tensor for tensor in graph.initializer if tensor.name in referenced]
    del graph.initializer[:]
    graph.initializer.extend(retained)
    for node in graph.node:
        for child in _subgraphs(node):
            removed.extend(_remove_unused_initializers(child))
    return removed


def prepare_encoder(
    source: Path,
    output: Path,
    *,
    family: str,
    source_sha256: str,
    expected_output_sha256: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    source_sha256 = _validated_sha256(source_sha256, "Source encoder SHA-256")
    if expected_output_sha256 is not None:
        expected_output_sha256 = _validated_sha256(
            expected_output_sha256, "Prepared encoder SHA-256"
        )
    source = _validated_source(source, source_sha256)
    output = _absolute_without_symlink_resolution(output)
    if output == source:
        raise ValueError("Output encoder must differ from the source encoder")
    _reject_symlink_components(output)
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output}")
    if output.exists() and (output.is_symlink() or not output.is_file()):
        raise ValueError("Output encoder must be a regular non-symlink file")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output)

    model = onnx.load_model(source, load_external_data=False)
    onnx.checker.check_model(model)
    _validate_contract(model, family)
    source_producer = model.producer_name
    source_producer_version = model.producer_version
    if family == "sam2":
        repaired_values = _repair_sam2_metadata(model)
    else:
        repaired_values = _repair_sam21_metadata(model)
    removed_initializers = sorted(_remove_unused_initializers(model.graph))
    properties = {item.key: item.value for item in model.metadata_props}
    properties.update(
        {
            "anylearning.family": family,
            "anylearning.source_producer": source_producer,
            "anylearning.source_producer_version": source_producer_version,
            "anylearning.source_sha256": source_sha256,
            "anylearning.transform": _TRANSFORM_VERSION,
        }
    )
    model.producer_name = "AnyLearning"
    model.producer_version = _TRANSFORM_VERSION
    helper.set_model_props(model, properties)
    onnx.checker.check_model(model)

    raw_descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".raw.tmp", dir=output.parent
    )
    os.close(raw_descriptor)
    inferred_descriptor, inferred_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".inferred.tmp", dir=output.parent
    )
    os.close(inferred_descriptor)
    raw_path = Path(raw_name)
    inferred_path = Path(inferred_name)
    try:
        onnx.save_model(model, raw_path)
        onnx.shape_inference.infer_shapes_path(
            raw_path,
            inferred_path,
            check_type=True,
            strict_mode=True,
            data_prop=True,
        )
        onnx.checker.check_model(inferred_path, full_check=True)
        output_sha256 = _sha256(inferred_path)
        if (
            expected_output_sha256 is not None
            and output_sha256 != expected_output_sha256
        ):
            raise ValueError(
                "Prepared encoder SHA-256 mismatch: expected "
                f"{expected_output_sha256}, got {output_sha256}"
            )
        os.replace(inferred_path, output)
        output.chmod(0o644)
    finally:
        raw_path.unlink(missing_ok=True)
        inferred_path.unlink(missing_ok=True)

    return {
        "family": family,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": output_sha256,
        "removed_initializers": removed_initializers,
        "repaired_values": repaired_values,
        "source": str(source),
        "source_sha256": source_sha256,
        "transform": _TRANSFORM_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--family", choices=sorted(_SUPPORTED_FAMILIES), required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    report = prepare_encoder(
        arguments.source,
        arguments.output,
        family=arguments.family,
        source_sha256=arguments.source_sha256,
        expected_output_sha256=arguments.expected_output_sha256,
        force=arguments.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared bounded ONNX Runtime session construction for inference backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..runtime import CancellationToken
from .onnx_safety import (
    select_providers,
    stable_external_data_files,
    stable_onnx_artifact,
    validate_onnx_artifact,
)


def _session_options(
    onnxruntime: Any,
    *,
    enable_cpu_mem_arena: bool,
    intra_op_threads: int,
    inter_op_threads: int,
) -> Any:
    options = onnxruntime.SessionOptions()
    options.enable_cpu_mem_arena = enable_cpu_mem_arena
    if intra_op_threads:
        options.intra_op_num_threads = intra_op_threads
    if inter_op_threads:
        options.inter_op_num_threads = inter_op_threads
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def create_checked_onnx_session(
    path: Path,
    *,
    providers: Sequence[str],
    allow_cpu_fallback: bool,
    max_model_bytes: int,
    expected_sha256: str | None,
    external_data_sha256: Mapping[str, str] | None,
    max_external_data_bytes: int,
    enable_cpu_mem_arena: bool,
    intra_op_threads: int,
    inter_op_threads: int,
    cancellation: CancellationToken,
) -> tuple[Any, Any, tuple[str, ...]]:
    """Validate one stable graph and construct its runtime session.

    The parsed graph is returned for backend-specific contract checks. External
    tensor mappings remain live until ONNX Runtime has finished constructing its
    own session.
    """
    import onnxruntime

    selected, warnings = select_providers(
        providers,
        onnxruntime.get_available_providers(),
        allow_cpu_fallback=allow_cpu_fallback,
    )
    options = _session_options(
        onnxruntime,
        enable_cpu_mem_arena=enable_cpu_mem_arena,
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
    )
    cancellation.raise_if_cancelled()
    with stable_onnx_artifact(
        path,
        max_bytes=max_model_bytes,
        expected_sha256=expected_sha256,
        cancellation=cancellation,
    ) as (artifact, runtime_path, _digest):
        graph = validate_onnx_artifact(
            artifact,
            max_bytes=max_model_bytes,
            allow_external_data=True,
        )
        cancellation.raise_if_cancelled()
        with stable_external_data_files(
            graph,
            model_path=path,
            expected_sha256=external_data_sha256,
            max_bytes=max_external_data_bytes,
            cancellation=cancellation,
        ) as external_data:
            runtime_model = (
                external_data.add_to_session_options(options, graph)
                if external_data.locations
                else None
            )
            session = onnxruntime.InferenceSession(
                runtime_model if runtime_model is not None else runtime_path,
                sess_options=options,
                providers=list(selected),
            )
    return session, graph, warnings


__all__ = ["create_checked_onnx_session"]

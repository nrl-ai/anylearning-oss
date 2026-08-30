from types import SimpleNamespace

import pytest

from anylearning.inference.backends.onnx_session import _session_options


class FakeSessionOptions:
    enable_cpu_mem_arena = None
    intra_op_num_threads = None
    inter_op_num_threads = None
    graph_optimization_level = None


@pytest.mark.parametrize("enable_cpu_mem_arena", [False, True])
def test_session_options_apply_explicit_cpu_arena_policy(enable_cpu_mem_arena):
    runtime = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
    )

    options = _session_options(
        runtime,
        enable_cpu_mem_arena=enable_cpu_mem_arena,
        intra_op_threads=2,
        inter_op_threads=3,
    )

    assert options.enable_cpu_mem_arena is enable_cpu_mem_arena
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 3
    assert options.graph_optimization_level == "all"

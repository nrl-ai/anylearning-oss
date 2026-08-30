import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from anylearning.inference.backends import onnx_session as onnx_session_module
from anylearning.inference.backends.onnx_session import (
    _session_options,
    release_unused_cpu_memory,
)


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


def test_release_unused_cpu_memory_relaxes_every_macos_zone(monkeypatch):
    collect = Mock()
    pressure_relief = Mock(return_value=123)
    libc = SimpleNamespace(malloc_zone_pressure_relief=pressure_relief)
    monkeypatch.setattr(onnx_session_module.gc, "collect", collect)
    monkeypatch.setattr(onnx_session_module.sys, "platform", "darwin")
    monkeypatch.setattr(onnx_session_module.ctypes, "CDLL", lambda _name: libc)

    release_unused_cpu_memory()

    collect.assert_called_once_with()
    pressure_relief.assert_called_once_with(None, 0)
    assert pressure_relief.argtypes == [ctypes.c_void_p, ctypes.c_size_t]
    assert pressure_relief.restype is ctypes.c_size_t


def test_release_unused_cpu_memory_tolerates_missing_macos_api(monkeypatch):
    collect = Mock()
    monkeypatch.setattr(onnx_session_module.gc, "collect", collect)
    monkeypatch.setattr(onnx_session_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        onnx_session_module.ctypes,
        "CDLL",
        lambda _name: SimpleNamespace(),
    )

    release_unused_cpu_memory()

    collect.assert_called_once_with()

"""Tests for the shared device helpers used by every trainer.

These only need torch, so they run on GPU machines and CPU-only ones alike.
"""

import pytest
import torch
import torch.nn as nn

from anylearning.training import device_utils
from anylearning.training.device_utils import (
    get_device,
    get_model_device,
    load_torch_model,
    load_torch_model_for_export,
)


def tiny_model():
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))


@pytest.fixture
def saved_model(tmp_path):
    """A checkpoint written the way the trainers write it: the whole module."""
    model = tiny_model()
    model.train()  # trainers may save while still in training mode
    path = tmp_path / "best_model.pth"
    torch.save(model, path)
    return path


# --------------------------------------------------------------------------
# get_device
# --------------------------------------------------------------------------


def test_get_device_prefers_gpu_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device().type == "cuda"


def test_get_device_falls_back_to_cpu_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device().type == "cpu"


def test_get_device_honours_prefer_gpu_false(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device(prefer_gpu=False).type == "cpu"


# --------------------------------------------------------------------------
# get_model_device
# --------------------------------------------------------------------------


def test_get_model_device_reads_parameters():
    assert get_model_device(tiny_model()).type == "cpu"


def test_get_model_device_reads_buffers_when_there_are_no_parameters():
    class BufferOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(3))

    assert get_model_device(BufferOnly()).type == "cpu"


def test_get_model_device_defaults_to_cpu_for_an_empty_module():
    assert get_model_device(nn.Module()).type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_get_model_device_reports_cuda_for_a_gpu_model():
    assert get_model_device(tiny_model().to("cuda")).type == "cuda"


# --------------------------------------------------------------------------
# load_torch_model
# --------------------------------------------------------------------------


def test_load_torch_model_returns_an_eval_mode_model(saved_model):
    model = load_torch_model(saved_model, device="cpu")
    assert not model.training


def test_load_torch_model_can_keep_training_mode(saved_model):
    model = load_torch_model(saved_model, device="cpu", eval_mode=False)
    assert model.training


def test_load_torch_model_places_the_model_on_the_requested_device(saved_model):
    model = load_torch_model(saved_model, device="cpu")
    assert get_model_device(model).type == "cpu"


def test_load_torch_model_defaults_to_get_device(monkeypatch, saved_model):
    monkeypatch.setattr(device_utils, "get_device", lambda: torch.device("cpu"))
    assert get_model_device(load_torch_model(saved_model)).type == "cpu"


def test_load_torch_model_always_passes_map_location(monkeypatch, saved_model):
    """Guards the "Attempting to deserialize object on a CUDA device" failure.

    Without map_location a GPU-trained checkpoint cannot be opened at all on a
    machine without a GPU. This holds on any machine, GPU or not.
    """
    recorded = {}
    real_load = torch.load

    def spy(path, *args, **kwargs):
        recorded.update(kwargs)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    load_torch_model(saved_model, device="cpu")

    assert recorded.get("map_location") is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_load_torch_model_moves_a_cpu_checkpoint_to_the_gpu(tmp_path):
    path = tmp_path / "cpu_model.pth"
    torch.save(tiny_model(), path)

    model = load_torch_model(path, device="cuda")
    assert get_model_device(model).type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_load_torch_model_reads_a_gpu_checkpoint_onto_the_cpu(tmp_path):
    """The cross-machine case: trained on a GPU box, opened on a CPU-only one."""
    path = tmp_path / "gpu_model.pth"
    torch.save(tiny_model().to("cuda"), path)

    model = load_torch_model(path, device="cpu")
    assert get_model_device(model).type == "cpu"
    with torch.no_grad():
        model(torch.rand(1, 3, 8, 8))  # runs without a device mismatch


# --------------------------------------------------------------------------
# load_torch_model_for_export
# --------------------------------------------------------------------------


def test_load_torch_model_for_export_returns_a_cpu_eval_model(saved_model):
    model = load_torch_model_for_export(saved_model)
    assert get_model_device(model).type == "cpu"
    assert not model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_load_torch_model_for_export_pulls_a_gpu_checkpoint_to_the_cpu(tmp_path):
    path = tmp_path / "gpu_model.pth"
    torch.save(tiny_model().to("cuda"), path)

    assert get_model_device(load_torch_model_for_export(path)).type == "cpu"


# --------------------------------------------------------------------------
# accelerator, device_type: which backend, on which machine
# --------------------------------------------------------------------------


def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def with_mps(monkeypatch, available: bool):
    monkeypatch.setattr(device_utils, "mps_available", lambda: available)


def test_accelerator_prefers_cuda_over_mps(monkeypatch):
    """A machine with both is a machine with a real GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with_mps(monkeypatch, True)
    assert device_utils.accelerator() == device_utils.CUDA


def test_accelerator_reports_mps_when_that_is_all_there_is(monkeypatch):
    no_cuda(monkeypatch)
    with_mps(monkeypatch, True)
    assert device_utils.accelerator() == device_utils.MPS


def test_accelerator_is_none_on_a_cpu_only_machine(monkeypatch):
    no_cuda(monkeypatch)
    with_mps(monkeypatch, False)
    assert device_utils.accelerator() is None


def test_get_device_selects_metal_on_a_mac(monkeypatch):
    """The gap this whole thing exists to close: no CUDA is not no GPU."""
    no_cuda(monkeypatch)
    with_mps(monkeypatch, True)
    assert get_device().type == "mps"
    assert device_utils.gpu_available() is True
    # ...and it is still not CUDA, which is what pinned memory, cudnn and
    # GradScaler ask about.
    assert device_utils.cuda_available() is False


def test_choosing_the_cpu_beats_metal_too(monkeypatch):
    no_cuda(monkeypatch)
    with_mps(monkeypatch, True)
    monkeypatch.setenv(device_utils.DEVICE_PREFERENCE_ENV, "cpu")
    assert device_utils.device_type() == device_utils.CPU
    assert get_device().type == "cpu"
    assert device_utils.gpu_available() is False


def test_an_accelerator_id_reads_back_as_the_gpu(monkeypatch):
    monkeypatch.setenv(device_utils.DEVICE_PREFERENCE_ENV, "mps")
    assert device_utils.device_preference() == device_utils.GPU


def test_mps_available_survives_a_torch_without_the_backend(monkeypatch):
    class NoBackend:
        @property
        def mps(self):
            raise AttributeError("no mps in this build")

    monkeypatch.setattr(torch, "backends", NoBackend())
    assert device_utils.mps_available() is False


# --------------------------------------------------------------------------
# What an accelerator cannot be asked to do
# --------------------------------------------------------------------------


def test_instance_segmentation_is_not_offered_metal(monkeypatch):
    no_cuda(monkeypatch)
    with_mps(monkeypatch, True)
    reason = device_utils.excluded_reason("Instance Segmentation")
    assert reason and "roi_align" in reason


def test_handpose_is_not_offered_metal_because_it_is_slower(monkeypatch):
    """Not broken -- measured at 8.8x the CPU's time, which is reason enough."""
    no_cuda(monkeypatch)
    with_mps(monkeypatch, True)
    reason = device_utils.excluded_reason("Handpose Classification")
    assert reason and "slower" in reason


def test_the_three_working_trainers_may_use_metal(monkeypatch):
    no_cuda(monkeypatch)
    with_mps(monkeypatch, True)
    for project_type in (
        "Image Classification",
        "Image Segmentation",
        # NanoDet only after the Integral rewrite; see gfl_head.py.
        "Object Detection",
    ):
        assert device_utils.excluded_reason(project_type) is None


def test_cuda_has_no_exclusions(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device_utils.excluded_reason("Instance Segmentation") is None

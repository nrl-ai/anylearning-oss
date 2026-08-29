"""What /api/settings/devices answers, on each kind of machine.

The training dialog builds its Hardware menu from this, so the shape is a
contract: Automatic and CPU are always offered, and each entry in
`accelerators` becomes one more option named after the hardware it is.
"""

import torch

from anylearning.routers.settings import training_devices
from anylearning.training import device_utils


def machine(monkeypatch, cuda: bool, mps: bool, name="Test GPU"):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(device_utils, "mps_available", lambda: mps)
    monkeypatch.setattr(device_utils, "gpu_name", lambda: name)


def test_a_cpu_only_machine_offers_no_accelerator(monkeypatch):
    machine(monkeypatch, cuda=False, mps=False)
    answer = training_devices()
    assert answer["accelerators"] == []
    assert answer["cuda"] is False


def test_a_cuda_machine_offers_its_card(monkeypatch):
    machine(monkeypatch, cuda=True, mps=False, name="NVIDIA GeForce RTX 4090")
    answer = training_devices()
    (entry,) = answer["accelerators"]
    assert entry["id"] == "cuda"
    assert entry["name"] == "NVIDIA GeForce RTX 4090"
    assert "NVIDIA GeForce RTX 4090" in entry["label"]
    assert entry["excluded_project_types"] == []
    # Kept for a frontend that predates the list.
    assert answer["cuda"] is True


def test_an_apple_machine_offers_metal_and_says_so(monkeypatch):
    machine(monkeypatch, cuda=False, mps=True, name="Apple M1")
    answer = training_devices()
    (entry,) = answer["accelerators"]
    assert entry["id"] == "mps"
    assert entry["name"] == "Apple M1"
    assert "Metal" in entry["label"]
    # A Mac's GPU is not CUDA, and the old field has to keep telling the truth
    # -- pinned memory, cudnn and GradScaler are all downstream of it.
    assert answer["cuda"] is False


def test_metal_declares_what_it_should_not_be_asked_to_train(monkeypatch):
    machine(monkeypatch, cuda=False, mps=True, name="Apple M1")
    (entry,) = training_devices()["accelerators"]
    excluded = entry["excluded_project_types"]
    # Mask R-CNN aborts inside Metal; the landmark MLP is simply slower there.
    assert "Instance Segmentation" in excluded
    assert "Handpose Classification" in excluded
    # And the three that Metal is good at are offered it.
    for offered in ("Image Classification", "Image Segmentation", "Object Detection"):
        assert offered not in excluded


def test_a_broken_torch_answers_rather_than_raises(monkeypatch):
    def explode():
        raise RuntimeError("torch is not importable in this build")

    monkeypatch.setattr(device_utils, "accelerator", explode)
    answer = training_devices()
    assert answer["accelerators"] == []
    assert answer["cuda"] is False
    assert "torch is not importable" in answer["error"]

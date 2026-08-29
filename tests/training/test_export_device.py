"""Regression tests for the device handling of ONNX export and inference.

Trainers persist the whole ``nn.Module`` (``torch.save(model, ...)``), so a
checkpoint written on a GPU machine restores its weights onto CUDA. Exporting or
running inference then has to place its own tensors on the same device, otherwise
torch raises::

    Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)
    should be the same

which is what used to break ONNX export right after a successful training run.
"""

import onnx
import pytest
import torch
import torch.nn as nn
import yaml

from anylearning.training.device_utils import get_model_device
from anylearning.training.trainers.classification_trainer import ClassificationTrainer
from anylearning.training.trainers.semseg_trainer import SemSegTrainer

IMG_SIZE = 32
CLASS_NAMES = ["class1", "class2"]

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU to produce a CUDA checkpoint"
)


class StubTrainer:
    """Stands in for a real trainer so export_onnx runs without a database.

    ``export_onnx`` only reads these three members, so binding it to a stub keeps
    the test focused on device handling.
    """

    def __init__(self, config_path, output_folder, model_path=None):
        self.config_path = config_path
        self.output_folder = output_folder
        self._model_path = model_path

    def get_model_path(self):
        return self._model_path is not None, self._model_path


def segmentation_model():
    """Conv-first, like the ResNet encoder that raised the reported error."""
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, 3, 1))


def classification_model():
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, len(CLASS_NAMES)),
    )


TRAINERS = {
    "semseg": (SemSegTrainer, segmentation_model),
    "classification": (ClassificationTrainer, classification_model),
}


def write_config(tmp_path):
    config = {
        "data": {
            "img_size": IMG_SIZE,
            "class_names": CLASS_NAMES,
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "label_set": [
                {"id": 1, "name": "class1", "color": "#FF0000"},
                {"id": 2, "name": "class2", "color": "#00FF00"},
            ],
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def save_checkpoint(tmp_path, model, device="cpu"):
    output_folder = tmp_path / "output"
    output_folder.mkdir(exist_ok=True)
    model_path = output_folder / "best_model.pth"
    torch.save(model.to(device), model_path)
    return output_folder, model_path


# --------------------------------------------------------------------------
# ONNX export
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(TRAINERS))
def test_export_onnx_writes_a_valid_model(tmp_path, name):
    trainer_cls, build_model = TRAINERS[name]
    output_folder, model_path = save_checkpoint(tmp_path, build_model())

    onnx_path = trainer_cls.export_onnx(
        StubTrainer(write_config(tmp_path), output_folder, model_path)
    )

    assert onnx_path is not None
    onnx.checker.check_model(onnx.load(onnx_path))


@pytest.mark.parametrize("name", list(TRAINERS))
def test_export_onnx_puts_the_dummy_input_on_the_weights_device(
    monkeypatch, tmp_path, name
):
    """The invariant that broke: traced input and weights share a device."""
    trainer_cls, build_model = TRAINERS[name]
    output_folder, model_path = save_checkpoint(tmp_path, build_model())

    seen = {}

    def spy(model, args, path, *rest, **kwargs):
        tensor = args[0] if isinstance(args, tuple) else args
        seen["weights"] = get_model_device(model)
        seen["dummy_input"] = tensor.device
        seen["training_mode"] = model.training
        open(path, "wb").close()

    monkeypatch.setattr(torch.onnx, "export", spy)
    trainer_cls.export_onnx(
        StubTrainer(write_config(tmp_path), output_folder, model_path)
    )

    assert seen["weights"].type == seen["dummy_input"].type
    assert seen["training_mode"] is False, "export must trace in eval mode"


@pytest.mark.parametrize("name", list(TRAINERS))
def test_export_onnx_traces_on_the_cpu(monkeypatch, tmp_path, name):
    """Export stays on the CPU so it never fights the finished run for VRAM."""
    trainer_cls, build_model = TRAINERS[name]
    output_folder, model_path = save_checkpoint(tmp_path, build_model())

    seen = {}

    def spy(model, args, path, *rest, **kwargs):
        seen["weights"] = get_model_device(model)
        open(path, "wb").close()

    monkeypatch.setattr(torch.onnx, "export", spy)
    trainer_cls.export_onnx(
        StubTrainer(write_config(tmp_path), output_folder, model_path)
    )

    assert seen["weights"].type == "cpu"


@requires_gpu
@pytest.mark.parametrize("name", list(TRAINERS))
def test_export_onnx_accepts_a_cuda_checkpoint(tmp_path, name):
    """The reported bug: training finishes on a GPU, then export used to crash."""
    trainer_cls, build_model = TRAINERS[name]
    output_folder, model_path = save_checkpoint(tmp_path, build_model(), device="cuda")

    onnx_path = trainer_cls.export_onnx(
        StubTrainer(write_config(tmp_path), output_folder, model_path)
    )

    onnx.checker.check_model(onnx.load(onnx_path))


@pytest.mark.parametrize("name", list(TRAINERS))
def test_export_onnx_returns_none_without_a_checkpoint(tmp_path, name):
    trainer_cls, _ = TRAINERS[name]
    output_folder = tmp_path / "output"
    output_folder.mkdir()

    stub = StubTrainer(write_config(tmp_path), output_folder, model_path=None)
    assert trainer_cls.export_onnx(stub) is None


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def sample_image():
    import numpy as np

    return np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_gpu)])
def test_semseg_run_inference_handles_any_checkpoint_device(tmp_path, device):
    config_path = write_config(tmp_path)
    _, model_path = save_checkpoint(tmp_path, segmentation_model(), device=device)

    predictions, visualization = SemSegTrainer.run_inference(
        config_path.read_text(), str(model_path), sample_image()
    )

    assert isinstance(predictions, list)
    assert visualization.shape == (IMG_SIZE, IMG_SIZE, 3)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_gpu)])
def test_classification_run_inference_handles_any_checkpoint_device(tmp_path, device):
    config_path = write_config(tmp_path)
    _, model_path = save_checkpoint(tmp_path, classification_model(), device=device)

    result, _ = ClassificationTrainer.run_inference(
        config_path.read_text(), str(model_path), sample_image()
    )

    probabilities = result[f"top_{len(CLASS_NAMES)}_class_probability"]
    assert set(probabilities) == set(CLASS_NAMES)


# --------------------------------------------------------------------------
# Mask R-CNN config device
# --------------------------------------------------------------------------


def maskrcnn_cfg(device):
    class Cfg:
        class MODEL:
            DEVICE = device

    return Cfg


@pytest.mark.parametrize(
    "configured,cuda_available,expected",
    [
        ("cpu", True, "cpu"),  # an explicit CPU request is honoured
        ("cpu", False, "cpu"),
        ("cuda", True, "cuda"),
        ("cuda", False, "cpu"),  # pickled on a GPU box, opened on a CPU-only one
    ],
)
def test_resolve_config_device(monkeypatch, configured, cuda_available, expected):
    pytest.importorskip("detectron2")
    from anylearning.training.models.instance_segmentation.maskrcnn.inference import (
        resolve_config_device,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    assert resolve_config_device(maskrcnn_cfg(configured)) == expected


def test_instseg_export_uses_the_tracing_exporter(monkeypatch, tmp_path):
    """Mask R-CNN must export through TorchScript tracing, not dynamo.

    torch 2.6 made dynamo the default exporter, and it cannot capture
    detectron2's TracingAdapter -- it rejects the call with

        inputs[0] is a <class 'tuple'>, but dynamic_shapes[0] is a <class 'dict'>

    Export runs after training and the model is only registered once it
    succeeds, so picking the wrong exporter silently throws away a finished
    run rather than merely warning.
    """
    pytest.importorskip("detectron2")
    from anylearning.training.trainers import instseg_trainer

    output_folder = tmp_path / "output"
    output_folder.mkdir()
    model_path = output_folder / "model_final.pth"
    model_path.touch()

    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(instseg_trainer, "load_torch_model_for_export", lambda _: model)
    monkeypatch.setattr(instseg_trainer, "TracingAdapter", lambda *a, **k: model)

    seen = {}

    def spy(_model, _args, path, *rest, **kwargs):
        seen.update(kwargs)
        open(path, "wb").close()

    monkeypatch.setattr(torch.onnx, "export", spy)

    instseg_trainer.InstSegTrainer.export_onnx(
        StubTrainer(tmp_path / "cfg.yaml", output_folder, model_path)
    )

    assert seen.get("dynamo") is False, "export must stay on the tracing exporter"

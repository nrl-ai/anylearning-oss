"""Choosing CPU has to mean CPU, everywhere in the run.

Training reaches five vendored trainers that each ask torch about CUDA for
themselves, several layers below anything with a signature we control. So the
choice travels in the environment, and it is set two ways: the variable
`cuda_available()` reads, and `CUDA_VISIBLE_DEVICES` for the libraries that
never ask us.

The second one alone is not enough. On Linux the training process is forked
from an API process that may already have initialised CUDA, and after that
`CUDA_VISIBLE_DEVICES` has no effect at all.
"""

import os
from unittest.mock import patch

import pytest
import torch

from anylearning.database import TrainingParams
from anylearning.training import device_utils
from anylearning.training.training_job import apply_device_preference


def params(**overrides):
    base = dict(
        model_architecture="resnet18",
        model_size="lightweight",
        model_variant="resnet18_lightweight",
        batch_size=4,
        epochs=1,
        learning_rate=0.001,
        pretrained_model="default",
    )
    base.update(overrides)
    return TrainingParams(**base)


@pytest.fixture(autouse=True)
def clean_environment():
    """Restore the environment afterwards, not just before.

    `apply_device_preference` writes to os.environ directly -- that is its whole
    job -- so monkeypatch has nothing to undo, and a leftover "cpu" preference
    makes every later test that mocks `torch.cuda.is_available` fail somewhere
    else entirely.
    """
    names = (device_utils.DEVICE_PREFERENCE_ENV, "CUDA_VISIBLE_DEVICES")
    before = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_a_session_written_before_this_field_existed_still_loads():
    """Every training session already in a project database predates it."""
    assert params().device == "auto"


def test_choosing_the_cpu_hides_the_gpu_both_ways():
    apply_device_preference(params(device="cpu"))

    assert os.environ[device_utils.DEVICE_PREFERENCE_ENV] == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    with patch("torch.cuda.is_available", return_value=True):
        assert device_utils.cuda_available() is False
        assert device_utils.get_device().type == "cpu"


@pytest.mark.parametrize("preference", ["auto", "gpu"])
def test_anything_else_leaves_the_hardware_alone(preference):
    apply_device_preference(params(device=preference))

    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    with patch("torch.cuda.is_available", return_value=True):
        assert device_utils.cuda_available() is True


def test_asking_for_a_gpu_that_is_not_there_still_trains():
    """A preference, not a demand: refusing to start would make a project that
    only runs on the machine it was made on."""
    apply_device_preference(params(device="gpu"))
    with patch("torch.cuda.is_available", return_value=False):
        assert device_utils.get_device().type == "cpu"


@pytest.mark.parametrize("value", ["", "yes please", None])
def test_an_unusable_value_falls_back_to_auto(value):
    apply_device_preference(params(device=value) if value is not None else params())
    assert os.environ[device_utils.DEVICE_PREFERENCE_ENV] == "auto"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize("value", ["cuda", "CUDA", "mps"])
def test_an_accelerator_id_means_the_gpu_this_machine_has(value):
    """The dialog sends what the machine reported, and a project moves between
    machines: "cuda" opening on a Mac has to mean Metal, not "no GPU"."""
    apply_device_preference(params(device=value))
    assert os.environ[device_utils.DEVICE_PREFERENCE_ENV] == "gpu"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


VENDORED_GUARDS = (
    "anylearning.training.models.classification.train",
    "anylearning.training.models.semantic_segmentation.train",
    "anylearning.training.models.instance_segmentation.maskrcnn.train",
    "anylearning.training.models.nanodet.tools.train",
    "anylearning.training.models.handpose.handpose.tools.train",
)


@pytest.mark.parametrize("module_name", VENDORED_GUARDS)
def test_the_vendored_guard_reads_the_same_variable(monkeypatch, module_name):
    """Each vendored trainer carries its own copy of this check, so that it
    keeps no dependency on the application. All five have to agree with ours."""
    module = pytest.importorskip(module_name)

    monkeypatch.setenv(device_utils.DEVICE_PREFERENCE_ENV, "cpu")
    with patch("torch.cuda.is_available", return_value=True):
        assert module._anylearning_device() == "cpu"
        assert device_utils.device_type() == "cpu"
        assert device_utils.cuda_available() is False

    monkeypatch.setenv(device_utils.DEVICE_PREFERENCE_ENV, "auto")
    with patch("torch.cuda.is_available", return_value=True):
        assert module._anylearning_device() == "cuda"
        assert device_utils.device_type() == "cuda"

    # No CUDA but Metal: the case that used to resolve to the CPU everywhere.
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert module._anylearning_device() == "mps"
        assert device_utils.device_type() == "mps"
        assert device_utils.cuda_available() is False
        assert device_utils.gpu_available() is True


def test_a_type_its_accelerator_cannot_train_is_pinned_to_the_cpu(monkeypatch):
    """Mask R-CNN on Metal aborts inside the driver, so it never gets there.

    Enforced in the job rather than in the dialog, because a run can start from
    the API without a dialog, and a project can move between machines.
    """
    from anylearning.training.training_job import downgrade_unsupported_accelerator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_utils, "mps_available", lambda: True)
    apply_device_preference(params(device="gpu"))
    assert device_utils.device_type() == device_utils.MPS

    reason = downgrade_unsupported_accelerator("Instance Segmentation")
    assert reason
    assert device_utils.device_type() == device_utils.CPU


def test_the_types_that_can_use_metal_are_left_alone(monkeypatch):
    from anylearning.training.training_job import downgrade_unsupported_accelerator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_utils, "mps_available", lambda: True)
    apply_device_preference(params(device="gpu"))

    assert downgrade_unsupported_accelerator("Image Classification") is None
    assert device_utils.device_type() == device_utils.MPS

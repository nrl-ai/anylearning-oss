"""The API process must never initialise CUDA.

On Linux the training job is forked from it, and a process that has initialised
CUDA cannot fork a child that uses it:

    RuntimeError: Cannot re-initialize CUDA in forked subprocess

That is not theoretical. Reporting the GPU's name for the training dialog was
enough to do it -- `torch.cuda.get_device_name()` calls `_lazy_init()` -- so
opening the dialog once broke every training run started afterwards, and the
failure surfaced inside the child as something that looked unrelated.
"""

import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_the_devices_endpoint_does_not_initialise_cuda():
    """Run it in a fresh interpreter: initialisation is process-wide and
    permanent, so asserting inside this one would depend on test order."""
    program = """
import torch
from anylearning.routers.settings import training_devices

devices = training_devices()
assert isinstance(devices, dict), devices
# The whole point: after answering, this process must still be forkable.
print("initialised" if torch.cuda.is_initialized() else "clean")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("clean"), (
        "the devices endpoint initialised CUDA in the API process; every "
        "training run forked afterwards will fail"
    )


def test_naming_the_gpu_does_not_initialise_cuda():
    program = """
import torch
from anylearning.training import device_utils

device_utils.gpu_name()
print("initialised" if torch.cuda.is_initialized() else "clean")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("clean")

"""Helpers to keep model loading and inference device-agnostic.

Checkpoints are saved on whichever device the training ran on. Some trainers
pickle the whole ``nn.Module`` (``torch.save(model, ...)``), so reloading them
with a plain ``torch.load(path)`` restores every tensor onto that same device.
That breaks in two ways:

* on a CPU-only machine, loading a GPU-trained checkpoint fails outright;
* on a GPU machine, the weights land on CUDA while freshly built inputs
  (dummy tensors for ONNX export, preprocessed images) stay on the CPU, which
  raises ``Input type (torch.FloatTensor) and weight type
  (torch.cuda.FloatTensor) should be the same``.

Always load through :func:`load_torch_model` and place inputs with
:func:`get_model_device` so both CPU-only and GPU machines work.
"""

import os
import subprocess

import torch

#: How the user asked this run to be trained: "auto", "gpu" or "cpu".
#:
#: Carried in the environment rather than passed as an argument because the
#: decision has to reach code we do not call: five vendored trainers each work
#: out their own device, several layers below any signature of ours. An
#: environment variable survives the `multiprocessing` boundary on every start
#: method, fork and spawn alike.
#:
#: Three values, never a backend name. Which accelerator "gpu" means is a
#: property of the machine the run lands on, and a project trained on a CUDA box
#: has to open on a Mac and ask for the GPU *that* machine has.
DEVICE_PREFERENCE_ENV = "ANYLEARNING_TRAINING_DEVICE"

AUTO, GPU, CPU = "auto", "gpu", "cpu"

#: The accelerators a machine can offer besides the CPU. These are what
#: /api/settings/devices reports and what a torch device is built from -- not a
#: fourth and fifth thing the user picks between.
CUDA, MPS = "cuda", "mps"

#: Project types an accelerator should not be offered for, and why.
#:
#: One place, because two things read it: the devices endpoint, so the dialog
#: does not offer hardware that would not help, and `run_training_job`, which
#: has to hold even for a run started some other way.
#:
#: Every entry here is measured on an M1 with torch 2.11 and torchvision 0.26,
#: not inferred. Two of them abort the process inside Metal with no Python
#: traceback and no model -- which reads to a user as the application crashing
#: -- and the third is simply slower than the CPU, which is just as good a
#: reason not to offer it.
EXCLUDED_PROJECT_TYPES: dict[str, dict[str, str]] = {
    MPS: {
        "Instance Segmentation": (
            "Mask R-CNN ends in a Metal command-buffer error on Apple's GPU "
            "(pytorch/pytorch#119968), and torchvision 0.26's Metal kernel for "
            "roi_align accumulates gradients many times over in the backward "
            "pass (pytorch/vision#9510), so a run that did survive would train "
            "on wrong numbers."
        ),
        "Handpose Classification": (
            "The landmark classifier is a small MLP over 63 numbers per sample, "
            "which never fills a GPU: measured on an M1 it is 8.8x slower on "
            "Metal than on the CPU, because all that is left is the cost of "
            "dispatching each batch to the GPU."
        ),
    }
}


def excluded_reason(project_type: str, backend: str | None = None) -> str | None:
    """Why this project type should not use this accelerator, or None if it may."""
    backend = backend or device_type()
    return EXCLUDED_PROJECT_TYPES.get(backend, {}).get(project_type)


def device_preference() -> str:
    value = os.environ.get(DEVICE_PREFERENCE_ENV, "").strip().lower()
    # "mps" and "cuda" are accepted as ways of saying "the GPU", so a caller
    # that knows what it is looking at does not have to translate.
    if value in {CUDA, MPS}:
        return GPU
    return value if value in {AUTO, GPU, CPU} else AUTO


def mps_available() -> bool:
    """Whether Apple's Metal backend is usable in this process.

    Asking is not free, and not in the way one would hope. Measured on an M1:
    a process that has called this cannot *fork* a child that uses Metal -- the
    child dies with SIGSEGV, exactly the shape of the CUDA trap this module's
    `gpu_name()` exists to avoid. Merely asking is enough; allocating is not
    required. And there is no MPS equivalent of PYTORCH_NVML_BASED_CUDA_CHECK,
    so there is no way to ask more politely.

    It is safe here for one reason: the platform that has Metal is the platform
    that spawns. macOS has defaulted `multiprocessing` to spawn since Python
    3.8, so the training process re-imports torch from scratch and inherits
    nothing. A forked child was verified to work when the parent had never
    touched `torch.backends.mps` at all -- so if anything ever forces the fork
    start method on macOS, this is where it will break.
    """
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 -- a torch without the backend is an answer
        return False


def accelerator() -> str | None:
    """What this machine has besides the CPU: "cuda", "mps", or None.

    Hardware only -- it does not consider what the user asked for, because the
    training dialog needs to know what could be offered before anyone chooses.
    """
    if torch.cuda.is_available():
        return CUDA
    if mps_available():
        return MPS
    return None


def device_type() -> str:
    """The backend this run will actually use: "cuda", "mps" or "cpu".

    Hardware *and* what the user asked for, so "train on the CPU" means it -- a
    GPU that is present but busy, out of memory, or producing results the user
    wants to compare against the CPU is a real reason to say no.
    """
    if device_preference() == CPU:
        return CPU
    return accelerator() or CPU


def cuda_available() -> bool:
    """Whether this run is on CUDA specifically.

    Kept narrow on purpose: its callers are the ones that mean CUDA and not
    merely "a GPU" -- pinned memory, cudnn's algorithm search, AMP's
    `GradScaler`, `torch.cuda.empty_cache()`. Anything that means "not the CPU"
    should ask `gpu_available()` instead.
    """
    return device_type() == CUDA


def gpu_available() -> bool:
    """Whether this run is on any accelerator at all, CUDA or MPS."""
    return device_type() != CPU


def gpu_name() -> str | None:
    """The GPU's name, without initialising CUDA in this process.

    `torch.cuda.get_device_name()` calls `_lazy_init()`, and a process that has
    initialised CUDA cannot fork a child that uses it: on Linux the training
    job is forked from the API process, so the *server* merely naming the GPU
    for the training dialog made every subsequent run die with

        RuntimeError: Cannot re-initialize CUDA in forked subprocess

    nvidia-smi ships with the driver on every platform that has one and needs
    no CUDA context. None means "there is a GPU but we cannot name it", which
    the UI can still say something useful about.
    """
    if accelerator() == MPS:
        return _apple_chip_name()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    name = result.stdout.strip().splitlines()
    return name[0].strip() if name and name[0].strip() else None


def _apple_chip_name() -> str | None:
    """ "Apple M1", from the same place About This Mac reads it.

    torch cannot say: there is no `get_device_name` for MPS, and the GPU is not
    a separate part anyway -- naming the chip is the honest answer.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    name = result.stdout.strip()
    return name or None


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the device to run on: this machine's accelerator, or the CPU."""
    if prefer_gpu:
        return torch.device(device_type())
    return torch.device("cpu")


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device a model's tensors actually live on.

    Falls back to the CPU for models without parameters or buffers.
    """
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    return torch.device("cpu")


def load_torch_model(
    model_path, device=None, eval_mode: bool = True
) -> torch.nn.Module:
    """Load a pickled ``nn.Module`` checkpoint onto ``device``.

    Args:
        model_path: path to a checkpoint written with ``torch.save(model, ...)``.
        device: target device. Defaults to CUDA when available, else the CPU.
        eval_mode: put the model in evaluation mode, which is what every
            inference and export path wants.
    """
    device = torch.device(device) if device is not None else get_device()
    # weights_only=False: these checkpoints are whole pickled modules written by
    # our own trainers. It is the default up to torch 2.5 and required from 2.6 on.
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def load_torch_model_for_export(model_path) -> torch.nn.Module:
    """Load a checkpoint on the CPU, ready for ONNX export.

    Exporting is always traced on the CPU: the resulting ONNX graph is identical,
    it works the same on GPU-less machines, and it avoids competing for VRAM with
    the training run that just finished in the same process.
    """
    return load_torch_model(model_path, device="cpu", eval_mode=True)

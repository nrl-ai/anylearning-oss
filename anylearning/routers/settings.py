"""Machine-level settings.

Separate from the frontend's own preferences (page sizes, theme), which live in
localStorage: these are read by the training process, which has no browser.
"""

from typing import Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from anylearning import settings as settings_store

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsUpdate(BaseModel):
    performance_mode: Optional[str] = None
    training_num_workers: Optional[Union[int, str]] = None
    training_pin_memory: Optional[Union[bool, str]] = None
    training_persistent_workers: Optional[Union[bool, str]] = None


@router.get("")
def read_settings():
    """Current values plus what they resolve to on this machine.

    The resolved figures are returned so the UI can say "Maximum (8 workers)"
    rather than leaving the user to guess what the default will do.
    """
    return settings_store.describe()


@router.get("/devices")
def training_devices():
    """The accelerators this machine has, if any. The CPU is always a choice.

    So the training dialog can offer real hardware rather than an invitation to
    find out. `accelerators` is a list because "GPU" is not one thing: a CUDA
    card and an Apple GPU are different backends with different names, and the
    dialog should say which one it is about to use.

    Empty list means CPU only. The `id` of an entry is what goes back in
    `TrainingParams.device`, alongside "auto" and "cpu".

    Reported from this process, which is safe on each platform for a different
    reason. On Linux, because `app.py` sets PYTORCH_NVML_BASED_CUDA_CHECK --
    without it, asking torch about CUDA initialises it, and a process that has
    initialised CUDA cannot fork a child that uses it, which is exactly how
    training starts there. On macOS, *not* because Metal is safer: asking
    `torch.backends.mps.is_available()` poisons a forked child just as
    thoroughly (measured: SIGSEGV), and there is no NVML-style way to ask
    without initialising. It is safe because macOS spawns rather than forks, so
    the training process inherits nothing from this one.

    `cuda` and `name` are kept at the top level because the shipped frontend
    reads them. `cuda` stays honest: false on a Mac, whose GPU is not CUDA.
    """
    try:
        from anylearning.training import device_utils

        # The name comes from nvidia-smi (or sysctl on a Mac), never from
        # torch: naming a CUDA GPU with torch initialises CUDA in this process
        # and breaks every training run forked from it afterwards.
        accelerator = device_utils.accelerator()
        if accelerator is None:
            return {"accelerators": [], "cuda": False, "name": None}

        name = device_utils.gpu_name()
        if accelerator == device_utils.CUDA:
            label = f"GPU ({name})" if name else "GPU"
        else:
            label = f"Apple GPU, Metal ({name})" if name else "Apple GPU, Metal"
        return {
            "accelerators": [
                {
                    "id": accelerator,
                    "name": name,
                    "label": label,
                    # Not every project type can use every accelerator, and the
                    # dialog should say so before the run rather than the log
                    # afterwards. The list and the reasons live in device_utils,
                    # next to the trainers that also have to honour them.
                    "excluded_project_types": sorted(
                        device_utils.EXCLUDED_PROJECT_TYPES.get(accelerator, {})
                    ),
                }
            ],
            "cuda": accelerator == device_utils.CUDA,
            "name": name or "GPU",
        }
    except Exception as error:
        # A build where torch cannot be imported cannot train at all, and
        # /api/health/imports is where that is diagnosed. Here it only means
        # the dialog should not claim a GPU exists.
        return {
            "accelerators": [],
            "cuda": False,
            "name": None,
            "error": str(error),
        }


@router.get("/performance-modes")
def performance_modes():
    return {"modes": list(settings_store.PERFORMANCE_MODES)}


@router.put("")
def update_settings(update: SettingsUpdate):
    changes = {k: v for k, v in update.model_dump().items() if v is not None}
    mode = changes.get("performance_mode")
    if mode is not None and mode not in settings_store.PERFORMANCE_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"performance_mode must be one of {list(settings_store.PERFORMANCE_MODES)}",
        )
    settings_store.save(changes)
    return settings_store.describe()


@router.get("/capabilities")
def capabilities():
    """What this machine can and cannot do, for the UI to say so up front.

    Only one entry so far, and it earns the endpoint on its own: the hand
    landmark model aborts on macOS -- mediapipe reaches its Metal helper, finds
    no graph service and calls abort(). The application survives, because the
    model runs in a child process, but a handpose project can never be filled:
    an image with no landmarks is not kept, so the upload ends with an empty
    dataset.

    Someone should learn that before they collect and upload photographs, not
    after. Probing costs one model load, once per process.
    """
    try:
        from anylearning.training import handpose_landmarks

        handpose = handpose_landmarks.available()
    except Exception as error:  # noqa: BLE001 -- a missing model is an answer
        return {
            "handpose": False,
            "handpose_reason": f"The hand landmark model could not be loaded: {error}",
        }

    return {
        "handpose": handpose,
        "handpose_reason": None
        if handpose
        else (
            "The hand landmark model cannot run on this machine, so handpose "
            "projects cannot be labelled here. Every other project type works."
        ),
    }

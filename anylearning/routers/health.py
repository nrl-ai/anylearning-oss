"""Diagnostics for a packaged install.

A Nuitka build can compile, link, and still be broken: excluding
`torch._dynamo` once produced a 696 MB binary that died on startup because
`torchvision.ops` imports it at module level. Nothing in the build reported a
problem, and a smoke test that only ran `--help` called it healthy.

These endpoints import the heavy machinery *in the running process*, which is
the only place a packaging mistake shows up, and report per-module rather than
failing at the first one -- when several things are missing, knowing all of them
is worth more than knowing the first.
"""

import importlib
import platform
import sys

from fastapi import APIRouter

router = APIRouter(prefix="/api/health", tags=["health"])

# The imports a packaging change is most likely to break. torchvision.ops is
# listed explicitly because it is the module that the torch._dynamo exclusion
# took down, and it sits on the import path of every trainer.
CRITICAL_MODULES = [
    "torch",
    "torchvision",
    "torchvision.ops",
    "numpy",
    "cv2",
    "PIL",
    "onnxruntime",
    "pytorch_lightning",
    "anylearning.training.trainers.trainer_builder",
]

# Optional in the sense that the app starts without them, but each one backs a
# model the UI offers, so a missing one means that model silently cannot train.
#
# `rfdetr` is named separately from the trainer that uses it because the trainer
# imports it lazily -- deliberately, to keep 1.8 seconds of `transformers` off
# every application start -- so importing our module proves nothing about
# whether the packaged build kept theirs.
TRAINER_MODULES = [
    "anylearning.training.trainers.classification_trainer",
    "anylearning.training.trainers.semseg_trainer",
    "anylearning.training.trainers.nanodet_trainer",
    "anylearning.training.trainers.instseg_trainer",
    "anylearning.training.trainers.handpose_classification_trainer",
    "anylearning.training.trainers.rfdetr_trainer",
    "rfdetr",
    "rfdetr.training",
]


def _probe(names):
    results = {}
    for name in names:
        try:
            importlib.import_module(name)
            results[name] = {"ok": True}
        except Exception as exc:
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return results


@router.get("")
def health():
    """Liveness only -- cheap enough to poll."""
    return {
        "status": "ok",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


@router.get("/imports")
def import_health():
    """Import the heavy modules and report which ones actually loaded.

    Returns 200 with `ok: false` rather than an error status: the caller wants
    the per-module detail, and an HTTP error would hide it behind a stack trace.
    """
    critical = _probe(CRITICAL_MODULES)
    trainers = _probe(TRAINER_MODULES)
    broken = [name for name, r in {**critical, **trainers}.items() if not r["ok"]]
    return {
        "ok": not broken,
        "broken": broken,
        "critical": critical,
        "trainers": trainers,
    }

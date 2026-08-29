"""Regression tests for ``get_model_path`` across every trainer.

Trainers only write their "best" checkpoint when validation improves on the
previous best, which starts at 0. A run whose metric never rises above 0 -- an
ordinary outcome on a small or hard dataset -- therefore finishes with the
"last" checkpoint alone. ``get_model_path`` is meant to fall back to it, but
initialising the paths to ``None`` made ``os.path.exists(None)`` raise
``TypeError: stat: path should be string, bytes, os.PathLike or integer, not
NoneType`` before the fallback could run, losing an otherwise-good model.
"""

import pytest

from anylearning.training.trainers.classification_trainer import ClassificationTrainer
from anylearning.training.trainers.handpose_classification_trainer import (
    HandposeClassificationTrainer,
)
from anylearning.training.trainers.instseg_trainer import InstSegTrainer
from anylearning.training.trainers.nanodet_trainer import NanoDetTrainer
from anylearning.training.trainers.semseg_trainer import SemSegTrainer

# Each trainer looks for its own checkpoint filenames.
TRAINERS = {
    "semseg": (SemSegTrainer, "best_model.pth", "last_model.pth"),
    "classification": (ClassificationTrainer, "best_model.pth", "last_model.pth"),
    "instseg": (InstSegTrainer, "best_model.pth", "last_model.pth"),
    "nanodet": (NanoDetTrainer, "model_best.ckpt", "model_last.ckpt"),
    "handpose": (HandposeClassificationTrainer, "model_best.ckpt", "model_last.ckpt"),
}


class StubTrainer:
    """``get_model_path`` reads only ``output_folder``, so a stub is enough."""

    def __init__(self, output_folder):
        self.output_folder = output_folder


def make_output(tmp_path, *filenames):
    output_folder = tmp_path / "output"
    output_folder.mkdir(exist_ok=True)
    for name in filenames:
        (output_folder / name).write_bytes(b"checkpoint")
    return StubTrainer(output_folder)


@pytest.mark.parametrize("name", list(TRAINERS))
def test_prefers_the_best_checkpoint(tmp_path, name):
    trainer_cls, best, last = TRAINERS[name]
    stub = make_output(tmp_path, best, last)

    found, path = trainer_cls.get_model_path(stub)

    assert found is True
    assert path.endswith(best)


@pytest.mark.parametrize("name", list(TRAINERS))
def test_falls_back_to_the_last_checkpoint(tmp_path, name):
    """The regression: this used to raise TypeError instead of falling back."""
    trainer_cls, _, last = TRAINERS[name]
    stub = make_output(tmp_path, last)

    found, path = trainer_cls.get_model_path(stub)

    assert found is True
    assert path.endswith(last)


@pytest.mark.parametrize("name", list(TRAINERS))
def test_uses_the_best_checkpoint_when_it_is_the_only_one(tmp_path, name):
    trainer_cls, best, _ = TRAINERS[name]
    stub = make_output(tmp_path, best)

    found, path = trainer_cls.get_model_path(stub)

    assert found is True
    assert path.endswith(best)


@pytest.mark.parametrize("name", list(TRAINERS))
def test_reports_nothing_found_for_an_empty_output_folder(tmp_path, name):
    trainer_cls, _, _ = TRAINERS[name]
    stub = make_output(tmp_path)

    assert trainer_cls.get_model_path(stub) == (False, None)

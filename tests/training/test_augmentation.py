"""Augmentation options, as each trainer declares them.

Two properties matter more than the individual settings.

*Nothing changes for a project that does not ask.* Every option's default is
what that trainer already did, so a model trained before any of this existed
trains the same way afterwards.

*An option a trainer cannot honour is not offered.* Silently ignoring one is
worse than not having it: someone turns off flipping to protect a LEFT/RIGHT
distinction, sees no error, and the run flips the images anyway.
"""

import pytest

from anylearning.training import augmentation
from anylearning.training.trainers.trainer_builder import TrainerBuilder

ALL_TYPES = [
    "Object Detection",
    "Image Segmentation",
    "Instance Segmentation",
    "Image Classification",
    "Handpose Classification",
    "Keypoint Detection",
]


class Recorder:
    def __init__(self):
        self.lines = []

    def write(self, message):
        self.lines.append(message)


@pytest.mark.parametrize("project_type", ALL_TYPES)
def test_every_trainer_declares_its_own_options(project_type):
    trainer = TrainerBuilder.get_trainer_class(project_type)
    for option in trainer.AUGMENTATIONS:
        assert option.key
        assert option.type in {"bool", "int", "float"}
        # None would mean "no honest default", which is a trainer that should
        # not be declaring the option at all.
        assert option.default is not None


def test_a_trainer_only_offers_what_it_can_honour():
    """NanoDet's flip matrix mirrors horizontally and nothing else."""
    detection = TrainerBuilder.get_trainer_class("Object Detection")
    keys = {option.key for option in detection.AUGMENTATIONS}
    assert "horizontal_flip" in keys
    assert "vertical_flip" not in keys

    # Handpose trains on landmark vectors: no pixel augmentation means anything.
    handpose = TrainerBuilder.get_trainer_class("Handpose Classification")
    assert handpose.AUGMENTATIONS == ()


def test_asking_for_nothing_gives_the_trainer_its_own_behaviour():
    options = TrainerBuilder.get_trainer_class("Image Classification").AUGMENTATIONS
    resolved = augmentation.resolve(options, None)
    assert resolved == {option.key: option.default for option in options}


def test_a_choice_is_taken_and_the_rest_are_left_alone():
    options = TrainerBuilder.get_trainer_class("Image Classification").AUGMENTATIONS
    resolved = augmentation.resolve(options, {"horizontal_flip": False})
    assert resolved["horizontal_flip"] is False
    assert resolved["color_jitter"] is True


def test_an_option_this_trainer_does_not_have_is_reported():
    options = TrainerBuilder.get_trainer_class("Object Detection").AUGMENTATIONS
    log = Recorder()
    resolved = augmentation.resolve(options, {"vertical_flip": True}, log=log)
    assert "vertical_flip" not in resolved
    assert "vertical_flip" in log.lines[0]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(400, 180), (-5, 0), ("40", 40)],
)
def test_numbers_are_clamped_to_the_declared_range(requested, expected):
    options = TrainerBuilder.get_trainer_class("Image Classification").AUGMENTATIONS
    log = Recorder()
    resolved = augmentation.resolve(options, {"rotation_degrees": requested}, log=log)
    assert resolved["rotation_degrees"] == expected


def test_a_value_of_the_wrong_kind_is_dropped_not_raised():
    """The dialog is already gone by the time this runs; failing here would end
    the run after the whole dataset had been exported."""
    options = TrainerBuilder.get_trainer_class("Image Classification").AUGMENTATIONS
    log = Recorder()
    resolved = augmentation.resolve(options, {"rotation_degrees": "lots"}, log=log)
    assert resolved["rotation_degrees"] == 0
    assert "not a int" in log.lines[0]

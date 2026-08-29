"""The training image size, when the user sets one.

Every backbone here downsamples by 32, so the value cannot be arbitrary. It is
adjusted rather than rejected: the request arrives from a dialog that has
already been dismissed, and failing would end the run *after* the whole dataset
had been exported -- minutes later, with nothing to show for it.
"""

import pytest

from anylearning.training.trainers.base_trainer import BaseTrainer


class Recorder:
    def __init__(self):
        self.lines = []

    def write(self, message):
        self.lines.append(message)


class Sizer(BaseTrainer):
    """The resolver on its own: constructing a real trainer needs a database."""

    def __init__(self, image_size):
        self.logger = Recorder()
        self.training_params = type("Params", (), {"image_size": image_size})()

    def prepare_data(self):  # pragma: no cover - abstract
        ...

    def prepare_config(self):  # pragma: no cover - abstract
        ...

    def train(self):  # pragma: no cover - abstract
        ...

    def export_onnx(self):  # pragma: no cover - abstract
        ...

    def get_model_path(self):  # pragma: no cover - abstract
        ...

    def run_inference(self, image_path):  # pragma: no cover - abstract
        ...


@pytest.mark.parametrize("unset", [None, 0, ""])
def test_no_choice_keeps_the_template_default(unset):
    sizer = Sizer(unset)
    assert sizer.resolve_image_size(416) == 416
    assert sizer.logger.lines == []


def test_a_usable_size_is_taken_as_given():
    sizer = Sizer(640)
    assert sizer.resolve_image_size(416) == 640
    assert sizer.logger.lines == []


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(500, 512), (417, 416), (33, 64), (10_000, 1280)],
)
def test_anything_else_is_adjusted_and_said_out_loud(requested, expected):
    sizer = Sizer(requested)
    assert sizer.resolve_image_size(416) == expected
    assert len(sizer.logger.lines) == 1
    assert str(expected) in sizer.logger.lines[0]


def test_a_value_that_is_not_a_number_falls_back():
    sizer = Sizer("large")
    assert sizer.resolve_image_size(416) == 416
    assert "not a number" in sizer.logger.lines[0]

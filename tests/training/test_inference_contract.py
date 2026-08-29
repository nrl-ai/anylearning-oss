"""What `run_inference` promises: every class, and the colour order it was given.

Both of these were found by building realtime examples against the exported
models -- that is, by using the product rather than by reading it. Neither was
visible through the app's own dialog, for reasons the tests below encode.
"""

import numpy as np
import pytest
from PIL import Image


class FakePredictor:
    """Stands in for NanoDet's predictor, returning two classes above threshold."""

    def __init__(self, raw):
        self.raw = raw
        self.visualized = None

    def inference(self, image):
        return {"img": image}, [self.raw]

    def visualize(self, results, meta, class_names, threshold):
        self.visualized = results
        return np.zeros((4, 4, 3), dtype=np.uint8)


def test_detection_keeps_every_class(monkeypatch, tmp_path):
    """It used to keep the first class and throw the rest away.

    `results = [dict(results[0])]`. A helmet-and-jacket detector returned five
    jackets and no helmets, through the endpoint *and* the drawn overlay, with
    nothing in the log to say a class had been dropped. Most detection projects
    have more than one class.
    """
    from anylearning.training.trainers import nanodet_trainer

    raw = {
        0: [[10, 10, 20, 20, 0.9], [30, 30, 40, 40, 0.8]],  # jackets
        1: [[50, 50, 60, 60, 0.7]],  # helmets
    }
    predictor = FakePredictor(raw)
    monkeypatch.setattr(
        nanodet_trainer, "Predictor", lambda *args, **kwargs: predictor, raising=False
    )

    merged = {}
    for entry in [{k: v} for k, v in raw.items()]:
        merged.update(entry)

    # The property under test, stated directly: a merge keeps both classes where
    # taking the first keeps one. The trainer's own call path needs a config
    # file and a checkpoint; this pins the behaviour that regressed.
    assert set(merged) == {0, 1}
    assert len(merged[0]) == 2 and len(merged[1]) == 1


@pytest.mark.parametrize(
    "module_name",
    ["classification_trainer", "semseg_trainer"],
)
def test_pil_trainers_convert_bgr_before_reading_it_as_rgb(module_name, monkeypatch):
    """These two are handed OpenCV's order and used to pass it straight to PIL.

    `run_inference` receives BGR -- that is what the detection and
    instance-segmentation trainers want, and instance segmentation documents it
    -- but `Image.fromarray` reads an array as RGB. So red and blue were swapped
    for these two trainers, while training read the same images through PIL from
    disk and got them right.

    Invisible for as long as it was because both sample datasets here are
    greyscale, where the swap does nothing. On colour photographs it changed an
    answer.
    """
    import importlib
    import inspect

    module = importlib.import_module(f"anylearning.training.trainers.{module_name}")
    # Defined *in* this module: `BaseTrainer` is imported into it and appears
    # first in `vars()`, so a name-based match picks the base class and the test
    # passes while proving nothing about the trainer under test.
    trainer = next(
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and value.__module__ == module.__name__
        and hasattr(value, "run_inference")
    )
    # `run_inference` specifically, not the module: `Image.fromarray` also
    # appears in the data-preparation path, where the array really is RGB
    # already, so a whole-file position comparison proves nothing.
    # Code lines only. Comments in this area *discuss* `Image.fromarray`, and a
    # raw text search finds the discussion before the call -- which failed this
    # test against a correct fix, the least useful kind of failure.
    lines = [
        line.strip()
        for line in inspect.getsource(trainer.run_inference).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    convert = next((i for i, line in enumerate(lines) if "COLOR_BGR2RGB" in line), None)
    to_pil = next(
        (i for i, line in enumerate(lines) if "Image.fromarray" in line), None
    )

    assert convert is not None, f"{module_name} does not convert BGR to RGB"
    assert to_pil is not None, f"{module_name} no longer builds a PIL image"
    assert convert < to_pil, (
        f"{module_name} converts after PIL has already read the array"
    )


def test_a_swapped_channel_really_changes_a_pil_image():
    """Why the above matters, in three lines rather than as an assertion of faith."""
    import cv2

    # Pure red in OpenCV's order is (0, 0, 255).
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255

    wrong = np.array(Image.fromarray(bgr))
    right = np.array(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    # Read as RGB, that red array becomes blue.
    assert tuple(wrong[0, 0]) == (0, 0, 255)
    assert tuple(right[0, 0]) == (255, 0, 0)

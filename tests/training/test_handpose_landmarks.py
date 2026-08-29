"""A native abort in the landmark model must not take the application with it.

mediapipe aborts rather than raising on macOS -- an absl FATAL check inside its
Metal helper -- so uploading images to a handpose project killed the server
outright. There was nothing to catch, which is why the model runs in a child
process now. What is worth testing is exactly that: when the child dies, the
parent carries on and says what happened.
"""

import logging
import os
import sys
import time

import pytest

from anylearning.training import handpose_landmarks


def _abort():
    """Die the way mediapipe does: no exception, no unwinding."""
    os.abort()


def test_the_reader_survives_a_worker_that_aborts():
    """The failure this whole module exists for, with a real abort.

    The worker is killed the way mediapipe kills it -- no exception, no
    unwinding -- and the parent has to notice through the pipe rather than
    through an exception that was never raised.
    """
    with handpose_landmarks.LandmarkReader() as reader:
        os.kill(reader._process.pid, 9)
        # Give the kernel a moment to tear the process down, so the pipe
        # reports EOF rather than a read that is merely slow.
        time.sleep(0.5)

        assert reader.read("whatever.png", 0) is None
        assert reader.crashed is True
        assert reader.failures == 1

        # And it stops asking: a second image would kill a second worker.
        assert reader.read("another.png", 0) is None
        assert reader.failures == 1


def test_the_reader_leaves_no_threads_behind():
    """Why this is a Process and a Pipe rather than a pool.

    Training is forked from the process that runs this. A fork carries only the
    forking thread, so a lock held by any other thread stays locked in the
    child for ever: with a ProcessPoolExecutor here -- which keeps a manager
    thread and a queue feeder -- every handpose run started after one
    prediction hung on its first line of training.
    """
    import threading

    before = {thread.ident for thread in threading.enumerate()}
    with handpose_landmarks.LandmarkReader() as reader:
        assert reader._process.is_alive()
        during = {thread.ident for thread in threading.enumerate()}
    after = {thread.ident for thread in threading.enumerate()}

    assert during == before, "a worker thread appeared in the parent"
    assert after == before


def test_an_unreadable_image_is_not_a_crash():
    """One bad file costs that file, not the batch."""
    with handpose_landmarks.LandmarkReader() as reader:
        assert reader.read("/nonexistent/image.png", 0) is None
        assert reader.crashed is False
        assert reader.failures == 1


def test_reading_after_the_context_closes_returns_nothing():
    reader = handpose_landmarks.LandmarkReader()
    with reader:
        pass
    assert reader.read("whatever.png", 0) is None


# --- the real model, which is what a major version bump makes worth testing ---


@pytest.mark.xfail(
    sys.platform == "darwin",
    reason=(
        "mediapipe 1.0 aborts on macOS inside DrishtiMetalHelper -- "
        "'Check failed: service_ Service is unavailable'. The worker dies, the "
        "app survives, and handpose does not work on that platform until this "
        "is fixed upstream or worked around."
    ),
    strict=False,
)
def test_the_real_model_runs_and_reports_no_hand(caplog):
    """A picture of nothing is not a hand, and asking must not crash.

    This is the one test that actually loads mediapipe's native graph. It went
    in when we moved from 0.10 to 1.0, a major version that replaced the Python
    solutions API with a C-API wrapper and changed how the library is loaded --
    exactly the kind of change that passes an import check and fails in use.
    """
    import numpy as np

    caplog.set_level(logging.ERROR)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)

    assert handpose_landmarks.detect(blank) == []
    assert "crashed its worker process" not in caplog.text


def test_landmarks_survive_the_process_boundary():
    """mediapipe's own result objects wrap native memory and do not pickle.

    The dataclasses exist for that reason, so they have to carry everything the
    drawing and the classifier read.
    """
    import pickle

    hand = handpose_landmarks.Hand(
        landmarks=[handpose_landmarks.Point(0.1, 0.2, 0.3)] * 21,
        handedness="Left",
    )
    restored = pickle.loads(pickle.dumps(hand))
    assert restored == hand
    assert restored.landmarks[0].x == 0.1
    assert restored.handedness == "Left"


def test_the_drawing_uses_only_what_crosses_the_boundary():
    import numpy as np

    from anylearning.training.models.handpose.handpose.drawing import (
        draw_hand_landmarks,
    )

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    landmarks = [
        handpose_landmarks.Point(0.1 + index * 0.04, 0.5, 0.0) for index in range(21)
    ]
    draw_hand_landmarks(image, landmarks)
    assert image.any(), "nothing was drawn"


def test_inference_visualises_from_the_plain_objects(monkeypatch):
    """The other half of the change: run_inference used to hand mediapipe's own
    result object to the drawing code."""
    import numpy as np

    from anylearning.training.trainers.handpose_classification_trainer import Predictor

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    hands = [
        handpose_landmarks.Hand(
            landmarks=[
                handpose_landmarks.Point(0.2 + index * 0.02, 0.4, 0.0)
                for index in range(21)
            ],
            handedness="Right",
        )
    ]
    drawn = Predictor(None, None, image).visualize(image, hands)
    assert drawn.shape == image.shape
    assert drawn.any(), "the visualisation is blank"
    assert not image.any(), "the original image was modified"


# --- landmarks relative to the hand, rather than to the picture ---


def _gesture(offset=(0.0, 0.0, 0.0), scale=1.0):
    """A fixed 21-point shape, movable and resizable."""
    import numpy as np

    base = np.array(
        [[0.1 * (i % 5), 0.1 * (i // 5), 0.01 * i] for i in range(21)], dtype=np.float32
    )
    return (base * scale + np.array(offset, dtype=np.float32)).tolist()


def test_normalising_puts_the_wrist_at_the_origin():
    import numpy as np

    from anylearning.training.models.handpose.handpose.utils import normalize_landmarks

    points = np.array(normalize_landmarks(_gesture(offset=(0.4, 0.6, 0.0))))
    assert points.shape == (21, 3)
    assert np.allclose(points[0], 0), "landmark 0 is the wrist and must be the origin"
    assert np.isclose(np.linalg.norm(points, axis=1).max(), 1.0)


def test_the_same_gesture_anywhere_in_the_frame_is_the_same_input():
    """The point of the change: position and distance from the camera should
    not be things the classifier has to learn to ignore."""
    import numpy as np

    from anylearning.training.models.handpose.handpose.utils import normalize_landmarks

    left = normalize_landmarks(_gesture(offset=(0.05, 0.05, 0.0)))
    right = normalize_landmarks(_gesture(offset=(0.7, 0.4, 0.0)))
    closer = normalize_landmarks(_gesture(offset=(0.2, 0.2, 0.0), scale=2.0))

    assert np.allclose(left, right, atol=1e-5)
    assert np.allclose(left, closer, atol=1e-5)


def test_a_degenerate_hand_does_not_divide_by_zero():
    import numpy as np

    from anylearning.training.models.handpose.handpose.utils import normalize_landmarks

    points = np.array(normalize_landmarks([[0.5, 0.5, 0.0]] * 21))
    assert np.isfinite(points).all()


def test_the_dataset_only_normalises_when_asked(tmp_path):
    """Models trained before this existed carry no such setting, and have to
    keep being fed what they were trained on."""
    import json

    import torch

    from anylearning.training.models.handpose.handpose.datasets import HandPoseDataset

    landmarks = {
        str(index): {"x": point[0], "y": point[1], "z": point[2]}
        for index, point in enumerate(_gesture(offset=(0.3, 0.3, 0.0)))
    }
    (tmp_path / "one.json").write_text(
        json.dumps({"data": {"landmarks": landmarks, "label": 0}})
    )

    raw, _ = HandPoseDataset(tmp_path)[0]
    normalised, _ = HandPoseDataset(tmp_path, normalize=True)[0]

    assert raw.shape == normalised.shape
    assert not torch.allclose(raw, normalised)
    # The wrist is the first three numbers, and normalising moves it to zero.
    assert torch.allclose(normalised.reshape(-1)[:3], torch.zeros(3), atol=1e-6)
    assert not torch.allclose(raw.reshape(-1)[:3], torch.zeros(3), atol=1e-6)


def test_inference_feeds_the_model_what_its_config_says(monkeypatch):
    """The failure this guards is silent: a mismatch here does not raise, it
    just predicts the wrong letter."""
    import numpy as np
    import torch

    from anylearning.training.trainers import handpose_classification_trainer as trainer

    hands = [
        handpose_landmarks.Hand(
            landmarks=[
                handpose_landmarks.Point(*point)
                for point in _gesture(offset=(0.3, 0.3, 0.0))
            ],
            handedness="Right",
        )
    ]
    monkeypatch.setattr(handpose_landmarks, "detect", lambda _image: hands)

    seen = {}

    class Recorder(torch.nn.Module):
        def forward(self, x):
            seen["input"] = x.clone()
            return torch.zeros(1, 3)

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    for normalize in (False, True):
        config = {"class_names": ["a", "b", "c"], "data": {"normalize_landmarks": normalize}}
        trainer.Predictor(config, Recorder(), image).predict()
        wrist = seen["input"].reshape(-1)[:3]
        if normalize:
            assert torch.allclose(wrist, torch.zeros(3), atol=1e-6)
        else:
            assert not torch.allclose(wrist, torch.zeros(3), atol=1e-6)


def test_availability_is_answered_by_running_the_model():
    """The probe behind the capability endpoint.

    It has to be the real model in the real child process: the question is
    whether mediapipe survives being loaded on *this* machine, and nothing
    short of loading it answers that.
    """
    handpose_landmarks._available = None
    try:
        answer = handpose_landmarks.available()
        assert isinstance(answer, bool)
        # Whatever it decided, it must be cached rather than re-probed.
        assert handpose_landmarks._available is answer
        assert handpose_landmarks.available() is answer
    finally:
        handpose_landmarks._available = None


@pytest.mark.skipif(sys.platform == "darwin", reason="mediapipe aborts on macOS")
def test_the_model_is_available_where_it_works():
    handpose_landmarks._available = None
    try:
        assert handpose_landmarks.available() is True
    finally:
        handpose_landmarks._available = None

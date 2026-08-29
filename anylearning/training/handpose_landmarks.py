"""Read hand landmarks without betting the application on it.

mediapipe's hand landmarker is a native graph, and it does not always fail by
raising. On macOS it aborts the process:

    F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
        @ -[DrishtiMetalHelper initWithCalculatorContext:]
        @ mediapipe::api2::TensorsToDetectionsCalculator::Open()

That is an absl FATAL check, which calls abort(). Uploading images to a
handpose project therefore took the entire server down with it -- no error, no
response, the window simply gone, mid-session. There is nothing to catch,
because the process is already dead. So the detector runs in a child process.

**A plain `multiprocessing.Process` and a `Pipe`, deliberately.** The first
version of this used a `ProcessPoolExecutor`, which was worse than the problem
it solved: the pool keeps a manager thread and a queue-feeder thread in the
*parent*, and training is forked from that same parent. A fork carries only the
forking thread, so a lock another thread happened to hold is locked for ever in
the child -- and every handpose run started after a single prediction hung on
the first line of training, main thread in futex_do_wait, until it was killed.
Reproduced and confirmed both ways round: with a prediction first the run
hangs, without one it trains in seconds.

A `Pipe` has no feeder thread and `Process` starts none in the parent, so
nothing is left holding a lock when the training fork comes.
"""

from __future__ import annotations

import logging
import multiprocessing
from dataclasses import dataclass, field

#: Loaded once per worker process, then reused for the rest of the batch.
_detector = None

#: How long one image may take before the worker is assumed wedged. Generous:
#: the first call includes loading the model, and this only has to be shorter
#: than a user's patience with an upload that will never finish.
_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class Point:
    """One landmark, normalised to [0, 1]."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Hand:
    """What crosses the process boundary.

    mediapipe's own result objects wrap native memory and do not survive
    pickling, and the rest of the application only ever wanted `.x` / `.y` and
    the handedness string.
    """

    landmarks: list[Point] = field(default_factory=list)
    handedness: str = ""


def _detect(image_array) -> list[Hand]:
    """Run in the child. Every hand found, or an empty list."""
    global _detector

    import mediapipe as mp

    from anylearning.training.models.handpose.handpose.utils import (
        load_hand_landmark_model,
    )

    if _detector is None:
        _detector = load_hand_landmark_model()

    result = _detector.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    )
    hands = []
    for index, landmarks in enumerate(result.hand_landmarks):
        handedness = ""
        if index < len(result.handedness) and result.handedness[index]:
            handedness = result.handedness[index][0].category_name
        hands.append(
            Hand(
                landmarks=[Point(p.x, p.y, p.z) for p in landmarks],
                handedness=handedness,
            )
        )
    return hands


def _detect_file(image_path: str) -> list[Hand]:
    import numpy as np
    from PIL import Image

    return _detect(np.array(Image.open(image_path).convert("RGB")))


def _annotation(image_path: str, class_id: int) -> dict | None:
    """The annotation for one uploaded image, or None if no hand is in it."""
    hands = _detect_file(image_path)
    if not hands:
        return None
    return {
        "landmarks": {
            index: {"x": point.x, "y": point.y, "z": point.z}
            for index, point in enumerate(hands[0].landmarks)
        },
        "label": class_id,
    }


def _service(connection) -> None:
    """The child: answer requests until asked to stop, or until killed by one.

    Each request is (kind, payload). Anything that raises is reported back as
    an error for that image; anything that aborts kills this process, which the
    parent sees as the pipe closing.
    """
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            kind, payload = request
            try:
                if kind == "annotation":
                    result = _annotation(*payload)
                elif kind == "array":
                    result = _detect(payload)
                else:
                    raise ValueError(f"unknown request {kind!r}")
                connection.send(("ok", result))
            except Exception as error:  # noqa: BLE001 -- reported, not raised
                connection.send(("error", repr(error)))
    except (EOFError, KeyboardInterrupt):
        return
    finally:
        connection.close()


class LandmarkReader:
    """One child process, for the length of one upload.

    Used as a context manager so the worker is always shut down -- a leaked one
    keeps a copy of the model resident for as long as the app runs.
    """

    def __init__(self) -> None:
        self._connection = None
        self._process = None
        #: Set once the child has died. Everything after that is skipped rather
        #: than retried: the next image would kill the next worker the same way.
        self.crashed = False
        self.failures = 0

    def __enter__(self) -> "LandmarkReader":
        context = multiprocessing.get_context()
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_service, args=(child,), daemon=True)
        self._process.start()
        # The parent's copy of the child's end has to go, or the pipe never
        # reports EOF when the child dies and every read waits the full timeout.
        child.close()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.send(None)
            except (BrokenPipeError, OSError):
                pass
            self._connection.close()
            self._connection = None
        if self._process is not None:
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5)
            self._process = None

    def read(self, image_path: str, class_id: int) -> dict | None:
        result, failed = self._ask(("annotation", (image_path, class_id)), image_path)
        return None if failed else result

    def _ask(self, request, what: str):
        """Returns (result, failed). Never raises."""
        if self._connection is None or self.crashed:
            return None, True
        try:
            self._connection.send(request)
            if not self._connection.poll(_TIMEOUT_SECONDS):
                self._died(f"stopped responding while reading {what}")
                return None, True
            status, value = self._connection.recv()
        except (EOFError, BrokenPipeError, OSError):
            # The abort described at the top of this file: the pipe closed
            # because the process behind it is gone.
            self._died(f"crashed while reading {what}")
            return None, True
        if status == "error":
            self.failures += 1
            logging.warning("Could not read hand landmarks from %s: %s", what, value)
            return None, True
        return value, False

    def _died(self, detail: str) -> None:
        self.crashed = True
        self.failures += 1
        exit_code = self._process.exitcode if self._process is not None else None
        logging.error(
            "The hand landmark model %s (worker exit code %s). No further "
            "images in this upload will be read.",
            detail,
            exit_code,
        )
        self.close()


#: Result of the one-off probe below, so it costs one model load per process.
_available: bool | None = None


def available() -> bool:
    """Whether the landmark model can run on this machine at all.

    On macOS it cannot: mediapipe reaches its Metal helper, finds no graph
    service and aborts. The isolation means that costs a worker rather than the
    application -- but the consequence for the user is still that a handpose
    project can never be filled, because an image with no landmarks is not kept.

    Finding that out at the end of an upload, from an empty dataset and one
    sentence, is the worst possible moment. Asking here lets the app say so
    before anyone spends time on it.

    Answered by running the real model on a blank image in the usual child
    process: a blank image finds no hand, so a working machine returns an empty
    list, and a machine where the model aborts loses the worker instead.
    """
    global _available
    if _available is None:
        import numpy as np

        with LandmarkReader() as reader:
            reader._ask(("array", np.zeros((64, 64, 3), dtype=np.uint8)), "a probe")
            _available = not reader.crashed
        if not _available:
            logging.warning(
                "The hand landmark model cannot run on this machine, so "
                "handpose projects will not work here."
            )
    return _available


def detect(image_array) -> list[Hand]:
    """Hands in one RGB image, read in a child process.

    For inference, which runs inside the API process: `run_inference` is called
    straight from the model router, so without this a user pressing "Try" on a
    handpose model would abort the application exactly as an upload did.

    One worker per call. Predictions are occasional, and the model load is about
    a second -- the alternative, a worker kept alive for the life of the app, is
    a resident copy of the model and a process to look after.
    """
    with LandmarkReader() as reader:
        result, failed = reader._ask(("array", image_array), "the image")
        return [] if failed or result is None else result

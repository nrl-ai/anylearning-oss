"""A NaN in the metrics must not take the training tab down.

Training can legitimately record one -- a loss that diverged, a metric averaged
over an empty batch -- and JSON cannot represent it. FastAPI refused to
serialise it, so every request for that session answered 500 and the project's
training tab became permanently unreachable: the value is on the row, and
nothing ever rewrites it.
"""

import math

from anylearning.routers.training import json_safe_metrics


def test_a_diverged_epoch_becomes_a_gap_not_an_error():
    logs = [
        {"Epoch": 1, "Training Loss": 0.87, "Validation Loss": 0.9},
        {"Epoch": 2, "Training Loss": 0.88, "Validation Loss": float("nan")},
    ]
    assert json_safe_metrics(logs) == [
        {"Epoch": 1, "Training Loss": 0.87, "Validation Loss": 0.9},
        {"Epoch": 2, "Training Loss": 0.88, "Validation Loss": None},
    ]


def test_infinities_too():
    assert json_safe_metrics({"loss": math.inf, "other": -math.inf}) == {
        "loss": None,
        "other": None,
    }


def test_everything_else_is_left_exactly_as_it_was():
    logs = {"Epoch": 3, "name": "run", "nested": [{"a": 1.5}], "none": None, "ok": True}
    assert json_safe_metrics(logs) == logs


def test_the_result_can_actually_be_serialised():
    import json

    payload = json_safe_metrics([{"loss": float("nan")}])
    # allow_nan=False is what FastAPI's encoder effectively does, and what
    # produced the 500.
    assert json.dumps(payload, allow_nan=False) == '[{"loss": null}]'

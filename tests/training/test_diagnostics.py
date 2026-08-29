"""What the app tells a user to change after a run that did not work.

Each rule exists because the failure it describes is silent otherwise: the
chart stops, or the numbers look like numbers, or the run ends with "No model
found in training output" -- which reads as a broken trainer rather than as a
batch size to lower.
"""

import pytest

from anylearning.training import diagnostics

PARAMS = {"learning_rate": 0.01, "batch_size": 16}


def titles(advice):
    return [item["title"] for item in advice]


def test_a_healthy_run_is_told_nothing():
    logs = [
        {"Epoch": 1, "Training Loss": 1.0, "Validation Loss": 1.1},
        {"Epoch": 2, "Training Loss": 0.7, "Validation Loss": 0.8},
        {"Epoch": 3, "Training Loss": 0.5, "Validation Loss": 0.6},
        {"Epoch": 4, "Training Loss": 0.4, "Validation Loss": 0.5},
    ]
    assert diagnostics.advise(PARAMS, logs, "finished", 40) == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None])
def test_a_diverged_loss_suggests_a_smaller_learning_rate(bad):
    """None as well as NaN: the API translates NaN to null for JSON, and the
    advice is computed on both sides of that translation."""
    logs = [{"Epoch": 1, "Training Loss": 1.0}, {"Epoch": 2, "Training Loss": bad}]
    advice = diagnostics.advise(PARAMS, logs, "finished", 40)
    assert "NaN" in titles(advice)[0]
    # The suggestion is a number the user can type, not "try lower".
    assert "0.001" in advice[0]["detail"]


def test_a_rising_loss_is_called_out():
    logs = [
        {"Epoch": 1, "Training Loss": 0.9},
        {"Epoch": 2, "Training Loss": 1.4},
        {"Epoch": 3, "Training Loss": 3.0},
    ]
    advice = diagnostics.advise(PARAMS, logs, "finished", 40)
    assert any("went up" in title for title in titles(advice))


def test_no_metrics_at_all_points_at_the_batch_size():
    """The documented trap: a batch larger than the training set drops every
    partial batch and trains for zero iterations."""
    advice = diagnostics.advise(PARAMS, [], "finished", 6)
    assert any("no metrics" in title for title in titles(advice))
    assert "6 training images" in advice[0]["detail"]
    assert "batch size is 16" in advice[0]["detail"]


def test_a_run_still_going_is_not_nagged():
    assert diagnostics.advise(PARAMS, [], "training", 40) == []


def test_overfitting_is_advice_rather_than_a_warning():
    logs = [
        {"Epoch": 1, "Training Loss": 1.0, "Validation Loss": 1.0},
        {"Epoch": 2, "Training Loss": 0.6, "Validation Loss": 0.7},
        {"Epoch": 3, "Training Loss": 0.3, "Validation Loss": 0.9},
        {"Epoch": 4, "Training Loss": 0.1, "Validation Loss": 1.4},
    ]
    advice = diagnostics.advise(PARAMS, logs, "finished", 40)
    assert any("memorise" in title for title in titles(advice))
    assert [item["level"] for item in advice] == [diagnostics.INFO]


def test_a_missing_learning_rate_does_not_crash_the_advice():
    logs = [{"Epoch": 1, "Training Loss": float("nan")}]
    advice = diagnostics.advise({}, logs, "finished", None)
    assert advice and "smaller" in advice[0]["detail"]


def test_a_real_error_beats_a_guess():
    """The run that started this: an empty validation split said so in its log,
    while the advice told the user their batch size was too large -- quoting
    real numbers, which made it convincing and wrong."""
    from anylearning.training import diagnostics

    logs = (
        "[2026-08-16 18:40:01] Exported data item 1 of 12\n"
        "Traceback (most recent call last):\n"
        '  File "training_job.py", line 133, in require_labelled_splits\n'
        "    raise ValueError(message)\n"
        "ValueError: The val split has no labelled images (train: 12, val: 0). "
        "Assign images to it in the Dataset tab before training.\n"
    )
    advice = diagnostics.advise(
        {"batch_size": 4, "epochs": 1},
        [],
        status="error",
        training_images=12,
        training_logs=logs,
    )

    titles = [entry["title"] for entry in advice]
    assert "The run stopped with an error" in titles
    assert "The run produced no metrics at all" not in titles
    detail = next(
        e["detail"] for e in advice if e["title"] == "The run stopped with an error"
    )
    assert "val split has no labelled images" in detail


def test_without_an_error_the_inference_still_runs():
    """A run that recorded nothing and raised nothing is still worth a guess."""
    from anylearning.training import diagnostics

    advice = diagnostics.advise(
        {"batch_size": 32, "epochs": 1},
        [],
        status="error",
        training_images=6,
        training_logs="[18:40:01] Training started...\n",
    )
    assert [e["title"] for e in advice] == ["The run produced no metrics at all"]

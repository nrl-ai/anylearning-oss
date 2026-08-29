"""Reading a finished run and saying what to change.

A training run that goes wrong does not announce it. The loss becomes NaN and
the chart simply stops; the loss climbs instead of falling and the numbers look
like numbers; a batch larger than the dataset trains for zero iterations and
the run ends with "No model found in training output", which reads as a broken
trainer rather than as a setting to change.

Every rule here is one of those, phrased as the next thing to try. They are
advice, not verdicts: each says what was observed as well as what to do, so
someone who knows better can disagree with it.

Deliberately not automatic. Changing the learning rate behind someone's back
makes two runs incomparable, and the whole point of storing the parameters with
the session is that runs can be compared.
"""

from __future__ import annotations

import math
import re
from typing import Optional

#: What each entry means to the UI. "warn" is a run that did not work;
#: "info" is a run that worked and could work better.
WARN = "warn"
INFO = "info"


def _numbers(metric_logs, key: str) -> list[float]:
    values = []
    for row in metric_logs or []:
        if not isinstance(row, dict):
            continue
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _has_non_finite(metric_logs) -> bool:
    for row in metric_logs or []:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return True
            # Already translated for JSON, which is how the UI sees it.
            if value is None:
                return True
    return False


def _suggested_rate(learning_rate) -> str:
    try:
        return f"{float(learning_rate) / 10:g}"
    except (TypeError, ValueError):
        return "ten times smaller"


def _explicit_error(training_logs: Optional[str]) -> Optional[str]:
    """The exception a failed run ended on, if it named one.

    Guessing is only worth doing when nothing better is available. A run that
    stopped because the validation split was empty said so in its log, while
    this module told the user their batch size was too large -- quoting real
    numbers, which made it convincing and wrong.
    """
    if not training_logs:
        return None
    for line in reversed(str(training_logs).strip().splitlines()):
        match = re.match(r"^\s*(?:\w+\.)*(\w*(?:Error|Exception)):\s*(\S.*)$", line)
        if match:
            return match.group(2).strip()
    return None


def advise(
    params: Optional[dict],
    metric_logs,
    status: Optional[str] = None,
    training_images: Optional[int] = None,
    classes: Optional[int] = None,
    training_logs: Optional[str] = None,
) -> list[dict]:
    """What to change before running again. Empty when the run looks healthy."""
    params = params or {}
    advice: list[dict] = []

    learning_rate = params.get("learning_rate")
    batch_size = params.get("batch_size")

    if _has_non_finite(metric_logs):
        advice.append(
            {
                "level": WARN,
                "title": "The loss became NaN, so the model stopped learning",
                "detail": (
                    f"Almost always the learning rate is too high. Try "
                    f"{_suggested_rate(learning_rate)} instead of {learning_rate}. "
                    "A smaller batch size makes it worse, not better, so lower "
                    "the rate first."
                ),
            }
        )

    training_loss = _numbers(metric_logs, "Training Loss")
    if len(training_loss) >= 3 and training_loss[-1] > training_loss[0]:
        advice.append(
            {
                "level": WARN,
                "title": "The training loss went up rather than down",
                "detail": (
                    f"It started at {training_loss[0]:.3g} and ended at "
                    f"{training_loss[-1]:.3g}. A learning rate of "
                    f"{_suggested_rate(learning_rate)} is the first thing to try; "
                    "if it still climbs, the labels may not match the images."
                ),
            }
        )

    # An error the run actually raised beats anything inferred from the shape
    # of the metrics.
    reported = _explicit_error(training_logs) if status == "error" else None
    if reported:
        advice.append(
            {
                "level": WARN,
                "title": "The run stopped with an error",
                "detail": reported,
            }
        )

    if not reported and status in {"finished", "error"} and not (metric_logs or []):
        detail = (
            "No epoch was recorded, which happens when the batch size is larger "
            "than the training set: the loader drops the last partial batch, so "
            "there is nothing left to train on."
        )
        if training_images is not None and batch_size:
            detail += (
                f" There {'is' if training_images == 1 else 'are'} "
                f"{training_images} training image"
                f"{'' if training_images == 1 else 's'} and the batch size is "
                f"{batch_size}."
            )
        advice.append(
            {
                "level": WARN,
                "title": "The run produced no metrics at all",
                "detail": detail,
            }
        )

    # A model no better than guessing, on a run short enough to explain it.
    # The handpose classifier trains from scratch and spends its first ten
    # epochs at chance -- 3% on 26 ASL letters -- before climbing to the high
    # seventies by 300. Someone who accepted a ten-epoch default and got 3%
    # concluded the software did not work, which is a fair reading of the
    # evidence they had.
    accuracy = _numbers(metric_logs, "Validation Accuracy")
    epochs = params.get("epochs")
    if accuracy and classes and classes > 1:
        chance = 1.0 / classes
        best_accuracy = max(accuracy)
        if best_accuracy <= chance * 1.5 and len(accuracy) < 50:
            advice.append(
                {
                    "level": WARN,
                    "title": "The model is still guessing",
                    "detail": (
                        f"Its best validation accuracy was {best_accuracy:.1%}, and "
                        f"guessing between {classes} classes would give about "
                        f"{chance:.1%}. {len(accuracy)} epoch"
                        f"{'' if len(accuracy) == 1 else 's'} is usually too few "
                        "for this to mean anything -- train for longer before "
                        "changing anything else."
                    ),
                }
            )
        elif best_accuracy > chance * 1.5 and epochs and accuracy[-1] >= max(accuracy):
            advice.append(
                {
                    "level": INFO,
                    "title": "It was still improving when the run ended",
                    "detail": (
                        f"Validation accuracy was highest on the last epoch "
                        f"({accuracy[-1]:.1%}), so more than {epochs} epochs would "
                        "probably do better. Nothing is wrong with this model; "
                        "there is more to be had."
                    ),
                }
            )

    validation_loss = _numbers(metric_logs, "Validation Loss")
    if len(validation_loss) >= 4 and len(training_loss) >= 4:
        best = min(validation_loss)
        # Rising well past its best while training loss keeps falling is the
        # textbook shape of overfitting, and the threshold keeps ordinary noise
        # from tripping it.
        if validation_loss[-1] > best * 1.25 and training_loss[-1] < training_loss[0]:
            advice.append(
                {
                    "level": INFO,
                    "title": "The model is starting to memorise the training set",
                    "detail": (
                        f"Validation loss was {best:.3g} at its best and is "
                        f"{validation_loss[-1]:.3g} now, while training loss kept "
                        "falling. Fewer epochs, more images, or more augmentation "
                        "all help."
                    ),
                }
            )

    return advice

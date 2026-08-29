"""Carry RF-DETR's Lightning metrics into the project database.

The training process has no stdout anyone reads and no way to return a value:
the only channel to the user is ``TrainingLogsWriter``, which the frontend
polls. RF-DETR trains through PyTorch Lightning, and Lightning's channel for
numbers is a ``Logger`` -- so the adapter is a Logger rather than a callback,
and it is passed to ``build_trainer`` where it *replaces* the CSV logger RF-DETR
would otherwise write into the run folder (a folder that is deleted when the run
ends).

Two decisions worth writing down:

* **One row per validation.** Lightning calls ``log_metrics`` several times an
  epoch -- training losses on a step schedule, validation metrics at the end --
  and the frontend charts one point per epoch. So everything is accumulated and
  a row is emitted when a ``val/`` key arrives, which happens exactly once per
  validated epoch.

* **The keys are renamed.** Lightning's ``val/mAP_50_95`` is the chart's title
  in the UI, and the other four trainers already write "Validation mAP". A
  metric name is not a place to leak the framework underneath.
"""

from __future__ import annotations

from typing import Any

from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

#: Lightning's metric key -> what the training-details chart is titled.
#:
#: Anything not named here is dropped rather than shown. RF-DETR logs about
#: forty keys per epoch -- every loss component, per-class AP, cardinality error
#: -- and a page of forty charts is not more information than five.
METRIC_LABELS = {
    "train/loss": "Training Loss",
    "val/loss": "Validation Loss",
    "val/mAP_50_95": "Validation mAP",
    "val/mAP_50": "Validation mAP@50",
    "val/segm_mAP_50_95": "Validation Mask mAP",
    "val/segm_mAP_50": "Validation Mask mAP@50",
    # RF-DETR keeps an exponential moving average of the weights and scores it
    # separately, and `_best_checkpoint()` registers `checkpoint_best_total.pth`
    # -- whichever of the two scored higher. So on the runs where the EMA wins,
    # every mAP the user could see belonged to a model they did not get. Two
    # keys, not four: enough to show which model was kept without doubling the
    # page.
    #
    # It also keeps the chart populated if `eval_ema_only` is ever turned on.
    # That setting skips the base model's forward pass, and then `val/mAP_50_95`
    # is never written at all -- with only the plain keys named here, the run
    # would chart its losses and no accuracy whatsoever.
    "val/ema_mAP_50_95": "Validation mAP (EMA)",
    "val/ema_segm_mAP_50_95": "Validation Mask mAP (EMA)",
    "val/keypoint_map_50_95": "Validation Keypoint mAP",
    "val/keypoint_map_50": "Validation Keypoint mAP@50",
    "val/ema_keypoint_map_50_95": "Validation Keypoint mAP (EMA)",
}


class RFDetrLogger(Logger):
    """A Lightning logger that writes into a project's training session."""

    def __init__(self, writer, save_dir: str):
        super().__init__()
        self._writer = writer
        self._save_dir = save_dir
        self._pending: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "RF-DETR"

    @property
    def version(self) -> str:
        return "1"

    @property
    def save_dir(self) -> str:
        return self._save_dir

    @rank_zero_only
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        # Deliberately nothing. The hyperparameters a user chose are already on
        # the training session as the stored config, and Lightning hands over
        # the whole resolved model config here -- several hundred fields, most
        # of them architecture constants.
        return None

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for key, label in METRIC_LABELS.items():
            if key in metrics:
                self._pending[label] = float(metrics[key])

        # A validation key means the epoch is over and the row is complete.
        # Emitting on every call instead would write a row per logging interval,
        # and the chart's x-axis is epochs.
        if not any(key.startswith("val/") for key in metrics):
            return
        if not self._pending:
            return

        epoch = metrics.get("epoch")
        summary = ", ".join(
            f"{name}: {value:.4f}" for name, value in self._pending.items()
        )
        if epoch is None:
            self._writer.write(summary)
        else:
            self._writer.write(f"Epoch {int(epoch) + 1} -- {summary}")
        self._writer.write_metrics(dict(self._pending))
        self._pending.clear()

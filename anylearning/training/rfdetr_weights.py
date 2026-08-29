"""The RF-DETR starting checkpoints, and why they are not the ones Roboflow ships.

RF-DETR downloads its COCO-pretrained weights the first time a model is
constructed, from ``storage.googleapis.com/rfdetr/...`` into ``$RF_HOME`` (by
default ``~/.roboflow/models``). That is the same failure the rest of
``anylearning/weights.py`` exists to prevent: a training run that prepares the
whole dataset and then dies on a network the machine does not have.

So the checkpoints ship inside the application, and the trainer hands RF-DETR an
**absolute path** to one rather than a model id it would resolve and fetch.

Two things follow from that, and both are deliberate:

* **The bundled files are renamed.** ``download_pretrain_weights`` looks the
  basename up in its own registry, and a name it recognises is a name it will
  MD5-check and, on a mismatch, complain about in a log the user never sees. A
  name it does not recognise is left alone entirely -- no lookup, no checksum,
  no request. ``rf-detr-nano-anylearning.pth`` says whose file it is.

* **They are smaller than what Roboflow publishes.** The detection checkpoints
  are ``checkpoint_best_regular.pth`` files straight out of a training run:
  ``rf-detr-nano.pth`` is 349 MB, of which 233 MB is optimizer state that
  fine-tuning never reads -- ``load_pretrain_weights`` takes ``checkpoint["model"]``
  and, for provenance, ``checkpoint["args"]``. Stripping the rest turns three
  detection checkpoints from ~1.1 GB into ~370 MB of installer, which on a
  download that is already several gigabytes is worth the twenty lines below.
  The segmentation and keypoint checkpoints are published already stripped, so
  the same pass over them is a no-op -- which is the point of a denylist rather
  than a list of keys to keep: a key nobody here has heard of survives it.

The same stripping runs once more at the end of a training run, on the
checkpoint that gets registered: PyTorch Lightning writes the weights twice, as
``model`` and again as ``state_dict``, so every RF-DETR model a user trains
would otherwise cost twice its own size in their data folder forever.
"""

from __future__ import annotations

import pathlib

from anylearning import weights

#: Where the bundled checkpoints live inside the weights directory.
SUBDIRECTORY = "rfdetr"

#: Our file name -> the name RF-DETR knows it by.
#:
#: The right-hand side is what ``rfdetr.assets.model_weights.ModelWeights``
#: calls the checkpoint, and it is the only thing `fetch_weights.py` needs to
#: ask RF-DETR's own downloader for the file. Nothing at runtime uses it; the
#: trainer only ever sees the left-hand side.
CHECKPOINTS = {
    "rf-detr-nano-anylearning.pth": "rf-detr-nano.pth",
    "rf-detr-small-anylearning.pth": "rf-detr-small.pth",
    "rf-detr-seg-nano-anylearning.pth": "rf-detr-seg-nano.pt",
    "rf-detr-seg-small-anylearning.pth": "rf-detr-seg-small.pt",
    "rf-detr-keypoint-preview-anylearning.pth": ("rf-detr-keypoint-preview-xlarge.pth"),
}

#: Checkpoint keys that hold the state of a training run rather than a model.
#:
#: ``state_dict`` is deliberately absent from this list even though it is
#: usually dropped too: Lightning writes the same tensors under both ``model``
#: and ``state_dict``, so it is a duplicate only when ``model`` is there. See
#: :func:`strip_training_state`.
_TRAINING_STATE_KEYS = (
    "optimizer",
    "optimizer_states",
    "lr_scheduler",
    "lr_schedulers",
    "loops",
    "callbacks",
)


def directory() -> pathlib.Path | None:
    """The bundled RF-DETR weights folder, or None in a build without one."""
    root = weights.bundled_dir()
    return None if root is None else root / SUBDIRECTORY


def bundled_path(file_name: str) -> pathlib.Path | None:
    """Absolute path of a bundled starting checkpoint, or None if it is absent.

    None rather than a raise: the caller is a trainer that has to say something
    a user can act on, and "this build does not include the RF-DETR weights" is
    a different problem from "this file is corrupt".
    """
    folder = directory()
    if folder is None:
        return None
    candidate = folder / file_name
    return candidate if candidate.is_file() else None


def strip_training_state(checkpoint: dict) -> dict:
    """A checkpoint with the optimizer and Lightning bookkeeping removed.

    Returns a new dict; the input is not modified. Everything not named in
    :data:`_TRAINING_STATE_KEYS` is carried across untouched, including keys
    this module has never heard of -- ``model_name`` and ``model_config`` are
    what ``RFDETR.from_checkpoint`` uses to decide which architecture to
    rebuild, and losing either turns a trained model into one that cannot be
    opened.
    """
    kept = {
        key: value
        for key, value in checkpoint.items()
        if key not in _TRAINING_STATE_KEYS
    }
    # Only when the weights survive under the other name. A Lightning-native
    # .ckpt has `state_dict` and no `model`, and dropping it would leave a file
    # with no weights in it at all.
    if "model" in kept and "state_dict" in kept:
        del kept["state_dict"]
    return kept

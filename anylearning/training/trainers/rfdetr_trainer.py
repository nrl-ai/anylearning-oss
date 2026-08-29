"""RF-DETR: a second architecture for detection, and one for instance segmentation.

NanoDet and Mask R-CNN are what this application shipped first, and both are
still the right default on a laptop with no GPU. RF-DETR is the other end of
that trade: a DETR over a DINOv2 backbone, noticeably more accurate on small
datasets because almost all of what it knows came from pretraining, and
correspondingly heavier to train.

It is offered as extra *variants of the project types that already exist* rather
than as new project types. A detection project therefore keeps its data, its
labels and its exports, and the choice between NanoDet and RF-DETR is made per
training run -- which is the only place the difference is real.

Four things about the integration are worth knowing before changing it.

**The weights are handed over as an absolute path.** RF-DETR resolves a bare
model id by downloading it, and this application is sold on training offline.
See ``anylearning/training/rfdetr_weights.py`` for what ships and why the files
are renamed.

**Training does not go through ``RFDETR.train()``.** That method builds a
PyTorch Lightning trainer internally and forwards only ``accelerator`` and
``devices`` to it, so there is no way in to pass a logger -- and a logger is the
only channel a training process has to the user, because the job runs in its own
process and its stdout is read by nobody. So the three public pieces
``RFDETRModelModule``, ``RFDETRDataModule`` and ``build_trainer`` are assembled
here instead, which is what ``RFDETR.train()`` does with the parts this trainer
does not need (auto batch-size probing, dataset-derived class counts, syncing
the trained weights back onto an in-memory model that is then thrown away).

**``model_name`` has to be written into the model config.** It is what
``RFDETR.from_checkpoint`` reads back to decide which architecture to rebuild,
and inference has nothing else to go on. ``RFDETR.train()`` sets it as a side
effect; assembling the parts by hand means setting it deliberately.

**Precision is asked for, not applied here.** RF-DETR runs its own autocast
inside Lightning, so ``anylearning/training/precision.py`` decides *what* this
machine should train in and that decision is translated into RF-DETR's own
``amp`` / ``amp_dtype`` switches, rather than wrapping a context manager around
a loop this trainer does not own.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import sys
import traceback

import cv2
import numpy as np
import yaml
from sqlalchemy.orm import Session

from anylearning import config as anylearning_config
from anylearning import settings
from anylearning.database import DataItem, TrainingParams, db_manager
from anylearning.training import augmentation, keypoints, precision, rfdetr_weights
from anylearning.training.device_utils import get_device
from anylearning.training.logging import TrainingLogsWriter
from anylearning.training.trainers.base_trainer import BaseTrainer
from anylearning.utils.resources import resource_path

PACKAGE_NAME = "anylearning"

#: AnyLearning's subset index -> the folder name RF-DETR's loader looks for.
#:
#: "valid", not "val": ``build_roboflow_from_coco`` hard-codes Roboflow's export
#: layout, and a folder called "val" is simply not found.
SUBSET_FOLDERS = ("train", "valid", "test")

#: Fallback colours for drawing predictions, used when a label carries none.
#: Distinct hues rather than a gradient, so neighbouring classes do not look the
#: same on a small preview.
_PALETTE = (
    (56, 56, 255),
    (56, 255, 56),
    (255, 56, 56),
    (255, 194, 0),
    (0, 194, 255),
    (194, 0, 255),
    (255, 128, 0),
    (0, 255, 194),
)


def _loader_worker_count(
    configured,
    on_gpu: bool,
    *,
    platform: str | None = None,
    compiled: bool | None = None,
) -> int:
    """Resolve RF-DETR workers without breaking a frozen spawn process.

    A DataLoader worker on macOS and Windows starts by unpickling RF-DETR's
    dataset. That imports RF-DETR before the spawned copy of the AnyLearning
    entry point can call ``frozen_compat.apply()``, so transformers tries to
    inspect compiled source and the worker exits with ``OSError: could not get
    source code``. The training process itself is already spawned and remains
    isolated; loading batches in it is the only reliable frozen configuration
    on spawn platforms.

    Linux uses ``fork`` and inherits the repair, while source runs still have
    inspectable Python files. Keep their measured worker policy unchanged.
    """
    platform = sys.platform if platform is None else platform
    compiled = "__compiled__" in globals() if compiled is None else compiled
    if compiled and platform in {"darwin", "win32"}:
        return 0
    return settings.resolve_num_workers(configured, on_gpu=on_gpu)


def _hex_to_bgr(value):
    """A "#rrggbb" label colour as OpenCV's blue-green-red, or None if unusable.

    A list rather than a tuple, because this goes into the config that is
    stored on the model row and read back with ``yaml.safe_load``. PyYAML
    writes a tuple as ``!!python/tuple``, which the safe loader refuses -- so a
    tuple here would produce a model that trains and then cannot be used.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        red, green, blue = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None
    return [blue, green, red]


def _colour_for(index: int, colours) -> tuple:
    if colours and 0 <= index < len(colours) and colours[index]:
        return tuple(colours[index])
    return _PALETTE[index % len(_PALETTE)]


class _CocoWriter:
    """Accumulates one split's images and annotations in COCO form.

    Every image is recorded, annotated or not: RF-DETR's dataset iterates the
    ``images`` list, and dropping the empty ones would quietly remove the
    negative examples a detector needs to learn what is *not* an object.
    """

    def __init__(
        self,
        categories: list,
        with_masks: bool,
        keypoint_names: list[str] | None = None,
    ):
        self.categories = categories
        self.with_masks = with_masks
        self.keypoint_names = keypoint_names
        self.images: list[dict] = []
        self.annotations: list[dict] = []
        self._name_to_id = {category["name"]: category["id"] for category in categories}

    def add(self, file_name: str, width: int, height: int, annotation) -> None:
        image_id = len(self.images) + 1
        self.images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )
        if self.keypoint_names is not None:
            for instance in keypoints.instances(annotation, self.keypoint_names):
                self._add_keypoint_instance(image_id, instance)
            return
        for shape in (annotation or {}).get("data") or []:
            self._add_shape(image_id, shape)

    def _add_keypoint_instance(self, image_id: int, instance: dict) -> None:
        bbox = [float(value) for value in instance["bbox"]]
        self.annotations.append(
            {
                "id": len(self.annotations) + 1,
                "image_id": image_id,
                "category_id": self.categories[0]["id"],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "keypoints": instance["keypoints"],
                "num_keypoints": instance["num_keypoints"],
            }
        )

    def _add_shape(self, image_id: int, shape) -> None:
        if not isinstance(shape, dict):
            return
        categories = shape.get("categories")
        name = (
            categories[0] if isinstance(categories, list) and categories else categories
        )
        category_id = self._name_to_id.get(name)
        if category_id is None:
            return
        # float32 and nothing else: cv2.contourArea accepts only CV_32F or
        # CV_32S, and a polygon carrying one fractional coordinate -- which the
        # labelling canvas produces routinely -- otherwise aborts the export.
        points = np.array(shape.get("points") or [], dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            return

        x_min, y_min = points.min(axis=0)[:2]
        x_max, y_max = points.max(axis=0)[:2]
        width, height = float(x_max - x_min), float(y_max - y_min)
        if width <= 0 or height <= 0:
            return

        record = {
            "id": len(self.annotations) + 1,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [float(x_min), float(y_min), width, height],
            "area": width * height,
            "iscrowd": 0,
        }
        if self.with_masks:
            # A box has no outline to segment, so a rectangle becomes its four
            # corners. Without this an instance-segmentation project that was
            # labelled with boxes would train on annotations with no mask at
            # all, and RF-DETR's mask loss would have nothing to learn from.
            if points.shape[0] < 3:
                x_min, y_min, x_max, y_max = (
                    float(x_min),
                    float(y_min),
                    float(x_max),
                    float(y_max),
                )
                polygon = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            else:
                polygon = [float(value) for value in points[:, :2].flatten()]
            record["segmentation"] = [polygon]
            contour_area = float(cv2.contourArea(points[:, :2]))
            if contour_area > 0:
                record["area"] = contour_area
        self.annotations.append(record)

    def to_coco(self) -> dict:
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }


class RFDetrTrainer(BaseTrainer):
    """RF-DETR for object detection."""

    #: RF-DETR's own augmentation is a resize/crop schedule plus a horizontal
    #: flip. The richer set upstream offers (rotation, colour jitter, affine)
    #: is behind the `rfdetr[augment]` extra, which pulls albumentations and
    #: kornia into a build that has to freeze cleanly -- so only the flip is
    #: declared, because only the flip can actually be honoured.
    AUGMENTATIONS = (augmentation.HORIZONTAL_FLIP,)

    #: Whether this trainer builds the mask head.
    SEGMENTATION = False

    #: Whether the COCO annotations carry landmark triplets rather than shapes.
    KEYPOINTS = False

    #: RF-DETR uses a specialised config subclass for keypoint losses.
    TRAIN_CONFIG_CLASS = "TrainConfig"

    #: The image size has to be a multiple of `patch_size * num_windows`, which
    #: is 16 * 2 for both detection variants. RF-DETR raises rather than
    #: rounding, and it raises after the dataset has been exported.
    IMAGE_SIZE_STEP = 32

    CONFIG_TEMPLATES = {
        "nano": resource_path(PACKAGE_NAME, "training/configs/rfdetr-nano.yml"),
        "small": resource_path(PACKAGE_NAME, "training/configs/rfdetr-small.yml"),
    }

    def __init__(
        self,
        training_folder: str,
        logger: TrainingLogsWriter,
        project_id: int,
        training_params: TrainingParams,
    ):
        super().__init__(training_folder, logger, project_id, training_params)

    # -- data ------------------------------------------------------------

    def prepare_data(self):
        """Export the project into the COCO layout RF-DETR's loader expects.

        ``<data>/train|valid|test/`` holds the images and, beside them,
        ``_annotations.coco.json``.

        Every category is written with ``supercategory: "none"``, which makes
        the file a *flat* category list -- and RF-DETR returns a flat list
        untouched. The alternative matters: its loader drops categories that
        group other categories and carry no annotations of their own, so a
        hierarchy would let a class with no instances in the training split
        vanish and shift every label index after it.
        """
        engine = db_manager.get_project_engine(self.project_id)
        sorted_labels = sorted(self.labels, key=lambda label: label["id"])
        keypoint_names = self._keypoint_names(sorted_labels)
        categories = self._categories(sorted_labels, keypoint_names)
        writers = {
            folder: _CocoWriter(categories, self.SEGMENTATION, keypoint_names)
            for folder in SUBSET_FOLDERS
        }

        with Session(engine) as session:
            data_items = session.query(DataItem).all()
            for index, item in enumerate(data_items):
                image_path = (
                    pathlib.Path(anylearning_config.PROJECTS_ROOT)
                    / str(self.project_id)
                    / "data"
                    / item.path
                )
                if not image_path.is_file():
                    self.logger.write(f"Warning: {item.path} is missing. Skipping.")
                    continue

                subset = SUBSET_FOLDERS[item.subset]
                destination = self.data_folder / subset
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy(image_path, destination)

                image = cv2.imread(str(image_path))
                if image is None:
                    self.logger.write(f"Warning: {item.path} did not decode. Skipping.")
                    continue
                height, width = image.shape[:2]
                writers[subset].add(
                    os.path.basename(image_path), width, height, item.annotation
                )

                if index % 50 == 0:
                    self.logger.write(
                        f"Exported data item {index + 1} of {len(data_items)}"
                    )

        for subset, writer in writers.items():
            folder = self.data_folder / subset
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "_annotations.coco.json").write_text(json.dumps(writer.to_coco()))
            self.logger.write(
                f"{subset}: {len(writer.images)} images, "
                f"{len(writer.annotations)} annotations."
            )

        (self.training_folder / "labels.json").write_text(json.dumps(self.labels))
        self.logger.write("Data exported successfully.")

    def _keypoint_names(self, sorted_labels) -> list[str] | None:
        return None

    def _categories(self, sorted_labels, keypoint_names) -> list[dict]:
        return [
            {"id": index + 1, "name": label["name"], "supercategory": "none"}
            for index, label in enumerate(sorted_labels)
        ]

    # -- config ----------------------------------------------------------

    def prepare_config(self):
        self.config_path = self.training_folder / "rfdetr.yml"
        labels = json.loads((self.training_folder / "labels.json").read_text())
        sorted_labels = sorted(labels, key=lambda label: label["id"])

        template = self.CONFIG_TEMPLATES.get(self.training_params.model_size)
        if template is None:
            raise ValueError(
                f"RF-DETR has no {self.training_params.model_size!r} variant; "
                f"expected one of {sorted(self.CONFIG_TEMPLATES)}. "
                "See config.MODEL_VARIANTS."
            )
        with open(template) as handle:
            config = yaml.safe_load(handle)

        config["data"]["class_names"] = self._class_names(sorted_labels)
        config["data"]["class_colors"] = [
            _hex_to_bgr(label.get("color")) for label in sorted_labels
        ]
        config["data"]["dataset_dir"] = str(self.data_folder)
        config["data"]["augmentation"] = self.resolve_augmentation()
        config["model"]["num_classes"] = len(config["data"]["class_names"])
        config["model"]["resolution"] = self.resolve_image_size(
            config["model"]["resolution"]
        )
        config["output_dir"] = str(self.output_folder)

        config["model"]["pretrained_path"] = str(
            self._starting_weights(
                config["model"]["variant"], config["model"]["pretrained"]
            )
        )

        # Read before the rate is overwritten: RF-DETR fine-tunes the DINOv2
        # encoder at a higher rate than the head, and moving one without the
        # other is how a fine-tune stops converging. So the dialog sets the
        # head's rate and the backbone keeps the template's ratio to it.
        ratio = config["training"]["lr_encoder"] / config["training"]["learning_rate"]
        config["training"]["epochs"] = self.training_params.epochs
        config["training"]["batch_size"] = self.training_params.batch_size
        config["training"]["learning_rate"] = self.training_params.learning_rate
        config["training"]["lr_encoder"] = self.training_params.learning_rate * ratio
        self._apply_task_config(config, sorted_labels)

        with open(self.config_path, "w") as handle:
            yaml.dump(config, handle)
        return yaml.dump(config)

    def _class_names(self, sorted_labels) -> list[str]:
        return [label["name"] for label in sorted_labels]

    def _apply_task_config(self, config: dict, sorted_labels) -> None:
        return None

    def _starting_weights(self, variant: str, file_name: str):
        """The checkpoint this run fine-tunes from. Never nothing.

        Two candidates, in order: a model the user picked in the dialog, then
        the bundled COCO checkpoint. Having neither raises rather than falling
        through to training from scratch. ``load_pretrain_weights`` loads with
        ``strict=False``,
        so a missing or mismatched file does not raise: it loads no weights at
        all and trains a 30M-parameter transformer from random initialisation on
        someone's few hundred images. That looks exactly like a working run
        until the metrics arrive, which is the failure this product can least
        afford.
        """
        chosen = self.resolve_pretrained_model_path()
        if chosen:
            if self._is_same_architecture(chosen, variant):
                self.logger.write(f"Starting from your model at {chosen}.")
                return chosen
            self.logger.write(
                f"The chosen starting model is not an {variant} checkpoint; "
                "starting from the bundled COCO weights instead."
            )

        bundled = rfdetr_weights.bundled_path(file_name)
        if bundled is None:
            raise RuntimeError(
                f"The RF-DETR starting weights ({file_name}) are not in this "
                "installation. They ship with the application; a build made "
                "without running fetch_weights.py has none, and RF-DETR cannot "
                "be trained from scratch here."
            )
        return bundled

    @staticmethod
    def _is_same_architecture(checkpoint_path, variant: str) -> bool:
        """Whether a checkpoint was written by this RF-DETR variant.

        Read rather than trusted, because ``strict=False`` turns the wrong
        answer into a silent one. The training dialog already filters the
        starting-model list by architecture and size, so this only fires for a
        run started straight from the API -- which is exactly the case with
        nobody watching.
        """
        import torch

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except Exception:  # noqa: BLE001 -- an unreadable checkpoint is a "no"
            return False
        return isinstance(checkpoint, dict) and checkpoint.get("model_name") == variant

    # -- training --------------------------------------------------------

    def train(self):
        config = yaml.safe_load(self.config_path.read_text())
        try:
            self._fit(config)
        except Exception as error:
            self.logger.write(
                f"Error during training: {error} {traceback.format_exc()}"
            )
            raise RuntimeError(
                f"Training process failed due to Error: {error}"
            ) from error

    def _fit(self, config: dict) -> None:
        import rfdetr.config as rfdetr_config
        from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

        from anylearning.training.rfdetr_logging import RFDetrLogger

        device = get_device()
        settings.apply_torch_runtime(device)

        variant = config["model"]["variant"]
        config_class = getattr(rfdetr_config, f"{variant}Config", None)
        if config_class is None:
            raise ValueError(f"rfdetr has no configuration for {variant}.")

        plan = precision.from_config(config, device=device)
        self.logger.write(plan.describe())

        model_config = self._model_config(config_class, config, plan, device)
        train_config = self._train_config(config, plan, device)

        accelerator, devices = _lightning_device(device)
        module = RFDETRModelModule(model_config, train_config)
        datamodule = RFDETRDataModule(model_config, train_config)
        trainer = build_trainer(
            train_config,
            model_config,
            accelerator=accelerator,
            devices=devices,
            logger=RFDetrLogger(self.logger, str(self.output_folder)),
            log_every_n_steps=self._logging_interval(train_config),
        )
        with self._captured_stdout():
            trainer.fit(module, datamodule)

    @contextlib.contextmanager
    def _captured_stdout(self):
        """Send RF-DETR's printing to a file instead of to a pipe nobody drains.

        RF-DETR prints a formatted metrics table on every validation epoch,
        through rich, on top of a model summary and a page of dataset notes.
        None of it reaches the user -- the training log does -- and in the
        packaged application the process's stdout is whatever the parent had,
        which on Windows is a console that has been disabled and under the smoke
        tests was a pipe. A pipe holds 64KB and then blocks the writer, which
        reads as a hung run and has already cost this project a day to attribute.

        Kept as a file rather than discarded: it is the first thing to read when
        a run fails for a reason the training log does not explain, and
        `--development` preserves the folder for exactly that.

        Beside the training folder, deliberately, and not inside
        ``training_output``: Lightning refuses to start when its checkpoint
        directory already holds anything, so putting a log file there ends the
        run before the first epoch with "Checkpoint directory ... exists and is
        not empty" -- a message about our log file that names neither.
        """
        self.training_folder.mkdir(parents=True, exist_ok=True)
        # Name the encoding rather than inheriting the machine locale. Windows
        # still defaults ordinary text files to its legacy ANSI code page even
        # when PYTHONIOENCODING makes the process's standard streams UTF-8.
        # Rich prints RF-DETR's validation table with box-drawing characters;
        # redirecting stdout to a locale-encoded file therefore used to abort
        # keypoint training at the first validation epoch on Windows.
        with open(
            self.training_folder / "rfdetr-output.log",
            "a",
            buffering=1,
            encoding="utf-8",
        ) as sink:
            with contextlib.redirect_stdout(sink):
                yield

    def _logging_interval(self, train_config) -> int:
        """How often Lightning flushes metrics, in steps.

        ``build_trainer`` hardcodes 50, and Lightning warns -- loudly, and once
        per run -- when an epoch has fewer batches than that, because it will
        then log no training loss at all. A few hundred labelled images at a
        batch of four is well under 50 steps, which is most of the datasets this
        application is for.
        """
        images = 0
        annotations = self.data_folder / "train" / "_annotations.coco.json"
        if annotations.is_file():
            images = len(json.loads(annotations.read_text()).get("images", []))
        batches = max(1, -(-images // max(1, int(train_config.batch_size))))
        return max(1, min(50, batches))

    def _model_config(self, config_class, config: dict, plan, device):
        """The RF-DETR architecture config for this run.

        The defaults are read back off a throwaway instance so the resolution
        override can be applied the way RF-DETR itself applies it: the
        positional-encoding size is only recomputed when the variant derives it
        from the resolution. Two variants do not -- they carry DINOv2's own
        encoding size and changing it makes the pretrained weights unloadable --
        and neither is shipped here, but copying the rule is cheaper than
        depending on which variants a future release ships.
        """
        defaults = config_class()
        resolution = int(config["model"]["resolution"])
        block = defaults.patch_size * defaults.num_windows
        if resolution % block != 0:
            raise ValueError(
                f"Image size {resolution} does not work for this model: it has "
                f"to be a multiple of {block}."
            )

        overrides = {
            "num_classes": int(config["model"]["num_classes"]),
            "pretrain_weights": config["model"]["pretrained_path"],
            "resolution": resolution,
            # Written deliberately: RFDETR.from_checkpoint reads it back to
            # decide which architecture to rebuild, and inference has nothing
            # else to go on.
            "model_name": config["model"]["variant"],
            "amp": plan.enabled,
            "device": str(device),
        }
        overrides.update(self._model_config_overrides(config))
        if (
            defaults.positional_encoding_size
            == defaults.resolution // defaults.patch_size
        ):
            overrides["positional_encoding_size"] = resolution // defaults.patch_size
        return config_class(**overrides)

    def _model_config_overrides(self, config: dict) -> dict:
        return {}

    def _train_config(self, config: dict, plan, device):
        import rfdetr.config as rfdetr_config

        training = config["training"]
        on_gpu = device.type in ("cuda", "mps")
        workers = _loader_worker_count(training.get("num_workers"), on_gpu=on_gpu)
        # None disables augmentation's horizontal flip; {} disables the flip
        # while keeping the resize schedule. Anything else routes through
        # albumentations, which is not installed.
        flip = config["data"]["augmentation"].get("horizontal_flip", True)

        config_class = getattr(rfdetr_config, self.TRAIN_CONFIG_CLASS)
        overrides = dict(
            dataset_dir=config["data"]["dataset_dir"],
            output_dir=config["output_dir"],
            class_names=list(config["data"]["class_names"]),
            epochs=int(training["epochs"]),
            batch_size=self._training_batch_size(config, device),
            grad_accum_steps=int(training["grad_accum_steps"]),
            lr=float(training["learning_rate"]),
            lr_encoder=float(training["lr_encoder"]),
            weight_decay=float(training["weight_decay"]),
            num_workers=workers,
            checkpoint_interval=int(training["checkpoint_interval"]),
            early_stopping=bool(training["early_stopping"]),
            amp_dtype=_AMP_DTYPES.get(plan.label, "auto"),
            aug_config=None if flip else {},
            # Off, so that the image size means the size images are trained at.
            #
            # RF-DETR's multi-scale schedule does not sample around the chosen
            # resolution -- with square resizing it takes the *largest* candidate
            # and uses only that, which for a 384 px model is 544 px. So a user
            # who lowers the image size to fit their GPU changes the number in
            # the dialog and nothing else, while memory and time stay where they
            # were. Every other trainer here treats that control as the training
            # size, and a control that means something different for one model is
            # worse than one that offers less. `scale_jitter` stays on, so the
            # resize-crop-resize branch still varies what the model sees; it just
            # ends at the size that was asked for.
            multi_scale=False,
            # Off, all four of them. tensorboard, wandb, mlflow and the progress
            # bar each write somewhere the user cannot see -- and the progress
            # bar writes to a stdout that, in the packaged application, is a
            # pipe nobody drains. A pipe holds 64KB; a training run that fills
            # one stops mid-write, which reads as a hang.
            tensorboard=False,
            wandb=False,
            mlflow=False,
            progress_bar=None,
            # Forty numbers an epoch reach the log otherwise, and the training
            # details page draws one chart per key.
            log_per_class_metrics=False,
        )
        overrides.update(self._train_config_overrides(config))
        return config_class(**overrides)

    def _train_config_overrides(self, config: dict) -> dict:
        return {}

    def _training_batch_size(self, config: dict, device) -> int:
        return int(config["training"]["batch_size"])

    # -- results ---------------------------------------------------------

    def get_model_path(self):
        """The best checkpoint, slimmed, at a stable name.

        RF-DETR writes the weights twice into every checkpoint it saves -- once
        as ``model`` for its own loader and once as ``state_dict`` for
        Lightning's -- so registering the file as written would cost every user
        twice the disk their model needs, permanently. The slimmed copy is what
        gets exported and registered, which means ONNX export doubles as the
        proof that slimming produced a loadable file: export gates registration,
        so a broken copy cannot reach a model row.
        """
        import torch

        source = self._best_checkpoint()
        if source is None:
            self.logger.write("No model found in training output.")
            return False, None

        slimmed = self.output_folder / "model_best.pth"
        if not slimmed.is_file():
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            torch.save(rfdetr_weights.strip_training_state(checkpoint), slimmed)
        return True, str(slimmed)

    def _best_checkpoint(self):
        """The checkpoint to keep, in the order RF-DETR ranks them.

        ``checkpoint_best_total.pth`` is whichever of the plain and the
        exponential-moving-average models scored higher, so it is the honest
        first choice. The rest are fallbacks for a run that ended before any
        validation improved on nothing.
        """
        for name in (
            "checkpoint_best_total.pth",
            "checkpoint_best_regular.pth",
            "checkpoint_best_ema.pth",
            "last.ckpt",
        ):
            candidate = self.output_folder / name
            if candidate.is_file():
                return candidate
        return None

    def export_onnx(self):
        ret, model_path = self.get_model_path()
        if not ret:
            return None

        from rfdetr import RFDETR

        export_folder = self.output_folder / "onnx"
        # Always on the CPU, for the same reason device_utils.
        # load_torch_model_for_export exists: the graph is identical, it does
        # not compete for VRAM with the run that has just finished in this same
        # process, and it cannot bake a CUDA assumption into an artefact the
        # user may run anywhere.
        model = RFDETR.from_checkpoint(str(model_path), device="cpu")
        # Captured for the same reason training is: onnxsim prints a comparison
        # table and NanoDet's export once printed its entire ONNX graph, which
        # is where the "the export takes 26 minutes" misdiagnosis came from.
        with self._captured_stdout():
            exported = model.export(output_dir=str(export_folder), verbose=False)

        # The path export() returns, not a glob of the folder: it names the file
        # after the variant, and the folder may hold more than one graph if the
        # exporter ever writes an intermediate beside the finished one.
        produced = pathlib.Path(exported) if exported else None
        if produced is None or not produced.is_file():
            self.logger.write("RF-DETR's export wrote no ONNX file.")
            return None

        # Renamed to the name every other trainer produces, because the model
        # row's download link is built from it.
        onnx_path = self.output_folder / "exported_model.onnx"
        shutil.move(str(produced), onnx_path)
        return str(onnx_path)

    # -- inference -------------------------------------------------------

    @staticmethod
    def run_inference(config_data: str, model_path: str, image: np.ndarray):
        """Run one image through a trained model.

        A staticmethod called by ``routers/model.py`` without a trainer, so it
        reads everything it needs -- class names, colours, threshold -- from the
        config text stored on the model row.
        """
        from rfdetr import RFDETR

        config = yaml.safe_load(config_data) or {}
        data = config.get("data", {})
        class_names = list(data.get("class_names") or [])
        class_colors = list(data.get("class_colors") or [])
        threshold = float(config.get("inference", {}).get("threshold", 0.35))

        model = RFDETR.from_checkpoint(str(model_path), device=str(get_device()))
        # predict() takes RGB; the router hands over BGR because that is what
        # the rest of the application draws in.
        detections = model.predict(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=threshold
        )

        results = []
        visualisation = image.copy()
        masks = getattr(detections, "mask", None)
        for index in range(len(detections)):
            class_id = int(detections.class_id[index])
            confidence = float(detections.confidence[index])
            box = [float(value) for value in detections.xyxy[index]]
            name = (
                class_names[class_id]
                if 0 <= class_id < len(class_names)
                else str(class_id)
            )
            results.append(
                {
                    "class_id": class_id,
                    "label": name,
                    "confidence": confidence,
                    "box": box,
                }
            )
            colour = _colour_for(class_id, class_colors)
            if masks is not None:
                _draw_mask(visualisation, masks[index], colour)
            _draw_box(visualisation, box, f"{name} {confidence:.2f}", colour)

        return results, visualisation


class RFDetrSegTrainer(RFDetrTrainer):
    """RF-DETR for instance segmentation.

    Same trainer with the mask head on: the exported annotations gain a
    ``segmentation`` polygon, and predictions come back with masks to draw. The
    image-size step is 24 rather than 32 because the segmentation variants use a
    patch size of 12 -- Seg Nano runs one window and would accept any multiple
    of 12, but one rule that satisfies both variants is easier to reason about
    than a rule that changes with the size the user picked.
    """

    SEGMENTATION = True
    IMAGE_SIZE_STEP = 24

    CONFIG_TEMPLATES = {
        "nano": resource_path(PACKAGE_NAME, "training/configs/rfdetr-seg-nano.yml"),
        "small": resource_path(PACKAGE_NAME, "training/configs/rfdetr-seg-small.yml"),
    }


class RFDetrKeypointTrainer(RFDetrTrainer):
    """RF-DETR's preview keypoint model for one project-defined skeleton."""

    KEYPOINTS = True
    IMAGE_SIZE_STEP = 24
    TRAIN_CONFIG_CLASS = "KeypointTrainConfig"
    CONFIG_TEMPLATES = {
        "preview": resource_path(PACKAGE_NAME, "training/configs/rfdetr-keypoint.yml")
    }

    def _training_batch_size(self, config: dict, device) -> int:
        """Keep the preview keypoint head inside Apple GPU memory.

        RF-DETR keypoint training has a substantially larger activation graph
        than detection. Batch eight reproducibly consumes the entire shared
        memory budget on a 16 GB M1, and the dialog's old generic batch sixteen
        default was worse. The server owns the final guard so API clients and
        saved jobs created by older versions are safe as well as the current
        UI. CUDA and CPU retain the number the user requested.
        """
        requested = super()._training_batch_size(config, device)
        if device.type != "mps" or requested <= 2:
            return requested

        self.logger.write(
            f"Reducing keypoint batch size from {requested} to 2 for Apple Metal "
            "memory safety."
        )
        return 2

    def _keypoint_names(self, sorted_labels) -> list[str]:
        names = keypoints.keypoint_names(sorted_labels)
        if not names:
            raise ValueError("A keypoint project needs at least one landmark label.")
        return names

    def _categories(self, sorted_labels, keypoint_names) -> list[dict]:
        return [keypoints.coco_category(keypoint_names)]

    def _class_names(self, sorted_labels) -> list[str]:
        return [keypoints.DEFAULT_CATEGORY]

    def _apply_task_config(self, config: dict, sorted_labels) -> None:
        names = keypoints.keypoint_names(sorted_labels)
        config["data"]["keypoint_names"] = names
        config["data"]["keypoint_colors"] = [
            _hex_to_bgr(label.get("color")) for label in sorted_labels
        ]
        # The detected category is implicit, so its colour is not one of the
        # landmark colours above.
        config["data"]["class_colors"] = [None]
        config["model"]["num_keypoints_per_class"] = [len(names)]
        config["training"]["keypoint_flip_pairs"] = keypoints.flip_pairs(names)
        config["training"]["keypoint_oks_sigmas"] = [keypoints.DEFAULT_OKS_SIGMA] * len(
            names
        )

    def _model_config_overrides(self, config: dict) -> dict:
        return {
            "num_keypoints_per_class": list(config["model"]["num_keypoints_per_class"])
        }

    def _train_config_overrides(self, config: dict) -> dict:
        training = config["training"]
        return {
            "keypoint_flip_pairs": list(training["keypoint_flip_pairs"]),
            "keypoint_oks_sigmas": list(training["keypoint_oks_sigmas"]),
        }

    @staticmethod
    def run_inference(config_data: str, model_path: str, image: np.ndarray):
        """Return named landmarks and draw the visible ones on the image."""
        from rfdetr import RFDETR

        config = yaml.safe_load(config_data) or {}
        data = config.get("data", {})
        names = list(data.get("keypoint_names") or [])
        # Early preview checkpoints stored the single category's landmark
        # schema as ``[[name, ...]]`` to mirror RF-DETR's per-class training
        # input. Current AnyLearning configs store the project schema flat.
        # Accept both so models trained by either version remain usable.
        if len(names) == 1 and isinstance(names[0], (list, tuple)):
            names = list(names[0])
        names = [str(name) for name in names]
        colours = list(data.get("keypoint_colors") or [])
        inference = config.get("inference", {})
        threshold = float(inference.get("threshold", 0.35))
        keypoint_threshold = float(inference.get("keypoint_threshold", 0.25))

        model = RFDETR.from_checkpoint(str(model_path), device=str(get_device()))
        predictions = model.predict(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=threshold
        )

        visualisation = image.copy()
        results = []
        # Dense anatomy schemas quickly turn into a wall of overlapping text.
        # Their names and confidences remain available in the structured result;
        # the image stays useful by drawing labels only for compact schemas.
        draw_names = len(names) <= 12
        xy = np.asarray(getattr(predictions, "xy", []))
        confidence = getattr(predictions, "keypoint_confidence", None)
        confidence = None if confidence is None else np.asarray(confidence)
        detection_confidence = getattr(predictions, "detection_confidence", None)
        boxes = (getattr(predictions, "data", {}) or {}).get("xyxy")

        for instance_index in range(len(xy)):
            landmarks = []
            for point_index, point in enumerate(xy[instance_index]):
                x, y = float(point[0]), float(point[1])
                score = (
                    float(confidence[instance_index, point_index])
                    if confidence is not None
                    else 1.0
                )
                visible = score >= keypoint_threshold
                name = (
                    names[point_index] if point_index < len(names) else str(point_index)
                )
                landmarks.append(
                    {
                        "name": name,
                        "x": x,
                        "y": y,
                        "confidence": score,
                        "visible": visible,
                    }
                )
                if visible:
                    colour = _colour_for(point_index, colours)
                    cv2.circle(
                        visualisation,
                        (int(round(x)), int(round(y))),
                        4,
                        colour,
                        -1,
                        lineType=cv2.LINE_AA,
                    )
                    if draw_names:
                        cv2.putText(
                            visualisation,
                            name,
                            (int(round(x)) + 6, int(round(y)) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            colour,
                            1,
                            cv2.LINE_AA,
                        )

            record = {"keypoints": landmarks}
            if detection_confidence is not None:
                record["confidence"] = float(detection_confidence[instance_index])
            if boxes is not None:
                record["box"] = [float(value) for value in boxes[instance_index]]
            results.append(record)

        return results, visualisation


#: What `precision.Plan` calls a dtype -> what RF-DETR's TrainConfig calls it.
#:
#: A plan that chose 16-bit names the dtype rather than leaving RF-DETR to
#: choose again, which is what makes ANYLEARNING_MIXED_PRECISION reach this
#: trainer too. A plan that chose 32-bit is not in this table and falls back to
#: "auto" -- deliberately, because `model_config.amp` is already False by then
#: and RF-DETR warns about an amp_dtype set alongside it.
_AMP_DTYPES = {"bfloat16": "bf16", "float16": "fp16"}


def _lightning_device(device):
    """A torch device as PyTorch Lightning's (accelerator, devices) pair.

    ``devices=1`` rather than "auto" on purpose: auto-detection on a machine
    with two GPUs starts distributed training, and this already runs inside a
    ``multiprocessing`` child that the application spawned.
    """
    if device.type == "cuda":
        return "gpu", 1
    if device.type == "mps":
        return "mps", 1
    return "cpu", 1


def _draw_box(image, box, caption: str, colour) -> None:
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, 2)
    (width, height), baseline = cv2.getTextSize(
        caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    top = max(y0, height + baseline + 2)
    cv2.rectangle(
        image,
        (x0, top - height - baseline - 2),
        (x0 + width + 2, top),
        colour,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        caption,
        (x0 + 1, top - baseline - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_mask(image, mask, colour) -> None:
    """Tint the masked pixels, leaving the photograph underneath readable."""
    selected = np.asarray(mask).astype(bool)
    if selected.shape[:2] != image.shape[:2] or not selected.any():
        return
    tint = np.array(colour, dtype=np.float32)
    region = image[selected].astype(np.float32)
    image[selected] = (region * 0.6 + tint * 0.4).astype(np.uint8)

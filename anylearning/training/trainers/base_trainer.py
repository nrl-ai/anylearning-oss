import os
import pathlib
from abc import abstractmethod

from fastapi import HTTPException
from sqlalchemy.orm import Session

from anylearning import config, database
from anylearning.database import Model, Project, TrainingParams
from anylearning.training import augmentation
from anylearning.training.logging import TrainingLogsWriter


class BaseTrainer:
    def __init__(
        self,
        training_folder: str,
        logger: TrainingLogsWriter,
        project_id: int,
        training_params: TrainingParams,
    ):
        self.training_folder = pathlib.Path(training_folder)
        self.data_folder = self.training_folder / "data"
        self.output_folder = self.training_folder / "training_output"
        self.config_path = self.training_folder / "yolo.yml"
        self.logger = logger
        self.project_id = project_id
        self.training_params = training_params
        self.config_path = None
        self.prepare_folders()

        # `database.db_manager`, resolved here rather than imported by name at
        # the top of this file. A module-level `from ... import db_manager`
        # binds whichever manager existed at first import, and a test that
        # patches `anylearning.database.db_manager` then leaves this class
        # still pointing at the real one -- so sixteen trainer-config tests
        # were reading the developer's own ~/anylearning-data/anylearning.db.
        # They passed on the machine where that database happens to contain a
        # project with id 1, and failed on every machine where it does not.
        with Session(database.db_manager.main_engine) as session:
            project = session.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            self.labels = project.labels

    def prepare_folders(self):
        """
        Prepare the folders for the training job.
        """
        self.training_folder.mkdir(parents=True, exist_ok=True)
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def resolve_pretrained_model_path(self):
        """Path of the checkpoint to start from, or None to train from scratch.

        "default" means "no starting checkpoint"; anything else used to be
        handed straight to int(), so any other non-id value died with

            ValueError: invalid literal for int() with base 10: ''

        in prepare_config -- after the whole dataset had already been exported.
        The training dialog only ever sends "default" or a model id, so the
        crash needs a direct API call; the case that *is* reachable from the UI
        is picking a model that gets deleted before the run starts, which used
        to raise AttributeError on model.path.

        Anything that is not a usable model id means "train from scratch".
        """
        raw = (self.training_params.pretrained_model or "").strip()
        if not raw or raw == "default":
            return None

        try:
            model_id = int(raw)
        except ValueError:
            self.logger.write(
                f"Ignoring pretrained model {raw!r}: not a model id. "
                "Training from scratch."
            )
            return None

        with Session(
            database.db_manager.get_project_engine(self.project_id)
        ) as session:
            model = session.get(Model, model_id)
            # Read path inside the session; the instance detaches on exit.
            model_path = model.path if model else None

        if not model_path:
            # The model row can be gone if it was deleted between picking it in
            # the UI and the run starting.
            self.logger.write(
                f"Pretrained model {model_id} is no longer available. "
                "Training from scratch."
            )
            return None

        models_folder = (
            pathlib.Path(config.PROJECTS_ROOT) / str(self.project_id) / "models"
        )
        return str(models_folder / model_path)

    #: Every backbone here downsamples by 32, so an input that is not a
    #: multiple of 32 is silently padded or -- in NanoDet's case -- produces
    #: feature maps whose sizes no longer line up with the head's assumptions.
    IMAGE_SIZE_STEP = 32
    #: Below this there is nothing left to learn from; above it, memory becomes
    #: the limit long before accuracy does.
    IMAGE_SIZE_RANGE = (64, 1280)

    def resolve_image_size(self, default: int) -> int:
        """The training image size: what the user asked for, or the template's.

        Rounded to a multiple of 32 and clamped, with a line in the training log
        when either happens. A rejected value would be worse: the request comes
        from a dialog that has already been dismissed, and the run would fail
        after the whole dataset had been exported.
        """
        requested = getattr(self.training_params, "image_size", None)
        if not requested:
            return default

        try:
            size = int(requested)
        except (TypeError, ValueError):
            self.logger.write(
                f"Ignoring image size {requested!r}: not a number. Using {default}."
            )
            return default

        smallest, largest = self.IMAGE_SIZE_RANGE
        clamped = max(smallest, min(largest, size))
        rounded = max(
            self.IMAGE_SIZE_STEP,
            round(clamped / self.IMAGE_SIZE_STEP) * self.IMAGE_SIZE_STEP,
        )
        if rounded != size:
            self.logger.write(
                f"Image size {size} adjusted to {rounded}: it has to be a "
                f"multiple of {self.IMAGE_SIZE_STEP}, between {smallest} and {largest}."
            )
        return rounded

    #: What this trainer can be asked to do to its training images.
    #:
    #: Declared per trainer rather than shared, because the underlying
    #: pipelines genuinely differ: NanoDet warps boxes along with the image and
    #: has no vertical flip, detectron2 decides most of it inside its own
    #: dataset mapper, and handpose trains on landmark vectors where none of
    #: this means anything. Offering the union everywhere would mean silently
    #: ignoring most of what a user set -- see anylearning/training/augmentation.py.
    AUGMENTATIONS: tuple = ()

    def resolve_augmentation(self) -> dict:
        """This trainer's augmentation settings, after the user's choices.

        Every key it declares is present in the result, at the value that
        trainer already used when nobody chooses -- so a project trained before
        any of this existed keeps training the same way.
        """
        return augmentation.resolve(
            self.AUGMENTATIONS,
            getattr(self.training_params, "augmentation", None),
            log=self.logger,
        )

    @abstractmethod
    def prepare_data(self):
        """Query data from the database and export it to the data folder"""

    @abstractmethod
    def prepare_config(self):
        """Prepare the config for the training job"""

    @abstractmethod
    def train(self):
        """Run the training job"""

    def export_onnx(self):
        """Export the model to ONNX format"""
        onnx_path = os.path.join(self.output_folder, "exported_model.onnx")
        # Mock file creation for base class
        with open(onnx_path, "w") as f:
            f.write(
                "Mock ONNX file - this is a placeholder that should be overridden by subclasses"
            )
        return onnx_path

    def companion_files(self) -> list:
        """Files that have to travel with the checkpoint to be usable.

        A registered model is copied out of the run's folder and that folder is
        then deleted, so anything a trainer needs at inference time and does not
        store in the config has to be named here or it is gone. Instance
        segmentation learned this the hard way: its pickled detectron2 config
        lived only in the run folder, so trying any instance segmentation model
        answered 500.

        Paths that do not exist are skipped by the caller, so a trainer can name
        a file it only sometimes writes.
        """
        return []

    @abstractmethod
    def get_model_path(self):
        """Get the path to the best model"""

    @abstractmethod
    def run_inference(self, image_path: str):
        """Run inference on an image"""

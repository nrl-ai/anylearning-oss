"""Which trainer class runs a project's training and inference.

The project type used to be enough, because each type had exactly one
implementation. RF-DETR broke that: a detection project can now be trained with
NanoDet or with RF-DETR, and an instance-segmentation project with Mask R-CNN or
with RF-DETR-Seg, so the architecture chosen for the run is part of the lookup.

The architecture stays optional, and omitting it keeps the type's original
trainer. Two reasons rather than politeness: every existing model row was
written before this argument existed, and a caller that has a project but not a
run -- ``/api/augmentations`` asks each type what it can augment -- has no
architecture to pass.
"""

from anylearning.training.trainers.classification_trainer import ClassificationTrainer
from anylearning.training.trainers.handpose_classification_trainer import (
    HandposeClassificationTrainer,
)
from anylearning.training.trainers.instseg_trainer import InstSegTrainer
from anylearning.training.trainers.nanodet_trainer import NanoDetTrainer
from anylearning.training.trainers.semseg_trainer import SemSegTrainer

#: (project type, model_architecture) -> trainer.
#:
#: Only the architectures that share a project type with another one need an
#: entry; everything else is answered by DEFAULT_TRAINERS below. The strings are
#: the ``model_architecture`` values in ``config.MODEL_VARIANTS`` and must match
#: them exactly -- they are stored on every model row and read back years later.
ALTERNATIVE_TRAINERS = {
    ("Object Detection", "rfdetr"): "RFDetrTrainer",
    ("Instance Segmentation", "rfdetr-seg"): "RFDetrSegTrainer",
}

#: The trainer a project type uses when nothing else is said.
DEFAULT_TRAINERS = {
    # Kept lazy because importing a structured trainer must not import pandas,
    # scikit-learn or CatBoost during an image-only app launch.
    "Tabular AI": "StructuredTrainer",
    "Text AI": "StructuredTrainer",
    "Text AI & LLM Evaluation": "StructuredTrainer",
    "Text & LLM": "StructuredTrainer",
    "Sentiment Analysis": "StructuredTrainer",
    "Object Detection": NanoDetTrainer,
    "Image Segmentation": SemSegTrainer,
    "Instance Segmentation": InstSegTrainer,
    "Image Classification": ClassificationTrainer,
    "Handpose Classification": HandposeClassificationTrainer,
    # Kept as a string so importing the builder does not import RF-DETR and
    # Transformers on every application launch.
    "Keypoint Detection": "RFDetrKeypointTrainer",
}


def _rfdetr_trainer(name: str):
    """Import the RF-DETR trainer only when something actually asks for it.

    Imported here rather than at module scope because this module is imported
    by the API process at startup, and ``import rfdetr`` costs about 1.8
    seconds -- almost all of it ``transformers``. A user who never picks RF-DETR
    should not pay that on every launch, nor in every spawned training child on
    the platforms that spawn rather than fork.
    """
    from anylearning.training.trainers import rfdetr_trainer

    return getattr(rfdetr_trainer, name)


def _structured_trainer(name: str):
    from anylearning.training.trainers import structured_trainer

    return getattr(structured_trainer, name)


class TrainerBuilder:
    @staticmethod
    def get_trainer_class(project_type: str, model_architecture: str | None = None):
        alternative = ALTERNATIVE_TRAINERS.get((project_type, model_architecture))
        if alternative is not None:
            return _rfdetr_trainer(alternative)

        trainer = DEFAULT_TRAINERS.get(project_type)
        if trainer is None:
            raise ValueError(f"Unsupported project type: {project_type}")
        if isinstance(trainer, str):
            if trainer == "StructuredTrainer":
                return _structured_trainer(trainer)
            return _rfdetr_trainer(trainer)
        return trainer

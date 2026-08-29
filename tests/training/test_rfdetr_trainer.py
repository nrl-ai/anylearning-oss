"""The RF-DETR trainer's contract, without training anything.

Everything here runs in under a second and needs neither a GPU nor the bundled
checkpoints. The parts that need both -- a real fine-tune, an ONNX export, an
inference pass -- are in ``tests/e2e/test_rfdetr_e2e.py``.

What is pinned down here is the wiring that is easy to get silently wrong: the
annotation export RF-DETR's loader reads, the label indices it derives from it,
and the two lookups (trainer by architecture, augmentation by architecture) that
now have to distinguish two detectors sharing one project type.
"""

import builtins
import json

import numpy as np
import pytest
import yaml
from PIL import Image
from sqlalchemy.orm import Session

from anylearning.database import TrainingParams

pytest.importorskip("rfdetr")


LABELS = [
    {"id": 0, "name": "circle", "color": "#ff0000"},
    {"id": 1, "name": "square"},
]


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def write(self, message):
        self.messages.append(str(message))

    def write_metrics(self, metrics):
        self.messages.append(str(metrics))


def _box(x0, y0, x1, y1, name):
    return {
        "type": "rectangle",
        "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "categories": [name],
    }


def _polygon(name):
    return {
        "type": "polygon",
        # Fractional coordinates on purpose: cv2.contourArea takes only CV_32F
        # or CV_32S, and a polygon that decoded to float64 aborted the instance
        # segmentation trainer's export once already.
        "points": [[10.5, 10.5], [40.25, 12.0], [38.0, 44.75], [11.0, 41.5]],
        "categories": [name],
    }


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A detection project with one labelled image in each split."""
    from anylearning import config, database
    from anylearning.database import DataItem, Project

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setattr(config, "PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "main.db"))

    manager = database.DatabaseManager()
    monkeypatch.setattr(database, "db_manager", manager)
    from anylearning.training.trainers import base_trainer, rfdetr_trainer

    for module in (base_trainer, rfdetr_trainer):
        monkeypatch.setattr(module, "db_manager", manager, raising=False)
    monkeypatch.setattr(rfdetr_trainer, "anylearning_config", config, raising=False)

    with Session(manager.main_engine) as session:
        row = Project(name="P", type="Object Detection", labels=LABELS)
        session.add(row)
        session.commit()
        project_id = row.id

    data = projects_root / str(project_id) / "data"
    data.mkdir(parents=True)

    with Session(manager.get_project_engine(project_id)) as session:
        for subset, name in ((0, "circle"), (1, "square"), (2, "circle")):
            file_name = f"image_{subset}.png"
            Image.new("RGB", (64, 48), (240, 240, 240)).save(data / file_name)
            session.add(
                DataItem(
                    path=file_name,
                    original_name=file_name,
                    subset=subset,
                    labeled=1,
                    class_id=0,
                    annotation={
                        "data": [_box(5, 6, 30, 40, name), _polygon(name)],
                    },
                )
            )
        session.commit()

    yield {"id": project_id, "root": tmp_path}

    # This manager is not the one the shared autouse fixture disposes -- that
    # one looks up `anylearning.database.db_manager`, which monkeypatch has
    # already put back by then. Left open, its connections surface as
    # ResourceWarnings attributed to whichever test runs next.
    manager.dispose_all()


def _trainer(project, monkeypatch, klass=None, size="nano", **params):
    from anylearning.training.trainers.rfdetr_trainer import RFDetrTrainer

    klass = klass or RFDetrTrainer
    defaults = dict(
        model_architecture="rfdetr",
        model_size=size,
        model_variant=f"rfdetr_{size}",
        learning_rate=0.0002,
        batch_size=2,
        epochs=3,
        pretrained_model="default",
    )
    defaults.update(params)
    return klass(
        project["root"] / "run",
        RecordingLogger(),
        project["id"],
        TrainingParams(**defaults),
    )


# --------------------------------------------------------------------------
# The exported dataset
# --------------------------------------------------------------------------


def test_prepare_data_writes_the_layout_rfdetr_looks_for(project, monkeypatch):
    """ "valid", not "val": build_roboflow_from_coco hard-codes Roboflow's names."""
    trainer = _trainer(project, monkeypatch)
    trainer.prepare_data()

    for folder in ("train", "valid", "test"):
        annotations = trainer.data_folder / folder / "_annotations.coco.json"
        assert annotations.is_file(), f"{folder} has no _annotations.coco.json"
        assert list((trainer.data_folder / folder).glob("*.png"))


def test_categories_are_flat_so_no_class_can_be_dropped(project, monkeypatch):
    """A hierarchy would let an unannotated class vanish and shift the rest.

    RF-DETR drops a category that groups other categories and carries no
    annotations of its own, then assigns label indices to what is left. A flat
    list -- every supercategory a placeholder -- is returned untouched, which is
    the only way a project's label ids survive a split where one class happens
    to be unused.
    """
    trainer = _trainer(project, monkeypatch)
    trainer.prepare_data()
    coco = json.loads(
        (trainer.data_folder / "train" / "_annotations.coco.json").read_text()
    )

    assert [category["name"] for category in coco["categories"]] == ["circle", "square"]
    assert {category["supercategory"] for category in coco["categories"]} == {"none"}

    from rfdetr.datasets.coco import annotated_category_ids, filter_parent_categories

    kept = filter_parent_categories(coco["categories"], annotated_category_ids(coco))
    assert [category["name"] for category in kept] == ["circle", "square"]


def test_detection_export_has_boxes_and_no_masks(project, monkeypatch):
    trainer = _trainer(project, monkeypatch)
    trainer.prepare_data()
    coco = json.loads(
        (trainer.data_folder / "train" / "_annotations.coco.json").read_text()
    )

    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2
    for annotation in coco["annotations"]:
        assert len(annotation["bbox"]) == 4
        assert annotation["bbox"][2] > 0 and annotation["bbox"][3] > 0
        assert "segmentation" not in annotation


def test_segmentation_export_adds_polygons(project, monkeypatch):
    """Including for a box, which becomes its own four corners.

    An instance-segmentation project labelled with boxes would otherwise train
    on annotations with no mask at all, and the mask loss would have nothing to
    learn from.
    """
    from anylearning.training.trainers.rfdetr_trainer import RFDetrSegTrainer

    trainer = _trainer(project, monkeypatch, klass=RFDetrSegTrainer)
    trainer.prepare_data()
    coco = json.loads(
        (trainer.data_folder / "train" / "_annotations.coco.json").read_text()
    )

    assert len(coco["annotations"]) == 2
    for annotation in coco["annotations"]:
        polygon = annotation["segmentation"][0]
        assert len(polygon) >= 8 and len(polygon) % 2 == 0
        assert annotation["area"] > 0


def test_an_unlabelled_image_is_still_exported(project, monkeypatch):
    """Negative examples teach a detector what is not an object."""
    from anylearning import config
    from anylearning.database import DataItem, db_manager

    with Session(db_manager.get_project_engine(project["id"])) as session:
        Image.new("RGB", (64, 48)).save(
            f"{config.PROJECTS_ROOT}/{project['id']}/data/empty.png"
        )
        session.add(
            DataItem(
                path="empty.png",
                original_name="empty.png",
                subset=0,
                labeled=0,
                class_id=-1,
                annotation=None,
            )
        )
        session.commit()

    trainer = _trainer(project, monkeypatch)
    trainer.prepare_data()
    coco = json.loads(
        (trainer.data_folder / "train" / "_annotations.coco.json").read_text()
    )
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 2


# --------------------------------------------------------------------------
# The config
# --------------------------------------------------------------------------


@pytest.fixture
def bundled_weights(tmp_path, monkeypatch):
    """A stand-in for the shipped checkpoints, so config tests need no download."""
    from anylearning.training import rfdetr_weights

    folder = tmp_path / "weights" / "rfdetr"
    folder.mkdir(parents=True)
    for name in rfdetr_weights.CHECKPOINTS:
        (folder / name).write_bytes(b"not a real checkpoint")
    monkeypatch.setattr(rfdetr_weights, "directory", lambda: folder)
    return folder


def test_prepare_config_carries_the_run_into_the_stored_text(
    project, monkeypatch, bundled_weights
):
    trainer = _trainer(project, monkeypatch, epochs=7, batch_size=3)
    trainer.prepare_data()
    stored = yaml.safe_load(trainer.prepare_config())

    assert stored["model"]["variant"] == "RFDETRNano"
    assert stored["model"]["num_classes"] == 2
    assert stored["data"]["class_names"] == ["circle", "square"]
    assert stored["training"]["epochs"] == 7
    assert stored["training"]["batch_size"] == 3
    assert stored["training"]["learning_rate"] == 0.0002
    # The label's own colour, as OpenCV wants it, so inference can draw the
    # boxes the project's colours without a database.
    assert stored["data"]["class_colors"][0] == [0, 0, 255]
    assert stored["data"]["class_colors"][1] is None
    assert stored["model"]["pretrained_path"].endswith("rf-detr-nano-anylearning.pth")


def test_the_backbone_rate_keeps_its_ratio_to_the_one_the_user_set(
    project, monkeypatch, bundled_weights
):
    """Moving one without the other is how a DETR fine-tune stops converging."""
    trainer = _trainer(project, monkeypatch, learning_rate=0.001)
    trainer.prepare_data()
    stored = yaml.safe_load(trainer.prepare_config())
    assert stored["training"]["lr_encoder"] == pytest.approx(0.0015)


def test_image_size_is_rounded_to_what_the_architecture_accepts(
    project, monkeypatch, bundled_weights
):
    """RF-DETR raises on a bad resolution, and it raises after the export."""
    trainer = _trainer(project, monkeypatch, image_size=500)
    trainer.prepare_data()
    stored = yaml.safe_load(trainer.prepare_config())
    assert stored["model"]["resolution"] % 32 == 0

    from anylearning.training.trainers.rfdetr_trainer import RFDetrSegTrainer

    seg = _trainer(project, monkeypatch, klass=RFDetrSegTrainer, image_size=500)
    seg.prepare_data()
    stored = yaml.safe_load(seg.prepare_config())
    assert stored["model"]["resolution"] % 24 == 0


def test_missing_bundled_weights_stop_the_run(project, monkeypatch):
    """Rather than training a transformer from random initialisation.

    ``load_pretrain_weights`` loads with ``strict=False``, so an absent
    checkpoint does not raise: it loads nothing and the run looks healthy until
    the metrics arrive.
    """
    from anylearning.training import rfdetr_weights

    monkeypatch.setattr(rfdetr_weights, "directory", lambda: None)
    trainer = _trainer(project, monkeypatch)
    trainer.prepare_data()
    with pytest.raises(RuntimeError, match="starting weights"):
        trainer.prepare_config()


def test_an_unknown_size_says_so_before_the_framework_does(
    project, monkeypatch, bundled_weights
):
    trainer = _trainer(project, monkeypatch, size="enormous")
    trainer.prepare_data()
    with pytest.raises(ValueError, match="enormous"):
        trainer.prepare_config()


def test_every_template_names_a_config_class_and_a_bundled_checkpoint():
    """The three names in a template have to agree with rfdetr and with us."""
    import rfdetr.config as rfdetr_config

    from anylearning.training import rfdetr_weights
    from anylearning.training.trainers.rfdetr_trainer import (
        RFDetrKeypointTrainer,
        RFDetrSegTrainer,
        RFDetrTrainer,
    )

    for trainer in (RFDetrTrainer, RFDetrSegTrainer, RFDetrKeypointTrainer):
        for size, path in trainer.CONFIG_TEMPLATES.items():
            with open(path) as handle:
                template = yaml.safe_load(handle)
            variant = template["model"]["variant"]
            config_class = getattr(rfdetr_config, f"{variant}Config", None)
            assert config_class is not None, f"{variant} is not an rfdetr config"
            assert template["model"]["pretrained"] in rfdetr_weights.CHECKPOINTS

            defaults = config_class()
            assert defaults.segmentation_head == trainer.SEGMENTATION, size
            block = defaults.patch_size * defaults.num_windows
            assert template["model"]["resolution"] % block == 0
            # The rounding step the trainer applies has to produce sizes this
            # architecture accepts, or a user's image size fails after the
            # whole dataset has been exported.
            assert trainer.IMAGE_SIZE_STEP % block == 0


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def test_our_checkpoint_names_are_not_names_rfdetr_knows():
    """The rename is what stops RF-DETR checksumming a file we changed.

    ``download_pretrain_weights`` looks the basename up in its own registry; a
    name it recognises gets an MD5 comparison against the published file, which
    a stripped copy cannot pass.
    """
    from rfdetr.assets.model_weights import ModelWeights

    from anylearning.training import rfdetr_weights

    published = set(ModelWeights.list_models())
    assert not (set(rfdetr_weights.CHECKPOINTS) & published)
    # And the other half of the mapping does have to name real ones.
    assert set(rfdetr_weights.CHECKPOINTS.values()) <= published


def test_stripping_keeps_everything_that_is_not_training_state():
    from anylearning.training import rfdetr_weights

    stripped = rfdetr_weights.strip_training_state(
        {
            "model": {"weight": 1},
            "state_dict": {"weight": 1},
            "optimizer": {"big": 2},
            "optimizer_states": [{"big": 2}],
            "lr_schedulers": [],
            "loops": {},
            "callbacks": {},
            "args": {"lr": 3},
            "model_name": "RFDETRNano",
            "something_new": 4,
        }
    )
    assert stripped == {
        "model": {"weight": 1},
        "args": {"lr": 3},
        "model_name": "RFDETRNano",
        "something_new": 4,
    }


def test_a_lightning_checkpoint_keeps_its_only_copy_of_the_weights():
    """`state_dict` is dropped as a duplicate, never as the last one standing."""
    from anylearning.training import rfdetr_weights

    stripped = rfdetr_weights.strip_training_state(
        {"state_dict": {"weight": 1}, "optimizer_states": [{"big": 2}]}
    )
    assert stripped == {"state_dict": {"weight": 1}}


# --------------------------------------------------------------------------
# Reporting progress
# --------------------------------------------------------------------------


def test_one_metric_row_per_validated_epoch():
    """Lightning logs several times an epoch; the chart plots one point."""
    from anylearning.training.rfdetr_logging import RFDetrLogger

    writer = RecordingLogger()
    writer.metrics = []
    writer.write_metrics = writer.metrics.append
    logger = RFDetrLogger(writer, "/tmp")

    logger.log_metrics({"train/loss": 2.0}, step=10)
    logger.log_metrics({"train/loss": 1.5}, step=20)
    assert writer.metrics == [], "a training step is not an epoch"

    logger.log_metrics({"val/loss": 1.0, "val/mAP_50_95": 0.4, "epoch": 0}, step=20)
    assert writer.metrics == [
        {"Training Loss": 1.5, "Validation Loss": 1.0, "Validation mAP": 0.4}
    ]

    logger.log_metrics({"train/loss": 1.0}, step=40)
    logger.log_metrics({"val/loss": 0.8, "val/mAP_50_95": 0.6, "epoch": 1}, step=40)
    assert len(writer.metrics) == 2
    assert writer.metrics[1]["Validation mAP"] == 0.6


def test_the_forty_other_metrics_are_dropped():
    """One chart per key, and RF-DETR logs every loss component separately."""
    from anylearning.training.rfdetr_logging import RFDetrLogger

    writer = RecordingLogger()
    writer.metrics = []
    writer.write_metrics = writer.metrics.append

    RFDetrLogger(writer, "/tmp").log_metrics(
        {
            "val/loss": 1.0,
            "val/loss_bbox_0": 0.1,
            "val/cardinality_error_enc": 3.0,
            "val/AP/circle": 0.9,
        },
        step=1,
    )
    assert writer.metrics == [{"Validation Loss": 1.0}]


def test_a_segmentation_run_reports_its_mask_metric():
    from anylearning.training.rfdetr_logging import RFDetrLogger

    writer = RecordingLogger()
    writer.metrics = []
    writer.write_metrics = writer.metrics.append

    RFDetrLogger(writer, "/tmp").log_metrics(
        {"val/mAP_50_95": 0.5, "val/segm_mAP_50_95": 0.3, "epoch": 2}, step=1
    )
    assert writer.metrics[0]["Validation Mask mAP"] == 0.3


def test_training_output_goes_to_a_file_rather_than_to_stdout(
    project, monkeypatch, capsys
):
    """The capture is drained and UTF-8 even under Windows' legacy locale."""
    trainer = _trainer(project, monkeypatch)

    # Linux's default locale is already UTF-8, so simply printing RF-DETR's
    # table there would not catch the Windows failure. Make an unspecified
    # encoding behave like a native Windows cp1252 default. The implementation
    # must override it explicitly for Rich's box-drawing characters to survive.
    real_open = builtins.open

    def windows_default_encoding(*args, **kwargs):
        kwargs.setdefault("encoding", "cp1252")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", windows_default_encoding)
    with trainer._captured_stdout():
        print("a metrics table ┏━ nobody reads")

    assert "a metrics table" not in capsys.readouterr().out
    written = (trainer.training_folder / "rfdetr-output.log").read_text(
        encoding="utf-8"
    )
    assert "a metrics table ┏━ nobody reads" in written
    # And not in the folder Lightning checkpoints into: it refuses to start
    # when that directory already holds anything, so a log file there would end
    # the run before its first epoch.
    assert not (trainer.output_folder / "rfdetr-output.log").exists()


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def test_the_builder_picks_the_trainer_the_run_asked_for():
    from anylearning.training.trainers.instseg_trainer import InstSegTrainer
    from anylearning.training.trainers.nanodet_trainer import NanoDetTrainer
    from anylearning.training.trainers.rfdetr_trainer import (
        RFDetrSegTrainer,
        RFDetrTrainer,
    )
    from anylearning.training.trainers.trainer_builder import TrainerBuilder

    assert (
        TrainerBuilder.get_trainer_class("Object Detection", "rfdetr") is RFDetrTrainer
    )
    assert (
        TrainerBuilder.get_trainer_class("Instance Segmentation", "rfdetr-seg")
        is RFDetrSegTrainer
    )
    # Everything written before the argument existed still resolves.
    assert TrainerBuilder.get_trainer_class("Object Detection") is NanoDetTrainer
    assert (
        TrainerBuilder.get_trainer_class("Object Detection", "nanodet")
        is NanoDetTrainer
    )
    assert (
        TrainerBuilder.get_trainer_class("Instance Segmentation", None)
        is InstSegTrainer
    )


def test_every_advertised_variant_resolves_to_a_trainer():
    """A variant the dialog offers and the builder cannot map fails at run time.

    Late, and after the dataset has been exported -- so it is asserted here
    against the same list the UI is built from.
    """
    from anylearning import config
    from anylearning.training.trainers.trainer_builder import TrainerBuilder

    for project_type, variants in config.MODEL_VARIANTS.items():
        for variant in variants:
            trainer = TrainerBuilder.get_trainer_class(
                project_type, variant["model_architecture"]
            )
            assert trainer is not None, f"{project_type}/{variant['name']}"


def test_augmentations_are_reported_per_architecture():
    """Two detectors, one project type, and not the same set of options.

    NanoDet warps boxes along with the image and takes rotation and colour
    jitter; RF-DETR's equivalents are behind an optional dependency that is not
    installed. One answer per project type would have to lie about one of them.
    """
    from anylearning.routers.model import get_augmentations

    options = get_augmentations()
    keys = {option["key"] for option in options["Object Detection::rfdetr"]}
    assert keys == {"horizontal_flip"}
    assert "rotation_degrees" in {
        option["key"] for option in options["Object Detection::nanodet"]
    }
    # The plain key stays, unchanged, for a client that has not learnt the
    # compound ones.
    assert options["Object Detection"] == options["Object Detection::nanodet"]


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def test_a_mask_tints_its_pixels_and_leaves_the_rest_alone():
    from anylearning.training.trainers.rfdetr_trainer import _draw_mask

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True
    _draw_mask(image, mask, (0, 0, 255))

    assert image[2, 2].tolist() != [0, 0, 0]
    assert image[7, 7].tolist() == [0, 0, 0]


def test_a_mask_of_the_wrong_shape_is_skipped_rather_than_raising():
    """Inference must not fail on the drawing step after the model answered."""
    from anylearning.training.trainers.rfdetr_trainer import _draw_mask

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    _draw_mask(image, np.ones((4, 4), dtype=bool), (0, 0, 255))
    assert image.sum() == 0

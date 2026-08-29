"""RF-DETR end to end, through the trainer contract rather than around it.

`run_training_job` drives every trainer in a fixed order -- prepare_data ->
prepare_config -> train -> export_onnx -> get_model_path -- and only registers a
model once the ONNX export has succeeded. So a run that reaches an ONNX file and
then answers an inference request has proved the whole chain, which is more than
any of the steps prove separately: RF-DETR's checkpoint is slimmed before it is
registered, and the export is what demonstrates the slimmed copy still loads.

Skipped when the bundled checkpoints are absent. They are a build input
(`fetch_weights.py`), not source, and a suite that silently downloaded 900 MB
would be a suite nobody runs.
"""

import json

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw
from sqlalchemy.orm import Session

from anylearning.database import TrainingParams

pytest.importorskip("rfdetr")

from anylearning.training import rfdetr_weights  # noqa: E402

pytestmark = pytest.mark.skipif(
    rfdetr_weights.bundled_path("rf-detr-nano-anylearning.pth") is None,
    reason="the bundled RF-DETR checkpoints are not here; run fetch_weights.py",
)

CLASSES = ("circle", "square")


class RecordingLogger:
    def __init__(self):
        self.messages = []
        self.metrics = []

    def write(self, message):
        self.messages.append(str(message))

    def write_metrics(self, metrics):
        self.metrics.append(dict(metrics))


def _shape(name, size, offset):
    """One large shape on a plain background, drawn where the label says."""
    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    x0 = y0 = offset
    x1 = y1 = offset + size // 2
    fill = (220, 60, 60) if name == "circle" else (60, 120, 220)
    if name == "circle":
        draw.ellipse((x0, y0, x1, y1), fill=fill)
    else:
        draw.rectangle((x0, y0, x1, y1), fill=fill)
    return image, (x0, y0, x1, y1)


def _annotation(name, box, polygon):
    x0, y0, x1, y1 = box
    if polygon:
        points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, (y0 + y1) / 2]]
        return {"data": [{"type": "polygon", "points": points, "categories": [name]}]}
    points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return {"data": [{"type": "rectangle", "points": points, "categories": [name]}]}


@pytest.fixture
def project(tmp_path, monkeypatch, request):
    """A small labelled project, in a database of its own under tmp_path."""
    from anylearning import config, database
    from anylearning.database import DataItem, Project
    from anylearning.training import device_utils

    # The CPU, everywhere. `run_training_job` normally puts the dialog's choice
    # in this variable before a trainer is built; calling the trainer directly
    # skips that, and a test whose answer depends on the machine it ran on is
    # not the one to have here -- the packaged training matrix covers the GPU.
    monkeypatch.setenv(device_utils.DEVICE_PREFERENCE_ENV, "cpu")

    project_type = getattr(request, "param", "Object Detection")
    polygons = project_type == "Instance Segmentation"

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

    labels = [{"id": index, "name": name} for index, name in enumerate(CLASSES)]
    with Session(manager.main_engine) as session:
        row = Project(name="P", type=project_type, labels=labels)
        session.add(row)
        session.commit()
        project_id = row.id

    data = projects_root / str(project_id) / "data"
    data.mkdir(parents=True)

    with Session(manager.get_project_engine(project_id)) as session:
        for subset in (0, 1, 2):
            for index, name in enumerate(CLASSES):
                for repeat in range(2):
                    image, box = _shape(name, 128, 12 + 20 * repeat)
                    file_name = f"{subset}_{name}_{repeat}.png"
                    image.save(data / file_name)
                    session.add(
                        DataItem(
                            path=file_name,
                            original_name=file_name,
                            subset=subset,
                            labeled=1,
                            class_id=index,
                            annotation=_annotation(name, box, polygons),
                        )
                    )
        session.commit()

    yield {"id": project_id, "root": tmp_path, "type": project_type}
    manager.dispose_all()


def _params(architecture):
    return TrainingParams(
        model_architecture=architecture,
        model_size="nano",
        model_variant=f"{architecture}_nano",
        batch_size=2,
        epochs=1,
        learning_rate=0.0001,
        pretrained_model="default",
        # Far below what anyone would train at, and the point of the run is the
        # chain rather than the accuracy. RF-DETR interpolates the pretrained
        # positional encodings down to match.
        image_size=128,
        device="cpu",
    )


# RF-DETR's own notice that its default augmentation backend moved from
# albumentations to torchvision, raised once per run because `aug_config=None`
# is what asks for the default. Scoped here rather than ignored globally: it is
# a real message about pixel values changing, it just has no action behind it
# for a project that never used the albumentations path.
@pytest.mark.filterwarnings("ignore:RF-DETR has changed the default training")
@pytest.mark.filterwarnings("ignore:Converting a tensor to a Python")
@pytest.mark.filterwarnings("ignore:The epoch parameter in `scheduler.step")
@pytest.mark.parametrize(
    "project,architecture,trainer_name",
    [
        ("Object Detection", "rfdetr", "RFDetrTrainer"),
        ("Instance Segmentation", "rfdetr-seg", "RFDetrSegTrainer"),
    ],
    indirect=["project"],
)
def test_rfdetr_trains_exports_and_answers(
    project, architecture, trainer_name, tmp_path
):
    from anylearning.training.trainers.trainer_builder import TrainerBuilder

    trainer_class = TrainerBuilder.get_trainer_class(project["type"], architecture)
    assert trainer_class.__name__ == trainer_name

    logger = RecordingLogger()
    trainer = trainer_class(
        tmp_path / "run", logger, project["id"], _params(architecture)
    )

    trainer.prepare_data()
    exported = json.loads(
        (trainer.data_folder / "valid" / "_annotations.coco.json").read_text()
    )
    assert exported["annotations"], "the validation split exported no annotations"

    config_text = trainer.prepare_config()
    # Stored on the model row and read back by inference with safe_load, so a
    # value PyYAML can only round-trip unsafely is a model that cannot be used.
    assert yaml.safe_load(config_text)

    trainer.train()

    # Progress reached the database rather than only stdout: the training
    # process has no other channel to the user.
    assert trainer.logger.metrics, "no metrics were written for the session"
    assert "Validation mAP" in trainer.logger.metrics[-1]
    if architecture == "rfdetr-seg":
        # The mask head trained, rather than a detector wearing its name: this
        # metric only exists when model_config.segmentation_head is on and the
        # exported annotations carried polygons for it to learn from.
        assert "Validation Mask mAP" in trainer.logger.metrics[-1]

    onnx_path = trainer.export_onnx()
    assert onnx_path and onnx_path.endswith("exported_model.onnx")

    onnxruntime = pytest.importorskip("onnxruntime")
    session = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    assert session.get_inputs(), "the exported graph has no inputs"

    found, model_path = trainer.get_model_path()
    assert found

    image = np.array(_shape("square", 128, 20)[0])[:, :, ::-1].copy()
    results, visualisation = trainer_class.run_inference(config_text, model_path, image)
    # An undertrained detector may legitimately find nothing; the contract is
    # that it answers without raising and hands back something to draw.
    assert results is not None
    assert visualisation.shape == image.shape
    for detection in results:
        assert detection["label"] in CLASSES
        assert 0.0 <= detection["confidence"] <= 1.0


@pytest.mark.filterwarnings("ignore:RF-DETR has changed the default training")
@pytest.mark.filterwarnings("ignore:The epoch parameter in `scheduler.step")
def test_the_registered_checkpoint_is_the_slim_one(project, tmp_path):
    """Lightning writes the weights twice; a user should not store them twice.

    Checked on the real artefact rather than on the stripping helper, because
    what matters is the size of the file that ends up in someone's data folder.
    """
    from anylearning.training.trainers.rfdetr_trainer import RFDetrTrainer

    trainer = RFDetrTrainer(
        tmp_path / "run", RecordingLogger(), project["id"], _params("rfdetr")
    )
    trainer.prepare_data()
    trainer.prepare_config()
    trainer.train()

    found, model_path = trainer.get_model_path()
    assert found
    source = trainer._best_checkpoint()
    assert source.stat().st_size > 0
    import pathlib

    assert pathlib.Path(model_path).stat().st_size < source.stat().st_size

    # And it is still a checkpoint RF-DETR can identify and rebuild.
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    assert checkpoint["model_name"] == "RFDETRNano"
    assert "model" in checkpoint

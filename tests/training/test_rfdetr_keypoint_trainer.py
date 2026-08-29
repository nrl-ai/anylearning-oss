"""The RF-DETR keypoint trainer's data, config, dispatch and inference contract."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from sqlalchemy.orm import Session

from anylearning.database import TrainingParams

pytest.importorskip("rfdetr")


LABELS = [
    {"id": 1, "name": "left_eye", "color": "#ff0000"},
    {"id": 2, "name": "right_eye", "color": "#00ff00"},
    {"id": 3, "name": "nose", "color": "#0000ff"},
]


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def write(self, message):
        self.messages.append(str(message))

    def write_metrics(self, metrics):
        self.messages.append(dict(metrics))


def _dot(name, x, y, group):
    return {
        "type": "dot",
        "position": [x, y],
        "categories": [name],
        "group_id": group,
    }


@pytest.fixture
def project(tmp_path, monkeypatch):
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
        row = Project(name="Pose", type="Keypoint Detection", labels=LABELS)
        session.add(row)
        session.commit()
        project_id = row.id

    data = projects_root / str(project_id) / "data"
    data.mkdir(parents=True)
    with Session(manager.get_project_engine(project_id)) as session:
        for subset in (0, 1, 2):
            name = f"pose-{subset}.png"
            Image.new("RGB", (80, 60), "white").save(data / name)
            session.add(
                DataItem(
                    path=name,
                    original_name=name,
                    subset=subset,
                    labeled=1,
                    class_id=-1,
                    annotation={
                        "data": [
                            _dot("left_eye", 10, 10, 1),
                            _dot("right_eye", 20, 10, 1),
                            _dot("nose", 15, 20, 1),
                            _dot("left_eye", 50, 30, 2),
                            _dot("right_eye", 60, 30, 2),
                        ]
                    },
                )
            )
        session.commit()

    yield {"id": project_id, "root": tmp_path}
    manager.dispose_all()


@pytest.fixture
def bundled_weights(tmp_path, monkeypatch):
    from anylearning.training import rfdetr_weights

    folder = tmp_path / "weights" / "rfdetr"
    folder.mkdir(parents=True)
    for name in rfdetr_weights.CHECKPOINTS:
        (folder / name).write_bytes(b"checkpoint placeholder")
    monkeypatch.setattr(rfdetr_weights, "directory", lambda: folder)
    return folder


def _trainer(project):
    from anylearning.training.trainers.rfdetr_trainer import RFDetrKeypointTrainer

    return RFDetrKeypointTrainer(
        project["root"] / "run",
        RecordingLogger(),
        project["id"],
        TrainingParams(
            model_architecture="rfdetr-keypoint",
            model_size="preview",
            model_variant="rfdetr_keypoint_preview",
            learning_rate=0.0001,
            batch_size=2,
            epochs=3,
            pretrained_model="default",
        ),
    )


def test_prepare_data_writes_one_schema_and_two_instances(project):
    trainer = _trainer(project)
    trainer.prepare_data()
    coco = json.loads(
        (trainer.data_folder / "train" / "_annotations.coco.json").read_text()
    )

    assert coco["categories"] == [
        {
            "id": 1,
            "name": "object",
            "supercategory": "none",
            "keypoints": ["left_eye", "right_eye", "nose"],
            "skeleton": [],
        }
    ]
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2
    assert [row["num_keypoints"] for row in coco["annotations"]] == [3, 2]
    for row in coco["annotations"]:
        assert row["category_id"] == 1
        assert len(row["keypoints"]) == 9
        assert row["bbox"][2] > 0 and row["bbox"][3] > 0
        assert row["area"] > 0


def test_prepare_config_carries_landmark_schema(project, bundled_weights):
    trainer = _trainer(project)
    trainer.prepare_data()
    config = yaml.safe_load(trainer.prepare_config())

    assert config["model"]["variant"] == "RFDETRKeypointPreview"
    assert config["model"]["num_classes"] == 1
    assert config["model"]["num_keypoints_per_class"] == [3]
    assert config["data"]["class_names"] == ["object"]
    assert config["data"]["keypoint_names"] == [
        "left_eye",
        "right_eye",
        "nose",
    ]
    assert config["data"]["keypoint_colors"] == [
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
    ]
    assert config["training"]["keypoint_flip_pairs"] == [0, 1]
    assert config["training"]["keypoint_oks_sigmas"] == [0.05, 0.05, 0.05]
    assert config["model"]["pretrained_path"].endswith(
        "rf-detr-keypoint-preview-anylearning.pth"
    )


def test_specialised_rfdetr_configs_receive_the_schema(project, bundled_weights):
    from rfdetr.config import RFDETRKeypointPreviewConfig

    trainer = _trainer(project)
    trainer.prepare_data()
    config = yaml.safe_load(trainer.prepare_config())
    plan = SimpleNamespace(enabled=False, label="float32")

    model = trainer._model_config(
        RFDETRKeypointPreviewConfig, config, plan, torch.device("cpu")
    )
    training = trainer._train_config(config, plan, torch.device("cpu"))

    assert model.num_classes == 1
    assert model.num_keypoints_per_class == [3]
    assert type(training).__name__ == "KeypointTrainConfig"
    assert training.keypoint_flip_pairs == [0, 1]
    assert training.keypoint_oks_sigmas == [0.05, 0.05, 0.05]
    assert training.keypoint_nll_loss_coef == 1.0


def test_keypoint_batch_is_capped_only_on_apple_metal(project, bundled_weights):
    trainer = _trainer(project)
    trainer.prepare_data()
    config = yaml.safe_load(trainer.prepare_config())
    config["training"]["batch_size"] = 16
    plan = SimpleNamespace(enabled=False, label="float32")

    metal = trainer._train_config(config, plan, torch.device("mps"))
    cpu = trainer._train_config(config, plan, torch.device("cpu"))

    assert metal.batch_size == 2
    assert cpu.batch_size == 16
    assert trainer.logger.messages[-1] == (
        "Reducing keypoint batch size from 16 to 2 for Apple Metal memory safety."
    )


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_frozen_spawn_build_keeps_rfdetr_loading_in_the_training_process(
    monkeypatch, platform
):
    from anylearning.training.trainers import rfdetr_trainer

    monkeypatch.setattr(
        rfdetr_trainer.settings, "resolve_num_workers", lambda *args, **kwargs: 6
    )

    assert (
        rfdetr_trainer._loader_worker_count(
            4, on_gpu=True, platform=platform, compiled=True
        )
        == 0
    )
    assert (
        rfdetr_trainer._loader_worker_count(
            4, on_gpu=True, platform=platform, compiled=False
        )
        == 6
    )
    assert (
        rfdetr_trainer._loader_worker_count(
            4, on_gpu=True, platform="linux", compiled=True
        )
        == 6
    )


def test_builder_lazily_selects_keypoint_trainer():
    from anylearning.training.trainers.rfdetr_trainer import RFDetrKeypointTrainer
    from anylearning.training.trainers.trainer_builder import TrainerBuilder

    assert (
        TrainerBuilder.get_trainer_class("Keypoint Detection", "rfdetr-keypoint")
        is RFDetrKeypointTrainer
    )
    assert (
        TrainerBuilder.get_trainer_class("Keypoint Detection") is RFDetrKeypointTrainer
    )


def test_keypoint_metrics_reach_the_training_chart():
    from anylearning.training.rfdetr_logging import RFDetrLogger

    writer = RecordingLogger()
    writer.metrics = []
    writer.write_metrics = writer.metrics.append
    RFDetrLogger(writer, "/tmp").log_metrics(
        {
            "val/keypoint_map_50_95": 0.42,
            "val/keypoint_map_50": 0.67,
            "epoch": 1,
        },
        step=2,
    )
    assert writer.metrics == [
        {"Validation Keypoint mAP": 0.42, "Validation Keypoint mAP@50": 0.67}
    ]


def test_inference_returns_named_points_and_draws_them(monkeypatch):
    import rfdetr

    from anylearning.training.trainers.rfdetr_trainer import RFDetrKeypointTrainer

    predictions = SimpleNamespace(
        xy=np.array([[[10.0, 12.0], [20.0, 22.0]]], dtype=np.float32),
        keypoint_confidence=np.array([[0.9, 0.1]], dtype=np.float32),
        detection_confidence=np.array([0.8], dtype=np.float32),
        data={"xyxy": np.array([[2, 3, 30, 32]], dtype=np.float32)},
    )

    class FakeModel:
        def predict(self, image, threshold):
            assert threshold == 0.35
            return predictions

    monkeypatch.setattr(
        rfdetr.RFDETR,
        "from_checkpoint",
        staticmethod(lambda *args, **kwargs: FakeModel()),
    )
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    config = yaml.safe_dump(
        {
            "data": {
                "keypoint_names": ["nose", "tail"],
                "keypoint_colors": [[0, 0, 255], [0, 255, 0]],
            },
            "inference": {"threshold": 0.35, "keypoint_threshold": 0.25},
        }
    )

    results, visualisation = RFDetrKeypointTrainer.run_inference(
        config, "model.pth", image
    )

    assert results[0]["confidence"] == pytest.approx(0.8)
    assert results[0]["box"] == [2.0, 3.0, 30.0, 32.0]
    assert results[0]["keypoints"][0]["name"] == "nose"
    assert results[0]["keypoints"][0]["visible"] is True
    assert results[0]["keypoints"][1]["visible"] is False
    assert visualisation.sum() > 0

"""Config generation for every trainer.

`prepare_config()` is where a project's labels and the UI's training parameters
become the YAML each framework consumes. It runs before any GPU work, so a
mistake here wastes a whole training run -- and it is pure enough to test
directly, unlike the training loops themselves.

The recurring bug class it guards against: a label count that disagrees between
the head, the auxiliary head and the class-name list, which frameworks report
much later as a confusing shape mismatch.
"""

import json
import pathlib

import pytest
import yaml

from anylearning.database import Project, TrainingParams


@pytest.fixture
def trainer_env(tmp_path, monkeypatch):
    """A project row plus patched paths, shared by every trainer."""
    from sqlalchemy.orm import Session

    from anylearning import config, database

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setattr(config, "PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "main.db"))

    manager = database.DatabaseManager()
    monkeypatch.setattr(database, "db_manager", manager)

    labels = [
        {"name": "cat", "id": 0, "color": "#FF0000"},
        {"name": "dog", "id": 1, "color": "#00FF00"},
    ]
    with Session(manager.main_engine) as session:
        project = Project(name="T", type="Object Detection", labels=labels)
        session.add(project)
        session.commit()
        project_id = project.id

    # get_project_engine opens projects/<id>/database.db, and sqlite will not
    # create the file unless the directory is already there.
    (projects_root / str(project_id)).mkdir(parents=True, exist_ok=True)

    yield {
        "project_id": project_id,
        "labels": labels,
        "training_folder": tmp_path / "training",
        "manager": manager,
        "monkeypatch": monkeypatch,
    }
    manager.dispose_all()


class NullLogger:
    def write(self, message):
        pass

    def write_metrics(self, metrics):
        pass


def params(architecture, size, epochs=3, batch_size=4, lr=0.005):
    return TrainingParams(
        model_architecture=architecture,
        model_size=size,
        model_variant=f"{architecture}_{size}",
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=lr,
        pretrained_model="default",
    )


def build(trainer_class, env, training_params, patch_targets=()):
    """Instantiate a trainer against the throwaway database."""
    from anylearning import database

    for module in patch_targets:
        env["monkeypatch"].setattr(
            module, "db_manager", database.db_manager, raising=False
        )

    return trainer_class(
        str(env["training_folder"]),
        NullLogger(),
        env["project_id"],
        training_params,
    )


def write_labels(trainer, labels):
    """prepare_config reads labels.json, which prepare_data would have written."""
    path = trainer.training_folder / "labels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels))


# --------------------------------------------------------------------------
# Shared expectations
# --------------------------------------------------------------------------


def test_base_trainer_creates_its_folders(trainer_env):
    from anylearning.training.trainers import base_trainer

    trainer = build(
        base_trainer.BaseTrainer,
        trainer_env,
        params("resnet18", "lightweight"),
        patch_targets=(base_trainer,),
    )

    assert trainer.training_folder.is_dir()
    assert trainer.data_folder.is_dir()
    assert trainer.output_folder.is_dir()


def test_base_trainer_loads_the_project_labels(trainer_env):
    from anylearning.training.trainers import base_trainer

    trainer = build(
        base_trainer.BaseTrainer,
        trainer_env,
        params("resnet18", "lightweight"),
        patch_targets=(base_trainer,),
    )
    assert [label["name"] for label in trainer.labels] == ["cat", "dog"]


def test_base_trainer_rejects_an_unknown_project(trainer_env):
    from fastapi import HTTPException

    from anylearning.training.trainers import base_trainer

    trainer_env["monkeypatch"].setattr(
        base_trainer, "db_manager", trainer_env["manager"], raising=False
    )
    with pytest.raises(HTTPException) as excinfo:
        base_trainer.BaseTrainer(
            str(trainer_env["training_folder"]),
            NullLogger(),
            999999,
            params("resnet18", "lightweight"),
        )
    assert excinfo.value.status_code == 404


@pytest.mark.parametrize("value", ["", "   ", "default", "not-an-id"])
def test_pretrained_model_falls_back_to_training_from_scratch(trainer_env, value):
    """Anything that is not a usable model id means "no starting checkpoint".

    An unset selector arrives as an empty string, and every trainer used to
    hand it straight to int() -- so the run died in prepare_config with
    `invalid literal for int() with base 10: ''`, after the whole dataset had
    already been exported.
    """
    from anylearning.training.trainers import base_trainer

    training_params = params("resnet18", "lightweight")
    training_params.pretrained_model = value
    trainer = build(
        base_trainer.BaseTrainer,
        trainer_env,
        training_params,
        patch_targets=(base_trainer,),
    )

    assert trainer.resolve_pretrained_model_path() is None


def test_pretrained_model_that_no_longer_exists_is_ignored(trainer_env):
    """The row can be deleted between picking it in the UI and the run starting."""
    from anylearning.training.trainers import base_trainer

    training_params = params("resnet18", "lightweight")
    training_params.pretrained_model = "424242"
    trainer = build(
        base_trainer.BaseTrainer,
        trainer_env,
        training_params,
        patch_targets=(base_trainer,),
    )

    assert trainer.resolve_pretrained_model_path() is None


def test_pretrained_model_resolves_to_a_path_under_the_project(trainer_env):
    from sqlalchemy.orm import Session

    from anylearning.database import Model
    from anylearning.training.trainers import base_trainer

    manager = trainer_env["manager"]
    with Session(manager.get_project_engine(trainer_env["project_id"])) as session:
        model = Model(name="m", path="run-7/best.pt")
        session.add(model)
        session.commit()
        model_id = model.id

    training_params = params("resnet18", "lightweight")
    training_params.pretrained_model = str(model_id)
    trainer = build(
        base_trainer.BaseTrainer,
        trainer_env,
        training_params,
        patch_targets=(base_trainer,),
    )

    resolved = trainer.resolve_pretrained_model_path()
    assert resolved is not None
    # Compared as path parts: on Windows the separator is a backslash.
    assert pathlib.Path(resolved).parts[-2:] == ("run-7", "best.pt")
    assert str(trainer_env["project_id"]) in resolved


def test_base_export_onnx_is_a_placeholder(trainer_env):
    """The base writes a stub; every real trainer must override it."""
    from anylearning.training.trainers import base_trainer

    trainer = build(
        base_trainer.BaseTrainer,
        trainer_env,
        params("resnet18", "lightweight"),
        patch_targets=(base_trainer,),
    )
    path = trainer.export_onnx()
    with open(path) as stub:
        assert "placeholder" in stub.read().lower()


def test_trainer_builder_covers_every_project_type():
    from anylearning import config
    from anylearning.training.trainers.trainer_builder import TrainerBuilder

    for project_type in config.MODEL_VARIANTS:
        assert TrainerBuilder.get_trainer_class(project_type) is not None


# --------------------------------------------------------------------------
# NanoDet
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", ["lightweight", "medium", "large"])
def test_nanodet_config_agrees_on_the_class_count(trainer_env, size):
    from anylearning.training.trainers import nanodet_trainer

    trainer = build(
        nanodet_trainer.NanoDetTrainer,
        trainer_env,
        params("nanodet", size),
        patch_targets=(nanodet_trainer,),
    )
    write_labels(trainer, trainer_env["labels"])

    config = yaml.safe_load(trainer.prepare_config())

    expected = len(trainer_env["labels"])
    assert config["model"]["arch"]["head"]["num_classes"] == expected
    assert len(config["class_names"]) == expected
    if "aux_head" in config["model"]["arch"]:
        assert config["model"]["arch"]["aux_head"]["num_classes"] == expected


@pytest.mark.parametrize("size", ["lightweight", "medium", "large"])
def test_nanodet_config_carries_the_training_parameters(trainer_env, size):
    from anylearning.training.trainers import nanodet_trainer

    trainer = build(
        nanodet_trainer.NanoDetTrainer,
        trainer_env,
        params("nanodet", size, epochs=7, batch_size=3, lr=0.012),
        patch_targets=(nanodet_trainer,),
    )
    write_labels(trainer, trainer_env["labels"])

    config = yaml.safe_load(trainer.prepare_config())

    assert config["schedule"]["total_epochs"] == 7
    assert config["device"]["batchsize_per_gpu"] == 3
    assert config["schedule"]["optimizer"]["lr"] == pytest.approx(0.012)


def test_nanodet_class_names_are_ordered_by_label_id(trainer_env):
    """Out-of-order ids would silently swap the classes a model predicts."""
    from anylearning.training.trainers import nanodet_trainer

    trainer = build(
        nanodet_trainer.NanoDetTrainer,
        trainer_env,
        params("nanodet", "lightweight"),
        patch_targets=(nanodet_trainer,),
    )
    write_labels(trainer, [{"name": "dog", "id": 1}, {"name": "cat", "id": 0}])

    config = yaml.safe_load(trainer.prepare_config())
    assert config["class_names"] == ["cat", "dog"]


def test_nanodet_config_points_at_the_training_folders(trainer_env):
    from anylearning.training.trainers import nanodet_trainer

    trainer = build(
        nanodet_trainer.NanoDetTrainer,
        trainer_env,
        params("nanodet", "lightweight"),
        patch_targets=(nanodet_trainer,),
    )
    write_labels(trainer, trainer_env["labels"])

    config = yaml.safe_load(trainer.prepare_config())

    assert config["data"]["train"]["img_path"] == str(trainer.data_folder / "train")
    assert config["data"]["val"]["img_path"] == str(trainer.data_folder / "val")
    assert config["save_dir"] == str(trainer.output_folder)


# --------------------------------------------------------------------------
# Semantic segmentation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ["resnet18", "resnet34", "resnet50"])
def test_semseg_config_reserves_a_background_class(trainer_env, architecture):
    """num_classes counts foreground; the model adds background itself."""
    from anylearning.training.trainers import semseg_trainer

    trainer = build(
        semseg_trainer.SemSegTrainer,
        trainer_env,
        params(architecture, "medium"),
        patch_targets=(semseg_trainer,),
    )
    write_labels(trainer, trainer_env["labels"])

    config = yaml.safe_load(trainer.prepare_config())

    assert config["model"]["arch"] == architecture
    assert config["data"]["label_set"]


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ["resnet18", "resnet34"])
def test_classification_config_matches_the_label_count(trainer_env, architecture):
    from anylearning.training.trainers import classification_trainer

    trainer = build(
        classification_trainer.ClassificationTrainer,
        trainer_env,
        params(architecture, "lightweight"),
        patch_targets=(classification_trainer,),
    )
    write_labels(trainer, trainer_env["labels"])

    config = yaml.safe_load(trainer.prepare_config())

    assert config["model"]["arch"] == architecture
    assert config["model"]["num_classes"] == len(trainer_env["labels"])


# --------------------------------------------------------------------------
# Handpose
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", ["lightweight", "medium", "large"])
def test_handpose_config_sets_the_output_units(trainer_env, size):
    """The MLP head width has to equal the number of gesture classes."""
    from anylearning.training.trainers import handpose_classification_trainer as hp

    trainer = build(
        hp.HandposeClassificationTrainer,
        trainer_env,
        params("mlp", size),
        patch_targets=(hp,),
    )
    write_labels(trainer, trainer_env["labels"])

    config = yaml.safe_load(trainer.prepare_config())

    assert config["models"]["arch"]["head"]["output_units"] == len(
        trainer_env["labels"]
    )
    assert config["schedule"]["epochs"] == 3

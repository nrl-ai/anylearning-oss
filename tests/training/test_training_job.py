"""The orchestration in `run_training_job`.

This is the fixed order the trainer contract depends on -- prepare_data ->
prepare_config -> train -> export_onnx -> get_model_path -> copy and register --
and the place its failure modes are decided. It runs in its own process, so
nothing it raises reaches the request; the only evidence a user gets is the
session status and the log rows it writes.

The rule worth pinning down: the model is registered *after* export succeeds, so
an export failure discards an otherwise-good training run. That is deliberate,
but it means the ordering here is load-bearing rather than incidental.
"""

import json

import pytest

from anylearning.database import TrainingParams


@pytest.fixture
def job_env(tmp_path, monkeypatch):
    """A project and a training session, with all paths inside tmp_path."""
    from anylearning import config, database
    from anylearning.database import Project, TrainingSession
    from sqlalchemy.orm import Session

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setattr(config, "PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "main.db"))

    manager = database.DatabaseManager()
    monkeypatch.setattr(database, "db_manager", manager)

    from anylearning.training import logging as training_logging
    from anylearning.training import training_job

    for module in (training_job, training_logging):
        monkeypatch.setattr(module, "db_manager", manager, raising=False)
    monkeypatch.setattr(training_job, "config", config, raising=False)

    with Session(manager.main_engine) as session:
        project = Project(name="P", type="Image Classification", labels=[])
        session.add(project)
        session.commit()
        project_id = project.id

    (projects_root / str(project_id)).mkdir(parents=True, exist_ok=True)

    with Session(manager.get_project_engine(project_id)) as session:
        # A labelled image in each of train and val: the job refuses to start
        # without them, because a split with nothing in it used to fail from
        # inside whichever framework the trainer wrapped.
        from anylearning.database import DataItem

        for subset in (0, 1):
            session.add(
                DataItem(
                    path=f"image_{subset}.png",
                    original_name=f"image_{subset}.png",
                    subset=subset,
                    labeled=1,
                    class_id=0,
                )
            )

        training_session = TrainingSession(
            name="run", description="", status="not_started"
        )
        session.add(training_session)
        session.commit()
        session_id = training_session.id

    yield {
        "manager": manager,
        "projects_root": projects_root,
        "project_id": project_id,
        "session_id": session_id,
        "monkeypatch": monkeypatch,
        "module": training_job,
    }
    manager.dispose_all()


def params():
    return TrainingParams(
        model_architecture="resnet18",
        model_size="lightweight",
        model_variant="ResNet18-Lightweight",
        batch_size=2,
        epochs=1,
        learning_rate=0.001,
        pretrained_model="default",
    )


class FakeTrainer:
    """Records the call order and stands in for a real trainer.

    `calls` is class-level so the test can read it back without holding the
    instance the job constructs internally.
    """

    calls = []
    checkpoint = None
    onnx = None
    fail_in = None

    def __init__(self, training_folder, logger, project_id, training_params):
        import pathlib

        self.training_folder = pathlib.Path(training_folder)
        self.logger = logger
        # BaseTrainer.prepare_folders() does this; the job only computes the
        # path, so without it there is nothing for the cleanup step to remove.
        self.training_folder.mkdir(parents=True, exist_ok=True)

    def _step(self, name):
        type(self).calls.append(name)
        if type(self).fail_in == name:
            raise RuntimeError(f"{name} failed")

    def prepare_data(self):
        self._step("prepare_data")

    def prepare_config(self):
        self._step("prepare_config")
        return "config: yes"

    def train(self):
        self._step("train")

    def export_onnx(self):
        self._step("export_onnx")
        return str(type(self).onnx) if type(self).onnx else None

    def companion_files(self):
        return []

    def get_model_path(self):
        self._step("get_model_path")
        if type(self).checkpoint is None:
            return False, None
        return True, str(type(self).checkpoint)


@pytest.fixture
def fake_trainer(job_env, tmp_path):
    """Installs FakeTrainer as the trainer for the project's type."""
    from anylearning.training.trainers.trainer_builder import TrainerBuilder

    FakeTrainer.calls = []
    FakeTrainer.fail_in = None

    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"weights")
    FakeTrainer.checkpoint = checkpoint

    onnx = tmp_path / "exported_model.onnx"
    onnx.write_bytes(b"onnx")
    FakeTrainer.onnx = onnx

    # Two arguments: the job passes the run's architecture as well as the
    # project's type, because a detection project can be trained by either of
    # two trainers.
    job_env["monkeypatch"].setattr(
        TrainerBuilder,
        "get_trainer_class",
        staticmethod(lambda _type, _architecture=None: FakeTrainer),
    )
    return FakeTrainer


def run(job_env):
    job_env["module"].run_training_job(
        job_env["project_id"], job_env["session_id"], params()
    )


def read_session(job_env):
    from anylearning.database import TrainingSession
    from sqlalchemy.orm import Session

    with Session(
        job_env["manager"].get_project_engine(job_env["project_id"])
    ) as session:
        return (
            session.query(TrainingSession).filter_by(id=job_env["session_id"]).first()
        )


def read_models(job_env):
    from anylearning.database import Model
    from sqlalchemy.orm import Session

    with Session(
        job_env["manager"].get_project_engine(job_env["project_id"])
    ) as session:
        return session.query(Model).all()


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_runs_the_trainer_steps_in_the_documented_order(job_env, fake_trainer):
    run(job_env)

    assert fake_trainer.calls == [
        "prepare_data",
        "prepare_config",
        "train",
        "export_onnx",
        "get_model_path",
    ]


def test_finished_run_registers_a_model_and_copies_both_artefacts(
    job_env, fake_trainer
):
    run(job_env)

    assert read_session(job_env).status == "finished"

    models = read_models(job_env)
    assert len(models) == 1
    model = models[0]
    assert model.model_architecture == "resnet18"
    assert model.exported_path is not None

    models_folder = job_env["projects_root"] / str(job_env["project_id"]) / "models"
    assert (models_folder / model.path).is_file()
    assert (models_folder / model.exported_path).is_file()


def test_the_config_is_persisted_on_the_session(job_env, fake_trainer):
    """Inference reads the stored config, not the template it came from."""
    run(job_env)
    assert read_session(job_env).config_file == "config: yes"


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step", ["prepare_data", "prepare_config", "train"])
def test_a_failed_step_marks_the_session_as_error_and_registers_nothing(
    job_env, fake_trainer, step
):
    fake_trainer.fail_in = step

    with pytest.raises(RuntimeError):
        run(job_env)

    session = read_session(job_env)
    assert session.status == "error"
    # The reason reaches the user through the log rows the UI polls -- there is
    # no error column on the session.
    assert f"{step} failed" in (session.training_logs or "")
    assert read_models(job_env) == []


def test_an_export_failure_discards_an_otherwise_good_run(job_env, fake_trainer):
    """The documented consequence of registering the model only after export.

    Training has already succeeded at this point, so this is the case where a
    user loses finished work -- worth pinning so it cannot change silently.
    """
    fake_trainer.fail_in = "export_onnx"

    with pytest.raises(RuntimeError):
        run(job_env)

    assert "train" in fake_trainer.calls, "training really did complete"
    assert read_session(job_env).status == "error"
    assert read_models(job_env) == []


def test_a_missing_checkpoint_is_an_error_not_an_empty_model(job_env, fake_trainer):
    fake_trainer.checkpoint = None

    with pytest.raises(RuntimeError):
        run(job_env)

    assert read_session(job_env).status == "error"
    assert read_models(job_env) == []


def test_a_run_without_onnx_still_registers_the_checkpoint(job_env, fake_trainer):
    """export_onnx returning None is a warning, not a failure.

    Only a raising export loses the run; a trainer that legitimately has no
    ONNX to offer still gets its model registered, with no exported_path.
    """
    fake_trainer.onnx = None

    run(job_env)

    models = read_models(job_env)
    assert len(models) == 1
    assert models[0].exported_path is None
    assert read_session(job_env).status == "finished"


def test_every_run_stamps_ended_at(job_env, fake_trainer):
    """The finally block runs on both paths; the UI shows a run as open without it."""
    fake_trainer.fail_in = "train"
    with pytest.raises(RuntimeError):
        run(job_env)
    assert read_session(job_env).ended_at is not None


# --------------------------------------------------------------------------
# Working folder
# --------------------------------------------------------------------------


def test_development_mode_keeps_the_training_folder(job_env, fake_trainer, monkeypatch):
    """`--development` preserving the folder is what makes a failed job debuggable."""
    monkeypatch.setenv("ANYLEARNING_DEVELOPMENT", "TRUE")

    run(job_env)

    training_folder = (
        job_env["projects_root"]
        / str(job_env["project_id"])
        / "training"
        / str(job_env["session_id"])
    )
    assert training_folder.is_dir()


def test_the_training_folder_is_removed_otherwise(job_env, fake_trainer, monkeypatch):
    monkeypatch.delenv("ANYLEARNING_DEVELOPMENT", raising=False)

    run(job_env)

    training_folder = (
        job_env["projects_root"]
        / str(job_env["project_id"])
        / "training"
        / str(job_env["session_id"])
    )
    assert not training_folder.exists()


def test_metric_logs_stored_as_a_json_string_are_normalised(job_env, fake_trainer):
    """Some trainers write metric_logs as text; the model's test_result reads
    the last entry, which indexes a string by character without this."""
    from anylearning.database import TrainingSession
    from sqlalchemy.orm import Session

    with Session(
        job_env["manager"].get_project_engine(job_env["project_id"])
    ) as session:
        row = session.query(TrainingSession).filter_by(id=job_env["session_id"]).first()
        row.metric_logs = json.dumps([{"epoch": 0}, {"epoch": 1, "acc": 0.5}])
        session.commit()

    run(job_env)

    assert read_models(job_env)[0].test_result == {"epoch": 1, "acc": 0.5}

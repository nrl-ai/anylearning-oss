"""Training log writers.

Every trainer routes its progress through these, and the framework adapters
(NanoDetLogger, MLPLogger) are what Lightning calls during a run. A missing
method here does not surface until a training job is already underway, which is
the worst time to find it.
"""

from unittest.mock import MagicMock

import pytest

from anylearning.training.logging import (
    ConsoleLogsWriter,
    MLPLogger,
    NanoDetLogger,
    TrainingLogsWriter,
)


class RecordingWriter:
    """Captures what an adapter forwards, without touching a database."""

    def __init__(self):
        self.messages = []
        self.metrics = []

    def write(self, message):
        self.messages.append(message)

    def write_metrics(self, metrics):
        self.metrics.append(metrics)


# --------------------------------------------------------------------------
# ConsoleLogsWriter
# --------------------------------------------------------------------------


def test_console_writer_prints_messages(capsys):
    ConsoleLogsWriter().write("training started")
    assert "training started" in capsys.readouterr().out


def test_console_writer_prints_metrics(capsys):
    ConsoleLogsWriter().write_metrics({"loss": 0.5})
    assert "loss" in capsys.readouterr().out


def test_console_writer_satisfies_the_writer_interface():
    """It stands in for TrainingLogsWriter, so it must match its surface."""
    required = {"write", "write_metrics"}
    assert required <= set(dir(ConsoleLogsWriter))
    for name in required:
        assert callable(getattr(ConsoleLogsWriter, name))


def test_console_writer_needs_no_project_or_session():
    """The whole point: usable from a script, notebook or test."""
    ConsoleLogsWriter()  # must not raise


# --------------------------------------------------------------------------
# NanoDetLogger / MLPLogger adapters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("logger_class", [NanoDetLogger, MLPLogger])
def test_adapter_forwards_each_severity(logger_class, tmp_path):
    writer = RecordingWriter()
    logger = logger_class(writer=writer, save_dir=str(tmp_path))

    logger.info("plain")
    logger.warning("careful")
    logger.error("broken")

    assert writer.messages[0] == "plain"
    assert "careful" in writer.messages[1] and "Warning" in writer.messages[1]
    assert "broken" in writer.messages[2] and "Error" in writer.messages[2]


@pytest.mark.parametrize("logger_class", [NanoDetLogger, MLPLogger])
def test_adapter_lifecycle_hooks_do_not_raise(logger_class, tmp_path):
    """Lightning calls these; none may explode mid-run."""
    writer = RecordingWriter()
    logger = logger_class(writer=writer, save_dir=str(tmp_path))

    logger.finalize("success")
    logger.log_graph(MagicMock())
    logger.dump_cfg({"a": 1})
    logger.save()
    logger.after_save_checkpoint(MagicMock())

    assert len(writer.messages) == 5


@pytest.mark.parametrize("logger_class", [NanoDetLogger, MLPLogger])
def test_adapter_has_the_name_and_version_lightning_reads(logger_class, tmp_path):
    logger = logger_class(writer=RecordingWriter(), save_dir=str(tmp_path))
    assert isinstance(logger.name, str) and logger.name
    assert isinstance(logger.version, str) and logger.version


def test_nanodet_logger_records_detection_metrics(tmp_path):
    writer = RecordingWriter()
    logger = NanoDetLogger(writer=writer, save_dir=str(tmp_path))

    logger.log_metrics({"train_loss": 1.5, "val_loss": 1.2, "mAP": 0.42}, epoch=3)

    [metrics] = writer.metrics
    assert metrics["Training Loss"] == 1.5
    assert metrics["Validation Loss"] == 1.2
    assert metrics["Validation mAP"] == 0.42


def test_mlp_logger_records_classification_metrics(tmp_path):
    writer = RecordingWriter()
    logger = MLPLogger(writer=writer, save_dir=str(tmp_path))

    logger.log_metrics({"train_loss": 0.9, "val_loss": 0.8, "val_accuracy": 0.77}, epoch=1)

    [metrics] = writer.metrics
    assert metrics["Training Loss"] == 0.9
    assert metrics["Validation Loss"] == 0.8
    assert metrics["Validation Accuracy"] == 0.77


@pytest.mark.parametrize("logger_class", [NanoDetLogger, MLPLogger])
def test_adapter_defaults_absent_metrics_to_zero(logger_class, tmp_path):
    """An early epoch may not have every metric yet; that must not KeyError."""
    writer = RecordingWriter()
    logger = logger_class(writer=writer, save_dir=str(tmp_path))

    logger.log_metrics({}, epoch=0)

    [metrics] = writer.metrics
    assert all(value == 0 for value in metrics.values())


def test_nanodet_logger_experiment_is_lazy(tmp_path):
    """SummaryWriter creates files, so it must not be built until used."""
    logger = NanoDetLogger(writer=RecordingWriter(), save_dir=str(tmp_path))
    assert logger._experiment is None

    experiment = logger.experiment
    assert experiment is not None
    # Cached, not rebuilt per access.
    assert logger.experiment is experiment


def test_add_scalars_reaches_the_experiment(tmp_path):
    logger = NanoDetLogger(writer=RecordingWriter(), save_dir=str(tmp_path))
    fake = MagicMock()
    logger._experiment = fake

    logger.add_scalars("Loss", {"Train": 0.1}, 5)
    fake.add_scalars.assert_called_once_with("Loss", {"Train": 0.1}, 5)


# --------------------------------------------------------------------------
# TrainingLogsWriter
# --------------------------------------------------------------------------


def test_training_logs_writer_keeps_its_identifiers():
    writer = TrainingLogsWriter(project_id=4, training_session_id=9)
    assert writer.project_id == 4
    assert writer.training_session_id == 9


def test_training_logs_writer_warns_rather_than_raises_for_a_missing_session(
    capsys, tmp_path, monkeypatch
):
    """A deleted session must not take the training process down.

    Exercised against a real (empty) database rather than a mocked Session:
    the previous version stubbed the exact ORM calls the implementation made,
    so it silently stopped testing anything when the writer switched to a SQL
    append.
    """
    from anylearning import config, database
    from anylearning.training.logging import TrainingLogsWriter

    projects_root = tmp_path / "projects"
    (projects_root / "1").mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "main.db"))

    manager = database.DatabaseManager()
    monkeypatch.setattr(database, "db_manager", manager)
    from anylearning.training import logging as logging_module

    monkeypatch.setattr(logging_module, "db_manager", manager)

    try:
        TrainingLogsWriter(project_id=1, training_session_id=123).write("hello")
    finally:
        manager.dispose_all()

    assert "not found" in capsys.readouterr().out


def test_training_logs_writer_appends_without_rewriting_earlier_lines(
    tmp_path, monkeypatch
):
    """Each line is appended in SQL, so the log accumulates in order.

    The writer used to read the whole column back, concatenate in Python and
    write it again -- quadratic over a run that logs every iteration.
    """
    from anylearning import config, database
    from anylearning.database import TrainingSession
    from anylearning.training.logging import TrainingLogsWriter
    from sqlalchemy.orm import Session

    projects_root = tmp_path / "projects"
    (projects_root / "1").mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "main.db"))

    manager = database.DatabaseManager()
    monkeypatch.setattr(database, "db_manager", manager)
    from anylearning.training import logging as logging_module

    monkeypatch.setattr(logging_module, "db_manager", manager)

    try:
        with Session(manager.get_project_engine(1)) as session:
            row = TrainingSession(name="run", description="", status="training")
            session.add(row)
            session.commit()
            session_id = row.id

        writer = TrainingLogsWriter(project_id=1, training_session_id=session_id)
        for i in range(5):
            writer.write(f"line {i}")

        with Session(manager.get_project_engine(1)) as session:
            logs = session.get(TrainingSession, session_id).training_logs
    finally:
        manager.dispose_all()

    lines = [line for line in logs.splitlines() if line.strip()]
    assert len(lines) == 5
    assert [line.split("] ", 1)[1] for line in lines] == [f"line {i}" for i in range(5)]

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from anylearning.database import (
    DatabaseManager,
    DataItem,
    Dataset,
    Model,
    Project,
    TrainingParams,
    TrainingProcess,
    TrainingSession,
    TrainingSessionStatus,
)


@pytest.fixture
def db_manager():
    """A DatabaseManager that actually closes its engines afterwards.

    Returning it without teardown left SQLite connections open; Python 3.13
    reports those as unraisable ResourceWarnings when they are eventually
    collected, which surfaces as a failure in whatever unrelated test happens to
    be running at the time.
    """
    manager = DatabaseManager()
    yield manager
    manager.dispose_all()


def test_init(db_manager):
    assert db_manager.main_engine is not None
    assert db_manager.project_connections == {}


def test_get_project_engine(db_manager, tmp_path):
    with patch("anylearning.config.PROJECTS_ROOT", tmp_path):
        project_id = "123"
        os.makedirs(os.path.join(tmp_path, project_id), exist_ok=True)
        engine = db_manager.get_project_engine(project_id)

        assert engine is not None
        assert project_id in db_manager.project_connections
        assert os.path.exists(os.path.join(tmp_path, project_id, "database.db"))

        # Test caching works
        engine2 = db_manager.get_project_engine(project_id)
        assert engine is engine2


def test_release_database(db_manager, tmp_path):
    with patch("anylearning.config.PROJECTS_ROOT", tmp_path):
        project_id = "123"
        os.makedirs(os.path.join(tmp_path, project_id), exist_ok=True)
        db_manager.get_project_engine(project_id)
        assert project_id in db_manager.project_connections

        db_manager.release_database(project_id)
        assert project_id not in db_manager.project_connections


def test_get_project_connection(db_manager, tmp_path):
    with patch("anylearning.config.PROJECTS_ROOT", tmp_path):
        project_id = "123"
        os.makedirs(os.path.join(tmp_path, project_id), exist_ok=True)

        with db_manager.get_project_connection(project_id) as conn:
            assert conn is not None

        # Connection should be closed after context exit
        assert conn.closed


def test_project_model():
    project = Project(
        name="test",
        type="classification",
        description="test project",
        path="/test",
        size=1.5,
        dataset="test_dataset",
        num_train=100,
        num_val=10,
        num_test=10,
        labels=["label1", "label2"],
    )

    assert project.name == "test"
    assert project.type == "classification"
    assert project.description == "test project"
    assert project.path == "/test"
    assert project.size == 1.5
    assert project.dataset == "test_dataset"
    assert project.num_train == 100
    assert project.num_val == 10
    assert project.num_test == 10
    assert project.labels == ["label1", "label2"]


def test_dataset_model():
    dataset = Dataset(
        train_version="1.0",
        val_version="1.0",
        test_version="1.0",
        modified_at=datetime.now(timezone.utc),
    )

    assert dataset.train_version == "1.0"
    assert dataset.val_version == "1.0"
    assert dataset.test_version == "1.0"
    assert isinstance(dataset.modified_at, datetime)
    assert dataset.modified_at.tzinfo == timezone.utc


def test_data_item_model():
    data_item = DataItem(
        dataset_id=1,
        subset=0,
        path="/test/image.jpg",
        labeled=True,
        class_id=1,
        annotation={"bbox": [0, 0, 100, 100]},
        original_name="image.jpg",
    )

    assert data_item.dataset_id == 1
    assert data_item.subset == 0
    assert data_item.path == "/test/image.jpg"
    assert data_item.labeled is True
    assert data_item.class_id == 1
    assert data_item.annotation == {"bbox": [0, 0, 100, 100]}
    assert data_item.original_name == "image.jpg"


def test_training_session_model():
    session = TrainingSession(
        name="test_training",
        description="test session",
        status=TrainingSessionStatus.NOT_STARTED.value,
        metric_logs={"loss": [0.1, 0.2]},
        training_logs="Training log",
        params={"lr": 0.001},
        config_file="config.yaml",
    )

    assert session.name == "test_training"
    assert session.description == "test session"
    assert session.status == TrainingSessionStatus.NOT_STARTED.value
    assert session.metric_logs == {"loss": [0.1, 0.2]}
    assert session.training_logs == "Training log"
    assert session.params == {"lr": 0.001}
    assert session.config_file == "config.yaml"


def test_training_process_model():
    process = TrainingProcess(
        training_session_id=1, pid=1234, status="running", exit_code=None
    )

    assert process.training_session_id == 1
    assert process.pid == 1234
    assert process.status == "running"
    assert process.exit_code is None


def test_model_model():
    model = Model(
        training_session_id=1,
        name="test_model",
        description="test model",
        path="/models/test",
        config_file="config.yaml",
        exported_path="/exports/test",
        model_architecture="resnet",
        model_size="18",
        test_version="1.0",
        test_result={"accuracy": 0.95},
    )

    assert model.training_session_id == 1
    assert model.name == "test_model"
    assert model.description == "test model"
    assert model.path == "/models/test"
    assert model.config_file == "config.yaml"
    assert model.exported_path == "/exports/test"
    assert model.model_architecture == "resnet"
    assert model.model_size == "18"
    assert model.test_version == "1.0"
    assert model.test_result == {"accuracy": 0.95}


def test_training_params_model():
    params = TrainingParams(
        model_architecture="resnet",
        model_size="18",
        model_variant="b",
        batch_size=32,
        epochs=100,
        learning_rate=0.001,
        pretrained_model="imagenet",
    )

    assert params.model_architecture == "resnet"
    assert params.model_size == "18"
    assert params.model_variant == "b"
    assert params.batch_size == 32
    assert params.epochs == 100
    assert params.learning_rate == 0.001
    assert params.pretrained_model == "imagenet"

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship

from anylearning import config

Base = declarative_base()


class ProjectBase(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    path: Optional[str] = None
    size: Optional[float] = None
    dataset: Optional[str] = None
    num_train: Optional[int] = None
    num_val: Optional[int] = None
    num_test: Optional[int] = None
    labels: Optional[list] = None


class ProjectCreate(ProjectBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    type = Column(String, index=True)
    description = Column(String)
    path = Column(String)
    size = Column(Float)
    dataset = Column(String)
    num_train = Column(Integer)
    num_val = Column(Integer)
    num_test = Column(Integer)
    num_trained_models = Column(Integer)
    new_models_this_month = Column(Integer)
    labels = Column(JSON)

    def __repr__(self):
        return f"Project(id={self.id}, name={self.name}, created_at={self.created_at}, type={self.type}, description={self.description}, path={self.path}, size={self.size}, dataset={self.dataset}, num_train={self.num_train}, num_val={self.num_val}, num_test={self.num_test}, labels={self.labels})"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    train_version = Column(String)
    val_version = Column(String)
    test_version = Column(String)
    modified_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    data_items = relationship("DataItem", back_populates="dataset")


class DataItem(Base):
    __tablename__ = "data_items"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    subset = Column(Integer)  # 0: train, 1: val, 2: test
    path = Column(String)
    labeled = Column(Boolean, default=False)
    class_id = Column(Integer, default=-1)
    annotation = Column(JSON)
    original_name = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    modified_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    dataset = relationship("Dataset", back_populates="data_items")


class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)
    status = Column(String)  # not_started/training/evaluating/error
    metric_logs = Column(JSON)
    training_logs = Column(String)
    params = Column(JSON)
    config_file = Column(String)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime)

    model = relationship("Model", back_populates="training_session", uselist=False)
    process = relationship(
        "TrainingProcess", back_populates="training_session", uselist=False
    )


class TrainingProcess(Base):
    __tablename__ = "training_processes"

    id = Column(Integer, primary_key=True, index=True)
    training_session_id = Column(Integer, ForeignKey("training_sessions.id"))
    pid = Column(Integer)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime)
    status = Column(String)  # running/terminated/completed
    exit_code = Column(Integer)

    training_session = relationship("TrainingSession", back_populates="process")


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    training_session_id = Column(Integer, ForeignKey("training_sessions.id"))
    name = Column(String)
    description = Column(String)
    path = Column(String)
    config_file = Column(String)
    exported_path = Column(String)
    model_architecture = Column(String)
    model_size = Column(String)
    test_version = Column(String)
    test_result = Column(JSON)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    modified_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    training_session = relationship("TrainingSession", back_populates="model")


class TrainingSessionStatus(Enum):
    NOT_STARTED = "not_started"
    TRAINING = "training"
    EVALUATING = "evaluating"
    ERROR = "error"
    FINISHED = "finished"
    TERMINATED = "terminated"


class TrainingParams(BaseModel):
    # "model_" is a protected namespace in Pydantic v2; these names are part of
    # the existing API and database payloads, so opt out of the warning instead.
    model_config = ConfigDict(protected_namespaces=())

    model_architecture: str
    model_size: str
    model_variant: str
    batch_size: int
    epochs: int
    learning_rate: float
    pretrained_model: str
    # "auto" (GPU when there is one), "gpu" or "cpu". Defaulted rather than
    # required: every training session already stored in a project database was
    # written before this field existed, and they still have to load.
    device: str = "auto"
    # None means "whatever the model's config template says", which is what
    # every run before this field did.
    image_size: Optional[int] = None
    # None means each trainer keeps the augmentation it has always applied.
    # A dict overrides it -- see BaseTrainer.resolve_augmentation. Not a nested
    # model, because it is stored as JSON on rows written by older versions and
    # a strict schema would refuse to load them.
    augmentation: Optional[Dict[str, Any]] = None


class DatabaseManager:
    def __init__(self):
        # Main DB for project metadata
        self.main_engine = create_engine(f"sqlite:///{config.DATABASE_PATH}")
        Base.metadata.create_all(self.main_engine)
        self.project_connections = {}

    def get_project_engine(self, project_id):
        if project_id not in self.project_connections:
            project_db_path = os.path.join(
                config.PROJECTS_ROOT, str(project_id), "database.db"
            )
            url = f"sqlite:///{project_db_path}"
            logging.info(f"Creating project engine for {project_id} at {url}")
            engine = create_engine(url)
            Base.metadata.create_all(engine)
            self.project_connections[project_id] = engine
        return self.project_connections[project_id]

    def release_database(self, project_id):
        if project_id in self.project_connections:
            engine = self.project_connections[project_id]
            engine.dispose()
            del self.project_connections[project_id]

    def refresh_connection(self, project_id):
        self.release_database(project_id)
        return self.get_project_engine(project_id)

    def dispose_all(self):
        """Dispose every cached project engine, and the main one.

        Engines are cached per project and previously only ever released one at
        a time, so nothing closed them at shutdown. Python 3.13 reports the
        surviving sqlite3 connections as unraisable ResourceWarnings when they
        are finally garbage collected, which is the visible symptom of a real
        leak in a long-running desktop process.
        """
        for project_id in list(self.project_connections):
            self.release_database(project_id)
        self.main_engine.dispose()

    @contextmanager
    def get_project_connection(self, project_id):
        engine = self.get_project_engine(project_id)
        conn = engine.connect()
        try:
            yield conn
        finally:
            conn.close()


db_manager = DatabaseManager()

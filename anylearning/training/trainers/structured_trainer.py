"""Trainer adapter that lets structured runs use the existing job lifecycle."""

from __future__ import annotations

import json
import pathlib

import numpy as np

from anylearning import config
from anylearning.structured.modeling import train_tabular, train_text
from anylearning.structured.store import load_metadata
from anylearning.training.trainers.base_trainer import BaseTrainer


class StructuredTrainer(BaseTrainer):
    AUGMENTATIONS = ()

    def __init__(self, training_folder, logger, project_id, training_params):
        super().__init__(training_folder, logger, project_id, training_params)
        self.metadata = load_metadata(project_id, required=True)
        if not self.metadata.get("configured"):
            raise ValueError("Configure the dataset task and target before training.")
        self.artifact: pathlib.Path | None = None
        self.report: dict | None = None

    def prepare_data(self):
        source = (
            pathlib.Path(config.PROJECTS_ROOT)
            / str(self.project_id)
            / "structured"
            / "rows.parquet"
        )
        if not source.exists():
            raise ValueError("Upload a structured dataset before training.")
        # Structured trainers read the immutable project Parquet directly. A
        # run-local multi-GB copy added latency and disk pressure but provided
        # no isolation: corrections already live in a separate override file.
        self.logger.write(
            f"Prepared {self.metadata['source']['rows']} structured rows from paged Parquet storage."
        )

    def prepare_config(self):
        task = self.metadata["task"]
        ignored = set(task.get("ignored_columns") or [])
        if task.get("target"):
            ignored.add(task["target"])
        if task.get("id_column"):
            ignored.add(task["id_column"])
        if task["type"] == "text_classification":
            features = [task["text_column"]]
        else:
            features = [
                item["name"]
                for item in self.metadata["profile"]
                if item["name"] not in ignored
            ]
        categorical = (
            []
            if task["type"] == "text_classification"
            else [
                item["name"]
                for item in self.metadata["profile"]
                if item["name"] in features and item["type"] != "numeric"
            ]
        )
        value = {
            "version": 1,
            "task": task,
            "split": self.metadata["split"],
            "source": self.metadata["source"],
            "attribution": self.metadata.get("attribution", {}),
            "features": features,
            "categorical_features": categorical,
        }
        self.config_path = self.training_folder / "structured-model.json"
        self.config_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return self.config_path.read_text(encoding="utf-8")

    def train(self):
        task_type = self.metadata["task"]["type"]
        if task_type in {"classification", "regression"}:
            self.artifact, self.report = train_tabular(
                self.project_id, self.output_folder, self.training_params, self.logger
            )
        elif task_type == "text_classification":
            self.artifact, self.report = train_text(
                self.project_id, self.output_folder, self.training_params, self.logger
            )
        else:
            raise ValueError(
                f"{task_type.replace('_', ' ').title()} is evaluated from the Dataset workspace and does not train a predictive model."
            )
        self.logger.write_metrics(self.report["metrics"])

    def export_onnx(self):
        # Mixed-type CatBoost and sparse text pipelines do not have a faithful
        # ONNX export.  The native artifact is portable across supported OSes.
        return None

    def get_model_path(self):
        return (
            self.artifact is not None and self.artifact.exists(),
            str(self.artifact) if self.artifact else None,
        )

    def companion_files(self):
        return [self.output_folder / "report.json"]

    @staticmethod
    def run_inference(config_data: str, model_path: str, image: np.ndarray):
        raise ValueError(
            "Structured models accept rows, not images. Use the structured prediction endpoint."
        )

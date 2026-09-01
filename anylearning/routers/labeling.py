import copy
import hashlib
import logging
import pathlib
from functools import lru_cache
from typing import List, Literal

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from anylearning import config
from anylearning.auto_labeling.model_manager import ModelManager
from anylearning.database import DataItem, Project, db_manager
from anylearning.database import Model as TrainedModel

router = APIRouter(prefix="/api", tags=["Auto Labeling"])
model_manager = ModelManager()

_MAX_TRAINING_CONFIG_BYTES = 1024 * 1024
_MAX_LOCAL_ONNX_BYTES = 20 * 1024**3


@lru_cache(maxsize=128)
def _local_model_sha256(path: str, size: int, modified_ns: int) -> str:
    """Hash a local artifact once per exact file identity."""
    del modified_ns
    digest = hashlib.sha256()
    consumed = 0
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            consumed += len(chunk)
            if consumed > size or consumed > _MAX_LOCAL_ONNX_BYTES:
                raise ValueError("Local ONNX model changed while it was being verified")
            digest.update(chunk)
    if consumed != size:
        raise ValueError("Local ONNX model changed while it was being verified")
    return digest.hexdigest()


def _trained_auto_labeling_models(
    project_id: int, project: Project, *, verify_artifacts: bool
) -> list[dict]:
    """Discover inference-compatible ONNX artifacts produced by training."""
    models_root = (
        pathlib.Path(config.PROJECTS_ROOT) / str(project_id) / "models"
    ).resolve()
    discovered: list[dict] = []
    with Session(db_manager.get_project_engine(project_id)) as session:
        rows = (
            session.query(TrainedModel)
            .filter(TrainedModel.exported_path.is_not(None))
            .order_by(TrainedModel.created_at.desc())
            .all()
        )
        for row in rows:
            architecture = row.model_architecture
            if architecture not in {"rfdetr", "rfdetr-seg"}:
                continue
            candidate = (models_root / str(row.exported_path)).resolve()
            try:
                candidate.relative_to(models_root)
            except ValueError:
                logging.warning("Ignoring trained model with an unsafe exported path")
                continue
            if candidate.suffix.lower() != ".onnx" or not candidate.is_file():
                continue
            try:
                stat_result = candidate.stat()
            except OSError:
                logging.warning(
                    "Ignoring a trained model that changed during discovery"
                )
                continue
            if not 0 < stat_result.st_size <= _MAX_LOCAL_ONNX_BYTES:
                continue

            config_text = row.config_file
            if (
                not isinstance(config_text, str)
                or not 0
                < len(config_text.encode("utf-8"))
                <= _MAX_TRAINING_CONFIG_BYTES
            ):
                continue
            try:
                parsed = yaml.safe_load(config_text)
            except yaml.YAMLError:
                logging.warning("Ignoring a trained model with an invalid config")
                continue
            if not isinstance(parsed, dict):
                continue
            data_config = parsed.get("data")
            if not isinstance(data_config, dict):
                continue
            class_names = data_config.get("class_names")
            if (
                not isinstance(class_names, list)
                or not 0 < len(class_names) <= 10_000
                or any(
                    not isinstance(name, str) or not name or len(name) > 1_024
                    for name in class_names
                )
                or len(class_names) != len(set(class_names))
            ):
                continue

            try:
                digest = (
                    _local_model_sha256(
                        str(candidate), stat_result.st_size, stat_result.st_mtime_ns
                    )
                    if verify_artifacts
                    else None
                )
            except (OSError, ValueError):
                logging.warning("Ignoring a trained model that failed verification")
                continue
            segmentation = architecture == "rfdetr-seg"
            task = "instance_segmentation" if segmentation else "detection"
            discovered.append(
                {
                    "name": f"project-{project_id}-trained-{row.id}",
                    "display_name": f"{row.name} (Trained RF-DETR)",
                    "type": "inference",
                    "backend": "rfdetr_onnx",
                    "tasks": [task],
                    "interaction_mode": "automatic",
                    "output_modes": (
                        ["polygon", "rectangle"] if segmentation else ["rectangle"]
                    ),
                    "project_types": [project.type],
                    "archive_size_bytes": stat_result.st_size,
                    "has_downloaded": True,
                    "is_custom_model": False,
                    "is_project_model": True,
                    "project_id": project_id,
                    # Only its parent directory matters for resolving resources;
                    # no configuration file is read from this sentinel path.
                    "config_file": str(candidate.parent / ".inference.yaml"),
                    "inference_config": {
                        "name": f"project-{project_id}-trained-{row.id}",
                        "model_path": str(candidate),
                        "task": task,
                        "class_names": class_names,
                        "background_class_id": None,
                        "max_detections": 100,
                        "providers": ["CPUExecutionProvider"],
                        "intra_op_threads": 1,
                        "inter_op_threads": 1,
                        **(
                            {
                                "model_revision": f"trained-sha256:{digest}",
                                "sha256": digest,
                            }
                            if digest is not None
                            else {}
                        ),
                    },
                }
            )
    return discovered


def _sync_project_models(
    project_id: int, *, verify_artifacts: bool = False
) -> tuple[str, list]:
    with Session(db_manager.main_engine) as session:
        project = session.query(Project).filter(Project.id == project_id).first()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        model_manager.set_project_model_configs(
            project_id,
            _trained_auto_labeling_models(
                project_id, project, verify_artifacts=verify_artifacts
            ),
        )
        return project.type, copy.deepcopy(project.labels or [])


@router.get("/projects/{project_id}/auto_labeling/models")
def auto_labeling_models(
    project_id: int,
):
    _sync_project_models(project_id)
    public_fields = {
        "name",
        "display_name",
        "has_downloaded",
        "is_custom_model",
        "tasks",
        "interaction_mode",
        "output_modes",
        "project_types",
        "archive_size_bytes",
        "is_project_model",
    }
    public = []
    for model_config in model_manager.get_model_configs():
        item = {
            key: copy.deepcopy(value)
            for key, value in model_config.items()
            if key in public_fields
        }
        item.setdefault("tasks", ["promptable_segmentation"])
        item.setdefault("interaction_mode", "prompted")
        item.setdefault("output_modes", ["polygon", "rectangle"])
        item.setdefault(
            "project_types",
            ["Object Detection", "Image Segmentation", "Instance Segmentation"],
        )
        item.setdefault("archive_size_bytes", 0)
        public.append(item)
    return public


@router.get("/projects/{project_id}/auto_labeling/status")
def auto_labeling_status(
    project_id: int,
):
    _sync_project_models(project_id)
    return {
        "status": model_manager.status,
        "model_name": model_manager.loaded_model_name,
    }


class AutoLabelingLoadModelRequest(BaseModel):
    model_name: str


@router.post("/projects/{project_id}/auto_labeling/load_model")
def auto_labeling_load_model(
    project_id: int,
    request: AutoLabelingLoadModelRequest,
):
    _sync_project_models(project_id, verify_artifacts=True)
    try:
        model_manager.load_model_by_name(request.model_name)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success"}


class AutoLabelingInferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    data_item_id: int
    marks: List[dict] = Field(default_factory=list, max_length=1_024)
    preload_data_item_ids: List[int] = Field(default_factory=list, max_length=7)
    output_shape: Literal["polygon", "rectangle"] | None = None
    parameters: dict[str, float | int] = Field(default_factory=dict, max_length=16)
    #: Backward compatibility for clients that explicitly pre-warm an image.
    #: The current desktop no longer starts an unsolicited dummy inference.
    warm_up: bool = False


@router.post("/projects/{project_id}/auto_labeling/inference")
def auto_labeling_inference(
    project_id: int,
    request: AutoLabelingInferenceRequest,
):
    """
    Inference using auto labeling model.
    Example request body:
    {
        "model_name": "sam2_hiera_small_20240803",
        "data_item_id": 123,
        "marks": [{'data': [78, 53], 'label': 1, 'type': 'point'}, {'data': [115, 56], 'label': 0, 'type': 'point'}, {'data': [178, 50, 222, 102], 'label': 1, 'type': 'rectangle'}],
        "preload_data_item_ids": []
    }
    """
    project_type, stored_labels = _sync_project_models(
        project_id, verify_artifacts=True
    )
    project_labels = tuple(
        label["name"]
        for label in stored_labels
        if isinstance(label, dict)
        and isinstance(label.get("name"), str)
        and label["name"]
    )

    selected_config = next(
        (
            item
            for item in model_manager.get_model_configs()
            if item.get("name") == request.model_name
        ),
        None,
    )
    if selected_config is None:
        raise HTTPException(
            status_code=400, detail=f"Model {request.model_name} not found."
        )
    supported_projects = selected_config.get("project_types")
    if supported_projects and project_type not in supported_projects:
        raise HTTPException(
            status_code=400,
            detail=f"{selected_config['display_name']} does not support {project_type} projects.",
        )

    with Session(db_manager.get_project_engine(project_id)) as session:
        data_item = (
            session.query(DataItem).filter(DataItem.id == request.data_item_id).first()
        )
        if not data_item:
            raise HTTPException(status_code=404, detail="Data item not found")

        project_data_dir = (
            pathlib.Path(config.PROJECTS_ROOT) / str(project_id) / "data"
        ).resolve()

        def _data_path(relative_path: str) -> pathlib.Path:
            candidate = (project_data_dir / relative_path).resolve()
            try:
                candidate.relative_to(project_data_dir)
            except ValueError as error:
                raise HTTPException(
                    status_code=400, detail="Invalid image path"
                ) from error
            return candidate

        full_path = _data_path(data_item.path)
        if not full_path.is_file():
            raise HTTPException(status_code=404, detail="Image file not found")

        # Search for preload data items
        preload_data_items = (
            session.query(DataItem)
            .filter(DataItem.id.in_(request.preload_data_item_ids))
            .all()
        )
        if len(preload_data_items) != len(set(request.preload_data_item_ids)):
            raise HTTPException(status_code=404, detail="Preload data items not found")
        preload_data_items_paths = [
            _data_path(item.path) for item in preload_data_items
        ]
        if not all(path.is_file() for path in preload_data_items_paths):
            raise HTTPException(status_code=404, detail="Preload data items not found")

        try:
            model_manager.load_model_by_name(request.model_name)
            if not model_manager.is_model_ready(request.model_name):
                raise HTTPException(
                    status_code=409,
                    detail="Model is still loading. Wait until it is ready and try again.",
                )
            before = model_manager.status
            auto_labeling_result = model_manager.predict_shapes(
                str(full_path),
                preload_paths=[str(path) for path in preload_data_items_paths],
                allowed_labels=project_labels,
                parameters=request.parameters,
                # A prompt belongs to a prompted model, not to the image. Old
                # marks can still be present when the picker changes to an
                # automatic detector, so do not forward them across modes.
                marks=(
                    request.marks
                    if selected_config.get("interaction_mode") == "prompted"
                    else []
                ),
                output_mode=request.output_shape,
            )
            if request.warm_up:
                # Put the status back: the toolbar shows it, and a warm-up
                # reporting a finished inference is a message about something
                # the user did not do.
                model_manager.status = before
            return {"status": "success", "result": auto_labeling_result}
        except HTTPException:
            raise
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            logging.exception("Auto-labeling inference failed")
            raise HTTPException(
                status_code=500, detail="Auto-labeling inference failed"
            ) from error

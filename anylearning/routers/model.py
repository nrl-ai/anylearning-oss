import base64
import io
import json
import os
import pathlib
import traceback
import zipfile
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from PIL import Image
from pydantic import BaseModel, ConfigDict
from sqlalchemy import or_
from sqlalchemy.orm import Session

from anylearning import config
from anylearning.config import get_model_variant_name
from anylearning.database import Model, Project, TrainingSession, db_manager
from anylearning.training.trainers.trainer_builder import TrainerBuilder


def bytes_to_ndarray(bytes):
    bytes_io = bytearray(bytes)
    img = Image.open(io.BytesIO(bytes_io))
    return np.array(img)


router = APIRouter(prefix="/api", tags=["Model Management"])


class TrainingSessionResponse(BaseModel):
    id: int
    name: str
    description: str
    status: str
    started_at: datetime
    params: dict
    metric_logs: Optional[List[dict]] = None

    model_config = ConfigDict(from_attributes=True)


class ModelResponse(BaseModel):
    id: int
    training_session_id: int
    name: str
    description: str
    path: str
    # These must be explicitly Optional: Pydantic v1 silently treated a None
    # default as Optional, v2 does not, and these columns are nullable.
    model_architecture: Optional[str] = None
    model_size: Optional[str] = None
    model_variant: Optional[str] = None
    test_result: Optional[dict] = None
    metric_logs: Optional[List[dict]] = None
    exported_path: Optional[str] = None
    training_session: Optional[TrainingSessionResponse] = None

    # protected_namespaces=() because "model_" is reserved in Pydantic v2 and
    # these field names are part of the existing API contract.
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    @classmethod
    def from_orm(cls, model, model_variants: List[dict]):
        # Create a ModelResponse instance from the ORM model
        response = super().model_validate(model)
        # Add model_variant field using get_model_variant_name
        response.model_variant = get_model_variant_name(
            model_variants, model.model_architecture, model.model_size
        )
        return response


class ModelListResponse(BaseModel):
    offset: int
    limit: int
    total_count: int
    models: List[ModelResponse]


@router.get("/model-variants")
def get_model_variants():
    return config.MODEL_VARIANTS


#: Separates a project type from a model architecture in the augmentations map.
#:
#: Not "/" or "-": both appear inside architecture names already
#: ("maskrcnn-resnet50", "rfdetr-seg"), and a key a client has to parse is a key
#: that will one day be parsed wrongly.
ARCHITECTURE_SEPARATOR = "::"


@router.get("/augmentations")
def get_augmentations():
    """What each project type can be asked to do to its training images.

    Declared by the trainers themselves, so the dialog offers exactly what the
    model underneath can honour. A shared list would mean showing controls that
    are silently ignored for some project types -- someone turns off flipping to
    protect a LEFT/RIGHT distinction, sees no error, and the run flips anyway.

    Keyed by project type *and*, additionally, by
    ``"<type>::<model_architecture>"`` for every architecture that offers a
    different set. NanoDet warps boxes along with the image and takes rotation
    and colour jitter; RF-DETR's equivalents live behind an optional dependency
    that is not installed, so it declares only a flip. Both are detection, so a
    single answer per project type would have to lie about one of them.

    The plain type keys stay, unchanged, so a client that has not learnt about
    the compound ones keeps working -- it just sees the type's default trainer.
    """
    from anylearning.training.trainers.trainer_builder import TrainerBuilder

    def declared(project_type, architecture=None):
        try:
            trainer = TrainerBuilder.get_trainer_class(project_type, architecture)
        except Exception:
            # A project type whose trainer will not import cannot train at all;
            # /api/health/imports is where that is diagnosed.
            return []
        return [option.as_dict() for option in getattr(trainer, "AUGMENTATIONS", ())]

    options = {}
    for project_type, variants in config.MODEL_VARIANTS.items():
        options[project_type] = declared(project_type)
        for architecture in dict.fromkeys(
            variant["model_architecture"] for variant in variants
        ):
            options[f"{project_type}{ARCHITECTURE_SEPARATOR}{architecture}"] = declared(
                project_type, architecture
            )
    return options


@router.post("/projects/{project_id}/models/{model_id}/inference")
async def model_inference(project_id: int, model_id: int, file: UploadFile = File(...)):
    with Session(db_manager.main_engine) as global_session:
        db_project = (
            global_session.query(Project).filter(Project.id == project_id).first()
        )
        if db_project is None:
            raise HTTPException(status_code=404, detail="Project not found")

    with Session(db_manager.get_project_engine(project_id)) as session:
        model = session.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        if model.config_file is None:
            model.config_file = model.training_session.config_file
            session.commit()

        MODELS_FOLDER = pathlib.Path(config.PROJECTS_ROOT) / str(project_id) / "models"
        model_full_path = MODELS_FOLDER / model.path
        if not model.path or not os.path.exists(model_full_path):
            raise HTTPException(status_code=404, detail="Model file not found")

        # The model's own architecture, not the project's type alone: a
        # detection project can hold both a NanoDet and an RF-DETR model, and
        # asking the wrong one to load the checkpoint fails in a way that reads
        # as a corrupt model rather than as the wrong reader.
        TrainerClass = TrainerBuilder.get_trainer_class(
            db_project.type, model.model_architecture
        )
        try:
            # Read and process the uploaded image
            contents = await file.read()
            img = bytes_to_ndarray(contents)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Run inference
            results, visualization_image = TrainerClass.run_inference(
                model.config_file, model_full_path, img
            )

            # Encode the result image to base64
            _, buffer = cv2.imencode(".jpeg", visualization_image)
            img_base64 = base64.b64encode(buffer).decode("utf-8")
            img_base64 = f"data:image/jpeg;base64,{img_base64}"

            return {"results": results, "visualization_image": img_base64}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred during model inference: {str(e)}",
            )


@router.get("/projects/{project_id}/models", response_model=ModelListResponse)
def list_models(
    project_id: int,
    offset: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    model_architecture: Optional[str] = None,
    model_size: Optional[str] = None,
):
    with Session(db_manager.main_engine) as global_session:
        db_project = (
            global_session.query(Project).filter(Project.id == project_id).first()
        )
        if db_project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        model_variants = config.MODEL_VARIANTS.get(db_project.type, [])

    with Session(db_manager.get_project_engine(project_id)) as session:
        try:
            query = session.query(Model)

            # Apply search filter if provided
            if search:
                search_term = f"%{search}%"
                query = query.filter(
                    or_(
                        Model.name.ilike(search_term),
                        Model.description.ilike(search_term),
                        Model.model_architecture.ilike(search_term),
                    )
                )

            # Apply model architecture filter
            if model_architecture:
                query = query.filter(Model.model_architecture == model_architecture)

            # Apply model size filter
            if model_size:
                query = query.filter(Model.model_size == model_size)

            # Get total count before pagination
            total_count = query.count()

            # Apply pagination
            query = query.order_by(Model.created_at.desc())
            models = query.offset(offset).limit(limit).all()

            return ModelListResponse(
                offset=offset,
                limit=limit,
                total_count=total_count,
                models=[
                    ModelResponse.from_orm(model, model_variants=model_variants)
                    for model in models
                ],
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while fetching models: {str(e)}",
            )


@router.get("/projects/{project_id}/models/{model_id}", response_model=ModelResponse)
def get_model(project_id: int, model_id: int):
    with Session(db_manager.main_engine) as global_session:
        db_project = (
            global_session.query(Project).filter(Project.id == project_id).first()
        )
        if db_project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        model_variants = config.MODEL_VARIANTS.get(db_project.type, [])

    with Session(db_manager.get_project_engine(project_id)) as session:
        model = session.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        training_session = (
            session.query(TrainingSession)
            .filter(TrainingSession.id == model.training_session_id)
            .first()
        )

        response = ModelResponse.from_orm(model, model_variants=model_variants)
        response.metric_logs = (
            json.loads(training_session.metric_logs)
            if (
                training_session.metric_logs
                and isinstance(training_session.metric_logs, str)
            )
            else training_session.metric_logs
        )
        return response


@router.delete("/projects/{project_id}/models/{model_id}")
def delete_model(project_id: int, model_id: int):
    with Session(db_manager.get_project_engine(project_id)) as session:
        model = session.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        try:
            session.delete(model)
            session.commit()
            return {"message": "Model deleted successfully"}
        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while deleting the model: {str(e)}",
            )


class ModelUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router.put("/projects/{project_id}/models/{model_id}", response_model=ModelResponse)
def update_model(project_id: int, model_id: int, model_update: ModelUpdateRequest):
    with Session(db_manager.main_engine) as global_session:
        db_project = (
            global_session.query(Project).filter(Project.id == project_id).first()
        )
        if db_project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        model_variants = config.MODEL_VARIANTS.get(db_project.type, [])

    with Session(db_manager.get_project_engine(project_id)) as session:
        model = session.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        if model_update.name is not None:
            model.name = model_update.name
        if model_update.description is not None:
            model.description = model_update.description

        try:
            session.commit()
            return ModelResponse.from_orm(model, model_variants=model_variants)
        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while updating the model: {str(e)}",
            )


@router.get("/projects/{project_id}/models/{model_id}/download")
def download_model(project_id: int, model_id: int):
    with Session(db_manager.get_project_engine(project_id)) as session:
        model = session.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        MODELS_FOLDER = pathlib.Path(config.PROJECTS_ROOT) / str(project_id) / "models"
        model_full_path = MODELS_FOLDER / model.path
        print(model_full_path)
        if not model.path or not os.path.exists(model_full_path):
            raise HTTPException(status_code=404, detail="Model file not found")

        try:
            with open(model_full_path, "rb") as model_file:
                model_data = model_file.read()

            filename = os.path.basename(model_full_path)
            return Response(
                content=model_data,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while downloading the model: {str(e)}",
            )


@router.get("/projects/{project_id}/models/{model_id}/download_exported")
def download_exported_model(project_id: int, model_id: int):
    with Session(db_manager.get_project_engine(project_id)) as session:
        model = session.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        MODELS_FOLDER = pathlib.Path(config.PROJECTS_ROOT) / str(project_id) / "models"
        model_full_path = MODELS_FOLDER / model.exported_path
        config_content = model.config_file
        if not model.exported_path or not os.path.exists(model_full_path):
            raise HTTPException(status_code=404, detail="Exported model file not found")
        if not config_content:
            raise HTTPException(status_code=404, detail="Config content not found")

        try:
            # Create a zip file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # Add model file
                zip_file.write(model_full_path, os.path.basename(model_full_path))
                # Add config content as file
                zip_file.writestr("config.yml", config_content)

            zip_buffer.seek(0)
            model_name = os.path.splitext(os.path.basename(model_full_path))[0]
            return Response(
                content=zip_buffer.getvalue(),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename={model_name}.zip"
                },
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while creating model archive: {str(e)}",
            )

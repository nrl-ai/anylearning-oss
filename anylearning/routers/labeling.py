import copy
import os
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from anylearning import config
from anylearning.auto_labeling.model_manager import ModelManager
from anylearning.database import DataItem, db_manager

router = APIRouter(prefix="/api", tags=["Auto Labeling"])
model_manager = ModelManager()


@router.get("/projects/{project_id}/auto_labeling/models")
async def auto_labeling_models(
    project_id: int,
):
    configs = copy.deepcopy(model_manager.get_model_configs())
    for model_config in configs:
        if "config_file" in model_config:
            del model_config["config_file"]
    return configs


@router.get("/projects/{project_id}/auto_labeling/status")
async def auto_labeling_status(
    project_id: int,
):
    return {"status": model_manager.status}


class AutoLabelingLoadModelRequest(BaseModel):
    model_name: str


@router.post("/projects/{project_id}/auto_labeling/load_model")
async def auto_labeling_load_model(
    project_id: int,
    request: AutoLabelingLoadModelRequest,
):
    try:
        model_manager.load_model_by_name(request.model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success"}


class AutoLabelingInferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    data_item_id: int
    marks: List[dict] = Field(default_factory=list)
    preload_data_item_ids: List[int] = Field(default_factory=list)
    #: A warm-up, not something the user asked for. The labelling screen runs
    #: one dummy prediction on open so the first real click is fast; without
    #: this flag it left "Finished inferencing AI model. Check the result." in
    #: the toolbar of a screen where nothing had been inferred, and the status
    #: stayed there until the user did something.
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
    with Session(db_manager.get_project_engine(project_id)) as session:
        data_item = (
            session.query(DataItem).filter(DataItem.id == request.data_item_id).first()
        )
        if not data_item:
            raise HTTPException(status_code=404, detail="Data item not found")

        project_data_dir = os.path.join(config.PROJECTS_ROOT, str(project_id), "data")
        full_path = os.path.join(project_data_dir, data_item.path)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Image file not found")

        # Search for preload data items
        preload_data_items = (
            session.query(DataItem)
            .filter(DataItem.id.in_(request.preload_data_item_ids))
            .all()
        )
        preload_data_items_paths = [
            os.path.join(project_data_dir, data_item.path)
            for data_item in preload_data_items
        ]
        if not all(os.path.exists(path) for path in preload_data_items_paths):
            raise HTTPException(status_code=404, detail="Preload data items not found")

        try:
            model_manager.load_model_by_name(request.model_name)
            model_manager.set_auto_labeling_marks(request.marks)
            before = model_manager.status
            auto_labeling_result = model_manager.predict_shapes(
                full_path, preload_paths=preload_data_items_paths
            )
            if request.warm_up:
                # Put the status back: the toolbar shows it, and a warm-up
                # reporting a finished inference is a message about something
                # the user did not do.
                model_manager.status = before
            return {"status": "success", "result": auto_labeling_result}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

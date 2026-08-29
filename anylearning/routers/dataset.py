import contextlib
import json
import logging
import math
import os
import random
import shutil
import traceback
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from numbers import Real
from typing import Dict, List, Literal, Optional

import cv2  # Added for image processing
import yaml
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse  # Added FileResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from anylearning import config
from anylearning.database import DataItem, Dataset, Project, db_manager
from anylearning.training import handpose_landmarks, keypoints
from anylearning.utils import convert_anylabeling_to_anylearning
from anylearning.utils.converters import (
    convert_anylearning_to_anylabeling,
    convert_anylearning_to_coco,
    convert_anylearning_to_labelme,
    convert_anylearning_to_yolo,
    convert_coco_to_anylearning,
    convert_yolo_to_anylearning,
)

router = APIRouter(prefix="/api", tags=["Dataset Management"])


class DataItemResponse(BaseModel):
    id: int
    subset: int
    labeled: bool
    path: str
    original_name: str
    class_id: int

    model_config = ConfigDict(from_attributes=True)


class DataItemListResponse(BaseModel):
    offset: int
    limit: int
    total_count: int
    data_items: List[DataItemResponse]


class UploadStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadStatusResponse(BaseModel):
    status: UploadStatus
    total_files: int
    processed_files: int
    error_message: Optional[str] = None
    #: Classes the upload created, so "it did not pick up my categories" is
    #: something the user can see rather than infer. An upload that finds no
    #: annotations reports an empty list, which is the useful distinction: the
    #: archive had none, rather than the app having ignored them.
    created_labels: List[str] = []


# TODO: Refactor this to be a singleton
# Or use a database table to store the status
upload_status = {}


@router.get(
    "/projects/{project_id}/data_items",
    response_model=DataItemListResponse,
)
def list_data_items(
    project_id: int,
    subset: Optional[int] = None,
    offset: int = 0,
    limit: int = 100,
) -> DataItemListResponse:
    with Session(db_manager.get_project_engine(project_id)) as session:
        query = session.query(DataItem)

        if subset is not None:
            if subset not in [0, 1, 2]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid subset parameter. Must be 0 (train), 1 (val), or 2 (test).",
                )
            query = query.filter(DataItem.subset == subset)

        # Count total number of items matching the query
        total_count = query.count()

        # Order the query by id for consistent pagination
        query = query.order_by(DataItem.id)

        try:
            data_items = query.offset(offset).limit(limit).all()
        except Exception as e:
            logging.error(f"Error fetching data items: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while fetching data items.",
            )

        data_item_responses = []
        for item in data_items:
            if item is None:
                continue
            data_item_responses.append(
                DataItemResponse(
                    id=item.id,
                    subset=item.subset,
                    labeled=item.labeled,
                    path=f"/api/projects/{project_id}/data_items/{item.id}/download",
                    original_name=os.path.basename(item.original_name)
                    if item.original_name
                    else "",
                    class_id=item.class_id,
                )
            )

        return DataItemListResponse(
            offset=offset,
            limit=limit,
            total_count=total_count,
            data_items=data_item_responses,
        )


@router.delete("/projects/{project_id}/data_items")
def delete_data_items(
    project_id: int,
    item_ids: List[int],
):
    with Session(db_manager.get_project_engine(project_id)) as session:
        try:
            # Get the data items to delete
            items_to_delete = (
                session.query(DataItem).filter(DataItem.id.in_(item_ids)).all()
            )

            # Delete the image files
            project_data_dir = os.path.join(
                config.PROJECTS_ROOT, str(project_id), "data"
            )
            for item in items_to_delete:
                image_path = os.path.join(project_data_dir, item.path)
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                except Exception as e:
                    logging.error(f"Failed to delete image file {image_path}: {str(e)}")

            # Delete the database records
            deleted_count = (
                session.query(DataItem)
                .filter(DataItem.id.in_(item_ids))
                .delete(synchronize_session=False)
            )
            session.commit()

            return {"message": f"Successfully deleted {deleted_count} data items"}
        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while deleting data items: {str(e)}",
            )


def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def create_or_get_dataset(session):
    dataset = session.query(Dataset).first()
    if not dataset:
        dataset = Dataset()
        session.add(dataset)
        session.commit()
        session.flush()
        logging.info("Created new dataset")
    return dataset


def get_project(main_session, project_id):
    project = main_session.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if isinstance(project.labels, str):
        project.labels = json.loads(project.labels)

    if not project.labels:
        project.labels = []
        logging.info("Initialized empty labels list")

    return project


class HandposeUnavailable(RuntimeError):
    """The landmark model died, so this upload cannot produce a dataset.

    Raised rather than returned: an upload that keeps going drops every image
    for a reason that has nothing to do with the images, and reports success.
    """


def next_label_id(project) -> int:
    """One past the highest id in use, not the number of labels.

    `len(project.labels)` collides the moment a label has been deleted: three
    labels with ids 0, 1, 2, delete the middle one, and the next upload creates
    another id 2. For classification that is not cosmetic -- `class_id` points
    at a label id, so every image of one class silently reads as the other.
    """
    ids = [
        label.get("id") for label in project.labels if isinstance(label.get("id"), int)
    ]
    return max(ids) + 1 if ids else 0


def annotation_from_archive(zip_ref, file_info, coco, class_names, image_path):
    """The annotation for one image, from whichever format the archive uses.

    Order matters: COCO first because it is dataset-wide and unambiguous, then
    the AnyLabeling sidecar, then YOLO -- whose .txt sidecar carries no format
    marker at all, so it is only read when nothing better was found.
    """
    stem = os.path.splitext(file_info.filename)[0]

    from_coco = coco.get(os.path.basename(file_info.filename))
    if from_coco:
        return from_coco

    label_filename = stem + ".json"
    if label_filename in zip_ref.namelist():
        with zip_ref.open(label_filename) as label_file:
            return convert_anylabeling_to_anylearning(json.load(label_file))

    yolo_filename = stem + ".txt"
    if yolo_filename not in zip_ref.namelist():
        # YOLO's usual layout is images/<name>.jpg with labels/<name>.txt, so
        # the label sits in a sibling folder rather than beside the image.
        # Matched on the path component, not on a substring: "images" can be
        # the first segment, where a "/images/" search never finds it.
        parts = stem.split("/")
        if "images" in parts:
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            yolo_filename = "/".join(parts) + ".txt"
    if yolo_filename in zip_ref.namelist():
        if not class_names:
            logging.warning(
                f"Found {yolo_filename} but no classes.txt or data.yaml, so its "
                "class numbers cannot be named"
            )
            return None
        with Image.open(image_path) as image:
            size = image.size
        with zip_ref.open(yolo_filename) as label_file:
            text = label_file.read().decode("utf-8", "replace")
        return convert_yolo_to_anylearning(text, class_names, size)

    return None


def process_regular_image(
    zip_ref, file_info, project, class_id, coco=None, class_names=None, image_path=None
):
    annotation = None
    updated_project_labels = False

    try:
        annotation = annotation_from_archive(
            zip_ref, file_info, coco or {}, class_names or [], image_path
        )
        if annotation:
            # Add new categories to project labels
            for obj in annotation:
                for category in obj["categories"]:
                    is_new_label = True
                    for label in project.labels:
                        if label["name"] == category:
                            is_new_label = False
                            break
                    if is_new_label:
                        new_label = {
                            "name": category,
                            "color": obj.get(
                                "color", f"#{random.randint(0, 0xFFFFFF):06x}"
                            ),
                            "id": next_label_id(project),
                        }
                        project.labels.append(new_label)
                        logging.info(f"Created new category {category}")
                        updated_project_labels = True
    except Exception as e:
        logging.error(f"Error processing label file: {str(e)}")
        logging.error(traceback.format_exc())

    return annotation, updated_project_labels


def handle_auto_categories(folder_name, project):
    class_id = -1
    updated = False

    if folder_name:
        is_new_label = True
        for label in project.labels:
            if label["name"] == folder_name:
                class_id = label["id"]
                is_new_label = False
                break

        if is_new_label:
            new_label = {
                "name": folder_name,
                "color": f"#{random.randint(0, 0xFFFFFF):06x}",
                "id": next_label_id(project),
            }
            class_id = new_label["id"]
            project.labels.append(new_label)
            updated = True
            logging.info(f"Created new category {folder_name} with id {class_id}")

    return class_id, updated


def scan_archive_formats(zip_ref):
    """Find dataset-wide annotations in an archive: COCO, or YOLO class names.

    AnyLabeling puts one .json beside each image, so it needs no preparation.
    COCO keeps a single file for the whole dataset, and YOLO keeps its class
    names in a separate file from its labels -- both have to be found before
    any image is looked at.

    Returns (coco annotations by image file name, YOLO class names).
    """
    coco: Dict[str, List[Dict]] = {}
    class_names: List[str] = []

    for info in zip_ref.infolist():
        if info.is_dir() or info.filename.startswith("__MACOSX/"):
            continue
        name = os.path.basename(info.filename).lower()

        if name.endswith(".json"):
            # Only a file that looks like COCO: a per-image AnyLabeling sidecar
            # is also .json, and reading every one of them here would be slow
            # and wrong.
            if info.file_size > 200 * 1024 * 1024:
                continue
            try:
                with zip_ref.open(info) as handle:
                    payload = json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            if (
                isinstance(payload, dict)
                and {"images", "annotations"} <= payload.keys()
            ):
                coco.update(convert_coco_to_anylearning(payload))
                logging.info(f"Read COCO annotations from {info.filename}")

        elif name in {"classes.txt", "obj.names", "labels.txt"}:
            with zip_ref.open(info) as handle:
                class_names = [
                    line.strip()
                    for line in handle.read().decode("utf-8", "replace").splitlines()
                    if line.strip()
                ]
            logging.info(
                f"Read {len(class_names)} YOLO class names from {info.filename}"
            )

        elif name in {"data.yaml", "data.yml"} and not class_names:
            try:
                with zip_ref.open(info) as handle:
                    payload = yaml.safe_load(handle.read().decode("utf-8", "replace"))
            except Exception:  # noqa: BLE001 -- a malformed yaml is not fatal
                continue
            names = (payload or {}).get("names")
            if isinstance(names, dict):
                class_names = [names[key] for key in sorted(names)]
            elif isinstance(names, list):
                class_names = list(names)
            if class_names:
                logging.info(
                    f"Read {len(class_names)} YOLO class names from {info.filename}"
                )

    return coco, class_names


def process_uploaded_data(
    project_id: int,
    temp_file_path: str,
    subset: int,
    auto_create_categories: bool,
):
    with Session(db_manager.get_project_engine(project_id)) as session:
        project_data_dir = os.path.join(config.PROJECTS_ROOT, str(project_id), "data")
        os.makedirs(project_data_dir, exist_ok=True)

        allowed_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")

        try:
            dataset = create_or_get_dataset(session)

            with Session(db_manager.main_engine) as main_session:
                project = get_project(main_session, project_id)
                updated_project_labels = False
                labels_before = {label["name"] for label in project.labels}

                with (
                    zipfile.ZipFile(temp_file_path, "r") as zip_ref,
                    contextlib.ExitStack() as stack,
                ):
                    # COCO keeps one annotations file for the whole dataset and
                    # YOLO keeps its class names apart from its labels, so both
                    # have to be found before any image is looked at.
                    coco_annotations, yolo_class_names = scan_archive_formats(zip_ref)

                    valid_files = [
                        f
                        for f in zip_ref.infolist()
                        if not f.is_dir()
                        and not f.filename.startswith("__MACOSX/")
                        and not f.filename.startswith(".")
                        and os.path.splitext(f.filename)[1].lower()
                        in allowed_extensions
                    ]

                    total_files = len(valid_files)
                    processed_files = 0
                    upload_status[project_id].status = UploadStatus.PROCESSING
                    upload_status[project_id].total_files = total_files
                    logging.info(f"Found {total_files} valid files to process")

                    # The landmark model runs in a child process; see
                    # handpose_landmarks for why. ExitStack so the worker is
                    # shut down on every path out of this loop, and so nothing
                    # is started for the four project types that never ask.
                    landmarks = None
                    if project.type == "Handpose Classification":
                        landmarks = stack.enter_context(
                            handpose_landmarks.LandmarkReader()
                        )
                        logging.info("Started the hand landmark worker")

                    for file_info in valid_files:
                        logging.info(f"Processing file: {file_info.filename}")
                        sample_id = str(uuid.uuid4())
                        extension = os.path.splitext(file_info.filename)[1]
                        new_filename = f"{sample_id}{extension}"
                        extracted_path = os.path.join(project_data_dir, new_filename)

                        folder_path = os.path.dirname(file_info.filename)
                        if isinstance(folder_path, bytes):
                            folder_path = folder_path.decode("utf-8")
                        folder_name = (
                            folder_path.split("/")[-1] if folder_path else None
                        )

                        class_id = -1
                        if folder_name:
                            for label in project.labels:
                                if label["name"] == folder_name:
                                    class_id = label["id"]
                                    logging.info(
                                        f"Found matching category {folder_name} with id {class_id}"
                                    )
                                    break

                        with (
                            zip_ref.open(file_info) as source,
                            open(extracted_path, "wb") as target,
                        ):
                            shutil.copyfileobj(source, target)

                        if is_valid_image(extracted_path):
                            logging.info(f"Valid image found: {file_info.filename}")
                            annotation = None

                            if project.type == "Handpose Classification":
                                if auto_create_categories:
                                    class_id, label_updated = handle_auto_categories(
                                        folder_name, project
                                    )
                                    updated_project_labels |= label_updated

                                annotation = landmarks.read(extracted_path, class_id)
                                if not annotation:
                                    # An image with no hand in it is not usable
                                    # for this project type, so it is dropped.
                                    # But if the *model* died, every remaining
                                    # image would be dropped for a reason that
                                    # has nothing to do with the images -- and
                                    # the upload would report "completed" over
                                    # an empty dataset. Stop and say so.
                                    os.remove(extracted_path)
                                    if landmarks.crashed:
                                        raise HandposeUnavailable(
                                            "The hand landmark model could not run on "
                                            "this machine, so no landmarks could be read "
                                            "from these images. Handpose projects need "
                                            "it; the other project types do not."
                                        )
                                    continue

                            else:
                                annotation, updated_project_labels_this_file = (
                                    process_regular_image(
                                        zip_ref,
                                        file_info,
                                        project,
                                        class_id,
                                        coco=coco_annotations,
                                        class_names=yolo_class_names,
                                        # YOLO coordinates are normalised, so
                                        # reading them needs the extracted
                                        # image's real size.
                                        image_path=extracted_path,
                                    )
                                )
                                updated_project_labels |= (
                                    updated_project_labels_this_file
                                )

                                if (
                                    project.type == "Image Classification"
                                    and auto_create_categories
                                ):
                                    class_id, label_updated = handle_auto_categories(
                                        folder_name, project
                                    )
                                    updated_project_labels |= label_updated

                            data_item = DataItem(
                                dataset_id=dataset.id,
                                subset=subset,
                                path=new_filename,
                                labeled=annotation is not None or class_id != -1,
                                original_name=os.path.basename(file_info.filename),
                                annotation=(
                                    {"data": annotation} if annotation else None
                                ),
                                class_id=class_id,
                            )
                            session.add(data_item)
                            session.commit()
                            session.flush()
                        else:
                            os.remove(extracted_path)
                            logging.info(f"Removed invalid image: {file_info.filename}")

                        processed_files += 1
                        upload_status[project_id].processed_files = processed_files
                        logging.info(f"Processed {processed_files}/{total_files} files")

                    created_labels = [
                        label["name"]
                        for label in project.labels
                        if label["name"] not in labels_before
                    ]
                    if project_id in upload_status:
                        upload_status[project_id].created_labels = created_labels

                    if updated_project_labels:
                        logging.info("Updating project labels")
                        project.labels = [label.copy() for label in project.labels]
                        flag_modified(project, "labels")
                        main_session.add(project)
                        main_session.commit()
                        main_session.flush()

                session.commit()
                session.flush()
                upload_status[project_id].status = UploadStatus.COMPLETED
                logging.info("Upload completed successfully")
        except HandposeUnavailable as e:
            # Not a database problem and not the user's archive: the machine
            # cannot run the model this project type is built on. Reported as a
            # failure rather than a completed upload over an empty dataset.
            session.rollback()
            upload_status[project_id].status = UploadStatus.FAILED
            upload_status[project_id].error_message = str(e)
            logging.error("Upload failed: %s", e)
        except IntegrityError as e:
            session.rollback()
            upload_status[project_id].status = UploadStatus.FAILED
            upload_status[
                project_id
            ].error_message = f"Database integrity error: {str(e)}"
            logging.error(f"Upload failed with database integrity error: {str(e)}")
        except Exception as e:
            session.rollback()
            upload_status[project_id].status = UploadStatus.FAILED
            upload_status[project_id].error_message = str(e)
            logging.error(f"Upload failed with error: {str(e)}")
        finally:
            os.remove(temp_file_path)
            logging.info("Removed temporary file")

        # Update the dataset version with the datetime
        dataset.version = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        session.add(dataset)
        session.commit()
        session.flush()
        logging.info(f"Updated dataset version to {dataset.version}")


def count_dataset_items(conn, dataset_id):
    session = conn["session"]
    counts = defaultdict(lambda: defaultdict(int))

    subsets = {0: "train", 1: "validation", 2: "test"}

    for subset, subset_name in subsets.items():
        total_count = (
            session.query(func.count(DataItem.id))
            .filter(DataItem.dataset_id == dataset_id, DataItem.subset == subset)
            .scalar()
        )

        labeled_count = (
            session.query(func.count(DataItem.id))
            .filter(
                DataItem.dataset_id == dataset_id,
                DataItem.subset == subset,
                DataItem.labeled,
            )
            .scalar()
        )

        counts[subset_name]["total"] = total_count
        counts[subset_name]["labeled"] = labeled_count

    return counts


@router.get("/projects/{project_id}/datasets")
async def get_dataset_info(project_id: int):
    with Session(db_manager.get_project_engine(project_id)) as session:
        dataset = session.query(Dataset).first()

        # Create a dataset if it doesn't exist
        if not dataset:
            dataset = Dataset()
            session.add(dataset)
            session.flush()

        counts = count_dataset_items({"session": session}, dataset.id)
        for subset, data in counts.items():
            if subset == "train":
                data["version"] = (
                    dataset.train_version if dataset.train_version else "v0.0.0"
                )
            elif subset == "validation":
                data["version"] = (
                    dataset.val_version if dataset.val_version else "v0.0.0"
                )
            elif subset == "test":
                data["version"] = (
                    dataset.test_version if dataset.test_version else "v0.0.0"
                )

        result = [
            {
                "type": subset,
                "version": data["version"],
                "num_total": data["total"],
                "num_labeled": data["labeled"],
                "num_unlabeled": data["total"] - data["labeled"],
                "subset": (
                    0 if subset == "train" else 1 if subset == "validation" else 2
                ),
            }
            for subset, data in counts.items()
        ]

        return result


SUBSET_NAMES = {0: "train", 1: "validation", 2: "test"}


@router.get("/projects/{project_id}/class_distribution")
async def get_class_distribution(project_id: int):
    """How many annotations each label has, split by train/validation/test.

    Class imbalance is invisible in the split chart -- a project can be neatly
    60/20/20 and still have one class with four examples and another with four
    hundred, which is the difference between a model that works and one that
    reports high accuracy by ignoring the rare class. The per-subset breakdown
    matters for the same reason: a class present only in training is never
    validated, and one absent from training cannot be learned at all.

    Counted here rather than maintained incrementally. A full scan of a
    project's items costs a few hundred milliseconds at the sizes this app is
    used at, and a stored counter would have to be kept correct across
    labelling, upload, deletion and subset moves -- every one of which is a
    place for it to drift away from the truth.
    """
    with Session(db_manager.main_engine) as main_session:
        project = get_project(main_session, project_id)
        labels = [dict(label) for label in (project.labels or [])]

    # id -> name, for the class_id column classification and handpose use.
    names_by_id = {label.get("id"): label.get("name") for label in labels}
    counts: Dict[str, Dict[str, int]] = {
        label["name"]: {name: 0 for name in SUBSET_NAMES.values()} for label in labels
    }
    # Annotations can name a category the project no longer lists, if a label
    # was renamed or deleted after labelling. Those are real annotations and
    # training will see them, so they are reported rather than dropped.
    unknown: Dict[str, Dict[str, int]] = {}
    unlabeled = {name: 0 for name in SUBSET_NAMES.values()}

    def bucket(name: str) -> Optional[Dict[str, int]]:
        if name in counts:
            return counts[name]
        if name is None:
            return None
        return unknown.setdefault(name, {n: 0 for n in SUBSET_NAMES.values()})

    with Session(db_manager.get_project_engine(project_id)) as session:
        for item in session.query(DataItem).all():
            subset = SUBSET_NAMES.get(item.subset)
            if subset is None:
                continue

            counted = False
            if item.class_id is not None and item.class_id >= 0:
                target = bucket(names_by_id.get(item.class_id))
                if target is not None:
                    target[subset] += 1
                    counted = True

            # Not every annotation is a list of shapes. Handpose stores a dict
            # of hand landmarks under the same key, and its class lives in
            # class_id, which is already counted above.
            shapes = (item.annotation or {}).get("data")
            for shape in shapes if isinstance(shapes, list) else []:
                for category in shape.get("categories") or []:
                    target = bucket(category)
                    if target is not None:
                        target[subset] += 1
                        counted = True

            if not counted:
                unlabeled[subset] += 1

    def rows(source, known):
        return [
            {
                "name": name,
                "color": next(
                    (
                        label.get("color")
                        for label in labels
                        if label.get("name") == name
                    ),
                    None,
                ),
                "known": known,
                "total": sum(per_subset.values()),
                **per_subset,
            }
            for name, per_subset in source.items()
        ]

    return {
        "classes": rows(counts, True) + rows(unknown, False),
        "unlabeled": {**unlabeled, "total": sum(unlabeled.values())},
    }


#: What a plain (non-zip) upload may contain. Images are the point; JSON is
#: allowed because an AnyLabeling export is an image and its sidecar, and
#: separating them at the door would drop every annotation the user already has.
UPLOADABLE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".json",
)


def _zip_from_uploads(uploads, destination) -> None:
    """Repack loose files into the archive the processing path already reads.

    Dropping images on the window and choosing a .zip are the same operation as
    far as everything downstream is concerned -- extraction, validation, label
    creation, the progress counter. Converting at the boundary keeps one code
    path rather than a second, subtly different one.
    """
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for upload in uploads:
            name = os.path.basename(upload.filename or "")
            if not name:
                continue
            upload.file.seek(0)
            with archive.open(name, "w") as target:
                shutil.copyfileobj(upload.file, target)


class CopySubsetRequest(BaseModel):
    from_subset: int
    to_subset: int


@router.post("/projects/{project_id}/data_items/copy_subset")
def copy_subset(project_id: int, request: CopySubsetRequest):
    """Duplicate one subset's images into another.

    For the common small-dataset case: someone has 40 training and 10
    validation images and no test set, and a test set is the only thing
    standing between them and a trained model. Copying validation into test
    gives a number that is *measured* rather than absent -- while being clear
    about what it is worth, which the UI says: a test set that shares its
    images with validation cannot tell you anything validation did not.

    The files are copied, not referenced twice. Two rows pointing at one file
    means deleting either item deletes the image out from under the other.
    """
    if request.from_subset not in (0, 1, 2) or request.to_subset not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="Subsets are 0, 1 or 2.")
    if request.from_subset == request.to_subset:
        raise HTTPException(status_code=400, detail="Pick two different subsets.")

    project_data_dir = os.path.join(config.PROJECTS_ROOT, str(project_id), "data")
    os.makedirs(project_data_dir, exist_ok=True)

    copied = 0
    with Session(db_manager.get_project_engine(project_id)) as session:
        dataset = create_or_get_dataset(session)
        source = (
            session.query(DataItem).filter(DataItem.subset == request.from_subset).all()
        )
        if not source:
            raise HTTPException(status_code=400, detail="That subset has no images.")

        for item in source:
            extension = os.path.splitext(item.path)[1]
            new_path = f"{uuid.uuid4()}{extension}"
            try:
                shutil.copyfile(
                    os.path.join(project_data_dir, item.path),
                    os.path.join(project_data_dir, new_path),
                )
            except OSError as error:
                # One unreadable file should not abandon the whole copy: the
                # rest of the subset is still worth having.
                logging.warning(f"Could not copy {item.path}: {error}")
                continue

            session.add(
                DataItem(
                    dataset_id=dataset.id,
                    subset=request.to_subset,
                    path=new_path,
                    labeled=item.labeled,
                    original_name=item.original_name,
                    annotation=item.annotation,
                    class_id=item.class_id,
                )
            )
            copied += 1

        session.commit()

    return {"copied": copied}


@router.post("/projects/{project_id}/upload_data")
async def upload_data(
    project_id: int,
    # A list under the name the single-file form already used, so an existing
    # client keeps working and a drag-and-drop of forty images is one request.
    file: List[UploadFile] = File(...),
    subset: int = 0,
    auto_create_categories: bool = False,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    uploads = [item for item in file if item.filename]
    if not uploads:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    names = [item.filename.lower() for item in uploads]
    archives = [name for name in names if name.endswith(".zip")]
    if archives and len(uploads) > 1:
        raise HTTPException(
            status_code=400,
            detail="Upload a ZIP on its own, or a set of images -- not both.",
        )

    if not archives:
        unsupported = sorted(
            {
                os.path.splitext(name)[1] or "(no extension)"
                for name in names
                if not name.endswith(UPLOADABLE_EXTENSIONS)
            }
        )
        if unsupported:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot upload {', '.join(unsupported)}. Images or a ZIP.",
            )

    if subset not in [0, 1, 2]:
        raise HTTPException(
            status_code=400,
            detail="Invalid subset parameter. Must be 0 (train), 1 (val), or 2 (test).",
        )

    os.makedirs(os.path.join(config.DATA_ROOT, "tmp_files"), exist_ok=True)
    temp_file = os.path.join(config.DATA_ROOT, "tmp_files", f"temp_{uuid.uuid4()}.zip")
    try:
        if archives:
            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(uploads[0].file, buffer)
        else:
            _zip_from_uploads(uploads, temp_file)

        upload_status[project_id] = UploadStatusResponse(
            status=UploadStatus.PENDING, total_files=0, processed_files=0
        )

        logging.info(
            f"Starting data upload for project {project_id} with subset {subset}"
        )

        background_tasks.add_task(
            process_uploaded_data,
            project_id,
            temp_file,
            subset,
            auto_create_categories,
        )
        return {
            "message": "Data upload started. Processing in background. Only valid image files will be processed."
        }
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/upload_status", response_model=UploadStatusResponse)
async def get_upload_status(project_id: int):
    if project_id not in upload_status:
        raise HTTPException(
            status_code=404, detail="No upload status found for this project"
        )
    return upload_status[project_id]


@router.get("/projects/{project_id}/data_items/{item_id}/download")
async def download_data_item(project_id: int, item_id: int, download: bool = False):
    with Session(db_manager.get_project_engine(project_id)) as session:
        data_item = session.query(DataItem).filter(DataItem.id == item_id).first()

        if not data_item:
            raise HTTPException(status_code=404, detail="Data item not found")

        project_data_dir = os.path.join(config.PROJECTS_ROOT, str(project_id), "data")
        full_path = os.path.join(project_data_dir, data_item.path)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Image file not found")

        # Stream the file in chunks
        async def iterfile():
            with open(full_path, mode="rb") as file:
                while chunk := file.read(8192):
                    yield chunk

        # Determine media type from file extension
        file_extension = os.path.splitext(data_item.original_name)[1].lower()
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
        }.get(file_extension, "application/octet-stream")

        # URL encode the filename to handle unicode characters
        encoded_filename = (
            os.path.basename(data_item.original_name).encode("utf-8").decode("latin-1")
        )

        headers = {}
        if download:
            headers["Content-Disposition"] = (
                f"attachment; filename*=UTF-8''{encoded_filename}"
            )
        else:
            headers["Content-Disposition"] = (
                f"inline; filename*=UTF-8''{encoded_filename}"
            )

        return StreamingResponse(iterfile(), media_type=media_type, headers=headers)


@router.get("/projects/{project_id}/data_items/random_test_sample")
async def get_random_test_sample(project_id: int):
    with Session(db_manager.get_project_engine(project_id)) as session:
        # Test set first, validation second. A project with an empty test split
        # is ordinary -- the upload puts everything the user gives it wherever
        # they put it, and plenty of people never fill the third one -- and
        # answering "Failed to fetch test sample" to someone who just wants to
        # see whether their model works is a dead end where an image they have
        # never trained on is sitting one split over.
        #
        # The response says which split it came from, so the dialog can say so
        # rather than quietly implying it is a test image.
        subset_used = 2
        test_item = (
            session.query(DataItem)
            .filter(DataItem.subset == 2)
            .order_by(func.random())
            .first()
        )
        if not test_item:
            subset_used = 1
            test_item = (
                session.query(DataItem)
                .filter(DataItem.subset == 1)
                .order_by(func.random())
                .first()
            )

        if not test_item:
            raise HTTPException(
                status_code=404,
                detail=(
                    "This project has no images in its test or validation set. "
                    "Upload some, or use the other tab to try an image of your own."
                ),
            )

        project_data_dir = os.path.join(config.PROJECTS_ROOT, str(project_id), "data")
        full_path = os.path.join(project_data_dir, test_item.path)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Image file not found")

        # Stream the file in chunks
        async def iterfile():
            with open(full_path, mode="rb") as file:
                while chunk := file.read(8192):
                    yield chunk

        # Determine media type from file extension
        file_extension = os.path.splitext(test_item.original_name)[1].lower()
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
        }.get(file_extension, "application/octet-stream")

        return StreamingResponse(
            iterfile(),
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="{os.path.basename(test_item.original_name)}"',
                # Which split this came from, so the dialog can say "from your
                # validation set" instead of calling it a test image.
                "X-AnyLearning-Subset": "test" if subset_used == 2 else "validation",
            },
        )


@router.get("/projects/{project_id}/data_items/{item_id}/get_annotation")
async def get_annotation(project_id: int, item_id: int):
    with Session(db_manager.get_project_engine(project_id)) as session:
        data_item = session.query(DataItem).filter(DataItem.id == item_id).first()

        if not data_item:
            raise HTTPException(status_code=404, detail="Data item not found")

        return data_item.annotation["data"] if data_item.annotation else []


def _is_valid_rectangle_shape(shape) -> bool:
    """Whether a JSON annotation is a finite, non-degenerate four-corner box."""
    if not isinstance(shape, dict) or shape.get("type") != "rectangle":
        return False
    points = shape.get("points")
    if not isinstance(points, list) or len(points) != 4:
        return False

    corners = []
    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            return False
        x, y = point[:2]
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, Real)
            or not isinstance(y, Real)
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
        ):
            return False
        corners.append((float(x), float(y)))

    # A rectangle may be rotated, so test polygon area rather than requiring
    # only two distinct x/y values. Four collinear or duplicate corners are not
    # a trainable object box.
    twice_area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(corners, corners[1:] + corners[:1])
        )
    )
    return twice_area > 0


@router.post("/projects/{project_id}/data_items/{item_id}/set_annotation")
async def save_annotation(project_id: int, item_id: int, request: Request):
    # Read the body directly, as update_class_id below already does. A bare
    # `annotation: list` parameter used to accept a raw JSON array; current
    # FastAPI instead expects it wrapped as {"annotation": [...]}, and even
    # Body(embed=False) does not restore the old behaviour for a plain list.
    # The result was a 422 on every labelling auto-save, losing annotations.
    annotation = await request.json()
    if not isinstance(annotation, list):
        raise HTTPException(
            status_code=422, detail="Annotation body must be a JSON array of shapes"
        )

    with Session(db_manager.main_engine) as main_session:
        project = main_session.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        if project.type == "Object Detection" and any(
            not _is_valid_rectangle_shape(shape) for shape in annotation
        ):
            raise HTTPException(
                status_code=422,
                detail="Object Detection annotations must contain rectangles only",
            )

    with Session(db_manager.get_project_engine(project_id)) as session:
        data_item = session.query(DataItem).filter(DataItem.id == item_id).first()

        if not data_item:
            raise HTTPException(status_code=404, detail="Data item not found")

        try:
            data_item.annotation = {}
            data_item.annotation["data"] = annotation
            data_item.labeled = True
            session.commit()
            return {"message": "Annotation saved successfully"}
        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=500, detail=f"Error saving annotation: {str(e)}"
            )


@router.post("/projects/{project_id}/data_items/{item_id}/class_id")
async def update_class_id(project_id: int, item_id: int, request: Request):
    with Session(db_manager.get_project_engine(project_id)) as session:
        data_item = session.query(DataItem).filter(DataItem.id == item_id).first()
        class_id = (await request.json())["class_id"]

        if not data_item:
            raise HTTPException(status_code=404, detail="Data item not found")

        try:
            data_item.class_id = class_id
            data_item.labeled = True
            session.commit()
            return {"message": "Class ID updated successfully"}
        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=500, detail=f"Error updating class ID: {str(e)}"
            )


# Add export status tracking
class ExportStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportStatusResponse(BaseModel):
    status: ExportStatus
    total_files: int
    processed_files: int
    error_message: Optional[str] = None
    export_path: Optional[str] = None
    format: Optional[str] = None


# Dictionary to keep track of export status
export_status: Dict[int, ExportStatusResponse] = {}


class ExportRequest(BaseModel):
    format: Literal["yolo", "coco", "labelme", "anylabeling"]
    subset: Optional[int] = None  # If None, export all subsets


@router.post("/projects/{project_id}/export_data")
async def export_dataset(
    project_id: int,
    export_request: ExportRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    # Validate if project exists
    with Session(db_manager.main_engine) as session:
        project = get_project(session, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        if project.type == "Keypoint Detection" and export_request.format == "yolo":
            raise HTTPException(
                status_code=400,
                detail=(
                    "YOLO pose export is not supported yet. Export this keypoint "
                    "project as COCO, LabelMe, or AnyLabeling instead."
                ),
            )

    # Create exports directory if it doesn't exist
    exports_dir = os.path.join(config.PROJECTS_ROOT, str(project_id), "exports")
    os.makedirs(exports_dir, exist_ok=True)

    # Clean up previous exports
    if project_id in export_status and export_status[project_id].export_path:
        try:
            previous_export_path = export_status[project_id].export_path
            if os.path.exists(previous_export_path):
                os.remove(previous_export_path)
                logging.info(f"Deleted previous export file: {previous_export_path}")
        except Exception as e:
            logging.error(f"Error deleting previous export: {e}")

    # Generate a unique name for the export
    export_filename = f"{export_request.format}_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip"
    export_path = os.path.join(exports_dir, export_filename)

    # Initialize export status
    export_status[project_id] = ExportStatusResponse(
        status=ExportStatus.PENDING,
        total_files=0,
        processed_files=0,
        format=export_request.format,
        export_path=export_path,
    )

    # Start background job for export
    background_tasks.add_task(
        process_dataset_export,
        project_id,
        export_request.format,
        export_path,
        export_request.subset,
    )

    return {
        "message": f"Dataset export to {export_request.format} format started. Processing in background.",
        "task_id": str(project_id),
    }


@router.get("/projects/{project_id}/export_status", response_model=ExportStatusResponse)
async def get_export_status(project_id: int):
    """Get the status of the current export job"""
    if project_id not in export_status:
        raise HTTPException(
            status_code=404, detail="No export job found for this project"
        )

    return export_status[project_id]


@router.get("/projects/{project_id}/download_export")
async def download_export(project_id: int):
    """Download the exported dataset"""
    if (
        project_id not in export_status
        or export_status[project_id].status != ExportStatus.COMPLETED
    ):
        raise HTTPException(
            status_code=404, detail="No completed export found for this project"
        )

    export_path = export_status[project_id].export_path
    if not os.path.exists(export_path):
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=export_path,
        filename=os.path.basename(export_path),
        media_type="application/zip",
    )


def process_dataset_export(
    project_id: int, export_format: str, export_path: str, subset: Optional[int] = None
):
    """
    Background job to export the dataset to the specified format
    """
    try:
        # Update status to processing
        export_status[project_id].status = ExportStatus.PROCESSING

        with Session(db_manager.get_project_engine(project_id)) as session:
            # Get project data to access labels
            with Session(db_manager.main_engine) as main_session:
                try:
                    project = get_project(main_session, project_id)

                    # Validate project has labels
                    if not project.labels or not isinstance(project.labels, list):
                        error_msg = (
                            f"Project has invalid or missing labels: {project.labels}"
                        )
                        logging.error(error_msg)
                        export_status[project_id].status = ExportStatus.FAILED
                        export_status[project_id].error_message = error_msg
                        return

                    # Create temp directory for export
                    temp_dir = os.path.join(
                        config.DATA_ROOT, "tmp_export", str(uuid.uuid4())
                    )
                    os.makedirs(temp_dir, exist_ok=True)

                    try:
                        # Query data items based on subset parameter
                        query = session.query(DataItem)
                        if subset is not None:
                            query = query.filter(DataItem.subset == subset)

                        # Get total count for progress tracking
                        total_items = query.count()
                        if total_items == 0:
                            error_msg = "No data items found to export"
                            logging.error(error_msg)
                            export_status[project_id].status = ExportStatus.FAILED
                            export_status[project_id].error_message = error_msg
                            return

                        export_status[project_id].total_files = total_items

                        data_items = query.all()
                        processed_items = 0
                        error_count = 0
                        max_errors = 10  # Maximum errors before aborting export

                        # Handle export based on format
                        if export_format == "yolo":
                            # Create YOLO directory structure
                            yolo_dir = os.path.join(temp_dir, "yolo")
                            images_dir = os.path.join(yolo_dir, "images")
                            labels_dir = os.path.join(yolo_dir, "labels")
                            os.makedirs(images_dir, exist_ok=True)
                            os.makedirs(labels_dir, exist_ok=True)

                            # Export dataset.yaml file with class names
                            with open(os.path.join(yolo_dir, "dataset.yaml"), "w") as f:
                                f.write("path: ./\n")
                                f.write("train: images/train\n")
                                f.write("val: images/val\n")
                                f.write("test: images/test\n\n")

                                # Write class names
                                f.write(f"nc: {len(project.labels)}\n")
                                f.write(
                                    "names: ["
                                    + ", ".join(
                                        [
                                            f"'{label['name']}'"
                                            for label in project.labels
                                        ]
                                    )
                                    + "]\n"
                                )

                            # Create train/val/test directories
                            for subset_name in ["train", "val", "test"]:
                                os.makedirs(
                                    os.path.join(images_dir, subset_name), exist_ok=True
                                )
                                os.makedirs(
                                    os.path.join(labels_dir, subset_name), exist_ok=True
                                )

                            # Process each data item
                            for _i, item in enumerate(data_items):
                                try:
                                    subset_folder = (
                                        "train"
                                        if item.subset == 0
                                        else "val"
                                        if item.subset == 1
                                        else "test"
                                    )

                                    # Copy image to YOLO structure
                                    image_path = os.path.join(
                                        config.PROJECTS_ROOT,
                                        str(project_id),
                                        "data",
                                        item.path,
                                    )

                                    # Validate image path exists
                                    if not os.path.exists(image_path):
                                        logging.warning(
                                            f"Image path does not exist: {image_path}"
                                        )
                                        continue

                                    image_filename = os.path.basename(item.path)
                                    yolo_image_path = os.path.join(
                                        images_dir, subset_folder, image_filename
                                    )

                                    # Copy image file
                                    shutil.copy2(image_path, yolo_image_path)

                                    # Create annotation if item is labeled
                                    if item.labeled and item.annotation:
                                        try:
                                            # Get image dimensions for YOLO conversion
                                            img = cv2.imread(image_path)
                                            if img is None:
                                                logging.warning(
                                                    f"Could not read image: {image_path}"
                                                )
                                                continue

                                            img_height, img_width = img.shape[:2]
                                            image_size = (img_width, img_height)

                                            # Extract annotation data safely
                                            annotation_data = None
                                            if (
                                                isinstance(item.annotation, dict)
                                                and "data" in item.annotation
                                            ):
                                                annotation_data = item.annotation[
                                                    "data"
                                                ]
                                            else:
                                                logging.warning(
                                                    f"Unexpected annotation format for item {item.id}: {item.annotation}"
                                                )
                                                continue

                                            # Create YOLO annotation
                                            yolo_annotation = (
                                                convert_anylearning_to_yolo(
                                                    annotation_data,
                                                    project.labels,
                                                    image_size,
                                                )
                                            )

                                            # Only write if annotation is not empty
                                            if yolo_annotation:
                                                # Write to file
                                                label_filename = (
                                                    os.path.splitext(image_filename)[0]
                                                    + ".txt"
                                                )
                                                label_path = os.path.join(
                                                    labels_dir,
                                                    subset_folder,
                                                    label_filename,
                                                )
                                                with open(label_path, "w") as f:
                                                    f.write(yolo_annotation)
                                        except Exception as e:
                                            error_count += 1
                                            logging.error(
                                                f"Error processing annotation for item {item.id}: {str(e)}"
                                            )
                                            logging.error(traceback.format_exc())
                                            # Abort if too many errors
                                            if error_count >= max_errors:
                                                raise Exception(
                                                    f"Too many errors ({error_count}) during export. Last error: {str(e)}"
                                                )

                                    # Update progress
                                    processed_items += 1
                                    export_status[
                                        project_id
                                    ].processed_files = processed_items
                                except Exception as e:
                                    error_count += 1
                                    logging.error(
                                        f"Error processing item {item.id}: {str(e)}"
                                    )
                                    # Continue with next item but count errors
                                    if error_count >= max_errors:
                                        raise Exception(
                                            f"Too many errors ({error_count}) during export. Last error: {str(e)}"
                                        )

                        elif export_format == "coco":
                            # Create COCO directory structure
                            coco_dir = os.path.join(temp_dir, "coco")
                            images_dir = os.path.join(coco_dir, "images")
                            annotations_dir = os.path.join(coco_dir, "annotations")
                            os.makedirs(images_dir, exist_ok=True)
                            os.makedirs(annotations_dir, exist_ok=True)

                            # Create COCO dataset structure
                            coco_json = {
                                "info": {
                                    "description": "Dataset exported from AnyLearning",
                                    "url": "",
                                    "version": "1.0",
                                    "year": datetime.now().year,
                                    "contributor": "AnyLearning",
                                    "date_created": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                },
                                "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
                                "images": [],
                                "annotations": [],
                                "categories": [],
                            }

                            # A keypoint project's labels are landmark names,
                            # not object categories. COCO carries them on the
                            # one implicit category instead.
                            category_records = (
                                [
                                    keypoints.coco_category(
                                        keypoints.keypoint_names(project.labels)
                                    )
                                ]
                                if project.type == "Keypoint Detection"
                                else [
                                    {
                                        "id": label["id"],
                                        "name": label["name"],
                                        "supercategory": "none",
                                    }
                                    for label in project.labels
                                ]
                            )
                            for category in category_records:
                                try:
                                    coco_json["categories"].append(category)
                                except Exception as e:
                                    logging.warning(
                                        f"Error adding category {category}: {e}"
                                    )
                                    continue

                            # Process each data item
                            image_id = 1
                            annotation_id = 1

                            for _i, item in enumerate(data_items):
                                try:
                                    # Copy image to COCO structure
                                    image_path = os.path.join(
                                        config.PROJECTS_ROOT,
                                        str(project_id),
                                        "data",
                                        item.path,
                                    )

                                    # Validate image path exists
                                    if not os.path.exists(image_path):
                                        logging.warning(
                                            f"Could not find image: {image_path}"
                                        )
                                        continue

                                    image_filename = os.path.basename(item.path)
                                    coco_image_path = os.path.join(
                                        images_dir, image_filename
                                    )

                                    # Copy image file
                                    shutil.copy2(image_path, coco_image_path)

                                    # Get image dimensions
                                    img = cv2.imread(image_path)
                                    if img is None:
                                        logging.warning(
                                            f"Could not read image: {image_path}"
                                        )
                                        continue

                                    img_height, img_width = img.shape[:2]

                                    # Add image info to COCO json
                                    coco_image = {
                                        "id": image_id,
                                        "width": img_width,
                                        "height": img_height,
                                        "file_name": image_filename,
                                        "license": 1,
                                        "flickr_url": "",
                                        "coco_url": "",
                                        "date_captured": "",
                                    }

                                    coco_json["images"].append(coco_image)

                                    # Create annotation if item is labeled
                                    if item.labeled and item.annotation:
                                        try:
                                            # Extract annotation data safely
                                            annotation_data = None
                                            if (
                                                isinstance(item.annotation, dict)
                                                and "data" in item.annotation
                                            ):
                                                annotation_data = item.annotation[
                                                    "data"
                                                ]
                                            else:
                                                logging.warning(
                                                    f"Unexpected annotation format for item {item.id}: {item.annotation}"
                                                )
                                                continue

                                            # Convert to COCO format
                                            coco_annotations = (
                                                convert_anylearning_to_coco(
                                                    annotation_data,
                                                    project.labels,
                                                    image_id,
                                                    image_filename,
                                                    (img_width, img_height),
                                                    keypoint_names=(
                                                        keypoints.keypoint_names(
                                                            project.labels
                                                        )
                                                        if project.type
                                                        == "Keypoint Detection"
                                                        else None
                                                    ),
                                                )
                                            )

                                            # Add annotations to COCO json
                                            for ann in coco_annotations:
                                                # The per-image converter uses
                                                # local IDs. COCO annotation
                                                # IDs are dataset-wide.
                                                ann["id"] = annotation_id
                                                coco_json["annotations"].append(ann)
                                                annotation_id += 1

                                        except Exception as e:
                                            error_count += 1
                                            logging.error(
                                                f"Error processing COCO annotation for item {item.id}: {str(e)}"
                                            )
                                            logging.error(traceback.format_exc())
                                            if error_count >= max_errors:
                                                raise Exception(
                                                    f"Too many errors ({error_count}) during export. Last error: {str(e)}"
                                                )

                                    # Update image_id for next image
                                    image_id += 1

                                    # Update progress
                                    processed_items += 1
                                    export_status[
                                        project_id
                                    ].processed_files = processed_items

                                except Exception as e:
                                    error_count += 1
                                    logging.error(
                                        f"Error processing item for COCO: {str(e)}"
                                    )
                                    if error_count >= max_errors:
                                        raise Exception(
                                            f"Too many errors ({error_count}) during export. Last error: {str(e)}"
                                        )

                            # Write COCO json to file
                            with open(
                                os.path.join(annotations_dir, "instances.json"), "w"
                            ) as f:
                                json.dump(coco_json, f, indent=2)

                        elif export_format in ["labelme", "anylabeling"]:
                            # Both LabelMe and AnyLabeling use similar formats
                            export_dir = os.path.join(temp_dir, export_format)
                            os.makedirs(export_dir, exist_ok=True)

                            # Process each data item
                            for _i, item in enumerate(data_items):
                                try:
                                    # Get image path
                                    image_path = os.path.join(
                                        config.PROJECTS_ROOT,
                                        str(project_id),
                                        "data",
                                        item.path,
                                    )

                                    # Validate image path exists
                                    if not os.path.exists(image_path):
                                        logging.warning(
                                            f"Could not find image: {image_path}"
                                        )
                                        continue

                                    image_filename = os.path.basename(item.path)
                                    base_filename = os.path.splitext(image_filename)[0]
                                    target_image_path = os.path.join(
                                        export_dir, image_filename
                                    )

                                    # Copy image file
                                    shutil.copy2(image_path, target_image_path)

                                    # Get image dimensions
                                    img = cv2.imread(image_path)
                                    if img is None:
                                        logging.warning(
                                            f"Could not read image: {image_path}"
                                        )
                                        continue

                                    img_height, img_width = img.shape[:2]

                                    # Create annotation if item is labeled
                                    if item.labeled and item.annotation:
                                        try:
                                            # Extract annotation data safely
                                            annotation_data = None
                                            if (
                                                isinstance(item.annotation, dict)
                                                and "data" in item.annotation
                                            ):
                                                annotation_data = item.annotation[
                                                    "data"
                                                ]
                                            else:
                                                logging.warning(
                                                    f"Unexpected annotation format for item {item.id}: {item.annotation}"
                                                )
                                                continue

                                            # Convert to appropriate format
                                            if export_format == "labelme":
                                                annotation_json = (
                                                    convert_anylearning_to_labelme(
                                                        annotation_data,
                                                        image_filename,
                                                        (img_width, img_height),
                                                    )
                                                )
                                            else:  # anylabeling
                                                annotation_json = (
                                                    convert_anylearning_to_anylabeling(
                                                        annotation_data,
                                                        image_filename,
                                                        (img_width, img_height),
                                                    )
                                                )

                                            # Write annotation to JSON file
                                            json_path = os.path.join(
                                                export_dir, f"{base_filename}.json"
                                            )
                                            with open(json_path, "w") as f:
                                                json.dump(annotation_json, f, indent=2)
                                        except Exception as e:
                                            error_count += 1
                                            logging.error(
                                                f"Error processing {export_format} annotation for item {item.id}: {str(e)}"
                                            )
                                            logging.error(traceback.format_exc())
                                            if error_count >= max_errors:
                                                raise Exception(
                                                    f"Too many errors ({error_count}) during export. Last error: {str(e)}"
                                                )
                                    else:
                                        # For unannotated images, create an empty annotation file
                                        if export_format == "labelme":
                                            empty_json = convert_anylearning_to_labelme(
                                                [],
                                                image_filename,
                                                (img_width, img_height),
                                            )
                                        else:  # anylabeling
                                            empty_json = (
                                                convert_anylearning_to_anylabeling(
                                                    [],
                                                    image_filename,
                                                    (img_width, img_height),
                                                )
                                            )

                                        # Write empty annotation
                                        json_path = os.path.join(
                                            export_dir, f"{base_filename}.json"
                                        )
                                        with open(json_path, "w") as f:
                                            json.dump(empty_json, f, indent=2)

                                    # Update progress
                                    processed_items += 1
                                    export_status[
                                        project_id
                                    ].processed_files = processed_items

                                except Exception as e:
                                    error_count += 1
                                    logging.error(
                                        f"Error processing item for {export_format}: {str(e)}"
                                    )
                                    if error_count >= max_errors:
                                        raise Exception(
                                            f"Too many errors ({error_count}) during export. Last error: {str(e)}"
                                        )

                        else:
                            raise ValueError(
                                f"Unsupported export format: {export_format}"
                            )

                        # Zip the export directory
                        with zipfile.ZipFile(
                            export_path, "w", zipfile.ZIP_DEFLATED
                        ) as zipf:
                            for root, _, files in os.walk(temp_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, temp_dir)
                                    zipf.write(file_path, arcname)

                        # Update status to completed
                        export_status[project_id].status = ExportStatus.COMPLETED

                    except Exception as e:
                        error_msg = f"Error during export process: {str(e)}"
                        logging.error(error_msg)
                        logging.error(traceback.format_exc())
                        export_status[project_id].status = ExportStatus.FAILED
                        export_status[project_id].error_message = error_msg
                        raise
                    finally:
                        # Clean up temporary directory
                        if os.path.exists(temp_dir):
                            try:
                                shutil.rmtree(temp_dir)
                            except Exception as e:
                                logging.error(
                                    f"Error cleaning up temp directory: {str(e)}"
                                )
                except Exception as e:
                    error_msg = f"Error processing project data: {str(e)}"
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
                    export_status[project_id].status = ExportStatus.FAILED
                    export_status[project_id].error_message = error_msg
                    raise

    except Exception as e:
        error_msg = f"Error exporting dataset: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        # Ensure we set the status to failed
        if project_id in export_status:
            export_status[project_id].status = ExportStatus.FAILED
            export_status[project_id].error_message = error_msg

        # Ensure the export file is removed if it exists
        if os.path.exists(export_path):
            try:
                os.remove(export_path)
            except Exception as e2:
                logging.error(f"Error removing failed export file: {str(e2)}")


@router.delete("/projects/{project_id}/cleanup_export")
async def cleanup_export(project_id: int):
    """
    Clean up the previous export file for a project
    """
    if project_id not in export_status:
        return {"message": "No export found for this project"}

    # Get the export path
    if export_status[project_id].export_path and os.path.exists(
        export_status[project_id].export_path
    ):
        try:
            # Delete the export file
            os.remove(export_status[project_id].export_path)
            logging.info(
                f"Deleted export file: {export_status[project_id].export_path}"
            )

            # Reset the export status
            export_status[project_id] = ExportStatusResponse(
                status=ExportStatus.PENDING,
                total_files=0,
                processed_files=0,
                error_message=None,
                export_path=None,
                format=None,
            )

            return {"message": "Export file cleaned up successfully"}
        except Exception as e:
            logging.error(f"Error cleaning up export file: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error cleaning up export file: {str(e)}"
            )

    return {"message": "No export file to clean up"}

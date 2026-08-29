import json
import logging
import os
import pathlib
import re
import shutil
import stat
import tarfile
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Json
from sqlalchemy.orm import Session

from anylearning import config
from anylearning.database import Base, Project, ProjectBase, ProjectCreate, db_manager
from anylearning.migration_manager import MigrationManager

router = APIRouter(prefix="/api", tags=["Project Management"])

export_processes = {}
import_processes = {}


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    labels: Optional[Json] = None


class ProjectResponse(ProjectBase):
    id: Optional[int] = 0
    created_at: Optional[datetime] = datetime.now(timezone.utc)
    size: Optional[float] = 0.0
    dataset: Optional[str] = ""
    num_train: Optional[int] = 0
    num_val: Optional[int] = 0
    num_test: Optional[int] = 0
    num_trained_models: Optional[int] = 0
    new_models_this_month: Optional[int] = 0
    labels: Optional[List[dict]] = []
    type: Optional[str] = ""
    path: Optional[str] = ""

    # json_encoders is dropped: serialising datetime to ISO 8601 is Pydantic v2's
    # default behaviour, so the explicit encoder was redundant.
    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_orm(cls, obj):
        data = {field: getattr(obj, field) for field in cls.model_fields}
        return cls(**data)


def get_db():
    db = Session(db_manager.main_engine)
    try:
        yield db
    finally:
        db.close()


@router.post("/projects", response_model=ProjectResponse)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    db_project = Project(**project.model_dump())
    db.add(db_project)
    db.commit()
    db.refresh(db_project)

    # Create project folder
    project_folder = pathlib.Path(config.PROJECTS_ROOT) / str(db_project.id)
    project_folder.mkdir(parents=True, exist_ok=True)

    # Initialize project database and run migrations
    migration_manager = MigrationManager()
    migration_manager.create_new_project(db_project.id)

    return ProjectResponse.from_orm(db_project)


@router.get("/projects", response_model=List[ProjectResponse])
def read_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    # Validate skip and limit
    skip = max(0, skip)  # Ensure skip is not negative
    limit = max(1, min(100, limit))  # Ensure limit is between 1 and 100

    projects = db.query(Project).offset(skip).limit(limit).all()

    # Validate each project and handle any invalid data
    validated_projects = []
    for p in projects:
        try:
            validated_project = ProjectResponse.from_orm(p)
            validated_projects.append(validated_project)
        except Exception as e:
            # Log validation error but continue processing other projects
            # Extract field name from validation error message if possible
            error_msg = str(e)
            field_match = re.search(r"field '(\w+)'", error_msg)
            field_name = field_match.group(1) if field_match else "unknown field"
            logging.error(
                f"Validation failed for project {p.id} on field '{field_name}': {error_msg}"
            )
            continue

    return validated_projects


@router.get("/projects/{project_id}", response_model=ProjectResponse)
def read_project(project_id: int, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Calculate project size
    project_folder = pathlib.Path(config.PROJECTS_ROOT) / str(project_id)
    project_size = sum(
        f.stat().st_size for f in project_folder.glob("**/*") if f.is_file()
    )
    project_size_gb = round(project_size / (1024 * 1024 * 1024), 2)

    # Update project size in database
    db_project.size = project_size_gb
    db.commit()
    db.refresh(db_project)

    return ProjectResponse.from_orm(db_project)


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: int, project: ProjectUpdate, db: Session = Depends(get_db)
):
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # model_dump, not dict(): the v1 spelling is deprecated in Pydantic 2 and
    # removed in 3, and nothing exercised this endpoint to surface the warning.
    update_data = project.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_project, key, value)

    db.commit()
    db.refresh(db_project)

    return ProjectResponse.from_orm(db_project)


@router.delete("/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Release the database
    try:
        db_manager.release_database(project_id)
    except Exception as e:
        print(f"Error releasing database: {e}")

    # Delete project folder
    try:
        project_folder = pathlib.Path(config.PROJECTS_ROOT) / str(project_id)
        if project_folder.exists():
            try:
                shutil.rmtree(project_folder)
            except OSError:
                # If rmtree fails, try force removing read-only files and try again
                for root, dirs, files in os.walk(project_folder, topdown=False):
                    for name in files:
                        filename = os.path.join(root, name)
                        os.chmod(filename, stat.S_IWUSR)
                        os.remove(filename)
                    for name in dirs:
                        dirname = os.path.join(root, name)
                        os.chmod(dirname, stat.S_IWUSR)
                        os.rmdir(dirname)
                os.rmdir(project_folder)
    except Exception as e:
        print(f"Error deleting project folder: {e}")

    db.delete(db_project)
    db.commit()
    return "OK"


def create_export(project_id: int, project_folder: pathlib.Path, export_path: str):
    try:
        # Save project metadata
        db = Session(db_manager.main_engine)
        db_project = db.query(Project).filter(Project.id == project_id).first()

        metadata_path = project_folder / "project_metadata.json"
        project_data = ProjectResponse.from_orm(db_project).model_dump()

        # Ensure all fields are present with default values if missing
        project_data.setdefault("type", "")
        project_data.setdefault("path", "")
        project_data.setdefault("num_train", 0)
        project_data.setdefault("num_val", 0)
        project_data.setdefault("num_test", 0)

        with open(metadata_path, "w") as f:
            json.dump(project_data, f, indent=2, default=str)

        # Create tar archive with compression
        total_size = sum(
            f.stat().st_size for f in project_folder.glob("**/*") if f.is_file()
        )
        processed_size = 0

        with tarfile.open(export_path, "w:gz") as tar:
            for root, _, files in os.walk(project_folder):
                # Check if export was canceled
                if (
                    export_processes
                    and project_id in export_processes
                    and export_processes[project_id]["status"] == "canceled"
                ):
                    if os.path.exists(export_path):
                        os.remove(export_path)
                    if os.path.exists(metadata_path):
                        metadata_path.unlink()
                    db.close()
                    return

                for file in files:
                    file_path = os.path.join(root, file)
                    tar.add(
                        file_path,
                        arcname=os.path.relpath(file_path, project_folder.parent),
                    )
                    processed_size += os.path.getsize(file_path)
                    progress = int((processed_size / total_size) * 100)
                    export_processes[project_id]["progress"] = progress

        # Remove metadata file after adding to archive
        metadata_path.unlink()

        export_processes[project_id]["status"] = "completed"
        db.close()

    except Exception as e:
        logging.error(f"Error creating export: {e}")
        logging.error(traceback.format_exc())
        export_processes[project_id]["status"] = "failed"
        export_processes[project_id]["error"] = str(e)


@router.post("/projects/{project_id}/export")
def start_export_project(
    project_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start creating a compressed export of the project"""
    if (
        project_id in export_processes
        and export_processes[project_id]["status"] == "in_progress"
    ):
        raise HTTPException(status_code=400, detail="Export already in progress")

    db_project = db.query(Project).filter(Project.id == project_id).first()
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    project_folder = pathlib.Path(config.PROJECTS_ROOT) / str(project_id)
    if not project_folder.exists():
        raise HTTPException(status_code=404, detail="Project folder not found")

    export_dir = pathlib.Path(config.PROJECTS_ROOT) / "exports"
    export_dir.mkdir(exist_ok=True)
    export_path = export_dir / f"project_{project_id}_export.tar.gz"

    export_processes[project_id] = {
        "status": "in_progress",
        "progress": 0,
        "path": str(export_path),
    }

    # Start export in background task
    background_tasks.add_task(create_export, project_id, project_folder, export_path)

    return {"message": "Export started", "project_id": project_id}


# operation_id is explicit because FastAPI derives it from the function name plus
# the path with "/" normalised to "_". That turns this route's "export/status"
# into the same id as dataset.py's "export_status" route, and duplicate operation
# ids break generated API clients. The URLs themselves are distinct and unchanged.
@router.get(
    "/projects/{project_id}/export/status", operation_id="get_project_export_status"
)
def get_export_status(project_id: int):
    """Get the status of an export process"""
    if project_id not in export_processes:
        raise HTTPException(status_code=404, detail="No export process found")

    return export_processes[project_id]


@router.get("/projects/{project_id}/export/download")
def download_export(project_id: int):
    """Download the completed export file"""
    if project_id not in export_processes:
        raise HTTPException(status_code=404, detail="No export found")

    if export_processes[project_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Export not ready")

    export_path = export_processes[project_id]["path"]
    if not os.path.exists(export_path):
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        export_path,
        media_type="application/gzip",
        filename=f"project_{project_id}_export.tar.gz",
    )


@router.post("/projects/{project_id}/export/cancel")
def cancel_export(project_id: int):
    """Cancel an in-progress export"""
    if project_id not in export_processes:
        raise HTTPException(status_code=404, detail="No export process found")

    if export_processes[project_id]["status"] != "in_progress":
        raise HTTPException(status_code=400, detail="Export is not in progress")

    export_processes[project_id]["status"] = "canceled"
    return {"message": "Export canceled"}


@router.delete("/projects/{project_id}/export")
def delete_export(project_id: int):
    """Delete the export file for a project"""
    if project_id not in export_processes:
        raise HTTPException(status_code=404, detail="No export found")

    export_path = export_processes[project_id]["path"]
    if os.path.exists(export_path):
        os.remove(export_path)

    del export_processes[project_id]
    return {"message": "Export deleted successfully"}


def import_project_task(import_path: str, import_id: str, db: Session):
    """Background task to import a project"""
    try:
        # Create temporary directory to extract import
        tmp_dir = tempfile.mkdtemp()
        try:
            # Update status to extracting
            import_processes[import_id]["status"] = "extracting"
            import_processes[import_id]["progress"] = 10

            # Copy import file to temporary directory
            tmp_import_path = os.path.join(tmp_dir, os.path.basename(import_path))

            # Ensure import file exists before copying
            if not os.path.exists(import_path):
                raise FileNotFoundError(f"Import file not found at {import_path}")

            shutil.copy2(import_path, tmp_import_path)
            import_processes[import_id]["progress"] = 20

            # Extract import to temporary directory.
            #
            # filter="data" is not optional here. Without it tarfile runs in
            # its legacy "fully trusted" mode, and a member named `../../x`
            # writes outside tmp_dir -- on a file the user was handed by
            # someone else, because a project archive is exactly the kind of
            # thing people email each other. It also refuses absolute paths,
            # device nodes and links pointing out of the tree.
            #
            # Python 3.14 makes this the default and 3.13 warns about it; the
            # warning was being silenced by an ignore in pyproject.toml that
            # claimed the only unfiltered extraction was torch's, which stopped
            # being true here.
            with tarfile.open(tmp_import_path, "r:gz") as tar:
                tar.extractall(tmp_dir, filter="data")
            import_processes[import_id]["progress"] = 40

            # Read project metadata
            metadata_files = list(pathlib.Path(tmp_dir).glob("*/project_metadata.json"))
            if not metadata_files:
                raise Exception("No metadata file found in import")

            metadata_path = metadata_files[0]
            with open(metadata_path, "r") as f:
                project_data = json.load(f)
            import_processes[import_id]["progress"] = 50

            # Validate and sanitize imported data
            def validate_int(value, field_name, default=0):
                try:
                    return int(value) if value is not None else default
                except (ValueError, TypeError):
                    logging.warning(
                        f"Invalid {field_name} value: {value}, using default {default}"
                    )
                    return default

            def validate_str(value, field_name, default=""):
                return str(value) if value is not None else default

            def validate_float(value, field_name, default=0.0):
                """`size` is written only when a project's detail page is
                opened, so an export taken before that carries "size": null --
                and `.get("size", 0.0)` returns None for a key that is present
                and null. float(None) then ended the import at 50% with
                "float() argument must be a string or a real number", which
                names nothing the user could act on."""
                try:
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    logging.warning(
                        f"Invalid {field_name} value: {value}, using default {default}"
                    )
                    return default

            # Create new project in database with validated data
            db_project = Project(
                name=validate_str(project_data.get("name"), "name", "Imported Project"),
                description=validate_str(
                    project_data.get("description"), "description", ""
                ),
                labels=project_data.get("labels", []),
                dataset=validate_str(project_data.get("dataset"), "dataset", ""),
                num_train=validate_int(project_data.get("num_train"), "num_train"),
                num_val=validate_int(project_data.get("num_val"), "num_val"),
                num_test=validate_int(project_data.get("num_test"), "num_test"),
                num_trained_models=validate_int(
                    project_data.get("num_trained_models"),
                    "num_trained_models",
                ),
                new_models_this_month=validate_int(
                    project_data.get("new_models_this_month"),
                    "new_models_this_month",
                ),
                size=validate_float(project_data.get("size"), "size"),
                type=validate_str(project_data.get("type"), "type"),
                path=validate_str(project_data.get("path"), "path"),
            )

            try:
                db.add(db_project)
                db.commit()
                db.refresh(db_project)
                import_processes[import_id]["progress"] = 60

                # Move extracted files to new project directory
                project_folder = pathlib.Path(config.PROJECTS_ROOT) / str(db_project.id)
                if project_folder.exists():
                    shutil.rmtree(project_folder)
                project_folder.mkdir(parents=True, exist_ok=True)

                import_project_dir = metadata_path.parent
                total_files = len(list(import_project_dir.iterdir()))
                processed_files = 0

                for item in import_project_dir.iterdir():
                    if item.name != "project_metadata.json":
                        if item.is_dir():
                            shutil.copytree(item, project_folder / item.name)
                        else:
                            shutil.copy2(item, project_folder / item.name)
                    processed_files += 1
                    progress = 60 + int((processed_files / total_files) * 30)
                    import_processes[import_id]["progress"] = progress

                # Initialize project database
                project_engine = db_manager.refresh_connection(db_project.id)
                Base.metadata.create_all(project_engine)
                import_processes[import_id]["progress"] = 100
                import_processes[import_id]["status"] = "completed"
                import_processes[import_id]["project_id"] = db_project.id

            except Exception as e:
                # Rollback database changes
                db.rollback()
                # Clean up project folder if it was created
                if project_folder.exists():
                    shutil.rmtree(project_folder)
                # Release database connection
                db_manager.release_database(db_project.id)
                raise e

        finally:
            # Clean up temporary directory
            shutil.rmtree(tmp_dir)
            # Clean up import file
            if os.path.exists(import_path):
                os.remove(import_path)

    except FileNotFoundError as e:
        logging.error(f"Import file not found: {e}")
        logging.error(traceback.format_exc())
        import_processes[import_id]["status"] = "failed"
        import_processes[import_id]["error"] = str(e)
        raise
    except Exception as e:
        logging.error(f"Error importing project: {e}")
        logging.error(traceback.format_exc())
        import_processes[import_id]["status"] = "failed"
        import_processes[import_id]["error"] = str(e)
        raise


@router.post("/projects/import")
def import_project(
    background_tasks: BackgroundTasks,
    import_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Import a project from an export file"""
    if not import_file.filename.endswith(".tar.gz"):
        raise HTTPException(status_code=400, detail="Invalid import file format")

    # Generate unique ID for this import
    import_id = str(uuid.uuid4())
    import_processes[import_id] = {
        "status": "uploading",
        "progress": 0,
        "error": None,
        "project_id": None,
    }

    # Save uploaded file to a persistent location
    upload_dir = tempfile.gettempdir()
    import_path = os.path.join(upload_dir, f"import_{import_id}.tar.gz")

    # Save uploaded file
    with open(import_path, "wb") as f:
        content = import_file.file.read()
        f.write(content)

    # Start import in background
    background_tasks.add_task(import_project_task, import_path, import_id, db)

    return {"message": "Project import started", "import_id": import_id}


@router.get("/projects/import/{import_id}/status")
def get_import_status(import_id: str):
    """Get the status of an import process"""
    if import_id not in import_processes:
        raise HTTPException(status_code=404, detail="No import process found")

    return import_processes[import_id]

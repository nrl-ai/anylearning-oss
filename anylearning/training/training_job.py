import json
import logging
import os
import pathlib
import shutil
import time
import traceback
import uuid
from anylearning import config, frozen_compat
from anylearning.database import (
    Model,
    Project,
    TrainingParams,
    TrainingSession,
    TrainingSessionStatus,
    db_manager,
)
from anylearning.training import device_utils
from anylearning.training.logging import TrainingLogsWriter
from anylearning.training.trainers.trainer_builder import TrainerBuilder
from anylearning.structured import is_structured_project
from datetime import datetime, timezone
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified


def release_log_files_under(folder):
    """Close log files inside `folder`, so the folder can actually be deleted.

    The trainers attach a FileHandler inside the training folder -- nanodet's
    logger writes logs.txt there -- and Windows refuses to delete a file any
    handle still holds open. `rmtree(ignore_errors=True)` then swallows the
    failure and leaves the whole working folder behind, so every training run
    on Windows leaks one, quietly. Nothing needs these handlers after the run.
    """
    root = os.path.abspath(str(folder))
    loggers = [logging.getLogger()] + [
        logging.getLogger(name) for name in list(logging.Logger.manager.loggerDict)
    ]
    for log in loggers:
        for handler in list(getattr(log, "handlers", [])):
            filename = getattr(handler, "baseFilename", None)
            if not filename or not os.path.abspath(filename).startswith(root):
                continue
            try:
                handler.close()
            finally:
                log.removeHandler(handler)


def apply_device_preference(training_params: TrainingParams) -> None:
    """Make the user's CPU/GPU choice true for everything this process runs.

    Set before any training code touches torch, and set two ways on purpose:

    * `ANYLEARNING_TRAINING_DEVICE` is what `device_utils.device_type()`
      reads, and it is the only mechanism that works when CUDA has already been
      initialised in this process -- which happens on Linux, where the child is
      forked from an API process that may already have asked about the GPU.
    * `CUDA_VISIBLE_DEVICES=""` hides the hardware from libraries that never ask
      us, which is most of the vendored stack: torch, Lightning and detectron2
      each decide for themselves several layers down.

    "gpu" is a preference, not a demand. A machine without an accelerator still
    trains, on the CPU, with a line in the log saying so -- refusing to start
    would turn a portable project into one that only runs where it was made.
    The same reason an accelerator id is stored as plain "gpu": a project
    trained on "cuda" opens on a Mac, where "the GPU" is Metal, and asks for the
    GPU that machine has rather than for a card it does not.

    So whatever the dialog sends, only three values ever reach the environment
    and the vendored trainers that read it: auto, gpu, cpu.
    """
    preference = (getattr(training_params, "device", None) or device_utils.AUTO).lower()
    if preference in {device_utils.CUDA, device_utils.MPS}:
        preference = device_utils.GPU
    if preference not in {device_utils.AUTO, device_utils.GPU, device_utils.CPU}:
        preference = device_utils.AUTO
    os.environ[device_utils.DEVICE_PREFERENCE_ENV] = preference
    if preference == device_utils.CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def describe_device(training_params: TrainingParams) -> str:
    """One line for the training log: what was asked for, and what was used.

    The log is the only place a user can see this. A run that quietly fell back
    to the CPU because the GPU was invisible looks exactly like a slow run.

    The name comes from nvidia-smi rather than torch: `get_device_name()`
    initialises CUDA, and this line runs before training does. In a child forked
    from a parent that had already initialised CUDA it raised instead -- so
    writing the log line was what ended the run, before a single batch.
    """
    preference = device_utils.device_preference()
    using_gpu = device_utils.gpu_available()
    name = device_utils.gpu_name() if using_gpu else None
    if device_utils.device_type() == device_utils.MPS:
        # Say Metal, not just "GPU". On a Mac the two are easy to confuse, and
        # which one ran is the first thing to know when a result looks wrong.
        where = f"GPU (Apple Metal, {name})" if name else "GPU (Apple Metal)"
    else:
        where = f"GPU ({name})" if name else ("GPU" if using_gpu else "CPU")
    if preference == device_utils.GPU and not using_gpu:
        return f"{where} -- a GPU was requested, but this machine has none"
    if preference == device_utils.CPU:
        return f"{where} (you chose it)"
    return where


def require_labelled_splits(project_id: int, logger) -> None:
    """Both training and validation need labelled images. Say so here.

    The dataset tab uploads into whichever split is selected, so putting
    everything in Training and pressing Start training is something people do
    on their first afternoon. Only the classification trainer caught it; the
    others failed from inside their framework with messages that name neither
    the cause nor the fix -- handpose managed
    `RuntimeError: stack expects a non-empty TensorList` -- and the advice
    engine then blamed the batch size, confidently and wrongly.

    Checked here rather than in five trainers, because every one of them needs
    it and none of them should have to remember.
    """
    from anylearning.database import DataItem

    with Session(db_manager.get_project_engine(project_id)) as session:
        counts = {
            name: session.query(DataItem)
            .filter(DataItem.subset == subset, DataItem.labeled != 0)
            .count()
            for name, subset in (("train", 0), ("val", 1))
        }

    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        detail = ", ".join(f"{name}: {counts[name]}" for name in ("train", "val"))
        splits = " and ".join(empty)
        message = (
            f"The {splits} split{'s have' if len(empty) > 1 else ' has'} no "
            f"labelled images "
            f"({detail}). Assign images to it in the Dataset tab before "
            "training -- train and val are both required, test is optional."
        )
        logger.write(message)
        raise ValueError(message)


def downgrade_unsupported_accelerator(project_type: str) -> str | None:
    """Pin this run to the CPU when its accelerator cannot train this type.

    Applied here, in one place, rather than trusted to each trainer: it has to
    hold for a run started from the API by a client that never saw the dialog,
    and for a project moved from a machine where the choice was fine to one
    where it is not. Returns the reason it downgraded, for the log, or None.

    Refusing outright would be worse. The project is portable; the limitation is
    this machine's, and the run still produces a model.
    """
    reason = device_utils.excluded_reason(project_type)
    if not reason:
        return None
    os.environ[device_utils.DEVICE_PREFERENCE_ENV] = device_utils.CPU
    return reason


def run_training_job(
    project_id: int, training_session_id: int, training_params: TrainingParams
):
    """
    Training job for a single project.
    """
    # Runs in its own process -- and on macOS and Windows that process is
    # spawned, not forked, so it re-imports every library from scratch and
    # inherits nothing the parent repaired.
    frozen_compat.apply()
    apply_device_preference(training_params)

    start_time = time.time()
    logger = TrainingLogsWriter(project_id, training_session_id)
    training_folder = None

    # Get the project
    with Session(db_manager.main_engine) as global_session:
        project = global_session.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    try:
        logger.write(f"Training device: {describe_device(training_params)}")
        # After the project is known, because which accelerators can train it
        # depends on its type. Written as a second "Training device:" line
        # rather than folded into the first: the log is read from the bottom,
        # by people and by smoke_test_training's device assertion, so the last
        # one has to be where the run actually happened.
        downgraded = downgrade_unsupported_accelerator(project.type)
        if downgraded:
            logger.write(f"Training device: CPU -- {downgraded}")

        # Create training folder
        training_folder = (
            pathlib.Path(config.PROJECTS_ROOT)
            / str(project_id)
            / "training"
            / str(training_session_id)
        )

        # The architecture as well as the type: a detection project can be
        # trained with NanoDet or with RF-DETR, and which one this run means is
        # in the parameters the dialog sent, not in the project.
        trainer_class = TrainerBuilder.get_trainer_class(
            project.type, getattr(training_params, "model_architecture", None)
        )
        trainer = trainer_class(training_folder, logger, project_id, training_params)

        if not is_structured_project(project.type):
            require_labelled_splits(project_id, logger)

        # Prepare data
        trainer.prepare_data()

        # Prepare config
        config_data = trainer.prepare_config()

        # Update training session status
        with Session(db_manager.get_project_engine(project_id)) as session:
            training_session = (
                session.query(TrainingSession).filter_by(id=training_session_id).first()
            )
            training_session.status = TrainingSessionStatus.TRAINING.value
            training_session.config_file = config_data
            session.commit()

        # Run the actual training job
        logger.write("Training started...")

        # Run the training command as a subprocess
        trainer.train()
        onnx_path = trainer.export_onnx()

        # Save the model
        with Session(db_manager.get_project_engine(project_id)) as session:
            training_session = (
                session.query(TrainingSession).filter_by(id=training_session_id).first()
            )
            metric_logs = training_session.metric_logs
            if isinstance(metric_logs, str):
                metric_logs = json.loads(metric_logs)
                training_session.metric_logs = metric_logs
                flag_modified(training_session, "metric_logs")
            ret, model_best_path = trainer.get_model_path()
            if not ret:
                raise RuntimeError(
                    "Training output validation failed. No model found in training output."
                )
            MODELS_FOLDER = (
                pathlib.Path(config.PROJECTS_ROOT) / str(project_id) / "models"
            )
            relative_model_path = pathlib.Path(
                f"session_{training_session_id}"
            ) / os.path.basename(model_best_path)
            full_model_path = MODELS_FOLDER / relative_model_path
            full_model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(model_best_path, full_model_path)

            # Whatever else this trainer needs to be usable later. The run's
            # folder is deleted after this, so a file left behind there is a
            # file the model no longer has -- which is how every instance
            # segmentation model ended up unable to answer the Try dialog.
            # getattr, not a direct call: this runs after a successful train
            # and export, and an AttributeError here would throw away a
            # finished model over a method a trainer did not implement.
            for companion in getattr(trainer, "companion_files", list)():
                if os.path.isfile(companion):
                    shutil.copy(
                        companion, full_model_path.parent / os.path.basename(companion)
                    )
                    logger.write(f"Kept {os.path.basename(companion)} with the model.")

            relative_onnx_path = None
            if onnx_path:
                relative_onnx_path = pathlib.Path(
                    f"session_{training_session_id}"
                ) / os.path.basename(onnx_path)
                full_onnx_path = MODELS_FOLDER / relative_onnx_path
                shutil.copy(onnx_path, full_onnx_path)
            else:
                logger.write("Warning: No ONNX model found")

            # Save the model to the database
            model = Model(
                training_session_id=training_session_id,
                name=f"Model from training session {training_session_id} - "
                + uuid.uuid4().hex[:6],
                description="Trained model",
                path=str(relative_model_path),
                model_architecture=training_params.model_architecture,
                model_size=training_params.model_size,
                config_file=training_session.config_file,
                test_result=metric_logs[-1] if metric_logs else None,
                exported_path=str(relative_onnx_path) if relative_onnx_path else None,
            )
            session.add(model)
            session.commit()

            logger.write(
                f"Model saved successfully. Best model path: {model_best_path}"
            )

            # Update training session status when done
            training_session = (
                session.query(TrainingSession).filter_by(id=training_session_id).first()
            )
            training_session.status = TrainingSessionStatus.FINISHED.value
            session.commit()

        logger.write(f"Training completed in {time.time() - start_time:.2f} seconds.")

    except Exception as e:
        traceback.print_exc()
        logger.write(f"Error during training: {str(e)} {traceback.format_exc()}")
        with Session(db_manager.get_project_engine(project_id)) as session:
            training_session = (
                session.query(TrainingSession).filter_by(id=training_session_id).first()
            )
            if training_session:
                training_session.status = TrainingSessionStatus.ERROR.value
                # No error_message column here, and nothing reads one: assigning
                # it only set a transient attribute that the commit discarded.
                # The reason reaches the user through training_logs, written
                # just above.
                session.commit()
            else:
                logger.write(
                    "Warning: Could not update training session status - session not found"
                )
        raise  # Re-raise the exception to ensure proper error handling up the stack
    finally:
        # Clean up temporary folder
        if training_folder and os.environ.get("ANYLEARNING_DEVELOPMENT") != "TRUE":
            try:
                release_log_files_under(training_folder)
                shutil.rmtree(training_folder, ignore_errors=True)
            except Exception as cleanup_error:
                logger.write(
                    f"Warning: Failed to clean up temporary folder: {cleanup_error}"
                )

        # Mark the training session as finished.
        #
        # The row may be gone: this runs in a `finally`, and deleting a project
        # while one of its runs is in flight takes its sessions with it. The
        # except block above already handles that and says so -- and then this
        # raised `AttributeError: 'NoneType' object has no attribute 'ended_at'`
        # from the finally, which replaces the exception on its way out. The
        # user was shown a null-attribute error instead of whatever actually
        # ended their run.
        with Session(db_manager.get_project_engine(project_id)) as session:
            training_session = (
                session.query(TrainingSession).filter_by(id=training_session_id).first()
            )
            if training_session is not None:
                training_session.ended_at = datetime.now(timezone.utc)
                session.commit()

import math
import multiprocessing
import uuid
from datetime import datetime, timezone

import psutil
from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import desc
from sqlalchemy.orm import Session

from anylearning.database import (
    DataItem,
    Project,
    TrainingParams,
    TrainingProcess,
    TrainingSession,
    TrainingSessionStatus,
    db_manager,
)
from anylearning.training import diagnostics
from anylearning.training.training_job import run_training_job


def training_process_context():
    """A clean interpreter for every training run.

    Linux defaults to ``fork``, which is unsafe after the API process has run
    CUDA inference: the child inherits CUDA's locks and can wait on them
    forever before its first batch. macOS and Windows already spawn. Using the
    same method everywhere also keeps development and packaged behaviour
    aligned.
    """
    return multiprocessing.get_context("spawn")


def training_process_ended(pid: int | None) -> bool:
    """Whether the process behind a session is gone.

    Every "did this run die?" check goes through here, because getting it wrong
    is expensive in both directions and both were happening:

    * **A zombie is not alive.** The job runs in a `multiprocessing.Process`
      that nothing joins, so when it exits it stays in the process table as a
      zombie -- and `psutil.Process.is_running()` answers True for a zombie. A
      run whose child was killed (this one was killed by the OOM killer,
      mid-`torch.save`) therefore sat at "training" for as long as the app was
      open. Worse, `POST /training_sessions` refuses to start a run while
      another is ongoing, so that project could not be trained again at all.
      `active_children()` reaps the exited ones, which is also what stops this
      process from accumulating zombies in the first place.

    * **A dataloader worker exiting is not the run dying.** The detail endpoint
      used to walk `parent.children(recursive=True)`, and psutil raises
      NoSuchProcess when a child disappears *during* the walk -- which workers
      do constantly. That exception was caught by a handler that marked the
      session ERROR, so a healthy run reported "failed" to the UI for a poll or
      two, several times a run, and then carried on training.

    Anything else psutil raises (a permissions problem, a race reading /proc)
    is not evidence that the run died, so the status is left alone.
    """
    if not pid:
        return True
    multiprocessing.active_children()
    try:
        process = psutil.Process(pid)
        return not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True
    except psutil.Error:
        return False


def count_project_classes(project_id: int) -> int | None:
    """How many classes the project has, for the advice about chance accuracy.

    None when it cannot be read: the advice is a nicety and must never be the
    reason a session fails to load."""
    try:
        with Session(db_manager.main_engine) as session:
            project = session.query(Project).filter(Project.id == project_id).first()
            return len(project.labels) if project and project.labels else None
    except Exception:  # noqa: BLE001
        return None


def count_training_images(project_id: int) -> int | None:
    """How many images are in the training subset, for the advice about batch
    size. None when it cannot be read: advice is a nicety and must never be the
    reason a session fails to load."""
    try:
        with Session(db_manager.get_project_engine(project_id)) as session:
            return session.query(DataItem).filter(DataItem.subset == 0).count()
    except Exception:  # noqa: BLE001
        return None


def json_safe_metrics(metric_logs):
    """Replace NaN and infinity with null, recursively.

    A training run can legitimately record a NaN -- a loss that diverged, a
    metric averaged over an empty batch -- and JSON has no way to represent
    one. FastAPI refuses to serialise it, so every request for that session
    answered 500 and the project's whole training tab became unreachable, for
    good: the value is on the row and nothing rewrites it.

    Null is the honest translation. The chart already skips missing points, so
    a diverged epoch shows as a gap rather than as an outage.
    """
    if isinstance(metric_logs, float):
        return (
            None if math.isnan(metric_logs) or math.isinf(metric_logs) else metric_logs
        )
    if isinstance(metric_logs, dict):
        return {key: json_safe_metrics(value) for key, value in metric_logs.items()}
    if isinstance(metric_logs, list):
        return [json_safe_metrics(value) for value in metric_logs]
    return metric_logs


router = APIRouter(prefix="/api", tags=["Training"])


@router.post("/projects/{project_id}/training_sessions")
async def start_training(
    project_id: int,
    training_params: TrainingParams,
    background_tasks: BackgroundTasks,
):
    with Session(db_manager.main_engine) as global_session:
        project = global_session.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    with Session(db_manager.get_project_engine(project_id)) as session:
        try:
            # Check if there's an ongoing training session
            ongoing_session = (
                session.query(TrainingSession)
                .filter(
                    TrainingSession.status.in_(
                        [
                            TrainingSessionStatus.NOT_STARTED.value,
                            TrainingSessionStatus.TRAINING.value,
                            TrainingSessionStatus.EVALUATING.value,
                        ]
                    )
                )
                .first()
            )

            if ongoing_session:
                pid = ongoing_session.process.pid if ongoing_session.process else None
                if training_process_ended(pid):
                    # The previous run is over and did not get to say so --
                    # record that, and let this one start.
                    ongoing_session.status = TrainingSessionStatus.ERROR.value
                    ongoing_session.ended_at = datetime.now(timezone.utc)
                    if ongoing_session.process:
                        ongoing_session.process.status = "terminated"
                        ongoing_session.process.ended_at = datetime.now(timezone.utc)
                    session.commit()
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="A training session is already in progress. Please wait for it to complete.",
                    )

            new_training_session = TrainingSession(
                name=f"Training Session {str(uuid.uuid4())[:8]}",
                description="Automatically created training session",
                status=TrainingSessionStatus.NOT_STARTED.value,
                params=training_params.model_dump(),
                started_at=datetime.now(timezone.utc),
            )
            session.add(new_training_session)
            session.commit()

            # Start training in a separate process
            process = training_process_context().Process(
                target=run_training_job,
                args=(project_id, new_training_session.id, training_params),
            )
            process.start()

            # Create and store process info
            training_process = TrainingProcess(
                training_session_id=new_training_session.id,
                pid=process.pid,
                status="running",
            )
            session.add(training_process)
            session.commit()

            return {
                "message": "Training session created and started successfully",
                "session_id": new_training_session.id,
            }
        except HTTPException:
            raise
        except Exception as e:
            session.rollback()
            # Automatically set the job status to ERROR
            new_training_session.status = TrainingSessionStatus.ERROR.value
            new_training_session.ended_at = datetime.now(timezone.utc)
            session.add(new_training_session)
            session.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Error creating training session: {str(e)}",
            )


@router.get("/projects/{project_id}/training_sessions")
async def get_training_sessions(project_id: int):
    with Session(db_manager.main_engine) as global_session:
        project = global_session.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    with Session(db_manager.get_project_engine(project_id)) as session:
        try:
            # Mark all jobs with invalid process as ERROR
            ongoing_sessions = (
                session.query(TrainingSession)
                .filter(
                    TrainingSession.status.in_(
                        [
                            TrainingSessionStatus.NOT_STARTED.value,
                            TrainingSessionStatus.TRAINING.value,
                            TrainingSessionStatus.EVALUATING.value,
                        ]
                    )
                )
                .all()
            )

            for ts in ongoing_sessions:
                if not training_process_ended(ts.process.pid if ts.process else None):
                    continue
                ts.status = TrainingSessionStatus.ERROR.value
                ts.ended_at = datetime.now(timezone.utc)
                session.add(ts)
                if ts.process:
                    ts.process.status = "terminated"
                    ts.process.ended_at = datetime.now(timezone.utc)
                    session.add(ts.process)

            session.commit()

            training_sessions = (
                session.query(TrainingSession)
                .order_by(desc(TrainingSession.id))
                .limit(10)
                .all()
            )
            return [
                {
                    "id": ts.id,
                    "name": ts.name,
                    "description": ts.description,
                    "status": ts.status,
                    "started_at": ts.started_at,
                    "ended_at": ts.ended_at,
                    "params": ts.params,
                    "metric_logs": json_safe_metrics(ts.metric_logs),
                    "model": {
                        "id": ts.model.id if ts.model else None,
                        "name": ts.model.name if ts.model else None,
                    },
                }
                for ts in training_sessions
            ]
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching training sessions: {str(e)}",
            )


@router.get("/projects/{project_id}/training_sessions/{session_id}")
async def get_training_session(project_id: int, session_id: int):
    with Session(db_manager.main_engine) as global_session:
        project = global_session.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    with Session(db_manager.get_project_engine(project_id)) as session:
        try:
            training_session = (
                session.query(TrainingSession).filter_by(id=session_id).first()
            )
            if not training_session:
                raise HTTPException(
                    status_code=404, detail="Training session not found"
                )

            # Check if the process is still active for ongoing sessions
            if training_session.status in [
                TrainingSessionStatus.NOT_STARTED.value,
                TrainingSessionStatus.TRAINING.value,
                TrainingSessionStatus.EVALUATING.value,
            ]:
                pid = training_session.process.pid if training_session.process else None
                if training_process_ended(pid):
                    training_session.status = TrainingSessionStatus.ERROR.value
                    training_session.ended_at = datetime.now(timezone.utc)
                    if training_session.process:
                        training_session.process.status = "terminated"
                        training_session.process.ended_at = datetime.now(timezone.utc)
                    session.commit()

            return {
                "id": training_session.id,
                "name": training_session.name,
                "description": training_session.description,
                "status": training_session.status,
                "started_at": training_session.started_at,
                "ended_at": training_session.ended_at,
                "params": training_session.params,
                "metric_logs": json_safe_metrics(training_session.metric_logs),
                "training_logs": training_session.training_logs,
                # What to change before running again, when the numbers say
                # something went wrong. Computed rather than stored: the rules
                # improve, and an old session should get today's advice.
                "advice": diagnostics.advise(
                    training_session.params,
                    training_session.metric_logs,
                    training_session.status,
                    count_training_images(project_id),
                    count_project_classes(project_id),
                    training_session.training_logs,
                ),
                "model": {
                    "id": training_session.model.id if training_session.model else None,
                    "name": training_session.model.name
                    if training_session.model
                    else None,
                },
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching training session: {str(e)}",
            )


@router.get("/projects/{project_id}/last_training_session")
async def get_last_training_session(project_id: int):
    with Session(db_manager.main_engine) as global_session:
        project = global_session.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    with Session(db_manager.get_project_engine(project_id)) as session:
        try:
            last_training_session = (
                session.query(TrainingSession)
                .order_by(desc(TrainingSession.id))
                .first()
            )
            if not last_training_session:
                # A project that has never been trained is a normal state, not an
                # error. Returning 404 here made the UI -- which polls this
                # endpoint continuously -- log an error every few seconds for
                # every untrained project. null lets the client render its empty
                # state without treating it as a failure.
                return None

            # Check if the process is still active for ongoing sessions
            if last_training_session.status in [
                TrainingSessionStatus.NOT_STARTED.value,
                TrainingSessionStatus.TRAINING.value,
                TrainingSessionStatus.EVALUATING.value,
            ]:
                pid = (
                    last_training_session.process.pid
                    if last_training_session.process
                    else None
                )
                if training_process_ended(pid):
                    last_training_session.status = TrainingSessionStatus.ERROR.value
                    last_training_session.ended_at = datetime.now(timezone.utc)
                    if last_training_session.process:
                        last_training_session.process.status = "terminated"
                        last_training_session.process.ended_at = datetime.now(
                            timezone.utc
                        )
                    session.commit()

            return {
                "id": last_training_session.id,
                "name": last_training_session.name,
                "description": last_training_session.description,
                "status": last_training_session.status,
                "started_at": last_training_session.started_at,
                "ended_at": last_training_session.ended_at,
                "params": last_training_session.params,
                "metric_logs": json_safe_metrics(last_training_session.metric_logs),
                "advice": diagnostics.advise(
                    last_training_session.params,
                    last_training_session.metric_logs,
                    last_training_session.status,
                    count_training_images(project_id),
                    count_project_classes(project_id),
                    last_training_session.training_logs,
                ),
                "training_logs": last_training_session.training_logs,
                "model": {
                    "id": last_training_session.model.id
                    if last_training_session.model
                    else None,
                    "name": last_training_session.model.name
                    if last_training_session.model
                    else None,
                },
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching last training session: {str(e)}",
            )


@router.post("/projects/{project_id}/training_sessions/{session_id}/terminate")
async def terminate_training(project_id: int, session_id: int):
    with Session(db_manager.main_engine) as global_session:
        project = global_session.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    with Session(db_manager.get_project_engine(project_id)) as session:
        try:
            training_session = (
                session.query(TrainingSession).filter_by(id=session_id).first()
            )
            if not training_session:
                raise HTTPException(
                    status_code=404, detail="Training session not found"
                )

            if training_session.status not in [
                TrainingSessionStatus.NOT_STARTED.value,
                TrainingSessionStatus.TRAINING.value,
                TrainingSessionStatus.EVALUATING.value,
            ]:
                raise HTTPException(
                    status_code=400, detail="Training session is not in progress"
                )

            if training_session.process and training_session.process.pid:
                try:
                    # Children first: the dataloader workers outlive a
                    # terminated parent otherwise, and go on holding the GPU.
                    parent = psutil.Process(training_session.process.pid)
                    doomed = []
                    for child in parent.children(recursive=True):
                        try:
                            child.terminate()
                            doomed.append(child)
                        except psutil.Error:
                            # It exited between listing and terminating, which
                            # is the outcome we wanted anyway.
                            pass
                    parent.terminate()
                    doomed.append(parent)

                    # And kill whatever ignored that. SIGTERM is delivered to
                    # Python by setting a flag the interpreter checks between
                    # bytecodes, so a process wedged inside a C-level lock --
                    # which is exactly what a deadlocked training job is --
                    # never runs the handler and does not die. One was left
                    # sitting at "training" through two SIGTERMs.
                    _, alive = psutil.wait_procs(doomed, timeout=5)
                    for process in alive:
                        try:
                            process.kill()
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
                training_session.process.status = "terminated"
                training_session.process.ended_at = datetime.now(timezone.utc)

            training_session.status = TrainingSessionStatus.TERMINATED.value
            training_session.ended_at = datetime.now(timezone.utc)
            session.commit()

            return {"message": "Training session terminated successfully"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error terminating training session: {str(e)}",
            )

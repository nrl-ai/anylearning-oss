from anylearning.database import TrainingSession, db_manager
from datetime import datetime, timezone
from sqlalchemy import func, update
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified


class TrainingLogsWriter:
    def __init__(self, project_id: int, training_session_id: int):
        self.project_id = project_id
        self.training_session_id = training_session_id

    def write(self, message: str):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        new_log = f"[{timestamp}] {message}\n"

        # Append in SQL rather than reading the column back, concatenating in
        # Python and writing the whole thing again. This is on the training hot
        # path -- NanoDet logs every iteration -- and it saves a SELECT of a
        # column that grows all run.
        #
        # It does not change the growth curve: SQLite still rewrites the row, so
        # cost still climbs with log size (measured 1.3x faster at 220 KB, 1.6x
        # at 1.8 MB, both still superlinear). Making appends genuinely cheap
        # needs log lines in their own table; see docs/TODO.md.
        engine = db_manager.get_project_engine(self.project_id)
        with Session(engine) as session:
            result = session.execute(
                update(TrainingSession)
                .where(TrainingSession.id == self.training_session_id)
                .values(
                    training_logs=func.coalesce(TrainingSession.training_logs, "")
                    + new_log
                )
            )
            session.commit()
            if result.rowcount == 0:
                print(
                    f"Warning: Training session {self.training_session_id} not found."
                )

    def write_metrics(self, epoch_metrics: dict):
        engine = db_manager.get_project_engine(self.project_id)
        with Session(engine) as session:
            training_session = (
                session.query(TrainingSession)
                .filter_by(id=self.training_session_id)
                .first()
            )
            metric_logs = training_session.metric_logs or []
            metric_logs.append(epoch_metrics)
            training_session.metric_logs = metric_logs
            flag_modified(training_session, "metric_logs")
            session.add(training_session)
            session.commit()


class ConsoleLogsWriter:
    """Drop-in ``TrainingLogsWriter`` that prints instead of writing to the DB.

    The ``train_fn`` entry points declare ``logger: TrainingLogsWriter = None``,
    but then call ``logger.write(...)`` unconditionally -- so the documented
    default raised ``AttributeError`` the moment anyone ran training outside the
    app (a script, a notebook, or a test). This gives that default something
    real to fall back on, with no database or project id required.
    """

    def write(self, message: str):
        print(message)

    def write_metrics(self, epoch_metrics: dict):
        print(epoch_metrics)


class NanoDetLogger:
    name = "NanoDet"
    version = "1"

    def __init__(self, writer: TrainingLogsWriter, save_dir: str):
        self._experiment = None
        self._writer = writer
        self.save_dir = save_dir

    def info(self, msg):
        self._writer.write(msg)

    def warning(self, msg):
        self._writer.write(f"Warning: {msg}")

    def error(self, msg):
        self._writer.write(f"Error: {msg}")

    def finalize(self, status):
        self._writer.write(f"Finalize: {status}")

    def log_graph(self, model):
        self._writer.write(f"Log graph: {model}")

    def dump_cfg(self, cfg):
        self._writer.write(f"Dump config: {cfg}")

    def save(self):
        self._writer.write("Save")

    def after_save_checkpoint(self, model_checkpoint):
        self._writer.write("Saved the checkpoint.")

    def log_metrics(self, eval_results, epoch):
        self._writer.write(f"Epoch {epoch}, eval results: {eval_results}")
        current_epoch_metrics = {}
        current_epoch_metrics["Training Loss"] = eval_results.get("train_loss", 0)
        current_epoch_metrics["Validation Loss"] = eval_results.get("val_loss", 0)
        current_epoch_metrics["Validation mAP"] = eval_results.get("mAP", 0)
        self._writer.write_metrics(current_epoch_metrics)

    @property
    def experiment(self):
        if self._experiment is None:
            from torch.utils.tensorboard import SummaryWriter

            self._experiment = SummaryWriter(log_dir=self.save_dir)
        return self._experiment

    def add_scalars(self, tag, phase_value_dict, step):
        self.experiment.add_scalars(tag, phase_value_dict, step)


class MLPLogger:
    name = "MLP"
    version = "1"

    def __init__(self, writer: TrainingLogsWriter, save_dir: str):
        self._experiment = None
        self._writer = writer
        self.save_dir = save_dir

    def info(self, msg):
        self._writer.write(msg)

    def warning(self, msg):
        self._writer.write(f"Warning: {msg}")

    def error(self, msg):
        self._writer.write(f"Error: {msg}")

    def finalize(self, status):
        self._writer.write(f"Finalize: {status}")

    def log_graph(self, model):
        self._writer.write(f"Log graph: {model}")

    def dump_cfg(self, cfg):
        self._writer.write(f"Dump config: {cfg}")

    def save(self):
        self._writer.write("Save")

    def after_save_checkpoint(self, model_checkpoint):
        self._writer.write("Saved the checkpoint.")

    def log_metrics(self, eval_results, epoch):
        self._writer.write(f"Epoch {epoch}, eval results: {eval_results}")
        current_epoch_metrics = {}
        current_epoch_metrics["Validation Accuracy"] = eval_results.get(
            "val_accuracy", 0
        )
        current_epoch_metrics["Validation Loss"] = eval_results.get("val_loss", 0)
        current_epoch_metrics["Training Loss"] = eval_results.get("train_loss", 0)
        self._writer.write_metrics(current_epoch_metrics)

    @property
    def experiment(self):
        if self._experiment is None:
            from torch.utils.tensorboard import SummaryWriter

            self._experiment = SummaryWriter(log_dir=self.save_dir)
        return self._experiment

    def add_scalars(self, tag, phase_value_dict, step):
        self.experiment.add_scalars(tag, phase_value_dict, step)

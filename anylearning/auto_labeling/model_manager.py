import copy
import hashlib
import importlib.resources as pkg_resources
import logging
import os
import pathlib
import re
import shutil
import stat
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from enum import Enum
from threading import Lock, Thread
from typing import Any, Callable, Dict, List

import yaml

from anylearning.auto_labeling.types import AutoLabelingResult

# DATA_ROOT by name, not the module: `config` is already a local in
# load_model_configs (the app's own settings), and shadowing it there turned
# every model listing into an UnboundLocalError.
from anylearning.config import DATA_ROOT, get_config, save_config
from anylearning.configs import auto_labeling as auto_labeling_configs

logger = logging.getLogger(__name__)

GIBIBYTE = 1024**3
MAX_MODEL_DOWNLOAD_BYTES = 20 * GIBIBYTE
MAX_MODEL_ARCHIVE_BYTES = 40 * GIBIBYTE
MAX_MODEL_ARCHIVE_FILES = 256
MAX_MODEL_MEMBER_BYTES = 40 * GIBIBYTE
MAX_MODEL_COMPRESSION_RATIO = 100
MODEL_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MODEL_DOWNLOAD_TIMEOUT_SECONDS = 60
PINNED_MODEL_FIELDS = ("download_url", "sha256", "archive_size_bytes")
ALLOWED_MODEL_FILE_SUFFIXES = {
    ".json",
    ".names",
    ".onnx",
    ".txt",
    ".yaml",
    ".yml",
}


def _validate_https_url(url: str, *, redirected: bool = False) -> None:
    """Require an absolute HTTPS URL before and after network redirects."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        source = "redirected model URL" if redirected else "model URL"
        raise ValueError(f"{source} must use HTTPS.")


def _display_download_url(url: str) -> str:
    """Return a short URL label without credentials, query data, or fragments."""
    parsed = urllib.parse.urlsplit(url)
    filename = pathlib.PurePosixPath(parsed.path).name or "model archive"
    return f"{parsed.hostname or 'model host'}/.../{filename}"


def _download_model_archive(
    url: str,
    destination: pathlib.Path,
    *,
    expected_sha256: str,
    expected_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Stream a model archive to disk and verify its pinned manifest metadata."""
    _validate_https_url(url)
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise ValueError("Model archive requires a valid SHA-256 checksum.")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or not 0 < expected_size <= MAX_MODEL_DOWNLOAD_BYTES
    ):
        raise ValueError("Model archive requires a valid bounded size.")

    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AnyLearning model downloader"},
    )
    downloaded = 0
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(
            request, timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            _validate_https_url(response.geturl(), redirected=True)
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    reported_size = int(content_length)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Model archive response has an invalid size."
                    ) from error
                if reported_size != expected_size:
                    raise ValueError("Model archive size does not match manifest.")

            if progress is not None:
                progress(0, expected_size)
            with destination.open("xb") as output:
                while chunk := response.read(MODEL_DOWNLOAD_CHUNK_BYTES):
                    downloaded += len(chunk)
                    if (
                        downloaded > expected_size
                        or downloaded > MAX_MODEL_DOWNLOAD_BYTES
                    ):
                        raise ValueError("Model archive is larger than expected.")
                    output.write(chunk)
                    digest.update(chunk)
                    if progress is not None:
                        progress(downloaded, expected_size)

        if downloaded != expected_size:
            raise ValueError("Model archive size does not match manifest.")
        if digest.hexdigest().lower() != expected_sha256.lower():
            raise ValueError("Model archive checksum does not match manifest.")
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _complete_bundled_config(
    manifest: Dict[str, Any], stored: Dict[str, Any], config_file: str
) -> Dict[str, Any]:
    """Merge current defaults into a bundled model's persisted config.

    Older releases wrote a short download stub over ``config.yaml``. If the
    model files were later restored or bundled, that stub still lacked the
    model type and ONNX filenames and the loader crashed before it could use
    the files beside it. The manifest supplies stable metadata; filenames are
    recovered only when their role is unambiguous.
    """
    config = {**manifest, **stored}
    for field in PINNED_MODEL_FIELDS:
        if field in manifest:
            config[field] = manifest[field]
    config.setdefault("type", "segment_anything")
    model_dir = pathlib.Path(config_file).parent
    for key, pattern in (
        ("encoder_model_path", "*.encoder.onnx"),
        ("decoder_model_path", "*.decoder.onnx"),
    ):
        if config.get(key):
            continue
        candidates = sorted(model_dir.glob(pattern))
        if len(candidates) == 1:
            config[key] = candidates[0].name
    return config


class ModelStatus(Enum):
    READY = "Model loaded. Ready for labeling."
    NO_MODEL = "No model selected."
    LOADING = "Loading model: {model_name}. Please wait..."
    DOWNLOAD_PROGRESS = "Downloading {download_url}: {percent}%"
    ERROR_INVALID_PATH = "Error in loading custom model: Invalid path."
    ERROR_INVALID_CONFIG = "Error in loading custom model: Invalid config file."
    ERROR_INVALID_FORMAT = "Error in loading custom model: Invalid config file format."
    ERROR_INVALID_NAME = "Error in loading model: Invalid model name."
    ERROR_CONFIG_FILE = "Error in loading config file."
    ERROR_PREDICTION = "Error in model prediction. Please check the model."
    INFERENCE_WAIT = "Inferencing AI model. Please wait..."
    INFERENCE_DONE = "Finished inferencing AI model. Check the result."
    NOT_LOADED = "Model is not loaded. Choose a mode to continue."
    BUSY = "Another model is being executed. Please wait for it to finish."


class ModelManager:
    """Model manager"""

    MAX_NUM_CUSTOM_MODELS = 5

    def __init__(self):
        self.model_configs: List[Dict[str, Any]] = []
        self.loaded_model_config = None
        self.loaded_model_config_lock = Lock()
        self.status = ModelStatus.NOT_LOADED.value

        self.model_download_thread = None
        self.model_execution_thread = None
        self.model_execution_thread_lock = Lock()

        # Callbacks
        self._on_model_configs_changed: List[Callable[[List[Dict]], None]] = []
        self._on_model_status_changed: List[Callable[[str], None]] = []
        self._on_model_loaded: List[Callable[[Dict], None]] = []
        self._on_auto_labeling_result: List[Callable[[AutoLabelingResult], None]] = []
        self._on_auto_segmentation_selected: List[Callable[[], None]] = []
        self._on_auto_segmentation_unselected: List[Callable[[], None]] = []
        self._on_prediction_started: List[Callable[[], None]] = []
        self._on_prediction_finished: List[Callable[[], None]] = []
        self._on_request_next_files: List[Callable[[], None]] = []
        self._on_output_modes_changed: List[Callable[[Dict, str], None]] = []

        self.load_model_configs()

    def add_callback(self, event_name: str, callback: Callable):
        """Add callback for specific event"""
        callback_list = getattr(self, f"_on_{event_name}", None)
        if callback_list is None:
            raise ValueError(f"Unknown callback event: {event_name}")
        callback_list.append(callback)

    def notify_callbacks(self, event_name: str, *args, **kwargs):
        """Notify all callbacks for an event"""
        callback_list = getattr(self, f"_on_{event_name}", None)
        if callback_list is None:
            raise ValueError(f"Unknown callback event: {event_name}")
        if callback_list:
            for callback in callback_list:
                try:
                    callback(*args, **kwargs)
                except Exception:
                    logger.exception("Callback %s failed", event_name)
        if event_name == "model_status_changed":
            self.status = args[0]

    def load_model_configs(self):
        """Load model configs"""
        # Load list of default models
        with pkg_resources.open_text(auto_labeling_configs, "models.yaml") as f:
            model_list = yaml.safe_load(f)
            for model in model_list:
                model["is_custom_model"] = False

            # Check downloaded
            for model in model_list:
                # config.DATA_ROOT, not the home directory: the bundled
                # auto-labelling weights are seeded into
                # <data root>/models by weights.seed_auto_labeling_models(),
                # so a run with ANYLEARNING_DATA_ROOT set looked in the wrong
                # place, reported models it *had* as not downloaded, and
                # offered to fetch 160 MB from Hugging Face on a machine that
                # was carrying them all along.
                model_download_path = os.path.join(DATA_ROOT, "models", model["name"])
                pathlib.Path(model_download_path).mkdir(parents=True, exist_ok=True)
                config_file = os.path.join(model_download_path, "config.yaml")
                model["config_file"] = config_file

                # Initialize model config if needed
                if not os.path.isfile(config_file):
                    model["has_downloaded"] = False
                    with open(config_file, "w") as f:
                        yaml.dump(model, f)

        # Load list of custom models
        custom_models = get_config().get("custom_models", [])
        for custom_model in custom_models:
            custom_model["is_custom_model"] = True
            custom_model["has_downloaded"] = True

        # Remove invalid/not found custom models
        custom_models = [
            custom_model
            for custom_model in custom_models
            if os.path.isfile(custom_model.get("config_file", ""))
        ]
        config = get_config()
        config["custom_models"] = custom_models
        save_config(config)

        model_list += custom_models

        # Load model configs
        model_configs = []
        for model in model_list:
            model_config = copy.deepcopy(model)
            config_file = model.get("config_file", None)
            if config_file:
                with open(config_file, "r") as f:
                    stored_config = yaml.safe_load(f) or {}
                    model_config = (
                        {**model_config, **stored_config}
                        if model.get("is_custom_model", False)
                        else _complete_bundled_config(
                            model_config, stored_config, config_file
                        )
                    )
                    model_config["config_file"] = os.path.normpath(
                        os.path.abspath(config_file)
                    )
                    model_config["is_custom_model"] = model.get(
                        "is_custom_model", False
                    )
            model_configs.append(model_config)

        # Sort by last used
        for i, model_config in enumerate(model_configs):
            # Keep order for integrated models
            if not model_config.get("is_custom_model", False):
                model_config["last_used"] = -i
            else:
                model_config["last_used"] = model_config.get("last_used", time.time())
        model_configs.sort(key=lambda x: x.get("last_used", 0), reverse=True)

        self.model_configs = model_configs
        self.notify_callbacks("model_configs_changed", model_configs)

    def get_model_configs(self):
        """Return model infos"""
        return self.model_configs

    def set_output_mode(self, mode):
        """Set output mode"""
        if self.loaded_model_config and self.loaded_model_config["model"]:
            self.loaded_model_config["model"].set_output_mode(mode)

    def on_model_download_finished(self):
        """Handle model download thread finished"""
        if self.loaded_model_config and self.loaded_model_config["model"]:
            self.notify_callbacks("model_status_changed", ModelStatus.READY.value)
            self.notify_callbacks("model_loaded", self.loaded_model_config)
            self.notify_callbacks(
                "output_modes_changed",
                self.loaded_model_config["model"].Meta.output_modes,
                self.loaded_model_config["model"].Meta.default_output_mode,
            )
        else:
            self.notify_callbacks("model_loaded", {})

    def load_custom_model(self, config_file):
        """Run custom model loading in a thread"""
        config_file = os.path.normpath(os.path.abspath(config_file))
        if (
            self.model_download_thread is not None
            and self.model_download_thread.is_alive()
        ):
            logger.info("Another model is already loading")
            return

        # Check config file path
        if not config_file or not os.path.isfile(config_file):
            self.notify_callbacks(
                "model_status_changed", ModelStatus.ERROR_INVALID_PATH.value
            )
            return

        # Check config file content
        model_config = {}
        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)
            model_config["config_file"] = os.path.abspath(config_file)
        if not model_config:
            self.notify_callbacks(
                "model_status_changed", ModelStatus.ERROR_INVALID_CONFIG.value
            )
            return
        if (
            "type" not in model_config
            or "display_name" not in model_config
            or "name" not in model_config
            or model_config["type"] not in ["segment_anything", "yolov5", "yolov8"]
        ):
            self.notify_callbacks(
                "model_status_changed", ModelStatus.ERROR_INVALID_FORMAT.value
            )
            return

        # Add or replace custom model
        custom_models = get_config().get("custom_models", [])
        matched_index = None
        for i, model in enumerate(custom_models):
            if os.path.normpath(model["config_file"]) == os.path.normpath(config_file):
                matched_index = i
                break
        if matched_index is not None:
            model_config["last_used"] = time.time()
            custom_models[matched_index] = model_config
        else:
            if len(custom_models) >= self.MAX_NUM_CUSTOM_MODELS:
                custom_models.sort(key=lambda x: x.get("last_used", 0), reverse=True)
                removed_model = custom_models.pop()
                # Remove old model folder
                config_file = removed_model["config_file"]
                if os.path.exists(config_file):
                    try:
                        pathlib.Path(config_file).parent.rmdir()
                    except OSError:
                        pass
            custom_models = [model_config] + custom_models

        # Save config
        config = get_config()
        config["custom_models"] = custom_models
        save_config(config)

        # Reload model configs
        self.load_model_configs()

        # Load model
        self.load_model(model_config["config_file"])

    def load_model(self, config_file):
        """Run model loading in a thread"""
        # Don't reload if model is already loaded
        if (
            self.loaded_model_config is not None
            and self.loaded_model_config.get("config_file") == config_file
        ):
            return
        if (
            self.model_download_thread is not None
            and self.model_download_thread.is_alive()
        ):
            logger.info("Another model is already loading")
            return
        if not config_file:
            self.unload_model()
            self.notify_callbacks("model_status_changed", ModelStatus.NO_MODEL.value)
            return

        # Check and get model id
        model_id = None
        for i, model_config in enumerate(self.model_configs):
            if model_config["config_file"] == config_file:
                model_id = i
                break
        if model_id is None:
            self.notify_callbacks(
                "model_status_changed", ModelStatus.ERROR_INVALID_NAME.value
            )
            return

        self.model_download_thread = Thread(target=self._load_model, args=(model_id,))
        self.notify_callbacks(
            "model_status_changed",
            ModelStatus.LOADING.value.format(
                model_name=self.model_configs[model_id]["display_name"]
            ),
        )
        self.model_download_thread.start()

    def load_model_by_name(self, model_name):
        """Load model by name"""
        for model_config in self.model_configs:
            if model_config["name"] == model_name:
                self.load_model(model_config["config_file"])
                return
        raise ValueError(f"Model {model_name} not found.")

    @staticmethod
    def _safe_extract(archive: zipfile.ZipFile, destination: pathlib.Path) -> None:
        """Validate and extract a small, data-only model archive."""
        destination = destination.resolve()
        members = archive.infolist()
        if len(members) > MAX_MODEL_ARCHIVE_FILES:
            raise ValueError("Model archive contains too many files.")

        validated = []
        extracted_paths = set()
        total_size = 0
        for member in members:
            filename = member.filename
            path = pathlib.PurePosixPath(filename)
            if (
                not filename
                or "\x00" in filename
                or "\\" in filename
                or path.is_absolute()
                or ".." in path.parts
                or (path.parts and ":" in path.parts[0])
            ):
                raise ValueError(f"Unsafe path in model archive: {filename}")

            target = (destination / pathlib.Path(*path.parts)).resolve()
            try:
                target.relative_to(destination)
            except ValueError as error:
                raise ValueError(f"Unsafe path in model archive: {filename}") from error

            path_key = str(target).casefold()
            if path_key in extracted_paths:
                raise ValueError(f"Duplicate path in model archive: {filename}")
            extracted_paths.add(path_key)

            unix_mode = member.external_attr >> 16 if member.create_system == 3 else 0
            file_type = stat.S_IFMT(unix_mode)
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(
                    f"Model archive contains a link or special file: {filename}"
                )

            if member.is_dir():
                validated.append((member, target))
                continue
            if path.suffix.lower() not in ALLOWED_MODEL_FILE_SUFFIXES:
                raise ValueError(f"Unsupported file type in model archive: {filename}")
            if member.file_size > MAX_MODEL_MEMBER_BYTES:
                raise ValueError("Model archive member is too large.")
            total_size += member.file_size
            if total_size > MAX_MODEL_ARCHIVE_BYTES:
                raise ValueError("Model archive is too large after extraction.")
            if member.file_size and (
                member.compress_size == 0
                or member.file_size > member.compress_size * MAX_MODEL_COMPRESSION_RATIO
            ):
                raise ValueError("Model archive has an unsafe compression ratio.")
            validated.append((member, target))

        destination.mkdir(parents=True, exist_ok=True)
        extracted_size = 0
        for member, target in validated:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            member_size = 0
            with archive.open(member) as source, target.open("xb") as output:
                while chunk := source.read(MODEL_DOWNLOAD_CHUNK_BYTES):
                    member_size += len(chunk)
                    extracted_size += len(chunk)
                    if (
                        member_size > member.file_size
                        or member_size > MAX_MODEL_MEMBER_BYTES
                        or extracted_size > MAX_MODEL_ARCHIVE_BYTES
                    ):
                        raise ValueError("Model archive is too large after extraction.")
                    output.write(chunk)
            if member_size != member.file_size:
                raise ValueError("Model archive member size does not match metadata.")

    def _download_and_extract_model(self, model_config):
        """Download a model archive and atomically install its model folder."""
        config_file = model_config["config_file"]
        # Check if model is already downloaded
        if not os.path.exists(config_file):
            raise ValueError(ModelStatus.ERROR_CONFIG_FILE.value)
        manifest_config = model_config
        with open(config_file, "r") as f:
            stored_config = yaml.safe_load(f) or {}
        model_config = {**manifest_config, **stored_config}
        for field in PINNED_MODEL_FIELDS:
            if field in manifest_config:
                model_config[field] = manifest_config[field]
        if model_config.get("has_downloaded", False):
            return

        # Download model
        download_url = model_config.get("download_url", None)
        if not download_url:
            raise ValueError("Missing download_url in config file.")
        expected_sha256 = model_config.get("sha256")
        expected_size = model_config.get("archive_size_bytes")
        extract_dir = pathlib.Path(config_file).parent
        display_download_url = _display_download_url(download_url)
        with tempfile.TemporaryDirectory(prefix="anylearning-model-") as tmp_name:
            tmp_dir = pathlib.Path(tmp_name)
            zip_model_path = tmp_dir / "model.zip"
            logging.info("Downloading model from %s", display_download_url)

            # Download and show progress
            def _progress(downloaded, total_size):
                percent = min(100, int(downloaded * 100 / max(total_size, 1)))
                self.notify_callbacks(
                    "model_status_changed",
                    ModelStatus.DOWNLOAD_PROGRESS.value.format(
                        download_url=display_download_url, percent=percent
                    ),
                )

            try:
                _download_model_archive(
                    download_url,
                    zip_model_path,
                    expected_sha256=expected_sha256,
                    expected_size=expected_size,
                    progress=_progress,
                )
                unpacked = tmp_dir / "extract"
                with zipfile.ZipFile(zip_model_path) as archive:
                    self._safe_extract(archive, unpacked)
                configs = list(unpacked.rglob("config.yaml"))
                if len(configs) != 1:
                    raise ValueError(
                        "Model archive must contain exactly one config.yaml"
                    )

                staged = tmp_dir / "staged"
                shutil.copytree(configs[0].parent, staged)
                backup = tmp_dir / "previous"
                if extract_dir.exists():
                    shutil.move(str(extract_dir), backup)
                try:
                    shutil.move(str(staged), extract_dir)
                except Exception:
                    if backup.exists():
                        shutil.move(str(backup), extract_dir)
                    raise
            except Exception as error:
                logger.exception(
                    "Could not install model from %s", display_download_url
                )
                self.notify_callbacks(
                    "model_status_changed", f"Could not download model: {error}"
                )
                return None

        with open(config_file, encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
        model_config["has_downloaded"] = True
        model_config["config_file"] = config_file
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(model_config, f)

        return model_config

    def _load_model(self, model_id):
        """Load and return model info"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None
            self.notify_callbacks("auto_segmentation_unselected")

        model_config = copy.deepcopy(self.model_configs[model_id])

        # Download and extract model
        if not model_config.get("has_downloaded", True):
            model_config = self._download_and_extract_model(model_config)
            if model_config is None:
                return

            self.model_configs[model_id].update(model_config)

        if model_config["type"] == "segment_anything":
            from .segment_anything import SegmentAnything

            try:
                model_config["model"] = SegmentAnything(
                    model_config,
                    on_message=lambda msg: self.notify_callbacks(
                        "model_status_changed", msg
                    ),
                )
                self.notify_callbacks("auto_segmentation_selected")
            except Exception as e:  # noqa
                logger.exception("Error loading model")
                self.notify_callbacks(
                    "model_status_changed", f"Error in loading model: {str(e)}"
                )
                return

            # Request next files for prediction
            self.notify_callbacks("request_next_files")
        else:
            raise Exception(f"Unknown model type: {model_config['type']}")

        self.loaded_model_config = model_config
        self.on_model_download_finished()
        return self.loaded_model_config

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks
        (For example, for segment_anything model, it is the marks for)
        """
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"] != "segment_anything"
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_marks(marks)

    def unload_model(self):
        """Unload model"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None

    def predict_shapes(self, image, filename=None, preload_paths=None):
        """Predict shapes.
        NOTE: This function is blocking. The model can take a long time to
        predict. So it is recommended to use predict_shapes_threading instead.
        """
        if self.loaded_model_config is None:
            self.notify_callbacks("model_status_changed", ModelStatus.NOT_LOADED.value)
            self.notify_callbacks("prediction_finished")
            return
        auto_labeling_result = None
        try:
            auto_labeling_result = self.loaded_model_config["model"].predict_shapes(
                image, filename, preload_paths=preload_paths
            )
            self.notify_callbacks("auto_labeling_result", auto_labeling_result)
        except Exception as e:  # noqa
            logger.exception("Error predicting shapes")
            self.notify_callbacks(
                "model_status_changed", ModelStatus.ERROR_PREDICTION.value
            )
        self.notify_callbacks("model_status_changed", ModelStatus.INFERENCE_DONE.value)
        self.notify_callbacks("prediction_finished")
        return auto_labeling_result

    def predict_shapes_threading(self, image, filename=None):
        """Predict shapes.
        This function starts a thread to run the prediction.
        """
        if self.loaded_model_config is None:
            self.notify_callbacks("model_status_changed", ModelStatus.NOT_LOADED.value)
            return
        self.notify_callbacks("model_status_changed", ModelStatus.INFERENCE_WAIT.value)
        self.notify_callbacks("prediction_started")

        with self.model_execution_thread_lock:
            if (
                self.model_execution_thread is not None
                and self.model_execution_thread.is_alive()
            ):
                self.notify_callbacks("model_status_changed", ModelStatus.BUSY.value)
                self.notify_callbacks("prediction_finished")
                return

            self.model_execution_thread = Thread(
                target=self.predict_shapes,
                args=(image, filename),
            )
            self.model_execution_thread.start()

    def on_next_files_changed(self, next_files):
        """Run prediction on next files in advance to save inference time later"""
        if self.loaded_model_config is None:
            return

        # Currently only segment_anything model supports this feature
        if self.loaded_model_config["type"] != "segment_anything":
            return

        self.loaded_model_config["model"].on_next_files_changed(next_files)

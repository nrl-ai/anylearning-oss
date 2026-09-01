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
PINNED_MODEL_FIELDS = (
    "name",
    "display_name",
    "download_url",
    "sha256",
    "archive_size_bytes",
    "archive_members",
    "type",
    "backend",
    "tasks",
    "interaction_mode",
    "output_modes",
    "project_types",
    "inference_config",
)
ALLOWED_MODEL_FILE_SUFFIXES = {
    ".json",
    ".md",
    ".names",
    ".onnx",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_MODEL_FILE_NAMES = {"LICENSE", "NOTICE"}


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
        self.project_model_scope: int | None = None
        self.project_model_scope_generation = 0
        self.status = ModelStatus.NOT_LOADED.value

        self.model_download_thread = None
        self.model_load_lock = Lock()
        self.loading_model_name: str | None = None
        self.queued_model_request = None
        self.model_execution_thread = None
        self.model_execution_thread_lock = Lock()
        self.model_inference_lock = Lock()

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
        model_catalog = pkg_resources.files(auto_labeling_configs).joinpath(
            "models.yaml"
        )
        with model_catalog.open("r", encoding="utf-8") as f:
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

    def set_project_model_configs(
        self, project_id: int, model_configs: List[Dict[str, Any]]
    ) -> None:
        """Replace discoverable trained models with those for one project.

        These entries point at local, already-exported ONNX artifacts, so they
        never enter the download path. Names include the project and database
        model IDs and cannot collide with bundled catalog names. Only one
        project's entries may be visible at a time; this prevents model names
        and label spaces leaking between projects in the long-lived desktop
        process.
        """
        copied_configs = [copy.deepcopy(item) for item in model_configs]
        desired_names = {item.get("name") for item in copied_configs}
        with self.model_load_lock:
            current_names = {
                item.get("name")
                for item in self.model_configs
                if item.get("is_project_model")
            }
            if self.project_model_scope != project_id or current_names != desired_names:
                self.project_model_scope_generation += 1
            self.project_model_scope = project_id
            if self.queued_model_request is not None:
                queued_config, _queued_generation = self.queued_model_request
                if (
                    queued_config.get("is_project_model")
                    and queued_config.get("project_id") != project_id
                ):
                    self.queued_model_request = None
            retained = [
                item for item in self.model_configs if not item.get("is_project_model")
            ]
            self.model_configs = retained + copied_configs

        stale_loaded_model = None
        with self.model_inference_lock, self.loaded_model_config_lock:
            if (
                self.loaded_model_config
                and self.loaded_model_config.get("is_project_model")
                and (
                    self.loaded_model_config.get("project_id") != project_id
                    or self.loaded_model_config.get("name") not in desired_names
                )
            ):
                stale_loaded_model = self.loaded_model_config.get("model")
                self.loaded_model_config = None

        if stale_loaded_model is not None:
            stale_loaded_model.unload()
            self.notify_callbacks("model_status_changed", ModelStatus.NOT_LOADED.value)

        self.notify_callbacks("model_configs_changed", self.model_configs)

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
        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)
        if not isinstance(model_config, dict):
            self.notify_callbacks(
                "model_status_changed", ModelStatus.ERROR_INVALID_CONFIG.value
            )
            return
        model_config["config_file"] = os.path.abspath(config_file)
        if (
            "type" not in model_config
            or "display_name" not in model_config
            or "name" not in model_config
            or model_config["type"] not in ["segment_anything", "inference"]
        ):
            self.notify_callbacks(
                "model_status_changed", ModelStatus.ERROR_INVALID_FORMAT.value
            )
            return

        self.register_custom_model(model_config)

        # Preserve the legacy method's explicit "load" behavior. New desktop
        # imports call register_custom_model directly and load only after the
        # user confirms a model selection.
        self.load_model(model_config["config_file"])

    def register_custom_model(self, model_config):
        """Persist a validated custom config without doing expensive inference setup."""
        model_config = copy.deepcopy(model_config)
        config_file = os.path.normpath(
            os.path.abspath(str(model_config.get("config_file", "")))
        )
        if not config_file or not os.path.isfile(config_file):
            raise ValueError(ModelStatus.ERROR_INVALID_PATH.value)
        if model_config.get("type") not in {"segment_anything", "inference"}:
            raise ValueError(ModelStatus.ERROR_INVALID_FORMAT.value)
        model_config["config_file"] = config_file
        model_config["is_custom_model"] = True
        model_config["has_downloaded"] = True

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
                removed_config_file = removed_model["config_file"]
                if os.path.exists(removed_config_file):
                    try:
                        pathlib.Path(removed_config_file).parent.rmdir()
                    except OSError:
                        pass
            custom_models = [model_config] + custom_models

        # Save config
        config = get_config()
        config["custom_models"] = custom_models
        save_config(config)

        # Reload model configs
        self.load_model_configs()
        return next(
            item
            for item in self.model_configs
            if item.get("config_file") == config_file
        )

    def load_model(self, config_file):
        """Run model loading in a thread"""
        with self.model_load_lock:
            if not config_file:
                self.unload_model()
                self.notify_callbacks(
                    "model_status_changed", ModelStatus.NO_MODEL.value
                )
                return

            selected_config = next(
                (
                    copy.deepcopy(model_config)
                    for model_config in self.model_configs
                    if model_config["config_file"] == config_file
                ),
                None,
            )
            if selected_config is None:
                self.notify_callbacks(
                    "model_status_changed", ModelStatus.ERROR_INVALID_NAME.value
                )
                raise ValueError(ModelStatus.ERROR_INVALID_NAME.value)

            project_scope_generation = (
                self.project_model_scope_generation
                if selected_config.get("is_project_model")
                else None
            )
            if (
                self.model_download_thread is not None
                and self.model_download_thread.is_alive()
            ):
                if self.loading_model_name == selected_config["name"]:
                    # The latest request is the model already loading, so drop
                    # a previously queued alternative.
                    self.queued_model_request = None
                    return
                # Model construction and downloads are not safely cancellable.
                # Keep only the user's latest selection and start it as soon as
                # the current worker exits.
                self.queued_model_request = (
                    selected_config,
                    project_scope_generation,
                )
                self.notify_callbacks(
                    "model_status_changed",
                    f"Queued {selected_config['display_name']} while another model loads.",
                )
                return

            # Don't reload if the requested model is already active.
            with self.loaded_model_config_lock:
                loaded_inference_config = (
                    self.loaded_model_config.get("inference_config", {})
                    if self.loaded_model_config
                    else {}
                )
                selected_inference_config = selected_config.get("inference_config", {})
                if (
                    self.loaded_model_config is not None
                    and self.loaded_model_config.get("config_file") == config_file
                    and (
                        not selected_config.get("is_project_model")
                        or loaded_inference_config.get("sha256")
                        == selected_inference_config.get("sha256")
                    )
                ):
                    self.queued_model_request = None
                    return

            self._start_model_load_locked(selected_config, project_scope_generation)

    def load_model_by_name(self, model_name):
        """Load model by name"""
        for model_config in self.model_configs:
            if model_config["name"] == model_name:
                self.load_model(model_config["config_file"])
                return
        raise ValueError(f"Model {model_name} not found.")

    def _start_model_load_locked(
        self, model_config, project_scope_generation=None
    ) -> None:
        """Start one worker while ``model_load_lock`` is held."""
        self.loading_model_name = model_config["name"]
        self.model_download_thread = Thread(
            target=self._load_model_thread,
            args=(model_config, project_scope_generation),
        )
        self.notify_callbacks(
            "model_status_changed",
            ModelStatus.LOADING.value.format(model_name=model_config["display_name"]),
        )
        self.model_download_thread.start()

    def _load_model_thread(self, model_config, project_scope_generation=None):
        try:
            return self._load_model(model_config, project_scope_generation)
        except Exception as error:
            logger.exception("Unexpected error loading auto-labeling model")
            self.notify_callbacks(
                "model_status_changed", f"Error in loading model: {error}"
            )
            return None
        finally:
            with self.model_load_lock:
                if self.loading_model_name == model_config.get("name"):
                    self.loading_model_name = None
                queued_request = self.queued_model_request
                self.queued_model_request = None
                if queued_request is not None:
                    self._start_model_load_locked(*queued_request)

    @staticmethod
    def _safe_extract(
        archive: zipfile.ZipFile,
        destination: pathlib.Path,
        *,
        expected_members: Dict[str, Dict[str, Any]] | None = None,
    ) -> None:
        """Validate and extract a bounded data-only model archive.

        New catalog entries declare every file that may be installed, including
        its uncompressed size and SHA-256. Undeclared files are inspected for
        unsafe archive metadata but are not extracted. This permits provenance
        archives that contain exporter source while guaranteeing the desktop
        never installs or executes that source.
        """
        destination = destination.resolve()
        members = archive.infolist()
        if len(members) > MAX_MODEL_ARCHIVE_FILES:
            raise ValueError("Model archive contains too many files.")

        expected: Dict[str, Dict[str, Any]] | None = None
        if expected_members is not None:
            if not isinstance(expected_members, dict) or not expected_members:
                raise ValueError("Model archive member manifest must be non-empty.")
            expected = {}
            for filename, metadata in expected_members.items():
                if not isinstance(filename, str) or not isinstance(metadata, dict):
                    raise ValueError("Model archive member manifest is invalid.")
                digest = metadata.get("sha256")
                size = metadata.get("size_bytes")
                if not isinstance(digest, str) or not re.fullmatch(
                    r"[0-9a-fA-F]{64}", digest
                ):
                    raise ValueError("Model archive member requires a SHA-256.")
                if (
                    isinstance(size, bool)
                    or not isinstance(size, int)
                    or not 0 <= size <= MAX_MODEL_MEMBER_BYTES
                ):
                    raise ValueError("Model archive member requires a bounded size.")
                expected[filename] = metadata

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
                if expected is None:
                    validated.append((member, target, None))
                continue
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

            metadata = expected.get(filename) if expected is not None else None
            if expected is not None and metadata is None:
                continue
            if (
                path.suffix.lower() not in ALLOWED_MODEL_FILE_SUFFIXES
                and path.name not in ALLOWED_MODEL_FILE_NAMES
            ):
                raise ValueError(f"Unsupported file type in model archive: {filename}")
            if metadata is not None and member.file_size != metadata["size_bytes"]:
                raise ValueError(
                    f"Model archive member size does not match manifest: {filename}"
                )
            validated.append((member, target, metadata))

        if expected is not None:
            archived_names = {
                member.filename for member in members if not member.is_dir()
            }
            missing = sorted(set(expected) - archived_names)
            if missing:
                raise ValueError(
                    "Model archive is missing declared member(s): " + ", ".join(missing)
                )

        destination.mkdir(parents=True, exist_ok=True)
        extracted_size = 0
        for member, target, metadata in validated:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            member_size = 0
            digest = hashlib.sha256()
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
                    digest.update(chunk)
            if member_size != member.file_size:
                raise ValueError("Model archive member size does not match metadata.")
            if (
                metadata is not None
                and digest.hexdigest().lower() != str(metadata["sha256"]).lower()
            ):
                raise ValueError(
                    f"Model archive member checksum does not match manifest: "
                    f"{member.filename}"
                )

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
                    self._safe_extract(
                        archive,
                        unpacked,
                        expected_members=manifest_config.get("archive_members"),
                    )
                staged = tmp_dir / "staged"
                if manifest_config.get("archive_members"):
                    shutil.copytree(unpacked, staged)
                    installed_config = copy.deepcopy(manifest_config)
                    installed_config.pop("config_file", None)
                    installed_config["has_downloaded"] = True
                    with (staged / "config.yaml").open("x", encoding="utf-8") as stream:
                        yaml.safe_dump(installed_config, stream, sort_keys=False)
                else:
                    configs = list(unpacked.rglob("config.yaml"))
                    if len(configs) != 1:
                        raise ValueError(
                            "Model archive must contain exactly one config.yaml"
                        )
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
            stored_config = yaml.safe_load(f) or {}
        # Legacy SAM bundles include their own config. Its internal release
        # name can differ from the public catalog name selected by the picker;
        # allowing that identity to replace the manifest makes a successful
        # load look like the wrong model and every immediate inference returns
        # 409. Preserve graph filenames from the bundle while pinning the
        # catalog identity and behavior just as load_model_configs does.
        model_config = _complete_bundled_config(
            manifest_config, stored_config, config_file
        )
        model_config["has_downloaded"] = True
        model_config["config_file"] = config_file
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(model_config, f)

        return model_config

    def _load_model(self, model_config, project_scope_generation=None):
        """Load and return model info"""
        # Download and extract model
        if not model_config.get("has_downloaded", True):
            model_config = self._download_and_extract_model(model_config)
            if model_config is None:
                return
            for current in self.model_configs:
                if current.get("name") == model_config.get("name") and current.get(
                    "config_file"
                ) == model_config.get("config_file"):
                    current.update(model_config)
                    break

        if model_config["type"] == "segment_anything":
            from .segment_anything import SegmentAnything

            try:
                model_config["model"] = SegmentAnything(
                    model_config,
                    on_message=lambda msg: self.notify_callbacks(
                        "model_status_changed", msg
                    ),
                )
            except Exception as e:  # noqa
                logger.exception("Error loading model")
                self.notify_callbacks(
                    "model_status_changed", f"Error in loading model: {str(e)}"
                )
                return

        elif model_config["type"] == "inference":
            from .inference_model import InferenceModel

            try:
                model_config["model"] = InferenceModel(
                    model_config,
                    on_message=lambda msg: self.notify_callbacks(
                        "model_status_changed", msg
                    ),
                )
            except Exception as e:
                logger.exception("Error loading inference model")
                self.notify_callbacks(
                    "model_status_changed", f"Error in loading model: {str(e)}"
                )
                return
        else:
            self.notify_callbacks(
                "model_status_changed",
                f"Error in loading model: unknown type {model_config['type']}",
            )
            return

        previous_model_config = None
        with self.model_inference_lock:
            with self.loaded_model_config_lock:
                stale_project_load = bool(
                    model_config.get("is_project_model")
                    and (
                        project_scope_generation != self.project_model_scope_generation
                        or model_config.get("project_id") != self.project_model_scope
                    )
                )
                if not stale_project_load:
                    previous_model_config = self.loaded_model_config
                    self.loaded_model_config = model_config

            if previous_model_config is not None:
                previous_model_config["model"].unload()

        if stale_project_load:
            model_config["model"].unload()
            self.notify_callbacks("model_status_changed", ModelStatus.NOT_LOADED.value)
            return None

        if previous_model_config is not None:
            self.notify_callbacks("auto_segmentation_unselected")
        if (
            model_config["type"] == "segment_anything"
            or model_config.get("interaction_mode") == "prompted"
        ):
            self.notify_callbacks("auto_segmentation_selected")
        self.notify_callbacks("request_next_files")
        self.on_model_download_finished()
        return self.loaded_model_config

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks
        (For example, for segment_anything model, it is the marks for)
        """
        if self.loaded_model_config is None:
            return
        setter = getattr(
            self.loaded_model_config["model"], "set_auto_labeling_marks", None
        )
        if callable(setter):
            setter(marks)

    def is_model_ready(self, model_name: str) -> bool:
        with self.loaded_model_config_lock:
            return bool(
                self.loaded_model_config
                and self.loaded_model_config.get("name") == model_name
                and self.loaded_model_config.get("model") is not None
            )

    @property
    def loaded_model_name(self) -> str | None:
        with self.loaded_model_config_lock:
            if self.loaded_model_config is None:
                return None
            value = self.loaded_model_config.get("name")
        return value if isinstance(value, str) else None

    def unload_model(self):
        """Unload model"""
        with self.model_inference_lock:
            with self.loaded_model_config_lock:
                model_config = self.loaded_model_config
                self.loaded_model_config = None
            if model_config is not None:
                model_config["model"].unload()

    def predict_shapes(
        self,
        image,
        filename=None,
        preload_paths=None,
        *,
        allowed_labels=None,
        parameters=None,
        marks=None,
        output_mode=None,
    ):
        """Predict shapes.
        NOTE: This function is blocking. The model can take a long time to
        predict. So it is recommended to use predict_shapes_threading instead.
        """
        auto_labeling_result = None
        try:
            with self.model_inference_lock:
                if self.loaded_model_config is None:
                    raise ValueError(ModelStatus.NOT_LOADED.value)
                model_config = self.loaded_model_config
                model = model_config["model"]
                if output_mode is not None:
                    model.set_output_mode(output_mode)
                if marks is not None:
                    setter = getattr(model, "set_auto_labeling_marks", None)
                    if callable(setter):
                        setter(marks)
                if model_config["type"] == "inference":
                    auto_labeling_result = model.predict_shapes(
                        image,
                        filename,
                        preload_paths=preload_paths,
                        allowed_labels=allowed_labels,
                        parameters=parameters,
                    )
                else:
                    auto_labeling_result = model.predict_shapes(
                        image, filename, preload_paths=preload_paths
                    )
            self.notify_callbacks("auto_labeling_result", auto_labeling_result)
        except Exception as error:
            logger.exception("Error predicting shapes")
            self.notify_callbacks(
                "model_status_changed",
                (
                    ModelStatus.NOT_LOADED.value
                    if isinstance(error, ValueError)
                    and str(error) == ModelStatus.NOT_LOADED.value
                    else ModelStatus.ERROR_PREDICTION.value
                ),
            )
            self.notify_callbacks("prediction_finished")
            raise
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

        self.loaded_model_config["model"].on_next_files_changed(next_files)

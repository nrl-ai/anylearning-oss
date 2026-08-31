import hashlib
import io
import re
import stat
import time
import zipfile
from importlib import resources
from threading import Event, Thread
from unittest.mock import Mock, patch

import pytest
import yaml

from anylearning.auto_labeling.model_manager import (
    MAX_MODEL_DOWNLOAD_BYTES,
    ModelManager,
    _complete_bundled_config,
    _download_model_archive,
)
from anylearning.configs import auto_labeling as auto_labeling_configs


class FakeResponse(io.BytesIO):
    def __init__(self, content: bytes, url: str = "https://cdn.example/model.zip"):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content))}
        self.url = url

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def make_archive(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)
    return path


def test_bundled_model_downloads_are_pinned_and_checksum_verified():
    model_file = resources.files(auto_labeling_configs).joinpath("models.yaml")
    models = yaml.safe_load(model_file.read_text())

    assert len(models) == 13
    for model in models:
        assert re.match(
            r"^https://huggingface\.co/nrl-ai/anylearning-labeling-models/resolve/"
            r"[0-9a-f]{40}/[^/]+\.zip$",
            model["download_url"],
        )
        assert len(model["sha256"]) == 64
        int(model["sha256"], 16)
        assert 0 < model["archive_size_bytes"] <= MAX_MODEL_DOWNLOAD_BYTES

    assert MAX_MODEL_DOWNLOAD_BYTES >= 10 * 1024**3
    assert {model["name"] for model in models}.issuperset(
        {
            "efficientvit_sam_l0",
            "dfine_n_coco",
            "rfdetr_nano_detection",
            "rfdetr_nano_segmentation",
            "sam2_1_hiera_small",
        }
    )
    for model in (item for item in models if item["type"] == "inference"):
        assert model["archive_members"]
        assert model["inference_config"]


def test_project_models_are_isolated_to_the_active_project():
    with patch.object(ModelManager, "load_model_configs"):
        manager = ModelManager()

    bundled = {"name": "bundled", "is_project_model": False}
    first = {
        "name": "project-1-trained-1",
        "project_id": 1,
        "is_project_model": True,
    }
    second = {
        "name": "project-2-trained-2",
        "project_id": 2,
        "is_project_model": True,
    }
    manager.model_configs = [bundled]
    manager.set_project_model_configs(1, [first])
    assert [item["name"] for item in manager.model_configs] == [
        "bundled",
        "project-1-trained-1",
    ]

    loaded = Mock()
    manager.loaded_model_config = {**first, "model": loaded}
    manager.set_project_model_configs(2, [second])

    assert [item["name"] for item in manager.model_configs] == [
        "bundled",
        "project-2-trained-2",
    ]
    assert manager.loaded_model_config is None
    loaded.unload.assert_called_once_with()


def test_project_model_load_is_discarded_after_project_switch():
    with patch.object(ModelManager, "load_model_configs"):
        manager = ModelManager()

    manager.set_project_model_configs(1, [])
    old_generation = manager.project_model_scope_generation
    loaded = Mock()
    stale_config = {
        "name": "project-1-trained-1",
        "display_name": "Old project model",
        "project_id": 1,
        "is_project_model": True,
        "type": "inference",
        "has_downloaded": True,
    }
    manager.set_project_model_configs(2, [])

    with patch(
        "anylearning.auto_labeling.inference_model.InferenceModel",
        return_value=loaded,
    ):
        result = manager._load_model(stale_config, old_generation)

    assert result is None
    assert manager.loaded_model_config is None
    loaded.unload.assert_called_once_with()


def test_prediction_keeps_each_requests_prompts_isolated():
    with patch.object(ModelManager, "load_model_configs"):
        manager = ModelManager()

    first_entered = Event()
    release_first = Event()

    class RecordingModel:
        def __init__(self):
            self.marks = []
            self.seen = []

        def set_auto_labeling_marks(self, marks):
            self.marks = list(marks)

        def predict_shapes(self, _image, _filename, *, preload_paths):
            del preload_paths
            if self.marks == [{"request": "first"}]:
                first_entered.set()
                release_first.wait(timeout=1)
            self.seen.append(list(self.marks))
            return []

    model = RecordingModel()
    manager.loaded_model_config = {
        "name": "promptable",
        "type": "segment_anything",
        "model": model,
    }
    errors = []

    def run(marks):
        try:
            manager.predict_shapes("image", marks=marks)
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    first = Thread(target=run, args=([{"request": "first"}],))
    second = Thread(target=run, args=([{"request": "second"}],))
    first.start()
    assert first_entered.wait(timeout=1)
    second.start()
    time.sleep(0.05)
    release_first.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert model.seen == [
        [{"request": "first"}],
        [{"request": "second"}],
    ]


def test_model_selection_queues_only_the_latest_request_while_loading():
    with patch.object(ModelManager, "load_model_configs"):
        manager = ModelManager()

    configs = [
        {
            "name": name,
            "display_name": name.upper(),
            "config_file": f"/{name}/config.yaml",
        }
        for name in ("first", "second", "third")
    ]
    manager.model_configs = configs
    manager.model_download_thread = Mock()
    manager.model_download_thread.is_alive.return_value = True
    manager.loading_model_name = "first"

    manager.load_model(configs[1]["config_file"])
    assert manager.queued_model_request[0]["name"] == "second"
    manager.load_model(configs[2]["config_file"])
    assert manager.queued_model_request[0]["name"] == "third"

    # Choosing the model already loading cancels the queued alternative.
    manager.load_model(configs[0]["config_file"])
    assert manager.queued_model_request is None


def test_persisted_config_cannot_override_pinned_download_metadata(tmp_path):
    manifest = {
        "download_url": "https://models.example/release/model.zip",
        "sha256": "a" * 64,
        "archive_size_bytes": 123,
    }
    stored = {
        "download_url": "https://attacker.example/model.zip",
        "sha256": "b" * 64,
        "archive_size_bytes": 456,
    }

    config = _complete_bundled_config(
        manifest,
        stored,
        str(tmp_path / "config.yaml"),
    )

    for field in ("download_url", "sha256", "archive_size_bytes"):
        assert config[field] == manifest[field]


def test_download_rejects_non_https_urls(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        _download_model_archive(
            "file:///private/model.zip",
            tmp_path / "model.zip",
            expected_sha256="0" * 64,
            expected_size=1,
        )


def test_download_rejects_redirect_to_non_https(tmp_path):
    response = FakeResponse(b"archive", url="file:///private/model.zip")
    with patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(ValueError, match="redirected"):
            _download_model_archive(
                "https://example.test/model.zip",
                tmp_path / "model.zip",
                expected_sha256=hashlib.sha256(b"archive").hexdigest(),
                expected_size=7,
            )


def test_download_rejects_wrong_size_or_checksum(tmp_path):
    content = b"archive"
    with patch("urllib.request.urlopen", return_value=FakeResponse(content)):
        with pytest.raises(ValueError, match="size"):
            _download_model_archive(
                "https://example.test/model.zip",
                tmp_path / "model.zip",
                expected_sha256=hashlib.sha256(content).hexdigest(),
                expected_size=len(content) + 1,
            )

    with patch("urllib.request.urlopen", return_value=FakeResponse(content)):
        with pytest.raises(ValueError, match="checksum"):
            _download_model_archive(
                "https://example.test/model.zip",
                tmp_path / "model.zip",
                expected_sha256="0" * 64,
                expected_size=len(content),
            )


def test_download_writes_verified_content_and_reports_progress(tmp_path):
    content = b"verified archive"
    progress = []
    destination = tmp_path / "model.zip"
    with patch("urllib.request.urlopen", return_value=FakeResponse(content)):
        _download_model_archive(
            "https://example.test/model.zip",
            destination,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            expected_size=len(content),
            progress=lambda downloaded, total: progress.append((downloaded, total)),
        )

    assert destination.read_bytes() == content
    assert progress[-1] == (len(content), len(content))


@pytest.mark.parametrize(
    "member",
    ["../escape.onnx", "/absolute.onnx", "folder\\escape.onnx", "C:/escape.onnx"],
)
def test_safe_extract_rejects_unsafe_paths(tmp_path, member):
    archive_path = make_archive(tmp_path / "model.zip", [(member, b"model")])

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="Unsafe path"):
            ModelManager._safe_extract(archive, tmp_path / "extract")


def test_safe_extract_rejects_links(tmp_path):
    archive_path = tmp_path / "model.zip"
    link = zipfile.ZipInfo("model.onnx")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "target.onnx")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="link or special file"):
            ModelManager._safe_extract(archive, tmp_path / "extract")


def test_safe_extract_rejects_unexpected_file_types_and_duplicates(tmp_path):
    script_archive = make_archive(
        tmp_path / "script.zip", [("install.py", b"print('unexpected')")]
    )
    with zipfile.ZipFile(script_archive) as archive:
        with pytest.raises(ValueError, match="Unsupported file type"):
            ModelManager._safe_extract(archive, tmp_path / "extract-script")

    duplicate_archive = tmp_path / "duplicate.zip"
    make_archive(
        duplicate_archive,
        [("model.onnx", b"first"), ("MODEL.ONNX", b"second")],
    )
    with zipfile.ZipFile(duplicate_archive) as archive:
        with pytest.raises(ValueError, match="Duplicate path"):
            ModelManager._safe_extract(archive, tmp_path / "extract-duplicate")


def test_safe_extract_enforces_member_and_size_limits(tmp_path):
    many_archive = make_archive(
        tmp_path / "many.zip",
        [(f"model-{index}.onnx", b"x") for index in range(257)],
    )
    with zipfile.ZipFile(many_archive) as archive:
        with pytest.raises(ValueError, match="too many files"):
            ModelManager._safe_extract(archive, tmp_path / "extract-many")

    large_archive = make_archive(tmp_path / "large.zip", [("model.onnx", b"x" * 1024)])
    with zipfile.ZipFile(large_archive) as archive:
        with patch(
            "anylearning.auto_labeling.model_manager.MAX_MODEL_ARCHIVE_BYTES", 512
        ):
            with pytest.raises(ValueError, match="too large"):
                ModelManager._safe_extract(archive, tmp_path / "extract-large")


def test_safe_extract_accepts_expected_model_files(tmp_path):
    archive_path = make_archive(
        tmp_path / "model.zip",
        [
            ("config.yaml", b"type: segment_anything\n"),
            ("encoder.onnx", b"encoder"),
            ("decoder.onnx", b"decoder"),
        ],
    )
    destination = tmp_path / "extract"

    with zipfile.ZipFile(archive_path) as archive:
        ModelManager._safe_extract(archive, destination)

    assert (destination / "config.yaml").read_text() == "type: segment_anything\n"
    assert (destination / "encoder.onnx").read_bytes() == b"encoder"
    assert (destination / "decoder.onnx").read_bytes() == b"decoder"


def test_safe_extract_installs_only_declared_checksum_verified_members(tmp_path):
    model = b"verified onnx"
    archive_path = make_archive(
        tmp_path / "model.zip",
        [
            ("model.onnx", model),
            ("export_checked.py", b"raise RuntimeError('must not be installed')"),
        ],
    )
    destination = tmp_path / "extract"
    with zipfile.ZipFile(archive_path) as archive:
        ModelManager._safe_extract(
            archive,
            destination,
            expected_members={
                "model.onnx": {
                    "sha256": hashlib.sha256(model).hexdigest(),
                    "size_bytes": len(model),
                }
            },
        )

    assert (destination / "model.onnx").read_bytes() == model
    assert not (destination / "export_checked.py").exists()


def test_safe_extract_rejects_declared_member_checksum_mismatch(tmp_path):
    archive_path = make_archive(tmp_path / "model.zip", [("model.onnx", b"model")])
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="checksum"):
            ModelManager._safe_extract(
                archive,
                tmp_path / "extract",
                expected_members={"model.onnx": {"sha256": "0" * 64, "size_bytes": 5}},
            )

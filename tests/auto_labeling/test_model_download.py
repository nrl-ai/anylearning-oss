import hashlib
import io
import stat
import zipfile
from importlib import resources
from unittest.mock import patch

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

    assert len(models) == 5
    for model in models:
        assert model["download_url"].startswith(
            "https://huggingface.co/nrl-ai/anylearning-labeling-models/resolve/v1.0.0/"
        )
        assert len(model["sha256"]) == 64
        int(model["sha256"], 16)
        assert 0 < model["archive_size_bytes"] <= MAX_MODEL_DOWNLOAD_BYTES

    assert MAX_MODEL_DOWNLOAD_BYTES >= 10 * 1024**3


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

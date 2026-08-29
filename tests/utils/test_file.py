"""Tests for cross-platform file helpers."""

import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from anylearning.utils.file import download_file, extract_frontend_dist, open_file

# --------------------------------------------------------------------------
# open_file
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system,expected",
    [("Darwin", "open"), ("Linux", "xdg-open")],
)
def test_open_file_uses_the_platform_opener(system, expected):
    with patch("platform.system", return_value=system):
        with patch("subprocess.Popen") as popen:
            open_file("/tmp/thing.txt")
    assert popen.call_args[0][0][0] == expected


def test_open_file_uses_startfile_on_windows():
    with patch("platform.system", return_value="Windows"):
        # os.startfile only exists on Windows, so it has to be injected.
        with patch.object(os, "startfile", create=True) as startfile:
            open_file("C:/thing.txt")
    startfile.assert_called_once_with("C:/thing.txt")


# --------------------------------------------------------------------------
# extract_frontend_dist
# --------------------------------------------------------------------------


def test_extract_frontend_dist_copies_the_bundled_ui(tmp_path):
    source = tmp_path / "bundled"
    source.mkdir()
    (source / "index.html").write_text("<h1>app</h1>")
    static = tmp_path / "static"

    with patch("anylearning.utils.file.resource_path", return_value=str(source)):
        extract_frontend_dist(str(static))

    assert (static / "index.html").read_text() == "<h1>app</h1>"


def test_extract_frontend_dist_replaces_a_stale_copy(tmp_path):
    """A previous build's assets must not survive into the new one."""
    source = tmp_path / "bundled"
    source.mkdir()
    (source / "new.html").write_text("new")

    static = tmp_path / "static"
    static.mkdir()
    (static / "stale.html").write_text("stale")

    with patch("anylearning.utils.file.resource_path", return_value=str(source)):
        extract_frontend_dist(str(static))

    assert (static / "new.html").exists()
    assert not (static / "stale.html").exists()


def test_extract_frontend_dist_writes_a_hint_when_the_ui_is_missing(tmp_path):
    """Running from source without building the frontend should explain itself."""
    static = tmp_path / "static"

    with patch(
        "anylearning.utils.file.resource_path", return_value=str(tmp_path / "absent")
    ):
        extract_frontend_dist(str(static))

    body = (static / "index.html").read_text()
    assert "build_frontend.sh" in body


# --------------------------------------------------------------------------
# download_file
# --------------------------------------------------------------------------


def _response(status=200, chunks=(b"abc", b"def")):
    response = MagicMock()
    response.status_code = status
    response.headers = {"content-length": str(sum(len(c) for c in chunks))}
    response.iter_content.return_value = iter(chunks)
    return response


def test_download_file_writes_the_body(tmp_path):
    target = tmp_path / "nested" / "model.bin"
    with patch("requests.get", return_value=_response()):
        download_file("https://example.invalid/model.bin", str(target))

    assert target.read_bytes() == b"abcdef"


def test_download_file_creates_missing_parent_directories(tmp_path):
    target = tmp_path / "a" / "b" / "c" / "model.bin"
    with patch("requests.get", return_value=_response()):
        download_file("https://example.invalid/model.bin", str(target))
    assert target.exists()


def test_download_file_leaves_no_file_on_http_error(tmp_path):
    """A 404 must not leave a truncated file that later looks like a model."""
    target = tmp_path / "model.bin"
    with patch("requests.get", return_value=_response(status=404, chunks=())):
        download_file("https://example.invalid/missing.bin", str(target))

    assert not target.exists()


def test_download_file_streams_rather_than_buffering(tmp_path):
    """stream=True matters: model files are hundreds of megabytes."""
    target = tmp_path / "model.bin"
    with patch("requests.get", return_value=_response()) as get:
        download_file("https://example.invalid/model.bin", str(target))

    assert get.call_args.kwargs.get("stream") is True


def test_download_file_does_not_leak_the_temporary_file(tmp_path):
    """The temp file is moved into place, so nothing should be left behind."""
    target = tmp_path / "model.bin"
    before = set(pathlib.Path(os.environ.get("TMPDIR", "/tmp")).glob("tmp*"))

    with patch("requests.get", return_value=_response()):
        download_file("https://example.invalid/model.bin", str(target))

    after = set(pathlib.Path(os.environ.get("TMPDIR", "/tmp")).glob("tmp*"))
    assert not (after - before), "temporary download file was left behind"

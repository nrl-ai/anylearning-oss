"""Extracting the frontend from a read-only installation, twice.

The app shipped a version that started exactly once from an installation the
user does not own. `shutil.copytree` copies permission bits, so a read-only
`/opt` tree or `.app` bundle produced 0555 directories and 0444 files *in the
user's own data root*; `rmtree(..., ignore_errors=True)` then could not delete
inside them and said nothing, and the next launch died on the `FileExistsError`
from copytree. No server, no window, and no way for a user to guess that
chmod-ing a cache directory is the cure.
"""

import os
import stat

import pytest

from anylearning.utils import file as file_utils


@pytest.fixture
def installation(tmp_path, monkeypatch):
    """A shipped frontend-dist, read-only like an installed copy."""
    source = tmp_path / "install" / "anylearning" / "frontend-dist"
    (source / "_next" / "static").mkdir(parents=True)
    (source / "index.html").write_text("<html></html>")
    (source / "_next" / "static" / "app.js").write_text("console.log(1)")

    for path in [source, *source.rglob("*")]:
        path.chmod(0o555 if path.is_dir() else 0o444)

    monkeypatch.setattr(file_utils, "resource_path", lambda *parts: str(source))
    return source


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_it_extracts_from_a_read_only_installation(installation, tmp_path):
    static = tmp_path / "data" / "frontend-dist"

    file_utils.extract_frontend_dist(str(static))

    assert (static / "index.html").is_file()
    assert (static / "_next" / "static" / "app.js").is_file()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_a_second_launch_replaces_what_the_first_extracted(installation, tmp_path):
    """The bug: launch one worked, launch two exited 1."""
    static = tmp_path / "data" / "frontend-dist"

    file_utils.extract_frontend_dist(str(static))
    file_utils.extract_frontend_dist(str(static))  # would raise FileExistsError

    assert (static / "index.html").is_file()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_what_it_leaves_behind_is_writable(installation, tmp_path):
    """Because the *next* launch has to be able to delete it."""
    static = tmp_path / "data" / "frontend-dist"

    file_utils.extract_frontend_dist(str(static))

    for path in [static, *static.rglob("*")]:
        assert os.stat(path).st_mode & stat.S_IWUSR, f"not writable: {path}"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_an_undeletable_target_says_so_rather_than_raising_file_exists(
    installation, tmp_path, monkeypatch
):
    """If the refresh really cannot happen, the error names the cause."""
    static = tmp_path / "data" / "frontend-dist"
    file_utils.extract_frontend_dist(str(static))

    # A tree that survives rmtree, as a read-only parent directory produces.
    monkeypatch.setattr(file_utils.shutil, "rmtree", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="Could not replace"):
        file_utils.extract_frontend_dist(str(static))


def test_a_build_without_a_frontend_still_says_what_to_do(tmp_path, monkeypatch):
    missing = tmp_path / "install" / "absent"
    monkeypatch.setattr(file_utils, "resource_path", lambda *parts: str(missing))
    static = tmp_path / "data" / "frontend-dist"

    file_utils.extract_frontend_dist(str(static))

    assert "build_frontend.sh" in (static / "index.html").read_text()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_seeded_auto_labelling_models_are_writable(tmp_path, monkeypatch):
    """The other place the app copies out of its own installation.

    `seed_auto_labeling_models` copies bundled auto-labelling weights into
    `<data root>/models`, and copytree copies permission bits. Seeded out of a
    read-only installation the user could neither delete a model to reclaim
    space nor replace one by downloading it again -- in a directory that is
    theirs. The comment in that function already says the data root is the
    user's; this makes it true.
    """
    from anylearning import weights

    bundle = tmp_path / "install" / "weights"
    model = bundle / "auto_labeling" / "mobile_sam"
    model.mkdir(parents=True)
    (model / "config.yaml").write_text("name: mobile_sam\n")
    (model / "encoder.onnx").write_bytes(b"weights")
    for path in [bundle, *bundle.rglob("*")]:
        path.chmod(0o555 if path.is_dir() else 0o444)

    data_root = tmp_path / "data"
    monkeypatch.setattr("anylearning.config.DATA_ROOT", str(data_root))
    monkeypatch.setattr(weights, "bundled_dir", lambda: bundle)

    seeded = weights.seed_auto_labeling_models()

    assert seeded == ["mobile_sam"]
    destination = data_root / "models" / "mobile_sam"
    for path in [destination, *destination.rglob("*")]:
        assert os.stat(path).st_mode & stat.S_IWUSR, f"not writable: {path}"

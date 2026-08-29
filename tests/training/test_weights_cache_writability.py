"""The weights caches, when the application is installed where the user cannot write.

detectron2 reads its checkpoints through iopath, which takes a portalocker lock
on `<weight>.pkl.lock` *beside the file* before opening it. A weights directory
inside the installation is read-only for a standard user, so instance
segmentation could not train from an installed copy at all unless the whole
application was run as administrator -- found on Windows, where
`C:\\Program Files` is read-only, and equally true of any installation the user
does not own.

Only fvcore is affected. torch.hub and huggingface_hub read without locking.
"""

import os
import pathlib
import stat

import pytest

from anylearning import weights


@pytest.fixture
def bundle(tmp_path):
    """A weights directory shaped like the shipped one."""
    root = tmp_path / "install" / "weights"
    (root / "fvcore" / "detectron2" / "mask_rcnn").mkdir(parents=True)
    (root / "fvcore" / "detectron2" / "mask_rcnn" / "model_final.pkl").write_bytes(
        b"checkpoint"
    )
    # Something that is not a checkpoint, so the mirror always has a file to
    # link even in the "checkpoints missing" case.
    (root / "fvcore" / "detectron2" / "mask_rcnn" / "notes.txt").write_bytes(b"x")
    # iopath leaves these behind when fetch_weights.py runs the real loaders,
    # so they ship inside the installation exactly like this: zero bytes, and
    # read-only once the installation is.
    (root / "fvcore" / "detectron2" / "mask_rcnn" / "model_final.pkl.lock").touch()
    (root / "huggingface" / "hub").mkdir(parents=True)
    return root


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    for variable in weights.CACHE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr("anylearning.config.DATA_ROOT", str(tmp_path / "data"))


def make_read_only(directory: pathlib.Path):
    directory.chmod(stat.S_IRUSR | stat.S_IXUSR)


def strip_checkpoints(fvcore: pathlib.Path):
    """A cache that still has to download, which is the only case that mirrors."""
    for checkpoint in fvcore.rglob("*.pkl"):
        checkpoint.unlink()


def test_a_writable_install_is_used_as_it_is(bundle, clean_environment):
    weights.use_bundled(bundle)
    assert os.environ["FVCORE_CACHE"] == str(bundle / "fvcore")


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to anything")
def test_a_read_only_install_with_its_checkpoints_needs_no_mirror(
    bundle, clean_environment
):
    """The mirror is a fallback, not the normal path.

    The trainers hand detectron2 a resolved local path rather than a URL, so
    nothing writes into this directory when the checkpoints are present: no
    download, and no lock, because a local path goes through
    `NativePathHandler`. Redirecting anyway put 426 MB of copies into a Windows
    user's data root, since `os.link` from Program Files is denied to a
    standard user and the fallback copies.
    """
    fvcore = bundle / "fvcore"
    make_read_only(fvcore)
    try:
        weights.use_bundled(bundle)

        from anylearning import config

        assert os.environ["FVCORE_CACHE"] == str(fvcore)
        assert not (pathlib.Path(config.DATA_ROOT) / "weights-cache").exists()
    finally:
        fvcore.chmod(stat.S_IRWXU)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to anything")
def test_a_read_only_install_missing_its_checkpoints_gets_a_writable_cache(
    bundle, clean_environment
):
    """The case the mirror is still for: a build that has to download."""
    fvcore = bundle / "fvcore"
    strip_checkpoints(fvcore)
    make_read_only(fvcore)
    try:
        weights.use_bundled(bundle)
        cache = pathlib.Path(os.environ["FVCORE_CACHE"])

        assert cache != fvcore
        assert weights._is_writable(cache)
    finally:
        fvcore.chmod(stat.S_IRWXU)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to anything")
def test_the_mirror_does_not_duplicate_the_bytes(bundle, clean_environment):
    """Hardlinked where the filesystem allows it: these checkpoints are 350-500 MB."""
    fvcore = bundle / "fvcore"
    strip_checkpoints(fvcore)
    original = fvcore / "detectron2" / "mask_rcnn" / "notes.txt"
    make_read_only(fvcore)
    try:
        weights.use_bundled(bundle)
        mirrored = (
            pathlib.Path(os.environ["FVCORE_CACHE"])
            / "detectron2"
            / "mask_rcnn"
            / "notes.txt"
        )
        assert mirrored.stat().st_ino == original.stat().st_ino
    finally:
        fvcore.chmod(stat.S_IRWXU)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to anything")
def test_the_other_caches_keep_pointing_at_the_bundle(bundle, clean_environment):
    """Only fvcore ever needed this. Mirroring the rest would copy for nothing."""
    make_read_only(bundle / "fvcore")
    try:
        weights.use_bundled(bundle)
        assert os.environ["TORCH_HOME"] == str(bundle)
        assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(bundle / "huggingface/hub")
    finally:
        (bundle / "fvcore").chmod(stat.S_IRWXU)


def test_an_explicit_setting_still_wins(bundle, clean_environment, monkeypatch, tmp_path):
    """fetch_weights.py points these at a directory it is filling."""
    chosen = tmp_path / "chosen"
    monkeypatch.setenv("FVCORE_CACHE", str(chosen))
    weights.use_bundled(bundle)
    assert os.environ["FVCORE_CACHE"] == str(chosen)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to anything")
def test_the_mirror_leaves_the_lock_files_behind(bundle, clean_environment):
    """The bug the first version of this shipped.

    A hardlink shares an inode and mode travels with the inode, so linking a
    read-only `.pkl.lock` out of a read-only installation produced a read-only
    lock in the mirror -- and iopath failed to open it for writing exactly as
    before, one directory further along. Measured on macOS: redirect fine,
    checkpoints hardlinked fine, run still dead with EACCES.

    Not mirroring them is the fix: iopath creates the lock it wants, and the
    mirror directory is writable, which was the only thing ever missing.
    """
    fvcore = bundle / "fvcore"
    # Only a cache that still has to download gets mirrored, so this is a
    # partially-fetched tree: the lock survived, the checkpoint did not. The
    # lock is the file that must not be hardlinked out of a read-only tree.
    strip_checkpoints(fvcore)
    make_read_only(fvcore)
    try:
        weights.use_bundled(bundle)
        cache = pathlib.Path(os.environ["FVCORE_CACHE"])
        mirrored = cache / "detectron2" / "mask_rcnn"

        assert not (mirrored / "model_final.pkl.lock").exists()
        # And the directory must accept the lock iopath is about to create.
        probe = mirrored / "model_final.pkl.lock"
        probe.touch()
        assert os.access(probe, os.W_OK)
    finally:
        fvcore.chmod(stat.S_IRWXU)


def test_the_harness_does_not_hand_its_own_cache_paths_to_the_binary():
    """Why the read-only bug survived every harness we had.

    Importing anything from `anylearning` runs `use_bundled()`, which points the
    cache variables at the *checkout's* weights. `subprocess` then inherits them,
    and `use_bundled()` inside the packaged binary skips any variable that
    already has a value -- so the installed app read the developer's writable
    directory, the mirror was never created, and instance segmentation appeared
    to train fine from a read-only installation when it could not.
    """
    from anylearning.selftest.driver import (
        INHERITED_CACHE_VARIABLES,
        scrub_cache_variables,
    )

    env = {name: "/somewhere/in/the/checkout" for name in INHERITED_CACHE_VARIABLES}
    env["PATH"] = "/usr/bin"

    dropped = scrub_cache_variables(env)

    assert sorted(dropped) == sorted(INHERITED_CACHE_VARIABLES)
    assert list(env) == ["PATH"]
    # Every variable weights.py sets must be covered, or the gap reopens
    # silently the next time one is added.
    assert set(INHERITED_CACHE_VARIABLES) == set(weights.CACHE_VARIABLES)

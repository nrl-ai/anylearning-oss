"""Mask R-CNN's starting weights: a local path, not a URL.

Handing detectron2 a URL is what made it lock. iopath routes a URL through its
*caching* handler, which takes a portalocker lock on `<file>.lock` beside the
cached copy before it will read it -- and that lock is the entire reason an
installed copy needed a writable mirror of the weights directory. It cost a
release: instance segmentation could not train from `C:\\Program Files` or from
a read-only `/Applications` bundle unless the whole app ran elevated.

We ship these checkpoints, so the download never happens and the lock protects
nothing. A plain local path goes through `NativePathHandler`, which never locks.
"""

import pathlib

import pytest

from anylearning.training.models.instance_segmentation.maskrcnn.train import (
    bundled_checkpoint,
)

URL = (
    "https://dl.fbaipublicfiles.com/detectron2/new_baselines/"
    "mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl"
)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FVCORE_CACHE", str(tmp_path))
    return tmp_path


def test_a_bundled_checkpoint_resolves_to_its_local_path(cache):
    target = cache / "detectron2/new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571"
    target.mkdir(parents=True)
    (target / "model_final_14d201.pkl").write_bytes(b"weights")

    resolved = bundled_checkpoint(URL)

    assert resolved == str(target / "model_final_14d201.pkl")
    assert not resolved.startswith("http")


def test_a_missing_checkpoint_keeps_the_url(cache):
    """A build without bundled weights must still be able to download."""
    assert bundled_checkpoint(URL) == URL


def test_no_cache_configured_keeps_the_url(monkeypatch):
    monkeypatch.delenv("FVCORE_CACHE", raising=False)
    assert bundled_checkpoint(URL) == URL


def test_a_directory_in_the_way_keeps_the_url(cache):
    """is_file(), not exists(): a directory of that name is not a checkpoint."""
    (cache / "detectron2/new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl").mkdir(
        parents=True
    )
    assert bundled_checkpoint(URL) == URL


def test_the_resolved_path_needs_no_lock_file(cache):
    """The point of the whole exercise.

    iopath only locks paths it is caching. Nothing here should require a
    writable directory beside the checkpoint, which is what a read-only
    installation cannot provide.
    """
    target = cache / "detectron2/new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571"
    target.mkdir(parents=True)
    (target / "model_final_14d201.pkl").write_bytes(b"weights")

    resolved = pathlib.Path(bundled_checkpoint(URL))

    assert resolved.is_file()
    assert not resolved.with_suffix(".pkl.lock").exists()

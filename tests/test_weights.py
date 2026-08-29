"""The weights that ship with the app, so a machine with no network can train.

The app is sold on working offline and did not: the first training run of each
architecture downloaded its backbone, and on a machine with no internet that
failed inside the training subprocess, after the dataset had been prepared,
where the user saw only a run that stopped.
"""

import os
import pathlib

import pytest

from anylearning import weights


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A weights directory in the shape fetch_weights.py produces."""
    root = tmp_path / "weights"
    (root / "hub" / "checkpoints").mkdir(parents=True)
    (root / "hub" / "checkpoints" / "resnet18-f37072fd.pth").write_bytes(b"x" * 10)
    monkeypatch.setattr(weights, "_CANDIDATES", (root,))
    for variable in weights.CACHE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return root


def test_the_caches_point_at_the_bundle(bundle):
    weights.use_bundled()

    assert os.environ["TORCH_HOME"] == str(bundle)
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(bundle / "huggingface" / "hub")
    assert os.environ["FVCORE_CACHE"] == str(bundle / "fvcore")
    # RF-DETR's default is ~/.roboflow/models, a hidden folder in the user's
    # home directory rather than anywhere the installer put something.
    assert os.environ["RF_HOME"] == str(bundle / "rfdetr")
    # Without this, a machine with no network waits for a revision check to
    # time out before using files it already has.
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_a_deliberate_setting_is_not_overwritten(bundle, monkeypatch, tmp_path):
    """Someone who set TORCH_HOME means it more than this default does -- and
    fetch_weights.py relies on it to aim the same machinery at a directory it
    is in the middle of filling."""
    elsewhere = str(tmp_path / "mine")
    monkeypatch.setenv("TORCH_HOME", elsewhere)

    weights.use_bundled()

    assert os.environ["TORCH_HOME"] == elsewhere


def test_a_build_without_weights_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(weights, "_CANDIDATES", (tmp_path / "absent",))
    assert weights.use_bundled() is None
    assert weights.describe() == {
        "bundled": False,
        "path": None,
        "files": 0,
        "bytes": 0,
    }


def test_auto_labeling_models_are_installed_once(bundle, tmp_path, monkeypatch):
    from anylearning import config

    model = bundle / "auto_labeling" / "mobile_sam_20230629"
    model.mkdir(parents=True)
    (model / "config.yaml").write_text("has_downloaded: true\n")
    (model / "encoder.onnx").write_bytes(b"weights")

    data_root = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_ROOT", str(data_root))

    assert weights.seed_auto_labeling_models() == ["mobile_sam_20230629"]
    installed = data_root / "models" / "mobile_sam_20230629"
    assert (installed / "encoder.onnx").read_bytes() == b"weights"

    # Second run leaves it alone: a user who replaced a model keeps theirs.
    (installed / "encoder.onnx").write_bytes(b"mine")
    assert weights.seed_auto_labeling_models() == []
    assert (installed / "encoder.onnx").read_bytes() == b"mine"


def test_describe_counts_what_is_there(bundle):
    described = weights.describe()
    assert described["bundled"] is True
    assert described["files"] == 1
    assert described["bytes"] == 10
    assert pathlib.Path(described["path"]) == bundle

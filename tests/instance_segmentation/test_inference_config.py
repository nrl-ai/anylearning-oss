"""An instance segmentation model has to remain usable after its run is gone.

Training pickles detectron2's config into the run's output folder, and that
folder is deleted when the run finishes. Inference read it from there, so every
attempt to try an instance segmentation model answered 500 with

    No such file or directory: .../training/<n>/training_output/detectron2_cfg.pkl

which is the whole feature, broken from the user's first click.
"""

import pickle

import pytest

from anylearning.training.models.instance_segmentation.factory import (
    InstanceSegmentationModelFactory,
)

CONFIG = {
    "detectron2_cfg_file": "detectron2_cfg.pkl",
    "save_dir": "/gone/training/10/training_output",
    "model": {"arch": "maskrcnn", "backbone": "resnet50", "num_classes": 3},
    "data": {"img_size": 512, "label_set": [{"name": "particle", "id": 1}]},
}


def test_the_config_is_read_from_beside_the_model(tmp_path):
    """What a model registered by 0.26.1 onwards carries."""
    model_path = tmp_path / "best_model.pth"
    model_path.touch()
    marker = {"marker": "the real pickled config"}
    (tmp_path / "detectron2_cfg.pkl").write_bytes(pickle.dumps(marker))

    loaded = InstanceSegmentationModelFactory.load_inference_config(
        CONFIG, str(model_path)
    )
    assert loaded == marker, "the exact config beside the model must win"


def test_the_run_folder_is_used_while_it_exists(tmp_path):
    """A run in progress, or anyone using --development, still has one."""
    model_path = tmp_path / "models" / "best_model.pth"
    model_path.parent.mkdir()
    model_path.touch()
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    marker = {"marker": "from the run folder"}
    (run_folder / "detectron2_cfg.pkl").write_bytes(pickle.dumps(marker))

    config = dict(CONFIG, save_dir=str(run_folder))
    assert (
        InstanceSegmentationModelFactory.load_inference_config(config, str(model_path))
        == marker
    )


def test_a_model_with_no_config_anywhere_is_rebuilt(tmp_path):
    """Every model trained by 0.26.0 is in this position, and cannot get its
    file back -- so the config is rebuilt from what the run stored instead of
    telling the user to train again."""
    model_path = tmp_path / "best_model.pth"
    model_path.touch()

    rebuilt = InstanceSegmentationModelFactory.load_inference_config(
        CONFIG, str(model_path)
    )

    # The fields inference actually reads, and nothing about a dataset that is
    # no longer registered.
    assert rebuilt.MODEL.ROI_HEADS.NUM_CLASSES == 3
    assert rebuilt.INPUT.MAX_SIZE_TEST == 512
    assert len(rebuilt.DATASETS.TEST) == 0
    assert rebuilt.MODEL.DEVICE in {"cpu", "cuda"}
    # Same thresholds as a config built during training, so a model behaves the
    # same way whichever path produced it.
    assert rebuilt.MODEL.ROI_HEADS.SCORE_THRESH_TEST == 0.2
    assert rebuilt.TEST.DETECTIONS_PER_IMAGE == 1000


def test_an_unknown_backbone_says_so(tmp_path):
    model_path = tmp_path / "best_model.pth"
    model_path.touch()
    config = dict(
        CONFIG, model={"arch": "maskrcnn", "backbone": "resnet9000", "num_classes": 1}
    )
    with pytest.raises(ValueError, match="resnet9000"):
        InstanceSegmentationModelFactory.load_inference_config(config, str(model_path))

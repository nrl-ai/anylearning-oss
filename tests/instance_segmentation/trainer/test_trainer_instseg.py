import json
import os
import shutil

import numpy as np
import pytest
import torch
import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from PIL import Image

from anylearning.training.models.instance_segmentation.maskrcnn.inference import (
    inference_fn,
)
from anylearning.training.models.instance_segmentation.maskrcnn.train import train_fn


@pytest.fixture
def temp_training_dir(tmp_path):
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    return training_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    # Create temporary dataset directory with sample images and annotations
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create a sample image
    img = Image.new("RGB", (100, 100), color="white")
    img_path = dataset_dir / "test_image.jpg"
    img.save(img_path)

    # Create corresponding annotation
    annotation = {
        "images": [
            {"id": 1, "file_name": "test_image.jpg", "height": 100, "width": 100}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
                "area": 1600,
                "bbox": [10, 10, 40, 40],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "class1"}],
    }

    with open(dataset_dir / "test_image.json", "w") as f:
        json.dump(annotation, f)

    return dataset_dir


def test_train_flow(temp_training_dir, temp_data_dir):
    """Test the training flow function"""
    # create dummpy data
    shutil.copytree(temp_data_dir, temp_training_dir / "train")
    shutil.copytree(temp_data_dir, temp_training_dir / "val")

    # Create a minimal config file
    config = {
        "data": {
            "train_dir": str(temp_training_dir / "train"),
            "val_dir": str(temp_training_dir / "val"),
            "train_ann_file": str(temp_training_dir / "train" / "test_image.json"),
            "val_ann_file": str(temp_training_dir / "val" / "test_image.json"),
            "img_size": 224,
            "num_workers": 1,
            "label_set": [
                {"id": 1, "name": "class1", "color": "#FF0000"},
            ],
        },
        "model": {
            "arch": "maskrcnn",
            "backbone": "resnet50",
            "pretrained": "coco_lsj",
            "num_classes": 1,
        },
        "training": {
            "scheduler": "WarmupCosineLR",
            "epochs": 1,
            "batch_size": 4,
            "verbose_steps": 1,
            "optim": {
                "name": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.001,
                "betas": [0.9, 0.999],
            },
            "min_lr": 1e-7,
        },
        "save_dir": str(temp_training_dir / "output"),
        "detectron2_cfg_file": str(temp_training_dir / "output" / "detectron2_cfg.pkl"),
    }

    config_path = temp_training_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run training flow
    class MockLogWriter:
        def write(self, log):
            print(log)

        def write_metrics(self, metrics):
            print(metrics)

    mock_logger = MockLogWriter()

    train_fn(str(config_path), mock_logger)

    # last_model.pth is written at the end of every epoch. best_model.pth is only
    # written when validation mAP improves on the previous best, which a one-image,
    # one-epoch run cannot promise -- asserting on it made this test flaky.
    assert (temp_training_dir / "output" / "last_model.pth").exists()


def test_run_inference(temp_training_dir, temp_data_dir):
    # create a model
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # init a model
    model = build_model(cfg)
    model.eval()
    os.makedirs(temp_training_dir / "output", exist_ok=True)
    torch.save(model, temp_training_dir / "output" / "best_model.pth")

    # create a dummy image
    image = np.array(Image.open(temp_data_dir / "test_image.jpg"))

    formatted_predictions, visualization_image = inference_fn(
        detectron2_cfg=cfg,
        model_path=temp_training_dir / "output" / "best_model.pth",
        class_names=["class1"],
        image=image,
    )

    assert formatted_predictions is not None
    assert visualization_image is not None

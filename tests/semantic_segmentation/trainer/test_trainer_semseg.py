import json
import shutil

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from anylearning.training.models.semantic_segmentation import train_fn
from anylearning.training.trainers.semseg_trainer import SemSegTrainer


@pytest.fixture
def temp_training_dir(tmp_path):
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    return training_dir


@pytest.fixture
def temp_dataset_dir(tmp_path):
    # Create temporary dataset directory with sample images and annotations
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create a sample image
    img = Image.new("RGB", (100, 100), color="white")
    img_path = dataset_dir / "test_image.jpg"
    img.save(img_path)

    # Create a sample image
    img = Image.new("RGB", (100, 100), color="white")
    img_path = dataset_dir / "test_image_2.jpg"
    img.save(img_path)

    # Create corresponding annotation
    annotation = [
        {"categories": ["class1"], "points": [[10, 10], [50, 10], [50, 50], [10, 50]]}
    ]

    with open(dataset_dir / "test_image.json", "w") as f:
        json.dump(annotation, f)

    with open(dataset_dir / "test_image_2.json", "w") as f:
        json.dump(annotation, f)

    return dataset_dir


def test_train_flow(temp_training_dir, temp_dataset_dir):
    """Test the training flow function"""
    # create dummpy data
    shutil.copytree(temp_dataset_dir, temp_training_dir / "train")
    shutil.copytree(temp_dataset_dir, temp_training_dir / "val")

    # Create a minimal config file
    config = {
        "data": {
            "train_dir": str(temp_training_dir / "train"),
            "val_dir": str(temp_training_dir / "val"),
            "img_size": 224,
            "num_workers": 1,
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "label_set": [
                {"id": 1, "name": "class1", "color": "#FF0000"},
                {"id": 2, "name": "class2", "color": "#00FF00"},
            ],
        },
        "model": {"arch": "resnet18", "num_classes": 2, "pretrained": None},
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "fp16": False,
            "optim": {
                "name": "Adam",
                "lr": 0.001,
                "weight_decay": 0.0001,
                "betas": [0.9, 0.999],
            },
            "min_lr": 1e-6,
        },
        "save_dir": str(temp_training_dir / "output"),
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

    # Assert that model files were created
    assert (temp_training_dir / "output" / "best_model.pth").exists()
    assert (temp_training_dir / "output" / "last_model.pth").exists()


def test_hex_to_rgb():
    assert SemSegTrainer.hex_to_rgb("#FF0000") == (255, 0, 0)
    assert SemSegTrainer.hex_to_rgb("#00FF00") == (0, 255, 0)
    assert SemSegTrainer.hex_to_rgb("#0000FF") == (0, 0, 255)


def test_run_inference(temp_training_dir):
    # Create sample config
    config = {
        "data": {
            "img_size": 224,
            "normalize": {"mean": [0.485], "std": [0.229]},
            "label_set": [
                {"id": 1, "name": "class1", "color": "#FF0000"},
                {"id": 2, "name": "class2", "color": "#00FF00"},
            ],
        }
    }
    config_str = yaml.dump(config)

    # Create dummy model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 1), torch.nn.ReLU(), torch.nn.Conv2d(3, 3, 1)
    )
    model_path = temp_training_dir / "model.pth"
    torch.save(model, model_path)

    # Create test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    predictions, visualization = SemSegTrainer.run_inference(
        config_str, str(model_path), test_image
    )

    assert isinstance(predictions, list)
    assert isinstance(visualization, np.ndarray)
    assert visualization.shape == test_image.shape

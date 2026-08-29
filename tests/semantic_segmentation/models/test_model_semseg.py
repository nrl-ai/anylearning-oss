import torch

import pytest
from anylearning.training.models.semantic_segmentation.train import (
    AverageMeter,
    calculate_iou,
    get_transformations,
    load_or_create_model,
)


@pytest.fixture
def sample_config():
    return {
        "model": {
            "arch": "resnet18",
            "num_classes": 2,
        },
        "data": {
            "img_size": 224,
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        "training": {"fp16": False},
    }


def test_average_meter():
    meter = AverageMeter()

    # Test initial state
    assert meter.sum == 0
    assert meter.count == 0
    assert meter.avg == 0

    # Test single update
    meter.update(5.0)
    assert meter.sum == 5.0
    assert meter.count == 1
    assert meter.avg == 5.0

    # Test multiple updates
    meter.update(3.0, n=2)
    assert meter.sum == 11.0
    assert meter.count == 3
    assert abs(meter.avg - 3.67) < 0.01

    # Test reset
    meter.reset()
    assert meter.sum == 0
    assert meter.count == 0
    assert meter.avg == 0


def test_get_transformations(sample_config):
    transform = get_transformations(
        sample_config["data"]["img_size"],
        sample_config["data"]["normalize"]["mean"],
        sample_config["data"]["normalize"]["std"],
        is_train=True,
    )
    assert transform is not None

    # Test transform composition
    assert len(transform.transforms) > 0

    # Test different transforms for train/val
    train_transform = get_transformations(224, [0.5], [0.5], is_train=True)
    val_transform = get_transformations(224, [0.5], [0.5], is_train=False)
    assert len(train_transform.transforms) > len(val_transform.transforms)


def test_load_or_create_model(sample_config):
    model = load_or_create_model(sample_config)
    assert model is not None

    # Test output shape
    dummy_input = torch.randn(
        2, 3, 224, 224
    )  # batch size == 1 will raise BatchNorm error
    output = model(dummy_input)
    expected_classes = sample_config["model"]["num_classes"] + 1  # +1 for background
    assert output.shape == (2, expected_classes, 224, 224)


def test_calculate_iou():
    # Create sample predictions and ground truth
    pred = torch.tensor([[0, 1, 1], [0, 1, 0]])
    true = torch.tensor([[0, 1, 1], [0, 1, 0]])

    # Test perfect prediction
    iou = calculate_iou(pred, true, num_classes=2, include_background=True)
    assert iou == 1.0

    # Test with some mismatch
    pred = torch.tensor([[0, 1, 1], [0, 0, 0]])
    true = torch.tensor([[0, 1, 1], [0, 1, 0]])
    iou = calculate_iou(pred, true, num_classes=2, include_background=True)
    assert iou < 1.0

    # Test excluding background
    iou_no_bg = calculate_iou(pred, true, num_classes=2, include_background=False)
    assert iou_no_bg != iou


def test_model_with_pretrained_weights(sample_config, tmp_path):
    # Create a temporary model file
    dummy_model = load_or_create_model(sample_config)
    model_path = tmp_path / "model.pth"
    torch.save(dummy_model, model_path)

    # Update config to use pretrained weights
    sample_config["model"]["resume_from"] = str(model_path)

    # Load model with pretrained weights
    model = load_or_create_model(sample_config)
    assert model is not None

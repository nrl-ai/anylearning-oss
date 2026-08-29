import json

import pytest
import torch
from PIL import Image
from torchvision import transforms

from anylearning.training.models.semantic_segmentation.dataset import (
    SegmentationDataset,
    is_an_image,
)


@pytest.fixture
def temp_dataset_dir(tmp_path):
    # Create temporary dataset directory with sample images and annotations
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create a sample image
    img = Image.new("RGB", (100, 100), color="white")
    img_path = dataset_dir / "test_image.jpg"
    img.save(img_path)

    # Create corresponding annotation
    annotation = [
        {"categories": ["class1"], "points": [[10, 10], [50, 10], [50, 50], [10, 50]]}
    ]

    with open(dataset_dir / "test_image.json", "w") as f:
        json.dump(annotation, f)

    return dataset_dir


@pytest.fixture
def temp_dataset_dir_invalid_annotation(tmp_path):
    # Create temporary dataset directory with sample images and annotations
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create a sample image
    img = Image.new("RGB", (100, 100), color="white")
    img_path = dataset_dir / "test_image.jpg"
    img.save(img_path)

    # Create invalid annotation
    invalid_annotation = [
        {
            "categories": ["class1"],
        }
    ]

    with open(dataset_dir / "test_image.json", "w") as f:
        json.dump(invalid_annotation, f)

    return dataset_dir


@pytest.fixture
def class_name2id():
    return {"class1": 1, "class2": 2}


def test_is_an_image():
    assert is_an_image("test.jpg")
    assert is_an_image("test.png")
    assert is_an_image("test.jpeg")
    assert is_an_image("test.bmp")
    assert not is_an_image("test.txt")
    assert not is_an_image("test.json")


def test_dataset_initialization(temp_dataset_dir, class_name2id):
    dataset = SegmentationDataset(str(temp_dataset_dir), class_name2id)
    assert len(dataset) == 1
    assert dataset.images[0] == "test_image.jpg"


def test_dataset_getitem(temp_dataset_dir, class_name2id):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = SegmentationDataset(
        str(temp_dataset_dir), class_name2id, transform=transform
    )
    image, mask = dataset[0]

    # Check image
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)

    # Check mask
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (224, 224)
    assert mask.dtype == torch.long

    # Verify mask values
    assert torch.any(mask == 0)  # background
    assert torch.any(mask == 1)  # class1


def test_dataset_with_missing_annotation(temp_dataset_dir, class_name2id):
    # Create image without annotation
    img = Image.new("RGB", (100, 100), color="white")
    img_path = temp_dataset_dir / "no_annotation.jpg"
    img.save(img_path)

    dataset = SegmentationDataset(str(temp_dataset_dir), class_name2id)
    # Look the sample up by name rather than assuming a position: hardcoding
    # dataset[1] silently depended on directory listing order.
    idx = dataset.images.index("no_annotation.jpg")
    image, mask = dataset[idx]

    # Mask should be all zeros (background)
    assert torch.all(mask == 0)


def test_dataset_with_invalid_annotation(
    temp_dataset_dir_invalid_annotation, class_name2id
):
    dataset = SegmentationDataset(
        str(temp_dataset_dir_invalid_annotation), class_name2id
    )

    dataset = SegmentationDataset(
        str(temp_dataset_dir_invalid_annotation), class_name2id
    )
    image, mask = dataset[0]

    print(str(temp_dataset_dir))
    print("mask != 0", torch.where(mask != 0))

    # Should handle invalid annotation gracefully
    assert torch.all(mask == 0)

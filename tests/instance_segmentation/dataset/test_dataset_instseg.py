import json
import os

import pytest
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from PIL import Image


@pytest.fixture
def temp_instance_dataset_dir(tmp_path):
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

    os.makedirs(dataset_dir / "images", exist_ok=True)
    with open(dataset_dir / "test_image.json", "w") as f:
        json.dump(annotation, f)

    return dataset_dir


def test_instance_dataset_initialization(temp_instance_dataset_dir):
    # create and register detectron2 coco dataset,
    # rename train_ds to train_ds1 since "train_ds" will be registered by other tests
    register_coco_instances(
        "train_ds1",
        {},
        temp_instance_dataset_dir / "test_image.json",
        temp_instance_dataset_dir,
    )

    metadata = MetadataCatalog.get("train_ds1")
    train_ds = DatasetCatalog.get("train_ds1")

    assert metadata is not None
    assert train_ds[0]["image_id"] == 1
    assert "test_image.jpg" in train_ds[0]["file_name"]
    assert (
        train_ds[0]["annotations"][0]["category_id"] == 1 - 1
    )  # detectron auto converts category_id to 0-based
    assert train_ds[0]["annotations"][0]["segmentation"] == [
        [10, 10, 50, 10, 50, 50, 10, 50]
    ]

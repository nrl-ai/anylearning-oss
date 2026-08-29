import importlib.util
import json
import pathlib
import zipfile

import pytest
from PIL import Image


SCRIPT = (
    pathlib.Path(__file__).parents[1]
    / "examples/keypoint_detection/generate_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("keypoint_example", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(subject)


def test_example_generates_import_ready_train_and_valid_archives(tmp_path):
    outputs = subject.generate(tmp_path, train_images=3, valid_images=2)

    assert outputs == [
        tmp_path / "stick-figures-train.zip",
        tmp_path / "stick-figures-valid.zip",
    ]
    for expected_count, output in zip((3, 2), outputs, strict=True):
        with zipfile.ZipFile(output) as archive:
            coco = json.loads(archive.read("annotations.coco.json"))
            assert "README.txt" in archive.namelist()
            assert len(coco["images"]) == expected_count
            assert len(coco["annotations"]) == expected_count * 2
            assert coco["categories"][0]["keypoints"] == list(subject.KEYPOINT_NAMES)
            first = coco["images"][0]
            with archive.open(f"images/{first['file_name']}") as source:
                assert Image.open(source).size == (subject.WIDTH, subject.HEIGHT)
            for annotation in coco["annotations"]:
                assert len(annotation["keypoints"]) == len(subject.KEYPOINT_NAMES) * 3
                assert annotation["num_keypoints"] == len(subject.KEYPOINT_NAMES)
                assert annotation["area"] > 0


def test_example_rejects_empty_splits(tmp_path):
    with pytest.raises(ValueError, match="must both be positive"):
        subject.generate(tmp_path, train_images=0, valid_images=1)

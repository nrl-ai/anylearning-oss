import hashlib
import io
import json
import zipfile

import pytest
from PIL import Image

import prepare_vertebral_keypoints as subject


def _jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (20, 10), "white").save(output, "JPEG")
    return output.getvalue()


def _source_archive(path):
    dataset = io.BytesIO()
    with zipfile.ZipFile(dataset, "w") as archive:
        for source_split in ("train", "val"):
            stem = f"case_{source_split}"
            root = "Dataset/Train/Keypointrcnn_data"
            archive.writestr(f"{root}/images/{source_split}/{stem}.jpg", _jpeg())
            if source_split == "train":
                archive.writestr(
                    f"{root}/labels/{source_split}/{stem}.json",
                    json.dumps(
                        {
                            "boxes": [[1, 2, 11, 8]],
                            "keypoints": [
                                [[1, 8, 1], [11, 8, 1], [1, 2, 1], [11, 2, 0]]
                            ],
                            "labels": [0],
                        }
                    ),
                )
            else:
                archive.writestr(
                    f"{root}/labels/{source_split}/{stem}.txt",
                    "0 0.3 0.5 0.5 0.6 0.05 0.2 2 0.55 0.2 0 0.05 0.8 2 0.55 0.8 2\n",
                )
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr(
            "Spondylolisthesis Vertebral Landmark/Dataset.zip", dataset.getvalue()
        )


def test_prepare_converts_nested_archive_to_importable_coco(tmp_path):
    source = tmp_path / "source.zip"
    _source_archive(source)

    result = subject.prepare(source, tmp_path / "output")

    assert result == {"train": (1, 1), "valid": (1, 1)}
    with zipfile.ZipFile(tmp_path / "output/vertebral-keypoints-train.zip") as archive:
        assert "ATTRIBUTION.md" in archive.namelist()
        assert "images/case_train.jpg" in archive.namelist()
        coco = json.loads(archive.read("annotations.coco.json"))

    assert coco["images"] == [
        {
            "id": 1,
            "file_name": "case_train.jpg",
            "width": 20,
            "height": 10,
            "license": 1,
        }
    ]
    assert coco["categories"][0]["keypoints"] == list(subject.KEYPOINT_NAMES)
    annotation = coco["annotations"][0]
    assert annotation["bbox"] == [1.0, 2.0, 10.0, 6.0]
    assert annotation["area"] == 60.0
    assert annotation["num_keypoints"] == 3
    assert annotation["keypoints"] == [
        1.0,
        2.0,
        2,
        11.0,
        2.0,
        0,
        1.0,
        8.0,
        2,
        11.0,
        8.0,
        2,
    ]

    with zipfile.ZipFile(tmp_path / "output/vertebral-keypoints-valid.zip") as archive:
        valid = json.loads(archive.read("annotations.coco.json"))
    assert valid["annotations"][0]["bbox"] == pytest.approx([1.0, 2.0, 10.0, 6.0])
    assert valid["annotations"][0]["keypoints"] == pytest.approx(
        annotation["keypoints"]
    )


def test_rejects_mismatched_box_and_keypoint_counts():
    payload = {"boxes": [[1, 2, 3, 4]], "keypoints": []}

    try:
        subject._annotation(payload, image_id=1, first_annotation_id=1)
    except ValueError as error:
        assert "1 boxes and 0 keypoint sets" in str(error)
    else:
        raise AssertionError("invalid source annotation was accepted")


def test_download_identifies_itself_and_verifies_checksum(tmp_path, monkeypatch):
    content = b"fixed version archive"
    seen = {}

    def urlopen(request):
        seen["user_agent"] = request.get_header("User-agent")
        seen["accept"] = request.get_header("Accept")
        return io.BytesIO(content)

    monkeypatch.setattr(subject.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(subject, "EXPECTED_SHA256", hashlib.sha256(content).hexdigest())
    destination = tmp_path / "dataset.zip"

    subject.download(destination)

    assert destination.read_bytes() == content
    assert seen == {
        "user_agent": "AnyLearning-Dataset-Preparation/0.26",
        "accept": "application/zip, application/octet-stream",
    }

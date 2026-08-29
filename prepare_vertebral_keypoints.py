#!/usr/bin/env python3
"""Prepare a CC BY 4.0 real-world keypoint dataset for AnyLearning.

The source archive is version 1 of "Spondylolisthesis Vertebral Landmark":
https://doi.org/10.17632/5jdfdgp762.1

Mendeley wraps the actual Dataset.zip in an outer download archive. This tool
accepts either archive, converts its mixture of per-image Keypoint R-CNN JSON
and YOLO-pose text files to standard COCO keypoints, and writes separate train
and validation ZIPs ready for AnyLearning's dataset uploader.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import shutil
import tempfile
import urllib.request
import zipfile

from PIL import Image

DATASET_DOI = "10.17632/5jdfdgp762.1"
DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/5jdfdgp762/download/1"
EXPECTED_SHA256 = "3a15ceb105c93716035df049b8ee8407ddee479184860fc885e71f18886859d1"

KEYPOINT_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
SKELETON = ((1, 2), (2, 4), (4, 3), (3, 1))
SPLITS = {
    "train": (
        "Dataset/Train/Keypointrcnn_data/images/train/",
        "Dataset/Train/Keypointrcnn_data/labels/train/",
    ),
    "valid": (
        "Dataset/Train/Keypointrcnn_data/images/val/",
        "Dataset/Train/Keypointrcnn_data/labels/val/",
    ),
}

ATTRIBUTION = """# Dataset attribution

This archive is a format-converted subset of **Spondylolisthesis Vertebral
Landmark**, version 1, by Karla Reyes.

- DOI: https://doi.org/10.17632/5jdfdgp762.1
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License text: https://creativecommons.org/licenses/by/4.0/

Changes: the source's mixture of per-image Keypoint R-CNN JSON and YOLO-pose
annotations was converted to COCO keypoints; their inconsistent landmark
orders were normalized geometrically to top-left, top-right, bottom-left,
bottom-right; binary visibility value 1 was mapped to COCO visible value 2;
the published train/validation split was retained. No image pixels were
modified.
"""


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name, suffix=".part", delete=False
    ) as temporary:
        temporary_path = pathlib.Path(temporary.name)
        try:
            request = urllib.request.Request(
                DOWNLOAD_URL,
                headers={
                    # Mendeley's Cloudflare layer rejects Python-urllib's
                    # default user agent with HTTP 403. Identify the tool
                    # honestly rather than pretending to be a browser.
                    "User-Agent": "AnyLearning-Dataset-Preparation/0.26",
                    "Accept": "application/zip, application/octet-stream",
                },
            )
            with urllib.request.urlopen(request) as response:  # noqa: S310
                shutil.copyfileobj(response, temporary)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    actual = sha256(temporary_path)
    if actual != EXPECTED_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded archive has SHA-256 {actual}, expected {EXPECTED_SHA256}."
        )
    temporary_path.replace(destination)


def _dataset_zip(path: pathlib.Path) -> tuple[zipfile.ZipFile, io.BytesIO | None]:
    outer = zipfile.ZipFile(path)
    if any(name.startswith("Dataset/") for name in outer.namelist()):
        return outer, None

    nested = [name for name in outer.namelist() if name.endswith("/Dataset.zip")]
    if len(nested) != 1:
        outer.close()
        raise ValueError(
            "Archive contains neither Dataset/ nor one nested Dataset.zip."
        )
    buffer = io.BytesIO(outer.read(nested[0]))
    outer.close()
    return zipfile.ZipFile(buffer), buffer


def _canonical_points(points: list) -> list:
    """Return four quadrilateral corners in one order, regardless of source format."""
    if len(points) != len(KEYPOINT_NAMES) or any(len(point) != 3 for point in points):
        raise ValueError("Expected four landmark triplets.")
    by_height = sorted(points, key=lambda point: (float(point[1]), float(point[0])))
    top = sorted(by_height[:2], key=lambda point: float(point[0]))
    bottom = sorted(by_height[2:], key=lambda point: float(point[0]))
    return top + bottom


def _annotation(payload: dict, image_id: int, first_annotation_id: int) -> list[dict]:
    boxes = payload.get("boxes") or []
    instances = payload.get("keypoints") or []
    if len(boxes) != len(instances):
        raise ValueError(
            f"Image {image_id} has {len(boxes)} boxes and {len(instances)} keypoint sets."
        )

    records = []
    for offset, (box, points) in enumerate(zip(boxes, instances, strict=True)):
        if len(box) != 4:
            raise ValueError(f"Image {image_id} has an invalid box or landmark count.")
        left, top, right, bottom = (float(value) for value in box)
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            raise ValueError(f"Image {image_id} has a non-positive bounding box.")

        flat = []
        visible = 0
        try:
            canonical_points = _canonical_points(points)
        except ValueError as error:
            raise ValueError(
                f"Image {image_id} has an invalid landmark triplet."
            ) from error
        for point in canonical_points:
            x, y, source_visibility = point
            coco_visibility = 2 if source_visibility else 0
            flat.extend((float(x), float(y), coco_visibility))
            visible += bool(coco_visibility)

        records.append(
            {
                "id": first_annotation_id + offset,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [left, top, width, height],
                "area": width * height,
                "iscrowd": 0,
                "keypoints": flat,
                "num_keypoints": visible,
            }
        )
    return records


def _yolo_payload(text: str, width: int, height: int) -> dict:
    """Read this dataset's YOLO-pose sidecars into its JSON-shaped structure."""
    boxes = []
    keypoint_sets = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError as error:
            raise ValueError(f"Unreadable YOLO line {line_number}.") from error
        expected = 5 + len(KEYPOINT_NAMES) * 3
        if len(values) != expected:
            raise ValueError(
                f"YOLO line {line_number} has {len(values)} values, expected {expected}."
            )
        _, centre_x, centre_y, box_width, box_height, *flat_points = values
        left = (centre_x - box_width / 2) * width
        top = (centre_y - box_height / 2) * height
        right = (centre_x + box_width / 2) * width
        bottom = (centre_y + box_height / 2) * height
        boxes.append([left, top, right, bottom])
        keypoint_sets.append(
            [
                [
                    flat_points[index] * width,
                    flat_points[index + 1] * height,
                    flat_points[index + 2],
                ]
                for index in range(0, len(flat_points), 3)
            ]
        )
    return {"boxes": boxes, "keypoints": keypoint_sets}


def convert_split(
    source: zipfile.ZipFile,
    split: str,
    destination: pathlib.Path,
    limit: int | None = None,
) -> tuple[int, int]:
    image_prefix, label_prefix = SPLITS[split]
    images = sorted(
        name
        for name in source.namelist()
        if name.startswith(image_prefix) and not name.endswith("/")
    )
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError(f"No {split} images found in the source archive.")

    coco = {
        "info": {
            "description": "Spondylolisthesis Vertebral Landmark",
            "version": "1",
            "url": f"https://doi.org/{DATASET_DOI}",
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "vertebra",
                "supercategory": "none",
                "keypoints": list(KEYPOINT_NAMES),
                "skeleton": [list(edge) for edge in SKELETON],
            }
        ],
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as output:
        for image_id, source_name in enumerate(images, 1):
            file_name = pathlib.PurePosixPath(source_name).name
            label_stem = label_prefix + pathlib.PurePosixPath(file_name).stem
            try:
                image_bytes = source.read(source_name)
            except KeyError as error:
                raise ValueError(f"Missing annotation for {source_name}.") from error

            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
            if label_stem + ".json" in source.NameToInfo:
                payload = json.loads(source.read(label_stem + ".json"))
            elif label_stem + ".txt" in source.NameToInfo:
                payload = _yolo_payload(
                    source.read(label_stem + ".txt").decode("utf-8"), width, height
                )
            else:
                raise ValueError(f"Missing annotation for {source_name}.")
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "license": 1,
                }
            )
            records = _annotation(payload, image_id, len(coco["annotations"]) + 1)
            coco["annotations"].extend(records)
            output.writestr(f"images/{file_name}", image_bytes)

        output.writestr("annotations.coco.json", json.dumps(coco, indent=2))
        output.writestr("ATTRIBUTION.md", ATTRIBUTION)

    return len(coco["images"]), len(coco["annotations"])


def prepare(
    archive: pathlib.Path,
    output_dir: pathlib.Path,
    train_limit: int | None = None,
    valid_limit: int | None = None,
) -> dict[str, tuple[int, int]]:
    source, buffer = _dataset_zip(archive)
    try:
        return {
            "train": convert_split(
                source,
                "train",
                output_dir / "vertebral-keypoints-train.zip",
                train_limit,
            ),
            "valid": convert_split(
                source,
                "valid",
                output_dir / "vertebral-keypoints-valid.zip",
                valid_limit,
            ),
        }
    finally:
        source.close()
        if buffer is not None:
            buffer.close()


def _positive_limit(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=pathlib.Path, help="downloaded version-1 ZIP")
    source.add_argument(
        "--download",
        action="store_true",
        help="download and verify version 1 from Mendeley Data",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--train-limit", type=_positive_limit)
    parser.add_argument("--valid-limit", type=_positive_limit)
    args = parser.parse_args()

    archive = args.archive
    if args.download:
        archive = args.output_dir / "spondylolisthesis-vertebral-landmark-v1.zip"
        if not archive.is_file():
            print(f"Downloading {DOWNLOAD_URL}")
            download(archive)
        elif sha256(archive) != EXPECTED_SHA256:
            raise ValueError(f"Existing download at {archive} has the wrong SHA-256.")

    results = prepare(archive, args.output_dir, args.train_limit, args.valid_limit)
    for split, (images, instance_count) in results.items():
        print(f"{split}: {images} images, {instance_count} vertebrae")


if __name__ == "__main__":
    main()

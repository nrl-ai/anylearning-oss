#!/usr/bin/env python3
"""Generate small, import-ready COCO keypoint archives for AnyLearning.

The images are drawn locally and contain two five-landmark stick figures. They
are useful for learning the Keypoint Detection workflow and for quick package
checks; they are deliberately too simple to be an accuracy benchmark.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pathlib
import random
import zipfile

from PIL import Image, ImageDraw

WIDTH = 512
HEIGHT = 384
KEYPOINT_NAMES = ("head", "left_hand", "right_hand", "left_foot", "right_foot")
SKELETON = ((1, 2), (1, 3), (2, 4), (3, 5))


def _figure(
    cx: float, cy: float, scale: float, lean: float
) -> list[tuple[float, float]]:
    """Return the five named points for one deterministic stick figure."""
    return [
        (cx + lean, cy - 70 * scale),
        (cx - 48 * scale, cy - 10 * scale),
        (cx + 48 * scale, cy - 10 * scale),
        (cx - 30 * scale, cy + 72 * scale),
        (cx + 30 * scale, cy + 72 * scale),
    ]


def _draw_figure(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    colour: tuple[int, int, int],
) -> None:
    head, left_hand, right_hand, left_foot, right_foot = points
    shoulder = ((left_hand[0] + right_hand[0]) / 2, (left_hand[1] + right_hand[1]) / 2)
    hip = ((left_foot[0] + right_foot[0]) / 2, left_foot[1] - 45)
    lines = (
        (head, shoulder),
        (left_hand, right_hand),
        (shoulder, hip),
        (hip, left_foot),
        (hip, right_foot),
    )
    for start, end in lines:
        draw.line((start, end), fill=colour, width=8)
    for x, y in points:
        radius = 9
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)


def _annotation(
    annotation_id: int,
    image_id: int,
    points: list[tuple[float, float]],
    occluded_index: int | None,
) -> dict:
    pad = 14
    left = max(0, min(point[0] for point in points) - pad)
    top = max(0, min(point[1] for point in points) - pad)
    right = min(WIDTH, max(point[0] for point in points) + pad)
    bottom = min(HEIGHT, max(point[1] for point in points) + pad)
    flat = []
    for index, (x, y) in enumerate(points):
        flat.extend((round(x, 2), round(y, 2), 1 if index == occluded_index else 2))
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [
            round(left, 2),
            round(top, 2),
            round(right - left, 2),
            round(bottom - top, 2),
        ],
        "area": round((right - left) * (bottom - top), 2),
        "iscrowd": 0,
        "keypoints": flat,
        "num_keypoints": len(points),
    }


def generate_split(path: pathlib.Path, split: str, count: int, seed: int) -> None:
    randomizer = random.Random(seed)
    coco = {
        "info": {
            "description": "AnyLearning generated stick-figure keypoint example",
            "version": "1.0",
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC0-1.0",
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "figure",
                "supercategory": "none",
                "keypoints": list(KEYPOINT_NAMES),
                "skeleton": [list(edge) for edge in SKELETON],
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for offset in range(count):
            image_id = offset + 1
            name = f"{split}-{image_id:03d}.png"
            image = Image.new("RGB", (WIDTH, HEIGHT), (244, 247, 250))
            draw = ImageDraw.Draw(image)
            # Grid lines make geometry and visual alignment easy to inspect.
            for x in range(0, WIDTH, 32):
                draw.line((x, 0, x, HEIGHT), fill=(225, 230, 236), width=1)
            for y in range(0, HEIGHT, 32):
                draw.line((0, y, WIDTH, y), fill=(225, 230, 236), width=1)

            instances = []
            for instance, base_x in enumerate((150, 360)):
                phase = (offset * 0.43) + instance * 1.7
                cx = base_x + math.sin(phase) * 16 + randomizer.uniform(-3, 3)
                cy = 190 + math.cos(phase) * 12 + randomizer.uniform(-3, 3)
                scale = 0.86 + randomizer.uniform(-0.04, 0.04)
                lean = math.sin(phase * 0.7) * 12
                points = _figure(cx, cy, scale, lean)
                _draw_figure(
                    draw, points, (31, 111, 235) if instance == 0 else (234, 88, 12)
                )
                occluded = 2 if (offset + instance) % 7 == 0 else None
                instances.append((points, occluded))

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            archive.writestr(f"images/{name}", buffer.getvalue())
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": name,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "license": 1,
                }
            )
            for points, occluded in instances:
                coco["annotations"].append(
                    _annotation(
                        len(coco["annotations"]) + 1, image_id, points, occluded
                    )
                )

        archive.writestr("annotations.coco.json", json.dumps(coco, indent=2))
        archive.writestr(
            "README.txt",
            "Generated by AnyLearning's keypoint example. The generated images and labels are dedicated to the public domain under CC0-1.0.\n",
        )


def generate(
    output_dir: pathlib.Path, train_images: int, valid_images: int
) -> list[pathlib.Path]:
    if train_images < 1 or valid_images < 1:
        raise ValueError("train-images and valid-images must both be positive")
    outputs = [
        output_dir / "stick-figures-train.zip",
        output_dir / "stick-figures-valid.zip",
    ]
    generate_split(outputs[0], "train", train_images, seed=2602)
    generate_split(outputs[1], "valid", valid_images, seed=2603)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("keypoint-example")
    )
    parser.add_argument("--train-images", type=int, default=24)
    parser.add_argument("--valid-images", type=int, default=8)
    args = parser.parse_args()
    for output in generate(args.output_dir, args.train_images, args.valid_images):
        print(output)


if __name__ == "__main__":
    main()

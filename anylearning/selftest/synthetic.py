"""Datasets the app can make for itself, with no network and no real images.

The self-test has to work on a machine that has nothing on it -- a customer's
laptop, a fresh CI runner -- so it cannot rely on `~/anylearning-data` holding
projects, or on anything being downloadable. It draws its own.

Each class gets a distinct shape *and* a distinct colour, which is the same
choice `tests/fixtures/datasets.py` makes and for the same reason: a run that
cannot separate two classes only proves the code did not crash, which is a much
weaker claim than it looks.

What comes out is a zip per subset -- the shape `upload_data` accepts -- and the
annotation for each image in AnyLearning's own format, so the caller can label
what it uploaded.
"""

from __future__ import annotations

import io
import random
import zipfile

from PIL import Image, ImageDraw

# name -> (shape, fill). Colours are far apart in every channel.
CLASSES = {
    "circle": ("circle", (220, 60, 60)),
    "square": ("square", (60, 120, 220)),
    "triangle": ("triangle", (60, 180, 90)),
}

LABEL_COLOURS = ("#e23c3c", "#3c78e2", "#3cb45a")

SUBSETS = (0, 1, 2)  # train, val, test -- the values DataItem.subset takes


def labels_for(names) -> list[dict]:
    """Project labels, in the shape the projects API stores them."""
    return [
        {"name": name, "color": LABEL_COLOURS[index % len(LABEL_COLOURS)], "id": index}
        for index, name in enumerate(names)
    ]


def _draw(draw, shape, box, fill):
    left, top, right, bottom = box
    if shape == "circle":
        draw.ellipse(box, fill=fill)
    elif shape == "square":
        draw.rectangle(box, fill=fill)
    else:
        draw.polygon(
            [(left, bottom), ((left + right) / 2, top), (right, bottom)], fill=fill
        )


def _polygon_points(shape, box) -> list[list[int]]:
    """The outline of a drawn shape, as a polygon.

    Segmentation trainers want an outline rather than a box, and a coarse one is
    enough: the point is that the geometry survives the round trip through the
    database and the trainer's own converter.

    Integer coordinates, because that is what the labelling canvas stores and
    what the trainers are therefore fed in practice.
    """
    left, top, right, bottom = box
    if shape == "triangle":
        return [[left, bottom], [round((left + right) / 2), top], [right, bottom]]
    if shape == "circle":
        middle_x, middle_y = (left + right) / 2, (top + bottom) / 2
        radius_x, radius_y = (right - left) / 2, (bottom - top) / 2
        # An octagon is close enough to a circle for a loss to go down.
        return [
            [round(middle_x + radius_x * dx), round(middle_y + radius_y * dy)]
            for dx, dy in (
                (1, 0),
                (0.7, 0.7),
                (0, 1),
                (-0.7, 0.7),
                (-1, 0),
                (-0.7, -0.7),
                (0, -1),
                (0.7, -0.7),
            )
        ]
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _rectangle_points(box) -> list[list[float]]:
    left, top, right, bottom = box
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _image_with_shapes(rng, size, class_names, instances):
    """One image, and what is in it. Returns (png bytes, [(name, box, shape)])."""
    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    placed = []
    for _ in range(instances):
        name = rng.choice(class_names)
        shape, fill = CLASSES[name]
        width = rng.randint(size // 4, size // 2)
        left = rng.randint(2, max(3, size - width - 2))
        top = rng.randint(2, max(3, size - width - 2))
        box = (left, top, left + width, top + width)
        _draw(draw, shape, box, fill)
        placed.append((name, box, shape))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), placed


def build_subset(kind: str, count: int, size: int, seed: int, class_names):
    """A zip of images plus the annotation for each, keyed by file name.

    `kind` decides what the annotation says: a class for classification, boxes
    for detection, outlines for the two segmentation types, or named points for
    keypoint detection.
    """
    rng = random.Random(seed)
    archive = io.BytesIO()
    annotations: dict[str, dict] = {}

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for index in range(count):
            single = kind == "classification"
            png, placed = _image_with_shapes(
                rng,
                size,
                list(CLASSES) if kind == "keypoint" else class_names,
                1 if single else rng.randint(1, 2),
            )
            name = f"{kind}_{index:03d}.png"
            zipped.writestr(name, png)

            if single:
                annotations[name] = {"class_name": placed[0][0]}
                continue

            shapes = []
            for shape_index, (_label, box, _shape) in enumerate(placed, start=1):
                if kind == "keypoint":
                    left, top, right, bottom = box
                    width, height = right - left, bottom - top
                    # Every synthetic subject carries the complete schema. The
                    # relative locations are intentionally asymmetric so a
                    # model cannot get credit by predicting one centre point
                    # for every landmark.
                    anchors = ((0.3, 0.3), (0.7, 0.3), (0.5, 0.7))
                    for landmark_index, label in enumerate(class_names):
                        dx, dy = anchors[landmark_index % len(anchors)]
                        shapes.append(
                            {
                                "id": len(shapes) + 1,
                                "position": [
                                    round(left + width * dx),
                                    round(top + height * dy),
                                ],
                                "categories": [label],
                                "type": "dot",
                                "group_id": shape_index,
                                "visible": 2,
                            }
                        )
                    continue
                outline = kind == "detection"
                shapes.append(
                    {
                        "id": shape_index,
                        "points": _rectangle_points(box)
                        if outline
                        else _polygon_points(_shape, box),
                        "phi": 0,
                        "categories": [_label],
                        "type": "rectangle" if outline else "polygon",
                    }
                )
            annotations[name] = {"shapes": shapes}

    return archive.getvalue(), annotations

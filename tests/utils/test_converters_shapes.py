"""Per-shape-type behaviour of the annotation converters.

Each target format expresses shapes differently -- COCO gives rectangles an empty
segmentation, LabelMe collapses a rectangle to two corners, YOLO can only encode
an axis-aligned box. These branches are where a shape type quietly gets dropped
or reshaped, which the round-trip tests in test_converters.py cannot see because
they only exercise polygons.
"""

import pytest

from anylearning.utils.converters import (
    convert_anylearning_to_coco,
    convert_anylearning_to_labelme,
    convert_anylearning_to_yolo,
)

LABELS = [{"name": "cat", "id": 0}, {"name": "dog", "id": 1}]
IMAGE_SIZE = (200, 100)


def shape(obj_type, points, label="cat", obj_id=1):
    return {
        "id": obj_id,
        "categories": [label],
        "points": points,
        "type": obj_type,
        "phi": 0,
    }


BOX_POINTS = [[50, 20], [150, 20], [150, 80], [50, 80]]
TRIANGLE_POINTS = [[10, 10], [90, 10], [50, 70]]


def coco_annotations(objects):
    result = convert_anylearning_to_coco(
        objects, LABELS, image_id=1, image_filename="a.jpg", image_size=IMAGE_SIZE
    )
    return result["annotations"] if isinstance(result, dict) else result


# --------------------------------------------------------------------------
# COCO
# --------------------------------------------------------------------------


def test_coco_rectangle_has_no_segmentation():
    """A box carries no mask, so segmentation stays empty."""
    [annotation] = coco_annotations([shape("rectangle", BOX_POINTS)])
    assert annotation["bbox"] == [50, 20, 100, 60]
    assert annotation["segmentation"] == []


def test_coco_polygon_carries_its_outline():
    [annotation] = coco_annotations([shape("polygon", TRIANGLE_POINTS)])
    assert annotation["segmentation"] == [[10, 10, 90, 10, 50, 70]]


def test_coco_polygon_area_is_the_enclosed_area_not_the_bbox():
    """A triangle covers half its bounding box; conflating them skews AP."""
    [annotation] = coco_annotations([shape("polygon", TRIANGLE_POINTS)])
    x, y, width, height = annotation["bbox"]
    assert annotation["area"] == pytest.approx(0.5 * 80 * 60)
    assert annotation["area"] < width * height


def test_coco_polyline_is_treated_as_a_polygon():
    [annotation] = coco_annotations([shape("polyline", TRIANGLE_POINTS)])
    assert annotation["segmentation"]


def test_coco_annotation_ids_are_unique_across_shapes():
    annotations = coco_annotations(
        [
            shape("rectangle", BOX_POINTS, obj_id=1),
            shape("polygon", TRIANGLE_POINTS, label="dog", obj_id=2),
        ]
    )
    ids = [annotation["id"] for annotation in annotations]
    assert len(ids) == len(set(ids))


def test_coco_uses_the_project_label_id():
    [annotation] = coco_annotations([shape("polygon", TRIANGLE_POINTS, label="dog")])
    assert annotation["category_id"] == 1


def test_coco_skips_unknown_categories():
    assert coco_annotations([shape("polygon", TRIANGLE_POINTS, label="ghost")]) == []


def test_coco_skips_degenerate_points():
    """A single click should not become an annotation with a zero-area box."""
    assert coco_annotations([shape("polygon", [[5, 5]])]) == []


# --------------------------------------------------------------------------
# LabelMe
# --------------------------------------------------------------------------


def test_labelme_rectangle_is_reduced_to_two_corners():
    """LabelMe stores rectangles as (top-left, bottom-right)."""
    data = convert_anylearning_to_labelme(
        [shape("rectangle", BOX_POINTS)], "a.jpg", IMAGE_SIZE
    )
    [labelme_shape] = data["shapes"]
    assert labelme_shape["shape_type"] == "rectangle"
    assert labelme_shape["points"] == [[50, 20], [150, 80]]


def test_labelme_polygon_keeps_every_vertex():
    data = convert_anylearning_to_labelme(
        [shape("polygon", TRIANGLE_POINTS)], "a.jpg", IMAGE_SIZE
    )
    [labelme_shape] = data["shapes"]
    assert labelme_shape["shape_type"] == "polygon"
    assert labelme_shape["points"] == TRIANGLE_POINTS


def test_labelme_shapes_carry_the_expected_keys():
    """Downstream LabelMe tooling reads all of these."""
    data = convert_anylearning_to_labelme(
        [shape("polygon", TRIANGLE_POINTS)], "a.jpg", IMAGE_SIZE
    )
    [labelme_shape] = data["shapes"]
    assert set(labelme_shape) == {"label", "points", "group_id", "shape_type", "flags"}
    assert labelme_shape["label"] == "cat"


def test_labelme_handles_mixed_shape_types_in_one_image():
    data = convert_anylearning_to_labelme(
        [shape("rectangle", BOX_POINTS), shape("polygon", TRIANGLE_POINTS, label="dog")],
        "a.jpg",
        IMAGE_SIZE,
    )
    assert [s["shape_type"] for s in data["shapes"]] == ["rectangle", "polygon"]


def test_labelme_with_no_objects_still_describes_the_image():
    data = convert_anylearning_to_labelme([], "empty.jpg", IMAGE_SIZE)
    assert data["shapes"] == []
    assert data["imagePath"] == "empty.jpg"
    assert data["imageWidth"] == 200


# --------------------------------------------------------------------------
# YOLO across shape types
# --------------------------------------------------------------------------


@pytest.mark.parametrize("obj_type", ["rectangle", "polygon", "polyline"])
def test_yolo_encodes_every_closed_shape_as_its_box(obj_type):
    """Regression: polygons used to be dropped, producing empty label files."""
    text = convert_anylearning_to_yolo(
        [shape(obj_type, BOX_POINTS)], LABELS, IMAGE_SIZE
    )
    assert text.strip(), f"{obj_type} produced no YOLO annotation"
    class_id, cx, cy, w, h = text.strip().split()
    assert class_id == "0"
    assert float(cx) == pytest.approx(0.5)
    assert float(w) == pytest.approx(0.5)


def test_yolo_boxes_a_triangle_by_its_extremes():
    text = convert_anylearning_to_yolo(
        [shape("polygon", TRIANGLE_POINTS)], LABELS, IMAGE_SIZE
    )
    _, cx, cy, w, h = text.strip().split()
    # x spans 10..90 of 200 -> centre 0.25, width 0.4
    assert float(cx) == pytest.approx(0.25)
    assert float(w) == pytest.approx(0.4)
    # y spans 10..70 of 100 -> centre 0.4, height 0.6
    assert float(cy) == pytest.approx(0.4)
    assert float(h) == pytest.approx(0.6)


def test_yolo_skips_objects_without_a_type():
    obj = shape("polygon", BOX_POINTS)
    del obj["type"]
    assert convert_anylearning_to_yolo([obj], LABELS, IMAGE_SIZE) == ""


def test_yolo_skips_objects_without_points():
    obj = shape("polygon", BOX_POINTS)
    del obj["points"]
    assert convert_anylearning_to_yolo([obj], LABELS, IMAGE_SIZE) == ""


def test_yolo_skips_non_dict_entries():
    text = convert_anylearning_to_yolo(
        ["not an object", shape("rectangle", BOX_POINTS)], LABELS, IMAGE_SIZE
    )
    assert len(text.strip().splitlines()) == 1


def test_yolo_accepts_a_string_category():
    obj = shape("rectangle", BOX_POINTS)
    obj["categories"] = "cat"
    assert convert_anylearning_to_yolo([obj], LABELS, IMAGE_SIZE).strip()


def test_yolo_survives_malformed_labels():
    """A corrupt project label list should return nothing, not raise."""
    assert convert_anylearning_to_yolo([shape("rectangle", BOX_POINTS)], None, IMAGE_SIZE) == ""
    assert convert_anylearning_to_yolo([shape("rectangle", BOX_POINTS)], [{}], IMAGE_SIZE) == ""


def test_yolo_skips_points_that_are_not_pairs():
    assert convert_anylearning_to_yolo(
        [shape("rectangle", [[1], [2], [3]])], LABELS, IMAGE_SIZE
    ) == ""

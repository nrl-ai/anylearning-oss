"""Annotation format conversion.

These functions sit between the labelling UI and every trainer, so a bug here
does not raise -- it silently mislabels a dataset and the model quietly gets
worse. That makes them worth testing far more thoroughly than their line count
suggests, especially the coordinate maths and the empty/degenerate inputs the
UI can legitimately produce.
"""

import pytest

from anylearning.utils.converters import (
    convert_anylabeling_to_anylearning,
    convert_anylearning_to_anylabeling,
    convert_anylearning_to_coco,
    convert_anylearning_to_labelme,
    convert_anylearning_to_yolo,
    convert_coco_to_anylearning,
)

LABELS = [{"name": "cat", "id": 0}, {"name": "dog", "id": 1}]
IMAGE_SIZE = (200, 100)  # width, height -- deliberately non-square


def rectangle(label="cat", x0=50, y0=20, x1=150, y1=80, obj_id=1):
    """An AnyLearning polygon covering an axis-aligned box."""
    return {
        "id": obj_id,
        "categories": [label],
        "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "type": "polygon",
        "phi": 0,
    }


# --------------------------------------------------------------------------
# AnyLabeling -> AnyLearning
# --------------------------------------------------------------------------


def test_anylabeling_rectangle_becomes_four_corners():
    """AnyLabeling stores a rectangle as two opposite corners, not four."""
    payload = {
        "shapes": [
            {
                "label": "cat",
                "shape_type": "rectangle",
                "points": [[10, 20], [110, 220]],
            }
        ]
    }
    [obj] = convert_anylabeling_to_anylearning(payload)

    assert obj["categories"] == ["cat"]
    assert obj["type"] == "rectangle"
    assert obj["points"] == [[10, 20], [110, 20], [110, 220], [10, 220]]


def test_anylabeling_polygon_points_pass_through():
    points = [[1, 2], [30, 4], [5, 60]]
    payload = {"shapes": [{"label": "dog", "shape_type": "polygon", "points": points}]}
    [obj] = convert_anylabeling_to_anylearning(payload)
    assert obj["points"] == points


def test_anylabeling_point_keeps_only_first_coordinate():
    payload = {"shapes": [{"label": "cat", "shape_type": "point", "points": [[7, 8]]}]}
    [obj] = convert_anylabeling_to_anylearning(payload)
    assert obj["points"] == [[7, 8]]


def test_anylabeling_preserves_keypoint_instance_and_visibility():
    payload = {
        "shapes": [
            {
                "label": "nose",
                "shape_type": "point",
                "points": [[7, 8]],
                "group_id": 3,
                "flags": {"visibility": 1},
            }
        ]
    }
    [obj] = convert_anylabeling_to_anylearning(payload)
    assert obj["group_id"] == 3
    assert obj["visible"] == 1


def test_anylabeling_ids_are_sequential():
    payload = {
        "shapes": [
            {
                "label": "cat",
                "shape_type": "polygon",
                "points": [[0, 0], [1, 1], [2, 2]],
            },
            {
                "label": "dog",
                "shape_type": "polygon",
                "points": [[3, 3], [4, 4], [5, 5]],
            },
        ]
    }
    objects = convert_anylabeling_to_anylearning(payload)
    assert [obj["id"] for obj in objects] == [1, 2]


def test_anylabeling_empty_and_missing_shapes():
    assert convert_anylabeling_to_anylearning({}) == []
    assert convert_anylabeling_to_anylearning({"shapes": []}) == []


def test_anylabeling_malformed_shape_is_skipped_not_fatal():
    """A rectangle with one corner should not abort the whole file."""
    payload = {
        "shapes": [
            {"label": "cat", "shape_type": "rectangle", "points": [[1, 2]]},
            {
                "label": "dog",
                "shape_type": "polygon",
                "points": [[0, 0], [1, 1], [2, 2]],
            },
        ]
    }
    objects = convert_anylabeling_to_anylearning(payload)
    assert [obj["categories"] for obj in objects] == [["dog"]]


# --------------------------------------------------------------------------
# AnyLearning -> YOLO
# --------------------------------------------------------------------------


def test_yolo_centre_and_extent_are_normalised():
    """YOLO is `class cx cy w h`, all normalised to [0, 1]."""
    text = convert_anylearning_to_yolo([rectangle()], LABELS, IMAGE_SIZE)
    class_id, cx, cy, w, h = text.strip().split()

    assert class_id == "0"
    # box x 50..150 of 200 wide -> centre 0.5, width 0.5
    assert float(cx) == pytest.approx(0.5)
    assert float(w) == pytest.approx(0.5)
    # box y 20..80 of 100 high -> centre 0.5, height 0.6
    assert float(cy) == pytest.approx(0.5)
    assert float(h) == pytest.approx(0.6)


def test_yolo_uses_the_label_id_not_list_position():
    """Label ids come from the project, so they need not match list order."""
    labels = [{"name": "cat", "id": 7}, {"name": "dog", "id": 3}]
    text = convert_anylearning_to_yolo([rectangle(label="dog")], labels, IMAGE_SIZE)
    assert text.strip().split()[0] == "3"


def test_yolo_all_values_within_unit_range():
    objects = [rectangle(), rectangle(label="dog", x0=0, y0=0, x1=200, y1=100)]
    text = convert_anylearning_to_yolo(objects, LABELS, IMAGE_SIZE)
    for line in text.strip().splitlines():
        for value in line.split()[1:]:
            assert 0.0 <= float(value) <= 1.0, line


def test_yolo_empty_input_is_empty_string():
    assert convert_anylearning_to_yolo([], LABELS, IMAGE_SIZE) == ""
    assert convert_anylearning_to_yolo(None, LABELS, IMAGE_SIZE) == ""


def test_yolo_accepts_a_single_object_not_wrapped_in_a_list():
    text = convert_anylearning_to_yolo(rectangle(), LABELS, IMAGE_SIZE)
    assert text.strip().split()[0] == "0"


def test_yolo_unknown_label_does_not_crash():
    """A label removed from the project must not take the export down."""
    text = convert_anylearning_to_yolo([rectangle(label="ghost")], LABELS, IMAGE_SIZE)
    assert "ghost" not in text


# --------------------------------------------------------------------------
# AnyLearning -> COCO
# --------------------------------------------------------------------------


def test_coco_bbox_is_xywh_in_pixels():
    """COCO uses absolute [x, y, width, height], unlike YOLO."""
    result = convert_anylearning_to_coco(
        [rectangle()], LABELS, image_id=1, image_filename="a.jpg", image_size=IMAGE_SIZE
    )
    annotations = result["annotations"] if isinstance(result, dict) else result
    [annotation] = annotations
    assert annotation["bbox"] == [50, 20, 100, 60]
    assert annotation["area"] == pytest.approx(100 * 60)
    assert annotation["iscrowd"] == 0


def test_coco_segmentation_is_flat_xy_pairs():
    result = convert_anylearning_to_coco(
        [rectangle()], LABELS, image_id=1, image_filename="a.jpg", image_size=IMAGE_SIZE
    )
    annotations = result["annotations"] if isinstance(result, dict) else result
    [segmentation] = annotations[0]["segmentation"]
    assert len(segmentation) % 2 == 0
    assert len(segmentation) >= 6, "a polygon needs at least three points"


def test_coco_empty_input():
    assert convert_anylearning_to_coco([], LABELS, 1, "a.jpg", IMAGE_SIZE) == []


# --------------------------------------------------------------------------
# AnyLearning -> LabelMe / AnyLabeling
# --------------------------------------------------------------------------


def test_labelme_records_image_geometry():
    data = convert_anylearning_to_labelme([rectangle()], "a.jpg", IMAGE_SIZE)
    assert data["imagePath"] == "a.jpg"
    assert data["imageWidth"] == 200
    assert data["imageHeight"] == 100
    assert data["imageData"] is None
    assert len(data["shapes"]) == 1


def test_anylabeling_is_labelme_plus_its_own_version():
    labelme = convert_anylearning_to_labelme([rectangle()], "a.jpg", IMAGE_SIZE)
    anylabeling = convert_anylearning_to_anylabeling([rectangle()], "a.jpg", IMAGE_SIZE)

    assert anylabeling["version"] != labelme["version"]
    assert anylabeling["imagePath"] == labelme["imagePath"]
    assert len(anylabeling["shapes"]) == len(labelme["shapes"])


def test_anylabeling_export_does_not_mutate_the_labelme_base():
    """The two share a dict; a shallow copy that leaked would corrupt exports."""
    objects = [rectangle()]
    anylabeling = convert_anylearning_to_anylabeling(objects, "a.jpg", IMAGE_SIZE)
    fresh = convert_anylearning_to_labelme(objects, "a.jpg", IMAGE_SIZE)
    assert fresh["version"] != anylabeling["version"]


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_anylabeling_round_trip_preserves_geometry_and_labels():
    """Export then re-import must not move the annotation."""
    original = [
        rectangle(),
        rectangle(label="dog", x0=10, y0=10, x1=60, y1=40, obj_id=2),
    ]

    exported = convert_anylearning_to_anylabeling(original, "a.jpg", IMAGE_SIZE)
    reimported = convert_anylabeling_to_anylearning(exported)

    assert [obj["categories"] for obj in reimported] == [["cat"], ["dog"]]
    for before, after in zip(original, reimported):
        assert [[float(x), float(y)] for x, y in after["points"]] == [
            [float(x), float(y)] for x, y in before["points"]
        ]


def test_labelme_keypoint_round_trip_preserves_group_and_visibility():
    original = [
        {
            "id": "p1",
            "type": "dot",
            "position": [12, 34],
            "categories": ["nose"],
            "group_id": 7,
            "visible": 1,
        }
    ]
    exported = convert_anylearning_to_anylabeling(original, "pose.jpg", IMAGE_SIZE)
    [shape] = exported["shapes"]
    assert shape["shape_type"] == "point"
    assert shape["points"] == [[12, 34]]
    assert shape["group_id"] == 7
    assert shape["flags"] == {"visibility": 1}

    [reimported] = convert_anylabeling_to_anylearning(exported)
    assert reimported["points"] == [[12, 34]]
    assert reimported["group_id"] == 7
    assert reimported["visible"] == 1


def test_coco_keypoints_import_as_grouped_canvas_dots():
    coco = {
        "images": [{"id": 1, "file_name": "images/pose.jpg"}],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "keypoints": ["left_eye", "right_eye", "nose"],
            }
        ],
        "annotations": [
            {
                "id": 42,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [10, 20, 2, 30, 40, 1, 0, 0, 0],
                "num_keypoints": 2,
            }
        ],
    }
    points = convert_coco_to_anylearning(coco)["pose.jpg"]
    assert [point["categories"] for point in points] == [
        ["left_eye"],
        ["right_eye"],
    ]
    assert [point["position"] for point in points] == [[10, 20], [30, 40]]
    assert {point["group_id"] for point in points} == {42}
    assert [point["visible"] for point in points] == [2, 1]


def test_anylearning_keypoints_export_as_one_coco_instance():
    points = [
        {
            "type": "dot",
            "position": [10, 20],
            "categories": ["left_eye"],
            "group_id": 4,
        },
        {
            "type": "point",
            "points": [[30, 40]],
            "categories": ["right_eye"],
            "group_id": 4,
            "visible": 1,
        },
    ]
    [annotation] = convert_anylearning_to_coco(
        points,
        LABELS,
        1,
        "pose.jpg",
        IMAGE_SIZE,
        keypoint_names=["left_eye", "right_eye", "nose"],
    )
    assert annotation["category_id"] == 1
    assert annotation["keypoints"] == [10.0, 20.0, 2, 30.0, 40.0, 1, 0.0, 0.0, 0]
    assert annotation["num_keypoints"] == 2
    assert annotation["bbox"][2] > 0 and annotation["bbox"][3] > 0

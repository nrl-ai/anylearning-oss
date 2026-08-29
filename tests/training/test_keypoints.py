"""What labelled points have to become before RF-DETR can train on them.

The interesting cases are all degenerate ones: a single visible landmark, points
that never got placed, two instances in one image, a schema with no left or
right in it. Those are what a real project produces on its first day.
"""

import pytest

from anylearning.training import keypoints


def label_set(*names):
    return [{"id": index + 1, "name": name} for index, name in enumerate(names)]


def point(name, x, y, group=None, **extra):
    shape = {"type": "point", "categories": [name], "points": [[x, y]]}
    if group is not None:
        shape["group_id"] = group
    shape.update(extra)
    return shape


def annotation(*shapes):
    return {"data": list(shapes)}


def test_names_follow_label_id_not_list_order():
    """The index of a keypoint is baked into every annotation already written.

    So the order comes from the id, which a rename does not change, rather than
    from the position in the list -- which the label editor may reorder.
    """
    labels = [{"id": 3, "name": "tail"}, {"id": 1, "name": "nose"}]
    assert keypoints.keypoint_names(labels) == ["nose", "tail"]


def test_a_label_with_no_name_is_not_a_landmark():
    assert keypoints.keypoint_names([{"id": 1, "name": ""}, {"id": 2}]) == []


def test_one_instance_from_ungrouped_points():
    names = ["nose", "tail"]
    built = keypoints.instances(
        annotation(point("nose", 10, 20), point("tail", 30, 40)), names
    )
    assert len(built) == 1
    assert built[0]["keypoints"] == [10.0, 20.0, 2, 30.0, 40.0, 2]
    assert built[0]["num_keypoints"] == 2


def test_canvas_dots_are_points_too():
    """The React canvas writes ``dot``/``position``, unlike LabelMe imports."""
    built = keypoints.instances(
        {"data": [{"type": "dot", "categories": ["nose"], "position": [3, 4]}]},
        ["nose"],
    )
    assert built[0]["keypoints"] == [3.0, 4.0, keypoints.VISIBLE]


def test_group_ids_separate_instances():
    """Two subjects in one image are two annotations, not one of four points."""
    names = ["nose", "tail"]
    built = keypoints.instances(
        annotation(
            point("nose", 10, 10, group=1),
            point("tail", 20, 20, group=1),
            point("nose", 100, 100, group=2),
            point("tail", 110, 110, group=2),
        ),
        names,
    )
    assert len(built) == 2
    assert [instance["num_keypoints"] for instance in built] == [2, 2]
    # Not merged into one box spanning both subjects.
    assert built[0]["bbox"][0] == 10 and built[1]["bbox"][0] == 100


def test_numeric_and_text_group_ids_can_coexist():
    built = keypoints.instances(
        annotation(point("nose", 1, 1, group=1), point("nose", 2, 2, group="other")),
        ["nose"],
    )
    assert len(built) == 2


def test_numeric_group_ids_are_ordered_as_numbers():
    built = keypoints.instances(
        annotation(point("nose", 20, 20, group=10), point("nose", 2, 2, group=2)),
        ["nose"],
    )
    assert [instance["keypoints"][0] for instance in built] == [2.0, 20.0]


def test_duplicate_landmark_in_one_group_uses_the_last_point_once():
    built = keypoints.instances(
        annotation(point("nose", 1, 1, group=1), point("nose", 9, 9, group=1)),
        ["nose"],
    )
    assert built[0]["keypoints"] == [9.0, 9.0, 2]
    assert built[0]["num_keypoints"] == 1


def test_group_zero_is_not_the_same_as_no_group():
    """`group_id: 0` is a group. Treating it as absent merges two instances.

    A dict keyed on the raw value gets this right and a falsy check does not,
    which is the entire reason for the test.
    """
    names = ["nose"]
    built = keypoints.instances(
        annotation(point("nose", 1, 1, group=0), point("nose", 50, 50)), names
    )
    assert len(built) == 2


def test_unplaced_landmarks_are_zero_zero_zero():
    names = ["nose", "tail", "ear"]
    built = keypoints.instances(annotation(point("tail", 5, 6)), names)
    assert built[0]["keypoints"] == [0.0, 0.0, 0, 5.0, 6.0, 2, 0.0, 0.0, 0]
    assert built[0]["num_keypoints"] == 1


def test_an_instance_with_nothing_placed_is_dropped():
    """It carries no supervision and its box would be a point at the origin."""
    names = ["nose"]
    assert keypoints.instances(annotation(point("nose", 4, 4, visible=0)), names) == []


@pytest.mark.parametrize(
    "value,expected",
    [
        ("visible", keypoints.VISIBLE),
        ("occluded", keypoints.OCCLUDED),
        ("hidden", keypoints.OCCLUDED),
        (2, keypoints.VISIBLE),
        (1, keypoints.OCCLUDED),
        (0, keypoints.NOT_LABELLED),
        (True, keypoints.VISIBLE),
        (False, keypoints.OCCLUDED),
        (None, keypoints.VISIBLE),
        ("something else", keypoints.VISIBLE),
    ],
)
def test_visibility_is_three_state(value, expected):
    """v=0 is excluded from the loss and v=1 is not, so the difference is real.

    `False` maps to occluded rather than to unlabelled deliberately: a user who
    placed a point and ticked "not visible" has told us where an occluded joint
    is, which is supervision. Never placing it is what means nothing is known.
    """
    shape = point("nose", 1, 2)
    if value is not None:
        shape["visible"] = value
    built = keypoints.instances(annotation(shape), ["nose"])
    if expected == keypoints.NOT_LABELLED:
        assert built == []
    else:
        assert built[0]["keypoints"][2] == expected


def test_occluded_points_still_count_and_still_bound_the_box():
    built = keypoints.instances(
        annotation(point("nose", 10, 10), point("tail", 90, 90, visible="occluded")),
        ["nose", "tail"],
    )
    assert built[0]["num_keypoints"] == 2
    assert built[0]["bbox"] == [10.0, 10.0, 80.0, 80.0]


def test_a_single_point_gets_a_box_with_area():
    """RF-DETR's loss divides by the box; a zero-sized one ends the run."""
    built = keypoints.instances(annotation(point("nose", 7, 7)), ["nose"])
    x, y, width, height = built[0]["bbox"]
    assert width > 0 and height > 0
    # Centred on the point rather than growing away from it.
    assert x < 7 < x + width and y < 7 < y + height


def test_collinear_points_get_a_box_with_area():
    """Horizontal landmarks -- a row of teeth, a fin -- have zero height."""
    built = keypoints.instances(
        annotation(point("a", 0, 5), point("b", 40, 5)), ["a", "b"]
    )
    _, _, width, height = built[0]["bbox"]
    assert width == 40
    assert height > 0


def test_only_the_degenerate_dimension_is_padded():
    built = keypoints.instances(
        annotation(point("a", 0, 0), point("b", 40, 30)), ["a", "b"]
    )
    assert built[0]["bbox"] == [0.0, 0.0, 40.0, 30.0]


def test_shapes_that_are_not_points_are_ignored():
    """A keypoint project may still hold boxes from an import or an auto-label."""
    box = {
        "type": "rectangle",
        "categories": ["nose"],
        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
    }
    assert keypoints.instances(annotation(box), ["nose"]) == []


def test_a_point_labelled_with_something_not_in_the_schema_is_ignored():
    assert keypoints.instances(annotation(point("elbow", 1, 1)), ["nose"]) == []


def test_malformed_shapes_do_not_raise():
    for shape in (
        {"type": "point", "categories": ["nose"], "points": []},
        {"type": "point", "categories": ["nose"], "points": [[1]]},
        {"type": "point", "categories": ["nose"], "points": [["x", "y"]]},
        {"type": "point", "categories": [], "points": [[1, 2]]},
        {"type": "point", "points": [[1, 2]]},
        "not a shape at all",
    ):
        assert keypoints.instances({"data": [shape]}, ["nose"]) == []


def test_no_names_means_nothing_to_build():
    assert keypoints.instances(annotation(point("nose", 1, 1)), []) == []


def test_missing_annotation_is_not_an_error():
    for empty in (None, {}, {"data": None}, {"data": []}):
        assert keypoints.instances(empty, ["nose"]) == []


@pytest.mark.parametrize(
    "names,expected",
    [
        (["left_eye", "right_eye"], [0, 1]),
        (["right_eye", "left_eye"], [0, 1]),
        (["l_paw", "r_paw"], [0, 1]),
        (["nose", "left_ear", "right_ear"], [1, 2]),
        (["Left_eye", "right_eye"], [0, 1]),
        (["nose", "tail"], []),
        (["left_eye"], []),
        (["front_left_paw", "front_right_ear"], []),
    ],
)
def test_flip_pairs_come_from_names(names, expected):
    """Flipping an image swaps left with right, and the labels must swap too.

    Otherwise every horizontally flipped example teaches the model that the two
    sides are interchangeable. Inferred only from these prefixes: a guess from
    any shared substring pairs a left paw with a right ear, which is worse than
    not flipping.
    """
    assert keypoints.flip_pairs(names) == expected


def test_flip_pairs_never_pairs_a_landmark_with_itself():
    assert keypoints.flip_pairs(["left", "left"]) == []


def test_the_category_carries_the_schema_and_no_invented_skeleton():
    category = keypoints.coco_category(["nose", "tail"])
    assert category["keypoints"] == ["nose", "tail"]
    # Empty on purpose: nothing in training reads it, and a made-up skeleton is
    # a claim about anatomy.
    assert category["skeleton"] == []
    assert category["supercategory"] == "none"

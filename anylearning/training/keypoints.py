"""Turning labelled points into COCO keypoint instances.

A keypoint project is labelled as points, one per named landmark, and trained
from a COCO file whose categories carry a ``keypoints`` list. This module is the
translation between the two, and it is separate from any trainer because the
same translation is needed by the exporter, the class-distribution endpoint and
the tests -- and because it is the part with all the decisions in it.

Three of those decisions are worth stating, because none of them is forced:

**The keypoint names are the project's labels.** A keypoint schema needs an
ordered list of landmark names, and the project already has an ordered list of
named things with add, rename, delete and reorder built around it. Making them
the same list means the labelling UI, the label editor and the class-balance
chart all work on a keypoint project without knowing it is one, and it is why
this feature needs no migration on the main database.

The cost is that the *object* class is implicit: a project has one kind of
subject with one skeleton, not "person" and "car" with different landmarks each.
COCO permits the general case and so does RF-DETR
(``num_keypoints_per_class``); this writes a single category. A project that
needs two skeletons needs two projects for now.

**Instances group by ``group_id``.** COCO keypoints are per *instance* -- three
people means three annotations of seventeen points -- so the points in an image
have to be attributable. ``group_id`` is the field AnyLabeling and LabelMe
already use for exactly this, and our own importer was dropping it. Points
without one are read as a single unnamed instance, which is the common case
(one subject per image) and means a project labelled without ever thinking about
groups still trains.

**Visibility is COCO's three-state flag, not a boolean.** ``v=0`` unlabelled,
``v=1`` labelled but occluded, ``v=2`` visible. It matters because ``v=0``
points are excluded from the loss while ``v=1`` points are not: marking an
occluded wrist as absent teaches the model nothing, and marking an absent one as
occluded teaches it something false. A point that was never placed is 0 here,
which is the only answer available and the right one.
"""

from __future__ import annotations

import re

#: COCO's visibility values, named.
NOT_LABELLED = 0
OCCLUDED = 1
VISIBLE = 2

#: What a point shape's own visibility field may say, mapped to COCO's number.
#: Anything absent or unrecognised is treated as visible, because a point the
#: user placed and said nothing about is one they can see.
#:
#: Booleans are handled before this lookup rather than in it. `True == 1` and
#: they hash alike, so a dict cannot hold both: written as entries here, `1:
#: OCCLUDED` silently replaced `True: VISIBLE` and every point a UI sent as a
#: boolean came out occluded. Two of the tests below exist because of it.
_VISIBILITY = {
    "visible": VISIBLE,
    "occluded": OCCLUDED,
    "hidden": OCCLUDED,
    2: VISIBLE,
    1: OCCLUDED,
    0: NOT_LABELLED,
}

#: The single category a keypoint project exports under.
DEFAULT_CATEGORY = "object"

#: OKS sigma used for every landmark when the project has not measured its own.
#:
#: COCO publishes seventeen hand-fitted sigmas for human pose, derived from how
#: much independent annotators disagreed about each joint. A project labelling
#: something else has no such study, and inventing per-landmark numbers would be
#: worse than admitting they are unknown: the sigma scales the distance at which
#: a prediction counts as correct, so a wrong one silently makes a metric
#: generous or harsh. 0.05 is COCO's own value for the eyes and the smallest it
#: publishes, so it is the strict end of the range rather than a flattering one.
DEFAULT_OKS_SIGMA = 0.05

#: Prefix pairs that make two landmark names each other's mirror image.
#:
#: Horizontal flipping is the one augmentation a keypoint model cannot do
#: naively: flipping the image turns a left wrist into a right wrist, so the
#: labels have to swap with the pixels or every flipped example teaches the
#: model that left and right are interchangeable. Inferred from names because
#: that is where the information is, and only from these prefixes -- a guess
#: from any shared substring would pair "front_left_paw" with "front_right_ear".
_MIRROR_PREFIXES = (("left_", "right_"), ("l_", "r_"), ("left", "right"))

# A number of pose datasets use compact suffixes instead: ``eyeL``/``eyeR``
# or ``forelegL2``/``forelegR2``. Require an uppercase marker and an optional
# numeric suffix so ordinary words ending in a lowercase l/r are not guessed
# to be anatomical pairs.
_MIRROR_SUFFIX = re.compile(r"^(?P<stem>.+)(?P<side>[LR])(?P<number>\d*)$")


def _mirror_partner(name: str) -> str | None:
    lowered = name.lower()
    for left, right in _MIRROR_PREFIXES:
        if lowered.startswith(left):
            return right + name[len(left) :]
        if lowered.startswith(right):
            return left + name[len(right) :]
    suffix = _MIRROR_SUFFIX.fullmatch(name)
    if suffix:
        side = "R" if suffix.group("side") == "L" else "L"
        return f"{suffix.group('stem')}{side}{suffix.group('number')}"
    return None


def keypoint_names(labels) -> list[str]:
    """The landmark names, in the order their indices are assigned.

    Sorted by label id rather than by name or by list position: the id is what
    survives a rename, and the *index* of a keypoint is part of every annotation
    already written. Reordering the schema after labelling would silently
    relabel every point in the project.
    """
    ordered = sorted(
        (label for label in labels or [] if label.get("name")),
        key=lambda label: label.get("id") or 0,
    )
    # Early keypoint projects stored one object label (for example ``locust``)
    # with the landmark schema nested under ``label["keypoints"]``. New
    # projects store landmarks as labels directly. Both representations exist
    # in exported projects, so normalize them at this boundary rather than
    # requiring a destructive database migration.
    nested = []
    for label in ordered:
        for landmark in label.get("keypoints") or []:
            name = landmark.get("name") if isinstance(landmark, dict) else landmark
            if name:
                nested.append(str(name))
    if nested:
        return nested
    return [label["name"] for label in ordered]


def flip_pairs(names) -> list[int]:
    """Index pairs that swap under a horizontal flip, flattened as RF-DETR wants.

    Returns ``[]`` when nothing pairs up, which is the honest answer for a
    schema with no left and right in it -- a single fin, a centre line -- and
    which leaves flipping to swap nothing rather than to swap wrongly.
    """
    index = {name: position for position, name in enumerate(names)}
    pairs: list[int] = []
    seen: set[int] = set()
    for position, name in enumerate(names):
        if position in seen:
            continue
        partner = _mirror_partner(name)
        if partner is None:
            continue
        # Matched case-insensitively: a schema mixing "Left_eye" with
        # "right_eye" is a typo, not two unrelated landmarks.
        match = next(
            (
                other
                for other in index
                if other.lower() == partner.lower() and index[other] != position
            ),
            None,
        )
        if match is not None:
            pairs.extend([position, index[match]])
            seen.update({position, index[match]})
    return pairs


def coco_category(names, category_name: str = DEFAULT_CATEGORY) -> dict:
    """The one category a keypoint project writes, with its schema attached.

    ``skeleton`` is deliberately empty. It is a drawing instruction -- which
    landmarks to join with a line -- and nothing in training reads it:
    RF-DETR's schema takes names, counts, sigmas and flip pairs. Writing an
    invented skeleton would put a claim about anatomy in a training file to no
    effect.
    """
    return {
        "id": 1,
        "name": category_name,
        "supercategory": "none",
        "keypoints": list(names),
        "skeleton": [],
    }


def _visibility(shape) -> int:
    raw = shape.get("visible", shape.get("visibility"))
    if raw is None:
        return VISIBLE
    if isinstance(raw, bool):
        return VISIBLE if raw else OCCLUDED
    if isinstance(raw, str):
        raw = raw.strip().lower()
    return _VISIBILITY.get(raw, VISIBLE)


def _point_of(shape):
    # AnyLabeling/LabelMe call this ``points``; AnyLearning's canvas has a
    # dedicated Dot shape and serialises the same coordinate as ``position``.
    # Accept both at the boundary so imports and hand-labelled projects share
    # one representation from here onward.
    position = shape.get("position")
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        points = [position]
    else:
        points = shape.get("points") or []
    if not points:
        return None
    first = points[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        return None
    try:
        return float(first[0]), float(first[1])
    except (TypeError, ValueError):
        return None


def _label_of(shape):
    categories = shape.get("categories")
    if isinstance(categories, list):
        return categories[0] if categories else None
    return categories


def instances(annotation, names) -> list[dict]:
    """Group an image's point shapes into COCO keypoint instances.

    Each instance is ``{"keypoints": [x, y, v] * len(names), "num_keypoints":
    n, "bbox": [x, y, w, h]}``, with unplaced landmarks written as ``0, 0, 0``.
    The flat triplet list is COCO's format and RF-DETR reads it directly.

    An instance whose landmarks are all unplaced is dropped: it carries no
    supervision, and its box would be a point at the origin.
    """
    if not names:
        return []
    index = {name: position for position, name in enumerate(names)}
    grouped: dict[object, list] = {}
    built = []

    for shape in (annotation or {}).get("data") or []:
        if not isinstance(shape, dict):
            continue
        shape_type = (shape.get("type") or "").lower()
        # The original keypoint canvas serialized a whole pose as one
        # ``keypoints`` shape: points and visibility arrays share the schema's
        # order. Preserve support for those projects alongside the newer
        # one-point-per-shape representation below.
        if shape_type == "keypoints":
            points = shape.get("points") or []
            visibility = shape.get("visibility") or []
            triplets = [0.0, 0.0, NOT_LABELLED] * len(names)
            placed = {}
            for position, point in enumerate(points[: len(names)]):
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    x, y = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    continue
                visible = _visibility(
                    {"visibility": visibility[position]}
                    if position < len(visibility)
                    else {}
                )
                triplets[position * 3 : position * 3 + 3] = [x, y, visible]
                if visible != NOT_LABELLED:
                    placed[position] = (x, y)
            if placed:
                xs = [x for x, _ in placed.values()]
                ys = [y for _, y in placed.values()]
                built.append(
                    {
                        "keypoints": triplets,
                        "num_keypoints": len(placed),
                        "bbox": _box(xs, ys),
                    }
                )
            continue
        if shape_type not in {"dot", "point", "keypoint"}:
            continue
        position = index.get(_label_of(shape))
        if position is None:
            continue
        point = _point_of(shape)
        if point is None:
            continue
        # `None` is a real key here: it is the group every ungrouped point
        # belongs to, and it must not collide with group 0.
        grouped.setdefault(shape.get("group_id"), []).append(
            (position, point, _visibility(shape))
        )

    # Ungrouped points last, so that numbering follows the groups a user made.
    for key in sorted(grouped, key=_group_sort_key):
        triplets = [0.0, 0.0, NOT_LABELLED] * len(names)
        placed: dict[int, tuple[float, float]] = {}
        for position, (x, y), visibility in grouped[key]:
            triplets[position * 3 : position * 3 + 3] = [x, y, visibility]
            if visibility != NOT_LABELLED:
                placed[position] = (x, y)
            else:
                placed.pop(position, None)
        if not placed:
            continue
        xs = [x for x, _ in placed.values()]
        ys = [y for _, y in placed.values()]
        built.append(
            {
                "keypoints": triplets,
                "num_keypoints": len(placed),
                "bbox": _box(xs, ys),
            }
        )
    return built


def _group_sort_key(group_id):
    """Order numeric groups naturally, text groups safely, and implicit last."""
    if group_id is None:
        return (1, 0, 0)
    if isinstance(group_id, (int, float)) and not isinstance(group_id, bool):
        return (0, 0, group_id)
    return (0, 1, str(group_id))


def _box(xs, ys) -> list[float]:
    """A box around the landmarks, never zero-sized.

    RF-DETR trains a detector alongside the landmarks, and its loss divides by
    the box: a single visible point, or a row of collinear ones, produces a box
    with no width or no height and takes the run down with a non-finite loss.
    The padding is a minimum rather than a margin -- it applies only to the
    degenerate dimension, so a normal box is untouched.
    """
    minimum = 1.0
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    if width < minimum:
        x_min -= (minimum - width) / 2
        width = minimum
    if height < minimum:
        y_min -= (minimum - height) / 2
        height = minimum
    return [x_min, y_min, width, height]

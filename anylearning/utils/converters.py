import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

from anylearning.training import keypoints


def convert_anylabeling_to_anylearning(anylabeling_json):
    """
    Convert a single image label from AnyLabeling (LabelMe) format to AnyLearning format.

    Args:
    anylabeling_json (dict): The input JSON in AnyLabeling format.

    Returns:
    list: A list of converted objects in AnyLearning format.
    """
    converted_objects = []

    for shape in anylabeling_json.get("shapes", []):
        new_obj = {
            "id": len(converted_objects) + 1,  # Use incremental ID
            "points": [],
            "phi": 0,  # Assuming no rotation by default
            "categories": [shape.get("label", "")],
            "type": shape.get("shape_type", "").lower(),
            # LabelMe/AnyLabeling use this to say which landmarks belong to
            # the same subject. Dropping it merges every person in an image
            # into one impossible skeleton.
            "group_id": shape.get("group_id"),
        }

        flags = shape.get("flags") or {}
        visibility = shape.get("visible", flags.get("visibility"))
        if visibility is not None:
            new_obj["visible"] = visibility

        # Convert points based on shape type
        try:
            if new_obj["type"] == "rectangle":
                top_left, bottom_right = shape["points"]
                new_obj["points"] = [
                    top_left,
                    [bottom_right[0], top_left[1]],  # Top-right
                    bottom_right,
                    [top_left[0], bottom_right[1]],  # Bottom-left
                ]
            elif new_obj["type"] in ["polygon", "polyline"]:
                new_obj["points"] = shape["points"]
            elif new_obj["type"] == "point":
                new_obj["points"] = [shape["points"][0]]
        except Exception as e:
            print(f"Error converting shape to AnyLearning format: {e}")
            continue

        converted_objects.append(new_obj)

    return converted_objects


def convert_anylearning_to_yolo(anylearning_json, labels, image_size):
    """
    Convert AnyLearning format to YOLO format.

    Args:
    anylearning_json (dict or list): The input JSON in AnyLearning format.
    labels (list): List of label dictionaries with 'name' and 'id' keys.
    image_size (tuple): The size of the image (width, height).
    Returns:
    str: A string in YOLO format.
    """
    yolo_annotations = []

    # Check if anylearning_json is None or empty
    if not anylearning_json:
        return ""

    # Ensure we're working with a list
    if isinstance(anylearning_json, dict):
        anylearning_objects = [anylearning_json]
    else:
        anylearning_objects = anylearning_json

    # Create a mapping of label names to their IDs
    try:
        label_map = {label["name"]: label["id"] for label in labels}
    except (TypeError, KeyError) as e:
        logging.error(f"Error processing labels: {e}")
        logging.error(f"Labels: {labels}")
        return ""

    img_width, img_height = image_size

    for obj in anylearning_objects:
        try:
            # Skip any non-dictionary objects
            if not isinstance(obj, dict):
                logging.warning(f"Skipping non-dictionary object: {obj}")
                continue

            # Safe category extraction
            category = None
            if "categories" in obj:
                categories = obj["categories"]
                if isinstance(categories, list) and len(categories) > 0:
                    category = categories[0]
                elif isinstance(categories, str):
                    category = categories
                else:
                    logging.warning(f"Unsupported categories format: {categories}")
                    continue
            else:
                logging.warning(f"No categories field in object: {obj}")
                continue

            if not isinstance(category, str):
                logging.warning(f"Category is not a string: {category}")
                continue

            if category not in label_map:
                logging.warning(f"Category not found in label map: {category}")
                continue

            label_id = label_map[category]

            # Skip objects with no points
            if "points" not in obj:
                logging.warning(f"No points in object: {obj}")
                continue

            points = obj["points"]

            # Skip empty point arrays
            if not points or not isinstance(points, list):
                logging.warning(f"Empty or invalid points: {points}")
                continue

            if "type" not in obj:
                logging.warning(f"No type specified in object: {obj}")
                continue

            obj_type = obj["type"]

            # Polygons are boxed, not dropped. YOLO only expresses axis-aligned
            # boxes, so any closed shape converts to its bounding box. Previously
            # only "rectangle" was handled and everything else fell through to a
            # log warning, which meant importing polygon annotations into a
            # detection project produced an *empty* label file -- the training
            # run then silently learned from images with no objects.
            if obj_type in ("rectangle", "polygon", "polyline"):
                try:
                    # Check if we have enough points
                    if len(points) < 3:
                        logging.warning(f"Not enough points for {obj_type}: {points}")
                        continue

                    # Make sure points are valid
                    valid_points = True
                    for point in points:
                        if not isinstance(point, list) or len(point) < 2:
                            valid_points = False
                            break

                    if not valid_points:
                        logging.warning(
                            f"Invalid points format for rectangle: {points}"
                        )
                        continue

                    x_min = min(point[0] for point in points if len(point) >= 2)
                    y_min = min(point[1] for point in points if len(point) >= 2)
                    x_max = max(point[0] for point in points if len(point) >= 2)
                    y_max = max(point[1] for point in points if len(point) >= 2)

                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    yolo_annotations.append(
                        f"{label_id} {x_center} {y_center} {width} {height}"
                    )
                except Exception as e:
                    logging.error(f"Error converting rectangle to YOLO format: {e}")
                    logging.error(f"Points: {points}")
                    continue
        except Exception as e:
            logging.error(f"Error processing object: {e}")
            logging.error(f"Object: {obj}")
            continue

    return "\n".join(yolo_annotations)


def convert_anylearning_to_coco(
    anylearning_objects: List[Dict],
    labels: List[Dict],
    image_id: int,
    image_filename: str,
    image_size: Tuple[int, int],
    keypoint_names: List[str] | None = None,
) -> Dict:
    """
    Convert AnyLearning format to COCO format for a single image.

    Args:
    anylearning_objects (list): List of objects in AnyLearning format
    labels (list): List of label dictionaries with 'name' and 'id' keys
    image_id (int): ID for the image in COCO format
    image_filename (str): Name of the image file
    image_size (tuple): Size of the image as (width, height)

    Returns:
    dict: COCO format annotations for the image
    """
    if not anylearning_objects:
        return []

    # Ensure we're working with a list
    if isinstance(anylearning_objects, dict):
        anylearning_objects = [anylearning_objects]

    if keypoint_names is not None:
        annotations = []
        for annotation_id, instance in enumerate(
            keypoints.instances({"data": anylearning_objects}, keypoint_names), 1
        ):
            bbox = [float(value) for value in instance["bbox"]]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "segmentation": [],
                    "iscrowd": 0,
                    "keypoints": instance["keypoints"],
                    "num_keypoints": instance["num_keypoints"],
                }
            )
        return annotations

    # Create label map for quick lookup
    label_map = {}
    for label in labels:
        try:
            label_map[label["name"]] = label["id"]
        except (KeyError, TypeError) as e:
            logging.error(f"Error processing label: {e}")
            continue

    img_width, img_height = image_size
    coco_annotations = []
    annotation_id = 1

    for obj in anylearning_objects:
        try:
            # Skip invalid objects
            if not isinstance(obj, dict):
                continue

            # Get category
            if "categories" not in obj:
                continue

            categories = obj["categories"]
            category = None

            if isinstance(categories, list) and len(categories) > 0:
                category = categories[0]
            elif isinstance(categories, str):
                category = categories
            else:
                continue

            if not category or category not in label_map:
                continue

            category_id = label_map[category]

            # Get points
            if "points" not in obj or not obj["points"]:
                continue

            points = obj["points"]
            obj_type = obj.get("type", "").lower()

            # Handle different shape types
            if obj_type == "rectangle":
                # Convert rectangle to bbox format [x, y, width, height]
                try:
                    # Calculate bounding box
                    x_coords = [
                        p[0] for p in points if isinstance(p, list) and len(p) >= 2
                    ]
                    y_coords = [
                        p[1] for p in points if isinstance(p, list) and len(p) >= 2
                    ]

                    if not x_coords or not y_coords:
                        continue

                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)

                    width = x_max - x_min
                    height = y_max - y_min

                    area = width * height

                    coco_ann = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x_min, y_min, width, height],
                        "area": area,
                        "segmentation": [],
                        "iscrowd": 0,
                    }

                    coco_annotations.append(coco_ann)
                    annotation_id += 1
                except Exception as e:
                    logging.error(f"Error converting rectangle to COCO: {e}")
                    continue

            elif obj_type in ["polygon", "polyline"]:
                # Convert polygon to COCO segmentation format
                try:
                    # Flatten the points list for COCO [[x1,y1,x2,y2,...]]
                    flat_points = []
                    for point in points:
                        if isinstance(point, list) and len(point) >= 2:
                            flat_points.extend([point[0], point[1]])

                    if len(flat_points) < 6:  # Need at least 3 points for a polygon
                        continue

                    # Calculate the bounding box from the polygon points
                    x_coords = [flat_points[i] for i in range(0, len(flat_points), 2)]
                    y_coords = [flat_points[i] for i in range(1, len(flat_points), 2)]

                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)

                    width = x_max - x_min
                    height = y_max - y_min

                    # Polygon area by the shoelace formula:
                    #   0.5 * |sum(x_i * y_i+1 - x_i+1 * y_i)|
                    #
                    # The previous implementation took abs() of each term rather
                    # than of the sum, and stopped before the closing edge, so
                    # every polygon area came out too large -- a 100x60 box
                    # reported 9500 instead of 6000. COCO buckets AP into
                    # small/medium/large by this value, so the error showed up as
                    # skewed evaluation metrics rather than an exception.
                    vertices = [
                        (flat_points[i], flat_points[i + 1])
                        for i in range(0, len(flat_points) - 1, 2)
                    ]
                    cross_sum = 0.0
                    for index, (x1, y1) in enumerate(vertices):
                        x2, y2 = vertices[(index + 1) % len(vertices)]
                        cross_sum += x1 * y2 - x2 * y1
                    area = abs(cross_sum) / 2.0

                    coco_ann = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x_min, y_min, width, height],
                        "area": area,
                        "segmentation": [flat_points],
                        "iscrowd": 0,
                    }

                    coco_annotations.append(coco_ann)
                    annotation_id += 1
                except Exception as e:
                    logging.error(f"Error converting polygon to COCO: {e}")
                    continue

            # TODO: Handle other shape types as needed

        except Exception as e:
            logging.error(f"Error processing object for COCO: {e}")
            continue

    return coco_annotations


def convert_anylearning_to_labelme(
    anylearning_objects: List[Dict], image_filename: str, image_size: Tuple[int, int]
) -> Dict:
    """
    Convert AnyLearning format to LabelMe format.

    Args:
    anylearning_objects (list): List of objects in AnyLearning format
    image_filename (str): Name of the image file
    image_size (tuple): Size of the image as (width, height)

    Returns:
    dict: LabelMe format annotations for the image
    """
    img_width, img_height = image_size

    # Create basic LabelMe structure
    labelme_data = {
        "version": "4.5.13",
        "flags": {},
        "shapes": [],
        "imagePath": image_filename,
        "imageData": None,  # We don't include actual image data, just reference
        "imageHeight": img_height,
        "imageWidth": img_width,
    }

    # Ensure we're working with a list
    if not anylearning_objects:
        return labelme_data

    if isinstance(anylearning_objects, dict):
        anylearning_objects = [anylearning_objects]

    # Convert each object to LabelMe shape format
    for obj in anylearning_objects:
        try:
            # Skip invalid objects
            if not isinstance(obj, dict):
                continue

            # Get category
            if "categories" not in obj:
                continue

            categories = obj["categories"]
            category = None

            if isinstance(categories, list) and len(categories) > 0:
                category = categories[0]
            elif isinstance(categories, str):
                category = categories
            else:
                continue

            if not category:
                continue

            # Get points and type. The canvas serialises its point primitive as
            # a Dot with one ``position``; LabelMe calls it a point with a
            # one-element ``points`` array.
            obj_type = obj.get("type", "").lower()
            if obj_type == "dot":
                position = obj.get("position")
                points = [position] if position else []
                obj_type = "point"
            else:
                points = obj.get("points") or []
            if not points:
                continue

            # Map AnyLearning shape types to LabelMe
            shape_type = None

            if obj_type == "rectangle":
                shape_type = "rectangle"
                # LabelMe rectangles use just 2 points (top-left, bottom-right)
                try:
                    # Calculate extreme points
                    x_coords = [
                        p[0] for p in points if isinstance(p, list) and len(p) >= 2
                    ]
                    y_coords = [
                        p[1] for p in points if isinstance(p, list) and len(p) >= 2
                    ]

                    if not x_coords or not y_coords:
                        continue

                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)

                    # Use only two points for LabelMe rectangle
                    simplified_points = [[x_min, y_min], [x_max, y_max]]

                    labelme_shape = {
                        "label": category,
                        "points": simplified_points,
                        "group_id": obj.get("group_id"),
                        "shape_type": shape_type,
                        "flags": {},
                    }

                    labelme_data["shapes"].append(labelme_shape)
                except Exception as e:
                    logging.error(f"Error converting rectangle to LabelMe: {e}")
                    continue

            elif obj_type == "polygon":
                shape_type = "polygon"
                # Check if we have a valid polygon (at least 3 points)
                if len(points) < 3:
                    continue

                # Keep all points for polygon
                labelme_shape = {
                    "label": category,
                    "points": points,
                    "group_id": obj.get("group_id"),
                    "shape_type": shape_type,
                    "flags": {},
                }

                labelme_data["shapes"].append(labelme_shape)

            elif obj_type == "polyline":
                shape_type = "line"
                # Check if we have a valid line (at least 2 points)
                if len(points) < 2:
                    continue

                labelme_shape = {
                    "label": category,
                    "points": points,
                    "group_id": obj.get("group_id"),
                    "shape_type": shape_type,
                    "flags": {},
                }

                labelme_data["shapes"].append(labelme_shape)

            elif obj_type == "point":
                shape_type = "point"
                # Only need one point
                if len(points) < 1:
                    continue

                labelme_shape = {
                    "label": category,
                    "points": points[0:1],  # Just use the first point
                    "group_id": obj.get("group_id"),
                    "shape_type": shape_type,
                    "flags": (
                        {"visibility": obj["visible"]} if "visible" in obj else {}
                    ),
                }

                labelme_data["shapes"].append(labelme_shape)

        except Exception as e:
            logging.error(f"Error processing object for LabelMe: {e}")
            continue

    return labelme_data


def convert_anylearning_to_anylabeling(ann_objects, image_filename, image_size):
    """
    Convert from AnyLearning annotation format to AnyLabeling format.
    AnyLabeling format is based on LabelMe with additional fields.

    Args:
        ann_objects: List of annotation objects in AnyLearning format
        image_filename: Filename of the image (used in the output)
        image_size: Tuple of (width, height) of the image

    Returns:
        A dictionary in AnyLabeling format
    """
    # Start with LabelMe format as base
    labelme_data = convert_anylearning_to_labelme(
        ann_objects, image_filename, image_size
    )

    # Add AnyLabeling specific fields
    anylabeling_data = labelme_data.copy()
    anylabeling_data["version"] = "0.2.0"
    anylabeling_data["flags"] = {}
    anylabeling_data["lineColor"] = [0, 255, 0, 128]
    anylabeling_data["fillColor"] = [255, 0, 0, 128]
    anylabeling_data["imagePath"] = image_filename
    anylabeling_data["imageHeight"] = image_size[1]
    anylabeling_data["imageWidth"] = image_size[0]
    anylabeling_data["imageData"] = None

    # Use the current time for lastSaved field
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    anylabeling_data["lastSaved"] = current_time

    return anylabeling_data


def convert_coco_to_anylearning(coco_json: Dict) -> Dict[str, List[Dict]]:
    """COCO in. Returns annotations keyed by image file name.

    The counterpart of convert_anylearning_to_coco, so a dataset exported from
    here can be read back in -- which is the round trip people assume exists and
    did not. COCO keeps one file for the whole dataset rather than a sidecar per
    image, so this returns the whole mapping at once.

    Segmentation is preferred over the bounding box when both are present: a
    polygon carries strictly more information, and a project that has masks
    wants them.
    """
    categories = {
        category["id"]: category
        for category in coco_json.get("categories", [])
        if "id" in category
    }
    images = {
        image["id"]: os.path.basename(image.get("file_name", ""))
        for image in coco_json.get("images", [])
        if "id" in image
    }

    by_image: Dict[str, List[Dict]] = {}
    for annotation in coco_json.get("annotations", []):
        file_name = images.get(annotation.get("image_id"))
        if not file_name:
            continue
        category = categories.get(annotation.get("category_id"))
        if category is None:
            continue

        category_keypoints = category.get("keypoints") or []
        flat_keypoints = annotation.get("keypoints") or []
        if category_keypoints and len(flat_keypoints) >= 3:
            objects = by_image.setdefault(file_name, [])
            group_id = annotation.get("id", len(objects) + 1)
            for index, name in enumerate(category_keypoints):
                offset = index * 3
                if offset + 2 >= len(flat_keypoints):
                    break
                x, y, visibility = flat_keypoints[offset : offset + 3]
                if not visibility:
                    continue
                objects.append(
                    {
                        "id": len(objects) + 1,
                        "position": [x, y],
                        "phi": 0,
                        "categories": [name],
                        "type": "dot",
                        "group_id": group_id,
                        "visible": visibility,
                    }
                )
            continue

        label = category.get("name", str(annotation.get("category_id")))

        points: List[List[float]] = []
        shape_type = "rectangle"

        segmentation = annotation.get("segmentation") or []
        # RLE masks arrive as a dict rather than a list of polygons. Decoding
        # them needs pycocotools and a real mask -> polygon step; skipping to
        # the bounding box keeps the import working and loses only precision.
        if isinstance(segmentation, list) and segmentation:
            flat = segmentation[0]
            if isinstance(flat, list) and len(flat) >= 6:
                points = [
                    [flat[index], flat[index + 1]]
                    for index in range(0, len(flat) - 1, 2)
                ]
                shape_type = "polygon"

        if not points:
            bbox = annotation.get("bbox") or []
            if len(bbox) != 4:
                continue
            left, top, width, height = bbox
            points = [
                [left, top],
                [left + width, top],
                [left + width, top + height],
                [left, top + height],
            ]

        objects = by_image.setdefault(file_name, [])
        objects.append(
            {
                "id": len(objects) + 1,
                "points": points,
                "phi": 0,
                "categories": [label],
                "type": shape_type,
            }
        )

    return by_image


def convert_yolo_to_anylearning(
    yolo_text: str, class_names: List[str], image_size: Tuple[int, int]
) -> List[Dict]:
    """One YOLO label file in, for one image.

    YOLO stores coordinates normalised to the image, so the image size is not
    optional -- without it every box would land in the top-left corner. Both
    layouts are read: `class cx cy w h` for boxes, and `class x1 y1 x2 y2 ...`
    for segmentation polygons, which is what YOLO-seg writes.
    """
    width, height = image_size
    objects: List[Dict] = []

    for line in (yolo_text or "").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_index = int(float(parts[0]))
            numbers = [float(value) for value in parts[1:]]
        except ValueError:
            logging.warning(f"Skipping unreadable YOLO line: {line!r}")
            continue

        if class_index < 0 or class_index >= len(class_names):
            logging.warning(f"YOLO class {class_index} is not in the class list")
            continue
        label = class_names[class_index]

        if len(numbers) == 4:
            centre_x, centre_y, box_width, box_height = numbers
            left = (centre_x - box_width / 2) * width
            top = (centre_y - box_height / 2) * height
            right = (centre_x + box_width / 2) * width
            bottom = (centre_y + box_height / 2) * height
            points = [[left, top], [right, top], [right, bottom], [left, bottom]]
            shape_type = "rectangle"
        elif len(numbers) >= 6 and len(numbers) % 2 == 0:
            points = [
                [numbers[index] * width, numbers[index + 1] * height]
                for index in range(0, len(numbers), 2)
            ]
            shape_type = "polygon"
        else:
            logging.warning(f"Skipping YOLO line with {len(numbers)} coordinates")
            continue

        objects.append(
            {
                "id": len(objects) + 1,
                "points": points,
                "phi": 0,
                "categories": [label],
                "type": shape_type,
            }
        )

    return objects

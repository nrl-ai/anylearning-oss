"""The packaged self-test can create a real keypoint training dataset."""

import io
import zipfile

from anylearning.selftest import synthetic
from anylearning.training import keypoints


def test_keypoint_subset_has_a_complete_schema_per_subject():
    names = ["left_eye", "right_eye", "nose"]
    archive, annotations = synthetic.build_subset("keypoint", 2, 64, 7, names)

    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        image_names = sorted(zipped.namelist())

    assert image_names == ["keypoint_000.png", "keypoint_001.png"]
    for image_name in image_names:
        shapes = annotations[image_name]["shapes"]
        assert shapes
        assert all(shape["type"] == "dot" for shape in shapes)
        assert all(shape["visible"] == 2 for shape in shapes)

        instances = keypoints.instances({"data": shapes}, names)
        assert instances
        assert all(instance["num_keypoints"] == len(names) for instance in instances)
        assert all(instance["bbox"][2] > 0 for instance in instances)
        assert all(instance["bbox"][3] > 0 for instance in instances)

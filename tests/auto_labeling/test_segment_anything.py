import cv2
import numpy as np

from anylearning.auto_labeling.segment_anything import SegmentAnything


def test_rectangle_post_process_returns_no_shape_for_an_empty_mask():
    model = object.__new__(SegmentAnything)
    model.output_mode = "rectangle"

    assert model.post_process(np.zeros((16, 16), dtype=np.float32)) == []


def test_contour_filter_removes_full_image_background_when_objects_exist():
    mask = np.ones((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 0
    mask[35:65, 35:65] = 1

    contours = SegmentAnything._contours(mask)

    assert all(cv2.contourArea(contour) < 9000 for contour in contours)

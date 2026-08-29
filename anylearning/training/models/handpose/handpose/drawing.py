"""Hand-landmark drawing, independent of mediapipe's removed `solutions` module.

mediapipe 1.0 dropped the legacy `mediapipe.solutions` package (drawing_utils,
drawing_styles, hands) in favour of the Tasks API. Detection had already moved to
Tasks -- `HandLandmarker.detect()` -- so only the visualisation helpers were left
behind, and they are simple enough to own outright rather than pin the whole
project to mediapipe 0.10.x (which caps numpy below 2 and ships no cp313 wheel).

The connection topology below is MediaPipe's standard 21-point hand model, and
the colours match `get_default_hand_landmarks_style()` closely enough that
existing screenshots still look right.
"""

import cv2

# MediaPipe's standard 21-landmark hand skeleton, as (start, end) index pairs.
#   0 wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm closure
)

_LANDMARK_COLOR = (0, 0, 255)      # BGR red, as in the default landmark style
_CONNECTION_COLOR = (224, 224, 224)  # BGR near-white, as in the default connection style
_LANDMARK_RADIUS = 3
_CONNECTION_THICKNESS = 2


def draw_hand_landmarks(image, landmarks):
    """Draw one hand's landmarks and skeleton onto ``image`` in place.

    Args:
        image: HxWx3 BGR array, modified in place.
        landmarks: sequence of 21 objects with normalised ``.x`` / ``.y`` in
            [0, 1], as returned by mediapipe Tasks' HandLandmarker.

    Returns the image, for convenience.
    """
    height, width = image.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks
    ]

    # Connections first so the joints sit on top of the bones.
    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(
                image, points[start], points[end], _CONNECTION_COLOR, _CONNECTION_THICKNESS
            )

    for point in points:
        cv2.circle(image, point, _LANDMARK_RADIUS, _LANDMARK_COLOR, -1)

    return image

"""Where a point in the window lands: a resize edge, the title bar, or the page.

This is the whole behaviour of the Windows custom frame expressed as
arithmetic, deliberately kept out of `win32.py` and free of ctypes: it is the
part that decides how the window feels, and the part that can be tested on a
machine that is not Windows.

Coordinates are client pixels -- physical, not CSS -- with the origin at the
top-left of the web view, which is where WM_NCHITTEST leaves us after
`ScreenToClient`.
"""

from __future__ import annotations

from typing import Sequence

Rect = Sequence[float]
"""(left, top, right, bottom) in client pixels."""

CLIENT = "client"
CAPTION = "caption"
LEFT = "left"
RIGHT = "right"
TOP = "top"
BOTTOM = "bottom"
TOP_LEFT = "topleft"
TOP_RIGHT = "topright"
BOTTOM_LEFT = "bottomleft"
BOTTOM_RIGHT = "bottomright"


def _contains(rect: Rect, x: float, y: float) -> bool:
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom


def in_caption(
    x: float, y: float, regions: Sequence[Rect], exclusions: Sequence[Rect]
) -> bool:
    """Is (x, y) on a title-bar surface rather than on something inside one?

    `exclusions` are the child elements of each drag region. Excluding them is
    what keeps a button that happens to sit in the title bar clickable, and it
    mirrors pywebview's own DRAG_REGION_DIRECT_TARGET_ONLY rule so the two
    drag paths agree about what is draggable.
    """
    if not any(_contains(region, x, y) for region in regions):
        return False
    return not any(_contains(exclusion, x, y) for exclusion in exclusions)


def hit_test(
    x: float,
    y: float,
    width: float,
    height: float,
    border: float,
    regions: Sequence[Rect] = (),
    exclusions: Sequence[Rect] = (),
    maximized: bool = False,
) -> str:
    """Classify a point in a client area of `width` x `height`.

    Edges win over the title bar: the top border of the window overlaps the
    top of the title bar, and a user aiming at a 4px resize strip has to be
    able to hit it. A maximised window has no edges to grab -- Windows resizes
    it by un-maximising -- so the border is skipped there.
    """
    if x < 0 or y < 0 or x >= width or y >= height:
        return CLIENT

    if not maximized and border > 0:
        left = x < border
        right = x >= width - border
        top = y < border
        bottom = y >= height - border

        if top and left:
            return TOP_LEFT
        if top and right:
            return TOP_RIGHT
        if bottom and left:
            return BOTTOM_LEFT
        if bottom and right:
            return BOTTOM_RIGHT
        if left:
            return LEFT
        if right:
            return RIGHT
        if top:
            return TOP
        if bottom:
            return BOTTOM

    if in_caption(x, y, regions, exclusions):
        return CAPTION

    return CLIENT

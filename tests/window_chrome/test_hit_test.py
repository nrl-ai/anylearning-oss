"""The Windows custom frame, minus Windows.

`anylearning/window_chrome/win32.py` cannot run here, but the arithmetic that
decides how its window behaves can: every resize edge, the title bar, and the
controls sitting inside the title bar that must stay clickable. These are the
cases that break a frameless window in ways a screenshot would not show.

The window in these tests is 1000 x 800 with a 8px grab border and a title bar
across the top 60px whose right end holds a 100px-wide group of buttons.
"""

import pytest

from anylearning.window_chrome.hit_test import (
    BOTTOM,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
    CAPTION,
    CLIENT,
    LEFT,
    RIGHT,
    TOP,
    TOP_LEFT,
    TOP_RIGHT,
    hit_test,
)

WIDTH = 1000
HEIGHT = 800
BORDER = 8
TITLE_BAR = [(0, 0, WIDTH, 60)]
CONTROLS = [(880, 14, 980, 46)]


def zone(x, y, **overrides):
    kwargs = {
        "width": WIDTH,
        "height": HEIGHT,
        "border": BORDER,
        "regions": TITLE_BAR,
        "exclusions": CONTROLS,
    }
    kwargs.update(overrides)
    return hit_test(x, y, **kwargs)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (2, 2, TOP_LEFT),
        (WIDTH - 2, 2, TOP_RIGHT),
        (2, HEIGHT - 2, BOTTOM_LEFT),
        (WIDTH - 2, HEIGHT - 2, BOTTOM_RIGHT),
        (2, 400, LEFT),
        (WIDTH - 2, 400, RIGHT),
        (500, 2, TOP),
        (500, HEIGHT - 2, BOTTOM),
    ],
)
def test_the_frame_is_grabbable_on_every_edge_and_corner(x, y, expected):
    assert zone(x, y) == expected


def test_the_top_edge_wins_over_the_title_bar():
    """Both cover y < 8. Losing the resize strip is worse than losing 8px of drag."""
    assert zone(500, 4) == TOP
    assert zone(500, 9) == CAPTION


def test_the_title_bar_drags_the_window():
    assert zone(400, 30) == CAPTION


def test_controls_inside_the_title_bar_stay_clickable():
    """Reported as the caption, the buttons would drag the window, not click."""
    assert zone(900, 30) == CLIENT


def test_the_gap_between_controls_and_the_edge_still_drags():
    assert zone(985, 30) == CAPTION


def test_the_page_below_the_title_bar_is_the_page():
    assert zone(400, 300) == CLIENT


def test_a_maximised_window_has_no_resize_edges():
    """Windows resizes a maximised window by un-maximising it, not by the edge."""
    assert zone(2, 400, maximized=True) == CLIENT
    assert zone(500, 2, maximized=True) == CAPTION


def test_points_outside_the_client_area_are_left_alone():
    assert zone(-5, 30) == CLIENT
    assert zone(WIDTH + 5, 30) == CLIENT


def test_a_window_with_no_title_bar_reported_yet_is_all_page():
    """The regions arrive from the page after it renders; until then, nothing drags."""
    assert zone(400, 30, regions=(), exclusions=()) == CLIENT
    assert zone(2, 400, regions=(), exclusions=()) == LEFT

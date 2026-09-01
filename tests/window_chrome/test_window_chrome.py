"""The title bar's Python half, without a window manager in sight.

What is worth pinning here is the state the buttons render from. It is easy to
get right while clicking the app's own maximise button and wrong the moment
the user uses the window manager instead -- minimise a maximised window and
restore it, and a naive reading of pywebview's events says it is no longer
maximised.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from anylearning.window_chrome import WindowChrome


class FakeEvent:
    """pywebview's Event, as far as this code uses it."""

    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def fire(self):
        for handler in self.handlers:
            handler()


class FakeWindow:
    def __init__(self):
        self.exposed = []
        self.scripts = []
        self.native = None
        self.maximize = MagicMock()
        self.restore = MagicMock()
        self.minimize = MagicMock()
        self.destroy = MagicMock()
        self.events = SimpleNamespace(
            shown=FakeEvent(), maximized=FakeEvent(), restored=FakeEvent()
        )

    def expose(self, *functions):
        self.exposed.extend(functions)

    def evaluate_js(self, script):
        self.scripts.append(script)


@pytest.fixture
def chrome():
    return WindowChrome(FakeWindow()).attach()


def test_the_title_bar_api_is_exposed_to_the_page(chrome):
    exposed = {function.__name__ for function in chrome.window.exposed}

    assert exposed == {
        "window_chrome_state",
        "window_minimize",
        "window_toggle_maximize",
        "window_close",
        "window_begin_drag",
        "window_begin_resize",
        "window_set_drag_regions",
    }


def test_maximise_round_trips(chrome):
    chrome.window_toggle_maximize()
    assert chrome.maximized is True
    chrome.window.maximize.assert_called_once()

    chrome.window_toggle_maximize()
    assert chrome.maximized is False
    chrome.window.restore.assert_called_once()


def test_the_window_manager_gets_the_last_word(chrome):
    """Maximised from the window manager, with no button of ours pressed."""
    chrome.window.events.maximized.fire()

    assert chrome.maximized is True
    assert chrome.window_chrome_state()["maximized"] is True


def test_state_tells_the_page_when_the_platform_owns_the_frame():
    chrome = WindowChrome(FakeWindow(), native_frame=True).attach()

    assert chrome.window_chrome_state() == {
        "maximized": False,
        "native_frame": True,
    }


def test_un_minimising_a_maximised_window_leaves_it_maximised(chrome):
    """`restored` fires for un-minimising too, so the event alone is not evidence."""
    chrome.native.is_maximized = lambda: True
    chrome.window.events.restored.fire()

    assert chrome.maximized is True


def test_restoring_a_window_the_platform_cannot_report_believes_the_event(chrome):
    chrome._set_maximized(True)
    chrome.window.events.restored.fire()

    assert chrome.maximized is False


def test_state_changes_reach_the_page(chrome):
    chrome.window.events.maximized.fire()

    assert any("anylearning:window-state" in script for script in chrome.window.scripts)
    assert any('"maximized": true' in script for script in chrome.window.scripts)


def test_drag_regions_reach_the_platform(chrome):
    chrome.native.set_drag_regions = MagicMock()

    chrome.window_set_drag_regions([[0, 0, 100, 60]], [[10, 10, 20, 20]])

    chrome.native.set_drag_regions.assert_called_once_with(
        [[0, 0, 100, 60]], [[10, 10, 20, 20]]
    )


def test_a_platform_with_no_native_drag_says_so(chrome):
    """False is the frontend's cue to leave the drag to pywebview's own JS."""
    assert chrome.window_begin_drag(10, 10) is False
    assert chrome.window_begin_resize("left", 10, 10) is False

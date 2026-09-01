"""Qt window gestures without requiring a running GUI in the test suite."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from anylearning.window_chrome import qt
from anylearning.window_chrome.base import NativeChrome


class Signal:
    def __init__(self):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


def fake_qt():
    edge = SimpleNamespace(
        TopEdge=1,
        LeftEdge=2,
        RightEdge=4,
        BottomEdge=8,
    )
    return SimpleNamespace(Qt=SimpleNamespace(Edge=edge))


def test_move_and_resize_are_queued_as_native_gestures(monkeypatch):
    bridge = SimpleNamespace(
        move_requested=Signal(), resize_requested=Signal(), regions=(), exclusions=()
    )
    window = SimpleNamespace(native=SimpleNamespace(isMaximized=lambda: True))
    chrome = qt.QtChrome(window, bridge)
    monkeypatch.setattr(qt, "_qt", fake_qt)

    assert chrome.begin_drag(100, 200) is True
    assert chrome.begin_resize("topleft", 100, 200) is True
    assert chrome.is_maximized() is True
    assert bridge.move_requested.values == [()]
    assert bridge.resize_requested.values == [(3,)]


def test_drag_regions_are_copied_for_gui_thread_hit_testing():
    bridge = SimpleNamespace(
        move_requested=Signal(), resize_requested=Signal(), regions=(), exclusions=()
    )
    chrome = qt.QtChrome(SimpleNamespace(native=None), bridge)

    regions = [[0, 0, 800, 48]]
    exclusions = [[700, 8, 790, 40]]
    chrome.set_drag_regions(regions, exclusions)
    regions[0][2] = 0
    exclusions.clear()

    assert bridge.regions == ((0, 0, 800, 48),)
    assert bridge.exclusions == ((700, 8, 790, 40),)


def test_unknown_resize_edge_is_not_consumed(monkeypatch):
    bridge = SimpleNamespace(move_requested=Signal(), resize_requested=Signal())
    chrome = qt.QtChrome(SimpleNamespace(native=None), bridge)
    monkeypatch.setattr(qt, "_qt", fake_qt)

    assert chrome.begin_resize("middle", 0, 0) is False
    assert bridge.resize_requested.values == []


def test_install_falls_back_until_qt_has_a_window_handle(monkeypatch):
    window = SimpleNamespace(
        native=SimpleNamespace(windowHandle=MagicMock(return_value=None))
    )

    chrome = qt.install(window)

    assert type(chrome) is NativeChrome


def test_install_keeps_the_signal_bridge_alive(monkeypatch):
    handle = object()
    bridge = SimpleNamespace(
        move_requested=Signal(), resize_requested=Signal(), regions=(), exclusions=()
    )
    window = SimpleNamespace(
        native=SimpleNamespace(windowHandle=MagicMock(return_value=handle))
    )
    monkeypatch.setattr(qt, "_make_bridge", lambda candidate, native: bridge)
    install_filter = MagicMock(return_value=True)
    monkeypatch.setattr(qt, "_install_event_filter", install_filter)

    chrome = qt.install(window)

    assert isinstance(chrome, qt.QtChrome)
    assert chrome._gesture_bridge is bridge
    install_filter.assert_called_once_with(bridge)

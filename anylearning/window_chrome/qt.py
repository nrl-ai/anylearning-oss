"""Qt: hand frameless move and resize gestures to the window manager.

pywebview's generic drag region moves a window by sending every mousemove
through the JavaScript bridge and setting a new position. Besides being
visibly laggy, that cannot work on Wayland, where applications may not place
their own top-level windows. Qt exposes the native operations on ``QWindow``:
``startSystemMove`` and ``startSystemResize``.

The JavaScript API is served on worker threads. Qt window objects belong to the
GUI thread, so a small signal bridge queues both operations onto that thread.
"""

from __future__ import annotations

from loguru import logger

from anylearning.window_chrome.base import NativeChrome
from anylearning.window_chrome.hit_test import in_caption

EDGE_NAMES = {
    "top": ("TopEdge",),
    "topleft": ("TopEdge", "LeftEdge"),
    "topright": ("TopEdge", "RightEdge"),
    "left": ("LeftEdge",),
    "right": ("RightEdge",),
    "bottom": ("BottomEdge",),
    "bottomleft": ("BottomEdge", "LeftEdge"),
    "bottomright": ("BottomEdge", "RightEdge"),
}


def _qt():
    # Qt is an optional desktop dependency. Import it only after pywebview has
    # selected a Qt renderer, never while the server or another GUI is loading.
    from qtpy import QtCore

    return QtCore


def _make_bridge(handle, native):
    qt_core = _qt()

    class GestureBridge(qt_core.QObject):
        install_requested = qt_core.Signal()
        move_requested = qt_core.Signal()
        resize_requested = qt_core.Signal(object)

        def __init__(self):
            super().__init__()
            self.regions = ()
            self.exclusions = ()

        def _install(self) -> None:
            from qtpy import QtWidgets

            application = QtWidgets.QApplication.instance()
            if application is None:
                logger.warning("Qt application is unavailable; native hit testing is off")
                return
            application.installEventFilter(self)

        def _start_move(self) -> bool:
            started = bool(handle.startSystemMove())
            if not started:
                logger.warning("Qt rejected the native window move gesture")
            return started

        def _start_resize(self, edge) -> bool:
            started = bool(handle.startSystemResize(edge))
            if not started:
                logger.warning("Qt rejected the native window resize gesture")
            return started

        def eventFilter(self, watched, event):  # noqa: N802 - Qt API spelling
            """Start the move inside Qt's physical mouse-press event.

            Calling ``startSystemMove`` later through the JavaScript bridge is
            too late on a number of Wayland compositors. The DOM still defines
            the precise drag surfaces; this filter only moves the call to the
            GUI event that authorises the compositor gesture.
            """
            try:
                if event.type() != qt_core.QEvent.Type.MouseButtonPress:
                    return False
                if event.button() != qt_core.Qt.MouseButton.LeftButton:
                    return False
                top_level = watched.window()
                if top_level is not native and top_level.windowHandle() is not handle:
                    return False
                point = handle.mapFromGlobal(event.globalPosition().toPoint())
                if not in_caption(point.x(), point.y(), self.regions, self.exclusions):
                    return False
                return self._start_move()
            except (AttributeError, RuntimeError):
                # Not every QApplication mouse receiver is a QWidget. Those
                # events simply belong to something other than this webview.
                return False

    bridge = GestureBridge()
    queued = qt_core.Qt.ConnectionType.QueuedConnection
    # pywebview emits `shown` outside Qt's GUI thread. The filter and every
    # slot touching QWindow must live on the same thread as the native widget.
    bridge.moveToThread(native.thread())
    bridge.install_requested.connect(bridge._install, queued)
    bridge.move_requested.connect(bridge._start_move, queued)
    bridge.resize_requested.connect(bridge._start_resize, queued)
    return bridge


def _install_event_filter(bridge) -> bool:
    # Queue the installation onto the GUI thread along with the physical event
    # it will later filter. Calling QApplication.installEventFilter from the
    # pywebview `shown` thread silently leaves the filter inactive.
    bridge.install_requested.emit()
    return True


def _edge_value(edge: str):
    names = EDGE_NAMES.get(edge)
    if names is None:
        return None
    qt_edge = _qt().Qt.Edge
    value = getattr(qt_edge, names[0])
    for name in names[1:]:
        value |= getattr(qt_edge, name)
    return value


class QtChrome(NativeChrome):
    def __init__(self, window, bridge) -> None:
        super().__init__(window)
        # Keep the QObject alive for the lifetime of the native adapter.
        self._gesture_bridge = bridge

    def begin_drag(self, x: float, y: float) -> bool:
        del x, y  # Qt reads the active pointer gesture from the window system.
        self._gesture_bridge.move_requested.emit()
        return True

    def begin_resize(self, edge: str, x: float, y: float) -> bool:
        del x, y
        value = _edge_value(edge)
        if value is None:
            logger.warning("Unknown window edge {}", edge)
            return False
        self._gesture_bridge.resize_requested.emit(value)
        return True

    def set_drag_regions(self, regions, exclusions) -> None:
        # A tuple assignment is atomic in CPython. The pywebview API thread can
        # update it safely while the Qt GUI thread reads the previous or next
        # complete geometry set.
        self._gesture_bridge.regions = tuple(tuple(rect) for rect in regions)
        self._gesture_bridge.exclusions = tuple(tuple(rect) for rect in exclusions)

    def is_maximized(self) -> bool | None:
        native = self.window.native
        if native is None:
            return None
        try:
            return bool(native.isMaximized())
        except Exception:
            logger.exception("Could not read the Qt window state")
            return None


def install(window) -> NativeChrome:
    native = window.native
    if native is None:
        logger.warning("Qt window is not ready; leaving gestures to pywebview")
        return NativeChrome(window)
    handle = native.windowHandle()
    if handle is None:
        logger.warning("Qt window handle is unavailable; leaving gestures to pywebview")
        return NativeChrome(window)
    try:
        bridge = _make_bridge(handle, native)
        chrome = QtChrome(window, bridge)
        if _install_event_filter(bridge):
            logger.info("Installed native Qt window hit testing, move and resize gestures")
        else:
            logger.warning("Qt application is unavailable; using bridged window gestures")
        return chrome
    except Exception:
        logger.exception("Could not install native Qt window gestures")
        return NativeChrome(window)

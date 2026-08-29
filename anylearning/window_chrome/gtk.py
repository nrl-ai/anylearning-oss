"""GTK: a window with no decorations still has to move and resize.

pywebview implements `frameless=True` here as `set_decorated(False)`
(webview/platforms/gtk.py), and on most window managers that takes the resize
borders with it -- there is nothing left to grab. Dragging survives only
through pywebview's JS drag region, which walks the window across the screen
one bridged mousemove at a time.

Both gestures belong to the window manager, and GTK has the calls that hand
them over: `begin_move_drag` and `begin_resize_drag` start the real thing, so
the window snaps and tiles like any other. The title bar and the invisible
resize edges in the React shell call in here on mouse-down.
"""

from __future__ import annotations

from loguru import logger

from anylearning.window_chrome.base import NativeChrome

# Gdk.WindowEdge members, by the name the frontend uses for each edge.
EDGES = {
    "top": "NORTH",
    "topleft": "NORTH_WEST",
    "topright": "NORTH_EAST",
    "left": "WEST",
    "right": "EAST",
    "bottom": "SOUTH",
    "bottomleft": "SOUTH_WEST",
    "bottomright": "SOUTH_EAST",
}

LEFT_BUTTON = 1


def _gtk():
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk, GLib

    return Gdk, GLib


class GtkChrome(NativeChrome):
    def _on_gtk_thread(self, func) -> bool:
        """Queue `func` on the GTK main loop, where window calls belong."""
        native = self.window.native
        if native is None:
            return False
        try:
            _, glib = _gtk()
        except Exception:
            logger.exception(
                "PyGObject is unavailable; leaving the gesture to pywebview"
            )
            return False

        def once() -> bool:
            func(native)
            return False  # an idle callback runs again until it says otherwise

        glib.idle_add(once)
        return True

    def is_maximized(self) -> bool | None:
        native = self.window.native
        if native is None:
            return None
        try:
            return bool(native.is_maximized())
        except Exception:
            logger.exception("Could not read the window state")
            return None

    def begin_drag(self, x: float, y: float) -> bool:
        try:
            gdk, _ = _gtk()
        except Exception:
            return False
        return self._on_gtk_thread(
            lambda native: native.begin_move_drag(
                LEFT_BUTTON, int(x), int(y), gdk.CURRENT_TIME
            )
        )

    def begin_resize(self, edge: str, x: float, y: float) -> bool:
        name = EDGES.get(edge)
        if name is None:
            logger.warning("Unknown window edge {}", edge)
            return False
        try:
            gdk, _ = _gtk()
        except Exception:
            return False
        window_edge = getattr(gdk.WindowEdge, name)
        return self._on_gtk_thread(
            lambda native: native.begin_resize_drag(
                window_edge, LEFT_BUTTON, int(x), int(y), gdk.CURRENT_TIME
            )
        )

    def toggle_maximize(self, maximized: bool) -> None:
        # `Window.restore()` on GTK de-iconifies and presents; it never
        # un-maximises, so pywebview's maximise/restore pair is a one-way trip.
        if self._on_gtk_thread(
            lambda native: native.unmaximize() if maximized else native.maximize()
        ):
            return
        super().toggle_maximize(maximized)


def install(window) -> NativeChrome:
    return GtkChrome(window)

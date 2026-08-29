"""What every platform shares: driving the window through pywebview's own API.

Each backend subclasses this and replaces only the parts it can do better --
or, in a couple of cases, the parts pywebview gets wrong for a frameless
window. `Window.restore()` on GTK is the clearest example: it de-iconifies and
presents, and leaves a maximised window maximised, so a title-bar button built
on it cannot round-trip.
"""

from __future__ import annotations

from typing import Sequence

import webview


class NativeChrome:
    """The fallback: no native help, just the pywebview window API."""

    def __init__(self, window: webview.Window) -> None:
        self.window = window

    def minimize(self) -> None:
        self.window.minimize()

    def maximize(self) -> None:
        self.window.maximize()

    def restore(self) -> None:
        self.window.restore()

    def toggle_maximize(self, maximized: bool) -> None:
        if maximized:
            self.restore()
        else:
            self.maximize()

    def is_maximized(self) -> bool | None:
        """Whether the window is maximised, or None when we cannot ask it.

        The caller falls back to what the event that prompted the question
        implied, which is right often enough and wrong only until the next
        one.
        """
        return None

    def begin_drag(self, x: float, y: float) -> bool:
        """Start a native title-bar drag from a pointer at screen (x, y).

        Returning False means "no native drag here" and leaves the field to
        pywebview's own drag-region JS, which moves the window a mousemove at a
        time over the JS bridge. Backends that can hand the gesture to the
        window manager say so by returning True.
        """
        return False

    def begin_resize(self, edge: str, x: float, y: float) -> bool:
        """Start a native resize from a window edge. See `begin_drag`."""
        return False

    def drag_regions(self) -> Sequence[Sequence[float]]:
        """The title-bar rectangles last reported by the page, for diagnostics."""
        return ()

    def set_drag_regions(
        self, regions: Sequence[Sequence[float]], exclusions: Sequence[Sequence[float]]
    ) -> None:
        """Tell the native side where the app painted its title bar.

        Only Windows needs this: it answers WM_NCHITTEST from these rectangles
        (see `win32.py`). Everywhere else the drag surface is decided in the
        DOM, so this is a no-op.
        """

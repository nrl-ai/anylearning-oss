"""Native support for desktop window chrome.

Windows and macOS keep the app-drawn title bar; Linux keeps a platform frame.
Every backend charges a different price for that choice, and this package is
where each one is paid:

* Windows loses the entire system frame. `win32.py` puts back what a window
  has no business losing: resize borders, Aero Snap, double-click-to-maximise
  and a native title-bar drag.
* macOS keeps its resize borders but hides the traffic lights. `cocoa.py`
  shows them again, and the shell reserves the corner they sit in.
* Linux uses the compositor-owned frame in the normal desktop app. That is the
  only route that moves, resizes, snaps and exposes the system menu uniformly
  on both X11 and Wayland. `gtk.py` and `qt.py` remain the native fallback for
  callers that deliberately create a frameless Linux window.

Everything here is best-effort. A window with no native help is still a usable
window -- pywebview's own JS drag region keeps working -- so a failure to
install any of it is logged and swallowed rather than taken out on the user.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Sequence

import webview
from loguru import logger

from anylearning.window_chrome.base import NativeChrome

# The class the React shell puts on every draggable surface. It is pywebview's
# own default selector (`settings['DRAG_REGION_SELECTOR']`), which is what
# makes the JS drag path work unchanged when the native one is unavailable.
DRAG_REGION_CLASS = "pywebview-drag-region"

_chrome: WindowChrome | None = None


def _backend() -> str | None:
    """Which native implementation applies, from the renderer pywebview chose."""
    renderer = getattr(webview, "renderer", None)
    if renderer in ("edgechromium", "mshtml"):
        return "win32"
    if renderer == "wkwebview":
        return "cocoa"
    if renderer == "gtkwebkit2":
        return "gtk"
    if renderer in ("qtwebengine", "qtwebkit"):
        return "qt"
    return None


class WindowChrome:
    """The window's title bar, as seen from Python.

    Owns the JS API the React title bar calls, the maximised/restored state it
    renders from, and the platform object that does the native work.
    """

    def __init__(self, window: webview.Window, *, native_frame: bool = False) -> None:
        self.window = window
        self.native_frame = native_frame
        self.native = NativeChrome(window)
        self.maximized = False

    def attach(self) -> WindowChrome:
        window = self.window
        window.expose(
            self.window_chrome_state,
            self.window_minimize,
            self.window_toggle_maximize,
            self.window_close,
            self.window_begin_drag,
            self.window_begin_resize,
            self.window_set_drag_regions,
        )
        # `shown` rather than `loaded`: the native window has to exist before
        # any of the platform code can get at its handle.
        window.events.shown += self._install_native
        window.events.maximized += self._on_maximized
        window.events.restored += self._on_restored
        return self

    # -- platform installation ------------------------------------------

    def _install_native(self) -> None:
        # A compositor-owned frame already provides every operation this
        # package restores for a frameless window. Installing a second hit
        # tester would only compete with it.
        if self.native_frame:
            logger.info("Using the platform window frame")
            return

        backend = _backend()
        try:
            if backend == "win32":
                from anylearning.window_chrome import win32

                self.native = win32.install(self.window)
            elif backend == "cocoa":
                from anylearning.window_chrome import cocoa

                self.native = cocoa.install(self.window)
            elif backend == "gtk":
                from anylearning.window_chrome import gtk

                self.native = gtk.install(self.window)
            elif backend == "qt":
                from anylearning.window_chrome import qt

                self.native = qt.install(self.window)
            else:
                logger.info(
                    "No native window chrome for renderer {}; using the pywebview API",
                    getattr(webview, "renderer", None),
                )
        except Exception:
            logger.exception(
                "Native window chrome unavailable; using the pywebview API"
            )

    # -- maximised state -------------------------------------------------

    def _on_maximized(self) -> None:
        self._refresh_maximized(assumed=True)

    def _on_restored(self) -> None:
        # `restored` also fires on un-minimising, and a window that was
        # maximised when it went to the taskbar comes back maximised. Ask the
        # window rather than believe the event.
        self._refresh_maximized(assumed=False)

    def _refresh_maximized(self, assumed: bool) -> None:
        actual = self.native.is_maximized()
        self._set_maximized(assumed if actual is None else actual)

    def _set_maximized(self, maximized: bool) -> None:
        self.maximized = maximized
        detail = json.dumps({"maximized": maximized})
        try:
            self.window.evaluate_js(
                "window.dispatchEvent(new CustomEvent('anylearning:window-state', "
                f"{{detail: {detail}}}))"
            )
        except Exception:
            # Before the first load there is no page to tell; it asks for the
            # state itself on mount.
            logger.debug("Could not push the window state to the page")

    # -- the API the React title bar calls -------------------------------

    def window_chrome_state(self) -> dict:
        """What the title bar cannot work out for itself when it mounts.

        Not the platform: the page reads that from `window.pywebview.platform`,
        which is set before any of our code runs and spells the backends
        differently from `webview.renderer` -- 'cocoa' against 'wkwebview'.
        One vocabulary in the frontend is worth more than one round trip saved.
        """
        return {
            "maximized": self.maximized,
            "native_frame": self.native_frame,
        }

    def window_minimize(self) -> None:
        self.native.minimize()

    def window_toggle_maximize(self) -> bool:
        self.native.toggle_maximize(self.maximized)
        # Optimistic: the platform event follows and corrects this if the
        # window did something other than what was asked.
        self._set_maximized(not self.maximized)
        return self.maximized

    def window_close(self) -> None:
        close_window()

    def window_begin_drag(self, x: float, y: float) -> bool:
        return self.native.begin_drag(x, y)

    def window_begin_resize(self, edge: str, x: float, y: float) -> bool:
        return self.native.begin_resize(edge, x, y)

    def window_set_drag_regions(
        self,
        regions: Sequence[Sequence[float]] | None,
        exclusions: Sequence[Sequence[float]] | None,
    ) -> None:
        self.native.set_drag_regions(regions or (), exclusions or ())


def needs_transparency() -> bool:
    """Whether the window has to carry an alpha channel to have round corners.

    Only GTK does. It will not round an undecorated window, so the corners have
    to be transparent and drawn by the page instead (`.window-shell` in
    globals.css). macOS rounds a titled window itself, and Windows is asked to
    in `win32.py` -- and there an alpha channel would cost the drop shadow,
    which DWM only extends into an opaque frame.

    Read from `sys.platform` rather than the renderer: this is needed when the
    window is created, and pywebview does not pick a backend until it starts.
    """
    return sys.platform.startswith("linux")


def attach(window: webview.Window, *, native_frame: bool = False) -> WindowChrome:
    """Wire the chrome to the app's window. Called once, at window creation."""
    global _chrome
    _chrome = WindowChrome(window, native_frame=native_frame).attach()
    return _chrome


def _active() -> WindowChrome | None:
    """The chrome driving the live window, if there is a window at all.

    Served in a browser -- `--development`, or `--server` -- there is none,
    which is a normal state and not an error.
    """
    if not webview.windows:
        return None
    if _chrome is not None and _chrome.window is webview.windows[0]:
        return _chrome
    # A window nobody attached to: still drivable, just without native help.
    return WindowChrome(webview.windows[0])


def minimize_window() -> bool:
    chrome = _active()
    if chrome is None:
        return False
    chrome.native.minimize()
    return True


def maximize_window() -> bool:
    chrome = _active()
    if chrome is None:
        return False
    chrome.native.maximize()
    return True


def restore_window() -> bool:
    chrome = _active()
    if chrome is None:
        return False
    chrome.native.restore()
    return True


def close_window() -> bool:
    chrome = _active()
    if chrome is None:
        return False
    chrome.window.destroy()
    # Closing the window is not enough to end the run: uvicorn holds a daemon
    # thread and training runs in child processes.
    os._exit(0)

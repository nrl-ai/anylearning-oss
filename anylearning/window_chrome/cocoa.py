"""macOS: keep the traffic lights, lose the title bar.

`frameless=True` on Cocoa gives pywebview's window a full-size content view
and a transparent title bar -- which is exactly what an app-drawn title bar
wants -- and then hides the close, minimise and zoom buttons with it
(webview/platforms/cocoa.py). Hiding them is the one part we do not want: a
Mac window without traffic lights is a Mac window people cannot close, and
nothing the app draws in their place would be as familiar.

So they go back on, and the React side keeps the top-left corner of the
window clear for them (`--titlebar-inset` in globals.css). The rest of the
frame needs no help here: the window keeps NSTitledWindowMask and
NSResizableWindowMask, so every edge still resizes natively and AppKit still
rounds and shadows the window itself.
"""

from __future__ import annotations

from loguru import logger

from anylearning.window_chrome.base import NativeChrome


def _call_on_main_thread(func, *args) -> None:
    from PyObjCTools import AppHelper

    AppHelper.callAfter(func, *args)


class CocoaChrome(NativeChrome):
    def show_traffic_lights(self) -> bool:
        window = self.window.native
        if window is None:
            logger.warning("No native window yet; the traffic lights stay hidden")
            return False

        def _show():
            try:
                import AppKit

                for button in (
                    AppKit.NSWindowCloseButton,
                    AppKit.NSWindowMiniaturizeButton,
                    AppKit.NSWindowZoomButton,
                ):
                    window.standardWindowButton_(button).setHidden_(False)
            except Exception:
                logger.exception("Could not show the window buttons")

        _call_on_main_thread(_show)
        return True

    def toggle_maximize(self, maximized: bool) -> None:
        # The green button's own behaviour, rather than pywebview's pair:
        # `Window.maximize()` resizes to the screen and `Window.restore()` only
        # de-miniaturises, so the two cannot return a window to where it was.
        window = self.window.native
        if window is None:
            super().toggle_maximize(maximized)
            return
        _call_on_main_thread(window.zoom_, None)


def install(window) -> NativeChrome:
    chrome = CocoaChrome(window)
    chrome.show_traffic_lights()
    return chrome

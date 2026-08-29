"""Windows: put back the frame we asked the system not to draw.

pywebview implements `frameless=True` on Windows as `FormBorderStyle.None`
(webview/platforms/winforms.py) and stops there, which is what a frameless
window costs on this platform: no resize borders, no Aero Snap, no snap
layouts, no double-click-to-maximise, and a title-bar drag that runs
JS -> Python -> `window.move()` on every mouse move.

The fix is the one Chromium uses for its own custom frame. Keep a real sizing
frame on the window, tell Windows the client area covers all of it, and answer
WM_NCHITTEST ourselves so the strip the app paints as a title bar is reported
as the caption. Windows then runs its own move, resize and snap loops: the
gestures are native again, and dragging never touches JavaScript.

The window procedure is replaced with SetWindowLongPtrW and chains to the
previous one for every message we do not handle. If any of this fails the
original procedure stays in place and the app falls back to pywebview's JS
drag region -- slower, but not broken -- so nothing here may raise into the
caller.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Sequence

from loguru import logger

from anylearning.window_chrome import hit_test
from anylearning.window_chrome.base import NativeChrome

# winuser.h
WM_NCCALCSIZE = 0x0083
WM_NCHITTEST = 0x0084
WM_NCDESTROY = 0x0082
WM_GETMINMAXINFO = 0x0024

MONITOR_DEFAULTTONEAREST = 2

GWL_STYLE = -16
GWLP_WNDPROC = -4

WS_MAXIMIZEBOX = 0x00010000
WS_MINIMIZEBOX = 0x00020000
WS_THICKFRAME = 0x00040000

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020

SM_CXSIZEFRAME = 32
SM_CYSIZEFRAME = 33
SM_CXPADDEDBORDER = 92

# dwmapi.h. Windows 11 rounds a window with a frame, but a window whose client
# area covers its whole frame is not obviously one, so it is asked outright.
# Windows 10 has no such attribute and fails the call harmlessly.
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_ROUND = 2

HIT_CODES = {
    hit_test.CAPTION: 2,  # HTCAPTION
    hit_test.LEFT: 10,
    hit_test.RIGHT: 11,
    hit_test.TOP: 12,
    hit_test.TOP_LEFT: 13,
    hit_test.TOP_RIGHT: 14,
    hit_test.BOTTOM: 15,
    hit_test.BOTTOM_LEFT: 16,
    hit_test.BOTTOM_RIGHT: 17,
}

# Floor for the grab strip. Windows' own frame is SM_CXSIZEFRAME plus
# SM_CXPADDEDBORDER -- 8px at 100% scaling -- and that is what a user's aim is
# calibrated to; the floor only guards a theme reporting something unusably
# thin.
MIN_RESIZE_BORDER = 4

_LRESULT = ctypes.c_ssize_t


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class _NCCALCSIZE_PARAMS(ctypes.Structure):
    _fields_ = [("rgrc", _RECT * 3), ("lppos", ctypes.c_void_p)]


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", _RECT),
        ("rcWork", _RECT),
        ("dwFlags", wintypes.DWORD),
    ]


class _MINMAXINFO(ctypes.Structure):
    _fields_ = [
        ("ptReserved", wintypes.POINT),
        ("ptMaxSize", wintypes.POINT),
        ("ptMaxPosition", wintypes.POINT),
        ("ptMinTrackSize", wintypes.POINT),
        ("ptMaxTrackSize", wintypes.POINT),
    ]


def _wndproc_type():
    return ctypes.WINFUNCTYPE(
        _LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    )


def _user32():
    """user32 with the prototypes we depend on declared.

    ctypes defaults every return value to a 32-bit int, which silently
    truncates the LRESULT of CallWindowProcW and the previous procedure
    returned by SetWindowLongPtrW -- both pointer-sized. `ctypes.windll`
    caches the library, so declaring them here is a one-off.
    """
    user32 = ctypes.windll.user32
    if getattr(user32, "_anylearning_prototypes", False):
        return user32

    user32.CallWindowProcW.restype = _LRESULT
    user32.CallWindowProcW.argtypes = [
        ctypes.c_void_p,
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    setter = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
    setter.restype = _LRESULT
    setter.argtypes = [wintypes.HWND, ctypes.c_int, _LRESULT]
    user32.GetWindowLongW.restype = wintypes.LONG
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetWindowLongW.restype = wintypes.LONG
    user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
    user32.ScreenToClient.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]
    user32.IsZoomed.argtypes = [wintypes.HWND]
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    # A monitor handle is pointer-sized, so the default 32-bit restype would
    # hand GetMonitorInfoW half of one.
    user32.MonitorFromWindow.restype = ctypes.c_void_p
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MONITORINFO)]

    user32._anylearning_prototypes = True
    return user32


def _set_window_long_ptr(hwnd: int, index: int, value) -> int:
    """SetWindowLongPtrW, or its 32-bit spelling on a 32-bit Python."""
    user32 = _user32()
    setter = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
    if not isinstance(value, int):
        value = ctypes.cast(value, ctypes.c_void_p).value
    return setter(hwnd, index, value)


def _window_handle(form) -> int:
    """The HWND behind pywebview's WinForms form, as a plain int."""
    handle = form.Handle
    to_int64 = getattr(handle, "ToInt64", None)
    return int(to_int64()) if to_int64 else int(handle)


def _on_ui_thread(form, action) -> None:
    """Run `action` on the thread that owns the window.

    Window events arrive on a worker thread -- pywebview's `Event.set` starts
    one per notification -- and subclassing a window from a thread that does
    not own it is asking for trouble.
    """
    try:
        from System import Action  # pythonnet, installed with the Windows backend

        form.Invoke(Action(action))
    except Exception:
        logger.debug("Form.Invoke unavailable; installing the frame in place")
        action()


class WindowsChrome(NativeChrome):
    """A custom frame: the app paints the title bar, Windows still owns the gestures."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self._hwnd = 0
        self._previous_proc = 0
        self._proc = None  # kept alive: ctypes will not hold the callback for us
        # Replaced wholesale rather than mutated, so the message thread always
        # reads one self-consistent set of rectangles.
        self._regions: tuple[tuple[float, ...], ...] = ()
        self._exclusions: tuple[tuple[float, ...], ...] = ()
        self._caption_bottom = 0.0

    # -- installation ---------------------------------------------------

    def install(self) -> bool:
        form = self.window.native
        if form is None:
            logger.warning("No native window yet; leaving the system frame alone")
            return False

        installed = [False]

        def _install():
            try:
                user32 = _user32()
                hwnd = _window_handle(form)

                # WinForms drops these along with the border. WS_THICKFRAME is
                # what makes Windows willing to run a resize loop at all, and
                # the box styles are what Aero Snap and the maximise animation
                # look for. None of the frame becomes visible: WM_NCCALCSIZE
                # below hands the whole window to the client area.
                style = user32.GetWindowLongW(hwnd, GWL_STYLE)
                user32.SetWindowLongW(
                    hwnd,
                    GWL_STYLE,
                    style | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX,
                )

                proc = _wndproc_type()(self._wnd_proc)
                previous = _set_window_long_ptr(hwnd, GWLP_WNDPROC, proc)
                if not previous:
                    raise OSError(ctypes.GetLastError(), "SetWindowLongPtrW failed")

                self._hwnd = hwnd
                self._proc = proc
                self._previous_proc = previous
                _round_corners(hwnd)

                # Nothing has asked the window to recalculate its frame yet, so
                # the style change is inert until we say so.
                user32.SetWindowPos(
                    hwnd,
                    None,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE
                    | SWP_NOSIZE
                    | SWP_NOZORDER
                    | SWP_NOACTIVATE
                    | SWP_FRAMECHANGED,
                )
                installed[0] = True
            except Exception:
                logger.exception("Could not install the custom window frame")

        _on_ui_thread(form, _install)
        return installed[0]

    def _restore_wnd_proc(self) -> None:
        if not self._previous_proc:
            return
        try:
            _set_window_long_ptr(self._hwnd, GWLP_WNDPROC, self._previous_proc)
        except Exception:
            logger.exception("Could not restore the original window procedure")
        finally:
            self._previous_proc = 0
            self._proc = None

    def is_maximized(self) -> bool | None:
        if not self._hwnd:
            return None
        return bool(_user32().IsZoomed(self._hwnd))

    # -- drag regions ---------------------------------------------------

    def drag_regions(self) -> tuple[tuple[float, ...], ...]:
        return self._regions

    def set_drag_regions(
        self, regions: Sequence[Sequence[float]], exclusions: Sequence[Sequence[float]]
    ) -> None:
        self._regions = tuple(tuple(rect) for rect in regions)
        self._exclusions = tuple(tuple(rect) for rect in exclusions)
        self._caption_bottom = max((rect[3] for rect in self._regions), default=0.0)

    # -- message handling -----------------------------------------------

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        try:
            if msg == WM_NCCALCSIZE and wparam:
                return self._client_covers_the_window()
            if msg == WM_GETMINMAXINFO:
                limits = self._maximised_geometry(hwnd, lparam)
                if limits is not None:
                    return limits
            if msg == WM_NCHITTEST:
                zone = self._zone_at(hwnd, lparam)
                if zone is not None:
                    return zone
            elif msg == WM_NCDESTROY:
                self._restore_wnd_proc()
        except Exception:
            # A window procedure that raises takes the window's input with it.
            # Fall through to the original one instead.
            logger.exception("Custom frame failed to handle message {}", msg)

        return _user32().CallWindowProcW(
            ctypes.c_void_p(self._previous_proc), hwnd, msg, wparam, lparam
        )

    def _client_covers_the_window(self) -> int:
        """WM_NCCALCSIZE: no non-client area, so the app paints every pixel.

        Nothing to correct when maximised: `_maximised_geometry` below sizes
        the window to the work area exactly, so the client area is already
        where it should be. Insetting here as well -- the usual recipe, for
        windows that let Windows pick the maximised rect -- would leave a gap
        around a maximised window instead.
        """
        return 0

    def _maximised_geometry(self, hwnd, lparam) -> int | None:
        """WM_GETMINMAXINFO: maximise onto the work area, not over the taskbar.

        Windows derives the maximised rectangle from the window's frame, and a
        window that has handed its whole frame to the client area gives it
        nothing to work from: it picks the entire monitor, and the taskbar
        disappears underneath. Measured on Windows 11 before this existed --
        client 0,0-1324,878 against a work area of 0,0-1324,830.
        """
        user32 = _user32()
        monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if not monitor:
            return None

        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None

        limits = ctypes.cast(lparam, ctypes.POINTER(_MINMAXINFO)).contents
        work, screen = info.rcWork, info.rcMonitor
        # ptMaxPosition is relative to the monitor, not to the desktop.
        limits.ptMaxPosition.x = work.left - screen.left
        limits.ptMaxPosition.y = work.top - screen.top
        limits.ptMaxSize.x = work.right - work.left
        limits.ptMaxSize.y = work.bottom - work.top
        return 0

    def _zone_at(self, hwnd, lparam) -> int | None:
        """WM_NCHITTEST: None means "not ours", which reads as the page."""
        user32 = _user32()

        point = wintypes.POINT(_low_word(lparam), _high_word(lparam))
        if not user32.ScreenToClient(hwnd, ctypes.byref(point)):
            return None

        rect = _RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None

        width = rect.right - rect.left
        height = rect.bottom - rect.top
        border = max(
            _system_metric(SM_CXSIZEFRAME) + _system_metric(SM_CXPADDEDBORDER),
            MIN_RESIZE_BORDER,
        )

        # This runs on every mouse move over the window, so the common case --
        # the pointer in the page, below the title bar and away from the edges
        # -- gets out before touching the rectangle lists.
        if (
            point.y >= self._caption_bottom
            and border <= point.x < width - border
            and point.y < height - border
        ):
            return None

        zone = hit_test.hit_test(
            point.x,
            point.y,
            width=width,
            height=height,
            border=border,
            regions=self._regions,
            exclusions=self._exclusions,
            maximized=bool(user32.IsZoomed(hwnd)),
        )
        return HIT_CODES.get(zone)


def _round_corners(hwnd: int) -> None:
    """Ask DWM for Windows 11's rounded corners. Older Windows just says no."""
    try:
        preference = ctypes.c_int(DWMWCP_ROUND)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            ctypes.c_uint(DWMWA_WINDOW_CORNER_PREFERENCE),
            ctypes.byref(preference),
            ctypes.sizeof(preference),
        )
    except Exception:
        logger.debug("Rounded corners are unavailable on this Windows")


def _low_word(value: int) -> int:
    return ctypes.c_short(value & 0xFFFF).value


def _high_word(value: int) -> int:
    return ctypes.c_short((value >> 16) & 0xFFFF).value


def _system_metric(index: int) -> int:
    return _user32().GetSystemMetrics(index)


def install(window) -> NativeChrome:
    chrome = WindowsChrome(window)
    if not chrome.install():
        # The frame stays as WinForms left it: no resize borders and no snap,
        # but pywebview's JS drag region still moves the window.
        logger.warning("Falling back to the plain frameless window")
    return chrome

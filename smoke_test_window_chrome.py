"""Check the app-drawn title bar by asking the window, not by looking at it.

Most of what `anylearning/window_chrome/` does is invisible in a screenshot and
plain to the window itself, so this opens the real window and interrogates it
through the same platform calls the frame is built on.

On Windows that is the custom frame: what the hit test answers where, which
styles survived, and whether a maximised client area lands on the work area
rather than over the taskbar. On macOS it is the traffic lights, the style
mask, how much of the corner they need, and whether AppKit's title bar lets
presses through to the page in the top strip -- which decides whether anything
the app draws up there can be clicked at all.

    python3 -m venv .venv                   # py -3.13 -m venv .venv on Windows
    .venv/bin/pip install pywebview loguru  # .venv\\Scripts\\pip
    .venv/bin/python smoke_test_window_chrome.py

That is the whole dependency list. No ML stack and no backend: the page's API
calls fail and the project list comes up empty, which does not matter, because
the title bar, the drag surfaces and the frame are all still there and they are
what is under test.

The window stays open afterwards for the gestures no probe can fake -- they
belong to the window manager, and it prints the list for the platform it is on.
"""

from __future__ import annotations

import ctypes
import functools
import os
import socket
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import webview

from anylearning import window_chrome

FRONTEND_DIST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "anylearning", "frontend-dist"
)

# Long enough for the page to render and, on Windows, to report where it put
# its drag surfaces -- the frontend debounces that by 200ms.
SETTLE_SECONDS = 4

# Hit test codes, from winuser.h.
HTCLIENT = 1
HTCAPTION = 2
HTLEFT = 10
HTTOP = 12
HTTOPLEFT = 13
HTBOTTOMRIGHT = 17
HIT_NAMES = {
    0: "HTNOWHERE",
    HTCLIENT: "HTCLIENT",
    HTCAPTION: "HTCAPTION",
    3: "HTSYSMENU",
    8: "HTMINBUTTON",
    9: "HTMAXBUTTON",
    HTLEFT: "HTLEFT",
    11: "HTRIGHT",
    HTTOP: "HTTOP",
    HTTOPLEFT: "HTTOPLEFT",
    14: "HTTOPRIGHT",
    15: "HTBOTTOM",
    16: "HTBOTTOMLEFT",
    HTBOTTOMRIGHT: "HTBOTTOMRIGHT",
}

WM_NCHITTEST = 0x0084
GWL_STYLE = -16
WS_MAXIMIZEBOX = 0x00010000
WS_MINIMIZEBOX = 0x00020000
WS_THICKFRAME = 0x00040000
SW_MAXIMIZE = 3
SW_RESTORE = 9
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_ROUND = 2

# The gestures belong to the window manager, so they are the part a person has
# to judge. Per platform, because each one snaps and zooms differently.
BY_HAND = {
    "win32": [
        "drag the title bar, and drag it to the top of the screen to snap",
        "resize from an edge and from a corner",
        "double-click the title bar to maximise, and again to restore",
        "Win+Left / Win+Right, and the snap layouts under the maximise button",
        "maximise and check the taskbar is still visible",
        "the corners are rounded (Windows 11 only)",
    ],
    "darwin": [
        "drag the title bar, and the sidebar's wordmark row",
        "double-click the title bar to zoom",
        "resize from an edge and from a corner",
        "the traffic lights work, and sit clear of the wordmark",
        "click something in the top strip -- the theme button, the stage rail",
    ],
    "linux": [
        "drag the title bar, and drag it to an edge to tile",
        "resize from an edge and from a corner (the strips are invisible)",
        "double-click the title bar to maximise, and again to restore",
        "the corners are rounded, with the desktop showing through",
    ],
}

# Unknown API requests get an empty list, which is a state the app is built for.
API_STUBS = {}

results: list[tuple[bool | None, str, str]] = []


def check(passed: bool | None, name: str, detail: str = "") -> None:
    results.append((passed, name, detail))


def serve(directory: str) -> str:
    """Serve the packaged frontend on a free port, and return its URL."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    handler = functools.partial(QuietHandler, directory=directory)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        """The probe output is the point; a request log would bury it."""

    def do_GET(self):
        """Answer the API as an empty app would.

        There is no backend here on purpose. Two answers matter. Serving
        index.html for /api/… hands the app HTML where it expects JSON, and the
        sidebar dies on `projects.filter is not a function` -- which Next
        answers with its built-in "This page couldn't load" page, so the window
        under test shows no title bar at all.
        """
        if self.path.startswith("/api/"):
            body = API_STUBS.get(self.path.split("?", 1)[0], b"[]")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def send_head(self):
        """Resolve a route the way a static export expects to be served.

        Next writes `projects/overview.html`, not `projects/overview`, so a
        plain file server answers every route with a 404 and the app never
        starts. Try the exported page first, then a directory index, and only
        then fall back to the root document.
        """
        route = self.path.split("?", 1)[0].split("#", 1)[0]
        target = self.translate_path(route)
        for candidate in (target, target + ".html", os.path.join(target, "index.html")):
            if os.path.isfile(candidate):
                self.path = route + candidate[len(target) :]
                break
        else:
            self.path = "/index.html"
        return super().send_head()


# ---------------------------------------------------------------------------
# The probes
# ---------------------------------------------------------------------------


def hit_test_at(user32, hwnd: int, x: int, y: int) -> int:
    """What the window says is at a screen point."""
    return user32.SendMessageW(hwnd, WM_NCHITTEST, 0, (y << 16) | (x & 0xFFFF))


def probe_windows(window: webview.Window) -> None:
    import ctypes.wintypes as wintypes

    # A private handle, not ctypes.windll.user32: that one is cached process
    # wide, and win32.py has already declared GetClientRect against its own
    # RECT. Sharing it makes every call from here fail with
    #   ArgumentError: expected LP__RECT instance instead of pointer to RECT
    user32 = ctypes.WinDLL("user32")
    user32.SendMessageW.restype = ctypes.c_ssize_t
    hwnd = int(window.native.Handle.ToInt64())

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    def window_rect() -> RECT:
        rect = RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return rect

    def client_rect_on_screen() -> RECT:
        rect = RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        origin = wintypes.POINT(rect.left, rect.top)
        user32.ClientToScreen(hwnd, ctypes.byref(origin))
        return RECT(
            origin.x,
            origin.y,
            origin.x + (rect.right - rect.left),
            origin.y + (rect.bottom - rect.top),
        )

    # 1. The styles the resize loop and Aero Snap look for.
    style = user32.GetWindowLongW(hwnd, GWL_STYLE)
    missing = [
        name
        for name, bit in (
            ("WS_THICKFRAME", WS_THICKFRAME),
            ("WS_MAXIMIZEBOX", WS_MAXIMIZEBOX),
            ("WS_MINIMIZEBOX", WS_MINIMIZEBOX),
        )
        if not style & bit
    ]
    check(
        not missing,
        "Sizing styles restored",
        f"style=0x{style:08X} missing={missing or 'none'}",
    )

    # 2. WM_NCCALCSIZE gave the whole window to the client area.
    frame, client = window_rect(), client_rect_on_screen()
    same = (frame.left, frame.top, frame.right, frame.bottom) == (
        client.left,
        client.top,
        client.right,
        client.bottom,
    )
    check(
        same,
        "Client area covers the window",
        f"window={frame.left},{frame.top},{frame.right},{frame.bottom} "
        f"client={client.left},{client.top},{client.right},{client.bottom}",
    )

    # 3. What the page told us about itself. The hit test below is answered
    #    from these rectangles, so an empty list moves the blame off the frame.
    chrome = window_chrome._active()
    regions = list(chrome.native.drag_regions()) if chrome else []
    check(
        bool(regions),
        "Page reported its drag surfaces",
        f"{len(regions)} rectangles, tallest={max(regions, key=lambda r: r[3]) if regions else None}",
    )

    # 4. The hit test: the frame, the title bar, the page, and the controls
    #    that sit in the title bar and must stay clickable.
    mid_x = (frame.left + frame.right) // 2
    mid_y = (frame.top + frame.bottom) // 2
    for label, x, y, expected in (
        ("top-left corner resizes", frame.left + 2, frame.top + 2, HTTOPLEFT),
        ("left edge resizes", frame.left + 2, mid_y, HTLEFT),
        (
            "bottom-right corner resizes",
            frame.right - 3,
            frame.bottom - 3,
            HTBOTTOMRIGHT,
        ),
        ("title bar drags", mid_x, frame.top + 30, HTCAPTION),
        ("page is the page", mid_x, mid_y, HTCLIENT),
        ("window controls stay clickable", frame.right - 30, frame.top + 30, HTCLIENT),
    ):
        answer = hit_test_at(user32, hwnd, x, y)
        check(
            answer == expected,
            label,
            f"at ({x},{y}) got {HIT_NAMES.get(answer, answer)}, want {HIT_NAMES[expected]}",
        )

    # 5. Maximised, the client area has to land on the work area exactly. Too
    #    large and it covers the taskbar; too small and there is a gap.
    user32.ShowWindow(hwnd, SW_MAXIMIZE)
    time.sleep(1)
    work = RECT()
    user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work), 0)  # SPI_GETWORKAREA
    maximised = client_rect_on_screen()
    fits = (maximised.left, maximised.top, maximised.right, maximised.bottom) == (
        work.left,
        work.top,
        work.right,
        work.bottom,
    )
    check(
        fits,
        "Maximised client area is the work area",
        f"client={maximised.left},{maximised.top},{maximised.right},{maximised.bottom} "
        f"work={work.left},{work.top},{work.right},{work.bottom}",
    )
    user32.ShowWindow(hwnd, SW_RESTORE)

    # 6. Rounded corners, on the Windows that has them.
    preference = ctypes.c_int(0)
    status = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd),
        ctypes.c_uint(DWMWA_WINDOW_CORNER_PREFERENCE),
        ctypes.byref(preference),
        ctypes.sizeof(preference),
    )
    if status != 0:
        check(
            None,
            "Rounded corners requested",
            "not supported on this Windows (expected on 10)",
        )
    else:
        check(
            preference.value == DWMWCP_ROUND,
            "Rounded corners requested",
            f"corner preference={preference.value}, want {DWMWCP_ROUND}",
        )


def probe_page(window: webview.Window) -> None:
    """Did the app actually render?

    Everything else here assumes it did. The Windows hit test in particular is
    answered from rectangles the page reports, so a window showing the
    browser's own "couldn't load" page fails the title-bar check for a reason
    that has nothing to do with the frame.
    """
    url = window.get_current_url()
    title = window.evaluate_js("document.title")
    surfaces = window.evaluate_js(
        "document.querySelectorAll('.pywebview-drag-region').length"
    )
    body = window.evaluate_js(
        "document.body.innerText.replace(/\\s+/g, ' ').slice(0, 90)"
    )

    check(
        bool(surfaces),
        "Page rendered the title bar",
        f"{surfaces} drag surfaces, title={title!r}, url={url}",
    )
    if not surfaces:
        check(None, "  what the window is showing", repr(body))

    # Did the React side recognise the shell it is in? `data-desktop` is the
    # first thing <DesktopChrome> sets, and everything it does afterwards --
    # the controls, the reserved corners, publishing the drag surfaces to
    # Windows -- follows from that same detection.
    seen = window.evaluate_js("(window.pywebview && window.pywebview.platform) || null")
    detected = window.evaluate_js("document.documentElement.dataset.desktop || null")
    exposed = window.evaluate_js(
        "Object.keys((window.pywebview && window.pywebview.api) || {}).sort().join(' ')"
    )
    check(
        bool(detected),
        "Page recognised the desktop shell",
        f"pywebview.platform={seen!r}, data-desktop={detected!r}",
    )
    check(
        "window_set_drag_regions" in (exposed or ""),
        "Window API reached the page",
        f"api: {exposed or '(none)'}",
    )


def on_main_thread(func, timeout: float = 5.0):
    """Run `func` on the AppKit main thread and bring back what it returned.

    Reading a window from another thread is how a probe ends up hanging the
    app it is probing.
    """
    from PyObjCTools import AppHelper

    outcome: dict = {}
    done = threading.Event()

    def run():
        try:
            outcome["value"] = func()
        except Exception as error:
            outcome["error"] = error
        finally:
            done.set()

    AppHelper.callAfter(run)
    if not done.wait(timeout):
        raise TimeoutError("the main thread did not answer")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def probe_macos(window: webview.Window) -> None:
    import AppKit

    native = window.native

    # 1. The traffic lights. pywebview hides all three for a frameless window;
    #    cocoa.py puts them back, because nothing the app draws would be as
    #    familiar and a Mac window has to close the way every other one does.
    hidden = on_main_thread(
        lambda: [
            bool(native.standardWindowButton_(button).isHidden())
            for button in (
                AppKit.NSWindowCloseButton,
                AppKit.NSWindowMiniaturizeButton,
                AppKit.NSWindowZoomButton,
            )
        ]
    )
    check(
        not any(hidden),
        "Traffic lights are shown",
        f"close/minimise/zoom hidden={hidden}",
    )

    # 2. The style mask the rest of this depends on: titled (so AppKit still
    #    rounds and shadows the window), resizable (so the edges still work
    #    without any help from us) and full-size content (so the page reaches
    #    under the title bar).
    mask = on_main_thread(native.styleMask)
    wanted = {
        "titled": AppKit.NSWindowStyleMaskTitled,
        "resizable": AppKit.NSWindowStyleMaskResizable,
        "fullSizeContentView": AppKit.NSWindowStyleMaskFullSizeContentView,
    }
    missing = [name for name, bit in wanted.items() if not mask & bit]
    check(
        not missing,
        "Window style is right",
        f"mask=0x{mask:X} missing={missing or 'none'}",
    )

    # 3. Where the traffic lights actually are, so the shell knows how much of
    #    the corner to keep clear. --titlebar-inset reserves 24pt at the top,
    #    --titlebar-inset-left 72pt for the short bars.
    def button_box():
        frames = [
            native.standardWindowButton_(button).convertRect_toView_(
                native.standardWindowButton_(button).bounds(), None
            )
            for button in (
                AppKit.NSWindowCloseButton,
                AppKit.NSWindowMiniaturizeButton,
                AppKit.NSWindowZoomButton,
            )
        ]
        height = native.frame().size.height
        right = max(frame.origin.x + frame.size.width for frame in frames)
        # Cocoa measures from the bottom; the shell reserves from the top.
        top = min(height - (frame.origin.y + frame.size.height) for frame in frames)
        bottom = max(height - frame.origin.y for frame in frames)
        return right, top, bottom

    right, top, bottom = on_main_thread(button_box)
    check(
        bottom <= 24 and right <= 72,
        "Reserved corner is big enough",
        f"lights occupy {right:.0f}x{bottom:.0f}pt from the top-left, reserved 72x24pt",
    )

    # 4. The question a screenshot cannot answer: with a full-size content
    #    view, does AppKit's title bar swallow presses in the top strip, or do
    #    they reach the page? Everything the app draws up there -- the theme
    #    toggle, the stage rail -- depends on the answer.
    def view_at(y_from_top: float) -> str:
        frame_view = native.contentView().superview()
        height = native.frame().size.height
        point = AppKit.NSMakePoint(400, height - y_from_top)
        hit = frame_view.hitTest_(point)
        return hit.__class__.__name__ if hit else "nothing"

    in_strip = on_main_thread(lambda: view_at(12))
    below = on_main_thread(lambda: view_at(120))
    check(
        in_strip == below,
        "Clicks in the top strip reach the page",
        f"at 12pt from the top: {in_strip}; at 120pt: {below}",
    )


def report() -> None:
    print()
    print(f"  {'':4} {'check':40} detail")
    print(f"  {'-' * 4} {'-' * 40} {'-' * 40}")
    for passed, name, detail in results:
        mark = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        print(f"  {mark:4} {name:40} {detail}")

    failed = [name for passed, name, _ in results if passed is False]
    print()
    print(f"  {len(failed)} failed of {len(results)}")
    print()
    print("  The window is still open. What no probe can check:")
    for line in BY_HAND.get(sys.platform, BY_HAND["linux"]):
        print(f"    - {line}")


def main() -> int:
    if not os.path.isdir(FRONTEND_DIST):
        print(
            f"No packaged frontend at {FRONTEND_DIST} -- run build_frontend.sh first",
            file=sys.stderr,
        )
        return 1

    url = serve(FRONTEND_DIST)
    print(f"serving {FRONTEND_DIST} at {url}")

    webview.settings.update(
        {
            "ALLOW_DOWNLOADS": True,
            "ALLOW_FILE_URLS": True,
            "OPEN_EXTERNAL_LINKS_IN_BROWSER": True,
            "DRAG_REGION_DIRECT_TARGET_ONLY": True,
        }
    )
    window = webview.create_window(
        "AnyLearning",
        url,
        width=1200,
        height=800,
        resizable=True,
        frameless=True,
        easy_drag=False,
        transparent=window_chrome.needs_transparency(),
    )
    window_chrome.attach(window)

    def run_probes():
        time.sleep(SETTLE_SECONDS)
        try:
            probe_page(window)
            if sys.platform == "win32":
                probe_windows(window)
            elif sys.platform == "darwin":
                probe_macos(window)
            else:
                check(
                    None, "Native frame probes", f"nothing to check on {sys.platform}"
                )
        except Exception as error:  # a broken probe must not take the window down
            check(False, "Probes ran", f"{type(error).__name__}: {error}")
        report()

    window.events.loaded += lambda: threading.Thread(
        target=run_probes, daemon=True
    ).start()
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

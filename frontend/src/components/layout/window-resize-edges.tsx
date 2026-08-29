"use client"

import { ResizeEdge, beginWindowResize } from "@/lib/desktop"

/**
 * Invisible grab strips around the window, for GTK only.
 *
 * `set_decorated(False)` is how pywebview makes a window frameless on Linux,
 * and on most window managers it takes the resize borders with it -- there is
 * nothing left to grab. macOS keeps its own (the window stays resizable), and
 * Windows gets them back in the hit test (`window_chrome/win32.py`), so this
 * is the one platform that needs the page to offer the edges itself.
 *
 * A press hands the gesture straight to the window manager through
 * `begin_resize_drag`, so what follows -- the outline, the snapping, the
 * tiling shortcuts -- is the real thing rather than a JS approximation.
 */

const EDGES: { edge: ResizeEdge; className: string }[] = [
    { edge: "top", className: "top-0 right-3 left-3 h-1 cursor-n-resize" },
    { edge: "bottom", className: "right-3 bottom-0 left-3 h-1 cursor-s-resize" },
    { edge: "left", className: "top-3 bottom-3 left-0 w-1 cursor-w-resize" },
    { edge: "right", className: "top-3 right-0 bottom-3 w-1 cursor-e-resize" },
    // Corners last: they overlap the strips above and have to win.
    { edge: "topleft", className: "top-0 left-0 size-3 cursor-nw-resize" },
    { edge: "topright", className: "top-0 right-0 size-3 cursor-ne-resize" },
    { edge: "bottomleft", className: "bottom-0 left-0 size-3 cursor-sw-resize" },
    { edge: "bottomright", className: "right-0 bottom-0 size-3 cursor-se-resize" },
]

export function WindowResizeEdges() {
    return (
        <div data-window-no-drag className="pointer-events-none fixed inset-0 z-[80]">
            {EDGES.map(({ edge, className }) => (
                <div
                    key={edge}
                    aria-hidden
                    className={`pointer-events-auto fixed ${className}`}
                    onMouseDown={(event) => {
                        if (event.button !== 0) return
                        event.preventDefault()
                        event.stopPropagation()
                        void beginWindowResize(edge, event.screenX, event.screenY)
                    }}
                />
            ))}
        </div>
    )
}

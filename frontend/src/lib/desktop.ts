/**
 * The desktop window, from the page's side.
 *
 * The app runs in two places. Inside the pywebview window it draws its own
 * title bar -- the window is created frameless -- and drives the real window
 * through `window.pywebview.api.window_*`, implemented in
 * `anylearning/window_chrome/`. In a plain browser (`pnpm dev`, or the backend
 * run with `--server`) none of that exists and there is no window to control,
 * so every entry point here answers "is this the desktop app?" first and does
 * nothing when it is not.
 */

export type DesktopPlatform = "macos" | "windows" | "linux"

export type ResizeEdge = "top" | "topleft" | "topright" | "left" | "right" | "bottom" | "bottomleft" | "bottomright"

/**
 * The class that marks a surface as title bar.
 *
 * It is pywebview's own default drag selector, so these surfaces keep working
 * through its JS drag path when the native one is unavailable -- a Windows
 * build where the custom frame failed to install, most plausibly.
 *
 * Only a press landing *directly* on the element drags: the backend sets
 * DRAG_REGION_DIRECT_TARGET_ONLY, and the Windows hit test excludes child
 * elements to match. So this belongs on the bar's own background and on inert
 * text, never on a wrapper around the controls.
 */
export const DRAG_REGION = "pywebview-drag-region"

/** Fired at the page whenever the native window is maximised or restored. */
export const WINDOW_STATE_EVENT = "anylearning:window-state"

/** pywebview's platform names, mapped to the platform each one implies. */
const PLATFORMS: Record<string, DesktopPlatform> = {
    cocoa: "macos",
    edgechromium: "windows",
    mshtml: "windows",
    gtkwebkit2: "linux",
    qtwebengine: "linux",
    qtwebkit: "linux",
}

/** Rectangles are handed to Windows in physical pixels, the units it hit-tests in. */
type Rect = [number, number, number, number]

export function isDesktop(): boolean {
    return typeof window !== "undefined" && !!window.pywebview
}

export function desktopPlatform(): DesktopPlatform | null {
    if (typeof window === "undefined") return null
    const platform = window.pywebview?.platform
    return platform ? (PLATFORMS[platform] ?? null) : null
}

/** Whether a DOM press can be handed to a native Linux window-manager drag. */
export function usesNativeDomWindowDrag(): boolean {
    if (typeof window === "undefined") return false
    // The GTK adapter implements begin_move_drag. Qt has no AnyLearning native
    // adapter, so its event must keep bubbling to pywebview's JS drag region.
    return window.pywebview?.platform === "gtkwebkit2"
}

function api() {
    return typeof window !== "undefined" ? window.pywebview?.api : undefined
}

/**
 * Run `callback` once the Python API is reachable.
 *
 * `window.pywebview` is injected before the page's scripts run, but the API
 * object it exposes is built a moment later and announced with `pywebviewready`.
 * Returns a cleanup function, so a React effect can drop the listener.
 */
export function onDesktopReady(callback: () => void): () => void {
    if (typeof window === "undefined") return () => {}
    if (window.pywebview?.api) {
        callback()
        return () => {}
    }
    const handler = () => callback()
    window.addEventListener("pywebviewready", handler, { once: true })
    return () => window.removeEventListener("pywebviewready", handler)
}

export async function windowState(): Promise<{ maximized: boolean } | null> {
    const state = await api()?.window_chrome_state?.()
    return state ?? null
}

export function minimizeWindow(): void {
    void api()?.window_minimize?.()
}

export function toggleWindowMaximized(): void {
    void api()?.window_toggle_maximize?.()
}

export function closeWindow(): void {
    void api()?.window_close?.()
}

/**
 * Hand a title-bar press to the window manager, where the platform can.
 *
 * Resolves false when the platform has no native drag -- macOS, or a Windows
 * build whose custom frame did not install -- and the caller must then leave
 * the event alone so pywebview's own drag region can pick it up.
 */
export async function beginWindowDrag(x: number, y: number): Promise<boolean> {
    return (await api()?.window_begin_drag?.(x, y)) ?? false
}

export async function beginWindowResize(edge: ResizeEdge, x: number, y: number): Promise<boolean> {
    return (await api()?.window_begin_resize?.(edge, x, y)) ?? false
}

function rectOf(element: Element, scale: number): Rect {
    const { left, top, right, bottom } = element.getBoundingClientRect()
    return [left * scale, top * scale, right * scale, bottom * scale]
}

/**
 * Anything that must stay clickable even where it overlaps a drag surface:
 * the window controls, and the full-screen cover a modal puts over the app.
 *
 * The overlays used to suppress every drag surface instead, on the reasoning
 * that an inert app should not be draggable. That was wrong twice over -- it
 * relied on spotting an open dialog by role, even though dialogs can stay
 * mounted while closed, so the title bar silently stopped dragging on Windows.
 * Excluding the rectangle the overlay actually covers says the same thing
 * without guessing.
 */
const NO_DRAG = [
    "[data-window-no-drag]",
    '[data-slot="dialog-overlay"]',
    '[data-slot="alert-dialog-overlay"]',
    '[data-slot="sheet-overlay"]',
].join(", ")

let lastReport = ""

/**
 * Tell Windows where the title bar is.
 *
 * Only Windows asks: its custom frame answers WM_NCHITTEST from geometry and
 * cannot consult the DOM. Exclusions are every element *inside* a drag surface
 * -- which keeps the controls in the bar clickable and matches the
 * DIRECT_TARGET_ONLY rule the other platforms drag by -- plus anything marked
 * `data-window-no-drag`, for the controls that float above the bar rather than
 * sit in it, and any modal overlay covering it. Rectangles are scaled to
 * physical pixels, the units the window is measured in.
 */
export function reportDragRegions(): void {
    const send = api()?.window_set_drag_regions
    if (!send) return

    const scale = window.devicePixelRatio || 1
    const regions: Rect[] = []
    const exclusions: Rect[] = []

    document.querySelectorAll(`.${DRAG_REGION}`).forEach((surface) => {
        regions.push(rectOf(surface, scale))
        // Everything inside a drag surface that is not itself one. Drag
        // surfaces nest -- the bar is one, and so is the project name inside
        // it -- and excluding them along with the buttons would punch holes in
        // the bar they sit in.
        surface.querySelectorAll(`*:not(.${DRAG_REGION})`).forEach((child) => exclusions.push(rectOf(child, scale)))
    })
    document.querySelectorAll(NO_DRAG).forEach((element) => exclusions.push(rectOf(element, scale)))

    // Layout settles far more often than it moves the title bar -- every
    // training log line is a DOM mutation -- and each call over the bridge
    // costs a thread on the Python side.
    const report = JSON.stringify([regions, exclusions])
    if (report === lastReport) return
    lastReport = report

    void send(regions, exclusions)
}

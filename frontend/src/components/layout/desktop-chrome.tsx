"use client"

import { useEffect } from "react"

import { WindowControls } from "@/components/layout/window-controls"
import { WindowResizeEdges } from "@/components/layout/window-resize-edges"
import { useWindowChrome } from "@/hooks/useWindowChrome"
import {
    DRAG_REGION,
    WINDOW_STATE_EVENT,
    beginWindowDrag,
    desktopPlatform,
    onDesktopReady,
    reportDragRegions,
    toggleWindowMaximized,
    windowState,
} from "@/lib/desktop"

/**
 * The app's side of the window frame it draws instead of the platform's.
 *
 * Mounted once, at the root, because the window is one thing and every route
 * shares it: the controls have to be reachable on a page that has no workbench
 * bar as much as on one that does. It does four jobs -- work out which shell
 * this is, keep the maximised state honest, start the gestures the DOM has to
 * start, and tell Windows where the title bar ended up.
 *
 * In a browser it does nothing and renders nothing.
 */
export function DesktopChrome() {
    const platform = useWindowChrome((state) => state.platform)
    const setPlatform = useWindowChrome((state) => state.setPlatform)
    const setMaximized = useWindowChrome((state) => state.setMaximized)

    // Which shell is this? `data-desktop` carries the answer into CSS, where
    // the shell reserves the corner each platform needs (see globals.css).
    useEffect(
        () =>
            onDesktopReady(() => {
                const detected = desktopPlatform()
                setPlatform(detected)
                if (detected) document.documentElement.dataset.desktop = detected
                void windowState().then((state) => state && setMaximized(state.maximized))
            }),
        [setPlatform, setMaximized]
    )

    // The window manager has the last word on whether the window is maximised:
    // Win+Up, a drag to the top of the screen and a double click on the bar all
    // change it without going through our buttons.
    useEffect(() => {
        const handler = (event: Event) => {
            const detail = (event as CustomEvent<{ maximized: boolean }>).detail
            if (detail) setMaximized(detail.maximized)
        }
        window.addEventListener(WINDOW_STATE_EVENT, handler)
        return () => window.removeEventListener(WINDOW_STATE_EVENT, handler)
    }, [setMaximized])

    // Title-bar gestures that only the page can start. Windows has none: its
    // hit test reports the bar as the caption, so the press never reaches the
    // DOM and the drag, the double click and the snap are all the system's.
    useEffect(() => {
        if (platform !== "macos" && platform !== "linux") return

        const onMouseDown = (event: MouseEvent) => {
            const target = event.target
            if (event.button !== 0 || !(target instanceof Element)) return
            if (!target.classList.contains(DRAG_REGION)) return

            if (event.detail >= 2) {
                // The second press of a double click. Read here rather than
                // from a dblclick listener because on Linux the first press
                // has already handed the pointer to the window manager, and
                // the click that would complete the pair never arrives.
                event.preventDefault()
                event.stopPropagation()
                toggleWindowMaximized()
                return
            }

            if (platform !== "linux") return
            // GTK moves the window properly -- snapping and tiling included --
            // so stop the event here before pywebview's own handler on <body>
            // starts walking it across the screen a bridged mousemove at a
            // time. macOS has no equivalent call, and keeps that JS path.
            event.preventDefault()
            event.stopPropagation()
            void beginWindowDrag(event.screenX, event.screenY)
        }

        document.addEventListener("mousedown", onMouseDown, true)
        return () => document.removeEventListener("mousedown", onMouseDown, true)
    }, [platform])

    // Windows hit-tests geometry and cannot see the DOM, so the page has to
    // keep it posted: a collapsed sidebar, a loaded project name and a resized
    // window all move the bar.
    useEffect(() => {
        if (platform !== "windows") return

        let timer = 0
        const schedule = () => {
            window.clearTimeout(timer)
            timer = window.setTimeout(reportDragRegions, 200)
        }

        schedule()
        window.addEventListener("resize", schedule)
        const observer = new MutationObserver(schedule)
        observer.observe(document.body, { subtree: true, childList: true, attributes: true })

        return () => {
            window.clearTimeout(timer)
            window.removeEventListener("resize", schedule)
            observer.disconnect()
        }
    }, [platform])

    if (!platform) return null

    return (
        <>
            {/* Fixed rather than parked in the workbench bar: routes without a
                bar -- a 404, say -- still have to be closable, and the
                labelling screen replaces the shell wholesale. It lines up with
                the centre line of whichever bar is beneath it, and that bar
                reserves the width it takes. */}
            <WindowControls className="fixed top-0 right-2 z-[70] h-[var(--titlebar-height)]" />
            {platform === "linux" && <WindowResizeEdges />}
        </>
    )
}

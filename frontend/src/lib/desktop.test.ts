import {
    desktopPlatform,
    onDesktopReady,
    usesNativeDomWindowDrag,
    usesNativeDragRegionHitTest,
    windowState,
} from "./desktop"

describe("desktop renderer window dragging", () => {
    afterEach(() => {
        delete window.pywebview
    })

    it("uses the native gesture only for the GTK renderer", () => {
        window.pywebview = { platform: "gtkwebkit2" }

        expect(desktopPlatform()).toBe("linux")
        expect(usesNativeDomWindowDrag()).toBe(true)
        expect(usesNativeDragRegionHitTest()).toBe(false)
    })

    it("uses Qt's native system gesture instead of bridged mousemoves", () => {
        window.pywebview = { platform: "qtwebengine" }

        expect(desktopPlatform()).toBe("linux")
        expect(usesNativeDomWindowDrag()).toBe(true)
        expect(usesNativeDragRegionHitTest()).toBe(true)
    })

    it("also supports the older Qt WebKit renderer", () => {
        window.pywebview = { platform: "qtwebkit" }

        expect(desktopPlatform()).toBe("linux")
        expect(usesNativeDomWindowDrag()).toBe(true)
        expect(usesNativeDragRegionHitTest()).toBe(true)
    })

    it("waits until pywebview has populated the callable API", () => {
        const ready = jest.fn()
        window.pywebview = { platform: "qtwebengine", api: {} }

        const cleanup = onDesktopReady(ready)
        expect(ready).not.toHaveBeenCalled()

        window.pywebview.api!.window_chrome_state = async () => ({
            maximized: false,
            native_frame: true,
        })
        window.dispatchEvent(new Event("pywebviewready"))
        expect(ready).toHaveBeenCalledTimes(1)
        cleanup()
    })

    it("recovers when the ready event preceded React hydration", () => {
        jest.useFakeTimers()
        const ready = jest.fn()
        window.pywebview = { platform: "qtwebengine", api: {} }
        const cleanup = onDesktopReady(ready)

        window.pywebview.api!.window_chrome_state = async () => ({
            maximized: false,
            native_frame: true,
        })
        jest.advanceTimersByTime(50)

        expect(ready).toHaveBeenCalledTimes(1)
        cleanup()
        jest.useRealTimers()
    })

    it("reads whether the platform owns the frame", async () => {
        window.pywebview = {
            platform: "qtwebengine",
            api: {
                window_chrome_state: async () => ({
                    maximized: false,
                    native_frame: true,
                }),
            },
        }

        await expect(windowState()).resolves.toEqual({
            maximized: false,
            native_frame: true,
        })
    })
})

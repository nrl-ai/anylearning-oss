import { desktopPlatform, usesNativeDomWindowDrag } from "./desktop"

describe("desktop renderer window dragging", () => {
    afterEach(() => {
        delete window.pywebview
    })

    it("uses the native gesture only for the GTK renderer", () => {
        window.pywebview = { platform: "gtkwebkit2" }

        expect(desktopPlatform()).toBe("linux")
        expect(usesNativeDomWindowDrag()).toBe(true)
    })

    it("leaves Qt presses to pywebview's drag-region fallback", () => {
        window.pywebview = { platform: "qtwebengine" }

        expect(desktopPlatform()).toBe("linux")
        expect(usesNativeDomWindowDrag()).toBe(false)
    })
})

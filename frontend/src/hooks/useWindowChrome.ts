import { create } from "zustand"

import { DesktopPlatform } from "@/lib/desktop"

interface WindowChromeStore {
    /**
     * Which platform's frame the app is standing in for, or null in a browser,
     * where there is no window to control. Null until `<DesktopChrome>` has
     * mounted and asked -- the static export renders before it can know.
     */
    platform: DesktopPlatform | null
    maximized: boolean
    setPlatform: (platform: DesktopPlatform | null) => void
    setMaximized: (maximized: boolean) => void
}

/**
 * The state the app-drawn title bar renders from.
 *
 * Not persisted, unlike the sidebar's: it describes the window this run is
 * living in, and restoring yesterday's answer would just be a guess.
 */
export const useWindowChrome = create<WindowChromeStore>()((set) => ({
    platform: null,
    maximized: false,
    setPlatform: (platform) => set({ platform }),
    setMaximized: (maximized) => set({ maximized }),
}))

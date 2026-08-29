import { create } from "zustand"
import { persist } from "zustand/middleware"

interface SidebarStore {
    isMinimized: boolean
    toggle: () => void
    expand: () => void
    collapse: () => void
    hoverState: boolean
    setHoverState: (state: boolean) => void
}

export const useSidebar = create<SidebarStore>()(
    persist(
        (set) => ({
            isMinimized: false,
            toggle: () => set((state) => ({ isMinimized: !state.isMinimized })),
            expand: () => set({ isMinimized: false }),
            collapse: () => set({ isMinimized: true }),
            hoverState: false,
            setHoverState: (state: boolean) => set({ hoverState: state }),
        }),
        {
            name: "sidebar-state",
        }
    )
)

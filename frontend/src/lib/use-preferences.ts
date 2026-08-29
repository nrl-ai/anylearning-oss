import { create } from "zustand"
import { persist } from "zustand/middleware"

export type Preferences = {
    /** How many thumbnails the dataset grid shows per page. */
    gridPageSize: number
    /** Rows per page in the models table. */
    modelsPageSize: number
    setGridPageSize: (value: number) => void
    setModelsPageSize: (value: number) => void
}

/**
 * Preferences that belong to this machine rather than to a project.
 *
 * Kept in localStorage, not on the backend: they describe how this person likes
 * to work, and the backend has no notion of a user.
 */
export const usePreferences = create<Preferences>()(
    persist(
        (set) => ({
            gridPageSize: 20,
            modelsPageSize: 5,
            setGridPageSize: (gridPageSize) => set({ gridPageSize }),
            setModelsPageSize: (modelsPageSize) => set({ modelsPageSize }),
        }),
        { name: "anylearning-preferences" }
    )
)

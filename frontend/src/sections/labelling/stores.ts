import { create } from "zustand"
import { persist } from "zustand/middleware"

export type AutoSaveSettingStoreState = {
    isEnabled: boolean
    toggleEnabled: () => void
    setEnabled: (value: boolean) => void
}

/**
 * Persisted: this is a preference, and losing it on every launch meant users
 * who deliberately turn auto-save off had to turn it off again each session.
 */
export const useAutoSaveSettingStore = create<AutoSaveSettingStoreState>()(
    persist(
        (set) => ({
            isEnabled: true,
            toggleEnabled() {
                set((state) => ({ ...state, isEnabled: !state.isEnabled }))
            },
            setEnabled(value) {
                set((state) => ({ ...state, isEnabled: value }))
            },
        }),
        { name: "anylearning-auto-save" }
    )
)

import { create } from "zustand"
import { persist } from "zustand/middleware"

export type SettingStoreState = {
    isShowLabels: boolean
    isShowKeypointInstances: boolean
    isShowKeypointVisibility: boolean
    isDimOccludedKeypoints: boolean
    setShowLabels: (value: boolean) => void
    setShowKeypointInstances: (value: boolean) => void
    setShowKeypointVisibility: (value: boolean) => void
    setDimOccludedKeypoints: (value: boolean) => void
}

/** Canvas annotation display preferences. Persisted per browser. */
export const useSettingStore = create<SettingStoreState>()(
    persist(
        (set) => ({
            isShowLabels: false,
            // Instance membership is essential context for keypoints, so show
            // it even when the more verbose landmark names are hidden.
            isShowKeypointInstances: true,
            isShowKeypointVisibility: true,
            isDimOccludedKeypoints: true,
            setShowLabels(value) {
                set((state) => ({ ...state, isShowLabels: value }))
            },
            setShowKeypointInstances(value) {
                set((state) => ({ ...state, isShowKeypointInstances: value }))
            },
            setShowKeypointVisibility(value) {
                set((state) => ({ ...state, isShowKeypointVisibility: value }))
            },
            setDimOccludedKeypoints(value) {
                set((state) => ({ ...state, isDimOccludedKeypoints: value }))
            },
        }),
        // Keep the original storage key so an existing "show names" choice
        // survives the addition of the keypoint-specific fields.
        { name: "anylearning-show-labels" }
    )
)

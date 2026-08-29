import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import { api, getJson } from "@/lib/api"
import { qk } from "@/lib/query-keys"

export type PerformanceMode = "maximum" | "balanced" | "power_saving"

export type AppSettings = {
    performance_mode: PerformanceMode
    training_num_workers: number | "auto"
    training_pin_memory: boolean | "auto"
    training_persistent_workers: boolean | "auto"
    resolved: {
        cpu_count: number
        physical_cores: number
        num_workers_gpu: number
        num_workers_cpu: number
        cudnn_benchmark: boolean
    }
}

/**
 * Machine-level settings, which live on the backend rather than in
 * localStorage: the training process reads them, and it has no browser.
 *
 * `resolved` is what the current choice actually works out to on this machine,
 * so the UI can say "Maximum (8 workers)" instead of leaving the user to guess.
 */
export function useAppSettings() {
    const queryClient = useQueryClient()

    const { data, isLoading, error } = useQuery({
        queryKey: qk.appSettings(),
        queryFn: () => getJson<AppSettings>("/api/settings"),
    })

    const update = useMutation({
        mutationFn: (changes: Partial<Omit<AppSettings, "resolved">>) =>
            api.put("/api/settings", changes).then((r) => r.data as AppSettings),
        onSuccess: (fresh) => queryClient.setQueryData(qk.appSettings(), fresh),
    })

    return {
        settings: data,
        isLoading,
        error,
        setPerformanceMode: (performance_mode: PerformanceMode) => update.mutateAsync({ performance_mode }),
        isSaving: update.isPending,
    }
}

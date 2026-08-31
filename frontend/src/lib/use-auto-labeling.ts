import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useCallback } from "react"

import { api, getJson } from "@/lib/api"
import { POLL, qk } from "@/lib/query-keys"

export interface AutoLabelingModel {
    name: string
    display_name: string
    has_downloaded: boolean
    is_custom_model: boolean
    tasks: ("detection" | "instance_segmentation" | "promptable_segmentation")[]
    interaction_mode: "prompted" | "automatic"
    output_modes: ("polygon" | "rectangle")[]
    project_types: string[]
    archive_size_bytes: number
    is_project_model?: boolean
}

interface AutoLabelingStatus {
    status: string
    model_name: string | null
}

const isBusyStatus = (status: string): boolean => /^(Loading model:|Downloading |Queued )/i.test(status)

const useAutoLabeling = (projectId: number | null) => {
    const queryClient = useQueryClient()
    const enabled = projectId !== null

    const { data: modelsData, error: modelsError } = useQuery({
        queryKey: qk.autoLabelingModels(projectId as number),
        queryFn: () => getJson<AutoLabelingModel[]>(`/api/projects/${projectId}/auto_labeling/models`),
        enabled,
        staleTime: 60_000,
    })

    const {
        data: statusData,
        error: statusError,
        refetch: refetchStatus,
    } = useQuery({
        queryKey: qk.autoLabelingStatus(projectId as number),
        queryFn: () => getJson<AutoLabelingStatus>(`/api/projects/${projectId}/auto_labeling/status`),
        enabled,
        // The backend downloads and loads model weights in the background, and
        // the status string is the only progress the user gets. Poll while it
        // is working, then stop.
        refetchInterval: (query) => {
            const status = (query.state.data as AutoLabelingStatus | undefined)?.status ?? ""
            return isBusyStatus(status) ? POLL.autoLabeling : false
        },
    })

    const {
        mutateAsync: loadModelAsync,
        isPending: isLoadPending,
        error: modelLoadError,
    } = useMutation({
        mutationFn: (modelName: string) =>
            api.post(`/api/projects/${projectId}/auto_labeling/load_model`, { model_name: modelName }),
        onSuccess: () => queryClient.invalidateQueries({ queryKey: qk.autoLabelingStatus(projectId as number) }),
    })

    const loadModel = useCallback(
        async (modelName: string) => {
            if (!enabled) return
            await loadModelAsync(modelName)
        },
        [enabled, loadModelAsync]
    )

    return {
        models: modelsData || [],
        status: statusData?.status || "",
        loadedModel: statusData?.model_name ?? null,
        loadModel,
        loadError: modelLoadError instanceof Error ? modelLoadError.message : "",
        isLoadingModel: isLoadPending || isBusyStatus(statusData?.status ?? ""),
        isLoading: enabled ? !modelsData && !modelsError : false,
        isError: enabled ? modelsError || statusError : null,
        mutateStatus: refetchStatus,
    }
}

export default useAutoLabeling

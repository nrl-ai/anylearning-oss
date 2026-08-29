import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import { api, getJson } from "@/lib/api"
import { POLL, qk } from "@/lib/query-keys"

export interface AutoLabelingModel {
    name: string
    display_name: string
    has_downloaded: boolean
    is_custom_model: boolean
}

interface AutoLabelingStatus {
    status: string
}

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
            return /download|load|prepar/i.test(status) ? POLL.autoLabeling : false
        },
    })

    const loadModelMutation = useMutation({
        mutationFn: (modelName: string) =>
            api.post(`/api/projects/${projectId}/auto_labeling/load_model`, { model_name: modelName }),
        onSuccess: () => queryClient.invalidateQueries({ queryKey: qk.autoLabelingStatus(projectId as number) }),
    })

    return {
        models: modelsData || [],
        status: statusData?.status || "",
        loadModel: async (modelName: string) => {
            if (!enabled) return
            await loadModelMutation.mutateAsync(modelName)
        },
        isLoading: enabled ? !modelsData && !modelsError : false,
        isError: enabled ? modelsError || statusError : null,
        mutateStatus: refetchStatus,
    }
}

export default useAutoLabeling

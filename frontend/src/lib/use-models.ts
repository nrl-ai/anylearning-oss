import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useCallback, useState } from "react"

import { api, getJson } from "@/lib/api"
import { POLL, qk } from "@/lib/query-keys"
import { Model } from "@/types"

type ModelPage = { models: Model[]; total_count: number }

type ListParams = {
    offset: number
    limit: number
    search: string
    model_architecture: string
    model_size: string
}

const useModels = (projectId: number | null) => {
    const queryClient = useQueryClient()
    const enabled = projectId !== null
    const [params, setParams] = useState<ListParams>({
        offset: 0,
        limit: 20,
        search: "",
        model_architecture: "",
        model_size: "",
    })

    const { data, error, isLoading } = useQuery({
        queryKey: qk.modelList(projectId as number, params),
        queryFn: () =>
            getJson<ModelPage>(`/api/projects/${projectId}/models`, {
                params: {
                    offset: params.offset,
                    limit: params.limit,
                    ...(params.search ? { search: params.search } : {}),
                    ...(params.model_architecture ? { model_architecture: params.model_architecture } : {}),
                    ...(params.model_size ? { model_size: params.model_size } : {}),
                },
            }),
        enabled,
        placeholderData: (previous) => previous,
        // A model only appears when a training run finishes, and that happens
        // in another process.
        refetchInterval: POLL.models,
    })

    const fetchModels = useCallback(
        (
            offset: number = -1,
            limit: number = -1,
            search: string = "",
            model_architecture: string = "",
            model_size: string = ""
        ) => {
            setParams((prev) => ({
                offset: offset === -1 ? prev.offset : offset,
                limit: limit === -1 ? prev.limit : limit,
                search,
                model_architecture,
                model_size,
            }))
        },
        []
    )

    const invalidateModels = useCallback(() => {
        if (projectId === null) return
        queryClient.invalidateQueries({ queryKey: qk.models(projectId) })
    }, [queryClient, projectId])

    const updateMutation = useMutation({
        mutationFn: ({ modelId, data: body }: { modelId: number; data: Partial<Model> }) =>
            api.put(`/api/projects/${projectId}/models/${modelId}`, body).then((r) => r.data as Model),
        onSuccess: invalidateModels,
    })

    const deleteMutation = useMutation({
        mutationFn: (modelId: number) => api.delete(`/api/projects/${projectId}/models/${modelId}`),
        onSuccess: invalidateModels,
    })

    const getModelById = useCallback(
        (modelId: number) =>
            queryClient.fetchQuery({
                queryKey: [...qk.models(projectId as number), modelId],
                queryFn: () => getJson<Model>(`/api/projects/${projectId}/models/${modelId}`),
            }),
        [queryClient, projectId]
    )

    return {
        models: data?.models ?? [],
        totalCount: data?.total_count ?? 0,
        fetchModels,
        invalidateModels,
        getModelById,
        updateModel: (modelId: number, updateData: Partial<Model>) =>
            updateMutation.mutateAsync({ modelId, data: updateData }),
        deleteModel: deleteMutation.mutateAsync,
        isLoading: enabled ? isLoading : false,
        isError: enabled && !!error,
        error,
    }
}

export default useModels

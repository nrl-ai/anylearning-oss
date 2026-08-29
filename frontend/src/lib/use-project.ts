import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import axios from "axios"
import { useEffect } from "react"

import { api, getJson } from "@/lib/api"
import { qk } from "@/lib/query-keys"
import { Project } from "@/types"

export default function useProject(projectId: number | null) {
    const queryClient = useQueryClient()

    const {
        data: project,
        error,
        isLoading,
        refetch,
    } = useQuery({
        queryKey: qk.project(projectId as number),
        queryFn: () => getJson<Project>(`/api/projects/${projectId}`),
        enabled: projectId !== null,
    })

    const { data: modelVariants } = useQuery({
        queryKey: qk.modelVariants(),
        queryFn: () => getJson<Record<string, any[]>>("/api/model-variants"),
        // Model variants are compiled into the build; they never change while
        // the app is open.
        staleTime: Infinity,
    })

    const updateMutation = useMutation({
        mutationFn: (projectData: Partial<Project>) =>
            api.patch(`/api/projects/${projectId}`, projectData).then((r) => r.data as Project),
        onSuccess: (data) => {
            queryClient.setQueryData(qk.project(projectId as number), data)
            queryClient.invalidateQueries({ queryKey: qk.projects() })
        },
        onError: (err) => {
            // A project deleted in another window is gone, not broken.
            if (axios.isAxiosError(err) && err.response?.status === 404) window.location.href = "/"
        },
    })

    const deleteMutation = useMutation({
        mutationFn: () => api.delete(`/api/projects/${projectId}`),
        onSuccess: () => {
            queryClient.removeQueries({ queryKey: qk.project(projectId as number) })
            queryClient.invalidateQueries({ queryKey: qk.projects() })
        },
    })

    // Handle 404 from the initial fetch the same way. In an effect, not in the
    // render body: navigating during render is a side effect React may run more
    // than once (and does under StrictMode), and it touches `window`, which does
    // not exist during the prerender the static export performs at build time.
    useEffect(() => {
        if (error && axios.isAxiosError(error) && error.response?.status === 404) {
            window.location.href = "/"
        }
    }, [error])

    const availableModelVariants = project?.type && modelVariants ? modelVariants[project.type] || [] : []
    const loading = isLoading && projectId !== null

    return {
        project,
        loading,
        isLoading: loading,
        error,
        refetch,
        update: updateMutation.mutateAsync,
        updateProject: updateMutation.mutateAsync,
        deleteProject: deleteMutation.mutateAsync,
        modelVariants: availableModelVariants,
    }
}

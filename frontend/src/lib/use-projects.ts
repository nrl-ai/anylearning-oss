import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import { api, getJson } from "@/lib/api"
import { qk } from "@/lib/query-keys"
import { Project, ProjectCreation } from "@/types"

export function useProjects() {
    const queryClient = useQueryClient()

    const {
        data: projects,
        error,
        isLoading,
        refetch,
    } = useQuery({
        queryKey: qk.projects(),
        queryFn: () => getJson<Project[]>("/api/projects"),
    })

    const invalidate = () => queryClient.invalidateQueries({ queryKey: qk.projects() })

    const createMutation = useMutation({
        mutationFn: (projectData: ProjectCreation) =>
            api.post("/api/projects", projectData).then((r) => r.data as Project),
        onSuccess: invalidate,
    })

    const updateMutation = useMutation({
        mutationFn: ({ id, data }: { id: number; data: Partial<Project> }) =>
            api.patch(`/api/projects/${id}`, data).then((r) => r.data as Project),
        onSuccess: (_data, { id }) => {
            invalidate()
            queryClient.invalidateQueries({ queryKey: qk.project(id) })
        },
    })

    const deleteMutation = useMutation({
        mutationFn: (id: number) => api.delete(`/api/projects/${id}`),
        onSuccess: invalidate,
    })

    return {
        projects,
        loading: isLoading,
        error,
        /** Kept for call sites that still ask for a manual refresh. */
        mutate: invalidate,
        refetch,
        createProject: createMutation.mutateAsync,
        updateProject: (id: number, data: Partial<Project>) => updateMutation.mutateAsync({ id, data }),
        deleteProject: deleteMutation.mutateAsync,
        isCreating: createMutation.isPending,
        isDeleting: deleteMutation.isPending,
    }
}

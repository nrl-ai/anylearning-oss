import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import { api, getJson, getJsonOrNull } from "@/lib/api"
import { POLL, qk } from "@/lib/query-keys"
import { isActiveStatus } from "@/lib/status"
import { DetailedTrainingSession, TrainingParams, TrainingResponse, TrainingSession } from "@/types"

export const useTraining = (projectId: number) => {
    const queryClient = useQueryClient()

    const {
        data: lastTrainingSession,
        error: lastSessionError,
        isLoading: lastLoading,
    } = useQuery({
        queryKey: qk.lastTrainingSession(projectId),
        queryFn: () => getJsonOrNull<DetailedTrainingSession>(`/api/projects/${projectId}/last_training_session`),
        enabled: Number.isFinite(projectId),
        // Training runs in a separate process and reports progress by writing
        // to the project database, so the only way to see it is to poll -- but
        // only while something is actually running. The old code polled every
        // 10s for the life of the screen even on a project that had never
        // trained.
        refetchInterval: (query) =>
            isActiveStatus((query.state.data as DetailedTrainingSession | null)?.status) ? POLL.training : false,
    })

    const isRunning = isActiveStatus(lastTrainingSession?.status)

    const {
        data: trainingSessions,
        error: sessionsError,
        isLoading,
    } = useQuery({
        queryKey: qk.trainingSessions(projectId),
        queryFn: () => getJson<TrainingSession[]>(`/api/projects/${projectId}/training_sessions`),
        enabled: Number.isFinite(projectId),
        refetchInterval: isRunning ? POLL.training : false,
    })

    /** Refreshes everything a run can change, including the models it produces. */
    const invalidateRun = () => {
        queryClient.invalidateQueries({ queryKey: qk.trainingSessions(projectId) })
        queryClient.invalidateQueries({ queryKey: qk.lastTrainingSession(projectId) })
        queryClient.invalidateQueries({ queryKey: qk.models(projectId) })
    }

    const startMutation = useMutation({
        mutationFn: (params: TrainingParams) =>
            api.post<TrainingResponse>(`/api/projects/${projectId}/training_sessions`, params).then((r) => r.data),
        onSuccess: invalidateRun,
    })

    const terminateMutation = useMutation({
        mutationFn: (sessionId: number) =>
            api.post(`/api/projects/${projectId}/training_sessions/${sessionId}/terminate`).then((r) => r.data),
        onSuccess: invalidateRun,
    })

    /** One session with its logs. Cached so reopening the dialog is instant. */
    const getTrainingSession = async (sessionId: number) =>
        queryClient.fetchQuery({
            queryKey: qk.trainingSession(projectId, sessionId),
            queryFn: () =>
                getJson<DetailedTrainingSession>(`/api/projects/${projectId}/training_sessions/${sessionId}`),
            staleTime: 0,
        })

    return {
        startTraining: async (params: TrainingParams) => {
            try {
                return await startMutation.mutateAsync(params)
            } catch (err) {
                console.error("Error starting training:", err)
                return null
            }
        },
        terminateTraining: async (sessionId: number) => {
            try {
                return await terminateMutation.mutateAsync(sessionId)
            } catch (err) {
                console.error("Error terminating training:", err)
                return null
            }
        },
        getTrainingSession,
        trainingSessions,
        lastTrainingSession,
        isLoading: isLoading || lastLoading,
        error: sessionsError || lastSessionError,
    }
}

export default useTraining

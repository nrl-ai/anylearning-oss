/**
 * Every React Query key in the app, in one place.
 *
 * Keys are hierarchical so a mutation can invalidate a whole project's data
 * with `qk.project(id)` without knowing which individual queries exist. Keeping
 * them here is what makes that safe: invalidation written as an inline array
 * literal at the call site drifts from the query that produced it, and the
 * symptom is stale data that only some screens notice.
 */
export const qk = {
    projects: () => ["projects"] as const,
    project: (projectId: number) => ["projects", projectId] as const,
    modelVariants: () => ["model-variants"] as const,
    appSettings: () => ["app-settings"] as const,

    datasets: (projectId: number) => ["projects", projectId, "datasets"] as const,
    classDistribution: (projectId: number) => ["projects", projectId, "class-distribution"] as const,
    dataItems: (projectId: number, subset: number, offset: number, limit: number) =>
        ["projects", projectId, "data-items", subset, offset, limit] as const,
    dataItemsAll: (projectId: number) => ["projects", projectId, "data-items"] as const,
    annotation: (projectId: number, itemId: number) =>
        ["projects", projectId, "data-items", itemId, "annotation"] as const,

    trainingSessions: (projectId: number) => ["projects", projectId, "training-sessions"] as const,
    trainingSession: (projectId: number, sessionId: number) =>
        ["projects", projectId, "training-sessions", sessionId] as const,
    lastTrainingSession: (projectId: number) => ["projects", projectId, "last-training-session"] as const,

    models: (projectId: number) => ["projects", projectId, "models"] as const,
    modelList: (projectId: number, params: Record<string, unknown>) =>
        ["projects", projectId, "models", "list", params] as const,

    autoLabelingModels: (projectId: number) => ["projects", projectId, "auto-labeling-models"] as const,
    autoLabelingStatus: (projectId: number) => ["projects", projectId, "auto-labeling-status"] as const,

}

/** How often the app polls things that a background process can change. */
export const POLL = {
    /** Training writes progress to the DB from another process. */
    training: 5_000,
    /** Models only appear when a run finishes. */
    models: 30_000,
    /** Model download / load progress while auto-labelling. */
    autoLabeling: 2_000,
} as const

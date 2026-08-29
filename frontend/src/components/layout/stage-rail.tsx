"use client"

import { useQuery } from "@tanstack/react-query"
import { AlertTriangle } from "lucide-react"

import { getJson } from "@/lib/api"
import { isStructuredProject, structuredTaskLabel } from "@/lib/project-types"
import { qk } from "@/lib/query-keys"
import { isActiveStatus } from "@/lib/status"
import useDatasets from "@/lib/use-datasets"
import { useTraining } from "@/lib/use-training"
import { cn } from "@/lib/utils"
import { Project } from "@/types"

/**
 * The stage rail — the app's signature element and its primary navigation.
 *
 * Getting a model out of AnyLearning is a real sequence with real
 * preconditions: you cannot train without labelled data, and you cannot have a
 * model without a finished run. The rail reports where the project actually
 * stands rather than where you happen to be looking. The previous stepper drew
 * a green check on every stage behind the current tab, so a brand-new project
 * with no images and no models still claimed three stages complete.
 *
 * Colour discipline (see DESIGN.md): completion is shown by *fill*, not hue.
 * Only two things earn colour here — a live process (amber, breathing) and a
 * failure (red) — plus the mark on the stage you are viewing.
 */

export type StageId = "overview" | "dataset" | "training" | "models"

type StageState = "empty" | "partial" | "running" | "done" | "failed"

type Stage = {
    id: StageId
    label: string
    /** How much of this stage is satisfied, 0–1. Drives the segment fill. */
    progress: number
    state: StageState
    /** One line of real numbers, shown under the label. */
    detail: string
}

/** Reads live project state and describes each stage. */
export function useStages(projectId: number | null, project?: Project | null): Stage[] {
    const structured = isStructuredProject(project?.type)
    const { datasets } = useDatasets(projectId)
    const { trainingSessions, lastTrainingSession } = useTraining(projectId as number)
    const { data: modelsData } = useQuery({
        queryKey: qk.models(projectId as number),
        queryFn: () => getJson<{ total_count: number }>(`/api/projects/${projectId}/models`),
        enabled: projectId !== null,
    })
    const { data: structuredData } = useQuery({
        queryKey: ["structured", projectId],
        queryFn: () =>
            getJson<{
                configured: boolean
                source?: { rows: number; columns: number }
                task?: { type: string }
            }>(`/api/projects/${projectId}/structured`),
        enabled: projectId !== null && structured,
    })

    const labelCount = project?.labels?.length ?? 0
    const totals = Object.values(datasets).reduce(
        (acc, d) => ({
            total: acc.total + (d.info?.num_total ?? 0),
            labeled: acc.labeled + (d.info?.num_labeled ?? 0),
        }),
        { total: 0, labeled: 0 }
    )
    const modelCount = modelsData?.total_count ?? 0
    const sessionCount = trainingSessions?.length ?? 0
    const running = isActiveStatus(lastTrainingSession?.status)
    const lastFailed = lastTrainingSession?.status?.toLowerCase() === "error"

    const epochsDone = lastTrainingSession?.metric_logs ? Object.keys(lastTrainingSession.metric_logs).length : 0
    const epochsTotal = lastTrainingSession?.params?.epochs ?? 0

    return [
        {
            id: "overview",
            label: "Overview",
            progress: structured ? (structuredData?.configured ? 1 : 0.5) : labelCount > 0 ? 1 : 0,
            state: structured ? (structuredData?.configured ? "done" : "partial") : labelCount > 0 ? "done" : "empty",
            detail: structured
                ? structuredData?.configured
                    ? structuredTaskLabel(structuredData.task?.type)
                    : "Choose a workflow"
                : labelCount > 0
                  ? plural(labelCount, "label")
                  : "No labels yet",
        },
        {
            id: "dataset",
            label: "Dataset",
            progress: structured
                ? structuredData?.source
                    ? 1
                    : 0
                : totals.total === 0
                  ? 0
                  : totals.labeled / totals.total,
            state: structured
                ? structuredData?.source
                    ? "done"
                    : "empty"
                : totals.total === 0
                  ? "empty"
                  : totals.labeled === totals.total
                    ? "done"
                    : "partial",
            detail: structured
                ? structuredData?.source
                    ? plural(structuredData.source.rows, "row")
                    : "No dataset yet"
                : totals.total === 0
                  ? "No images yet"
                  : totals.labeled === totals.total
                    ? plural(totals.total, "image")
                    : `${totals.labeled}/${totals.total} labelled`,
        },
        {
            id: "training",
            label: "Training",
            progress: running
                ? epochsTotal > 0
                    ? Math.min(epochsDone / epochsTotal, 0.99)
                    : 0.1
                : sessionCount > 0
                  ? 1
                  : 0,
            state: running ? "running" : lastFailed ? "failed" : sessionCount > 0 ? "done" : "empty",
            detail: running
                ? epochsTotal > 0
                    ? `${structured ? "Iteration" : "Epoch"} ${epochsDone}/${epochsTotal}`
                    : "Starting…"
                : lastFailed
                  ? "Last run failed"
                  : sessionCount > 0
                    ? plural(sessionCount, "run")
                    : "No runs yet",
        },
        {
            id: "models",
            label: "Models",
            progress: modelCount > 0 ? 1 : 0,
            state: modelCount > 0 ? "done" : "empty",
            detail: modelCount > 0 ? plural(modelCount, "model") : "No models yet",
        },
    ]
}

function plural(n: number, word: string) {
    return `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`
}

const fillFor: Record<StageState, string> = {
    empty: "bg-transparent",
    // Completion is presence, not colour: a finished stage reads as a solid
    // neutral bar so a healthy project stays calm.
    partial: "bg-foreground/35",
    done: "bg-foreground/55",
    running: "bg-run animate-breathe",
    failed: "bg-fail",
}

export function StageRail({
    stages,
    current,
    onSelect,
    className,
}: {
    stages: Stage[]
    current: StageId
    onSelect: (id: StageId) => void
    className?: string
}) {
    return (
        <nav aria-label="Project stages" className={cn("flex items-stretch gap-1", className)}>
            {stages.map((stage) => {
                const isCurrent = stage.id === current
                return (
                    <button
                        key={stage.id}
                        type="button"
                        onClick={() => onSelect(stage.id)}
                        aria-current={isCurrent ? "page" : undefined}
                        className={cn(
                            "group hover:bg-accent/60 min-w-0 flex-1 rounded-md px-2.5 pt-1.5 pb-1 text-left transition-colors",
                            "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none"
                        )}
                    >
                        {/* The segment: a track that fills with real progress. */}
                        <span
                            aria-hidden
                            className={cn(
                                "block h-[3px] w-full overflow-hidden rounded-full",
                                isCurrent ? "bg-mark-border" : "bg-border"
                            )}
                        >
                            <span
                                className={cn(
                                    "block h-full rounded-full transition-[width] duration-500",
                                    isCurrent && stage.state !== "running" && stage.state !== "failed"
                                        ? "bg-mark"
                                        : fillFor[stage.state]
                                )}
                                style={{ width: `${Math.round(stage.progress * 100)}%` }}
                            />
                        </span>
                        <span className="mt-1.5 flex items-center gap-1.5">
                            <span
                                className={cn(
                                    "t-section truncate text-[0.8125rem]",
                                    isCurrent ? "text-mark" : "text-foreground/80 group-hover:text-foreground"
                                )}
                            >
                                {stage.label}
                            </span>
                            {stage.state === "failed" && (
                                <AlertTriangle className="text-fail size-3 shrink-0" strokeWidth={2} />
                            )}
                        </span>
                        <span
                            className={cn(
                                "tabular block truncate font-mono text-[0.6875rem]",
                                stage.state === "running"
                                    ? "text-run"
                                    : stage.state === "failed"
                                      ? "text-fail"
                                      : "text-muted-foreground"
                            )}
                        >
                            {stage.detail}
                        </span>
                    </button>
                )
            })}
        </nav>
    )
}

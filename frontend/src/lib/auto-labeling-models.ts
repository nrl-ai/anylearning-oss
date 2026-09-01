import { AutoLabelingModel } from "@/lib/use-auto-labeling"

export interface AutoLabelingModelGroup {
    label: string
    models: AutoLabelingModel[]
}

const GROUPS = ["Project models", "Interactive segmentation", "Object detection", "Instance segmentation"] as const

function groupName(model: AutoLabelingModel): (typeof GROUPS)[number] {
    if (model.is_project_model) return "Project models"
    if (model.interaction_mode === "prompted") return "Interactive segmentation"
    if (model.tasks.includes("detection")) return "Object detection"
    return "Instance segmentation"
}

/** Keep the catalogue order within a small number of task-oriented sections. */
export function groupAutoLabelingModels(models: AutoLabelingModel[]): AutoLabelingModelGroup[] {
    const grouped = new Map<string, AutoLabelingModel[]>()
    models.forEach((model) => {
        const name = groupName(model)
        grouped.set(name, [...(grouped.get(name) ?? []), model])
    })
    return GROUPS.flatMap((label) => {
        const entries = grouped.get(label)
        return entries?.length ? [{ label, models: entries }] : []
    })
}

export function formatDownloadSize(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) return "Download required"
    const megabytes = bytes / (1024 * 1024)
    if (megabytes < 1024) return `${Math.max(1, Math.round(megabytes))} MB download`
    return `${(megabytes / 1024).toFixed(1)} GB download`
}

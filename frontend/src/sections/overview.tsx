import { useQuery } from "@tanstack/react-query"
import {
    ArrowRight,
    BoxIcon,
    BrainCircuit,
    CheckIcon,
    Database,
    EditIcon,
    FileSearch,
    Scale,
    XIcon,
} from "lucide-react"
import { useEffect, useState } from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Panel, PanelBody, PanelHeader, Stat } from "@/components/ui/panel"
import { getJson } from "@/lib/api"
import { isStructuredProject, projectTypeLabel, structuredTaskLabel } from "@/lib/project-types"
import ClassDistribution from "@/sections/class-distribution"
import DataDistribution from "@/sections/data-distribution"
import LabelListEditor from "@/sections/project/label-list-editor"
import { Project } from "@/types"

/**
 * An editable field: eyebrow, current value, and a pencil that swaps the value
 * for an input in place. Used for both name and description so the two behave
 * identically — the previous screen had two hand-rolled variants that saved on
 * different keys and offered no way to cancel.
 */
function EditableField({
    label,
    value,
    placeholder,
    empty,
    onSave,
}: {
    label: string
    value: string
    placeholder: string
    empty: string
    onSave: (next: string) => void
}) {
    const [isEditing, setIsEditing] = useState(false)
    const [draft, setDraft] = useState(value)

    useEffect(() => {
        setIsEditing(false)
        setDraft(value)
    }, [value])

    const save = () => {
        onSave(draft)
        setIsEditing(false)
    }

    return (
        <div className="min-w-0">
            <p className="t-eyebrow mb-1.5">{label}</p>
            {isEditing ? (
                <div className="flex items-center gap-2">
                    <Input
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === "Enter") save()
                            if (e.key === "Escape") {
                                setDraft(value)
                                setIsEditing(false)
                            }
                        }}
                        className="h-8 flex-grow"
                        placeholder={placeholder}
                        autoFocus
                    />
                    <Button size="icon-sm" aria-label={`Save ${label.toLowerCase()}`} onClick={save}>
                        <CheckIcon />
                    </Button>
                    <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label="Cancel"
                        onClick={() => {
                            setDraft(value)
                            setIsEditing(false)
                        }}
                    >
                        <XIcon />
                    </Button>
                </div>
            ) : (
                <div className="flex min-h-8 items-start justify-between gap-2">
                    <p className="pt-1 text-sm break-words">
                        {value ? value.slice(0, 240) : <span className="text-muted-foreground">{empty}</span>}
                    </p>
                    <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label={`Edit ${label.toLowerCase()}`}
                        onClick={() => setIsEditing(true)}
                    >
                        <EditIcon />
                    </Button>
                </div>
            )}
        </div>
    )
}

function ProjectOverview({
    project,
    onSaveName,
    onSaveDescription,
}: {
    project: Project
    onSaveName: (name: string) => void
    onSaveDescription: (description: string) => void
}) {
    const structured = isStructuredProject(project.type)
    const { data: structuredData } = useQuery({
        queryKey: ["structured", project.id],
        queryFn: () =>
            getJson<{
                configured: boolean
                source?: { rows: number; columns: number; filename: string }
                task?: { type: string }
            }>(`/api/projects/${project.id}/structured`),
        enabled: structured,
    })

    if (structured) {
        return (
            <div className="grid items-start gap-4 lg:grid-cols-3">
                <div className="grid content-start gap-4 lg:col-span-2">
                    <Panel>
                        <PanelHeader icon={BoxIcon} title="Project" />
                        <PanelBody className="space-y-4">
                            <EditableField
                                label="Name"
                                value={project.name ?? ""}
                                placeholder="Project name"
                                empty="Untitled project"
                                onSave={onSaveName}
                            />
                            <EditableField
                                label="Description"
                                value={project.description ?? ""}
                                placeholder="What is this project for?"
                                empty="No description yet."
                                onSave={onSaveDescription}
                            />
                            <div className="grid grid-cols-2 gap-4 border-t pt-4 sm:grid-cols-4">
                                <Stat label="Workspace" value={projectTypeLabel(project.type)} mono={false} />
                                <Stat label="Rows" value={structuredData?.source?.rows?.toLocaleString() ?? "—"} />
                                <Stat
                                    label="Columns"
                                    value={structuredData?.source?.columns?.toLocaleString() ?? "—"}
                                />
                                <Stat
                                    label="Workflow"
                                    value={structuredTaskLabel(structuredData?.task?.type)}
                                    mono={false}
                                />
                            </div>
                        </PanelBody>
                    </Panel>
                    <Panel>
                        <PanelHeader
                            icon={BrainCircuit}
                            title="From raw rows to a defensible result"
                            description="Every stage stays in the project archive, including source attribution and review decisions."
                        />
                        <PanelBody className="grid gap-3 md:grid-cols-3">
                            {[
                                [
                                    Database,
                                    "1. Import & profile",
                                    "CSV, Excel, Parquet, TSV or JSONL. Inspect types, missingness and examples.",
                                ],
                                [
                                    Scale,
                                    "2. Configure & train",
                                    "Save the target and split, train against a baseline, and keep held-out metrics.",
                                ],
                                [
                                    FileSearch,
                                    "3. Review & improve",
                                    "Correct uncertain, diverse and duplicate rows from the transparent Smart Review queue.",
                                ],
                            ].map(([Icon, title, description]) => {
                                const StageIcon = Icon as typeof Database
                                return (
                                    <div key={String(title)} className="rounded-lg border p-4">
                                        <StageIcon className="text-muted-foreground size-4" />
                                        <p className="mt-3 text-sm font-medium">{String(title)}</p>
                                        <p className="text-muted-foreground mt-1 text-xs leading-relaxed">
                                            {String(description)}
                                        </p>
                                    </div>
                                )
                            })}
                        </PanelBody>
                    </Panel>
                </div>
                <Panel>
                    <PanelHeader title="Next step" />
                    <PanelBody className="space-y-3">
                        <p className="text-sm font-medium">
                            {structuredData?.source
                                ? structuredData.configured
                                    ? "Run or inspect the workflow"
                                    : "Choose the workflow and columns"
                                : "Import a dataset"}
                        </p>
                        <p className="text-muted-foreground text-xs leading-relaxed">
                            {structuredData?.source
                                ? `${structuredData.source.filename} is stored locally with a checksum. Configuration changes do not rewrite the original.`
                                : "Use your own data, or start with a real CC BY 4.0 example from the curated catalog."}
                        </p>
                        <Button className="w-full" asChild>
                            <a href={`/projects/dataset?projectId=${project.id}`}>
                                Open data workspace <ArrowRight />
                            </a>
                        </Button>
                    </PanelBody>
                </Panel>
            </div>
        )
    }

    return (
        // Two independent columns rather than a shared row grid: when the split
        // panel is taller than the project panel, a row grid pushes the labels
        // panel down and leaves a band of dead space beside it.
        <div className="grid items-start gap-4 lg:grid-cols-3">
            <div className="grid content-start gap-4 lg:col-span-2">
                <Panel>
                    {/* The project name and task live in the workbench bar, so
                        this panel names the job it does instead of repeating
                        them back. */}
                    <PanelHeader icon={BoxIcon} title="Project" />
                    <PanelBody className="space-y-4">
                        <EditableField
                            label="Name"
                            value={project.name ?? ""}
                            placeholder="Project name"
                            empty="Untitled project"
                            onSave={onSaveName}
                        />
                        <EditableField
                            label="Description"
                            value={project.description ?? ""}
                            placeholder="What is this project for?"
                            empty="No description yet."
                            onSave={onSaveDescription}
                        />
                        <div className="grid grid-cols-2 gap-4 border-t pt-4 sm:grid-cols-3">
                            <Stat label="Task" value={projectTypeLabel(project.type)} mono={false} />
                            <Stat label="On disk" value={project.size ? `${project.size} GB` : "—"} />
                            <Stat label="Classes" value={project.labels?.length ?? 0} />
                        </div>
                    </PanelBody>
                </Panel>

                <LabelListEditor projectId={project.id} />
            </div>

            <div className="grid content-start gap-4">
                <DataDistribution projectId={project.id} />
                <ClassDistribution projectId={project.id} />
            </div>
        </div>
    )
}

export default ProjectOverview

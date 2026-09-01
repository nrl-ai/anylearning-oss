import { SlidersHorizontal, TagIcon } from "lucide-react"

import { useSettingStore } from "@/components/react-image-label/base/store"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { EmptyState } from "@/components/ui/empty-state"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Switch } from "@/components/ui/switch"
import { isClassificationProject } from "@/lib/project-types"
import { DataItem, Project } from "@/types"

type LabelListProps = {
    projectId: number
    project: Project | undefined
    currentImage: DataItem | undefined
    handleClassChange: (classId: number) => void
}

export default function LabelList({ projectId, project, currentImage, handleClassChange }: LabelListProps) {
    const {
        isShowLabels,
        isShowKeypointInstances,
        isShowKeypointVisibility,
        isDimOccludedKeypoints,
        setShowLabels,
        setShowKeypointInstances,
        setShowKeypointVisibility,
        setDimOccludedKeypoints,
    } = useSettingStore()
    const isClassification = isClassificationProject(project?.type)
    const isKeypoint = project?.type === "Keypoint Detection"
    const keypointDisplayOptions: [string, boolean, (value: boolean) => void][] = [
        ["Landmark names", isShowLabels, setShowLabels],
        ["Instance IDs", isShowKeypointInstances, setShowKeypointInstances],
        ["Occlusion status", isShowKeypointVisibility, setShowKeypointVisibility],
        ["Dim occluded points", isDimOccludedKeypoints, setDimOccludedKeypoints],
    ]

    return (
        <div className="flex max-h-[280px] w-full flex-none flex-col overflow-hidden">
            <div className="mb-2 flex flex-none items-center justify-between gap-2">
                <p className="t-eyebrow">{isKeypoint ? "Landmarks" : "Classes"}</p>
                {!isClassification && !isKeypoint && (
                    <div className="flex items-center gap-2">
                        <Switch checked={isShowLabels} onCheckedChange={setShowLabels} id="showLabelsSwitch" />
                        <label className="cursor-pointer text-xs" htmlFor="showLabelsSwitch">
                            Names
                        </label>
                    </div>
                )}
                {isKeypoint && (
                    <Popover>
                        <PopoverTrigger asChild>
                            <Button variant="ghost" size="sm" className="h-7 px-2 text-xs">
                                <SlidersHorizontal />
                                Display
                            </Button>
                        </PopoverTrigger>
                        <PopoverContent align="end" className="w-72 space-y-3">
                            <div>
                                <p className="t-section">Keypoint display</p>
                                <p className="text-muted-foreground mt-0.5 text-xs">
                                    Choose what is drawn without changing the saved annotations.
                                </p>
                            </div>
                            {keypointDisplayOptions.map(([label, checked, change]) => (
                                <label key={label} className="flex cursor-pointer items-center justify-between gap-3">
                                    <span className="text-sm">{label}</span>
                                    <Switch checked={checked} onCheckedChange={change} aria-label={label} />
                                </label>
                            ))}
                        </PopoverContent>
                    </Popover>
                )}
            </div>

            <div className="mb-2 min-h-0 flex-1 overflow-y-auto">
                {isClassification ? (
                    <div className="space-y-1">
                        <label className="hover:bg-accent flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1">
                            <Checkbox
                                id="unlabeled"
                                checked={currentImage?.class_id === -1}
                                onCheckedChange={() => handleClassChange(-1)}
                            />
                            <span className="text-muted-foreground text-sm">Unlabelled</span>
                        </label>
                        {project?.labels?.map((label) => (
                            <label
                                key={label.id}
                                className="hover:bg-accent flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1"
                            >
                                <Checkbox
                                    id={`class-${label.id}`}
                                    checked={currentImage?.class_id === label.id}
                                    onCheckedChange={() => handleClassChange(label.id)}
                                />
                                <span
                                    aria-hidden
                                    className="size-3 shrink-0 rounded-[2px] border-2"
                                    style={{
                                        borderColor: label.color || "#888",
                                        backgroundColor: `${label.color || "#888888"}2e`,
                                    }}
                                />
                                <span className="min-w-0 truncate text-sm">{label.name}</span>
                            </label>
                        ))}
                    </div>
                ) : project?.labels?.length ? (
                    // A list, not a table: the previous <table> gave "Name / ID
                    // / Color" three header columns for what is really one row
                    // per class, and the swatch didn't match how the class is
                    // actually drawn on the canvas.
                    <ul className="space-y-0.5">
                        {project.labels.map((label) => (
                            <li key={label.id} className="flex items-center gap-2 px-1.5 py-1">
                                <span
                                    aria-hidden
                                    className="size-3 shrink-0 rounded-[2px] border-2"
                                    style={{
                                        borderColor: label.color || "#888",
                                        backgroundColor: `${label.color || "#888888"}2e`,
                                    }}
                                />
                                <span className="min-w-0 flex-1 truncate text-sm">{label.name}</span>
                                <span className="text-muted-foreground tabular font-mono text-[0.6875rem]">
                                    {label.id}
                                </span>
                            </li>
                        ))}
                    </ul>
                ) : (
                    <EmptyState compact icon={TagIcon} title={isKeypoint ? "No landmarks" : "No classes"} />
                )}
            </div>

            <p className="text-muted-foreground flex-none text-[0.6875rem]">
                Manage {isKeypoint ? "landmarks" : "classes"} in the{" "}
                <a
                    href={`/projects/overview?projectId=${projectId}`}
                    className="text-mark underline-offset-2 hover:underline"
                >
                    project overview
                </a>
                .
            </p>
        </div>
    )
}

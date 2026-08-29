import {
    BrainIcon,
    HelpCircle,
    Hexagon,
    LucideIcon,
    MapPin,
    MousePointer,
    RotateCcw,
    Save,
    SquareIcon,
    ZoomIn,
    ZoomOut,
} from "lucide-react"

import { AnnotatorHandles } from "@/components/react-image-label/annotator/hook"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

type LabelingMode = "labeling" | "auto_labeling"

type LeftBarProps = {
    mode: string
    aiEnabled: boolean
    projectType: string
    annotator: AnnotatorHandles | undefined
    selectedTool: string
    handleToolSelect: (tool: string) => void
    setMode: (value: LabelingMode) => void
    setAiEnabled: (value: boolean) => void
    saveAnnotation: () => void
    clearAll: () => void
}

/**
 * One tool button.
 *
 * Every button here used to carry its own hardcoded colour — blue for the
 * active tool, grey for the rest, amber for AI, red for clear — which meant
 * the toolbar was a small rainbow and had no dark mode at all. Now the only
 * coloured state is "this tool is armed" (the mark) and "this is destructive"
 * (fail), and both come from tokens.
 */
function ToolButton({
    icon: Icon,
    label,
    shortcut,
    active,
    danger,
    onClick,
}: {
    icon: LucideIcon
    label: string
    shortcut?: string
    active?: boolean
    danger?: boolean
    onClick: () => void
}) {
    return (
        <Tooltip>
            <TooltipTrigger asChild>
                <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={onClick}
                    aria-label={label}
                    aria-pressed={active}
                    className={cn(
                        "text-muted-foreground hover:text-foreground",
                        active && "bg-mark text-mark-ink hover:bg-mark-strong hover:text-mark-ink",
                        danger && "hover:bg-fail-surface hover:text-fail"
                    )}
                >
                    <Icon />
                </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
                {label}
                {shortcut && <span className="text-muted-foreground ml-1.5 font-mono">{shortcut}</span>}
            </TooltipContent>
        </Tooltip>
    )
}

export default function LeftBar({
    mode,
    projectType,
    aiEnabled,
    annotator,
    selectedTool,
    handleToolSelect,
    setMode,
    setAiEnabled,
    saveAnnotation,
    clearAll,
}: LeftBarProps) {
    if (projectType === "Image Classification" || projectType === "Handpose Classification") {
        return null
    }

    return (
        <div className="bg-surface/95 absolute top-3 left-3 z-40 flex w-fit flex-col gap-0.5 rounded-lg border p-1 shadow-md backdrop-blur">
            <ToolButton
                icon={MousePointer}
                label="Select"
                active={selectedTool === "select"}
                onClick={() => handleToolSelect("select")}
            />
            {projectType === "Object Detection" && (
                <ToolButton
                    icon={SquareIcon}
                    label="Draw rectangle"
                    active={selectedTool === "rectangle"}
                    onClick={() => handleToolSelect("rectangle")}
                />
            )}
            {projectType === "Image Segmentation" && (
                <ToolButton
                    icon={Hexagon}
                    label="Draw polygon"
                    active={selectedTool === "polygon"}
                    onClick={() => handleToolSelect("polygon")}
                />
            )}
            {projectType === "Keypoint Detection" && (
                <ToolButton
                    icon={MapPin}
                    label="Place keypoint"
                    active={selectedTool === "dot"}
                    onClick={() => handleToolSelect("dot")}
                />
            )}

            <span aria-hidden className="bg-border my-1 h-px w-full" />

            <ToolButton icon={ZoomIn} label="Zoom in" onClick={() => annotator?.zoom(1.25)} />
            <ToolButton icon={ZoomOut} label="Zoom out" onClick={() => annotator?.zoom(0.8)} />
            <ToolButton icon={Save} label="Save annotations" onClick={saveAnnotation} />

            <span aria-hidden className="bg-border my-1 h-px w-full" />

            {/* Shows and hides the auto-labelling toolbar, and nothing else:
                the mode follows which tool is selected, so setting it here as
                well meant the button read "Turn on auto-labelling" while lit,
                and pressing it hid the toolbar while switching *into* AI mode.
                Turning the toolbar off leaves the AI tools with no way to be
                selected, so it hands the pointer back. */}
            {projectType !== "Keypoint Detection" && (
                <ToolButton
                    icon={BrainIcon}
                    label={aiEnabled ? "Turn off auto-labelling" : "Turn on auto-labelling"}
                    active={aiEnabled}
                    onClick={() => {
                        const next = !aiEnabled
                        setAiEnabled(next)
                        if (!next && mode === "auto_labeling") {
                            setMode("labeling")
                            handleToolSelect("select")
                        }
                    }}
                />
            )}
            <ToolButton icon={RotateCcw} label="Clear all annotations" danger onClick={clearAll} />

            <Popover>
                <PopoverTrigger asChild>
                    <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label="Keyboard and mouse help"
                        className="text-muted-foreground hover:text-foreground"
                    >
                        <HelpCircle />
                    </Button>
                </PopoverTrigger>
                <PopoverContent side="right" className="w-72">
                    <p className="t-section mb-2">Mouse and keyboard</p>
                    <dl className="space-y-1.5 text-xs">
                        {[
                            ["Click", "Edit or stop editing an annotation"],
                            ["Ctrl + wheel", "Zoom"],
                            ["Ctrl + drag", "Pan"],
                            ["Drag", "Move, resize or rotate an annotation"],
                            ["Delete", "Delete the selected annotation"],
                            ["← / →", "Previous or next image"],
                        ].map(([keys, what]) => (
                            <div key={keys} className="flex gap-3">
                                <dt className="w-24 shrink-0 font-mono text-[0.6875rem]">{keys}</dt>
                                <dd className="text-muted-foreground">{what}</dd>
                            </div>
                        ))}
                    </dl>
                </PopoverContent>
            </Popover>
        </div>
    )
}

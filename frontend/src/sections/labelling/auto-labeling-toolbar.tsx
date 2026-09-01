import { BrainIcon, CheckCircle, Circle, CircleMinus, Loader2, Play, Square as SquareIcon, X } from "lucide-react"
import React, { useMemo } from "react"

import { Button } from "@/components/ui/button"
import {
    Select,
    SelectContent,
    SelectGroup,
    SelectItem,
    SelectLabel,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import { formatDownloadSize, groupAutoLabelingModels } from "@/lib/auto-labeling-models"
import useAutoLabeling from "@/lib/use-auto-labeling"
import { cn } from "@/lib/utils"

import ImportOnnxModelDialog from "./import-onnx-model-dialog"

type LabelingMode = "labeling" | "auto_labeling"
const isErrorStatus = (status: string) => /^(Error|Could not)/i.test(status)

interface LabelingScreenProps {
    setMode: (mode: LabelingMode) => void
    aiToolSelected: string
    projectId: number
    projectType: string
    projectLabels: string[]
    aiShape: string
    model: string
    mode: LabelingMode
    setAiShape: (value: string) => void
    selectModel: (value: string, interactionMode: "prompted" | "automatic") => void
    handleToolSelect: (tool: string, isAI: boolean) => void
    clear: () => void
    finish: () => void
    run: () => void
    isInferencing: boolean
}

/**
 * Prompt tools for the auto-labelling model.
 *
 * These were previously four different text colours — green, orange, grey,
 * blue — on a hardcoded dark grey bar, in an app that has a light theme. The
 * only distinction that carries meaning is add versus remove, so add/remove
 * are the two tones and everything else is a plain control.
 */
export default function AutoLabellingToolbar({
    aiToolSelected,
    projectId,
    projectType,
    projectLabels,
    aiShape,
    model,
    mode,
    setMode,
    setAiShape,
    selectModel,
    handleToolSelect,
    clear,
    finish,
    run,
    isInferencing,
}: LabelingScreenProps) {
    const { models, status, loadedModel, loadModel, loadError, isLoadingModel, isLoading, refreshModels } =
        useAutoLabeling(projectId)
    const requiresBoundingBoxes = projectType === "Object Detection"
    const compatibleModels = useMemo(
        () => models.filter((candidate) => candidate.project_types?.includes(projectType)),
        [models, projectType]
    )
    const modelGroups = useMemo(() => groupAutoLabelingModels(compatibleModels), [compatibleModels])
    const selectedModel = compatibleModels.find((candidate) => candidate.name === model)
    const isPromptable = selectedModel?.interaction_mode !== "automatic"
    const modelReady = loadedModel === model && !isLoadingModel
    const hasStatusError = isErrorStatus(status)
    const modelPlaceholder = isLoading
        ? "Loading models…"
        : compatibleModels.length
          ? "Select a model"
          : "No compatible models"

    const isArmed = (tool: string) => mode === "auto_labeling" && aiToolSelected === tool

    return (
        <div className="bg-surface-sunken flex shrink-0 flex-col border-b">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2">
                <div className="flex items-center gap-2">
                    <BrainIcon className="text-muted-foreground size-4 shrink-0" strokeWidth={2} />
                    <Select
                        value={model}
                        disabled={isLoading || compatibleModels.length === 0}
                        onValueChange={(value) => {
                            const candidate = compatibleModels.find((item) => item.name === value)
                            selectModel(value, candidate?.interaction_mode ?? "prompted")
                            setMode("auto_labeling")
                            void loadModel(value).catch(() => undefined)
                        }}
                    >
                        <SelectTrigger
                            size="sm"
                            className="w-[300px] truncate"
                            aria-label="Auto-labeling model"
                            title={selectedModel?.display_name}
                        >
                            <SelectValue placeholder={modelPlaceholder}>{selectedModel?.display_name}</SelectValue>
                        </SelectTrigger>
                        <SelectContent className="w-[340px]">
                            {modelGroups.map((group) => (
                                <SelectGroup key={group.label}>
                                    <SelectLabel>{group.label}</SelectLabel>
                                    {group.models.map((candidate) => (
                                        <SelectItem key={candidate.name} value={candidate.name}>
                                            <span className="flex min-w-0 flex-1 items-center justify-between gap-3">
                                                <span className="truncate">{candidate.display_name}</span>
                                                <span
                                                    className={cn(
                                                        "shrink-0 text-[0.6875rem]",
                                                        candidate.has_downloaded ? "text-ok" : "text-muted-foreground"
                                                    )}
                                                >
                                                    {candidate.has_downloaded
                                                        ? "Ready"
                                                        : formatDownloadSize(candidate.archive_size_bytes)}
                                                </span>
                                            </span>
                                        </SelectItem>
                                    ))}
                                </SelectGroup>
                            ))}
                        </SelectContent>
                    </Select>
                </div>

                <ImportOnnxModelDialog
                    projectType={projectType}
                    projectLabels={projectLabels}
                    onImported={async (modelName) => {
                        await refreshModels()
                        selectModel(modelName, "automatic")
                        setMode("auto_labeling")
                        await loadModel(modelName)
                    }}
                />

                <Select
                    value={requiresBoundingBoxes ? "rectangle" : aiShape}
                    onValueChange={setAiShape}
                    disabled={!selectedModel || requiresBoundingBoxes || selectedModel.output_modes?.length === 1}
                >
                    <SelectTrigger size="sm" className="w-[130px]">
                        <SelectValue placeholder="Shape" />
                    </SelectTrigger>
                    <SelectContent>
                        {!requiresBoundingBoxes && <SelectItem value="polygon">Polygon</SelectItem>}
                        <SelectItem value="rectangle">Rectangle</SelectItem>
                    </SelectContent>
                </Select>

                {!selectedModel ? (
                    <p className="text-muted-foreground text-xs">
                        Choose a model to enable AI tools. It will load only after you select it.
                    </p>
                ) : isPromptable ? (
                    <div className="flex items-center gap-0.5">
                        <Button
                            size="sm"
                            variant="ghost"
                            disabled={!modelReady || isInferencing}
                            className={cn("text-ok", isArmed("addPoint") && "bg-ok-surface")}
                            onClick={() => handleToolSelect("addPoint", true)}
                        >
                            <Circle />
                            Include point
                            <kbd className="text-muted-foreground ml-1 font-mono text-[0.6875rem]">a</kbd>
                        </Button>

                        <Button
                            size="sm"
                            variant="ghost"
                            disabled={!modelReady || isInferencing}
                            className={cn("text-fail", isArmed("removePoint") && "bg-fail-surface")}
                            onClick={() => handleToolSelect("removePoint", true)}
                        >
                            <CircleMinus />
                            Exclude point
                            <kbd className="text-muted-foreground ml-1 font-mono text-[0.6875rem]">d</kbd>
                        </Button>

                        <Button
                            size="sm"
                            variant="ghost"
                            disabled={!modelReady || isInferencing}
                            className={cn("text-ok", isArmed("addRect") && "bg-ok-surface")}
                            onClick={() => handleToolSelect("addRect", true)}
                        >
                            <SquareIcon />
                            Include box
                            <kbd className="text-muted-foreground ml-1 font-mono text-[0.6875rem]">r</kbd>
                        </Button>

                        <Button size="sm" variant="ghost" className="text-muted-foreground" onClick={clear}>
                            <X />
                            Clear
                            <kbd className="ml-1 font-mono text-[0.6875rem]">c</kbd>
                        </Button>

                        <Button size="sm" disabled={!modelReady || isInferencing} onClick={finish}>
                            <CheckCircle />
                            Finish object
                            <kbd className="ml-1 font-mono text-[0.6875rem] opacity-70">f</kbd>
                        </Button>
                    </div>
                ) : (
                    <div className="flex items-center gap-1">
                        <Button size="sm" disabled={!modelReady || isInferencing} onClick={run}>
                            <Play />
                            Run model
                        </Button>
                        <Button size="sm" variant="ghost" className="text-muted-foreground" onClick={clear}>
                            <X />
                            Clear predictions
                        </Button>
                    </div>
                )}
            </div>

            {(loadError || (selectedModel && status)) && (
                <p
                    role="status"
                    className={cn(
                        "flex items-center gap-2 truncate border-t px-3 py-1 text-xs",
                        loadError || hasStatusError ? "text-fail" : "text-muted-foreground"
                    )}
                >
                    {isLoadingModel && <Loader2 className="size-3.5 shrink-0 animate-spin" />}
                    <span className="truncate">{loadError || status}</span>
                </p>
            )}
        </div>
    )
}

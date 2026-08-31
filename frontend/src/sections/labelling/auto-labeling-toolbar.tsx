import { BrainIcon, CheckCircle, Circle, CircleMinus, Play, Square as SquareIcon, X } from "lucide-react"
import React, { useEffect, useMemo } from "react"

import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import useAutoLabeling from "@/lib/use-auto-labeling"
import { cn } from "@/lib/utils"

type LabelingMode = "labeling" | "auto_labeling"
interface LabelingScreenProps {
    setMode: (mode: LabelingMode) => void
    aiToolSelected: string
    projectId: number
    projectType: string
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
    const { models, status, loadedModel, loadModel, loadError, isLoadingModel } = useAutoLabeling(projectId)
    const requiresBoundingBoxes = projectType === "Object Detection"
    const compatibleModels = useMemo(
        () => models.filter((candidate) => candidate.project_types?.includes(projectType)),
        [models, projectType]
    )
    const selectedModel = compatibleModels.find((candidate) => candidate.name === model)
    const isPromptable = selectedModel?.interaction_mode !== "automatic"
    const modelReady = loadedModel === model && !isLoadingModel

    useEffect(() => {
        if (compatibleModels.length > 0 && !selectedModel) {
            const firstModel = compatibleModels[0].name
            selectModel(firstModel, compatibleModels[0].interaction_mode)
            void loadModel(firstModel).catch(() => undefined)
        }
    }, [compatibleModels, selectedModel, selectModel, loadModel])

    const isArmed = (tool: string) => mode === "auto_labeling" && aiToolSelected === tool

    return (
        <div className="bg-surface-sunken flex shrink-0 flex-col border-b">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2">
                <div className="flex items-center gap-2">
                    <BrainIcon className="text-muted-foreground size-4 shrink-0" strokeWidth={2} />
                    <Select
                        value={model}
                        onValueChange={(value) => {
                            void loadModel(value).catch(() => undefined)
                            const candidate = compatibleModels.find((item) => item.name === value)
                            selectModel(value, candidate?.interaction_mode ?? "prompted")
                            setMode("auto_labeling")
                        }}
                    >
                        <SelectTrigger size="sm" className="w-[230px] truncate">
                            <SelectValue placeholder="Select a model" />
                        </SelectTrigger>
                        <SelectContent>
                            {compatibleModels.map((model) => (
                                <SelectItem key={model.name} value={model.name} className="truncate">
                                    {model.display_name}
                                    {model.is_project_model ? " · This project" : ""}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>

                <Select
                    value={requiresBoundingBoxes ? "rectangle" : aiShape}
                    onValueChange={setAiShape}
                    disabled={requiresBoundingBoxes || selectedModel?.output_modes?.length === 1}
                >
                    <SelectTrigger size="sm" className="w-[130px]">
                        <SelectValue placeholder="Shape" />
                    </SelectTrigger>
                    <SelectContent>
                        {!requiresBoundingBoxes && <SelectItem value="polygon">Polygon</SelectItem>}
                        <SelectItem value="rectangle">Rectangle</SelectItem>
                    </SelectContent>
                </Select>

                {isPromptable ? (
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

            {(loadError || status) && (
                <p className="text-muted-foreground truncate border-t px-3 py-1 font-mono text-[0.6875rem]">
                    {loadError || status}
                </p>
            )}
        </div>
    )
}

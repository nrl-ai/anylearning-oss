import { BrainIcon, CheckCircle, Circle, CircleMinus, Square as SquareIcon, X } from "lucide-react"
import React, { useEffect } from "react"

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
    selectModel: (value: string) => void
    handleToolSelect: (tool: string, isAI: boolean) => void
    clear: () => void
    finish: () => void
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
    setAiShape,
    selectModel,
    handleToolSelect,
    clear,
    finish,
}: LabelingScreenProps) {
    const { models, status, loadModel } = useAutoLabeling(projectId)
    const requiresBoundingBoxes = projectType === "Object Detection"

    useEffect(() => {
        if (models.length > 0 && !model) {
            const firstModel = models[0].name
            selectModel(firstModel)
            loadModel(firstModel)
        }
    }, [models, selectModel, loadModel, model])

    const isArmed = (tool: string) => mode === "auto_labeling" && aiToolSelected === tool

    return (
        <div className="bg-surface-sunken flex shrink-0 flex-col border-b">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2">
                <div className="flex items-center gap-2">
                    <BrainIcon className="text-muted-foreground size-4 shrink-0" strokeWidth={2} />
                    <Select
                        value={model}
                        onValueChange={(value) => {
                            loadModel(value)
                            selectModel(value)
                        }}
                    >
                        <SelectTrigger size="sm" className="w-[230px] truncate">
                            <SelectValue placeholder="Select a model" />
                        </SelectTrigger>
                        <SelectContent>
                            {models.map((model) => (
                                <SelectItem key={model.name} value={model.name} className="truncate">
                                    {model.display_name}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>

                <Select
                    value={requiresBoundingBoxes ? "rectangle" : aiShape}
                    onValueChange={setAiShape}
                    disabled={requiresBoundingBoxes}
                >
                    <SelectTrigger size="sm" className="w-[130px]">
                        <SelectValue placeholder="Shape" />
                    </SelectTrigger>
                    <SelectContent>
                        {!requiresBoundingBoxes && <SelectItem value="polygon">Polygon</SelectItem>}
                        <SelectItem value="rectangle">Rectangle</SelectItem>
                    </SelectContent>
                </Select>

                <div className="flex items-center gap-0.5">
                    <Button
                        size="sm"
                        variant="ghost"
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

                    <Button size="sm" onClick={finish}>
                        <CheckCircle />
                        Finish object
                        <kbd className="ml-1 font-mono text-[0.6875rem] opacity-70">f</kbd>
                    </Button>
                </div>
            </div>

            {status && (
                <p className="text-muted-foreground truncate border-t px-3 py-1 font-mono text-[0.6875rem]">{status}</p>
            )}
        </div>
    )
}

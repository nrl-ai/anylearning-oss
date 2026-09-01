import { Loader2, Upload } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

import { Button } from "@/components/ui/button"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { isDesktop, onDesktopReady } from "@/lib/desktop"

type YoloTask = "detection" | "instance_segmentation"
type LabelSource = "project" | "coco80" | "custom"

const FORMATS = ["auto", "yolov5", "yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "yolo26", "yolox"]

export function parseClassNames(value: string): string[] {
    return value
        .split(/[\n,]/)
        .map((name) => name.trim())
        .filter(Boolean)
}

interface ImportOnnxModelDialogProps {
    projectType: string
    projectLabels: string[]
    onImported: (modelName: string) => Promise<void>
}

export default function ImportOnnxModelDialog({ projectType, projectLabels, onImported }: ImportOnnxModelDialogProps) {
    const [open, setOpen] = useState(false)
    const [desktopReady, setDesktopReady] = useState(false)
    const [displayName, setDisplayName] = useState("Custom ONNX model")
    const [format, setFormat] = useState("auto")
    const [task, setTask] = useState<YoloTask>(
        projectType === "Object Detection" ? "detection" : "instance_segmentation"
    )
    const [labelSource, setLabelSource] = useState<LabelSource>(projectLabels.length ? "project" : "custom")
    const [customLabels, setCustomLabels] = useState("")
    const [inputSize, setInputSize] = useState("")
    const [error, setError] = useState("")
    const [isImporting, setIsImporting] = useState(false)

    useEffect(() => onDesktopReady(() => setDesktopReady(true)), [])
    const canDetect = projectType === "Object Detection"
    const selectedLabels = useMemo(
        () => (labelSource === "project" ? projectLabels : parseClassNames(customLabels)),
        [customLabels, labelSource, projectLabels]
    )

    const submit = async () => {
        setError("")
        const api = window.pywebview?.api?.import_onnx_auto_labeling_model
        if (!api) {
            setError("ONNX import is available in the desktop app.")
            return
        }
        if (!displayName.trim()) {
            setError("Enter a model name.")
            return
        }
        if (labelSource !== "coco80" && !selectedLabels.length) {
            setError("Enter the class names in the model's exact output order.")
            return
        }
        const parsedInputSize = inputSize ? Number(inputSize) : undefined
        if (parsedInputSize !== undefined && (!Number.isInteger(parsedInputSize) || parsedInputSize < 16)) {
            setError("Input size must be an integer of at least 16 pixels.")
            return
        }

        setIsImporting(true)
        try {
            const result = await api({
                display_name: displayName.trim(),
                format,
                task,
                ...(labelSource === "coco80" ? { label_space: "coco80" as const } : { class_names: selectedLabels }),
                ...(parsedInputSize ? { input_size: parsedInputSize } : {}),
            })
            if (result.cancelled) return
            if (!result.ok || !result.model_name) {
                setError(result.error || "The selected ONNX model could not be imported.")
                return
            }
            await onImported(result.model_name)
            setOpen(false)
        } catch (caught) {
            setError(caught instanceof Error ? caught.message : "The selected ONNX model could not be imported.")
        } finally {
            setIsImporting(false)
        }
    }

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
                <Button
                    variant="outline"
                    size="sm"
                    disabled={!isDesktop() || !desktopReady}
                    title="Import an ONNX model"
                >
                    <Upload />
                    Import ONNX
                </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-xl">
                <DialogHeader>
                    <DialogTitle>Import a YOLO-family ONNX model</DialogTitle>
                    <DialogDescription>
                        Choose a single-file ONNX graph (up to 20 GiB). It is copied into AnyLearning and runs through
                        the shared bounded inference core.
                    </DialogDescription>
                </DialogHeader>

                <div className="grid gap-4 py-1">
                    <div className="grid gap-2">
                        <Label htmlFor="onnx-display-name">Model name</Label>
                        <Input
                            id="onnx-display-name"
                            value={displayName}
                            maxLength={256}
                            onChange={(event) => setDisplayName(event.target.value)}
                        />
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                        <div className="grid gap-2">
                            <Label>Export format</Label>
                            <Select value={format} onValueChange={setFormat}>
                                <SelectTrigger className="w-full">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    {FORMATS.map((value) => (
                                        <SelectItem
                                            key={value}
                                            value={value}
                                            disabled={value === "yolox" && !canDetect}
                                        >
                                            {value === "auto" ? "Auto-detect" : value.toUpperCase()}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="grid gap-2">
                            <Label>Task</Label>
                            <Select
                                value={task}
                                onValueChange={(value) => setTask(value as YoloTask)}
                                disabled={!canDetect}
                            >
                                <SelectTrigger className="w-full">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="detection">Object detection</SelectItem>
                                    <SelectItem value="instance_segmentation">Instance segmentation</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                    </div>

                    <div className="grid grid-cols-[1fr_10rem] gap-3">
                        <div className="grid gap-2">
                            <Label>Class order</Label>
                            <Select value={labelSource} onValueChange={(value) => setLabelSource(value as LabelSource)}>
                                <SelectTrigger className="w-full">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="project" disabled={!projectLabels.length}>
                                        Project labels ({projectLabels.length})
                                    </SelectItem>
                                    <SelectItem value="coco80">COCO 80 classes</SelectItem>
                                    <SelectItem value="custom">Custom class list</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="grid gap-2">
                            <Label htmlFor="onnx-input-size">Dynamic input size</Label>
                            <Input
                                id="onnx-input-size"
                                type="number"
                                min={16}
                                max={16384}
                                placeholder="Auto"
                                value={inputSize}
                                onChange={(event) => setInputSize(event.target.value)}
                            />
                        </div>
                    </div>

                    {labelSource === "custom" && (
                        <div className="grid gap-2">
                            <Label htmlFor="onnx-classes">Class names, in output index order</Label>
                            <Textarea
                                id="onnx-classes"
                                rows={5}
                                placeholder={"person\nbicycle\ncar"}
                                value={customLabels}
                                onChange={(event) => setCustomLabels(event.target.value)}
                            />
                        </div>
                    )}
                    <p className="text-muted-foreground text-xs">
                        Class order must match the model export exactly. Use an explicit input size only for dynamic
                        input graphs. External-data ONNX bundles are not imported yet.
                    </p>
                    {error && <p className="text-fail text-sm">{error}</p>}
                </div>

                <DialogFooter>
                    <Button variant="outline" onClick={() => setOpen(false)} disabled={isImporting}>
                        Cancel
                    </Button>
                    <Button onClick={submit} disabled={isImporting}>
                        {isImporting ? <Loader2 className="animate-spin" /> : <Upload />}
                        {isImporting ? "Importing…" : "Choose file, import and load"}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}

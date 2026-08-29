"use client"

import { useQuery } from "@tanstack/react-query"
import { Box, Check, ChevronsUpDown, Cpu, Database, Gauge, Maximize2, Repeat, Scale, Shuffle } from "lucide-react"
import { useEffect, useState } from "react"

import { Button } from "@/components/ui/button"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { InfoHint } from "@/components/ui/info-hint"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { getJson } from "@/lib/api"
import { isStructuredProject } from "@/lib/project-types"
import useModels from "@/lib/use-models"
import useProject from "@/lib/use-project"
import { cn } from "@/lib/utils"
import { Accelerator, AugmentationOption, TrainingDevices, TrainingParams } from "@/types"

/**
 * How long a first run should be, per task.
 *
 * One number for every task was wrong in one direction each way, and badly so
 * for handpose: the landmark classifier is trained from scratch, and on 26 ASL
 * letters ten epochs gives 3% -- chance, on 26 classes -- while the same run
 * at 300 reaches the high seventies. A first-time user with the old default
 * concluded, reasonably, that the software did not work.
 *
 * The others start from pretrained weights and are fine-tuning, so they need
 * far fewer passes to say something useful.
 */
const DEFAULT_EPOCHS: Record<string, number> & { default: number } = {
    "Tabular AI": 300,
    "Text AI": 300,
    "Text AI & LLM Evaluation": 300,
    "Text & LLM": 300,
    "Sentiment Analysis": 300,
    "Handpose Classification": 300,
    "Image Classification": 30,
    "Object Detection": 30,
    "Image Segmentation": 30,
    "Instance Segmentation": 20,
    "Keypoint Detection": 30,
    default: 30,
}

/**
 * The starting rate, per architecture rather than per task.
 *
 * One number here is wrong for the same reason one epoch count was. RF-DETR
 * fine-tunes a DINOv2 encoder and its own configs use 1e-4; 1e-3 is ten times
 * that, and the trainer scales the encoder's rate to match the head's, so the
 * dialog's default moved both. Measured cost of clicking straight through:
 * about 0.09 mAP against the same run at the template's rate.
 *
 * Keyed on `model_architecture`, which is what the rate belongs to -- two
 * detectors ship for Object Detection and they do not want the same one.
 */
const DEFAULT_LEARNING_RATE: Record<string, number> & { default: number } = {
    catboost: 0.05,
    "tfidf-logreg": 0.1,
    rfdetr: 0.0001,
    "rfdetr-seg": 0.0001,
    "rfdetr-keypoint": 0.0001,
    default: 0.001,
}

/** Architecture-specific starting points that fit the supported hardware. */
const DEFAULT_BATCH_SIZE: Record<string, number> & { default: number } = {
    catboost: 1,
    "tfidf-logreg": 1,
    // The keypoint head has a larger activation graph than detection. Two is
    // the verified Metal-safe default on a 16 GB M1; the backend also caps
    // older clients that send a larger value to an Apple GPU.
    "rfdetr-keypoint": 2,
    default: 16,
}

export function NewTrainingDialog({
    isOpen,
    onOpenChange,
    onStartTraining,
    projectId,
}: {
    isOpen: boolean
    onOpenChange: (open: boolean) => void
    onStartTraining: (params: TrainingParams) => Promise<void>
    projectId: number
}) {
    const { project, modelVariants } = useProject(projectId)
    const isStructured = isStructuredProject(project?.type)
    // Declared by the trainer for this project type, so the dialog shows what
    // the model underneath can actually honour rather than a fixed list.
    const { data: augmentations } = useQuery({
        queryKey: ["augmentations"],
        queryFn: () => getJson<Record<string, AugmentationOption[]>>("/api/augmentations"),
        staleTime: Infinity,
    })
    const { data: devices } = useQuery({
        queryKey: ["training-devices"],
        queryFn: () => getJson<TrainingDevices>("/api/settings/devices"),
        staleTime: Infinity,
    })
    // The list is what a current backend answers. `cuda` is the fallback for one
    // that predates it, so an older build still offers its GPU.
    const accelerators: Accelerator[] = (
        devices?.accelerators ?? (devices?.cuda ? [{ id: "cuda", name: devices.name, label: "GPU" }] : [])
    ).filter(
        // A Mac's GPU cannot train every project type, and is slower than
        // the CPU for one of them. An option that quietly does neither of
        // the things it says is worse than one that is not offered.
        (accelerator) => !(accelerator.excluded_project_types ?? []).includes(project?.type ?? "")
    )
    const { models, fetchModels } = useModels(projectId)
    const [modelSearch, setModelSearch] = useState("")
    const [open, setOpen] = useState(false)
    const [newJobParams, setNewJobParams] = useState<TrainingParams>({
        model_architecture: "resnet18",
        model_size: "lightweight",
        model_variant: "resnet18_lightweight",
        learning_rate: DEFAULT_LEARNING_RATE.default,
        // A batch size of 1 trains on a single image at a time: gradients are
        // extremely noisy and the run is slow. Clicking straight through the
        // dialog produced ~35% validation accuracy on a 3-class problem where
        // batch 16 reached ~76% with everything else identical. 16 is the
        // standard starting point and fits comfortably in CPU memory.
        batch_size: 16,
        epochs: DEFAULT_EPOCHS.default,
        pretrained_model: "default",
        device: "auto",
        image_size: null,
    })

    // The architecture-specific list when the backend offers one, the project
    // type's otherwise. Two detectors now ship for the same project type and
    // they do not augment the same way, so a list chosen by type alone would
    // show an RF-DETR run a rotation control its pipeline cannot honour. The
    // `??` fallback is for a backend that predates the compound keys.
    const options =
        (project?.type &&
            (augmentations?.[`${project.type}::${newJobParams.model_architecture}`] ??
                augmentations?.[project.type])) ||
        []

    // What the menu should show as selected. "gpu" is what older sessions and
    // older builds say for "the accelerator"; and a choice this machine cannot
    // honour falls back to Automatic rather than leaving the control blank.
    const chosen = newJobParams.device ?? "auto"
    const deviceValue =
        chosen === "auto" || chosen === "cpu"
            ? chosen
            : (accelerators.find((accelerator) => accelerator.id === chosen)?.id ??
              (chosen === "gpu" ? accelerators[0]?.id : undefined) ??
              "auto")

    useEffect(() => {
        fetchModels(0, 20, modelSearch, newJobParams.model_architecture, newJobParams.model_size)
    }, [fetchModels, modelSearch, newJobParams.model_architecture, newJobParams.model_size])

    useEffect(() => {
        if (modelVariants && modelVariants.length > 0) {
            const firstVariant = modelVariants[0]
            setNewJobParams((prev) => ({
                ...prev,
                model_architecture: firstVariant.model_architecture,
                model_size: firstVariant.model_size,
            }))
        }
    }, [modelVariants])

    // Only until the user touches the field: after that the number is theirs,
    // and having the dialog quietly rewrite it while they type would be worse
    // than any default.
    const [epochsEdited, setEpochsEdited] = useState(false)
    useEffect(() => {
        if (epochsEdited || !project?.type) return
        setNewJobParams((prev) => ({
            ...prev,
            epochs: DEFAULT_EPOCHS[project.type] ?? DEFAULT_EPOCHS.default,
        }))
    }, [project?.type, epochsEdited])

    useEffect(() => {
        if (!isStructured) return
        setNewJobParams((previous) => ({ ...previous, device: "cpu", image_size: null }))
    }, [isStructured])

    // Follow the selected architecture until the user deliberately chooses a
    // batch. This prevents the generic image-classification default from being
    // carried into a much heavier keypoint model.
    const [batchEdited, setBatchEdited] = useState(false)
    useEffect(() => {
        if (batchEdited) return
        setNewJobParams((previous) => ({
            ...previous,
            batch_size: DEFAULT_BATCH_SIZE[previous.model_architecture] ?? DEFAULT_BATCH_SIZE.default,
        }))
    }, [newJobParams.model_architecture, batchEdited])

    // Same rule for the rate, keyed on the architecture: it follows the variant
    // menu until the field is touched, and then it is the user's.
    const [rateEdited, setRateEdited] = useState(false)
    useEffect(() => {
        if (rateEdited) return
        setNewJobParams((prev) => ({
            ...prev,
            learning_rate: DEFAULT_LEARNING_RATE[prev.model_architecture] ?? DEFAULT_LEARNING_RATE.default,
        }))
    }, [newJobParams.model_architecture, rateEdited])

    const startNewJob = async () => {
        await onStartTraining(newJobParams)
    }

    return (
        <Dialog open={isOpen} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-2xl">
                <DialogHeader>
                    <DialogTitle>Start a training run</DialogTitle>
                    <DialogDescription>
                        These settings are saved with the run, so you can compare them against earlier ones.
                    </DialogDescription>
                </DialogHeader>
                <div className="-mx-1 max-h-[65vh] space-y-3 overflow-y-auto px-1 py-3">
                    <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                        <div className="space-y-4">
                            <div className="flex flex-col gap-1.5">
                                <div className="flex items-center gap-2">
                                    <Scale className="h-4 w-4" />
                                    <Label htmlFor="model_variant">Model variant</Label>
                                    <InfoHint label="Model variant">
                                        Larger variants are usually more accurate but need more memory and time.
                                    </InfoHint>
                                </div>
                                <Select
                                    value={`${newJobParams.model_architecture}_${newJobParams.model_size}`}
                                    onValueChange={(value) => {
                                        const selectedVariant = modelVariants.find(
                                            (variant) => `${variant.model_architecture}_${variant.model_size}` === value
                                        )
                                        if (selectedVariant) {
                                            setNewJobParams({
                                                ...newJobParams,
                                                model_architecture: selectedVariant.model_architecture,
                                                model_size: selectedVariant.model_size,
                                            })
                                        }
                                    }}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="Choose a variant">
                                            {modelVariants.find(
                                                (v) =>
                                                    v.model_architecture === newJobParams.model_architecture &&
                                                    v.model_size === newJobParams.model_size
                                            )?.name || "Choose a variant"}
                                        </SelectValue>
                                    </SelectTrigger>
                                    <SelectContent>
                                        {modelVariants.map((variant) => (
                                            <SelectItem
                                                key={`${variant.model_architecture}_${variant.model_size}`}
                                                value={`${variant.model_architecture}_${variant.model_size}`}
                                            >
                                                {variant.name}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>

                            <div className="flex flex-col gap-1.5">
                                <div className="flex items-center gap-2">
                                    <Gauge className="h-4 w-4" />
                                    <Label htmlFor="learning_rate">
                                        {newJobParams.model_architecture === "tfidf-logreg"
                                            ? "Regularization step"
                                            : "Learning rate"}
                                    </Label>
                                    <InfoHint label="Learning rate">
                                        How far the model moves on each correction. Usually between 0.0001 and 0.01.
                                    </InfoHint>
                                </div>
                                <Input
                                    id="learning_rate"
                                    value={newJobParams.learning_rate}
                                    onChange={(e) => {
                                        setRateEdited(true)
                                        setNewJobParams({
                                            ...newJobParams,
                                            learning_rate: parseFloat(e.target.value),
                                        })
                                    }}
                                    type="number"
                                    // 0.0001, not 0.001: a number input validates its value
                                    // as a multiple of the step from `min` (0 here), so with
                                    // a step of 0.001 RF-DETR's own default rate would be
                                    // rejected by the browser as invalid.
                                    step="0.0001"
                                />
                            </div>

                            <div className="flex flex-col gap-1.5">
                                <div className="flex items-center gap-2">
                                    <Repeat className="h-4 w-4" />
                                    <Label htmlFor="epochs">{isStructured ? "Maximum iterations" : "Epochs"}</Label>
                                    <InfoHint label={isStructured ? "Maximum iterations" : "Epochs"}>
                                        {isStructured
                                            ? "An upper limit. CatBoost stops early when validation quality no longer improves; text classification stops after convergence."
                                            : "How many times the model sees the whole training set. The default suits this task: fine-tuning a pretrained model needs few passes, while the handpose classifier is trained from scratch and needs hundreds — at ten epochs it is still guessing."}
                                    </InfoHint>
                                </div>
                                <Input
                                    id="epochs"
                                    value={newJobParams.epochs}
                                    onChange={(e) => {
                                        setEpochsEdited(true)
                                        setNewJobParams({
                                            ...newJobParams,
                                            epochs: parseInt(e.target.value),
                                        })
                                    }}
                                    type="number"
                                    step="1"
                                />
                            </div>
                        </div>

                        {isStructured ? (
                            <div className="bg-surface-sunken space-y-3 rounded-lg border p-4">
                                <p className="text-sm font-medium">Reproducible evaluation</p>
                                <p className="text-muted-foreground text-xs leading-relaxed">
                                    The run uses the split, primary metric, column roles and class controls saved in the
                                    Dataset workspace. It compares against a simple baseline and builds a Smart Review
                                    queue from uncertainty and class diversity. Structured models run on the CPU and
                                    keep the original data local.
                                </p>
                                <div className="grid grid-cols-3 gap-2 text-center font-mono text-xs">
                                    <div className="rounded border p-2">
                                        <span className="text-muted-foreground block">split</span>saved
                                    </div>
                                    <div className="rounded border p-2">
                                        <span className="text-muted-foreground block">metric</span>saved
                                    </div>
                                    <div className="rounded border p-2">
                                        <span className="text-muted-foreground block">seed</span>saved
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <div className="flex flex-col gap-1.5">
                                    <div className="flex items-center gap-2">
                                        <Database className="h-4 w-4" />
                                        <Label htmlFor="pretrained_model">Starting weights</Label>
                                        <InfoHint label="Starting weights">
                                            Start from the default weights, or continue from a model you already
                                            trained.
                                        </InfoHint>
                                    </div>
                                    <Popover open={open} onOpenChange={setOpen}>
                                        <PopoverTrigger asChild>
                                            <Button
                                                variant="outline"
                                                role="combobox"
                                                aria-expanded={open}
                                                className="w-full justify-between font-normal"
                                            >
                                                {newJobParams.pretrained_model === "default"
                                                    ? "Default"
                                                    : models?.find(
                                                          (model) =>
                                                              model.id.toString() === newJobParams.pretrained_model
                                                      )?.name || "Choose a model"}
                                                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                                            </Button>
                                        </PopoverTrigger>
                                        <PopoverContent className="w-full p-0">
                                            <Command>
                                                <CommandInput
                                                    placeholder="Search models"
                                                    value={modelSearch}
                                                    onValueChange={setModelSearch}
                                                />
                                                <CommandList>
                                                    <CommandEmpty>No models found.</CommandEmpty>
                                                    <CommandGroup>
                                                        <CommandItem
                                                            value="default"
                                                            onSelect={() => {
                                                                setNewJobParams({
                                                                    ...newJobParams,
                                                                    pretrained_model: "default",
                                                                })
                                                                setOpen(false)
                                                            }}
                                                        >
                                                            Default
                                                            <Check
                                                                className={cn(
                                                                    "ml-auto h-4 w-4",
                                                                    newJobParams.pretrained_model === "default"
                                                                        ? "opacity-100"
                                                                        : "opacity-0"
                                                                )}
                                                            />
                                                        </CommandItem>
                                                        {models?.map((model) => (
                                                            <CommandItem
                                                                key={model.id}
                                                                value={model.path}
                                                                onSelect={() => {
                                                                    setNewJobParams({
                                                                        ...newJobParams,
                                                                        pretrained_model: model.id.toString(),
                                                                    })
                                                                    setOpen(false)
                                                                }}
                                                            >
                                                                {model.name}
                                                                <Check
                                                                    className={cn(
                                                                        "ml-auto h-4 w-4",
                                                                        newJobParams.pretrained_model ===
                                                                            model.id.toString()
                                                                            ? "opacity-100"
                                                                            : "opacity-0"
                                                                    )}
                                                                />
                                                            </CommandItem>
                                                        ))}
                                                    </CommandGroup>
                                                </CommandList>
                                            </Command>
                                        </PopoverContent>
                                    </Popover>
                                </div>

                                <div className="flex flex-col gap-1.5">
                                    <div className="flex items-center gap-2">
                                        <Box className="h-4 w-4" />
                                        <Label htmlFor="batch_size">Batch size</Label>
                                        <InfoHint label="Batch size">
                                            How many images the model looks at before each correction. Larger is
                                            steadier and faster per epoch, and uses more memory — if a run stops with an
                                            out-of-memory error, halve this first.
                                        </InfoHint>
                                    </div>
                                    <Input
                                        id="batch_size"
                                        value={newJobParams.batch_size}
                                        onChange={(e) => {
                                            setBatchEdited(true)
                                            setNewJobParams({
                                                ...newJobParams,
                                                batch_size: parseInt(e.target.value),
                                            })
                                        }}
                                        type="number"
                                        step="1"
                                    />
                                </div>
                                <div className="flex flex-col gap-1.5">
                                    <div className="flex items-center gap-2">
                                        <Maximize2 className="h-4 w-4" />
                                        <Label htmlFor="image_size">Image size</Label>
                                        <InfoHint label="Image size">
                                            What each image is resized to for training. Larger sees smaller objects and
                                            costs memory and time — quadratically.
                                        </InfoHint>
                                    </div>
                                    <Select
                                        value={newJobParams.image_size ? String(newJobParams.image_size) : "default"}
                                        onValueChange={(value) =>
                                            setNewJobParams({
                                                ...newJobParams,
                                                image_size: value === "default" ? null : parseInt(value),
                                            })
                                        }
                                    >
                                        <SelectTrigger id="image_size">
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="default">Model default</SelectItem>
                                            {[320, 416, 512, 640, 768, 1024].map((size) => (
                                                <SelectItem key={size} value={String(size)}>
                                                    {size} × {size}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>

                                <div className="flex flex-col gap-1.5">
                                    <div className="flex items-center gap-2">
                                        <Cpu className="h-4 w-4" />
                                        <Label htmlFor="device">Hardware</Label>
                                        <InfoHint label="Hardware">
                                            {accelerators.length > 0
                                                ? `Automatic uses ${accelerators[0].name ?? accelerators[0].label}. Pick CPU when it is busy or short of memory.`
                                                : "This project type trains on the CPU on this machine."}
                                        </InfoHint>
                                    </div>
                                    <Select
                                        value={deviceValue}
                                        onValueChange={(value) =>
                                            setNewJobParams({
                                                ...newJobParams,
                                                device: value as TrainingParams["device"],
                                            })
                                        }
                                    >
                                        <SelectTrigger id="device">
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="auto">Automatic</SelectItem>
                                            {accelerators.map((accelerator) => (
                                                <SelectItem key={accelerator.id} value={accelerator.id}>
                                                    {accelerator.label}
                                                </SelectItem>
                                            ))}
                                            <SelectItem value="cpu">CPU</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>
                        )}

                        {options.length > 0 && (
                            <div className="flex flex-col gap-1.5">
                                <div className="flex items-center gap-2">
                                    <Shuffle className="h-4 w-4" />
                                    <Label>Augmentation</Label>
                                    <InfoHint label="Augmentation">
                                        Random changes applied to each training image, so the model learns the object
                                        rather than the photograph. Only the settings this model can actually honour are
                                        shown.
                                    </InfoHint>
                                </div>
                                <div className="space-y-3 rounded-md border p-3">
                                    {options.map((option) => {
                                        const current = newJobParams.augmentation?.[option.key] ?? option.default
                                        const set = (value: boolean | number) =>
                                            setNewJobParams({
                                                ...newJobParams,
                                                augmentation: {
                                                    ...newJobParams.augmentation,
                                                    [option.key]: value,
                                                },
                                            })
                                        return (
                                            <div key={option.key} className="flex items-start justify-between gap-4">
                                                <div className="flex min-w-0 items-center gap-2">
                                                    <p className="text-sm">{option.label}</p>
                                                    {option.help && (
                                                        <InfoHint label={option.label}>{option.help}</InfoHint>
                                                    )}
                                                </div>
                                                {option.type === "bool" ? (
                                                    <Switch
                                                        checked={Boolean(current)}
                                                        onCheckedChange={set}
                                                        aria-label={option.label}
                                                    />
                                                ) : (
                                                    <Input
                                                        type="number"
                                                        className="w-20 shrink-0"
                                                        value={Number(current)}
                                                        min={option.minimum}
                                                        max={option.maximum}
                                                        step={option.step}
                                                        onChange={(event) => set(Number(event.target.value))}
                                                        aria-label={option.label}
                                                    />
                                                )}
                                            </div>
                                        )
                                    })}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
                <DialogFooter>
                    <Button variant="outline" onClick={() => onOpenChange(false)}>
                        Cancel
                    </Button>
                    <Button onClick={startNewJob}>Start training</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}

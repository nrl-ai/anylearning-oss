"use client"

import { useQuery, useQueryClient } from "@tanstack/react-query"
import { AlertTriangle, ArrowRight, BarChart3, BrainCircuit, Check, Download, Play, Scale } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

import { Button } from "@/components/ui/button"
import { EmptyState } from "@/components/ui/empty-state"
import { Input } from "@/components/ui/input"
import { Panel, PanelBody, PanelHeader, Stat } from "@/components/ui/panel"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"
import { toast } from "@/components/ui/use-toast"
import { api, getJson, withToken } from "@/lib/api"
import { isTextAiProject } from "@/lib/project-types"
import useModels from "@/lib/use-models"
import { Project } from "@/types"

type ReviewRow = {
    rank: number
    row_id: number
    prediction: string
    actual: unknown
    uncertainty: number
    confidence: number
    duplicate_group: string | null
    reason: string
}

type ModelReport = {
    engine: string
    task: string
    target: string
    metrics: Record<string, number>
    primary_metric?: string
    evaluation?: {
        labels: string[]
        confusion_matrix: number[][]
        per_class: Array<{ label: string; precision: number; recall: number; f1: number; support: number }>
    } | null
    configuration?: {
        class_balance?: string
        text_features?: string
        iterations?: number
        learning_rate?: number
        depth?: number
    }
    baseline: { engine: string; metrics: Record<string, number> }
    feature_importance?: Array<{ feature: string; importance: number }>
    review_queue: ReviewRow[]
    dataset_rows?: number
    training_rows?: number
    training_sampled?: boolean
}

export default function StructuredModels({ projectId, project }: { projectId: number; project: Project }) {
    const queryClient = useQueryClient()
    const { models, fetchModels, isLoading } = useModels(projectId)
    const [modelId, setModelId] = useState<number | null>(null)
    const [predictionInput, setPredictionInput] = useState(
        isTextAiProject(project.type) ? '[\n  { "text": "My transfer is still pending" }\n]' : "[\n  { }\n]"
    )
    const [predictions, setPredictions] = useState<unknown[] | null>(null)

    useEffect(() => {
        fetchModels(0, 100, "")
    }, [fetchModels])
    useEffect(() => {
        if (modelId === null && models.length) setModelId(models[0].id)
    }, [modelId, models])

    const { data: report, isLoading: reportLoading } = useQuery({
        queryKey: ["structured-report", projectId, modelId],
        queryFn: () => getJson<ModelReport>(`/api/projects/${projectId}/structured/models/${modelId}/report`),
        enabled: modelId !== null,
    })
    const selectedModel = models.find((model) => model.id === modelId)
    const metricComparison = useMemo(() => {
        if (!report) return []
        return Object.entries(report.metrics).map(([name, value]) => ({
            name,
            value,
            baseline: report.baseline.metrics[name],
        }))
    }, [report])

    const runPrediction = async () => {
        if (!modelId) return
        try {
            const rows = JSON.parse(predictionInput)
            if (!Array.isArray(rows)) throw new Error("Input must be a JSON array of rows.")
            const response = await api.post(`/api/projects/${projectId}/structured/models/${modelId}/predict`, { rows })
            setPredictions(response.data.predictions)
        } catch (error) {
            toast({
                title: "Prediction input is not valid",
                description:
                    (error as { response?: { data?: { detail?: string } }; message?: string })?.response?.data
                        ?.detail ?? (error as Error).message,
                variant: "destructive",
            })
        }
    }

    const correctRow = async (rowId: number, value: string) => {
        if (!report?.target) return
        await api.patch(`/api/projects/${projectId}/structured/rows/${rowId}`, {
            values: { [report.target]: value },
        })
        await queryClient.invalidateQueries({ queryKey: ["structured-rows", projectId] })
        toast({ title: "Review saved", description: `Row ${rowId} will be corrected in the next training run.` })
    }

    if (isLoading) return <p className="text-muted-foreground p-8 text-sm">Loading models…</p>
    if (!models.length) {
        return (
            <Panel>
                <EmptyState
                    icon={BrainCircuit}
                    title="No structured model yet"
                    description="Configure the dataset first, then train. The finished run will include held-out metrics, a baseline and a Smart Review queue."
                    action={
                        <Button asChild>
                            <a href={`/projects/training?projectId=${projectId}`}>
                                Go to training <ArrowRight />
                            </a>
                        </Button>
                    }
                />
            </Panel>
        )
    }

    return (
        <div className="space-y-4">
            <Panel>
                <PanelHeader
                    icon={BrainCircuit}
                    title="Structured model report"
                    description={report ? `${report.engine} · held-out evaluation` : "Loading the model report…"}
                    actions={
                        <div className="flex items-center gap-2">
                            <Select
                                value={modelId ? String(modelId) : undefined}
                                onValueChange={(value) => setModelId(Number(value))}
                            >
                                <SelectTrigger className="h-8 w-64">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    {models.map((model) => (
                                        <SelectItem key={model.id} value={String(model.id)}>
                                            {model.name}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                            {selectedModel && (
                                <Button variant="outline" size="sm" asChild>
                                    <a
                                        href={withToken(
                                            `/api/projects/${projectId}/models/${selectedModel.id}/download`
                                        )}
                                        download
                                    >
                                        <Download /> Native model
                                    </a>
                                </Button>
                            )}
                        </div>
                    }
                />
                <PanelBody>
                    {reportLoading || !report ? (
                        <p className="text-muted-foreground text-sm">Loading report…</p>
                    ) : (
                        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                            {metricComparison.map((metric) => (
                                <Stat
                                    key={metric.name}
                                    label={
                                        metric.name === report.primary_metric ? `${metric.name} · primary` : metric.name
                                    }
                                    value={metric.value.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                                    hint={
                                        metric.baseline === undefined
                                            ? undefined
                                            : `${report.baseline.engine}: ${metric.baseline.toLocaleString(undefined, { maximumFractionDigits: 4 })}`
                                    }
                                />
                            ))}
                        </div>
                    )}
                    {report?.configuration && (
                        <p className="text-muted-foreground mt-4 border-t pt-3 text-xs">
                            {report.configuration.class_balance &&
                                `Class weighting: ${report.configuration.class_balance}`}
                            {report.configuration.text_features &&
                                ` · Text features: ${report.configuration.text_features.replaceAll("_", " + ")}`}
                            {report.configuration.depth && ` · Tree depth: ${report.configuration.depth}`}
                            {report.configuration.iterations &&
                                ` · Maximum iterations: ${report.configuration.iterations}`}
                        </p>
                    )}
                </PanelBody>
            </Panel>

            {report && (
                <div className="space-y-4">
                    {!!report.evaluation?.per_class.length && (
                        <div className="grid gap-4 xl:grid-cols-2">
                            <Panel>
                                <PanelHeader
                                    icon={BarChart3}
                                    title="Class performance"
                                    description="Held-out classes sorted from lowest to highest F1, so weak classes are visible first."
                                />
                                <PanelBody className="max-h-96 overflow-auto p-0">
                                    <Table>
                                        <TableHeader>
                                            <TableRow>
                                                <TableHead>Class</TableHead>
                                                <TableHead className="text-right">Precision</TableHead>
                                                <TableHead className="text-right">Recall</TableHead>
                                                <TableHead className="text-right">F1</TableHead>
                                                <TableHead className="text-right">Rows</TableHead>
                                            </TableRow>
                                        </TableHeader>
                                        <TableBody>
                                            {[...report.evaluation.per_class]
                                                .sort((left, right) => left.f1 - right.f1)
                                                .map((item) => (
                                                    <TableRow key={item.label}>
                                                        <TableCell className="max-w-52 truncate">
                                                            {item.label}
                                                        </TableCell>
                                                        <TableCell className="text-right font-mono">
                                                            {item.precision.toFixed(3)}
                                                        </TableCell>
                                                        <TableCell className="text-right font-mono">
                                                            {item.recall.toFixed(3)}
                                                        </TableCell>
                                                        <TableCell className="text-right font-mono">
                                                            {item.f1.toFixed(3)}
                                                        </TableCell>
                                                        <TableCell className="text-right font-mono">
                                                            {item.support.toLocaleString()}
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                        </TableBody>
                                    </Table>
                                </PanelBody>
                            </Panel>
                            {report.evaluation.labels.length <= 10 && (
                                <Panel>
                                    <PanelHeader
                                        icon={Scale}
                                        title="Confusion matrix"
                                        description="Rows are actual classes; columns are predictions on held-out data."
                                    />
                                    <PanelBody className="overflow-auto p-0">
                                        <Table>
                                            <TableHeader>
                                                <TableRow>
                                                    <TableHead>Actual ↓ / predicted →</TableHead>
                                                    {report.evaluation.labels.map((label) => (
                                                        <TableHead key={label} className="max-w-24 truncate text-right">
                                                            {label}
                                                        </TableHead>
                                                    ))}
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {report.evaluation.confusion_matrix.map((row, rowIndex) => (
                                                    <TableRow key={report.evaluation?.labels[rowIndex]}>
                                                        <TableCell className="max-w-40 truncate font-medium">
                                                            {report.evaluation?.labels[rowIndex]}
                                                        </TableCell>
                                                        {row.map((value, columnIndex) => (
                                                            <TableCell
                                                                key={report.evaluation?.labels[columnIndex]}
                                                                className={
                                                                    rowIndex === columnIndex
                                                                        ? "bg-mark/10 text-right font-mono font-medium"
                                                                        : "text-right font-mono"
                                                                }
                                                            >
                                                                {value}
                                                            </TableCell>
                                                        ))}
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </PanelBody>
                                </Panel>
                            )}
                        </div>
                    )}
                    <div className={report.feature_importance?.length ? "grid gap-4 xl:grid-cols-4" : "grid gap-4"}>
                        <Panel className={report.feature_importance?.length ? "xl:col-span-3" : undefined}>
                            <PanelHeader
                                icon={Scale}
                                title="Smart Review"
                                description={
                                    report.training_sampled
                                        ? `Model and review candidates use a reproducible ${report.training_rows?.toLocaleString()}-row sample of ${report.dataset_rows?.toLocaleString()} rows. Rows are ranked by uncertainty and diversified across classes.`
                                        : "Rows are ranked by uncertainty, then interleaved across predicted classes. Duplicate and disagreement reasons stay visible."
                                }
                            />
                            <PanelBody className="overflow-x-auto p-0">
                                <Table className="min-w-[1060px] table-fixed">
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead className="w-16">Rank</TableHead>
                                            <TableHead className="w-20">Row</TableHead>
                                            <TableHead className="w-56">Prediction</TableHead>
                                            <TableHead className="w-28">Confidence</TableHead>
                                            <TableHead className="w-72">Why selected</TableHead>
                                            <TableHead className="w-64">Reviewed label</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {report.review_queue.slice(0, 25).map((row) => (
                                            <TableRow key={row.row_id}>
                                                <TableCell className="w-16 font-mono">{row.rank}</TableCell>
                                                <TableCell className="w-20 font-mono">{row.row_id}</TableCell>
                                                <TableCell className="w-56 break-all whitespace-normal">
                                                    {row.prediction}
                                                </TableCell>
                                                <TableCell className="w-28 font-mono">
                                                    {Math.round(row.confidence * 100)}%
                                                </TableCell>
                                                <TableCell className="text-muted-foreground w-72 text-xs whitespace-normal">
                                                    <span className="flex items-start gap-1.5">
                                                        {row.duplicate_group && (
                                                            <AlertTriangle className="text-run mt-0.5 size-3 shrink-0" />
                                                        )}
                                                        {row.reason}
                                                    </span>
                                                </TableCell>
                                                <TableCell className="w-64">
                                                    <ReviewInput
                                                        value={row.actual == null ? "" : String(row.actual)}
                                                        onSave={(value) => correctRow(row.row_id, value)}
                                                    />
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </PanelBody>
                        </Panel>

                        {!!report.feature_importance?.length && (
                            <Panel>
                                <PanelHeader
                                    icon={BarChart3}
                                    title="Feature importance"
                                    description="Global CatBoost importance; association is not causation."
                                />
                                <PanelBody className="space-y-3">
                                    {report.feature_importance.slice(0, 12).map((item) => {
                                        const maximum = report.feature_importance?.[0]?.importance || 1
                                        return (
                                            <div key={item.feature} className="space-y-1">
                                                <div className="flex justify-between gap-3 text-xs">
                                                    <span className="truncate">{item.feature}</span>
                                                    <span className="font-mono">{item.importance.toFixed(2)}</span>
                                                </div>
                                                <div className="bg-muted h-1.5 overflow-hidden rounded-full">
                                                    <div
                                                        className="bg-mark h-full rounded-full"
                                                        style={{
                                                            width: `${Math.max(2, (item.importance / maximum) * 100)}%`,
                                                        }}
                                                    />
                                                </div>
                                            </div>
                                        )
                                    })}
                                </PanelBody>
                            </Panel>
                        )}
                    </div>
                </div>
            )}

            <Panel>
                <PanelHeader
                    icon={Play}
                    title="Try rows"
                    description="Paste a JSON array. Nothing is uploaded outside this machine."
                    actions={
                        <Button size="sm" onClick={runPrediction}>
                            <Play /> Predict
                        </Button>
                    }
                />
                <PanelBody className="grid gap-4 lg:grid-cols-2">
                    <Textarea
                        className="min-h-36 font-mono text-xs"
                        value={predictionInput}
                        onChange={(event) => setPredictionInput(event.target.value)}
                    />
                    <pre className="bg-surface-sunken min-h-36 overflow-auto rounded-md border p-3 text-xs">
                        {predictions ? JSON.stringify(predictions, null, 2) : "Predictions appear here."}
                    </pre>
                </PanelBody>
            </Panel>
        </div>
    )
}

function ReviewInput({ value, onSave }: { value: string; onSave: (value: string) => Promise<void> }) {
    const [current, setCurrent] = useState(value)
    return (
        <div className="flex min-w-40 gap-1">
            <Input value={current} onChange={(event) => setCurrent(event.target.value)} className="h-7" />
            {current !== value && (
                <Button
                    variant="ghost"
                    size="icon"
                    className="size-7"
                    onClick={() => onSave(current)}
                    aria-label="Save reviewed label"
                >
                    <Check className="size-3.5" />
                </Button>
            )}
        </div>
    )
}

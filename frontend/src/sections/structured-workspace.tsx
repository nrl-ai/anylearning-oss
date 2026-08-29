"use client"

import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
    ArrowRight,
    BarChart3,
    BrainCircuit,
    Check,
    Database,
    Download,
    FileSpreadsheet,
    Search,
    Sparkles,
    Upload,
} from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { EmptyState } from "@/components/ui/empty-state"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Panel, PanelBody, PanelHeader, Stat } from "@/components/ui/panel"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { toast } from "@/components/ui/use-toast"
import { api, getJson, withToken } from "@/lib/api"
import { Project } from "@/types"

type ProfileColumn = {
    name: string
    type: "numeric" | "text" | "datetime" | "boolean"
    missing: number
    missing_percent: number
    unique: number
    examples: unknown[]
    profile_rows?: number
    estimated?: boolean
    minimum?: number
    maximum?: number
    mean?: number
}

type StructuredMetadata = {
    version: number
    configured: boolean
    source?: {
        filename: string
        rows: number
        columns: number
        bytes: number
        sha256: string
    }
    task?: {
        type: string
        target: string | null
        text_column: string | null
        id_column?: string | null
        ignored_columns?: string[]
        primary_metric?: string | null
        class_balance?: "balanced" | "natural"
        text_features?: "word_character" | "word" | "character"
        prompt_column?: string | null
        response_column?: string | null
        reference_column?: string | null
    }
    split?: { train: number; validation: number; test: number; seed: number }
    profile?: ProfileColumn[]
    performance?: {
        storage_engine: string
        memory_limit?: string
        paged_loading: boolean
        batch_rows: number
        profile_rows: number
        profile_is_sampled: boolean
        tabular_training_row_limit: number
        text_training_row_limit: number
    }
    attribution?: { name?: string; url?: string; license?: string; citation?: string }
}

type CatalogDataset = {
    slug: string
    name: string
    description: string
    task: string
    rows: number
    target?: string
    text_column?: string
    prompt_column?: string
    response_column?: string
    reference_column?: string
    license: string
}

type RowsPage = {
    total: number
    dataset_total: number
    paged: boolean
    offset: number
    limit: number
    rows: Array<Record<string, unknown> & { _row_id: number }>
}

type HuggingFaceInfo = {
    dataset_id: string
    name: string
    url: string
    licenses: string[]
    gated: boolean
    configs: Record<string, Record<string, { files: number; bytes: number }>>
    downloads?: number
    likes?: number
}

const PAGE_SIZE = 25
const COLUMN_PAGE_SIZE = 10

export default function StructuredWorkspace({ projectId, project }: { projectId: number; project: Project }) {
    const queryClient = useQueryClient()
    const uploadRef = useRef<HTMLInputElement>(null)
    const [uploading, setUploading] = useState(false)
    const [query, setQuery] = useState("")
    const [tableQuery, setTableQuery] = useState("")
    const [offset, setOffset] = useState(0)
    const [columnOffset, setColumnOffset] = useState(0)
    const [taskType, setTaskType] = useState("")
    const [target, setTarget] = useState("")
    const [textColumn, setTextColumn] = useState("")
    const [promptColumn, setPromptColumn] = useState("")
    const [responseColumn, setResponseColumn] = useState("")
    const [referenceColumn, setReferenceColumn] = useState("")
    const [idColumn, setIdColumn] = useState("")
    const [ignoredColumns, setIgnoredColumns] = useState<string[]>([])
    const [primaryMetric, setPrimaryMetric] = useState("")
    const [classBalance, setClassBalance] = useState<"balanced" | "natural">("balanced")
    const [textFeatures, setTextFeatures] = useState<"word_character" | "word" | "character">("word_character")
    const [validationPercent, setValidationPercent] = useState(15)
    const [testPercent, setTestPercent] = useState(15)
    const [splitSeed, setSplitSeed] = useState(42)
    const [savingConfig, setSavingConfig] = useState(false)
    const [searchText, setSearchText] = useState("")
    const [searchResults, setSearchResults] = useState<Array<{ row_id: number; score: number; text: string }>>([])
    const [evaluation, setEvaluation] = useState<Record<string, number> | null>(null)
    const [hfDataset, setHfDataset] = useState("")
    const [hfInfo, setHfInfo] = useState<HuggingFaceInfo | null>(null)
    const [hfConfig, setHfConfig] = useState("")
    const [hfSplit, setHfSplit] = useState("")
    const [hfRows, setHfRows] = useState(50000)
    const [hfLoading, setHfLoading] = useState(false)

    useEffect(() => {
        const timer = window.setTimeout(() => setTableQuery(query), 250)
        return () => window.clearTimeout(timer)
    }, [query])

    const { data: metadata, isLoading } = useQuery({
        queryKey: ["structured", projectId],
        queryFn: () => getJson<StructuredMetadata>(`/api/projects/${projectId}/structured`),
    })
    const columns = metadata?.profile ?? []
    const columnNames = columns.map((column) => column.name)
    const visibleColumns = columnNames.slice(columnOffset, columnOffset + COLUMN_PAGE_SIZE)
    useEffect(() => {
        if (columnNames.length && columnOffset >= columnNames.length) setColumnOffset(0)
    }, [columnNames.length, columnOffset])
    const { data: catalog } = useQuery({
        queryKey: ["structured-catalog"],
        queryFn: () => getJson<{ source: string; datasets: CatalogDataset[] }>("/api/structured/catalog"),
        staleTime: 5 * 60 * 1000,
    })
    const { data: page, isFetching: rowsLoading } = useQuery({
        queryKey: ["structured-rows", projectId, offset, tableQuery, columnOffset],
        queryFn: () => {
            const parameters = new URLSearchParams({
                offset: String(offset),
                limit: String(PAGE_SIZE),
                query: tableQuery,
            })
            for (const name of visibleColumns) parameters.append("columns", name)
            return getJson<RowsPage>(`/api/projects/${projectId}/structured/rows?${parameters}`)
        },
        enabled: Boolean(metadata?.source),
    })

    const taskOptions =
        project.type === "Tabular AI"
            ? [
                  ["classification", "Classification"],
                  ["regression", "Regression"],
              ]
            : [
                  ["text_classification", "Text classification"],
                  ["lexical_search", "Lexical & fuzzy search"],
                  ["llm_evaluation", "Response evaluation"],
              ]

    const selectedTask =
        metadata?.task?.type === "semantic_search" ? "lexical_search" : (metadata?.task?.type ?? taskType)
    const activeTask = taskType || selectedTask
    const metricOptions =
        activeTask === "regression"
            ? ["RMSE", "MAE", "R²"]
            : activeTask === "classification" || activeTask === "text_classification"
              ? ["Balanced Accuracy", "Macro F1", "Accuracy", "Log Loss"]
              : []
    const defaultMetric =
        activeTask === "regression" ? "RMSE" : activeTask === "text_classification" ? "Macro F1" : "Balanced Accuracy"
    const workflowDisclosure =
        activeTask === "text_classification"
            ? {
                  title: "Trainable classical ML",
                  detail: "Word and character TF-IDF with balanced logistic regression; averaged SGD is used above 100,000 training rows. No LLM is loaded.",
              }
            : activeTask === "lexical_search"
              ? {
                    title: "Model-free search baseline",
                    detail: "Hashed character n-grams and cosine similarity match wording and spelling variants. This is not semantic embedding search.",
                }
              : activeTask === "llm_evaluation"
                ? {
                      title: "Deterministic response metrics",
                      detail: "Completion, normalized exact match and token-overlap F1 evaluate responses already in your table. AnyLearning does not run or judge with an LLM.",
                  }
                : null
    const effectiveTarget = target || metadata?.task?.target || ""
    const effectiveText = textColumn || metadata?.task?.text_column || ""
    const effectivePrompt = promptColumn || metadata?.task?.prompt_column || ""
    const effectiveResponse = responseColumn || metadata?.task?.response_column || ""
    const effectiveReference = referenceColumn || metadata?.task?.reference_column || ""
    const effectiveId = idColumn || metadata?.task?.id_column || ""
    const effectiveMetric = metricOptions.includes(primaryMetric || metadata?.task?.primary_metric || "")
        ? primaryMetric || metadata?.task?.primary_metric || defaultMetric
        : defaultMetric
    const trainPercent = 100 - validationPercent - testPercent
    const isTrainable = ["classification", "regression", "text_classification"].includes(activeTask)
    const splitValid =
        !isTrainable ||
        (trainPercent > 0 && validationPercent >= 0 && testPercent >= 0 && validationPercent + testPercent > 0)
    const usesTabularFeatures = activeTask === "classification" || activeTask === "regression"

    useEffect(() => {
        if (idColumn && [effectiveTarget, effectiveText].includes(idColumn)) setIdColumn("")
    }, [effectiveTarget, effectiveText, idColumn])

    const hydratedSource = useRef("")
    useEffect(() => {
        const source = metadata?.source?.sha256
        if (!source || hydratedSource.current === source) return
        hydratedSource.current = source
        setIdColumn(metadata.task?.id_column ?? "")
        setIgnoredColumns(metadata.task?.ignored_columns ?? [])
        setPrimaryMetric(metadata.task?.primary_metric ?? "")
        setClassBalance(metadata.task?.class_balance ?? "balanced")
        setTextFeatures(metadata.task?.text_features ?? "word_character")
        setValidationPercent(Math.round((metadata.split?.validation ?? 0.15) * 100))
        setTestPercent(Math.round((metadata.split?.test ?? 0.15) * 100))
        setSplitSeed(metadata.split?.seed ?? 42)
    }, [metadata])

    const summary = useMemo(() => {
        if (!columns.length) return null
        const missing = columns.reduce((sum, column) => sum + column.missing, 0)
        const numeric = columns.filter((column) => column.type === "numeric").length
        return { missing, numeric, categorical: columns.length - numeric }
    }, [columns])

    const refresh = async () => {
        await queryClient.invalidateQueries({ queryKey: ["structured", projectId] })
        await queryClient.invalidateQueries({ queryKey: ["structured-rows", projectId] })
    }

    const uploadFile = async (file: File) => {
        const data = new FormData()
        data.append("file", file)
        setUploading(true)
        try {
            await api.post(`/api/projects/${projectId}/structured/upload`, data)
            setOffset(0)
            setColumnOffset(0)
            await refresh()
            toast({ title: "Dataset ready", description: `${file.name} was profiled and stored with its checksum.` })
        } catch (error) {
            toast({
                title: "Could not import dataset",
                description:
                    (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
                    "Check the file format.",
                variant: "destructive",
            })
        } finally {
            setUploading(false)
        }
    }

    const installExample = async (dataset: CatalogDataset) => {
        setUploading(true)
        try {
            await api.post(`/api/projects/${projectId}/structured/catalog/${dataset.slug}`)
            setOffset(0)
            setColumnOffset(0)
            setTaskType(dataset.task)
            setTarget(dataset.target ?? "")
            setTextColumn(dataset.text_column ?? "")
            await refresh()
            toast({ title: "Example downloaded", description: `${dataset.name} is ready to configure.` })
        } catch (error) {
            toast({
                title: "Example download failed",
                description:
                    (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
                    "The CDN may be unavailable.",
                variant: "destructive",
            })
        } finally {
            setUploading(false)
        }
    }

    const inspectHuggingFace = async () => {
        setHfLoading(true)
        try {
            const info = await getJson<HuggingFaceInfo>(
                `/api/structured/huggingface/inspect?dataset_id=${encodeURIComponent(hfDataset.trim())}`
            )
            setHfInfo(info)
            const firstConfig = Object.keys(info.configs)[0] ?? ""
            setHfConfig(firstConfig)
            setHfSplit(Object.keys(info.configs[firstConfig] ?? {})[0] ?? "")
        } catch (error) {
            toast({
                title: "Could not inspect Hugging Face dataset",
                description:
                    (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
                    "Check the public dataset ID.",
                variant: "destructive",
            })
        } finally {
            setHfLoading(false)
        }
    }

    const importHuggingFace = async () => {
        if (!hfInfo || !hfConfig || !hfSplit) return
        setHfLoading(true)
        try {
            await api.post(`/api/projects/${projectId}/structured/huggingface`, {
                dataset_id: hfInfo.dataset_id,
                config: hfConfig,
                split: hfSplit,
                row_limit: hfRows,
            })
            setOffset(0)
            setColumnOffset(0)
            await refresh()
            toast({
                title: "Hugging Face dataset imported",
                description: `${hfInfo.dataset_id} is now a local, portable project snapshot.`,
            })
        } catch (error) {
            toast({
                title: "Hugging Face import failed",
                description:
                    (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
                    "The selected split could not be imported.",
                variant: "destructive",
            })
        } finally {
            setHfLoading(false)
        }
    }

    const saveConfiguration = async () => {
        setSavingConfig(true)
        try {
            const type = taskType || selectedTask
            await api.put(`/api/projects/${projectId}/structured/config`, {
                type,
                target: ["classification", "regression", "text_classification"].includes(type) ? effectiveTarget : null,
                text_column: ["text_classification", "lexical_search"].includes(type) ? effectiveText : null,
                prompt_column: type === "llm_evaluation" ? effectivePrompt : null,
                response_column: type === "llm_evaluation" ? effectiveResponse : null,
                reference_column: type === "llm_evaluation" && effectiveReference ? effectiveReference : null,
                id_column: isTrainable && effectiveId ? effectiveId : null,
                ignored_columns: isTrainable
                    ? ignoredColumns.filter((column) => ![effectiveTarget, effectiveText, effectiveId].includes(column))
                    : [],
                primary_metric: isTrainable ? effectiveMetric : null,
                class_balance: type === "classification" || type === "text_classification" ? classBalance : null,
                text_features: type === "text_classification" ? textFeatures : null,
                split: {
                    train: trainPercent / 100,
                    validation: validationPercent / 100,
                    test: testPercent / 100,
                    seed: splitSeed,
                },
            })
            await refresh()
            toast({
                title: "Workflow configured",
                description: "The schema and reproducible split are saved with this project.",
            })
        } catch (error) {
            toast({
                title: "Configuration needs attention",
                description:
                    (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
                    "Choose the required columns.",
                variant: "destructive",
            })
        } finally {
            setSavingConfig(false)
        }
    }

    const updateTarget = async (rowId: number, value: string) => {
        if (!effectiveTarget) return
        await api.patch(`/api/projects/${projectId}/structured/rows/${rowId}`, { values: { [effectiveTarget]: value } })
        await queryClient.invalidateQueries({ queryKey: ["structured-rows", projectId] })
        toast({ title: "Label corrected", description: `Row ${rowId} will use the reviewed value in the next run.` })
    }

    const runEvaluation = async () => {
        const response = await api.post(`/api/projects/${projectId}/structured/evaluate`)
        setEvaluation(response.data.metrics)
    }

    const runSearch = async () => {
        const response = await api.post(`/api/projects/${projectId}/structured/search`, {
            query: searchText,
            limit: 10,
        })
        setSearchResults(response.data.results)
    }

    if (isLoading) {
        return <div className="text-muted-foreground p-8 text-sm">Loading structured workspace…</div>
    }

    if (!metadata?.source) {
        const compatibleExamples = (catalog?.datasets ?? []).filter((item) =>
            project.type === "Tabular AI"
                ? ["classification", "regression"].includes(item.task)
                : ["text_classification", "lexical_search", "semantic_search", "llm_evaluation"].includes(item.task)
        )
        return (
            <div className="space-y-4">
                <Panel>
                    <EmptyState
                        icon={FileSpreadsheet}
                        title={
                            project.type === "Tabular AI"
                                ? "Bring a real table"
                                : "Bring text data or saved model responses"
                        }
                        description="CSV, TSV, Excel, Parquet and JSON Lines stay local. AnyLearning profiles the schema and preserves the original file and checksum."
                        action={
                            <>
                                <input
                                    ref={uploadRef}
                                    type="file"
                                    accept=".csv,.tsv,.xlsx,.xls,.parquet,.jsonl"
                                    className="hidden"
                                    onChange={(event) => event.target.files?.[0] && uploadFile(event.target.files[0])}
                                />
                                <Button onClick={() => uploadRef.current?.click()} disabled={uploading}>
                                    <Upload /> {uploading ? "Importing…" : "Choose dataset"}
                                </Button>
                            </>
                        }
                    />
                </Panel>
                {compatibleExamples.length > 0 && (
                    <Panel>
                        <PanelHeader
                            icon={Sparkles}
                            title="License-cleared examples"
                            description="Real datasets mirrored on the AnyLearning CDN with source, citation and license attached."
                        />
                        <PanelBody className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                            {compatibleExamples.map((dataset) => (
                                <button
                                    type="button"
                                    key={dataset.slug}
                                    onClick={() => installExample(dataset)}
                                    disabled={uploading}
                                    className="hover:border-mark/50 rounded-lg border p-4 text-left transition-colors disabled:opacity-60"
                                >
                                    <div className="flex items-start justify-between gap-3">
                                        <p className="font-medium">{dataset.name}</p>
                                        <span className="text-muted-foreground font-mono text-[0.6875rem]">
                                            {dataset.license}
                                        </span>
                                    </div>
                                    <p className="text-muted-foreground mt-2 text-sm leading-relaxed">
                                        {dataset.description}
                                    </p>
                                    <p className="text-mark mt-3 text-xs">
                                        {dataset.rows.toLocaleString()} rows · Download
                                    </p>
                                </button>
                            ))}
                        </PanelBody>
                    </Panel>
                )}
                <Panel>
                    <PanelHeader
                        icon={Database}
                        title="Import from Hugging Face"
                        description="Public Dataset Viewer Parquet only—no repository code, pickle files or dataset scripts are executed."
                    />
                    <PanelBody className="space-y-4">
                        <div className="flex gap-2">
                            <Input
                                value={hfDataset}
                                onChange={(event) => setHfDataset(event.target.value)}
                                placeholder="owner/dataset-name"
                                onKeyDown={(event) => event.key === "Enter" && hfDataset.trim() && inspectHuggingFace()}
                            />
                            <Button
                                variant="outline"
                                onClick={inspectHuggingFace}
                                disabled={!hfDataset.trim() || hfLoading}
                            >
                                {hfLoading && !hfInfo ? "Inspecting…" : "Inspect"}
                            </Button>
                        </div>
                        {hfInfo && (
                            <div className="bg-surface-sunken space-y-4 rounded-lg border p-4">
                                <div className="flex flex-wrap items-start justify-between gap-3">
                                    <div>
                                        <p className="font-medium">{hfInfo.name}</p>
                                        <a
                                            className="text-mark text-xs hover:underline"
                                            href={hfInfo.url}
                                            target="_blank"
                                            rel="noreferrer"
                                        >
                                            {hfInfo.dataset_id}
                                        </a>
                                    </div>
                                    <div className="text-right">
                                        <p className="font-mono text-xs">
                                            {hfInfo.licenses.join(", ") || "License not declared"}
                                        </p>
                                        <p className="text-muted-foreground mt-1 text-[0.6875rem]">
                                            {hfInfo.downloads?.toLocaleString() ?? "—"} downloads ·{" "}
                                            {hfInfo.likes?.toLocaleString() ?? "—"} likes
                                        </p>
                                    </div>
                                </div>
                                <div className="grid gap-3 md:grid-cols-3">
                                    <Field label="Config / subset">
                                        <Select
                                            value={hfConfig}
                                            onValueChange={(value) => {
                                                setHfConfig(value)
                                                setHfSplit(Object.keys(hfInfo.configs[value] ?? {})[0] ?? "")
                                            }}
                                        >
                                            <SelectTrigger>
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {Object.keys(hfInfo.configs).map((value) => (
                                                    <SelectItem key={value} value={value}>
                                                        {value}
                                                    </SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                    </Field>
                                    <Field label="Split">
                                        <Select value={hfSplit} onValueChange={setHfSplit}>
                                            <SelectTrigger>
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {Object.keys(hfInfo.configs[hfConfig] ?? {}).map((value) => (
                                                    <SelectItem key={value} value={value}>
                                                        {value}
                                                    </SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                    </Field>
                                    <Field label="Maximum rows">
                                        <Input
                                            type="number"
                                            min={100}
                                            max={200000}
                                            step={1000}
                                            value={hfRows}
                                            onChange={(event) => setHfRows(Number(event.target.value))}
                                        />
                                    </Field>
                                </div>
                                <div className="flex items-center justify-between gap-4">
                                    <p className="text-muted-foreground text-xs">
                                        The selected snapshot is capped at 200,000 rows and 2 GB. Original Hub identity,
                                        split and license metadata stay in the project archive.
                                    </p>
                                    <Button onClick={importHuggingFace} disabled={hfLoading}>
                                        {hfLoading ? "Importing…" : "Import snapshot"}
                                    </Button>
                                </div>
                            </div>
                        )}
                    </PanelBody>
                </Panel>
            </div>
        )
    }

    return (
        <div className="space-y-4">
            <Panel>
                <PanelHeader
                    icon={Database}
                    title={metadata.source.filename}
                    description={
                        metadata.attribution?.name
                            ? `${metadata.attribution.name} · ${metadata.attribution.license ?? "license not recorded"}`
                            : `SHA-256 ${metadata.source.sha256.slice(0, 12)}… · original preserved`
                    }
                    actions={
                        <div className="flex gap-2">
                            <Button variant="outline" size="sm" asChild>
                                <a href={withToken(`/api/projects/${projectId}/structured/export`)} download>
                                    <Download /> Export edited CSV
                                </a>
                            </Button>
                            <Button variant="outline" size="sm" onClick={() => uploadRef.current?.click()}>
                                Replace
                            </Button>
                            <input
                                ref={uploadRef}
                                type="file"
                                accept=".csv,.tsv,.xlsx,.xls,.parquet,.jsonl"
                                className="hidden"
                                onChange={(event) => event.target.files?.[0] && uploadFile(event.target.files[0])}
                            />
                        </div>
                    }
                />
                <PanelBody className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <Stat label="Rows" value={metadata.source.rows.toLocaleString()} />
                    <Stat label="Columns" value={metadata.source.columns.toLocaleString()} />
                    <Stat label="Numeric" value={(summary?.numeric ?? 0).toLocaleString()} />
                    <Stat label="Missing cells" value={(summary?.missing ?? 0).toLocaleString()} />
                    <div className="text-muted-foreground col-span-2 flex flex-wrap items-center gap-x-4 gap-y-1 border-t pt-3 text-xs sm:col-span-4">
                        <span className="text-foreground font-medium">
                            {metadata.performance?.storage_engine ?? "Parquet"}
                        </span>
                        <span>Pages load {PAGE_SIZE} rows on demand</span>
                        <span>
                            Batch operations use {metadata.performance?.batch_rows?.toLocaleString() ?? "8,192"} rows
                        </span>
                        <span>{metadata.performance?.memory_limit ?? "Bounded memory"}</span>
                        {metadata.performance?.profile_is_sampled && (
                            <span>
                                Profile estimated from {metadata.performance.profile_rows.toLocaleString()} rows
                            </span>
                        )}
                    </div>
                </PanelBody>
            </Panel>

            <Tabs defaultValue="data">
                <TabsList>
                    <TabsTrigger value="data">Data</TabsTrigger>
                    <TabsTrigger value="profile">Profile</TabsTrigger>
                    <TabsTrigger value="configure">Configure</TabsTrigger>
                    {(selectedTask === "lexical_search" || selectedTask === "llm_evaluation") && (
                        <TabsTrigger value="evaluate">Explore &amp; evaluate</TabsTrigger>
                    )}
                </TabsList>

                <TabsContent value="data" className="space-y-3">
                    <Panel>
                        <PanelHeader
                            icon={FileSpreadsheet}
                            title="Rows"
                            description={
                                effectiveTarget
                                    ? `Edit ${effectiveTarget} inline; changes are stored as review decisions.`
                                    : "Configure a target to review labels inline."
                            }
                            actions={
                                <div className="relative">
                                    <Search className="text-muted-foreground absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2" />
                                    <Input
                                        value={query}
                                        onChange={(event) => {
                                            setQuery(event.target.value)
                                            setOffset(0)
                                        }}
                                        placeholder="Search rows"
                                        className="h-8 w-52 pl-8"
                                    />
                                </div>
                            }
                        />
                        <PanelBody className="overflow-x-auto p-0">
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead className="w-14">Row</TableHead>
                                        {visibleColumns.map((name) => (
                                            <TableHead key={name}>{name}</TableHead>
                                        ))}
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {rowsLoading ? (
                                        <TableRow>
                                            <TableCell
                                                colSpan={visibleColumns.length + 1}
                                                className="text-muted-foreground py-10 text-center"
                                            >
                                                Loading rows…
                                            </TableCell>
                                        </TableRow>
                                    ) : (
                                        (page?.rows ?? []).map((row) => (
                                            <TableRow key={row._row_id}>
                                                <TableCell className="text-muted-foreground font-mono text-xs">
                                                    {row._row_id}
                                                </TableCell>
                                                {visibleColumns.map((name) => (
                                                    <TableCell key={name} className="max-w-64 truncate">
                                                        {name === effectiveTarget ? (
                                                            <InlineReviewValue
                                                                value={row[name] == null ? "" : String(row[name])}
                                                                onSave={(value) => updateTarget(row._row_id, value)}
                                                            />
                                                        ) : row[name] == null ? (
                                                            <span className="text-muted-foreground">—</span>
                                                        ) : (
                                                            String(row[name])
                                                        )}
                                                    </TableCell>
                                                ))}
                                            </TableRow>
                                        ))
                                    )}
                                </TableBody>
                            </Table>
                        </PanelBody>
                        <div className="flex items-center justify-between border-t px-4 py-3">
                            <div className="text-muted-foreground space-y-0.5 text-xs">
                                <p>
                                    {(page?.total ?? 0).toLocaleString()} matching rows · only this page is held in
                                    memory
                                </p>
                                <p>
                                    Columns {columnOffset + 1}–
                                    {Math.min(columnOffset + COLUMN_PAGE_SIZE, columnNames.length)} of{" "}
                                    {columnNames.length}
                                </p>
                            </div>
                            <div className="flex flex-wrap justify-end gap-2">
                                <Button
                                    variant="outline"
                                    size="sm"
                                    disabled={columnOffset === 0}
                                    onClick={() => setColumnOffset(Math.max(0, columnOffset - COLUMN_PAGE_SIZE))}
                                >
                                    Previous columns
                                </Button>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    disabled={columnOffset + COLUMN_PAGE_SIZE >= columnNames.length}
                                    onClick={() => setColumnOffset(columnOffset + COLUMN_PAGE_SIZE)}
                                >
                                    Next columns
                                </Button>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    disabled={offset === 0}
                                    onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                                >
                                    Previous rows
                                </Button>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    disabled={offset + PAGE_SIZE >= (page?.total ?? 0)}
                                    onClick={() => setOffset(offset + PAGE_SIZE)}
                                >
                                    Next rows
                                </Button>
                            </div>
                        </div>
                    </Panel>
                </TabsContent>

                <TabsContent value="profile">
                    <Panel>
                        <PanelHeader
                            icon={BarChart3}
                            title="Column profile"
                            description={
                                metadata.performance?.profile_is_sampled
                                    ? `Estimated from a deterministic ${metadata.performance.profile_rows.toLocaleString()}-row sample; total row count is exact.`
                                    : "Types, missingness, cardinality and ranges were measured during import."
                            }
                        />
                        <PanelBody className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                            {columns.map((column) => (
                                <div key={column.name} className="rounded-lg border p-4">
                                    <div className="flex items-center justify-between gap-3">
                                        <p className="truncate font-medium">{column.name}</p>
                                        <span className="bg-muted rounded px-2 py-0.5 font-mono text-[0.6875rem]">
                                            {column.type}
                                        </span>
                                    </div>
                                    <dl className="text-muted-foreground mt-3 grid grid-cols-2 gap-2 text-xs">
                                        <div>
                                            <dt>{column.estimated ? "Unique in sample" : "Unique"}</dt>
                                            <dd className="text-foreground mt-0.5 font-mono">
                                                {column.unique.toLocaleString()}
                                            </dd>
                                        </div>
                                        <div>
                                            <dt>Missing</dt>
                                            <dd className="text-foreground mt-0.5 font-mono">
                                                {column.missing_percent}%
                                            </dd>
                                        </div>
                                        {column.mean !== undefined && (
                                            <div>
                                                <dt>Mean</dt>
                                                <dd className="text-foreground mt-0.5 font-mono">{column.mean}</dd>
                                            </div>
                                        )}
                                        {column.minimum !== undefined && (
                                            <div>
                                                <dt>Range</dt>
                                                <dd className="text-foreground mt-0.5 truncate font-mono">
                                                    {column.minimum}–{column.maximum}
                                                </dd>
                                            </div>
                                        )}
                                    </dl>
                                    <p className="text-muted-foreground mt-3 truncate text-xs">
                                        {column.examples.map(String).join(" · ") || "No non-empty examples"}
                                    </p>
                                </div>
                            ))}
                        </PanelBody>
                    </Panel>
                </TabsContent>

                <TabsContent value="configure">
                    <Panel>
                        <PanelHeader
                            icon={BrainCircuit}
                            title="What should this project do?"
                            description="Choose column roles, evaluation and feature controls. The complete contract is saved with every run."
                        />
                        <PanelBody className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                            {workflowDisclosure && (
                                <div className="bg-muted/35 rounded-lg border p-3 md:col-span-2 xl:col-span-3">
                                    <p className="text-sm font-medium">{workflowDisclosure.title}</p>
                                    <p className="text-muted-foreground mt-1 text-xs leading-5">
                                        {workflowDisclosure.detail}
                                    </p>
                                </div>
                            )}
                            <Field label="Task">
                                <Select value={taskType || selectedTask} onValueChange={setTaskType}>
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {taskOptions.map(([value, label]) => (
                                            <SelectItem key={value} value={value}>
                                                {label}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </Field>
                            {["classification", "regression", "text_classification"].includes(activeTask) && (
                                <ColumnSelect
                                    label="Target column"
                                    value={effectiveTarget}
                                    onChange={setTarget}
                                    columns={columnNames}
                                />
                            )}
                            {["text_classification", "lexical_search"].includes(activeTask) && (
                                <ColumnSelect
                                    label="Text column"
                                    value={effectiveText}
                                    onChange={setTextColumn}
                                    columns={columnNames}
                                />
                            )}
                            {activeTask === "llm_evaluation" && (
                                <>
                                    <ColumnSelect
                                        label="Prompt column"
                                        value={effectivePrompt}
                                        onChange={setPromptColumn}
                                        columns={columnNames}
                                    />
                                    <ColumnSelect
                                        label="Response column"
                                        value={effectiveResponse}
                                        onChange={setResponseColumn}
                                        columns={columnNames}
                                    />
                                    <ColumnSelect
                                        label="Reference (optional)"
                                        value={effectiveReference}
                                        onChange={setReferenceColumn}
                                        columns={columnNames}
                                        optional
                                    />
                                </>
                            )}
                            {usesTabularFeatures && (
                                <>
                                    <ColumnSelect
                                        label="Row identifier (optional)"
                                        value={effectiveId}
                                        onChange={setIdColumn}
                                        columns={columnNames.filter(
                                            (column) => ![effectiveTarget, effectiveText].includes(column)
                                        )}
                                        optional
                                    />
                                    <Field label="Primary metric">
                                        <Select value={effectiveMetric} onValueChange={setPrimaryMetric}>
                                            <SelectTrigger>
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {metricOptions.map((metric) => (
                                                    <SelectItem key={metric} value={metric}>
                                                        {metric}
                                                    </SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                    </Field>
                                </>
                            )}
                            {(activeTask === "classification" || activeTask === "text_classification") && (
                                <Field label="Class weighting">
                                    <Select
                                        value={classBalance}
                                        onValueChange={(value: "balanced" | "natural") => setClassBalance(value)}
                                    >
                                        <SelectTrigger>
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="balanced">Balance rare classes</SelectItem>
                                            <SelectItem value="natural">Keep observed frequency</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </Field>
                            )}
                            {activeTask === "text_classification" && (
                                <Field label="Text features">
                                    <Select
                                        value={textFeatures}
                                        onValueChange={(value: "word_character" | "word" | "character") =>
                                            setTextFeatures(value)
                                        }
                                    >
                                        <SelectTrigger>
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="word_character">Words + spelling patterns</SelectItem>
                                            <SelectItem value="word">Words only</SelectItem>
                                            <SelectItem value="character">Spelling patterns only</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </Field>
                            )}
                            {activeTask === "text_classification" && (
                                <div className="bg-muted/35 rounded-lg border p-3 md:col-span-2 xl:col-span-3">
                                    <p className="text-sm font-medium">One text input</p>
                                    <p className="text-muted-foreground mt-1 text-xs leading-5">
                                        Only the selected text column becomes a model feature. Other columns remain in
                                        the dataset for provenance and review.
                                    </p>
                                </div>
                            )}
                            {usesTabularFeatures && (
                                <div className="space-y-3 rounded-lg border p-4 md:col-span-2 xl:col-span-3">
                                    <div>
                                        <p className="text-sm font-medium">Feature columns</p>
                                        <p className="text-muted-foreground mt-1 text-xs">
                                            Turn off IDs, post-outcome fields and anything unavailable when a real
                                            prediction is made.
                                        </p>
                                    </div>
                                    <div className="grid max-h-48 gap-2 overflow-y-auto sm:grid-cols-2 lg:grid-cols-3">
                                        {columnNames
                                            .filter(
                                                (column) =>
                                                    ![effectiveTarget, effectiveText, effectiveId].includes(column)
                                            )
                                            .map((column) => {
                                                const profile = columns.find((item) => item.name === column)
                                                const profileRows = profile?.profile_rows ?? metadata.source?.rows ?? 0
                                                const likelyId =
                                                    profileRows > 0 && (profile?.unique ?? 0) / profileRows >= 0.98
                                                return (
                                                    <label
                                                        key={column}
                                                        className="flex items-start gap-2 rounded-md border px-3 py-2 text-sm"
                                                    >
                                                        <Checkbox
                                                            checked={!ignoredColumns.includes(column)}
                                                            onCheckedChange={(checked) =>
                                                                setIgnoredColumns((current) =>
                                                                    checked
                                                                        ? current.filter((name) => name !== column)
                                                                        : [...new Set([...current, column])]
                                                                )
                                                            }
                                                        />
                                                        <span className="min-w-0">
                                                            <span className="block truncate">{column}</span>
                                                            {likelyId && (
                                                                <span className="text-run text-[0.6875rem]">
                                                                    Nearly unique—check for leakage
                                                                </span>
                                                            )}
                                                        </span>
                                                    </label>
                                                )
                                            })}
                                    </div>
                                </div>
                            )}
                            {isTrainable && (
                                <div className="grid gap-3 rounded-lg border p-4 md:col-span-2 md:grid-cols-4 xl:col-span-3">
                                    <div className="md:col-span-4">
                                        <p className="text-sm font-medium">Reproducible split</p>
                                        <p className="text-muted-foreground mt-1 text-xs">
                                            Training uses {trainPercent}%; the held-out test split is used only for the
                                            final report.
                                        </p>
                                    </div>
                                    <Field label="Train %">
                                        <Input value={trainPercent} disabled />
                                    </Field>
                                    <Field label="Validation %">
                                        <Input
                                            type="number"
                                            min={0}
                                            max={80}
                                            value={validationPercent}
                                            onChange={(event) => setValidationPercent(Number(event.target.value))}
                                        />
                                    </Field>
                                    <Field label="Test %">
                                        <Input
                                            type="number"
                                            min={0}
                                            max={80}
                                            value={testPercent}
                                            onChange={(event) => setTestPercent(Number(event.target.value))}
                                        />
                                    </Field>
                                    <Field label="Random seed">
                                        <Input
                                            type="number"
                                            step={1}
                                            value={splitSeed}
                                            onChange={(event) => setSplitSeed(Number(event.target.value))}
                                        />
                                    </Field>
                                </div>
                            )}
                        </PanelBody>
                        <div className="flex items-center justify-between border-t px-5 py-4">
                            <p className={splitValid ? "text-muted-foreground text-xs" : "text-fail text-xs"}>
                                {!splitValid
                                    ? "Split must leave training rows and at least one held-out partition"
                                    : isTrainable
                                      ? `${trainPercent}% train · ${validationPercent}% validation · ${testPercent}% held-out test · seed ${splitSeed}`
                                      : "Column choices are saved with this project"}
                            </p>
                            <div className="flex gap-2">
                                <Button onClick={saveConfiguration} disabled={savingConfig || !splitValid}>
                                    {metadata.configured && <Check />}
                                    {savingConfig
                                        ? "Saving…"
                                        : metadata.configured
                                          ? "Update workflow"
                                          : "Save workflow"}
                                </Button>
                                {metadata.configured &&
                                    ["classification", "regression", "text_classification"].includes(selectedTask) && (
                                        <Button variant="outline" asChild>
                                            <a href={`/projects/training?projectId=${projectId}`}>
                                                Train model <ArrowRight />
                                            </a>
                                        </Button>
                                    )}
                            </div>
                        </div>
                    </Panel>
                </TabsContent>

                <TabsContent value="evaluate">
                    {selectedTask === "lexical_search" ? (
                        <Panel>
                            <PanelHeader
                                icon={Search}
                                title="Lexical & fuzzy search"
                                description="Character n-gram hashing scans Parquet in bounded batches. It matches wording and spelling variants; it is not a neural embedding or semantic model."
                            />
                            <PanelBody className="space-y-4">
                                <div className="flex gap-2">
                                    <Input
                                        value={searchText}
                                        onChange={(event) => setSearchText(event.target.value)}
                                        placeholder="Search this dataset"
                                        onKeyDown={(event) => event.key === "Enter" && runSearch()}
                                    />
                                    <Button onClick={runSearch} disabled={!searchText}>
                                        Search
                                    </Button>
                                </div>
                                <div className="space-y-2">
                                    {searchResults.map((result) => (
                                        <div key={result.row_id} className="rounded-lg border p-3">
                                            <div className="flex justify-between gap-4">
                                                <p className="text-sm">{result.text}</p>
                                                <span className="text-mark font-mono text-xs">
                                                    {result.score.toFixed(3)}
                                                </span>
                                            </div>
                                            <p className="text-muted-foreground mt-1 font-mono text-[0.6875rem]">
                                                row {result.row_id}
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            </PanelBody>
                        </Panel>
                    ) : (
                        <Panel>
                            <PanelHeader
                                icon={Sparkles}
                                title="Response evaluation"
                                description="Measure saved model, LLM or human responses with completion, exact match and token F1—without sending data off the machine."
                                actions={<Button onClick={runEvaluation}>Run evaluation</Button>}
                            />
                            <PanelBody>
                                {evaluation ? (
                                    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                                        {Object.entries(evaluation).map(([name, value]) => (
                                            <Stat
                                                key={name}
                                                label={name.replaceAll("_", " ")}
                                                value={
                                                    typeof value === "number"
                                                        ? value.toLocaleString(undefined, { maximumFractionDigits: 3 })
                                                        : String(value)
                                                }
                                            />
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-muted-foreground text-sm">
                                        Run the evaluation to create a portable report beside the dataset.
                                    </p>
                                )}
                            </PanelBody>
                        </Panel>
                    )}
                </TabsContent>
            </Tabs>
        </div>
    )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div className="space-y-1.5">
            <Label>{label}</Label>
            {children}
        </div>
    )
}

function ColumnSelect({
    label,
    value,
    onChange,
    columns,
    optional = false,
}: {
    label: string
    value: string
    onChange: (value: string) => void
    columns: string[]
    optional?: boolean
}) {
    return (
        <Field label={label}>
            <Select
                value={value || (optional ? "__none__" : undefined)}
                onValueChange={(next) => onChange(next === "__none__" ? "" : next)}
            >
                <SelectTrigger>
                    <SelectValue placeholder="Choose a column" />
                </SelectTrigger>
                <SelectContent>
                    {optional && <SelectItem value="__none__">None</SelectItem>}
                    {columns.map((column) => (
                        <SelectItem key={column} value={column}>
                            {column}
                        </SelectItem>
                    ))}
                </SelectContent>
            </Select>
        </Field>
    )
}

function InlineReviewValue({ value, onSave }: { value: string; onSave: (value: string) => Promise<void> }) {
    const [current, setCurrent] = useState(value)
    const changed = current !== value
    return (
        <div className="flex min-w-40 items-center gap-1">
            <Input className="h-7 min-w-28" value={current} onChange={(event) => setCurrent(event.target.value)} />
            {changed && (
                <Button
                    variant="ghost"
                    size="icon"
                    className="size-7"
                    aria-label="Save reviewed value"
                    onClick={() => onSave(current)}
                >
                    <Check className="size-3.5" />
                </Button>
            )}
        </div>
    )
}

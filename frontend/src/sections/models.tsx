"use client"

import { useProjectContext } from "@/contexts/project"
import { ArrowRight, BarChart2, Boxes, ChevronLeft, ChevronRight, Download, Edit2, Play, Search } from "lucide-react"
import React, { useEffect, useState } from "react"

import TryModelDialog from "@/components/try-model"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { EmptyState } from "@/components/ui/empty-state"
import { Input } from "@/components/ui/input"
import { Panel, PanelBody, PanelFooter, PanelHeader } from "@/components/ui/panel"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { toast } from "@/components/ui/use-toast"
import { isStructuredProject } from "@/lib/project-types"
import useModels from "@/lib/use-models"
import { usePreferences } from "@/lib/use-preferences"
import { getStatusColor } from "@/lib/utils"
import { DetailedTrainingSession, Model } from "@/types"

import StructuredModels from "./structured-models"
import ViewTrainingDetails from "./view-training-details"

function Models({ projectId }: { projectId: number }) {
    const project = useProjectContext()
    if (project && isStructuredProject(project.type)) {
        return <StructuredModels projectId={projectId} project={project} />
    }
    return <ImageModels projectId={projectId} />
}

function ImageModels({ projectId }: { projectId: number }) {
    const [selectedModel, setSelectedModel] = useState<number | null>(null)
    const [editingModel, setEditingModel] = useState<number | null>(null)
    const [newModelName, setNewModelName] = useState<string>("")
    const [currentPage, setCurrentPage] = useState(1)
    const { modelsPageSize, setModelsPageSize } = usePreferences()
    const itemsPerPage = modelsPageSize
    const [searchQuery, setSearchQuery] = useState("")
    const [selectedTrainingSession, setSelectedTrainingSession] = useState<DetailedTrainingSession | null>(null)
    const [isTrainingDetailsOpen, setIsTrainingDetailsOpen] = useState(false)
    const [isTryDialogOpen, setIsTryDialogOpen] = useState(false)

    const { isLoading, models, totalCount, fetchModels, updateModel, getModelById } = useModels(projectId)
    const totalPages = Math.ceil(totalCount / itemsPerPage)

    // Fetch models when the component mounts or when the search query changes
    useEffect(() => {
        fetchModels((currentPage - 1) * itemsPerPage, itemsPerPage, searchQuery)
    }, [currentPage, itemsPerPage, fetchModels, searchQuery])

    useEffect(() => {
        // Get search param from URL
        const params = new URLSearchParams(window.location.search)
        const searchParam = params.get("search")
        if (searchParam) {
            setSearchQuery(searchParam)
        }
        setTimeout(() => {
            fetchModels((currentPage - 1) * itemsPerPage, itemsPerPage, searchParam || "")
        }, 200)
    }, [])

    const handleTry = (modelId: number) => {
        setSelectedModel(modelId)
        setIsTryDialogOpen(true)
    }

    const handleEditModelName = (modelId: number, currentName: string) => {
        setEditingModel(modelId)
        setNewModelName(currentName)
    }

    const handleSaveModelName = async (modelId: number) => {
        try {
            await updateModel(modelId, { name: newModelName })
            setEditingModel(null)
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to update model name",
                variant: "destructive",
            })
        }
    }

    const handleViewMetrics = async (modelId: number) => {
        try {
            const model = await getModelById(modelId)
            if (model && model.training_session) {
                setSelectedTrainingSession(model.training_session as DetailedTrainingSession)
                setIsTrainingDetailsOpen(true)
            }
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to fetch model metrics",
                variant: "destructive",
            })
        }
    }

    const handlePageChange = (page: number) => {
        setCurrentPage(page)
    }

    const calculateTrainingTime = (startedAt: string, endedAt: string | null) => {
        if (!endedAt) return "In progress..."

        const start = new Date(startedAt)
        const end = new Date(endedAt)
        const diff = end.getTime() - start.getTime()

        const hours = Math.floor(diff / (1000 * 60 * 60))
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60))
        const seconds = Math.floor((diff % (1000 * 60)) / 1000)

        return `${hours}h ${minutes}m ${seconds}s`
    }

    const goToModelPage = (modelName: string) => {
        // Implement navigation to model page
    }

    return (
        <Panel>
            <PanelHeader
                icon={Boxes}
                title="Models"
                description="Trained models you can try, download or hand to another tool."
                actions={
                    totalCount > 0 || searchQuery ? (
                        <div className="relative">
                            <Search className="text-muted-foreground absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2" />
                            <Input
                                placeholder="Search models"
                                value={searchQuery}
                                onChange={(e) => {
                                    setSearchQuery(e.target.value)
                                    setCurrentPage(1)
                                }}
                                className="h-8 w-56 pl-8"
                            />
                        </div>
                    ) : null
                }
            />
            <PanelBody className="p-0">
                {isLoading ? (
                    <>
                        <div className="mb-4 flex items-center gap-2">
                            <Skeleton className="h-4 w-4" />
                            <Skeleton className="h-10 w-[200px]" />
                        </div>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Model Name</TableHead>
                                    <TableHead>Model Variant</TableHead>
                                    <TableHead>Training Session</TableHead>
                                    <TableHead>Metrics</TableHead>
                                    <TableHead>Actions</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {[...Array(5)].map((_, i) => (
                                    <TableRow key={i}>
                                        <TableCell>
                                            <Skeleton className="h-6 w-[150px]" />
                                        </TableCell>
                                        <TableCell>
                                            <Skeleton className="h-6 w-[100px]" />
                                        </TableCell>
                                        <TableCell>
                                            <Skeleton className="h-6 w-[80px]" />
                                        </TableCell>
                                        <TableCell>
                                            <Skeleton className="h-6 w-[120px]" />
                                        </TableCell>
                                        <TableCell>
                                            <div className="flex space-x-2">
                                                <Skeleton className="h-8 w-[60px]" />
                                                <Skeleton className="h-8 w-8" />
                                                <Skeleton className="h-8 w-8" />
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </>
                ) : models.length === 0 && !searchQuery ? (
                    <EmptyState
                        icon={Boxes}
                        title="No models yet"
                        description="A model appears here once a training run finishes."
                        action={
                            <Button asChild>
                                <a href={`/projects/training?projectId=${projectId}`}>
                                    Go to training
                                    <ArrowRight />
                                </a>
                            </Button>
                        }
                    />
                ) : (
                    <>
                        {models.length === 0 ? (
                            <EmptyState
                                compact
                                icon={Search}
                                title="No models match that search"
                                description="Try a shorter search, or clear it to see every model."
                            />
                        ) : (
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>Model</TableHead>
                                        <TableHead>Variant</TableHead>
                                        <TableHead>Run</TableHead>
                                        <TableHead>Metrics</TableHead>
                                        <TableHead className="text-right">Actions</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {models.map((model: Model) => (
                                        <TableRow key={model.id} className="group/row">
                                            <TableCell>
                                                {editingModel === model.id ? (
                                                    <Input
                                                        value={newModelName}
                                                        onChange={(e) => setNewModelName(e.target.value)}
                                                        onKeyDown={(e) => {
                                                            if (e.key === "Enter") {
                                                                handleSaveModelName(model.id)
                                                            }
                                                        }}
                                                        onBlur={() => handleSaveModelName(model.id)}
                                                    />
                                                ) : (
                                                    <div className="flex items-center">
                                                        {model.name}
                                                        <TooltipProvider>
                                                            <Tooltip>
                                                                <TooltipTrigger asChild>
                                                                    <Button
                                                                        size="icon-xs"
                                                                        variant="ghost"
                                                                        aria-label={`Rename ${model.name}`}
                                                                        className="ml-1 opacity-0 transition-opacity group-hover/row:opacity-100 focus-visible:opacity-100"
                                                                        onClick={() =>
                                                                            handleEditModelName(model.id, model.name)
                                                                        }
                                                                    >
                                                                        <Edit2 />
                                                                    </Button>
                                                                </TooltipTrigger>
                                                                <TooltipContent>
                                                                    <p>Edit model name</p>
                                                                </TooltipContent>
                                                            </Tooltip>
                                                        </TooltipProvider>
                                                    </div>
                                                )}
                                            </TableCell>
                                            <TableCell className="text-muted-foreground">
                                                {model.model_variant}
                                            </TableCell>
                                            <TableCell className="text-muted-foreground tabular font-mono text-xs">
                                                {model.training_session_id}
                                            </TableCell>
                                            <TableCell>
                                                {model.test_result &&
                                                    Object.entries(model.test_result)
                                                        .filter(
                                                            ([key]) =>
                                                                ![
                                                                    "Epoch",
                                                                    "Validation Loss",
                                                                    "Training Loss",
                                                                    "Loss",
                                                                    "validation_loss",
                                                                    "train_loss",
                                                                    "val_loss",
                                                                    "epoch",
                                                                ].includes(key)
                                                        )
                                                        .map(([key, value]) => (
                                                            <div
                                                                key={key}
                                                                className="flex gap-2 text-xs whitespace-nowrap"
                                                            >
                                                                <span className="text-muted-foreground">{key}</span>
                                                                <span className="tabular font-mono">
                                                                    {typeof value === "number"
                                                                        ? value.toFixed(4)
                                                                        : value}
                                                                </span>
                                                            </div>
                                                        ))}
                                            </TableCell>
                                            <TableCell>
                                                <div className="flex justify-end gap-1">
                                                    <TooltipProvider>
                                                        <Tooltip>
                                                            <TooltipTrigger asChild>
                                                                <Button
                                                                    size="sm"
                                                                    variant="outline"
                                                                    onClick={() => handleTry(model.id)}
                                                                >
                                                                    <Play />
                                                                    Try
                                                                </Button>
                                                            </TooltipTrigger>
                                                            <TooltipContent>
                                                                <p>Try model inference</p>
                                                            </TooltipContent>
                                                        </Tooltip>
                                                    </TooltipProvider>
                                                    <TooltipProvider>
                                                        <Tooltip>
                                                            <TooltipTrigger asChild>
                                                                <DropdownMenu>
                                                                    <DropdownMenuTrigger asChild>
                                                                        <Button
                                                                            size="icon-sm"
                                                                            variant="ghost"
                                                                            aria-label="Download model"
                                                                        >
                                                                            <Download />
                                                                        </Button>
                                                                    </DropdownMenuTrigger>
                                                                    <DropdownMenuContent>
                                                                        <DropdownMenuItem asChild>
                                                                            <a
                                                                                href={`/api/projects/${projectId}/models/${model.id}/download?token=${window?.pywebview?.token}`}
                                                                                rel="noopener noreferrer"
                                                                            >
                                                                                Raw Model
                                                                            </a>
                                                                        </DropdownMenuItem>
                                                                        <DropdownMenuItem
                                                                            asChild
                                                                            disabled={!model.exported_path}
                                                                        >
                                                                            <a
                                                                                href={`/api/projects/${projectId}/models/${model.id}/download_exported?token=${window?.pywebview?.token}`}
                                                                                rel="noopener noreferrer"
                                                                            >
                                                                                ONNX Model
                                                                            </a>
                                                                        </DropdownMenuItem>
                                                                    </DropdownMenuContent>
                                                                </DropdownMenu>
                                                            </TooltipTrigger>
                                                            <TooltipContent>
                                                                <p>Download model</p>
                                                            </TooltipContent>
                                                        </Tooltip>
                                                    </TooltipProvider>
                                                    <TooltipProvider>
                                                        <Tooltip>
                                                            <TooltipTrigger asChild>
                                                                <Button
                                                                    size="icon-sm"
                                                                    variant="ghost"
                                                                    aria-label="View training details"
                                                                    onClick={() => handleViewMetrics(model.id)}
                                                                >
                                                                    <BarChart2 />
                                                                </Button>
                                                            </TooltipTrigger>
                                                            <TooltipContent>
                                                                <p>View training details</p>
                                                            </TooltipContent>
                                                        </Tooltip>
                                                    </TooltipProvider>
                                                </div>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        )}
                    </>
                )}
            </PanelBody>
            <PanelFooter>
                {totalCount > 0 && (
                    <div className="flex w-full items-center justify-between">
                        <div className="flex items-center gap-2 rounded-md border px-3 py-1">
                            <Button
                                onClick={() => handlePageChange(Math.max(currentPage - 1, 1))}
                                variant="ghost"
                                size="icon"
                                disabled={currentPage === 1}
                                className="h-6 w-6"
                            >
                                <ChevronLeft className="h-4 w-4" />
                            </Button>
                            <span className="tabular font-mono text-xs">
                                {currentPage} / {totalPages}
                            </span>
                            <Button
                                onClick={() => handlePageChange(Math.min(currentPage + 1, totalPages))}
                                variant="ghost"
                                size="icon"
                                disabled={currentPage === totalPages}
                                className="h-6 w-6"
                            >
                                <ChevronRight className="h-4 w-4" />
                            </Button>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="text-muted-foreground text-xs">Per page</span>
                            <Select
                                value={itemsPerPage.toString()}
                                onValueChange={(value) => {
                                    setModelsPageSize(parseInt(value))
                                    setCurrentPage(1) // Reset to first page when changing page size
                                }}
                            >
                                <SelectTrigger size="sm" className="w-[70px]">
                                    <SelectValue placeholder="5" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="5">5</SelectItem>
                                    <SelectItem value="10">10</SelectItem>
                                    <SelectItem value="20">20</SelectItem>
                                    <SelectItem value="50">50</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                    </div>
                )}
            </PanelFooter>
            <ViewTrainingDetails
                session={selectedTrainingSession}
                isOpen={isTrainingDetailsOpen}
                onOpenChange={setIsTrainingDetailsOpen}
                getStatusColor={getStatusColor}
                calculateTrainingTime={calculateTrainingTime}
                goToModelPage={goToModelPage}
            />
            <TryModelDialog
                isOpen={isTryDialogOpen}
                onOpenChange={setIsTryDialogOpen}
                selectedModel={models.find((m) => m.id === selectedModel)}
                projectId={projectId}
            />
        </Panel>
    )
}

export default Models

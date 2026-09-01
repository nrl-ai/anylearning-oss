"use client"

import { useProjectContext } from "@/contexts/project"
import {
    ArrowRight,
    Check,
    ChevronLeft,
    ChevronRight,
    Download,
    FileDown,
    HelpCircle,
    ImageOff,
    Tag,
    Trash2,
    Upload,
} from "lucide-react"
import Image from "next/image"
import React, { useEffect, useRef, useState } from "react"

import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { EmptyState } from "@/components/ui/empty-state"
import { Panel, PanelBody, PanelFooter, PanelHeader, Stat } from "@/components/ui/panel"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { toast } from "@/components/ui/use-toast"
import { api } from "@/lib/api"
import { isStructuredProject } from "@/lib/project-types"
import { putClassId } from "@/lib/use-annotation"
import useDataItems from "@/lib/use-data-items"
import useDatasets from "@/lib/use-datasets"
import { usePreferences } from "@/lib/use-preferences"
import { useProjects } from "@/lib/use-projects"
import { cn } from "@/lib/utils"
import { DataItem } from "@/types"

import ClassDistribution from "./class-distribution"
import DataDistribution from "./data-distribution"
import LabelingScreen from "./labelling"
import StructuredWorkspace from "./structured-workspace"

interface DatasetProps {
    projectId: number
}

interface UploadStatus {
    status: "pending" | "processing" | "completed" | "failed"
    total_files: number
    processed_files: number
    error_message?: string
    /** Classes this upload created. Empty means the archive had no annotations. */
    created_labels?: string[]
}

interface ExportStatus {
    status: "pending" | "processing" | "completed" | "failed"
    total_files: number
    processed_files: number
    error_message?: string
    export_path?: string
    format?: string
}

export function DatasetManager({ projectId }: DatasetProps) {
    const project = useProjectContext()
    if (project && isStructuredProject(project.type)) {
        return <StructuredWorkspace projectId={projectId} project={project} />
    }
    return <ImageDatasetManager projectId={projectId} project={project} />
}

function ImageDatasetManager({ projectId, project }: DatasetProps & { project: ReturnType<typeof useProjectContext> }) {
    const [selectedSubset, setSelectedSubset] = useState<number>(0)
    const { dataItems, setDataItems, totalCount, fetchDataItems, refetch, invalidateDataItems, isLoading, isError } =
        useDataItems(projectId, selectedSubset)
    const { datasetsInfo, invalidateDatasets } = useDatasets(projectId)
    const { refetch: refetchProjects } = useProjects()
    const [currentPage, setCurrentPage] = useState<number>(1)
    const [selectedItems, setSelectedItems] = useState<Set<number>>(new Set())
    const [uploadStatus, setUploadStatus] = useState<UploadStatus | null>(null)
    const [exportStatus, setExportStatus] = useState<ExportStatus | null>(null)
    const [isUploading, setIsUploading] = useState<boolean>(false)
    // Which subset card the pointer is over, so dragging onto Training does not
    // light up Validation and Test as well. All three cards are rendered by the
    // same function, so a boolean here would be shared by all of them.
    const [dropTarget, setDropTarget] = useState<string | null>(null)
    const [isExporting, setIsExporting] = useState<boolean>(false)
    const [isLabelingOpen, setIsLabelingOpen] = useState<boolean>(false)
    const [imagesLoaded, setImagesLoaded] = useState<{ [key: number]: boolean }>({})
    const [autoCreateCategories, setAutoCreateCategories] = useState<boolean>(true)
    const { gridPageSize, setGridPageSize } = usePreferences()
    const itemsPerPage = gridPageSize
    const fileInputRef = useRef<HTMLInputElement>(null)
    const totalPages = Math.ceil(totalCount / itemsPerPage)
    const isClassification = project?.type === "Image Classification" || project?.type === "Handpose Classification"
    const [exportDialogOpen, setExportDialogOpen] = useState<boolean>(false)
    const [exportFormat, setExportFormat] = useState<string>("yolo")
    const [openExportMenu, setOpenExportMenu] = useState(false)
    const [openExportDialog, setOpenExportDialog] = useState(false)
    const [exportError, setExportError] = useState<string | null>(null)

    useEffect(() => {
        // The existing YOLO converter writes detection boxes, not YOLO Pose.
        // Defaulting a keypoint project to it would produce an archive with
        // images and empty label files while reporting a successful export.
        if (project?.type === "Keypoint Detection" && exportFormat === "yolo") {
            setExportFormat("coco")
        }
    }, [exportFormat, project?.type])

    const toggleItemSelection = (id: number) => {
        const newSelectedItems = new Set(selectedItems)
        if (newSelectedItems.has(id)) {
            newSelectedItems.delete(id)
        } else {
            newSelectedItems.add(id)
        }
        setSelectedItems(newSelectedItems)
    }

    const toggleAllSelection = () => {
        if (selectedItems.size === dataItems.length) {
            setSelectedItems(new Set())
        } else {
            setSelectedItems(new Set(dataItems.map((item) => item.id)))
        }
    }

    const deleteSelectedItems = async () => {
        try {
            await api.delete(`/api/projects/${projectId}/data_items`, {
                data: Array.from(selectedItems),
            })

            setDataItems(dataItems.filter((item) => !selectedItems.has(item.id)))
            setSelectedItems(new Set())

            // Refresh the data items after deletion
            invalidateDataItems()
            const offset = (currentPage - 1) * itemsPerPage
            await fetchDataItems(offset, itemsPerPage)
            toast({
                title: "Items deleted",
                description: "Selected items have been successfully deleted.",
            })
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to delete items. Please try again.",
                variant: "destructive",
            })
        }
    }

    const uploadFiles = async (files: File[]) => {
        if (files.length === 0 || isUploading) return

        const formData = new FormData()
        // Repeated under one name: the endpoint takes a list, so dropping forty
        // images is one request rather than forty.
        files.forEach((file) => formData.append("file", file))

        setIsUploading(true)
        try {
            await api.post(`/api/projects/${projectId}/upload_data`, formData, {
                params: {
                    subset: selectedSubset,
                    auto_create_categories: autoCreateCategories,
                },
            })
            toast({
                title: "Upload started",
                description: "Your file upload has begun.",
            })
            setUploadStatus({
                status: "pending",
                total_files: 0,
                processed_files: 0,
            })
        } catch (error) {
            toast({
                title: "Upload failed",
                // The server says which file it could not take; repeating
                // "there was an error" instead would hide the one useful fact.
                description:
                    (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
                    "There was an error uploading your files. Please try again.",
                variant: "destructive",
            })
            setUploadStatus({
                status: "failed",
                total_files: 0,
                processed_files: 0,
                error_message: "Upload failed",
            })
            setIsUploading(false)
        }
    }

    useEffect(() => {
        let intervalId: NodeJS.Timeout

        const checkUploadStatus = async () => {
            if (uploadStatus && uploadStatus.status !== "completed" && uploadStatus.status !== "failed") {
                try {
                    const response = await api.get(`/api/projects/${projectId}/upload_status`)
                    setUploadStatus(response.data)
                    if (response.data.status === "completed" || response.data.status === "failed") {
                        setIsUploading(false)
                        clearInterval(intervalId)
                        if (response.data.status === "completed") {
                            invalidateDataItems() // Invalidate the data items after successful upload
                            refetch() // Refetch data items after successful upload
                            refetchProjects()
                            toast({
                                title: "Upload completed",
                                description: "Your file has been successfully uploaded and processed.",
                            })
                            invalidateDatasets()
                        }
                    }
                } catch (error) {
                    toast({
                        title: "Error",
                        description: "Failed to get upload status. Please check your connection.",
                        variant: "destructive",
                    })
                    setUploadStatus({
                        status: "failed",
                        total_files: 0,
                        processed_files: 0,
                        error_message: "Failed to get status",
                    })
                    refetchProjects()
                    setIsUploading(false)
                    clearInterval(intervalId)
                }
            }
        }

        if (isUploading) {
            intervalId = setInterval(checkUploadStatus, 1000)
        }

        return () => clearInterval(intervalId)
    }, [uploadStatus, projectId, isUploading, invalidateDataItems, fetchDataItems])

    useEffect(() => {
        const offset = (currentPage - 1) * itemsPerPage
        fetchDataItems(offset, itemsPerPage)
    }, [fetchDataItems, currentPage, itemsPerPage])

    // Add a function to clean up previous exports
    const cleanupPreviousExport = async () => {
        try {
            if (exportStatus?.status === "completed") {
                await api.delete(`/api/projects/${projectId}/cleanup_export`)
                console.log("Previous export cleaned up")
            }
        } catch (error) {
            console.error("Error cleaning up previous export:", error)
        }
    }

    // Start dataset export
    const handleExport = async () => {
        try {
            // Clean up previous export before starting a new one
            await cleanupPreviousExport()

            // If there's a completed export, alert the user that it will be discarded
            if (exportStatus?.status === "completed") {
                toast({
                    title: "Previous export discarded",
                    description: "Starting a new export will replace your previous export file.",
                    variant: "default",
                })
            }

            setIsExporting(true)
            setExportError(null)

            // Start the export process
            const response = await api.post(`/api/projects/${projectId}/export_data`, {
                format: exportFormat,
                subset: null, // Export all subsets
            })

            toast({
                title: "Export started",
                description: `Your dataset is being exported in ${exportFormat.toUpperCase()} format. You can track progress in the export dialog.`,
            })

            // Start checking export status
            checkExportStatus()
        } catch (error) {
            console.error("Error starting export:", error)
            toast({
                title: "Export Error",
                description: "Failed to start export process. Please try again.",
                variant: "destructive",
            })
            setIsExporting(false)
        }
    }

    // Download exported dataset
    const downloadExport = () => {
        if (exportStatus?.status !== "completed") return

        window.open(`/api/projects/${projectId}/download_export`, "_blank")
    }

    // Check export status
    const checkExportStatus = async () => {
        try {
            const response = await api.get(`/api/projects/${projectId}/export_status`)

            const status = response.data
            setExportStatus(status)

            if (status.status === "completed") {
                toast({
                    title: "Export Completed",
                    description: "Your dataset export is now ready for download.",
                })
                setIsExporting(false)
            } else if (status.status === "failed") {
                toast({
                    title: "Export Failed",
                    description: status.error_message || "An error occurred during export.",
                    variant: "destructive",
                })
                setIsExporting(false)
            } else {
                // Continue checking if still processing
                setTimeout(checkExportStatus, 2000)
            }
        } catch (error) {
            console.error("Error checking export status:", error)
            setIsExporting(false)
        }
    }

    const renderUploadCard = (subset: string) => (
        <Panel>
            <PanelHeader
                title={`${subset} set`}
                actions={
                    <Popover>
                        <PopoverTrigger asChild>
                            <Button variant="ghost" size="icon-xs" aria-label={`What the ${subset} set is for`}>
                                <HelpCircle />
                            </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-72 text-xs">
                            {subset === "Training" ? (
                                <p>The model learns from these images. Usually 60–80% of your data.</p>
                            ) : subset === "Validation" ? (
                                <p>Used during training to tune the model and catch overfitting. Usually 10–20%.</p>
                            ) : (
                                <p>
                                    Held back to measure final performance on images the model never saw. Usually
                                    10–20%.
                                </p>
                            )}
                        </PopoverContent>
                    </Popover>
                }
            />
            <PanelBody className="space-y-3 p-3">
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={(event) => {
                        uploadFiles(Array.from(event.target.files ?? []))
                        // Cleared so choosing the same file twice in a row
                        // still fires a change event.
                        event.target.value = ""
                    }}
                    accept=".zip,.json,image/*"
                    multiple
                    style={{ display: "none" }}
                    disabled={isUploading}
                />
                <div
                    onDragOver={(event) => {
                        event.preventDefault()
                        if (!isUploading) setDropTarget(subset)
                    }}
                    onDragLeave={() => setDropTarget(null)}
                    onDrop={(event) => {
                        event.preventDefault()
                        setDropTarget(null)
                        uploadFiles(Array.from(event.dataTransfer.files ?? []))
                    }}
                    className={cn(
                        "rounded-md border border-dashed p-3 text-center transition-colors",
                        dropTarget === subset && !isUploading ? "border-primary bg-primary/5" : "border-border"
                    )}
                >
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isUploading}
                        className="w-full"
                    >
                        <Upload /> Choose images or a .zip
                    </Button>
                    <p className="text-muted-foreground mt-2 text-xs">or drop them here</p>
                </div>
                <div className="flex items-center gap-2">
                    <Checkbox
                        id={`auto-create-categories-${subset}`}
                        checked={autoCreateCategories}
                        onCheckedChange={(checked) => setAutoCreateCategories(checked as boolean)}
                    />
                    <label htmlFor={`auto-create-categories-${subset}`} className="cursor-pointer text-xs">
                        Create classes from folder names
                    </label>
                </div>
                {uploadStatus && (
                    <div className="bg-muted space-y-1.5 rounded-md p-2 text-xs">
                        <div className="flex items-center justify-between">
                            <span className="text-muted-foreground capitalize">{uploadStatus.status}</span>
                            <span className="tabular font-mono">
                                {uploadStatus.processed_files}/{uploadStatus.total_files}
                            </span>
                        </div>
                        {uploadStatus.status === "completed" && uploadStatus.created_labels && (
                            <p className="text-muted-foreground">
                                {uploadStatus.created_labels.length > 0
                                    ? `Created ${uploadStatus.created_labels.length} class${
                                          uploadStatus.created_labels.length === 1 ? "" : "es"
                                      }: ${uploadStatus.created_labels.join(", ")}`
                                    : // Said out loud, because the alternative is a user
                                      // wondering whether the app ignored their annotations.
                                      "No annotations found in the upload, so no classes were created."}
                            </p>
                        )}
                        {uploadStatus.status !== "completed" && (
                            <Progress
                                value={(uploadStatus.processed_files / uploadStatus.total_files) * 100}
                                className="h-1 w-full"
                            />
                        )}
                        {uploadStatus.error_message && <p className="text-fail">{uploadStatus.error_message}</p>}
                    </div>
                )}
                <div className="grid grid-cols-2 gap-3 border-t pt-3">
                    <Stat label="Images" value={totalCount.toLocaleString()} />
                    <Stat
                        label="Labelled"
                        value={(datasetsInfo?.[selectedSubset]?.num_labeled || 0).toLocaleString()}
                    />
                </div>
            </PanelBody>
        </Panel>
    )

    // Export dataset dialog
    const ExportDialog = () => {
        return (
            <Dialog
                open={openExportDialog}
                onOpenChange={(open) => {
                    if (!open) {
                        // Clean up when closing the dialog
                        cleanupPreviousExport()
                        setOpenExportDialog(false)
                        setExportStatus(null)
                        setExportError(null)
                    }
                }}
            >
                <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                        <DialogTitle>Export Dataset</DialogTitle>
                        <p className="text-muted-foreground text-sm">Export your dataset to use with external tools.</p>
                    </DialogHeader>

                    {isExporting || exportStatus?.status === "processing" ? (
                        <div className="py-4">
                            <div className="mb-2 text-sm">Exporting dataset...</div>
                            {exportStatus && (
                                <div className="mt-2">
                                    <Progress
                                        value={
                                            exportStatus.total_files > 0
                                                ? (exportStatus.processed_files / exportStatus.total_files) * 100
                                                : 0
                                        }
                                    />
                                    <div className="text-muted-foreground mt-1 text-xs">
                                        {exportStatus.processed_files} / {exportStatus.total_files} files processed
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : exportStatus?.status === "completed" ? (
                        <div className="py-4">
                            <div className="text-ok mb-4 text-sm">Export ready.</div>
                            <Button variant="outline" className="w-full" onClick={downloadExport}>
                                <Download />
                                Download export
                            </Button>
                        </div>
                    ) : exportStatus?.status === "failed" ? (
                        <div className="py-4">
                            <div className="mb-4 space-y-2 text-sm">
                                <div className="text-fail font-medium">Export failed</div>
                                <div className="bg-surface-sunken max-h-[150px] overflow-auto rounded-md border p-2 font-mono text-xs">
                                    {exportStatus.error_message || "No error message was reported."}
                                </div>
                                {exportStatus.error_message?.includes("string indices") && (
                                    <p className="border-warn-border bg-warn-surface text-warn rounded-md border p-2 text-xs">
                                        Some annotations are in a format {exportFormat} export can't read. Check the
                                        shapes on a few images, then export again.
                                    </p>
                                )}
                            </div>
                            <Button
                                variant="outline"
                                className="w-full"
                                onClick={() => {
                                    cleanupPreviousExport()
                                    setExportStatus(null)
                                    setOpenExportDialog(false)
                                }}
                            >
                                Close
                            </Button>
                            <Button
                                variant="outline"
                                className="mt-2 w-full"
                                onClick={() => {
                                    setExportStatus(null)
                                    handleExport()
                                }}
                            >
                                Try Again
                            </Button>
                        </div>
                    ) : (
                        <>
                            <div className="py-4">
                                <div className="mb-4">
                                    <p className="text-muted-foreground text-sm">
                                        You are about to export your dataset in {exportFormat.toUpperCase()} format.
                                        {exportFormat === "yolo" &&
                                            " This will create a ZIP file containing all your images and annotations in a format compatible with YOLO training."}
                                        {exportFormat === "coco" &&
                                            " This will create a ZIP file containing all your images and a single JSON file with annotations in COCO format."}
                                        {exportFormat === "labelme" &&
                                            " This will create a ZIP file containing all your images and JSON files with annotations in LabelMe format."}
                                        {exportFormat === "anylabeling" &&
                                            " This will create a ZIP file containing all your images and JSON files with annotations in AnyLabeling format."}
                                    </p>
                                </div>

                                <div className="flex items-center space-x-2">
                                    <div className="grid flex-1 gap-2">
                                        <Select value={exportFormat} onValueChange={setExportFormat}>
                                            <SelectTrigger>
                                                <SelectValue placeholder="Select format" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {project?.type !== "Keypoint Detection" && (
                                                    <SelectItem value="yolo">YOLO Format</SelectItem>
                                                )}
                                                <SelectItem value="coco">COCO Format</SelectItem>
                                                <SelectItem value="labelme">LabelMe Format</SelectItem>
                                                <SelectItem value="anylabeling">AnyLabeling Format</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                </div>
                            </div>

                            <DialogFooter>
                                <Button onClick={handleExport}>Start Export</Button>
                            </DialogFooter>
                        </>
                    )}
                </DialogContent>
            </Dialog>
        )
    }

    return (
        <div className="flex h-full min-h-0 flex-col">
            <div className="flex min-h-0 flex-1 gap-4">
                <div className="flex w-[270px] flex-shrink-0 flex-col gap-3 overflow-y-auto">
                    <Tabs
                        defaultValue="Training"
                        onValueChange={(value: string) => {
                            const subset = value === "Training" ? 0 : value === "Validation" ? 1 : 2
                            setSelectedSubset(subset)
                            setCurrentPage(1)
                            setSelectedItems(new Set())
                            setUploadStatus(null)
                            setIsUploading(false)
                            fetchDataItems(0, itemsPerPage)
                        }}
                    >
                        <TabsList className="grid w-full grid-cols-3">
                            <TabsTrigger value="Training" className="py-1 text-sm">
                                Training
                            </TabsTrigger>
                            <TabsTrigger value="Validation" className="py-1 text-sm">
                                Validation
                            </TabsTrigger>
                            <TabsTrigger value="Test" className="py-1 text-sm">
                                Test
                            </TabsTrigger>
                        </TabsList>
                        <TabsContent value="Training">{renderUploadCard("Training")}</TabsContent>
                        <TabsContent value="Validation">{renderUploadCard("Validation")}</TabsContent>
                        <TabsContent value="Test">{renderUploadCard("Test")}</TabsContent>
                    </Tabs>
                    <DataDistribution projectId={projectId} />
                    <ClassDistribution projectId={projectId} />
                </div>
                <div className="flex min-w-0 flex-1 flex-col">
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-baseline gap-2">
                            <h2 className="t-title">
                                {selectedSubset === 0 ? "Training" : selectedSubset === 1 ? "Validation" : "Test"}
                            </h2>
                            <span className="text-muted-foreground tabular font-mono text-xs">
                                {totalCount.toLocaleString()} images
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            {selectedItems.size > 0 && (
                                <AlertDialog>
                                    <AlertDialogTrigger asChild>
                                        <Button variant="destructive" size="sm">
                                            <Trash2 />
                                            Delete {selectedItems.size} selected
                                        </Button>
                                    </AlertDialogTrigger>
                                    <AlertDialogContent>
                                        <AlertDialogHeader>
                                            <AlertDialogTitle>
                                                Delete {selectedItems.size}{" "}
                                                {selectedItems.size === 1 ? "image" : "images"}?
                                            </AlertDialogTitle>
                                            <AlertDialogDescription>
                                                The images and their annotations are removed from this machine. This
                                                can't be undone.
                                            </AlertDialogDescription>
                                        </AlertDialogHeader>
                                        <AlertDialogFooter>
                                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                                            <AlertDialogAction onClick={deleteSelectedItems}>
                                                Delete images
                                            </AlertDialogAction>
                                        </AlertDialogFooter>
                                    </AlertDialogContent>
                                </AlertDialog>
                            )}

                            <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                    <Button variant="outline" size="sm">
                                        <FileDown />
                                        Export
                                    </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent>
                                    {project?.type !== "Keypoint Detection" && (
                                        <DropdownMenuItem
                                            onClick={() => {
                                                setExportFormat("yolo")
                                                setOpenExportDialog(true)
                                            }}
                                        >
                                            YOLO format
                                        </DropdownMenuItem>
                                    )}
                                    <DropdownMenuItem
                                        onClick={() => {
                                            setExportFormat("coco")
                                            setOpenExportDialog(true)
                                        }}
                                    >
                                        COCO format
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                        onClick={() => {
                                            setExportFormat("labelme")
                                            setOpenExportDialog(true)
                                        }}
                                    >
                                        LabelMe format
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                        onClick={() => {
                                            setExportFormat("anylabeling")
                                            setOpenExportDialog(true)
                                        }}
                                    >
                                        AnyLabeling format
                                    </DropdownMenuItem>
                                </DropdownMenuContent>
                            </DropdownMenu>

                            <Button size="sm" onClick={() => setIsLabelingOpen(true)}>
                                <Tag />
                                Start labelling
                            </Button>
                        </div>
                    </div>

                    <Panel className="flex min-h-0 flex-1 flex-col">
                        <div className="min-h-0 flex-grow overflow-y-auto p-3">
                            {isLoading ? (
                                <div className="grid grid-cols-2 gap-3 pt-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
                                    {Array.from({ length: itemsPerPage }).map((_, index) => (
                                        <div key={index} className="aspect-square overflow-hidden rounded-md">
                                            <Skeleton className="h-full w-full" />
                                        </div>
                                    ))}
                                </div>
                            ) : isError ? (
                                <EmptyState
                                    icon={ImageOff}
                                    title="Couldn't load these images"
                                    description="Check that the project folder is still on disk, then try again."
                                />
                            ) : dataItems.length === 0 ? (
                                <EmptyState
                                    icon={ImageOff}
                                    title="No images in this set"
                                    description="Upload a .zip of images to get started. You can split them across training, validation and test."
                                    action={
                                        <Button size="sm" onClick={() => fileInputRef.current?.click()}>
                                            <Upload />
                                            Upload a .zip
                                        </Button>
                                    }
                                />
                            ) : (
                                <div className="grid grid-cols-2 gap-3 pt-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
                                    {dataItems.map((item: DataItem) => (
                                        <div
                                            key={item.id}
                                            className={`group bg-surface flex cursor-pointer flex-col overflow-hidden rounded-md border ${
                                                selectedItems.has(item.id) ? "ring-mark ring-2 ring-offset-1" : ""
                                            }`}
                                            onClick={() => toggleItemSelection(item.id)}
                                        >
                                            {/* The image and the text about it are separate bands.
                                                Laying the filename and the class control over the
                                                picture meant white text on whatever the photograph
                                                happened to be -- unreadable on anything pale, which
                                                is most of a segmentation or X-ray dataset. */}
                                            <div className="bg-surface-sunken relative aspect-square">
                                                <div className="absolute top-2 left-2 z-10">
                                                    <Checkbox
                                                        checked={selectedItems.has(item.id)}
                                                        className="data-[state=checked]:bg-mark data-[state=checked]:border-mark data-[state=checked]:text-mark-ink size-4 border-white/80 bg-black/30 opacity-0 transition-opacity group-hover:opacity-100 data-[state=checked]:opacity-100"
                                                    />
                                                </div>

                                                {!imagesLoaded[item.id] && (
                                                    <div className="absolute inset-0 flex items-center justify-center">
                                                        <Skeleton className="h-full w-full" />
                                                    </div>
                                                )}

                                                <Image
                                                    src={item.path + "?token=" + window?.pywebview?.token}
                                                    alt={item.original_name}
                                                    width={200}
                                                    height={200}
                                                    // contain, not cover: a cropped thumbnail hides the
                                                    // very object being labelled when it sits near an edge.
                                                    className={`h-full w-full object-contain ${
                                                        imagesLoaded[item.id] ? "" : "invisible"
                                                    }`}
                                                    onLoad={() =>
                                                        setImagesLoaded((prev) => ({
                                                            ...prev,
                                                            [item.id]: true,
                                                        }))
                                                    }
                                                />
                                            </div>

                                            <div className="space-y-1.5 border-t p-2">
                                                <p
                                                    className="text-foreground truncate text-xs font-medium"
                                                    title={item.original_name}
                                                >
                                                    {item.original_name}
                                                </p>
                                                <div className="flex items-center justify-between">
                                                    {/* What needs work gets the marker. On this screen
                                                        you are scanning for gaps, so "unlabelled" is
                                                        called out and "labelled" recedes to a quiet
                                                        check — a fully labelled set reads as calm. */}
                                                    {item.labeled ? (
                                                        <span className="text-muted-foreground flex items-center gap-1 text-[10px]">
                                                            <Check className="size-3" strokeWidth={2.5} />
                                                            Labelled
                                                        </span>
                                                    ) : (
                                                        <span className="bg-warn/25 text-warn inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-medium">
                                                            Unlabelled
                                                        </span>
                                                    )}
                                                </div>

                                                {isClassification && (
                                                    <div onClick={(e) => e.stopPropagation()}>
                                                        <Select
                                                            value={item.class_id === -1 ? "" : item.class_id.toString()}
                                                            onValueChange={async (value) => {
                                                                const classId = value === "" ? -1 : parseInt(value)
                                                                try {
                                                                    await putClassId(projectId, item.id, classId)
                                                                    refetch()
                                                                } catch (error) {
                                                                    console.error("Error updating class ID:", error)
                                                                }
                                                            }}
                                                        >
                                                            <SelectTrigger
                                                                // `truncate` on the trigger does nothing: the trigger is a
                                                                // flex row and the label lives in a nested span, so the
                                                                // text was clipped mid-word with no ellipsis
                                                                // ("PNEUMON..."). Target the span and let it shrink.
                                                                className="h-7 min-h-0 w-full min-w-0 text-xs [&>span]:min-w-0 [&>span]:truncate"
                                                                // The class colour as a dot rather than
                                                                // as the control's background: a
                                                                // translucent fill over an arbitrary
                                                                // photograph gave text no reliable
                                                                // contrast, and half the labels in a
                                                                // project are pale.
                                                                style={undefined}
                                                            >
                                                                <span className="flex min-w-0 items-center gap-1.5">
                                                                    <span
                                                                        aria-hidden
                                                                        className="size-2 shrink-0 rounded-full"
                                                                        style={{
                                                                            backgroundColor:
                                                                                item.class_id === -1
                                                                                    ? "var(--muted-foreground)"
                                                                                    : project?.labels?.find(
                                                                                          (label) =>
                                                                                              label.id === item.class_id
                                                                                      )?.color ||
                                                                                      "var(--muted-foreground)",
                                                                        }}
                                                                    />
                                                                    <SelectValue placeholder="unlabeled" />
                                                                </span>
                                                            </SelectTrigger>
                                                            <SelectContent>
                                                                <SelectItem value="" className="my-0.5 truncate">
                                                                    unlabeled
                                                                </SelectItem>
                                                                {project?.labels?.map((label) => (
                                                                    <SelectItem
                                                                        key={label.id}
                                                                        value={label.id.toString()}
                                                                        style={{
                                                                            backgroundColor: label.color || "#666",
                                                                            color: "#fff",
                                                                        }}
                                                                        className="my-0.5 truncate"
                                                                    >
                                                                        {label.name}
                                                                    </SelectItem>
                                                                ))}
                                                            </SelectContent>
                                                        </Select>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                        {dataItems.length != 0 && (
                            <PanelFooter>
                                <div className="flex w-full flex-wrap items-center justify-between gap-3">
                                    <div className="flex items-center gap-1 rounded-md border px-1.5 py-0.5">
                                        <Button
                                            onClick={() => {
                                                const newPage = Math.max(currentPage - 1, 1)
                                                setCurrentPage(newPage)
                                            }}
                                            variant="ghost"
                                            size="icon-xs"
                                            aria-label="Previous page"
                                            disabled={currentPage === 1}
                                        >
                                            <ChevronLeft />
                                        </Button>
                                        <span className="tabular font-mono text-xs">
                                            {currentPage} / {totalPages}
                                        </span>
                                        <Button
                                            onClick={() => {
                                                const newPage = Math.min(currentPage + 1, totalPages)
                                                setCurrentPage(newPage)
                                            }}
                                            variant="ghost"
                                            size="icon-xs"
                                            aria-label="Next page"
                                            disabled={currentPage === totalPages}
                                        >
                                            <ChevronRight />
                                        </Button>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-muted-foreground text-xs">Per page</span>
                                        <Select
                                            value={itemsPerPage.toString()}
                                            onValueChange={(value) => {
                                                setGridPageSize(parseInt(value))
                                                setCurrentPage(1) // Reset to first page when changing page size
                                            }}
                                        >
                                            <SelectTrigger size="sm" className="w-[72px]">
                                                <SelectValue placeholder="20" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="10">10</SelectItem>
                                                <SelectItem value="20">20</SelectItem>
                                                <SelectItem value="50">50</SelectItem>
                                                <SelectItem value="100">100</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                </div>
                            </PanelFooter>
                        )}
                    </Panel>
                </div>
            </div>

            <ExportDialog />

            {isLabelingOpen && (
                // The exit control now lives inside the labelling top bar. As a
                // floating overlay it sat on top of that bar and clipped the
                // auto-save toggle.
                <div className="bg-background fixed inset-0 z-50">
                    <LabelingScreen
                        projectId={projectId}
                        subset={selectedSubset}
                        onExit={() => {
                            setIsLabelingOpen(false)
                            refetch()
                        }}
                    />
                </div>
            )}
        </div>
    )
}

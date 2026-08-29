"use client"

import { motion } from "framer-motion"
import {
    Box,
    ChevronLeft,
    Download,
    FileQuestion,
    FolderOpen,
    Hand,
    Image as ImageIcon,
    Layers,
    Loader2,
    MapPin,
    MessageSquare,
    MoreVertical,
    Plus,
    Settings as SettingsIcon,
    Table,
    Trash2,
    Upload,
    X,
} from "lucide-react"
import Image from "next/image"
import { useRouter, useSearchParams } from "next/navigation"
import { useEffect, useRef, useState } from "react"

import { AppLogo } from "@/components/app-logo"
import ProjectCreationForm from "@/components/project-creation-form"
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
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { EmptyState } from "@/components/ui/empty-state"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useToast } from "@/components/ui/use-toast"
import { useSidebar } from "@/hooks/useSidebar"
import { api, withToken } from "@/lib/api"
import { APP_VERSION } from "@/lib/app-info"
import { DRAG_REGION } from "@/lib/desktop"
import { projectTypeLabel } from "@/lib/project-types"
import useMounted from "@/lib/use-mounted"
import { useProjects } from "@/lib/use-projects"
import { cn } from "@/lib/utils"
import { Project, ProjectCreation } from "@/types"

type SidebarProps = {
    className?: string
}

/** Compare dotted numeric versions. Returns >0 when a is newer than b. */
function compareVersions(a: string, b: string): number {
    const pa = a.split(".").map((n) => parseInt(n, 10) || 0)
    const pb = b.split(".").map((n) => parseInt(n, 10) || 0)
    for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
        const diff = (pa[i] ?? 0) - (pb[i] ?? 0)
        if (diff !== 0) return diff
    }
    return 0
}

export default function Sidebar({ className }: SidebarProps) {
    const { isMinimized, toggle, setHoverState } = useSidebar()
    const { projects, loading: isLoadingProjects, createProject, deleteProject } = useProjects()
    const mounted = useMounted()
    const searchParams = useSearchParams()
    const [projectToDelete, setProjectToDelete] = useState<Project | null>(null)
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false)
    const [confirmProjectName, setConfirmProjectName] = useState("")
    const router = useRouter()
    const [selectedProject, setSelectedProject] = useState<Project | null>(null)
    const { toast } = useToast()
    const [updateAvailable, setUpdateAvailable] = useState(false)
    const [latestVersion, setLatestVersion] = useState("")
    const [showUpdateNotification, setShowUpdateNotification] = useState(false)
    const [exportProgress, setExportProgress] = useState<{ [key: number]: number }>({})
    const [exportStatus, setExportStatus] = useState<{ [key: number]: string }>({})
    const [importProgress, setImportProgress] = useState<number>(0)
    const [importStatus, setImportStatus] = useState<string>("")
    const [importError, setImportError] = useState<string>("")
    const statusCheckInterval = useRef<{ [key: number]: NodeJS.Timeout }>({})
    const importStatusInterval = useRef<NodeJS.Timeout | undefined>(undefined)
    const fileInputRef = useRef<HTMLInputElement>(null)

    useEffect(() => {
        const checkForUpdates = async () => {
            try {
                const response = await fetch("https://anylearning-oss.nrl.ai/check-for-update.json")
                const data = await response.json()
                const latest = data.anylearning_oss?.latest_version
                if (!latest) return
                setLatestVersion(latest)
                // Only offer an *upgrade*. A plain !== also fires when the
                // published version is older than the installed one, which had
                // the app prompting users to "update" to 0.24.13 from 0.26.0.
                if (compareVersions(latest, APP_VERSION) > 0) {
                    setUpdateAvailable(true)
                    setShowUpdateNotification(true)
                }
            } catch (error) {
                console.error("Failed to check for updates:", error)
            }
        }
        checkForUpdates()
    }, [])

    useEffect(() => {
        const projectId = parseInt(searchParams.get("projectId") || "-1") || -1
        if (projectId !== -1) {
            const project = projects?.find((p) => p.id === projectId)
            setSelectedProject(project || null)
        }
    }, [searchParams, projects])

    useEffect(() => {
        const checkScreenSize = () => {
            const isSmallScreen = window.innerWidth < 768
            if (isSmallScreen && !isMinimized) {
                toggle()
            }
        }

        checkScreenSize()
    }, [])

    const addProject = async (newProject: ProjectCreation) => {
        try {
            // Open what was just created. Staying on the current project after
            // filling in a creation form reads as if nothing happened -- the new
            // project appears somewhere in a list the user then has to find.
            const project = await createProject(newProject)
            setSelectedProject(project)
            router.push(`/projects/overview?projectId=${project.id}`)
        } catch (error) {
            toast({
                title: "Error",
                description: "Error adding project",
                variant: "destructive",
            })
        }
    }

    const handleDeleteProject = (project: Project) => {
        setProjectToDelete(project)
        setIsDeleteDialogOpen(true)
        setConfirmProjectName("")
    }

    const confirmDeleteProject = async () => {
        if (projectToDelete && confirmProjectName === projectToDelete.name) {
            try {
                await deleteProject(projectToDelete.id)
                setIsDeleteDialogOpen(false)
                setProjectToDelete(null)
                setConfirmProjectName("")
                window.location.href = "/projects/overview"
            } catch (error) {
                toast({
                    title: "Error",
                    description: "Error deleting project",
                    variant: "destructive",
                })
            }
        }
    }

    const handleProjectClick = (project: Project) => {
        setSelectedProject(project)
        router.push(`/projects/overview?projectId=${project.id}`)
    }

    const startExport = async (projectId: number) => {
        try {
            await api.post(`/api/projects/${projectId}/export`)

            setExportStatus((prev) => ({ ...prev, [projectId]: "in_progress" }))
            setExportProgress((prev) => ({ ...prev, [projectId]: 0 }))

            statusCheckInterval.current[projectId] = setInterval(async () => {
                const { data: status } = await api.get(`/api/projects/${projectId}/export/status`)

                setExportProgress((prev) => ({ ...prev, [projectId]: status.progress }))

                if (status.status === "completed") {
                    clearInterval(statusCheckInterval.current[projectId])
                    setExportStatus((prev) => ({ ...prev, [projectId]: "completed" }))

                    // Trigger download
                    window.location.href = withToken(`/api/projects/${projectId}/export/download`)
                } else if (status.status === "failed") {
                    clearInterval(statusCheckInterval.current[projectId])
                    setExportStatus((prev) => ({ ...prev, [projectId]: "failed" }))
                    toast({
                        title: "Export Failed",
                        description: status.error || "Unknown error occurred",
                        variant: "destructive",
                    })
                }
            }, 1000)
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to start export",
                variant: "destructive",
            })
        }
    }

    const cancelExport = async (projectId: number) => {
        try {
            await api.post(`/api/projects/${projectId}/export/cancel`)
            clearInterval(statusCheckInterval.current[projectId])
            setExportStatus((prev) => ({ ...prev, [projectId]: "canceled" }))
            setExportProgress((prev) => ({ ...prev, [projectId]: 0 }))
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to cancel export",
                variant: "destructive",
            })
        }
    }

    const handleImport = async (file: File) => {
        const formData = new FormData()
        formData.append("import_file", file)

        try {
            const { data } = await api.post("/api/projects/import", formData)
            const importId = data.import_id

            setImportStatus("importing")
            setImportProgress(0)
            setImportError("")

            // Start polling for import status
            importStatusInterval.current = setInterval(async () => {
                try {
                    const { data: status } = await api.get(`/api/projects/import/${importId}/status`)

                    setImportProgress(status.progress)

                    if (status.status === "completed") {
                        clearInterval(importStatusInterval.current)
                        setImportStatus("completed")
                        toast({
                            title: "Success",
                            description: "Project imported successfully",
                        })
                        window.location.reload()
                    } else if (status.status === "failed") {
                        clearInterval(importStatusInterval.current)
                        setImportStatus("failed")
                        setImportError(status.error || "Unknown error occurred")
                        toast({
                            title: "Import Failed",
                            description: status.error || "Unknown error occurred",
                            variant: "destructive",
                        })
                    }
                } catch (error) {
                    clearInterval(importStatusInterval.current)
                    setImportStatus("failed")
                    setImportError("Failed to check import status")
                }
            }, 1000)
        } catch (error) {
            setImportStatus("failed")
            setImportError("Failed to start import")
            toast({
                title: "Error",
                description: "Failed to import project",
                variant: "destructive",
            })
        }
    }

    const getCategoryIcon = (category: Project["type"]) => {
        switch (category) {
            case "Image Classification":
                return <ImageIcon className="h-3.5 w-3.5" />
            case "Object Detection":
                return <Box className="h-3.5 w-3.5" />
            case "Image Segmentation":
                return <Layers className="h-3.5 w-3.5" />
            case "Sentiment Analysis":
            case "Text & LLM":
            case "Text AI & LLM Evaluation":
            case "Text AI":
                return <MessageSquare className="h-3.5 w-3.5" />
            case "Tabular AI":
                return <Table className="h-3.5 w-3.5" />
            case "Handpose Classification":
                return <Hand className="h-3.5 w-3.5" />
            case "Instance Segmentation":
                return <Layers className="h-3.5 w-3.5" />
            case "Keypoint Detection":
                return <MapPin className="h-3.5 w-3.5" />
            default:
                return <FileQuestion className="h-3.5 w-3.5" />
        }
    }

    // macOS style sidebar item
    const SidebarItem = ({
        icon,
        label,
        isActive = false,
        onClick,
        showLabel = true,
    }: {
        icon: React.ReactNode
        label: string
        isActive?: boolean
        onClick?: () => void
        showLabel?: boolean
    }) => {
        return (
            <TooltipProvider delayDuration={300}>
                <Tooltip>
                    <TooltipTrigger asChild>
                        <button
                            onClick={onClick}
                            className={cn(
                                "group flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all",
                                isActive
                                    ? "bg-primary/10 text-primary"
                                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                            )}
                        >
                            <div
                                className={cn(
                                    "flex h-7 w-7 items-center justify-center rounded-full",
                                    isActive
                                        ? "bg-primary text-primary-foreground"
                                        : "text-muted-foreground group-hover:text-foreground"
                                )}
                            >
                                {icon}
                            </div>
                            {showLabel && <span>{label}</span>}
                        </button>
                    </TooltipTrigger>
                    {!showLabel && <TooltipContent side="right">{label}</TooltipContent>}
                </Tooltip>
            </TooltipProvider>
        )
    }

    const renderProjectCard = (project: Project) => {
        const isActive = selectedProject && project.id === selectedProject.id

        return (
            <motion.div
                key={project.id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.18 }}
            >
                {/* The current project is marked the same way the current stage
                    is marked on the rail: a mark-coloured rule and label, not a
                    tinted block. */}
                <div
                    className={cn(
                        "group relative flex cursor-pointer items-center rounded-md",
                        // Collapsed, every row is a square centred on the rail's
                        // axis; expanded, it is a normal padded list row. The
                        // two used to share one asymmetric padding, which left
                        // the icons a few pixels off centre from each other and
                        // from the buttons above them.
                        isMinimized ? "h-10 justify-center" : "justify-between py-1.5 pr-1 pl-2.5",
                        isActive ? "bg-accent" : "hover:bg-accent/60"
                    )}
                    onClick={() => handleProjectClick(project)}
                    title={isMinimized ? `${project.name} — ${projectTypeLabel(project.type)}` : undefined}
                >
                    {isActive && (
                        <span aria-hidden className="bg-mark absolute top-1.5 bottom-1.5 left-0 w-0.5 rounded-full" />
                    )}
                    <div className={cn("flex min-w-0 items-center", !isMinimized && "gap-2.5")}>
                        <div
                            className={cn(
                                "flex size-5 shrink-0 items-center justify-center",
                                isActive ? "text-mark" : "text-muted-foreground group-hover:text-foreground"
                            )}
                        >
                            {getCategoryIcon(project.type)}
                        </div>
                        {!isMinimized && (
                            <div className="flex min-w-0 flex-col">
                                <span
                                    className={cn(
                                        "truncate text-xs font-medium",
                                        isActive ? "text-foreground" : "text-foreground/80"
                                    )}
                                >
                                    {project.name ? project.name.slice(0, 60) : "Untitled project"}
                                </span>
                                <span className="text-muted-foreground truncate text-[0.6875rem]">
                                    {projectTypeLabel(project.type)}
                                </span>
                            </div>
                        )}
                    </div>
                    {!isMinimized && (
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                                <Button
                                    variant="ghost"
                                    size="icon-sm"
                                    aria-label={`Options for ${project.name}`}
                                    className="opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
                                >
                                    <MoreVertical />
                                </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                                <DropdownMenuItem
                                    onClick={(e) => {
                                        e.stopPropagation()
                                        startExport(project.id)
                                    }}
                                >
                                    <Download className="mr-2 size-4" />
                                    Export project
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                    variant="destructive"
                                    onClick={(e) => {
                                        e.stopPropagation()
                                        handleDeleteProject(project)
                                    }}
                                >
                                    <Trash2 className="mr-2 size-4" />
                                    Delete project
                                </DropdownMenuItem>
                            </DropdownMenuContent>
                        </DropdownMenu>
                    )}
                </div>
            </motion.div>
        )
    }

    const handleImportClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click()
        }
    }

    return (
        <motion.aside
            initial={{ width: isMinimized ? 64 : 252 }}
            animate={{ width: isMinimized ? 64 : 252 }}
            transition={{ duration: 0.22, ease: [0.22, 0.61, 0.36, 1] }}
            // Elevation by surface step, not shadow: the sidebar is the raised
            // rail beside the workspace, so it simply sits one step lighter.
            className={cn("bg-surface relative hidden h-screen flex-none border-r md:flex md:flex-col", className)}
            onMouseEnter={() => setHoverState(true)}
            onMouseLeave={() => setHoverState(false)}
        >
            {/* Wordmark. Also the left end of the window's title bar: it
                drags the window, and on macOS it starts below the traffic
                lights, which the system draws over this corner
                (--titlebar-inset in globals.css). */}
            <div
                className={cn(
                    DRAG_REGION,
                    "flex items-center gap-2.5 px-3 pt-[calc(1rem+var(--titlebar-inset))] pb-3",
                    isMinimized && "justify-center px-2"
                )}
            >
                <AppLogo className="size-7 shrink-0" />
                {!isMinimized && (
                    <div className={cn(DRAG_REGION, "flex min-w-0 flex-col")}>
                        <span className={cn(DRAG_REGION, "t-section cursor-default truncate")}>AnyLearning</span>
                        <span
                            className={cn(
                                DRAG_REGION,
                                "text-muted-foreground cursor-default truncate text-[0.6875rem]"
                            )}
                        >
                            Local AI training
                        </span>
                    </div>
                )}
            </div>

            {/* Collapse handle. Rides down with the wordmark on macOS so the
                two stay on one line. */}
            <Button
                variant="outline"
                size="icon-xs"
                aria-label={isMinimized ? "Expand sidebar" : "Collapse sidebar"}
                className="bg-surface absolute top-[calc(1.5rem+var(--titlebar-inset))] -right-3 z-50 rounded-full p-0"
                onClick={() => toggle()}
            >
                <ChevronLeft className={cn("transition-transform", isMinimized && "rotate-180")} />
            </Button>

            {/* Projects section */}
            <div className="mt-4 flex min-h-0 flex-1 flex-col">
                <div
                    className={cn(
                        "mb-1 flex items-center",
                        isMinimized ? "justify-center px-0" : "justify-between px-3"
                    )}
                >
                    {!isMinimized && (
                        <h3 className="text-muted-foreground text-xs font-semibold tracking-wider uppercase">
                            Projects
                        </h3>
                    )}
                    <div className={cn("flex", isMinimized ? "w-full flex-col items-center gap-1" : "gap-1")}>
                        {/* Import Project Button */}
                        <TooltipProvider delayDuration={300}>
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <Button
                                        variant="ghost"
                                        size="icon-sm"
                                        aria-label="Import project"
                                        onClick={handleImportClick}
                                        disabled={importStatus === "importing"}
                                    >
                                        {importStatus === "importing" ? (
                                            <Loader2 className="animate-spin" />
                                        ) : (
                                            <Upload />
                                        )}
                                    </Button>
                                </TooltipTrigger>
                                <TooltipContent side={isMinimized ? "right" : "top"} className="z-[60]">
                                    Import project
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>

                        {/* Hidden file input for import */}
                        <Input
                            ref={fileInputRef}
                            type="file"
                            accept=".tar.gz"
                            onChange={(e) => {
                                const file = e.target.files?.[0]
                                if (file) handleImport(file)
                            }}
                            className="hidden"
                            id="import-input"
                        />

                        {/* Create Project Button */}
                        <Dialog>
                            <TooltipProvider delayDuration={300}>
                                <Tooltip>
                                    <TooltipTrigger asChild>
                                        <DialogTrigger asChild>
                                            <Button variant="ghost" size="icon-sm" aria-label="Create project">
                                                <Plus />
                                            </Button>
                                        </DialogTrigger>
                                    </TooltipTrigger>
                                    <TooltipContent side={isMinimized ? "right" : "top"} className="z-[60]">
                                        Create project
                                    </TooltipContent>
                                </Tooltip>
                            </TooltipProvider>
                            <DialogContent>
                                <DialogHeader>
                                    <DialogTitle>Create a project</DialogTitle>
                                    <DialogDescription>
                                        Name the project and pick the task you want to train for.
                                    </DialogDescription>
                                </DialogHeader>
                                <ProjectCreationForm onSubmit={addProject} />
                            </DialogContent>
                        </Dialog>
                    </div>
                </div>

                {/* Import Progress */}
                {importStatus === "importing" && !isMinimized && (
                    <div className="mb-2 px-3">
                        <div className="bg-muted rounded-md p-2">
                            <div className="flex items-center justify-between text-xs">
                                <span className="text-muted-foreground">Importing project…</span>
                                <span className="tabular font-mono">{importProgress}%</span>
                            </div>
                            <Progress value={importProgress} className="mt-1.5 h-1" />
                        </div>
                    </div>
                )}

                {/* Import Error */}
                {importStatus === "failed" && !isMinimized && (
                    <div className="mb-2 px-3">
                        <div className="border-fail-border bg-fail-surface text-fail rounded-md border p-2 text-xs">
                            <p className="font-medium">Import failed</p>
                            <p className="mt-1">{importError}</p>
                        </div>
                    </div>
                )}

                <ScrollArea className={cn("min-h-0 flex-1", isMinimized ? "px-3" : "px-2")}>
                    <div className="space-y-0.5 pb-2">
                        {/* "No projects yet" is a claim, and before the list has
                            loaded it is one this component cannot make: the
                            static export prerenders with nothing fetched, so
                            the empty state was baked into the HTML and then
                            replaced by the real projects, which React reports
                            as a hydration mismatch. Skeleton rows until we
                            actually know. */}
                        {!mounted || isLoadingProjects ? (
                            <div aria-hidden className="space-y-1.5 py-1">
                                {[0, 1, 2].map((row) => (
                                    <div key={row} className="bg-muted h-9 animate-pulse rounded-md" />
                                ))}
                            </div>
                        ) : projects && projects.length > 0 ? (
                            <>
                                {selectedProject && renderProjectCard(selectedProject)}
                                {projects?.filter &&
                                    projects
                                        ?.filter((p: any) => !selectedProject || p.id !== selectedProject.id)
                                        .map((project: any) => renderProjectCard(project))}
                            </>
                        ) : (
                            <div className={cn("rounded-md border border-dashed", isMinimized && "p-2 text-center")}>
                                {isMinimized ? (
                                    <FolderOpen className="text-muted-foreground mx-auto size-5" />
                                ) : (
                                    <EmptyState
                                        compact
                                        icon={FolderOpen}
                                        title="No projects yet"
                                        description="Create a project to start labelling images and training a model."
                                        action={
                                            <>
                                                <Button variant="outline" size="sm" onClick={handleImportClick}>
                                                    <Upload />
                                                    Import
                                                </Button>
                                                <Dialog>
                                                    <DialogTrigger asChild>
                                                        <Button size="sm">
                                                            <Plus />
                                                            Create
                                                        </Button>
                                                    </DialogTrigger>
                                                    <DialogContent>
                                                        <DialogHeader>
                                                            <DialogTitle>Create a project</DialogTitle>
                                                            <DialogDescription>
                                                                Name the project and pick the task you want to train
                                                                for.
                                                            </DialogDescription>
                                                        </DialogHeader>
                                                        <ProjectCreationForm onSubmit={addProject} />
                                                    </DialogContent>
                                                </Dialog>
                                            </>
                                        }
                                    />
                                )}
                            </div>
                        )}
                    </div>
                </ScrollArea>
            </div>

            {/* Footer: update offer in flow rather than as an overlay, so it can
                never sit on top of the last project in the list. */}
            <div className="mt-auto border-t">
                {showUpdateNotification && updateAvailable && !isMinimized && (
                    <div className="space-y-2 border-b p-3">
                        <div className="flex items-start justify-between gap-2">
                            <p className="text-sm leading-snug">Version {latestVersion} is available</p>
                            <Button
                                variant="ghost"
                                size="icon-xs"
                                aria-label="Dismiss update notice"
                                onClick={() => setShowUpdateNotification(false)}
                                className="shrink-0"
                            >
                                <X />
                            </Button>
                        </div>
                        <Button
                            size="sm"
                            onClick={() =>
                                window.open("https://anylearning-oss.nrl.ai/download", "_blank")
                            }
                            className="w-full"
                        >
                            Download update
                        </Button>
                    </div>
                )}
                <div
                    className={cn(
                        "flex items-center gap-1 px-2 py-1.5",
                        isMinimized ? "justify-center" : "justify-between"
                    )}
                >
                    {!isMinimized && (
                        <span className="text-muted-foreground tabular px-1 font-mono text-[0.6875rem]">
                            v{APP_VERSION}
                        </span>
                    )}
                    <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label="Settings"
                        title="Settings"
                        onClick={() => router.push("/settings")}
                    >
                        <SettingsIcon />
                    </Button>
                </div>
            </div>

            {/* Delete project dialog */}
            <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Delete {projectToDelete?.name || "project"}?</DialogTitle>
                        <DialogDescription>
                            This deletes the project's images, labels, training runs and models from this machine. It
                            can't be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="grid gap-4 py-4">
                        <div className="grid gap-2">
                            <p className="text-sm">
                                Type <span className="font-mono text-xs">{projectToDelete?.name}</span> to confirm.
                            </p>
                            <Input
                                value={confirmProjectName}
                                onChange={(e) => setConfirmProjectName(e.target.value)}
                                placeholder="Type project name to confirm"
                            />
                        </div>
                    </div>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setIsDeleteDialogOpen(false)}>
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            onClick={confirmDeleteProject}
                            disabled={confirmProjectName !== projectToDelete?.name}
                        >
                            Delete
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </motion.aside>
    )
}

"use client"

import { motion } from "framer-motion"
import { FolderOpen, Loader2, MenuIcon, MoreVertical, Plus, Upload } from "lucide-react"
import { useRouter } from "next/navigation"
import { useRef, useState } from "react"

import { AppLogo } from "@/components/app-logo"
import ProjectCreationForm from "@/components/project-creation-form"
import { Button } from "@/components/ui/button"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useToast } from "@/components/ui/use-toast"
import { api } from "@/lib/api"
import { projectTypeLabel } from "@/lib/project-types"
import { useProjects } from "@/lib/use-projects"
import { cn } from "@/lib/utils"
import { Project, ProjectCreation } from "@/types"

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> {}

export function MobileSidebar({ className: _className }: SidebarProps) {
    const router = useRouter()
    const [open, setOpen] = useState(false)
    const [selectedProject, setSelectedProject] = useState<Project | null>(null)
    const { projects, createProject, deleteProject } = useProjects()
    const [importProgress, setImportProgress] = useState<number>(0)
    const [importStatus, setImportStatus] = useState<string>("")
    const [importError, setImportError] = useState<string>("")
    const importStatusInterval = useRef<ReturnType<typeof setInterval> | undefined>(undefined)
    const fileInputRef = useRef<HTMLInputElement>(null)
    const { toast } = useToast()

    const handleProjectClick = (project: Project) => {
        setSelectedProject(project)
        setOpen(false)
        router.push(`/projects/overview?projectId=${project.id}`)
    }

    const handleDeleteProject = async (project: Project) => {
        await deleteProject(project.id)
    }

    const addProject = async (newProject: ProjectCreation) => {
        try {
            // Same as the desktop sidebar: open the project that was just
            // created rather than leaving the user where they were.
            const project = await createProject(newProject)
            setOpen(false)
            router.push(`/projects/overview?projectId=${project.id}`)
        } catch (error) {
            console.error("Error adding project:", error)
        }
    }

    const getCategoryIcon = (category: Project["type"]) => {
        switch (category) {
            case "Image Classification":
                return <div className="h-4 w-4" />
            case "Object Detection":
                return <div className="h-4 w-4" />
            case "Image Segmentation":
                return <div className="h-4 w-4" />
            case "Sentiment Analysis":
            case "Text & LLM":
            case "Text AI & LLM Evaluation":
            case "Text AI":
                return <div className="h-4 w-4" />
            case "Tabular AI":
                return <div className="h-4 w-4" />
            case "Handpose Classification":
                return <div className="h-4 w-4" />
            case "Instance Segmentation":
                return <div className="h-4 w-4" />
            case "Keypoint Detection":
                return <div className="h-4 w-4" />
            default:
                return <div className="h-4 w-4" />
        }
    }

    const renderProjectCard = (project: Project) => {
        const isActive = selectedProject && project.id === selectedProject.id

        return (
            <motion.div
                key={project.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2 }}
            >
                <div
                    className={cn(
                        "group mb-1 flex cursor-pointer items-center justify-between rounded-lg px-3 py-2",
                        isActive
                            ? "bg-primary/10 text-primary"
                            : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                    )}
                    onClick={() => handleProjectClick(project)}
                >
                    <div className="flex items-center gap-3">
                        <div
                            className={cn(
                                "flex h-8 w-8 items-center justify-center rounded-full",
                                isActive
                                    ? "bg-primary text-primary-foreground"
                                    : "text-muted-foreground group-hover:text-foreground"
                            )}
                        >
                            {getCategoryIcon(project.type)}
                        </div>
                        <div className="flex flex-col">
                            <span className="truncate text-sm font-medium">
                                {project.name ? project.name.slice(0, 80) : "Untitled Project"}
                            </span>
                            <span className="text-muted-foreground truncate text-xs">
                                {projectTypeLabel(project.type)}
                            </span>
                        </div>
                    </div>
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8 opacity-0 group-hover:opacity-100">
                                <MoreVertical className="h-4 w-4" />
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                            <DropdownMenuItem
                                onClick={(e) => {
                                    e.stopPropagation()
                                    handleDeleteProject(project)
                                }}
                            >
                                Delete
                            </DropdownMenuItem>
                        </DropdownMenuContent>
                    </DropdownMenu>
                </div>
            </motion.div>
        )
    }

    const handleImportClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click()
        }
    }

    const handleImport = async (file: File) => {
        try {
            setImportStatus("importing")
            setImportProgress(0)
            setImportError("")

            const formData = new FormData()
            formData.append("file", file)

            await api.post("/api/projects/import", formData)

            importStatusInterval.current = setInterval(async () => {
                const { data: status } = await api.get("/api/projects/import/status")

                setImportProgress(status.progress)

                if (status.status === "completed") {
                    clearInterval(importStatusInterval.current)
                    setImportStatus("completed")
                    toast({
                        title: "Success",
                        description: "Project imported successfully",
                    })
                    setTimeout(() => {
                        setImportStatus("")
                        window.location.reload()
                    }, 1000)
                } else if (status.status === "failed") {
                    clearInterval(importStatusInterval.current)
                    setImportStatus("failed")
                    setImportError(status.error || "Unknown error")
                }
            }, 1000)
        } catch (error) {
            setImportStatus("failed")
            setImportError(error instanceof Error ? error.message : "Unknown error")
            console.error("Import error:", error)
        }
    }

    return (
        <>
            <Sheet open={open} onOpenChange={setOpen}>
                <SheetTrigger asChild className="focus:outline-none">
                    <Button variant="ghost" size="icon" className="h-10 w-10 rounded-full">
                        <MenuIcon className="h-5 w-5" />
                    </Button>
                </SheetTrigger>
                <SheetContent
                    side="left"
                    className="bg-background/80 w-[280px] border-r p-0 backdrop-blur-md"
                    style={{
                        boxShadow: "0 0 15px rgba(0, 0, 0, 0.05)",
                    }}
                >
                    {/* Logo section */}
                    <div className="p-4 pt-6">
                        <div className="flex items-center">
                            <AppLogo className="mr-2 size-7" />
                            <div className="ml-1 flex flex-col">
                                <span className="text-foreground w-[150px] cursor-default text-sm font-semibold">
                                    AnyLearning
                                </span>
                                <span className="text-muted-foreground w-[150px] cursor-default text-xs">
                                    Local AI training
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Projects section */}
                    <div className="mt-6 px-3">
                        <div className="mb-2 flex items-center justify-between px-1">
                            <h3 className="text-muted-foreground text-xs font-semibold tracking-wider uppercase">
                                Projects
                            </h3>
                            <div className="mx-4 flex space-x-1">
                                {/* Import Project Button */}
                                <TooltipProvider delayDuration={300}>
                                    <Tooltip>
                                        <TooltipTrigger asChild>
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-7 w-7 rounded-full"
                                                onClick={handleImportClick}
                                                disabled={importStatus === "importing"}
                                            >
                                                {importStatus === "importing" ? (
                                                    <Loader2 className="h-4 w-4 animate-spin" />
                                                ) : (
                                                    <Upload className="h-4 w-4" />
                                                )}
                                            </Button>
                                        </TooltipTrigger>
                                        <TooltipContent side="top" className="z-[60]">
                                            Import Project
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
                                    id="mobile-import-input"
                                />

                                {/* Create Project Button */}
                                <Dialog>
                                    <TooltipProvider delayDuration={300}>
                                        <Tooltip>
                                            <TooltipTrigger asChild>
                                                <DialogTrigger asChild>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="h-7 w-7 rounded-full"
                                                    >
                                                        <Plus className="h-4 w-4" />
                                                    </Button>
                                                </DialogTrigger>
                                            </TooltipTrigger>
                                            <TooltipContent side="top" className="z-[60]">
                                                Create Project
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                    <DialogContent>
                                        <DialogHeader>
                                            <DialogTitle>Create New Project</DialogTitle>
                                            <DialogDescription>
                                                Fill in the details to create a new project.
                                            </DialogDescription>
                                        </DialogHeader>
                                        <ProjectCreationForm onSubmit={addProject} />
                                    </DialogContent>
                                </Dialog>
                            </div>
                        </div>

                        {/* Import Progress */}
                        {importStatus === "importing" && (
                            <div className="mb-2 px-1">
                                <div className="bg-primary/5 rounded-md p-2">
                                    <div className="flex items-center justify-between text-xs">
                                        <span className="text-muted-foreground">Importing project...</span>
                                        <span className="font-medium">{importProgress}%</span>
                                    </div>
                                    <Progress value={importProgress} className="mt-1 h-1" />
                                </div>
                            </div>
                        )}

                        {/* Import Error */}
                        {importStatus === "failed" && (
                            <div className="mb-2 px-1">
                                <div className="bg-destructive/10 text-destructive rounded-md p-2 text-xs">
                                    <p className="font-medium">Import failed</p>
                                    <p className="mt-1">{importError}</p>
                                </div>
                            </div>
                        )}

                        <div className="space-y-1 py-2">
                            {projects && projects.length > 0 ? (
                                <>
                                    {selectedProject && renderProjectCard(selectedProject)}
                                    {projects
                                        .filter((project) => !selectedProject || project.id !== selectedProject.id)
                                        .map((project) => renderProjectCard(project))}
                                </>
                            ) : (
                                <div className="rounded-lg border border-dashed p-6 text-center">
                                    <FolderOpen className="text-muted-foreground mx-auto h-8 w-8" />
                                    <h3 className="mt-2 text-sm font-medium">No projects</h3>
                                    <p className="text-muted-foreground mt-1 text-xs">
                                        Get started by creating a new project or importing an existing one.
                                    </p>
                                    <div className="mt-4 flex justify-center space-x-2">
                                        <Button variant="outline" size="sm" onClick={handleImportClick}>
                                            <Upload className="mr-2 h-4 w-4" />
                                            Import
                                        </Button>
                                        <Dialog>
                                            <DialogTrigger asChild>
                                                <Button variant="default" size="sm">
                                                    <Plus className="mr-2 h-4 w-4" />
                                                    Create
                                                </Button>
                                            </DialogTrigger>
                                            <DialogContent>
                                                <DialogHeader>
                                                    <DialogTitle>Create New Project</DialogTitle>
                                                    <DialogDescription>
                                                        Fill in the details to create a new project.
                                                    </DialogDescription>
                                                </DialogHeader>
                                                <ProjectCreationForm onSubmit={addProject} />
                                            </DialogContent>
                                        </Dialog>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </SheetContent>
            </Sheet>
        </>
    )
}

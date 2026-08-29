"use client"

import { ProjectContext } from "@/contexts/project"
import { FolderOpen, Loader2 } from "lucide-react"
import { useSearchParams } from "next/navigation"
import React, { Suspense } from "react"

import PageContainer from "@/components/layout/page-container"
import { EmptyState } from "@/components/ui/empty-state"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/use-toast"
import useProject from "@/lib/use-project"
import { useProjects } from "@/lib/use-projects"
import { DatasetManager } from "@/sections/dataset"
import Models from "@/sections/models"
import ProjectOverview from "@/sections/overview"
import Training from "@/sections/training"

function ProjectContent({ tab }: { tab: string }) {
    const searchParams = useSearchParams()
    const projectId = Number(searchParams.get("projectId")) || null

    const { project, loading, error, update, refetch } = useProject(projectId)
    const { mutate: mutateProjects } = useProjects()

    const handleSaveDescription = async (description: string) => {
        if (!project) return

        try {
            await update({ description })
            mutateProjects()
            refetch()
        } catch (error) {
            toast({
                title: "Couldn't save the description",
                description: "The change wasn't stored. Try again.",
                variant: "destructive",
            })
        }
    }

    const handleSaveName = async (name: string) => {
        if (!project) return

        try {
            await update({ name })
            mutateProjects()
            refetch()
        } catch (error) {
            toast({
                title: "Couldn't save the name",
                description: "The change wasn't stored. Try again.",
                variant: "destructive",
            })
        }
    }

    if (!projectId) {
        return (
            <div className="flex min-h-0 flex-1 items-center justify-center p-6">
                <EmptyState
                    icon={FolderOpen}
                    title="No project selected"
                    description="Pick a project from the sidebar, or create one to start labelling and training."
                />
            </div>
        )
    }

    if (loading) {
        return (
            <div className="flex min-h-0 flex-1 items-center justify-center">
                <div className="text-muted-foreground flex items-center gap-2 text-sm">
                    <Loader2 className="size-4 animate-spin" />
                    <span>Loading project…</span>
                </div>
            </div>
        )
    }

    if (error) {
        return (
            <PageContainer>
                <Panel className="border-fail-border bg-fail-surface text-fail p-4 text-sm">
                    Unable to load this project: {error instanceof Error ? error.message : String(error)}
                </Panel>
            </PageContainer>
        )
    }

    if (!project) {
        return null
    }

    // The stage rail in the workbench bar is the navigation for these; this
    // view only renders the stage the route asks for.
    return (
        <ProjectContext.Provider value={project}>
            <PageContainer scrollable={tab !== "dataset"}>
                {tab === "overview" && (
                    <ProjectOverview
                        project={project}
                        onSaveName={handleSaveName}
                        onSaveDescription={handleSaveDescription}
                    />
                )}
                {tab === "dataset" && <DatasetManager projectId={projectId} />}
                {tab === "training" && <Training projectId={projectId} />}
                {tab === "models" && <Models projectId={projectId} />}
            </PageContainer>
        </ProjectContext.Provider>
    )
}

export default function ProjectView({ tab }: { tab: string }) {
    return (
        <Suspense fallback={<ProjectSkeleton />}>
            <ProjectContent tab={tab} />
        </Suspense>
    )
}

function ProjectSkeleton() {
    return (
        <div className="mx-auto w-full max-w-[1600px] space-y-4 p-4 sm:p-5 md:px-6 md:py-5" aria-busy="true">
            <div className="grid gap-4 lg:grid-cols-3">
                <Skeleton className="h-64 rounded-lg lg:col-span-2" />
                <Skeleton className="h-64 rounded-lg" />
            </div>
        </div>
    )
}

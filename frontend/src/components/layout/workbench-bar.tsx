"use client"

import { usePathname, useRouter, useSearchParams } from "next/navigation"

import ThemeToggle from "@/components/layout/ThemeToggle/theme-toggle"
import { MobileSidebar } from "@/components/layout/mobile-sidebar"
import { StageId, StageRail, useStages } from "@/components/layout/stage-rail"
import { DRAG_REGION } from "@/lib/desktop"
import { projectTypeLabel } from "@/lib/project-types"
import useMounted from "@/lib/use-mounted"
import useProject from "@/lib/use-project"
import { cn } from "@/lib/utils"

const STAGES: StageId[] = ["overview", "dataset", "training", "models"]

/**
 * The one bar at the top of the workspace.
 *
 * It replaces an empty 48px header plus a separate floating stepper: the two
 * together spent ~110px of vertical space and told the user nothing. Now the
 * strip carries who you are working on (project + task), where the project
 * actually stands (the stage rail), and where the data lives.
 */
export default function WorkbenchBar() {
    const router = useRouter()
    const pathname = usePathname()
    const searchParams = useSearchParams()
    const projectId = Number(searchParams.get("projectId")) || null
    const { project } = useProject(projectId)
    const mounted = useMounted()
    const stages = useStages(projectId, project)

    const current = (STAGES.find((stage) => pathname?.includes(`/projects/${stage}`)) ?? "overview") as StageId

    const goToStage = (stage: StageId) => {
        router.push(`/projects/${stage}?projectId=${projectId}`)
    }

    return (
        // Also the window's title bar: the desktop window is frameless, so
        // this strip is what the user drags, and it stops short of wherever
        // the platform put the window controls (globals.css). DRAG_REGION
        // marks a surface that moves the window; only a press landing
        // directly on one counts, which is what keeps the controls inside the
        // bar clickable.
        <header
            className={cn(
                DRAG_REGION,
                "bg-background/85 supports-[backdrop-filter]:bg-background/70 sticky inset-x-0 top-0 z-40 border-b pt-[var(--titlebar-inset)] pr-[var(--window-controls-width)] backdrop-blur-xl"
            )}
        >
            <div className={cn(DRAG_REGION, "flex items-center gap-3 px-3 py-2 sm:px-4")}>
                <div className="md:hidden">
                    <MobileSidebar />
                </div>

                <div className={cn(DRAG_REGION, "min-w-0 shrink-0 md:w-56 lg:w-64")}>
                    {/* Until mounted, this bar cannot know which project is
                        open: the static export prerenders it with no search
                        params. Claiming "No project selected" in that gap made
                        the header flash the wrong answer on every load, so show
                        a placeholder and let the real name replace it. */}
                    {!mounted ? (
                        <div aria-hidden className="space-y-1.5">
                            <div className="bg-muted h-3.5 w-32 animate-pulse rounded" />
                            <div className="bg-muted h-2.5 w-20 animate-pulse rounded" />
                        </div>
                    ) : project ? (
                        <>
                            {/* The name of what you are working on is the one
                                thing a title bar always carries, so it drags
                                the window like the space around it. */}
                            <p className={cn(DRAG_REGION, "t-section truncate")} title={project.name}>
                                {project.name || "Untitled project"}
                            </p>
                            <p className={cn(DRAG_REGION, "text-muted-foreground truncate text-[0.6875rem]")}>
                                {projectTypeLabel(project.type)}
                            </p>
                        </>
                    ) : (
                        <p className={cn(DRAG_REGION, "text-muted-foreground text-xs")}>No project selected</p>
                    )}
                </div>

                {mounted && projectId && (
                    <StageRail
                        stages={stages}
                        current={current}
                        onSelect={goToStage}
                        className="hidden min-w-0 flex-1 md:flex"
                    />
                )}

                <div className="ml-auto flex shrink-0 items-center gap-1">
                    <ThemeToggle />
                </div>
            </div>

            {/* On narrow windows the rail moves to its own row rather than
                shrinking the labels into illegibility. */}
            {mounted && projectId && (
                <StageRail stages={stages} current={current} onSelect={goToStage} className="px-2 pb-2 md:hidden" />
            )}
        </header>
    )
}

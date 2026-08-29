import type { Metadata } from "next"
import { Suspense } from "react"

import Sidebar from "@/components/layout/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import { DRAG_REGION } from "@/lib/desktop"
import { cn } from "@/lib/utils"

export const metadata: Metadata = {
    title: "Settings — AnyLearning",
}

export default function SettingsLayout({ children }: { children: React.ReactNode }) {
    return (
        <div className="bg-background flex h-dvh overflow-hidden">
            <Suspense fallback={<aside className="hidden w-[252px] flex-none border-r md:block" />}>
                <Sidebar />
            </Suspense>
            <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
                {/* Settings has no project, so the bar carries a title rather
                    than the stage rail. Same height and hairline as the
                    workbench bar so the shell does not shift between screens —
                    and, like it, this strip is the window's title bar. */}
                <header
                    className={cn(
                        DRAG_REGION,
                        "bg-background/85 supports-[backdrop-filter]:bg-background/70 sticky inset-x-0 top-0 z-40 border-b pt-[var(--titlebar-inset)] pr-[var(--window-controls-width)] backdrop-blur-xl"
                    )}
                >
                    <div className={cn(DRAG_REGION, "flex items-center gap-3 px-3 py-2 sm:px-4")}>
                        <div className={cn(DRAG_REGION)}>
                            <p className={cn(DRAG_REGION, "t-section")}>Settings</p>
                            <p className={cn(DRAG_REGION, "text-muted-foreground text-[0.6875rem]")}>
                                Applies to every project
                            </p>
                        </div>
                    </div>
                </header>
                <Suspense fallback={<Skeleton className="m-6 h-64 rounded-lg" />}>{children}</Suspense>
            </main>
        </div>
    )
}

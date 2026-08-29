import type { Metadata } from "next"
import { Suspense } from "react"

import Sidebar from "@/components/layout/sidebar"
import WorkbenchBar from "@/components/layout/workbench-bar"
import { Skeleton } from "@/components/ui/skeleton"

export const metadata: Metadata = {
    title: "AnyLearning",
    description: "Build your own AI models!",
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
    return (
        // h-dvh, not min-h-dvh: the shell has to be exactly the window so the
        // scroll areas inside it have a height to resolve against. With a
        // minimum the shell grew past the viewport instead, and because <body>
        // is overflow-hidden everything below the fold was clipped with no way
        // to scroll to it.
        <div className="bg-background flex h-dvh overflow-hidden">
            <Suspense fallback={<SidebarSkeleton />}>
                <Sidebar />
            </Suspense>
            <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
                <Suspense fallback={<div className="h-[3.75rem] border-b" />}>
                    <WorkbenchBar />
                </Suspense>
                {children}
            </main>
        </div>
    )
}

function SidebarSkeleton() {
    return (
        <aside className="bg-surface hidden h-dvh w-[260px] flex-none border-r p-4 md:block" aria-busy="true">
            <div className="flex items-center gap-3 py-2">
                <Skeleton className="size-9 rounded-lg" />
                <div className="space-y-2">
                    <Skeleton className="h-3.5 w-28" />
                    <Skeleton className="h-2.5 w-20" />
                </div>
            </div>
            <div className="mt-10 space-y-2">
                {Array.from({ length: 6 }).map((_, index) => (
                    <Skeleton key={index} className="h-10 w-full rounded-md" />
                ))}
            </div>
        </aside>
    )
}

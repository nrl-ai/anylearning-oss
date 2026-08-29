import React from "react"

import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

/**
 * The workspace area under the workbench bar.
 *
 * It fills the remaining height via flex rather than subtracting a hard-coded
 * header height, so the bar can change size (it wraps the stage rail onto a
 * second row on narrow windows) without leaving a dead strip at the bottom.
 */
export default function PageContainer({
    children,
    scrollable = true,
    className,
}: {
    children: React.ReactNode
    scrollable?: boolean
    className?: string
}) {
    const inner = (indexed: boolean) => (
        <div
            className={cn(
                "mx-auto w-full max-w-[1600px] p-4 sm:p-5 md:px-6 md:py-5",
                // A page that scrolls its own panes (the dataset grid) needs a
                // height to resolve against. Without this the wrapper sizes to
                // its content, every `h-full` inside it resolves to auto, and
                // the panes grow past the viewport instead of scrolling -- so
                // the bottom of the column was simply unreachable.
                indexed && "flex h-full min-h-0 flex-col",
                className
            )}
        >
            {children}
        </div>
    )

    return scrollable ? (
        <ScrollArea className="min-h-0 flex-1">{inner(false)}</ScrollArea>
    ) : (
        <div className="min-h-0 flex-1 overflow-hidden">{inner(true)}</div>
    )
}

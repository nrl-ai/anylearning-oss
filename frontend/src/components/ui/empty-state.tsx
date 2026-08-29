import { LucideIcon } from "lucide-react"

import { cn } from "@/lib/utils"

/**
 * Every "there's nothing here yet" moment in the app, in one shape.
 *
 * An empty screen is an invitation to act, so the action is part of the
 * component and the copy says what to do next rather than apologising. The
 * icon is drawn in the muted tone: an empty dataset is not a warning, and
 * painting it amber taught users to ignore real warnings.
 */
export function EmptyState({
    icon: Icon,
    title,
    description,
    action,
    className,
    compact = false,
}: {
    icon?: LucideIcon
    title: string
    description?: string
    action?: React.ReactNode
    className?: string
    compact?: boolean
}) {
    return (
        <div
            className={cn(
                "flex flex-col items-center justify-center text-center",
                compact ? "gap-2 px-4 py-8" : "gap-3 px-6 py-14",
                className
            )}
        >
            {Icon && (
                <div
                    className={cn(
                        "text-muted-foreground/70 bg-muted flex items-center justify-center rounded-full",
                        compact ? "size-9" : "size-12"
                    )}
                >
                    <Icon className={compact ? "size-4" : "size-5"} strokeWidth={1.75} />
                </div>
            )}
            <div className="space-y-1">
                <p className={cn("t-section", compact && "text-[0.8125rem]")}>{title}</p>
                {description && <p className="text-muted-foreground mx-auto max-w-sm text-xs">{description}</p>}
            </div>
            {action && <div className="mt-1 flex items-center gap-2">{action}</div>}
        </div>
    )
}

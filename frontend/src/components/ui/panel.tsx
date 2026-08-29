import { LucideIcon } from "lucide-react"

import { cn } from "@/lib/utils"

/**
 * A panel is the app's unit of content: one surface, one hairline, one job.
 *
 * It exists so every screen gets the same header rhythm and the same padding
 * instead of each section hand-rolling a Card with its own spacing. Elevation
 * comes from the surface lightness step, not a shadow — see DESIGN.md.
 */
export function Panel({ className, inset = false, ...props }: React.ComponentProps<"section"> & { inset?: boolean }) {
    return (
        <section
            data-slot="panel"
            className={cn("bg-card text-card-foreground rounded-lg border", inset && "bg-surface-sunken", className)}
            {...props}
        />
    )
}

/** Header row: title on the left, actions on the right, hairline underneath. */
export function PanelHeader({
    icon: Icon,
    title,
    description,
    actions,
    className,
}: {
    icon?: LucideIcon
    title: React.ReactNode
    description?: React.ReactNode
    actions?: React.ReactNode
    className?: string
}) {
    return (
        <div className={cn("flex items-start justify-between gap-3 border-b px-4 py-3", className)}>
            <div className="flex min-w-0 items-start gap-2.5">
                {Icon && <Icon className="text-muted-foreground mt-0.5 size-4 shrink-0" strokeWidth={1.75} />}
                <div className="min-w-0 space-y-0.5">
                    <h2 className="t-section truncate">{title}</h2>
                    {description && <p className="text-muted-foreground text-xs">{description}</p>}
                </div>
            </div>
            {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </div>
    )
}

export function PanelBody({ className, ...props }: React.ComponentProps<"div">) {
    return <div className={cn("p-4", className)} {...props} />
}

export function PanelFooter({ className, ...props }: React.ComponentProps<"div">) {
    return <div className={cn("flex items-center gap-2 border-t px-4 py-2.5", className)} {...props} />
}

/**
 * A labelled value. Numbers get the mono face with tabular figures so a column
 * of them lines up and can be compared at a glance; pass mono={false} for
 * values that are words, which the mono face only makes harder to read.
 */
export function Stat({
    label,
    value,
    hint,
    mono = true,
    className,
}: {
    label: string
    value: React.ReactNode
    hint?: React.ReactNode
    mono?: boolean
    className?: string
}) {
    return (
        <div className={cn("min-w-0 space-y-1", className)}>
            <p className="t-eyebrow">{label}</p>
            <p className={cn("truncate text-sm leading-none font-medium", mono && "tabular font-mono")}>{value}</p>
            {hint && <p className="text-muted-foreground text-[0.6875rem]">{hint}</p>}
        </div>
    )
}

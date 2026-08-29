"use client"

import { AlertTriangle, ChevronDown, Info, X } from "lucide-react"
import { useCallback, useEffect, useState } from "react"

import { cn } from "@/lib/utils"

export interface Notice {
    /** Stable across polls: it is what a dismissal is remembered against. */
    key: string
    level: "warn" | "info"
    title: string
    detail?: string
    action?: React.ReactNode
}

/**
 * Advisory rows that stay out of the way.
 *
 * Three constraints shaped this. They repeat -- the training panel re-polls
 * every few seconds and the same advice comes back each time, so a banner the
 * user has read is a banner they have to keep reading. They vary in length:
 * "the loss became NaN" is a headline, the reason and the number to try are a
 * paragraph. And several can apply at once.
 *
 * So each is one line with its detail folded away, and each can be dismissed.
 * Dismissals are remembered in localStorage per scope, because the alternative
 * is that closing one only closes it until the next poll -- which reads as the
 * dismiss button being broken.
 *
 * Dismissal is per key, not per screen: a NaN loss dismissed on one project
 * should still be pointed out on another.
 */
export function Notices({
    notices,
    scope,
    className,
}: {
    notices: Notice[]
    /** Namespaces the dismissals, usually the project id. */
    scope: string
    className?: string
}) {
    const storageKey = `anylearning.dismissed-notices.${scope}`
    const [dismissed, setDismissed] = useState<string[]>([])
    const [expanded, setExpanded] = useState<string | null>(null)

    useEffect(() => {
        try {
            const stored = window.localStorage.getItem(storageKey)
            setDismissed(stored ? JSON.parse(stored) : [])
        } catch {
            // A browser with storage disabled still gets the notices, it just
            // cannot remember that they were dismissed.
            setDismissed([])
        }
    }, [storageKey])

    const dismiss = useCallback(
        (key: string) => {
            setDismissed((previous) => {
                const next = [...new Set([...previous, key])]
                try {
                    window.localStorage.setItem(storageKey, JSON.stringify(next))
                } catch {
                    /* not fatal, see above */
                }
                return next
            })
        },
        [storageKey]
    )

    const visible = notices.filter((notice) => !dismissed.includes(notice.key))
    if (visible.length === 0) return null

    return (
        <div className={cn("space-y-1.5", className)}>
            {visible.map((notice) => {
                const isOpen = expanded === notice.key
                const Icon = notice.level === "warn" ? AlertTriangle : Info
                return (
                    <div
                        key={notice.key}
                        className={cn(
                            "rounded-md border px-2.5 py-1.5 text-xs",
                            notice.level === "warn"
                                ? "border-warn-border bg-warn-surface text-warn"
                                : "text-muted-foreground"
                        )}
                    >
                        <div className="flex items-start gap-2">
                            <Icon className="mt-0.5 size-3.5 shrink-0" strokeWidth={2} />
                            <button
                                type="button"
                                onClick={() => setExpanded(isOpen ? null : notice.key)}
                                aria-expanded={isOpen}
                                disabled={!notice.detail}
                                className="min-w-0 flex-1 text-left disabled:cursor-default"
                            >
                                <span className={cn(!isOpen && "line-clamp-1")}>{notice.title}</span>
                            </button>
                            {notice.detail && (
                                <ChevronDown
                                    aria-hidden
                                    className={cn(
                                        "mt-0.5 size-3.5 shrink-0 transition-transform",
                                        !isOpen && "-rotate-90"
                                    )}
                                />
                            )}
                            <button
                                type="button"
                                onClick={() => dismiss(notice.key)}
                                aria-label={`Dismiss: ${notice.title}`}
                                className="mt-0.5 shrink-0 opacity-60 transition-opacity hover:opacity-100"
                            >
                                <X className="size-3.5" />
                            </button>
                        </div>
                        {isOpen && notice.detail && (
                            <p className="mt-1.5 pl-5.5 leading-relaxed opacity-90">{notice.detail}</p>
                        )}
                        {isOpen && notice.action && <div className="mt-2 pl-5.5">{notice.action}</div>}
                    </div>
                )
            })}
        </div>
    )
}

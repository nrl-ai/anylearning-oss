"use client"

import { Info } from "lucide-react"

import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"

/**
 * The (i) next to a setting: what it does, and how to decide.
 *
 * Explanations used to sit under each field as permanent grey text, which made
 * a dialog of eight settings a wall of prose that experienced users read past
 * and new users still had to interpret. Behind a button, the same words are
 * there for whoever wants them and out of the way of everyone else.
 *
 * A click rather than a hover: hover text cannot be reached from a touch screen
 * and disappears the moment the pointer moves toward it.
 */
export function InfoHint({
    children,
    label,
    className,
}: {
    children: React.ReactNode
    /** What the hint is about, for anyone using a screen reader. */
    label: string
    className?: string
}) {
    return (
        <Popover>
            <PopoverTrigger
                type="button"
                aria-label={`About ${label}`}
                className={cn(
                    "text-muted-foreground hover:text-foreground focus-visible:ring-ring inline-flex size-4 shrink-0 items-center justify-center rounded-full transition-colors focus-visible:ring-2 focus-visible:outline-none",
                    className
                )}
            >
                <Info className="size-3.5" strokeWidth={2} />
            </PopoverTrigger>
            <PopoverContent
                align="start"
                className="text-muted-foreground w-72 text-xs leading-relaxed"
                // Stops a click inside the explanation from closing the dialog
                // it is explaining.
                onClick={(event) => event.stopPropagation()}
            >
                {children}
            </PopoverContent>
        </Popover>
    )
}

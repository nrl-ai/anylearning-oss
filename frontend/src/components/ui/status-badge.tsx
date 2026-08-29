import { Tone, toneChip, toneSolid, trainingStatus } from "@/lib/status"
import { cn } from "@/lib/utils"

/**
 * The one way this app shows machine state. A dot carries the tone, the word
 * carries the meaning; running states breathe so a live process is legible
 * without reading the label.
 */
export function StatusBadge({ tone, label, className }: { tone: Tone; label: string; className?: string }) {
    return (
        <span
            className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap",
                toneChip[tone],
                className
            )}
        >
            <span
                aria-hidden
                className={cn("size-1.5 rounded-full", toneSolid[tone], tone === "run" && "animate-breathe")}
            />
            {label}
        </span>
    )
}

/** Convenience wrapper for the training-session statuses the backend reports. */
export function TrainingStatusBadge({ status, className }: { status: string; className?: string }) {
    const { tone, label } = trainingStatus(status)
    return <StatusBadge tone={tone} label={label} className={className} />
}

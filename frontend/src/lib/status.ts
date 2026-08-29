/**
 * Single source of truth for how machine state is named and coloured.
 *
 * Every status the backend can report maps to exactly one tone, and each tone
 * owns a token trio (text / surface / border). Screens ask for a tone rather
 * than picking their own colours, so a training session looks the same on the
 * training screen, the models table and the stage rail.
 */

export type Tone = "idle" | "run" | "ok" | "warn" | "fail"

/** Tailwind classes for a tone used as a chip. */
export const toneChip: Record<Tone, string> = {
    idle: "bg-idle-surface text-idle border-idle-border",
    run: "bg-run-surface text-run border-run-border",
    ok: "bg-ok-surface text-ok border-ok-border",
    warn: "bg-warn-surface text-warn border-warn-border",
    fail: "bg-fail-surface text-fail border-fail-border",
}

/** Tailwind classes for a tone used as a solid fill (dots, bars, rails). */
export const toneSolid: Record<Tone, string> = {
    idle: "bg-idle",
    run: "bg-run",
    ok: "bg-ok",
    warn: "bg-warn",
    fail: "bg-fail",
}

/** Tailwind classes for a tone used as foreground text or an icon. */
export const toneText: Record<Tone, string> = {
    idle: "text-idle",
    run: "text-run",
    ok: "text-ok",
    warn: "text-warn",
    fail: "text-fail",
}

/**
 * A training session's status, as tone plus the label we show the user.
 *
 * "terminated" reads as neutral, not failed: the user stopped it on purpose,
 * and painting a deliberate stop red makes the screen look broken.
 */
export function trainingStatus(status: string): { tone: Tone; label: string } {
    switch (status?.toLowerCase()) {
        case "not_started":
            return { tone: "idle", label: "Queued" }
        case "training":
            return { tone: "run", label: "Training" }
        case "evaluating":
            return { tone: "run", label: "Evaluating" }
        case "finished":
            return { tone: "ok", label: "Finished" }
        case "error":
            return { tone: "fail", label: "Failed" }
        case "terminated":
            return { tone: "idle", label: "Stopped" }
        default:
            return { tone: "idle", label: status ? status.replace(/_/g, " ") : "Unknown" }
    }
}

/** True while a session is still doing work and worth polling. */
export function isActiveStatus(status: string | undefined | null): boolean {
    return ["not_started", "training", "evaluating"].includes((status ?? "").toLowerCase())
}

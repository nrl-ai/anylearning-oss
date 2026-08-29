import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

import { Tone, toneChip, trainingStatus } from "@/lib/status"

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs))
}

/**
 * Chip classes for a training-session status.
 *
 * Prefer `<TrainingStatusBadge status={...} />`, which also gets the label and
 * the live dot right. This helper stays for the few call sites that pass a
 * class-name getter down as a prop, and now routes through the same tone table
 * instead of its own set of hand-picked palette classes.
 */
export const getStatusColor = (status: string): string => {
    const tone: Tone = trainingStatus(status).tone
    return `border ${toneChip[tone]}`
}

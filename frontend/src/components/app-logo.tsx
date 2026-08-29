import { cn } from "@/lib/utils"

/**
 * The AnyLearning mark: an annotation box caught mid-draw.
 *
 * It is the app's own vocabulary rather than a monogram -- a bounding box with
 * one live corner handle, exactly what the canvas draws when you label. The
 * frame takes `currentColor` and the handle takes `--mark`, so the mark is
 * correct in both themes without a second asset, and it has no baked tile: the
 * old logo carried its own dark background and read as a foreign chip sitting
 * on the sidebar.
 *
 * The open corner is deliberate. A closed rectangle is a generic crop glyph;
 * the gap plus the handle says a box is being drawn right now.
 */
export function AppLogo({ className }: { className?: string }) {
    return (
        <svg viewBox="0 0 24 24" fill="none" role="img" aria-label="AnyLearning" className={cn("size-8", className)}>
            <path d="M7.5 4H16.5A3.5 3.5 0 0 1 20 7.5V14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            <path
                d="M14 20H7.5A3.5 3.5 0 0 1 4 16.5V7.5A3.5 3.5 0 0 1 7.5 4"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
            />
            <rect x="17" y="17" width="5.5" height="5.5" rx="1.5" fill="var(--mark)" />
        </svg>
    )
}

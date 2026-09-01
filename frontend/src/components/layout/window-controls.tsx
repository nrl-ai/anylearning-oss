"use client"

import { useWindowChrome } from "@/hooks/useWindowChrome"
import { closeWindow, minimizeWindow, toggleWindowMaximized } from "@/lib/desktop"
import { cn } from "@/lib/utils"

/**
 * Minimise, maximise and close, for the platforms whose frame we replaced.
 *
 * macOS is absent on purpose: its traffic lights are put back natively
 * (`anylearning/window_chrome/cocoa.py`) because nothing drawn here would be
 * as familiar, and the shell leaves them the corner they sit in. In a browser
 * there is no window to control and this renders nothing at all.
 *
 * The glyphs are drawn rather than borrowed from the icon set: window controls
 * are 10px marks on a 1px grid, and lucide's 24px strokes scaled down to that
 * size read as fuzzy. They are also the one place in the app where a system
 * convention outranks the house style -- close is on the right, and it is the
 * only control that goes red.
 */

function Glyph({ children }: { children: React.ReactNode }) {
    return (
        <svg
            viewBox="0 0 10 10"
            fill="none"
            stroke="currentColor"
            strokeWidth={1}
            strokeLinecap="square"
            aria-hidden
            className="size-2.5"
        >
            {children}
        </svg>
    )
}

function ControlButton({
    label,
    onClick,
    className,
    children,
}: {
    label: string
    onClick: () => void
    className?: string
    children: React.ReactNode
}) {
    return (
        <button
            type="button"
            aria-label={label}
            title={label}
            onClick={onClick}
            className={cn(
                "text-muted-foreground hover:bg-accent hover:text-foreground focus-visible:ring-ring/50 flex size-7 items-center justify-center rounded-md transition-colors outline-none focus-visible:ring-[3px]",
                className
            )}
        >
            {children}
        </button>
    )
}

export function WindowControls({ className }: { className?: string }) {
    const platform = useWindowChrome((state) => state.platform)
    const maximized = useWindowChrome((state) => state.maximized)
    const nativeFrame = useWindowChrome((state) => state.nativeFrame)

    if (nativeFrame || (platform !== "windows" && platform !== "linux")) return null

    return (
        <div
            // Excluded from the drag surface underneath it, or Windows would
            // report these as title bar and the presses would drag the window.
            data-window-no-drag
            className={cn("pointer-events-auto flex items-center gap-0.5", className)}
        >
            <ControlButton label="Minimize" onClick={minimizeWindow}>
                <Glyph>
                    <line x1="1" y1="5" x2="9" y2="5" />
                </Glyph>
            </ControlButton>

            <ControlButton label={maximized ? "Restore" : "Maximize"} onClick={toggleWindowMaximized}>
                <Glyph>
                    {maximized ? (
                        // The restore mark: the window in front, the size it
                        // would go back to behind it.
                        <>
                            <rect x="1" y="3" width="6" height="6" />
                            <path d="M3 3V1h6v6H7" />
                        </>
                    ) : (
                        <rect x="1.5" y="1.5" width="7" height="7" />
                    )}
                </Glyph>
            </ControlButton>

            <ControlButton label="Close" onClick={closeWindow} className="hover:bg-fail hover:text-white">
                <Glyph>
                    <path d="M1.5 1.5l7 7M8.5 1.5l-7 7" />
                </Glyph>
            </ControlButton>
        </div>
    )
}

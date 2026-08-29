import { X } from "lucide-react"
import { useEffect } from "react"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { DRAG_REGION } from "@/lib/desktop"
import { cn } from "@/lib/utils"

import { useAutoSaveSettingStore } from "./stores"

const getSubsetName = (subset: number) => {
    switch (subset) {
        case 0:
            return "Training set"
        case 1:
            return "Validation set"
        case 2:
            return "Test set"
        default:
            return "Unknown set"
    }
}

type TopBarProps = {
    projectName?: string
    subset: number
    savingStatus: string
    onExit?: () => void
}

export default function TopBar({ projectName, subset, savingStatus, onExit }: TopBarProps) {
    const { isEnabled: isAutoSaveEnabled, setEnabled } = useAutoSaveSettingStore()

    // The window controls are drawn once, at the root, and centre themselves
    // on whichever bar is beneath them. Announce this one for as long as it is
    // the bar on screen.
    useEffect(() => {
        document.documentElement.dataset.titlebar = "compact"
        return () => {
            delete document.documentElement.dataset.titlebar
        }
    }, [])

    return (
        // The labelling screen covers the window, so this bar is the window's
        // title bar while it is open: it drags the window, steps aside for the
        // traffic lights on macOS, and stops short of the window controls on
        // Windows and Linux. It is shorter than the workbench bar, which is
        // what the effect above announces.
        <div
            className={cn(
                DRAG_REGION,
                "bg-surface flex h-10 shrink-0 items-center gap-3 border-b pr-[calc(0.75rem+var(--window-controls-width))] pl-[calc(0.75rem+var(--titlebar-inset-left))]"
            )}
        >
            <div className={cn(DRAG_REGION, "flex min-w-0 items-baseline gap-2")}>
                <span className={cn(DRAG_REGION, "t-section truncate")}>{projectName || "Loading…"}</span>
                <span className={cn(DRAG_REGION, "text-muted-foreground shrink-0 text-xs")}>
                    {getSubsetName(subset)}
                </span>
            </div>

            <div className="ml-auto flex items-center gap-3">
                {/* The saving state gets the ok tone rather than a bare green,
                    and holds its row so the bar does not reflow on every save. */}
                <span className="text-ok min-w-24 text-right text-xs transition-opacity duration-300">
                    {savingStatus}
                </span>
                <div className="flex items-center gap-2">
                    <Checkbox
                        id="auto-save"
                        checked={isAutoSaveEnabled}
                        onCheckedChange={(checked) => setEnabled(!!checked)}
                    />
                    <label htmlFor="auto-save" className="cursor-pointer text-xs">
                        Auto-save
                    </label>
                </div>
                {onExit && (
                    <Button variant="ghost" size="sm" onClick={onExit}>
                        <X />
                        Done labelling
                    </Button>
                )}
            </div>
        </div>
    )
}

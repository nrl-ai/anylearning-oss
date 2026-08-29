import { RefreshCcwIcon } from "lucide-react"
import { useReducer } from "react"

import { Shape } from "@/components/react-image-label"
import { Button } from "@/components/ui/button"

type ShapesDebuggerProps = {
    getShapes: () => Shape[]
}
export default function ShapesDebugger({ getShapes }: ShapesDebuggerProps) {
    const [, forceRerender] = useReducer((x: number) => x + 1, 0)
    const shapes = getShapes()

    if (process.env.NODE_ENV !== "development") {
        return null
    }

    return (
        <div className="flex w-full flex-none flex-col gap-2">
            <div className="flex items-center justify-between">
                <p className="t-eyebrow">Shapes · dev only</p>
                <Button size="icon-xs" variant="ghost" aria-label="Refresh shapes" onClick={forceRerender}>
                    <RefreshCcwIcon />
                </Button>
            </div>
            <div className="text-sm">
                <div className="relative">
                    <textarea
                        className="bg-surface-sunken h-[160px] w-full resize-none rounded-md border p-1.5 font-mono text-[0.6875rem]"
                        value={JSON.stringify(shapes, null, 2)}
                        readOnly
                    />
                    <Button
                        size="sm"
                        variant="ghost"
                        className="absolute top-2 right-2"
                        onClick={() => {
                            navigator.clipboard.writeText(JSON.stringify(shapes, null, 2))
                        }}
                    >
                        Copy
                    </Button>
                </div>
            </div>
        </div>
    )
}

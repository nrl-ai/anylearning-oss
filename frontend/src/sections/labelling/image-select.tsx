import { Check, ChevronLeft, ChevronRight } from "lucide-react"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { DataItem } from "@/types"

type ImageSelectProps = {
    currentPage: number
    currentImageIndex: number
    totalPages: number
    dataItems: DataItem[]
    isLoading: boolean
    isError: boolean
    setCurrentImageIndex: (index: number) => Promise<void>
    handlePageChange: (dir: number) => Promise<void>
}

export default function ImageSelect({
    isLoading,
    isError,
    dataItems,
    handlePageChange,
    currentPage,
    totalPages,
    setCurrentImageIndex,
    currentImageIndex,
}: ImageSelectProps) {
    return (
        <div className="flex min-h-0 w-full flex-1 flex-col overflow-hidden">
            <div className="mb-2 flex flex-none items-center justify-between gap-2">
                <p className="t-eyebrow">Images</p>
                <div className="flex items-center gap-1">
                    <Button
                        variant="ghost"
                        size="icon-xs"
                        aria-label="Previous page"
                        onClick={() => handlePageChange(currentPage - 1)}
                        disabled={currentPage === 1}
                    >
                        <ChevronLeft />
                    </Button>
                    <span className="text-muted-foreground tabular font-mono text-[0.6875rem]">
                        {currentPage}/{totalPages}
                    </span>
                    <Button
                        variant="ghost"
                        size="icon-xs"
                        aria-label="Next page"
                        onClick={() => handlePageChange(currentPage + 1)}
                        disabled={currentPage === totalPages}
                    >
                        <ChevronRight />
                    </Button>
                </div>
            </div>

            {isLoading ? (
                <p className="text-muted-foreground text-xs">Loading images…</p>
            ) : isError ? (
                <p className="text-fail text-xs">Couldn't load the image list.</p>
            ) : dataItems.length === 0 ? (
                <p className="text-muted-foreground text-xs">No images in this set.</p>
            ) : (
                <ul className="min-h-0 flex-1 space-y-0.5 overflow-y-auto">
                    {dataItems.map((item, index) => {
                        const isCurrent = index === currentImageIndex
                        return (
                            <li key={item.id}>
                                <button
                                    type="button"
                                    onClick={() => setCurrentImageIndex(index)}
                                    aria-current={isCurrent ? "true" : undefined}
                                    className={cn(
                                        "relative flex w-full items-center gap-2 rounded-md py-1 pr-1.5 pl-2 text-left",
                                        isCurrent ? "bg-accent" : "hover:bg-accent/60"
                                    )}
                                >
                                    {isCurrent && (
                                        <span
                                            aria-hidden
                                            className="bg-mark absolute top-1 bottom-1 left-0 w-0.5 rounded-full"
                                        />
                                    )}
                                    <span className="min-w-0 flex-1 truncate font-mono text-[0.6875rem]">
                                        {item.original_name}
                                    </span>
                                    {/* Same rule as the dataset grid: the state that
                                        needs work is the one that gets marked. */}
                                    {item.labeled ? (
                                        <Check
                                            className="text-muted-foreground/60 size-3.5 shrink-0"
                                            strokeWidth={2.25}
                                        />
                                    ) : (
                                        <span
                                            aria-label="Not labelled yet"
                                            className="bg-warn size-1.5 shrink-0 rounded-full"
                                        />
                                    )}
                                </button>
                            </li>
                        )
                    })}
                </ul>
            )}
        </div>
    )
}

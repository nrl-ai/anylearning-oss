import { ArrowLeft, ArrowRight } from "lucide-react"

import { Button } from "@/components/ui/button"
import { DataItem } from "@/types"

type BottomBarProps = {
    navigateImage: (dir: number) => void
    projectType: string
    selectedTool: string
    currentImageIndex: number
    currentPage: number
    totalPages: number
    dataItems: DataItem[]
}

export const hintFor = (projectType: string, selectedTool: string) => {
    switch (projectType) {
        case "Object Detection":
            return selectedTool === "rectangle"
                ? "Drag on the image to draw a box."
                : "Choose the rectangle tool to draw a box."
        case "Image Segmentation":
            return selectedTool === "polygon"
                ? "Click around an object to outline it."
                : "Choose the polygon tool to outline a region."
        case "Keypoint Detection":
            return "Place each named landmark and assign its instance number."
        default:
            return "Choose a class for each image."
    }
}

export default function BottomBar({
    navigateImage,
    projectType,
    selectedTool,
    currentImageIndex,
    currentPage,
    totalPages,
    dataItems,
}: BottomBarProps) {
    return (
        <div className="bg-surface flex h-12 shrink-0 items-center justify-between gap-3 border-t px-3">
            {/* Navigation is navigation, so both buttons are the same quiet
                control. They previously turned blue whenever the select tool
                happened to be active, which read as a state they don't have. */}
            <Button
                variant="outline"
                size="sm"
                onClick={() => navigateImage(-1)}
                disabled={currentImageIndex === 0 && currentPage === 1}
            >
                <ArrowLeft />
                Previous
            </Button>

            <div className="flex min-w-0 flex-col items-center">
                <span className="truncate font-mono text-xs">
                    {dataItems[currentImageIndex]?.original_name || "No image selected"}
                </span>
                <span className="text-muted-foreground truncate text-[0.6875rem]">
                    {hintFor(projectType, selectedTool)}
                </span>
            </div>

            <Button
                variant="outline"
                size="sm"
                onClick={() => navigateImage(1)}
                disabled={currentImageIndex === dataItems.length - 1 && currentPage === totalPages}
            >
                Next
                <ArrowRight />
            </Button>
        </div>
    )
}

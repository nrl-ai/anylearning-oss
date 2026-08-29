import { Shape } from "@/components/react-image-label"
import ShapesDebugger from "@/sections/labelling/shapes-debugger"
import { DataItem, Project } from "@/types"

import ImageSelect from "./image-select"
import LabelList from "./label-list"

type RightBarProps = {
    projectId: number
    project: Project | undefined
    currentImage: DataItem | undefined
    currentPage: number
    currentImageIndex: number
    totalPages: number
    getShapes: () => Shape[]
    dataItems: DataItem[]
    isLoading: boolean
    isError: boolean
    setCurrentImageIndex: (index: number) => Promise<void>
    handleClassChange: (classId: number) => void
    handlePageChange: (dir: number) => Promise<void>
}

export default function RightBar({
    projectId,
    project,
    currentImage,
    currentPage,
    currentImageIndex,
    totalPages,
    getShapes,
    dataItems,
    isLoading,
    isError,
    setCurrentImageIndex,
    handleClassChange,
    handlePageChange,
}: RightBarProps) {
    return (
        <div className="bg-surface flex w-[250px] shrink-0 flex-col gap-4 overflow-hidden border-l p-3">
            <LabelList
                projectId={projectId}
                project={project}
                currentImage={currentImage}
                handleClassChange={handleClassChange}
            />
            {/* Raw shape JSON is a debugging aid, not a product feature -- it
                was shipping to users as a "SHAPES · DEV ONLY" panel that ate a
                third of the sidebar. Kept for development only. */}
            {process.env.NODE_ENV === "development" && <ShapesDebugger getShapes={getShapes} />}
            <ImageSelect
                currentPage={currentPage}
                currentImageIndex={currentImageIndex}
                totalPages={totalPages}
                dataItems={dataItems}
                isLoading={isLoading}
                isError={isError}
                setCurrentImageIndex={setCurrentImageIndex}
                handlePageChange={handlePageChange}
            />
        </div>
    )
}

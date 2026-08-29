import React from "react"

import { Shape, ShapeColor } from "../base/types"

export const useImageAnnotator = () => {
    const [handles, setHandles] = React.useState<AnnotatorHandles>()
    return { setHandles, annotator: handles }
}

export type AnnotatorHandles = {
    setEditable(value: boolean): void
    setShapes: (shapes: Shape[]) => void
    drawRectangle(): void
    drawPolygon(): void
    drawCircle(): void
    drawEllipse(): void
    drawDot(): void
    stop: () => void
    stopEdit: () => void
    edit: (id: string) => void
    delete: (id: string) => void
    updateCategories: (id: string, categories: string[], color?: ShapeColor) => void
    updateKeypointMetadata: (id: string, groupId: string | number | null, visible: number) => void
    zoom: (factor: number, relative?: boolean) => void
    getShapes: () => Shape[] | null
    container: HTMLDivElement
}

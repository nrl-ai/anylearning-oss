export type AutoLabelingShape = "polygon" | "rectangle"

interface PreviewShape {
    type: AutoLabelingShape
    points: number[][]
    categories: string[]
    phi: number
    id: string
}

interface CategorizedShape {
    categories?: string[]
}

/** Prompt markers and the live SAM preview are canvas state, not annotations. */
export function persistableAnnotationShapes<T extends CategorizedShape>(shapes: T[]): T[] {
    return shapes.filter((shape) => !shape.categories?.some((category) => category.startsWith("AUTOLABEL_")))
}

export function autoLabelingOutputShape(projectType: string | undefined, selected: string): AutoLabelingShape {
    if (projectType === "Object Detection") return "rectangle"
    return selected === "rectangle" ? "rectangle" : "polygon"
}

/** Convert SAM's mask contour into the geometry this project can store. */
export function createAutoLabelingPreview(
    points: number[][],
    projectType: string | undefined,
    selected: string,
    id: string
): PreviewShape {
    const type = autoLabelingOutputShape(projectType, selected)
    const outputPoints =
        type === "rectangle"
            ? (() => {
                  const xs = points.map((point) => point[0])
                  const ys = points.map((point) => point[1])
                  const minX = Math.min(...xs)
                  const maxX = Math.max(...xs)
                  const minY = Math.min(...ys)
                  const maxY = Math.max(...ys)
                  return [
                      [minX, minY],
                      [minX, maxY],
                      [maxX, maxY],
                      [maxX, minY],
                  ]
              })()
            : points

    return {
        type,
        points: outputPoints,
        categories: ["AUTOLABEL_TMP_SHAPE"],
        phi: 0,
        id,
    }
}

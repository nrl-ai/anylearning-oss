export type AutoLabelingShape = "polygon" | "rectangle"

interface PreviewShape {
    type: AutoLabelingShape
    points: number[][]
    categories: string[]
    phi: number
    id: string
    score?: number | null
    group_id?: string | number | null
    attributes?: Record<string, string | number | boolean | null>
    auto_labeling_model?: string
}

export interface AutoLabelingPrediction {
    shape_type: AutoLabelingShape
    points: { x: number; y: number }[]
    label?: string | null
    score?: number | null
    group_id?: string | number | null
    attributes?: Record<string, string | number | boolean | null>
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
    return createAutoLabelingPrediction(
        {
            shape_type: selected === "rectangle" ? "rectangle" : "polygon",
            points: points.map(([x, y]) => ({ x, y })),
            label: "AUTOLABEL_TMP_SHAPE",
        },
        projectType,
        selected,
        id
    )
}

/** Convert one neutral inference shape into editable canvas geometry. */
export function createAutoLabelingPrediction(
    prediction: AutoLabelingPrediction,
    projectType: string | undefined,
    selected: string,
    id: string,
    modelName?: string
): PreviewShape {
    const type = autoLabelingOutputShape(projectType, selected)
    const points = prediction.points.map((point) => [point.x, point.y])
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
        categories: [prediction.label || "AUTOLABEL_TMP_SHAPE"],
        phi: 0,
        id,
        ...(prediction.score !== undefined ? { score: prediction.score } : {}),
        ...(prediction.group_id !== undefined ? { group_id: prediction.group_id } : {}),
        ...(prediction.attributes ? { attributes: { ...prediction.attributes } } : {}),
        ...(modelName ? { auto_labeling_model: modelName } : {}),
    }
}

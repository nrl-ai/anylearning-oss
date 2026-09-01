export const TEXT_AI_PROJECT_TYPE = "Text AI" as const

/** Preview names remain readable; the creation form only offers the current name. */
export function isTextAiProject(type: string | null | undefined): boolean {
    return (
        type === TEXT_AI_PROJECT_TYPE ||
        type === "Text AI & LLM Evaluation" ||
        type === "Text & LLM" ||
        type === "Sentiment Analysis"
    )
}

export function isStructuredProject(type: string | null | undefined): boolean {
    return type === "Tabular AI" || isTextAiProject(type)
}

/** Projects whose label is stored as one class id for the whole image.
 *
 * Handpose also stores detected landmarks in `annotation`, but those are
 * training metadata rather than editable canvas shapes. Treating that object
 * as a shape array can both crash annotation loading and overwrite the
 * landmarks during auto-save.
 */
export function isClassificationProject(type: string | null | undefined): boolean {
    return type === "Image Classification" || type === "Handpose Classification"
}

/** Whether the labeling workspace should load and save drawable shapes. */
export function usesCanvasAnnotations(type: string | null | undefined): boolean {
    return (
        type === "Object Detection" ||
        type === "Image Segmentation" ||
        type === "Instance Segmentation" ||
        type === "Keypoint Detection"
    )
}

export function projectTypeLabel(type: string): string {
    return isTextAiProject(type) ? TEXT_AI_PROJECT_TYPE : type
}

export function structuredTaskLabel(task: string | null | undefined): string {
    switch (task) {
        case "text_classification":
            return "Text classification"
        case "lexical_search":
        case "semantic_search":
            return "Lexical & fuzzy search"
        case "llm_evaluation":
            return "Response evaluation"
        case "classification":
            return "Classification"
        case "regression":
            return "Regression"
        default:
            return task ? task.replaceAll("_", " ") : "Not configured"
    }
}

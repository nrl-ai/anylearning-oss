import { isClassificationProject, isStructuredProject, usesCanvasAnnotations } from "./project-types"

describe("project type workflow contracts", () => {
    it.each(["Image Classification", "Handpose Classification"])(
        "%s uses one image-level class and never canvas annotations",
        (type) => {
            expect(isClassificationProject(type)).toBe(true)
            expect(usesCanvasAnnotations(type)).toBe(false)
        }
    )

    it.each(["Object Detection", "Image Segmentation", "Instance Segmentation", "Keypoint Detection"])(
        "%s uses drawable canvas annotations",
        (type) => {
            expect(isClassificationProject(type)).toBe(false)
            expect(usesCanvasAnnotations(type)).toBe(true)
        }
    )

    it.each(["Tabular AI", "Text AI"])("%s does not enter an image annotation workflow", (type) => {
        expect(isStructuredProject(type)).toBe(true)
        expect(usesCanvasAnnotations(type)).toBe(false)
    })

    it("keeps unknown future project types away from the destructive shape saver", () => {
        expect(usesCanvasAnnotations("Future workflow")).toBe(false)
        expect(usesCanvasAnnotations(undefined)).toBe(false)
    })
})

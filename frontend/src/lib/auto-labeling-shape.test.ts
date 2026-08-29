import { autoLabelingOutputShape, createAutoLabelingPreview, persistableAnnotationShapes } from "./auto-labeling-shape"

describe("SAM output geometry", () => {
    const contour = [
        [9, 7],
        [14, 12],
        [5, 15],
        [3, 10],
    ]

    it("always produces a bounding box for object detection", () => {
        expect(autoLabelingOutputShape("Object Detection", "polygon")).toBe("rectangle")
        expect(createAutoLabelingPreview(contour, "Object Detection", "polygon", "preview")).toEqual({
            type: "rectangle",
            points: [
                [3, 7],
                [3, 15],
                [14, 15],
                [14, 7],
            ],
            categories: ["AUTOLABEL_TMP_SHAPE"],
            phi: 0,
            id: "preview",
        })
    })

    it("keeps the contour for image segmentation", () => {
        expect(createAutoLabelingPreview(contour, "Image Segmentation", "polygon", "preview").points).toBe(contour)
    })

    it("keeps prompts and previews out of saved annotations", () => {
        const saved = { type: "rectangle", categories: ["helmet"] }
        const prompt = { type: "dot", categories: ["AUTOLABEL_ADD_POINT"] }
        const preview = { type: "rectangle", categories: ["AUTOLABEL_TMP_SHAPE"] }

        expect(persistableAnnotationShapes([saved, prompt, preview])).toEqual([saved])
    })
})

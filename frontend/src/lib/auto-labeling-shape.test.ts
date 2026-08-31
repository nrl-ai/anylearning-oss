import {
    autoLabelingOutputShape,
    createAutoLabelingPrediction,
    createAutoLabelingPreview,
    persistableAnnotationShapes,
} from "./auto-labeling-shape"

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
        expect(createAutoLabelingPreview(contour, "Image Segmentation", "polygon", "preview").points).toEqual(contour)
    })

    it("turns a neutral two-corner detector result into an editable labeled box", () => {
        expect(
            createAutoLabelingPrediction(
                {
                    shape_type: "rectangle",
                    points: [
                        { x: 10, y: 20 },
                        { x: 40, y: 60 },
                    ],
                    label: "dog",
                    score: 0.95,
                    group_id: 2,
                    attributes: { class_id: 16 },
                },
                "Object Detection",
                "rectangle",
                "prediction",
                "dfine_n_coco"
            )
        ).toEqual({
            type: "rectangle",
            points: [
                [10, 20],
                [10, 60],
                [40, 60],
                [40, 20],
            ],
            categories: ["dog"],
            phi: 0,
            id: "prediction",
            score: 0.95,
            group_id: 2,
            attributes: { class_id: 16 },
            auto_labeling_model: "dfine_n_coco",
        })
    })

    it("keeps prompts and previews out of saved annotations", () => {
        const saved = { type: "rectangle", categories: ["helmet"] }
        const prompt = { type: "dot", categories: ["AUTOLABEL_ADD_POINT"] }
        const preview = { type: "rectangle", categories: ["AUTOLABEL_TMP_SHAPE"] }

        expect(persistableAnnotationShapes([saved, prompt, preview])).toEqual([saved])
    })
})

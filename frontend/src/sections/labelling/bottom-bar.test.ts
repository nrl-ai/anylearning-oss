import { hintFor } from "./bottom-bar"

describe("labeling workflow hint", () => {
    it("describes the action when the required tool is already active", () => {
        expect(hintFor("Object Detection", "rectangle")).toBe("Drag on the image to draw a box.")
        expect(hintFor("Image Segmentation", "polygon")).toBe("Click around an object to outline it.")
    })

    it("directs the user to the tool when another tool is active", () => {
        expect(hintFor("Object Detection", "select")).toBe("Choose the rectangle tool to draw a box.")
    })
})

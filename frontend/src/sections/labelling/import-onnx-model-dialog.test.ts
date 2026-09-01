import { parseClassNames } from "./import-onnx-model-dialog"

describe("custom ONNX class list", () => {
    it("accepts newline and comma separated labels while preserving output order", () => {
        expect(parseClassNames("person\n bicycle,car\n\ntruck ")).toEqual(["person", "bicycle", "car", "truck"])
    })
})

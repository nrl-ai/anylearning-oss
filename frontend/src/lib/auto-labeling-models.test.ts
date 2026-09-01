import { formatDownloadSize, groupAutoLabelingModels } from "./auto-labeling-models"
import { AutoLabelingModel } from "./use-auto-labeling"

const model = (overrides: Partial<AutoLabelingModel>): AutoLabelingModel => ({
    name: "model",
    display_name: "Model",
    has_downloaded: false,
    is_custom_model: false,
    tasks: ["detection"],
    interaction_mode: "automatic",
    output_modes: ["rectangle"],
    project_types: ["Object Detection"],
    archive_size_bytes: 0,
    ...overrides,
})

describe("auto-labeling model picker", () => {
    it("groups project, prompted, detection and segmentation models in workflow order", () => {
        const groups = groupAutoLabelingModels([
            model({ name: "segment", tasks: ["instance_segmentation"], output_modes: ["polygon"] }),
            model({ name: "detect" }),
            model({ name: "prompt", interaction_mode: "prompted", tasks: ["promptable_segmentation"] }),
            model({ name: "project", is_project_model: true }),
        ])

        expect(groups.map((group) => group.label)).toEqual([
            "Project models",
            "Interactive segmentation",
            "Object detection",
            "Instance segmentation",
        ])
        expect(groups.map((group) => group.models[0].name)).toEqual(["project", "prompt", "detect", "segment"])
    })

    it("shows useful download sizes without pretending unknown sizes are free", () => {
        expect(formatDownloadSize(0)).toBe("Download required")
        expect(formatDownloadSize(512 * 1024 * 1024)).toBe("512 MB download")
        expect(formatDownloadSize(2.25 * 1024 * 1024 * 1024)).toBe("2.3 GB download")
    })
})

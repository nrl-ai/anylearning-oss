import "@testing-library/jest-dom"
import { fireEvent, render, renderHook, screen } from "@testing-library/react"

import LabelList from "@/sections/labelling/label-list"
import { Project } from "@/types"

import { ImageAnnotator } from "../annotator"
import { useImageAnnotator } from "../annotator/hook"
import { useSettingStore } from "../base/store"
import { Dot } from "../base/types"
import { annotationLabelText, isShowingLabels } from "../base/util"

const point = (visible = 1) => new Dot("point", [50, 50], ["top_left"], "#2e2999", 3, visible)

beforeAll(() => {
    Object.defineProperty(global.SVGElement.prototype, "getBBox", {
        configurable: true,
        value: jest.fn().mockReturnValue({ x: 0, y: 0, width: 54, height: 12 }),
    })
})

beforeEach(() => {
    useSettingStore.setState({
        isShowLabels: false,
        isShowKeypointInstances: false,
        isShowKeypointVisibility: false,
        isDimOccludedKeypoints: true,
    })
})

it("builds a keypoint label from the selected display fields", () => {
    const dot = point()
    expect(isShowingLabels(dot)).toBe(false)
    expect(annotationLabelText(dot)).toBe("")

    useSettingStore.setState({ isShowKeypointInstances: true })
    expect(isShowingLabels(dot)).toBe(true)
    expect(annotationLabelText(dot)).toBe("#3")

    useSettingStore.setState({ isShowLabels: true, isShowKeypointVisibility: true })
    expect(annotationLabelText(dot)).toBe("top_left · #3 · occluded")
})

it("does not draw an empty visibility label on a visible point", () => {
    const dot = point(2)
    useSettingStore.setState({ isShowKeypointVisibility: true })

    expect(isShowingLabels(dot)).toBe(false)
    expect(annotationLabelText(dot)).toBe("")
})

it("offers the keypoint display controls and updates their persisted settings", () => {
    const project: Project = {
        id: 7,
        name: "Vertebral landmarks",
        type: "Keypoint Detection",
        description: null,
        createdAt: null,
        updatedAt: null,
        size: 1,
        numTrainedModels: 0,
        newModelsThisMonth: 0,
        labels: [{ id: 1, name: "top_left", color: "#2e2999" }],
    }
    render(<LabelList projectId={project.id} project={project} currentImage={undefined} handleClassChange={jest.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Display" }))
    expect(screen.getByText("Choose what is drawn without changing the saved annotations.")).toBeVisible()
    const landmarkNames = screen.getByRole("switch", { name: "Landmark names" })
    expect(landmarkNames).toHaveAttribute("aria-checked", "false")
    expect(screen.getByRole("switch", { name: "Instance IDs" })).toHaveAttribute("aria-checked", "false")
    expect(screen.getByRole("switch", { name: "Occlusion status" })).toHaveAttribute("aria-checked", "false")
    expect(screen.getByRole("switch", { name: "Dim occluded points" })).toHaveAttribute("aria-checked", "true")

    fireEvent.click(landmarkNames)
    expect(useSettingStore.getState().isShowLabels).toBe(true)
    expect(landmarkNames).toHaveAttribute("aria-checked", "true")
})

it("renders a padded high-contrast chip with the landmark colour as its border", () => {
    useSettingStore.setState({
        isShowLabels: true,
        isShowKeypointInstances: true,
        isShowKeypointVisibility: true,
    })
    const handles = renderHook(useImageAnnotator)
    const view = render(
        <ImageAnnotator
            setHandles={handles.result.current.setHandles}
            naturalSize
            imageUrl="x-ray.png"
            shapes={[point()]}
            width={100}
            height={100}
        />
    )
    const image = view.container.querySelector("svg")!.children[0] as SVGImageElement
    fireEvent(
        image,
        new CustomEvent("testEvent", {
            detail: { testRil: { naturalWidth: 100, naturalHeight: 100 } },
        })
    )

    const text = view.container.querySelector(".annotation-label-text")
    const background = view.container.querySelector(".annotation-label-background")
    expect(text).toHaveTextContent("top_left · #3 · occluded")
    expect(text).toHaveAttribute("fill", "#f8fafc")
    expect(background).toHaveAttribute("fill", "#111827e6")
    expect(background).toHaveAttribute("stroke", "#2e2999")
    expect(background).toHaveAttribute("rx", "4")
    expect(background).toHaveAttribute("width", "64")
    expect(background).toHaveAttribute("height", "16")
})

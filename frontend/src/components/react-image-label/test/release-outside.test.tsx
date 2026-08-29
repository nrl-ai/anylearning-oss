import "@testing-library/jest-dom"
import { fireEvent, render, renderHook } from "@testing-library/react"
import React from "react"

import { AnnotatorHandles, useImageAnnotator } from "../annotator/hook"
import { ImageAnnotator } from "../annotator/index"
import { Rectangle } from "../base/types"
import Util from "../base/util"
import { FakeMouseEvent } from "./helper/MouseEventWithOffsets"

afterEach(() => {
    Util.maxId = 0
})

Object.defineProperty(global.SVGElement.prototype, "getBBox", {
    writable: true,
    value: jest.fn().mockReturnValue({ x: 0, y: 0, width: 0, height: 0 }),
})

const imageUrl = "https://raw.githubusercontent.com/TaqBostan/content/main/Fruit.jpeg"

function mountAnnotator(onReady: (annotator: AnnotatorHandles) => void) {
    const res = renderHook(useImageAnnotator)
    const { setHandles } = res.result.current
    const view = render(
        <ImageAnnotator
            setHandles={setHandles}
            naturalSize
            imageUrl={imageUrl}
            shapes={[]}
            width={700}
            height={400}
            onReady={onReady}
        />
    )
    const img = view.container.querySelector("svg")!.children[0] as SVGImageElement
    fireEvent(img, new CustomEvent("testEvent", { detail: { testRil: { naturalWidth: 1400, naturalHeight: 800 } } }))

    return {
        svg: view.container.querySelector("svg")!,
        container: view.container.children[0] as HTMLDivElement,
        annotator: res.result.current.annotator!,
    }
}

const at = (type: string, offsetX: number, offsetY: number, buttons = 1) =>
    new FakeMouseEvent(type, { bubbles: true, buttons, offsetX, offsetY })

/**
 * Drawing used to die permanently the first time a gesture ended off-canvas.
 *
 * The terminators were bound to the SVG, so releasing the button past the edge
 * of the image -- routine when boxing an object that runs to the border --
 * never cleared `rectOrigin`. The `!this.rectOrigin` guard on mousedown then
 * rejected every later box, with no visible sign of why.
 */
it("keeps drawing rectangles after a release outside the canvas", () => {
    const { svg, annotator } = mountAnnotator((a) => a.drawRectangle())

    // First box: press and drag on the canvas, then release over the page.
    fireEvent(svg, at("mousedown", 100, 100))
    fireEvent(svg, at("mousemove", 150, 200))
    fireEvent(window, new FakeMouseEvent("mouseup", { bubbles: true, buttons: 0 }))

    expect(annotator.getShapes()!).toHaveLength(1)

    // Second box: this is the one the stuck state used to swallow.
    fireEvent(svg, at("mousedown", 300, 300))
    fireEvent(svg, at("mousemove", 400, 380))
    fireEvent(svg, at("mouseup", 400, 380))

    const shapes = annotator.getShapes()!
    expect(shapes).toHaveLength(2)
    expect(shapes[1].type).toBe("rectangle")
})

/**
 * The same defect on the drag path: moving a shape and letting go off-canvas
 * left `lastPoint` set, so the shape could never be picked up again.
 */
it("keeps dragging shapes after a release outside the canvas", () => {
    const { svg, annotator } = mountAnnotator((a) => a.drawRectangle())

    fireEvent(svg, at("mousedown", 100, 100))
    fireEvent(svg, at("mousemove", 200, 200))
    fireEvent(svg, at("mouseup", 200, 200))

    const drawn = annotator.getShapes()![0]
    annotator.edit(drawn.id)

    const shapeEl = svg.querySelector("polyline.grabbable")!
    const originX = (annotator.getShapes()![0] as Rectangle).points[0][0]

    // Drag the shape, releasing outside the canvas.
    fireEvent(shapeEl, at("mousedown", 150, 150))
    fireEvent(svg, at("mousemove", 170, 160))
    fireEvent(window, new FakeMouseEvent("mouseup", { bubbles: true, buttons: 0 }))

    const afterFirstDrag = (annotator.getShapes()![0] as Rectangle).points[0][0]
    expect(afterFirstDrag).toBeGreaterThan(originX)

    // A second drag has to be accepted, which is what the stuck lastPoint broke.
    fireEvent(shapeEl, at("mousedown", 170, 160))
    fireEvent(svg, at("mousemove", 190, 170))
    fireEvent(svg, at("mouseup", 190, 170))

    expect((annotator.getShapes()![0] as Rectangle).points[0][0]).toBeGreaterThan(afterFirstDrag)
})

/**
 * Panning had a nastier variant: a stuck origin left the container's
 * mousemove handler bound, so the canvas scrolled while merely moving the
 * mouse, with no button held.
 */
it("stops panning when the button is released outside the canvas", () => {
    const { container } = mountAnnotator(() => {})

    fireEvent(
        container,
        new FakeMouseEvent("mousedown", { bubbles: true, ctrlKey: true, buttons: 1, clientX: 200, clientY: 200 })
    )
    fireEvent(
        container,
        new FakeMouseEvent("mousemove", { bubbles: true, ctrlKey: true, buttons: 1, clientX: 150, clientY: 150 })
    )
    fireEvent(window, new FakeMouseEvent("mouseup", { bubbles: true, buttons: 0 }))

    const scrolled = { left: container.scrollLeft, top: container.scrollTop }

    // A button-less move must no longer scroll the canvas.
    fireEvent(container, new FakeMouseEvent("mousemove", { bubbles: true, buttons: 0, clientX: 10, clientY: 10 }))

    expect(container.scrollLeft).toBe(scrolled.left)
    expect(container.scrollTop).toBe(scrolled.top)
})

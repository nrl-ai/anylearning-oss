/**
 * Hovering a shape brightens it, and leaving it puts it back.
 *
 * The case this is for is overlapping masks: where several polygons cover the
 * same pixels, nothing tells you which one is under the pointer until you have
 * already clicked it. The fill alpha is the whole mechanism -- 0.25 at rest,
 * 0.45 under the pointer -- so that is what these assertions read.
 */
import "@testing-library/jest-dom"
import { fireEvent, render, renderHook } from "@testing-library/react"
import React from "react"

import { AnnotatorHandles, useImageAnnotator } from "../annotator/hook"
import { ImageAnnotator } from "../annotator/index"
import Util from "../base/util"

afterEach(() => {
    Util.maxId = 0
})

Object.defineProperty(global.SVGElement.prototype, "getBBox", {
    writable: true,
    value: jest.fn().mockReturnValue({ x: 0, y: 0, width: 0, height: 0 }),
})

const rawShapes = [
    {
        type: "rectangle",
        categories: ["class 3"],
        points: [
            [150, 50],
            [200, 50],
            [200, 100],
            [150, 100],
        ],
        color: "#27f17640",
    },
]

function renderAnnotator(onReady?: (annotator: AnnotatorHandles) => void) {
    const res = renderHook(useImageAnnotator)
    const { setHandles } = res.result.current
    const annotator = render(
        <ImageAnnotator
            setHandles={setHandles}
            naturalSize={false}
            imageUrl="https://raw.githubusercontent.com/TaqBostan/content/main/Fruit.jpeg"
            shapes={rawShapes}
            width={700}
            height={800}
            onReady={onReady}
        />
    )
    const img = annotator.container.querySelector("svg")!.children[0] as SVGImageElement
    fireEvent(
        img,
        new CustomEvent("testEvent", {
            detail: { testRil: { naturalWidth: 600, naturalHeight: 700 } },
        })
    )
    return annotator.container.querySelector("svg")!
}

it("brightens a shape under the pointer and restores it on leaving", () => {
    const svg = renderAnnotator()
    const rect = svg.querySelectorAll('polyline:not([points=""])')[0]

    // 0x40 is 0.25 of 255: the resting fill.
    expect(rect).toHaveAttribute("fill", "#27f17640")

    fireEvent.mouseOver(rect)
    // 0x73 is 0.45 of 255.
    expect(rect).toHaveAttribute("fill", "#27f17673")

    fireEvent.mouseOut(rect)
    expect(rect).toHaveAttribute("fill", "#27f17640")
})

it("leaves shapes alone while a new one is being drawn", () => {
    // Drawing drags the pointer across whatever is already on the canvas, and a
    // shape flashing as it is crossed is noise rather than information.
    const svg = renderAnnotator((annotator) => annotator.drawRectangle())
    const rect = svg.querySelectorAll('polyline:not([points=""])')[0]

    fireEvent.mouseOver(rect)
    expect(rect).toHaveAttribute("fill", "#27f17640")
})

import "@testing-library/jest-dom"
import { fireEvent, render, renderHook } from "@testing-library/react"
import React from "react"

import { AnnotatorHandles, useImageAnnotator } from "../annotator/hook"
import { ImageAnnotator } from "../annotator/index"
import Util from "../base/util"
import { FakeMouseEvent } from "./helper/MouseEventWithOffsets"

afterEach(() => {
    Util.maxId = 0
})
export const ns = "http://www.w3.org/2000/svg"

Object.defineProperty(global.SVGElement.prototype, "getBBox", {
    writable: true,
    value: jest.fn().mockReturnValue({
        x: 0,
        y: 0,
        width: 0,
        height: 0,
    }),
})

it("hidden border", () => {
    const imageUrl = "https://raw.githubusercontent.com/TaqBostan/content/main/Fruit.jpeg"
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
    const onReady = (annotator: AnnotatorHandles) => {
        annotator.drawRectangle()
    }
    const res = renderHook(useImageAnnotator)
    const { setHandles } = res.result.current
    const _annotator = render(
        <ImageAnnotator
            setHandles={setHandles}
            naturalSize={false}
            imageUrl={imageUrl}
            shapes={rawShapes}
            width={700}
            height={800}
            hideBorder
            onReady={onReady}
        />
    )
    const _img = _annotator.container.querySelector("svg")!.children[0] as SVGImageElement
    fireEvent(
        _img,
        new CustomEvent("testEvent", {
            detail: { testRil: { naturalWidth: 600, naturalHeight: 700 } },
        })
    )

    const container = _annotator.container.children[0] as HTMLDivElement
    const svg = _annotator.container.querySelector("svg")!

    //#region elements
    let polylines = svg.querySelectorAll('polyline:not([points=""])')
    let discs = svg.querySelectorAll('circle[r="2"]')

    expect(polylines.length).toBe(2)
    expect(discs.length).toBe(0)
    //#endregion

    //#region rect1
    let rect1 = polylines[0]
    let rect1Shadow = polylines[1]

    expect(rect1).toHaveAttribute("fill", "#27f17640")
    expect(rect1).toHaveAttribute(
        "points",
        "171.42857142857142,57.14285714285714 171.42857142857142,114.28571428571428 228.57142857142856,114.28571428571428 228.57142857142856,57.14285714285714 171.42857142857142,57.14285714285714"
    )
    expect(rect1).toHaveAttribute("stroke", "#27f176")
    expect(rect1).toHaveAttribute("stroke-opacity", "0.7")
    expect(rect1).toHaveAttribute("stroke-width", "0")
    expect(rect1).toHaveAttribute("transform", "rotate(0,199.99999999999997,85.71428571428571)")

    expect(rect1Shadow).toHaveAttribute("fill", "none")
    expect(rect1Shadow).toHaveAttribute(
        "points",
        "171.42857142857142,57.14285714285714 171.42857142857142,114.28571428571428 228.57142857142856,114.28571428571428 228.57142857142856,57.14285714285714 171.42857142857142,57.14285714285714"
    )
    expect(rect1Shadow).toHaveAttribute("stroke", "#000000")
    expect(rect1Shadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(rect1Shadow).toHaveAttribute("stroke-width", "4")
    expect(rect1Shadow).toHaveAttribute("transform", "rotate(0,199.99999999999997,85.71428571428571)")
    expect(rect1Shadow).toHaveClass("il-hid")
    //#endregion

    //#region discs
    let _points = [
        [150, 50],
        [150, 100],
        [200, 100],
        [200, 50],
    ]
    discs.forEach((disc, index) => {
        expect(disc).toHaveAttribute("fill", "#000000")
        // Discs are drawn in display coordinates, so the source points above
        // are scaled by the fit ratio (600x700 image in a 700x800 box -> 8/7).
        // Compared numerically: the exact decimal string depends on the order
        // the implementation happens to multiply in.
        expect(parseFloat(disc.getAttribute("cx")!)).toBeCloseTo((_points[index][0] * 8) / 7)
        expect(parseFloat(disc.getAttribute("cy")!)).toBeCloseTo((_points[index][1] * 8) / 7)
        expect(disc).toHaveAttribute("r", "2")
        expect(disc).toHaveAttribute("transform", "rotate(0,199.99999999999997,85.71428571428571)")
        expect(disc).toHaveClass("il-hid")
    })
    //#endregion

    const annotator = res.result.current.annotator!
    const shapes = annotator.getShapes()!
    annotator.edit(shapes[0].id)

    //#region elements
    polylines = svg.querySelectorAll('polyline:not([points=""])')
    discs = svg.querySelectorAll('circle[r="3"]')
    expect(polylines.length).toBe(2)
    expect(discs.length).toBe(4)
    //#endregion

    //#region rect1
    rect1 = polylines[0]
    rect1Shadow = polylines[1]
    expect(rect1).toHaveAttribute("fill", "#27f17640")
    expect(rect1).toHaveAttribute(
        "points",
        "171.42857142857142,57.14285714285714 171.42857142857142,114.28571428571428 228.57142857142856,114.28571428571428 228.57142857142856,57.14285714285714 171.42857142857142,57.14285714285714"
    )
    expect(rect1).toHaveAttribute("stroke", "#27f176")
    expect(rect1).toHaveAttribute("stroke-opacity", "0.7")
    expect(rect1).toHaveAttribute("stroke-width", "2")
    expect(rect1).toHaveAttribute("transform", "rotate(0,199.99999999999997,85.71428571428571)")
    expect(rect1).toHaveClass("grabbable")

    expect(rect1Shadow).toHaveAttribute("fill", "none")
    expect(rect1Shadow).toHaveAttribute(
        "points",
        "171.42857142857142,57.14285714285714 171.42857142857142,114.28571428571428 228.57142857142856,114.28571428571428 228.57142857142856,57.14285714285714 171.42857142857142,57.14285714285714"
    )
    expect(rect1Shadow).toHaveAttribute("stroke", "#000000")
    expect(rect1Shadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(rect1Shadow).toHaveAttribute("stroke-width", "4")
    expect(rect1Shadow).toHaveAttribute("transform", "rotate(0,199.99999999999997,85.71428571428571)")
    //#endregion

    //#region discs
    _points = [
        [150, 50],
        [150, 100],
        [200, 100],
        [200, 50],
    ]
    discs.forEach((disc, index) => {
        expect(disc).toHaveAttribute("fill", "#009900")
        // Discs are drawn in display coordinates, so the source points above
        // are scaled by the fit ratio (600x700 image in a 700x800 box -> 8/7).
        // Compared numerically: the exact decimal string depends on the order
        // the implementation happens to multiply in.
        expect(parseFloat(disc.getAttribute("cx")!)).toBeCloseTo((_points[index][0] * 8) / 7)
        expect(parseFloat(disc.getAttribute("cy")!)).toBeCloseTo((_points[index][1] * 8) / 7)
        expect(disc).toHaveAttribute("r", "3")
        expect(disc).toHaveAttribute("transform", "rotate(0,199.99999999999997,85.71428571428571)")
        expect(disc).toHaveClass(" seg-point")
    })
    //#endregion

    annotator.stopEdit()

    //#region elements
    polylines = svg.querySelectorAll('polyline:not([points=""])')
    discs = svg.querySelectorAll('circle[r="2"]')

    expect(polylines.length).toBe(2)
    expect(discs.length).toBe(0)
    //#endregion

    //#region rect1
    rect1 = polylines[0]
    rect1Shadow = polylines[1]

    expect(rect1).toHaveAttribute("fill", "#27f17640")
    expect(rect1).toHaveAttribute(
        "points",
        "171.42857142857142,57.14285714285714 171.42857142857142,114.28571428571428 228.57142857142856,114.28571428571428 228.57142857142856,57.14285714285714 171.42857142857142,57.14285714285714"
    )
    expect(rect1).toHaveAttribute("stroke", "#27f176")
    expect(rect1).toHaveAttribute("stroke-opacity", "0.7")
    expect(rect1).toHaveAttribute("stroke-width", "0")
    expect(rect1).toHaveAttribute("transform", "rotate(0,199.99999999999997,85.71428571428571)")

    expect(rect1Shadow).toHaveAttribute("fill", "none")
    expect(rect1Shadow).toHaveAttribute(
        "points",
        "171.42857142857142,57.14285714285714 171.42857142857142,114.28571428571428 228.57142857142856,114.28571428571428 228.57142857142856,57.14285714285714 171.42857142857142,57.14285714285714"
    )
    expect(rect1Shadow).toHaveAttribute("stroke", "#000000")
    expect(rect1Shadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(rect1Shadow).toHaveAttribute("stroke-width", "4")
    expect(rect1Shadow).toHaveAttribute("transform", "rotate(0,199.99999999999997,85.71428571428571)")
    expect(rect1Shadow).toHaveClass("il-hid")
    //#endregion

    //#region discs
    _points = [
        [150, 50],
        [150, 100],
        [200, 100],
        [200, 50],
    ]
    discs.forEach((disc, index) => {
        expect(disc).toHaveAttribute("fill", "#000000")
        // Discs are drawn in display coordinates, so the source points above
        // are scaled by the fit ratio (600x700 image in a 700x800 box -> 8/7).
        // Compared numerically: the exact decimal string depends on the order
        // the implementation happens to multiply in.
        expect(parseFloat(disc.getAttribute("cx")!)).toBeCloseTo((_points[index][0] * 8) / 7)
        expect(parseFloat(disc.getAttribute("cy")!)).toBeCloseTo((_points[index][1] * 8) / 7)
        expect(disc).toHaveAttribute("r", "2")
        expect(disc).toHaveAttribute("transform", "rotate(0,199.99999999999997,85.71428571428571)")
        expect(disc).toHaveClass("il-hid")
    })
    //#endregion

    const points = [
        [100, 100],
        [100, 200],
        [150, 200],
        [150, 100],
    ]
    fireEvent(
        svg,
        new FakeMouseEvent("mousedown", {
            bubbles: true,
            buttons: 1,
            offsetX: points[0][0],
            offsetY: points[0][1],
        })
    )
    fireEvent(
        svg,
        new FakeMouseEvent("mousemove", {
            bubbles: true,
            buttons: 1,
            offsetX: points[2][0],
            offsetY: points[2][1],
        })
    )
    fireEvent(
        svg,
        new FakeMouseEvent("mouseup", {
            bubbles: true,
            buttons: 1,
            offsetX: points[2][0],
            offsetY: points[2][1],
        })
    )

    //#region elements
    polylines = svg.querySelectorAll('polyline:not([points=""])')
    discs = svg.querySelectorAll('circle:not([r="12"])')

    expect(polylines.length).toBe(4)
    expect(discs.length).toBe(4)
    //#endregion

    //#region rect2
    const rect2 = polylines[2]
    const rect2Shadow = polylines[3]

    expect(rect2).toHaveAttribute("fill", "#ffffff00")
    expect(rect2).toHaveAttribute("points", "100,100 100,200 150,200 150,100 100,100")
    expect(rect2).toHaveAttribute("stroke", "#ff0000")
    expect(rect2).toHaveAttribute("stroke-opacity", "0.7")
    expect(rect2).toHaveAttribute("stroke-width", "2")
    expect(rect2).toHaveAttribute("transform", "rotate(0,125,150)")
    expect(rect2).toHaveClass("grabbable")

    expect(rect2Shadow).toHaveAttribute("fill", "none")
    expect(rect2Shadow).toHaveAttribute("points", "100,100 100,200 150,200 150,100 100,100")
    expect(rect2Shadow).toHaveAttribute("stroke", "#000000")
    expect(rect2Shadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(rect2Shadow).toHaveAttribute("stroke-width", "4")
    expect(rect2Shadow).toHaveAttribute("transform", "rotate(0,125,150)")
    //#endregion

    //#region discs
    _points = [
        [100, 100],
        [100, 200],
        [150, 200],
        [150, 100],
    ]
    discs = svg.querySelectorAll('circle[r="3"]')
    discs.forEach((disc, index) => {
        expect(disc).toHaveAttribute("fill", "#009900")
        // This shape was drawn through mouse events, whose offsets are already
        // display coordinates -- so unlike the loaded shapes above, no ratio.
        expect(disc).toHaveAttribute("cx", _points[index][0].toString())
        expect(disc).toHaveAttribute("cy", _points[index][1].toString())
        expect(disc).toHaveAttribute("r", "3")
        expect(disc).toHaveAttribute("transform", "rotate(0,125,150)")
        expect(disc).toHaveClass("seg-point")
    })
    //#endregion
})

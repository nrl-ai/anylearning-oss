import "@testing-library/jest-dom"
import { fireEvent, render, renderHook } from "@testing-library/react"
import React from "react"

import { AnnotatorHandles, useImageAnnotator } from "../annotator/hook"
import { ImageAnnotator } from "../annotator/index"
import { Color, Polygon, Shape } from "../base/types"
import Util from "../base/util"
import { FakeMouseEvent } from "./helper/MouseEventWithOffsets"
import { flushAnimationFrame } from "./helper/flushAnimationFrame"

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

it("draw polygon", async () => {
    const imageUrl = "https://raw.githubusercontent.com/TaqBostan/content/main/Fruit.jpeg"
    const rawShapes = [
        {
            id: 1,
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
        {
            id: 2,
            type: "polygon",
            categories: ["class 1", "class 2"],
            points: [
                [150, 50],
                [150, 100],
                [175, 100],
                [175, 120],
                [190, 120],
                [190, 150],
                [220, 150],
                [220, 50],
            ],
            color: "#27f17640",
        },
    ]

    const onReady = (annotator: AnnotatorHandles) => {
        annotator.drawPolygon()
    }
    const res = renderHook(useImageAnnotator)
    const { setHandles } = res.result.current
    const onAdded = (shape: Shape) => {
        res.result.current.annotator!.updateCategories(shape.id, ["class 1", "class 2"], "blue")
    }
    const _annotator = render(
        <ImageAnnotator
            setHandles={setHandles}
            naturalSize
            imageUrl={imageUrl}
            shapes={rawShapes}
            width={700}
            height={400}
            onReady={onReady}
            onAdded={onAdded}
        />
    )
    const _img = _annotator.container.querySelector("svg")!.children[0] as SVGImageElement
    fireEvent(
        _img,
        new CustomEvent("testEvent", {
            detail: { testRil: { naturalWidth: 1400, naturalHeight: 800 } },
        })
    )

    const container = _annotator.container.children[0] as HTMLDivElement
    const svg = _annotator.container.querySelector("svg")!
    const annotator = res.result.current.annotator!

    //#region zoom
    expect(svg).toHaveAttribute("height", "800")
    expect(svg).toHaveAttribute("width", "1400")

    fireEvent(
        container,
        new FakeMouseEvent("wheel", {
            bubbles: true,
            ctrlKey: true,
            pageX: 100,
            pageY: 100,
            // A full notch. Zoom now scales with the wheel delta so a trackpad
            // zooms smoothly; a full notch still gives the historical 1.25.
            deltaY: -100,
        })
    )

    // Wheel zoom is applied once per animation frame rather than per event --
    // zoom() re-plots every shape, and a trackpad emits far more events than
    // frames. Let the frame run before asserting.
    await flushAnimationFrame()

    expect(svg).toHaveAttribute("height", "1000")
    expect(svg).toHaveAttribute("width", "1750")
    //#endregion

    //#region pan
    fireEvent(
        container,
        new FakeMouseEvent("mousedown", {
            bubbles: true,
            ctrlKey: true,
            buttons: 1,
            clientX: 200,
            clientY: 200,
        })
    )
    fireEvent(
        container,
        new FakeMouseEvent("mousemove", {
            bubbles: true,
            ctrlKey: true,
            buttons: 1,
            clientX: 100,
            clientY: 50,
        })
    )
    fireEvent(
        container,
        new FakeMouseEvent("mouseup", {
            bubbles: true,
            ctrlKey: true,
            buttons: 1,
            clientX: 100,
            clientY: 50,
        })
    )

    expect(container.scrollLeft).toBe(100)
    expect(container.scrollTop).toBe(150)
    //#endregion

    //#region draw
    const points = [
            [40, 40],
            [40, 80],
            [80, 80],
            [80, 120],
            [120, 120],
            [120, 160],
            [160, 160],
            [160, 40],
        ],
        pointsCopy = [...points]
    const originalPoints = [
        [32, 32],
        [32, 64],
        [64, 64],
        [64, 96],
        [96, 96],
        [96, 128],
        [128, 128],
        [128, 32],
    ]
    points.forEach((point, i) => {
        fireEvent(
            svg,
            new FakeMouseEvent("click", {
                bubbles: true,
                buttons: 1,
                offsetX: point[0],
                offsetY: point[1],
            })
        )
        if (i === points.length - 1)
            fireEvent(
                svg,
                new FakeMouseEvent("dblclick", {
                    bubbles: true,
                    buttons: 1,
                    offsetX: point[0],
                    offsetY: point[1],
                })
            )
    })
    //#endregion

    //#region zoom
    fireEvent(
        container,
        new FakeMouseEvent("wheel", {
            bubbles: true,
            ctrlKey: true,
            pageX: 100,
            pageY: 100,
            deltaY: -100,
        })
    )

    await flushAnimationFrame()

    expect(svg).toHaveAttribute("height", "1250")
    expect(svg).toHaveAttribute("width", "2187.5")

    points.forEach((p) => {
        ;((p[0] *= 1.25), (p[1] *= 1.25))
    })
    //#endregion

    //#region edit shape
    const point = points[3]
    const greenDisc = svg.querySelector(`circle[r="3"][cx="${point[0]}"][cy="${point[1]}"]`)!

    fireEvent(
        greenDisc,
        new FakeMouseEvent("mousedown", {
            bubbles: true,
            buttons: 1,
            offsetX: point[0],
            offsetY: point[1],
        })
    )
    fireEvent(
        svg,
        new FakeMouseEvent("mousemove", {
            bubbles: true,
            buttons: 1,
            offsetX: (point[0] * 3) / 2,
            offsetY: (point[1] * 2) / 3,
        })
    )
    fireEvent(
        greenDisc,
        new FakeMouseEvent("mouseup", {
            bubbles: true,
            buttons: 1,
            offsetX: (point[0] * 3) / 2,
            offsetY: (point[1] * 2) / 3,
        })
    )
    points[3][0] *= 3 / 2
    points[3][1] *= 2 / 3
    originalPoints[3][0] *= 3 / 2
    originalPoints[3][1] *= 2 / 3
    //#endregion

    //#region getShapes
    const shapes = annotator.getShapes()!

    expect(shapes).toHaveLength(3)

    const polygon = shapes.filter((s) => !["1", "2"].includes(s.id))[0] as Polygon

    expect(JSON.stringify(polygon.points)).toBe(JSON.stringify(originalPoints))
    expect(polygon.type).toBe("polygon")
    expect(polygon.categories).toHaveLength(2)
    expect(polygon.color).toBe("blue")
    //#endregion

    //#region elements
    const polylines = svg.querySelectorAll("polyline")
    const discs = svg.querySelectorAll('circle[r="2"]')
    const greenDiscs = svg.querySelectorAll('circle[r="3"]')
    const ellipses = svg.querySelectorAll("ellipse")

    expect(polylines.length).toBe(6)
    expect(discs.length).toBe(0)
    expect(greenDiscs.length).toBe(8)
    expect(ellipses.length).toBe(0)
    //#endregion

    //#region green discs
    greenDiscs.forEach((greenDisc) => {
        expect(greenDisc).toHaveClass("seg-point")
        expect(greenDisc).toHaveAttribute("fill", Color.GreenDisc)
        const cx = parseInt(greenDisc.getAttribute("cx")!)
        const cy = parseInt(greenDisc.getAttribute("cy")!)
        expect(pointsCopy.findIndex((p) => p[0] === cx && p[1] === cy)).toBeGreaterThanOrEqual(0)
        pointsCopy.splice(
            pointsCopy.findIndex((p) => p[0] === cx && p[1] === cy),
            1
        )[0]
    })

    expect(pointsCopy).toHaveLength(0)
    //#endregion

    //#region polyline
    const polyline = polylines[4]
    const polylineShadow = polylines[5]
    const _points = `${points[0][0]},${points[0][1]} ${points[1][0]},${points[1][1]} ${points[2][0]},${points[2][1]} ${points[3][0]},${points[3][1]} ${points[4][0]},${points[4][1]} ${points[5][0]},${points[5][1]} ${points[6][0]},${points[6][1]} ${points[7][0]},${points[7][1]} ${points[0][0]},${points[0][1]}`
    const _center = [
        points.reduce((sum: number, p: number[]) => sum + p[0], 0) / 8.0,
        points.reduce((sum: number, p: number[]) => sum + p[1], 0) / 8.0,
    ]

    expect(polyline).toHaveClass("grabbable")
    expect(polyline).toHaveAttribute("fill", Color.ShapeFill)
    expect(polyline).toHaveAttribute("points", _points)
    expect(polyline).toHaveAttribute("stroke", "#ff0000")
    expect(polyline).toHaveAttribute("stroke-opacity", "0.7")
    expect(polyline).toHaveAttribute("stroke-width", "2")
    expect(polyline).not.toHaveAttribute("transform")

    expect(polylineShadow).toHaveAttribute("fill", "none")
    expect(polylineShadow).toHaveAttribute("points", _points)
    expect(polylineShadow).toHaveAttribute("stroke", "#000000")
    expect(polylineShadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(polylineShadow).toHaveAttribute("stroke-width", "4")
    expect(polylineShadow).not.toHaveAttribute("transform")

    annotator.stopEdit()

    expect(polyline).not.toHaveClass("grabbable")
    expect(polyline).toHaveAttribute("fill", "blue")
    expect(polyline).toHaveAttribute("stroke", "blue")
    //#endregion
})

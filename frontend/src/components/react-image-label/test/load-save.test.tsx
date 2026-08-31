import "@testing-library/jest-dom"
import { fireEvent, render, renderHook } from "@testing-library/react"
import React from "react"

import { AnnotatorHandles, useImageAnnotator } from "../annotator/hook"
import { ImageAnnotator } from "../annotator/index"
import { Circle, Dot, Ellipse, Polygon, Rectangle } from "../base/types"
import Util from "../base/util"

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

it("load container, svg, image in natural size", () => {
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
        {
            type: "polygon",
            categories: ["class 1", "class 2"],
            points: [
                [50, 50],
                [50, 100],
                [75, 100],
                [75, 120],
                [90, 120],
                [90, 150],
                [120, 150],
                [120, 50],
            ],
            color: "#27f17640",
        },
        { type: "circle", categories: ["class 4"], centre: [250, 100], radius: 40 },
        {
            type: "ellipse",
            categories: ["class 3"],
            centre: [350, 150],
            radiusX: 60,
            radiusY: 40,
            color: "#27f17640",
        },
    ]

    const onReady = (annotator: AnnotatorHandles) => {}
    const res = renderHook(useImageAnnotator)
    const { setHandles, annotator } = res.result.current
    const _annotator = render(
        <ImageAnnotator
            setHandles={setHandles}
            naturalSize
            imageUrl={imageUrl}
            shapes={rawShapes}
            width={700}
            height={400}
            onReady={onReady}
        />
    )
    const _img = _annotator.container.querySelector("svg")!.children[0] as SVGImageElement
    fireEvent(
        _img,
        new CustomEvent("testEvent", {
            detail: { testRil: { naturalWidth: 800, naturalHeight: 500 } },
        })
    )

    const container = _annotator.container.children[0] as HTMLDivElement
    expect(container.style.height).toBe("400px")
    expect(container.style.width).toBe("700px")
    // data-img holds the whole URL, not just the file name. AnyLearning serves
    // every image in a project from .../data_items/{id}/download, so the last
    // path segment is identical for all of them and could not identify an image.
    expect(container).toHaveAttribute("data-img", imageUrl)

    const svg = _annotator.container.querySelector("svg")!
    expect(svg).toHaveClass("il-svg")
    expect(svg).toHaveAttribute("height", "500")
    expect(svg).toHaveAttribute("width", "800")

    const images = svg.querySelectorAll("image")
    expect(images.length).toBe(1)

    const image = images[0]
    expect(image).toHaveAttribute("height", "100%")
    expect(image).toHaveAttribute("width", "100%")
    expect(image).toHaveAttribute("oncontextmenu", "return false")
    expect(image).toHaveAttribute("onmousedown", "return false")
    expect(image).toHaveAttribute("href", imageUrl)
})

it("load container, svg, image 1", () => {
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
        {
            type: "polygon",
            categories: ["class 1", "class 2"],
            points: [
                [50, 50],
                [50, 100],
                [75, 100],
                [75, 120],
                [90, 120],
                [90, 150],
                [120, 150],
                [120, 50],
            ],
            color: "#27f17640",
        },
        { type: "circle", categories: ["class 4"], centre: [250, 100], radius: 40 },
        {
            type: "ellipse",
            categories: ["class 3"],
            centre: [350, 150],
            radiusX: 60,
            radiusY: 40,
            color: "#27f17640",
        },
    ]

    const onReady = (annotator: AnnotatorHandles) => {}
    const res = renderHook(useImageAnnotator)
    const { setHandles, annotator } = res.result.current
    const _annotator = render(
        <ImageAnnotator
            setHandles={setHandles}
            naturalSize={false}
            imageUrl={imageUrl}
            shapes={rawShapes}
            width={800}
            height={300}
            onReady={onReady}
        />
    )
    const _img = _annotator.container.querySelector("svg")!.children[0] as SVGImageElement
    fireEvent(
        _img,
        new CustomEvent("testEvent", {
            detail: { testRil: { naturalWidth: 800, naturalHeight: 400 } },
        })
    )

    const container = _annotator.container.children[0] as HTMLDivElement
    expect(container.style.height).toBe("300px")
    expect(container.style.width).toBe("800px")
    // data-img holds the whole URL, not just the file name. AnyLearning serves
    // every image in a project from .../data_items/{id}/download, so the last
    // path segment is identical for all of them and could not identify an image.
    expect(container).toHaveAttribute("data-img", imageUrl)

    const svg = _annotator.container.querySelector("svg")!
    expect(svg).toHaveClass("il-svg")
    expect(svg).toHaveAttribute("height", "300")
    expect(svg).toHaveAttribute("width", "600")

    const images = svg.querySelectorAll("image")
    expect(images.length).toBe(1)

    const image = images[0]
    expect(image).toHaveAttribute("height", "100%")
    expect(image).toHaveAttribute("width", "100%")
})

it("load container, svg, image 2", () => {
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
        {
            type: "polygon",
            categories: ["class 1", "class 2"],
            points: [
                [50, 50],
                [50, 100],
                [75, 100],
                [75, 120],
                [90, 120],
                [90, 150],
                [120, 150],
                [120, 50],
            ],
            color: "#27f17640",
        },
        { type: "circle", categories: ["class 4"], centre: [250, 100], radius: 40 },
        {
            type: "ellipse",
            categories: ["class 3"],
            centre: [350, 150],
            radiusX: 60,
            radiusY: 40,
            color: "#27f17640",
        },
    ]

    const onReady = (annotator: AnnotatorHandles) => {}
    const res = renderHook(useImageAnnotator)
    const { setHandles, annotator } = res.result.current
    const _annotator = render(
        <ImageAnnotator
            setHandles={setHandles}
            naturalSize={false}
            imageUrl={imageUrl}
            shapes={rawShapes}
            width={300}
            height={800}
            onReady={onReady}
        />
    )
    const _img = _annotator.container.querySelector("svg")!.children[0] as SVGImageElement
    fireEvent(
        _img,
        new CustomEvent("testEvent", {
            detail: { testRil: { naturalWidth: 400, naturalHeight: 800 } },
        })
    )

    const container = _annotator.container.children[0] as HTMLDivElement
    expect(container.style.height).toBe("800px")
    expect(container.style.width).toBe("300px")
    // data-img holds the whole URL, not just the file name. AnyLearning serves
    // every image in a project from .../data_items/{id}/download, so the last
    // path segment is identical for all of them and could not identify an image.
    expect(container).toHaveAttribute("data-img", imageUrl)

    const svg = _annotator.container.querySelector("svg")!
    expect(svg).toHaveClass("il-svg")
    expect(svg).toHaveAttribute("height", "600")
    expect(svg).toHaveAttribute("width", "300")

    const images = svg.querySelectorAll("image")
    expect(images.length).toBe(1)

    const image = images[0]
    expect(image).toHaveAttribute("height", "100%")
    expect(image).toHaveAttribute("width", "100%")
})

it("load container, svg, image 3", () => {
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
        {
            type: "polygon",
            categories: ["class 1", "class 2"],
            points: [
                [50, 50],
                [50, 100],
                [75, 100],
                [75, 120],
                [90, 120],
                [90, 150],
                [120, 150],
                [120, 50],
            ],
            color: "#27f17640",
        },
        { type: "circle", categories: ["class 4"], centre: [250, 100], radius: 40 },
        {
            type: "ellipse",
            categories: ["class 3"],
            centre: [350, 150],
            radiusX: 60,
            radiusY: 40,
            color: "#27f17640",
        },
    ]

    const onReady = (annotator: AnnotatorHandles) => {}
    const res = renderHook(useImageAnnotator)
    const { setHandles, annotator } = res.result.current
    const _annotator = render(
        <ImageAnnotator
            setHandles={setHandles}
            naturalSize={false}
            imageUrl={imageUrl}
            shapes={rawShapes}
            width={700}
            height={800}
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
    expect(container.style.height).toBe("800px")
    expect(container.style.width).toBe("700px")
    // data-img holds the whole URL, not just the file name. AnyLearning serves
    // every image in a project from .../data_items/{id}/download, so the last
    // path segment is identical for all of them and could not identify an image.
    expect(container).toHaveAttribute("data-img", imageUrl)

    const svg = _annotator.container.querySelector("svg")!
    expect(svg).toHaveClass("il-svg")
    // The canvas scales to fit the available box, enlarging as well as
    // shrinking: a 600x700 image in a 700x800 box is height-limited, so the
    // ratio is 800/700 = 8/7. (It used to refuse to scale above 1:1, which left
    // small images marooned in the middle of a large pane.)
    expect(svg).toHaveAttribute("height", "800")
    expect(svg).toHaveAttribute("width", "685.7142857142857")

    const images = svg.querySelectorAll("image")
    expect(images.length).toBe(1)

    const image = images[0]
    expect(image).toHaveAttribute("height", "100%")
    expect(image).toHaveAttribute("width", "100%")
})

it("load shapes", () => {
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
        {
            type: "polygon",
            categories: ["class 1", "class 2"],
            points: [
                [50, 50],
                [50, 100],
                [75, 100],
                [75, 120],
                [90, 120],
                [90, 150],
                [120, 150],
                [120, 50],
            ],
            color: "#27f17640",
        },
        { type: "circle", categories: ["class 4"], centre: [250, 100], radius: 40 },
        {
            type: "rectangle",
            categories: ["class 2"],
            points: [
                [250, 150],
                [300, 150],
                [300, 200],
                [250, 200],
            ],
            color: "red",
        },
        {
            type: "ellipse",
            categories: ["class 3"],
            centre: [350, 150],
            radiusX: 60,
            radiusY: 40,
            color: "#27f17640",
        },
    ]

    const onReady = (annotator: AnnotatorHandles) => {}
    const res = renderHook(useImageAnnotator)
    const { setHandles, annotator } = res.result.current
    const _annotator = render(
        <ImageAnnotator
            setHandles={setHandles}
            naturalSize={false}
            imageUrl={imageUrl}
            shapes={rawShapes}
            width={700}
            height={800}
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
    const polylines = svg.querySelectorAll("polyline")
    const discs = svg.querySelectorAll('circle[r="2"]')
    const ellipses = svg.querySelectorAll("ellipse")

    expect(polylines.length).toBe(6)
    expect(discs.length).toBe(0)
    expect(ellipses.length).toBe(4)
    //#endregion

    //#region rect1
    const rect1 = polylines[0]
    const rect1Shadow = polylines[1]

    expect(rect1).toHaveAttribute("fill", "#27f17640")
    expect(rect1).toHaveAttribute(
        "points",
        "171.42857142857142,57.14285714285714 171.42857142857142,114.28571428571428 228.57142857142856,114.28571428571428 228.57142857142856,57.14285714285714 171.42857142857142,57.14285714285714"
    )
    expect(rect1).toHaveAttribute("stroke", "#27f176")
    expect(rect1).toHaveAttribute("stroke-opacity", "0.7")
    expect(rect1).toHaveAttribute("stroke-width", "2")
    expect(rect1).toHaveAttribute("transform", "rotate(0,199.99999999999997,85.71428571428571)")
    expect(rect1).not.toHaveClass("il-hid")

    expect(rect1Shadow).toHaveAttribute("fill", "none")
    expect(rect1Shadow).toHaveAttribute(
        "points",
        "171.42857142857142,57.14285714285714 171.42857142857142,114.28571428571428 228.57142857142856,114.28571428571428 228.57142857142856,57.14285714285714 171.42857142857142,57.14285714285714"
    )
    expect(rect1Shadow).toHaveAttribute("stroke", "#000000")
    expect(rect1Shadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(rect1Shadow).toHaveAttribute("stroke-width", "4")
    expect(rect1Shadow).toHaveAttribute("transform", "rotate(0,199.99999999999997,85.71428571428571)")
    expect(rect1Shadow).not.toHaveClass("il-hid")

    //#endregion

    //#region rect2
    const rect2 = polylines[2]
    const rect2Shadow = polylines[3]

    expect(rect2).toHaveAttribute("fill", "red")
    expect(rect2).toHaveAttribute(
        "points",
        "285.7142857142857,171.42857142857142 285.7142857142857,228.57142857142856 342.85714285714283,228.57142857142856 342.85714285714283,171.42857142857142 285.7142857142857,171.42857142857142"
    )
    expect(rect2).toHaveAttribute("stroke", "red")
    expect(rect2).toHaveAttribute("stroke-opacity", "0.7")
    expect(rect2).toHaveAttribute("stroke-width", "2")
    expect(rect2).toHaveAttribute("transform", "rotate(0,314.2857142857143,200)")

    expect(rect2Shadow).toHaveAttribute("fill", "none")
    expect(rect2Shadow).toHaveAttribute(
        "points",
        "285.7142857142857,171.42857142857142 285.7142857142857,228.57142857142856 342.85714285714283,228.57142857142856 342.85714285714283,171.42857142857142 285.7142857142857,171.42857142857142"
    )
    expect(rect2Shadow).toHaveAttribute("stroke", "#000000")
    expect(rect2Shadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(rect2Shadow).toHaveAttribute("stroke-width", "4")
    expect(rect2Shadow).toHaveAttribute("transform", "rotate(0,314.2857142857143,200)")
    //#endregion

    //#region polyline
    const polyline = polylines[4]
    const polylineShadow = polylines[5]

    expect(polyline).toHaveAttribute("fill", "#27f17640")
    expect(polyline).toHaveAttribute(
        "points",
        "57.14285714285714,57.14285714285714 57.14285714285714,114.28571428571428 85.71428571428571,114.28571428571428 85.71428571428571,137.14285714285714 102.85714285714285,137.14285714285714 102.85714285714285,171.42857142857142 137.14285714285714,171.42857142857142 137.14285714285714,57.14285714285714 57.14285714285714,57.14285714285714"
    )
    expect(polyline).toHaveAttribute("stroke", "#27f176")
    expect(polyline).toHaveAttribute("stroke-opacity", "0.7")
    expect(polyline).toHaveAttribute("stroke-width", "2")
    expect(polyline).not.toHaveAttribute("transform")

    expect(polylineShadow).toHaveAttribute("fill", "none")
    expect(polylineShadow).toHaveAttribute(
        "points",
        "57.14285714285714,57.14285714285714 57.14285714285714,114.28571428571428 85.71428571428571,114.28571428571428 85.71428571428571,137.14285714285714 102.85714285714285,137.14285714285714 102.85714285714285,171.42857142857142 137.14285714285714,171.42857142857142 137.14285714285714,57.14285714285714 57.14285714285714,57.14285714285714"
    )
    expect(polylineShadow).toHaveAttribute("stroke", "#000000")
    expect(polylineShadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(polylineShadow).toHaveAttribute("stroke-width", "4")
    expect(polylineShadow).not.toHaveAttribute("transform")
    //#endregion

    // Geometry below is in display coordinates, so it carries the 8/7 fit
    // ratio (a 600x700 image scaled to fit a 700x800 box). Compared numerically
    // because the exact decimal depends on the implementation's multiply order.
    //#region circle
    const circle = ellipses[0]
    const circleShadow = ellipses[1]

    expect(circle).toHaveAttribute("fill", "#ffffff00")
    expect(parseFloat(circle.getAttribute("cx")!)).toBeCloseTo(250.0 * (8 / 7))
    expect(parseFloat(circle.getAttribute("cy")!)).toBeCloseTo(100.0 * (8 / 7))
    expect(parseFloat(circle.getAttribute("rx")!)).toBeCloseTo(40.0 * (8 / 7))
    expect(parseFloat(circle.getAttribute("ry")!)).toBeCloseTo(40.0 * (8 / 7))
    expect(circle).toHaveAttribute("stroke", "#ffffff")
    expect(circle).toHaveAttribute("stroke-opacity", "0.7")
    expect(circle).toHaveAttribute("stroke-width", "2")
    expect(circle).not.toHaveAttribute("transform")

    expect(circleShadow).toHaveAttribute("fill", "none")
    expect(parseFloat(circleShadow.getAttribute("cx")!)).toBeCloseTo(250.0 * (8 / 7))
    expect(parseFloat(circleShadow.getAttribute("cy")!)).toBeCloseTo(100.0 * (8 / 7))
    expect(parseFloat(circleShadow.getAttribute("rx")!)).toBeCloseTo(40.0 * (8 / 7))
    expect(parseFloat(circleShadow.getAttribute("ry")!)).toBeCloseTo(40.0 * (8 / 7))
    expect(circleShadow).toHaveAttribute("stroke", "#000000")
    expect(circleShadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(circleShadow).toHaveAttribute("stroke-width", "4")
    expect(circleShadow).not.toHaveAttribute("transform")
    //#endregion

    //#region ellipse
    const ellipse = ellipses[2]
    const ellipseShadow = ellipses[3]

    expect(ellipse).toHaveAttribute("fill", "#27f17640")
    expect(parseFloat(ellipse.getAttribute("cx")!)).toBeCloseTo(350.0 * (8 / 7))
    expect(parseFloat(ellipse.getAttribute("cy")!)).toBeCloseTo(150.0 * (8 / 7))
    expect(parseFloat(ellipse.getAttribute("rx")!)).toBeCloseTo(60.0 * (8 / 7))
    expect(parseFloat(ellipse.getAttribute("ry")!)).toBeCloseTo(40.0 * (8 / 7))
    expect(ellipse).toHaveAttribute("stroke", "#27f176")
    expect(ellipse).toHaveAttribute("stroke-opacity", "0.7")
    expect(ellipse).toHaveAttribute("stroke-width", "2")
    expect(ellipse).toHaveAttribute("transform", "rotate(0,400,171.42857142857142)")

    expect(ellipseShadow).toHaveAttribute("fill", "none")
    expect(parseFloat(ellipseShadow.getAttribute("cx")!)).toBeCloseTo(350.0 * (8 / 7))
    expect(parseFloat(ellipseShadow.getAttribute("cy")!)).toBeCloseTo(150.0 * (8 / 7))
    expect(parseFloat(ellipseShadow.getAttribute("rx")!)).toBeCloseTo(60.0 * (8 / 7))
    expect(parseFloat(ellipseShadow.getAttribute("ry")!)).toBeCloseTo(40.0 * (8 / 7))
    expect(ellipseShadow).toHaveAttribute("stroke", "#000000")
    expect(ellipseShadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(ellipseShadow).toHaveAttribute("stroke-width", "4")
    expect(ellipseShadow).toHaveAttribute("transform", "rotate(0,400,171.42857142857142)")
    //#endregion
})

it("getShapes", () => {
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
        {
            type: "polygon",
            categories: ["class 1", "class 2"],
            points: [
                [50, 50],
                [50, 100],
                [75, 100],
                [75, 120],
                [90, 120],
                [90, 150],
                [120, 150],
                [120, 50],
            ],
            color: "blue",
        },
        { type: "circle", categories: ["class 4"], centre: [250, 100], radius: 40 },
        {
            type: "rectangle",
            categories: ["class 2"],
            points: [
                [250, 150],
                [300, 150],
                [300, 200],
                [250, 200],
            ],
            color: "red",
        },
        {
            type: "ellipse",
            categories: ["class 3"],
            centre: [350, 150],
            radiusX: 60,
            radiusY: 40,
            color: "yellow",
        },
    ]

    const onReady = (annotator: AnnotatorHandles) => {}
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
    const shapes = res.result.current.annotator!.getShapes()!

    const rectangles = shapes.filter((c) => c.type === "rectangle")
    const polygons = shapes.filter((c) => c.type === "polygon")
    const circles = shapes.filter((c) => c.type === "circle")
    const ellipses = shapes.filter((c) => c.type === "ellipse")

    expect(rectangles.length).toBe(2)
    expect(polygons.length).toBe(1)
    expect(circles.length).toBe(1)
    expect(ellipses.length).toBe(1)

    const rectangle = rectangles[0] as Rectangle
    const rectangle2 = rectangles[1] as Rectangle
    const polygon = polygons[0] as Polygon
    const circle = circles[0] as Circle
    const ellipse = ellipses[0] as Ellipse

    expect(rectangle.color).toBe("#27f17640")
    expect(rectangle.categories.length).toBe(1)
    expect(rectangle.categories[0]).toBe("class 3")
    expect(rectangle.getCenterWithOffset().X).toBeCloseTo(200.0)
    expect(rectangle.getCenterWithOffset().Y).toBeCloseTo(85.7143)
    expect(JSON.stringify(rectangle.points)).toBe("[[150,50],[150,100],[200,100],[200,50]]")
    expect(rectangle.type).toBe("rectangle")

    expect(rectangle2.color).toBe("red")
    expect(rectangle2.categories.length).toBe(1)
    expect(rectangle2.categories[0]).toBe("class 2")
    expect(rectangle2.getCenterWithOffset().X).toBeCloseTo(314.2857)
    expect(rectangle2.getCenterWithOffset().Y).toBeCloseTo(200.0)
    expect(JSON.stringify(rectangle2.points)).toBe("[[250,150],[250,200],[300,200],[300,150]]")
    expect(rectangle2.type).toBe("rectangle")

    expect(polygon.color).toBe("blue")
    expect(polygon.categories.length).toBe(2)
    expect(polygon.categories[0]).toBe("class 1")
    expect(polygon.getCenterWithOffset().X).toBeCloseTo(95.7143)
    expect(polygon.getCenterWithOffset().Y).toBeCloseTo(114.2857)
    expect(JSON.stringify(polygon.points)).toBe(
        "[[50,50],[50,100],[75,100],[75,120],[90,120],[90,150],[120,150],[120,50]]"
    )
    expect(polygon.type).toBe("polygon")

    expect(circle.color).not.toBeDefined
    expect(circle.categories.length).toBe(1)
    expect(circle.categories[0]).toBe("class 4")
    expect(circle.getCenterWithOffset().X).toBeCloseTo(285.7143)
    expect(circle.getCenterWithOffset().Y).toBeCloseTo(114.2857)
    expect(JSON.stringify(circle.centre)).toBe("[250,100]")
    expect(circle.radius).toBe(40)
    expect(circle.type).toBe("circle")

    expect(ellipse.color).toBe("yellow")
    expect(ellipse.categories.length).toBe(1)
    expect(ellipse.categories[0]).toBe("class 3")
    expect(ellipse.getCenterWithOffset().X).toBeCloseTo(400.0)
    expect(ellipse.getCenterWithOffset().Y).toBeCloseTo(171.4286)
    expect(JSON.stringify(ellipse.centre)).toBe("[350,150]")
    expect(ellipse.radiusX).toBe(60)
    expect(ellipse.radiusY).toBe(40)
    expect(ellipse.type).toBe("ellipse")
})

it("preserves auto-labeling metadata through canvas load and save", () => {
    const imageUrl = "test-auto-labeling-metadata.png"
    const rawShapes = [
        {
            id: "detected-dog",
            type: "rectangle",
            categories: ["dog"],
            points: [
                [10, 20],
                [80, 20],
                [80, 90],
                [10, 90],
            ],
            score: 0.9123,
            group_id: 4,
            attributes: { class_id: 16, occluded: false },
            auto_labeling_model: "rfdetr-nano",
        },
        {
            id: "segmented-dog",
            type: "polygon",
            categories: ["dog"],
            points: [
                [20, 30],
                [70, 30],
                [60, 80],
            ],
            score: 0.8765,
            group_id: "instance-5",
            attributes: { class_id: 16 },
            auto_labeling_model: "rfdetr-nano-seg",
        },
    ]
    const res = renderHook(useImageAnnotator)
    const view = render(
        <ImageAnnotator
            setHandles={res.result.current.setHandles}
            naturalSize
            imageUrl={imageUrl}
            shapes={rawShapes}
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

    const firstSave = res.result.current.annotator!.getShapes()!
    expect(firstSave).toMatchObject([
        {
            score: 0.9123,
            group_id: 4,
            attributes: { class_id: 16, occluded: false },
            auto_labeling_model: "rfdetr-nano",
        },
        {
            score: 0.8765,
            group_id: "instance-5",
            attributes: { class_id: 16 },
            auto_labeling_model: "rfdetr-nano-seg",
        },
    ])

    res.result.current.annotator!.setShapes(firstSave)
    expect(res.result.current.annotator!.getShapes()).toMatchObject([
        {
            score: 0.9123,
            group_id: 4,
            attributes: { class_id: 16, occluded: false },
            auto_labeling_model: "rfdetr-nano",
        },
        {
            score: 0.8765,
            group_id: "instance-5",
            attributes: { class_id: 16 },
            auto_labeling_model: "rfdetr-nano-seg",
        },
    ])
})

it("loads keypoints from canvas and LabelMe formats and preserves metadata", () => {
    const imageUrl = "test-keypoints.png"
    const rawShapes = [
        {
            id: "canvas-dot",
            type: "dot",
            categories: ["nose"],
            position: [10, 20],
            group_id: 1,
            visible: 2,
        },
        {
            id: "labelme-point",
            type: "point",
            categories: ["tail"],
            points: [[30, 40]],
            group_id: "second",
            visible: "occluded",
        },
        {
            id: "absent-point",
            type: "point",
            categories: ["ear"],
            points: [[50, 60]],
            visible: 0,
        },
    ]
    const res = renderHook(useImageAnnotator)
    const view = render(
        <ImageAnnotator
            setHandles={res.result.current.setHandles}
            naturalSize
            imageUrl={imageUrl}
            shapes={rawShapes}
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

    let dots = res.result.current.annotator!.getShapes() as Dot[]
    expect(dots).toHaveLength(2)
    expect(dots.map((dot) => dot.position)).toEqual([
        [10, 20],
        [30, 40],
    ])
    expect(dots.map((dot) => [dot.group_id, dot.visible])).toEqual([
        [1, 2],
        ["second", 1],
    ])
    expect(dots.map((dot) => dot.labelText())).toEqual(["nose · #1", "tail · #second · occluded"])

    res.result.current.annotator!.updateKeypointMetadata("canvas-dot", 3, 1)
    dots = res.result.current.annotator!.getShapes() as Dot[]
    expect([dots[0].group_id, dots[0].visible]).toEqual([3, 1])
    expect(dots[0].labelText()).toBe("nose · #3 · occluded")
})

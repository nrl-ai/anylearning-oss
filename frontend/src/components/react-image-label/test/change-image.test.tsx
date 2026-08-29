import "@testing-library/jest-dom"
import { fireEvent, render, renderHook } from "@testing-library/react"
import React from "react"

import { useImageAnnotator } from "../annotator/hook"
import { ImageAnnotator } from "../annotator/index"
import { Circle, Ellipse, Polygon, Rectangle, Shape } from "../base/types"
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

it("change image", () => {
    const imageUrl = "https://raw.githubusercontent.com/TaqBostan/content/main/Fruit.jpeg"
    const imageUrl2 = "https://svgjs.dev/docs/3.0/assets/images/logo-svg-js-01d-128.png"
    const _rect1 = new Rectangle(
        undefined,
        [
            [150, 50],
            [200, 50],
            [200, 100],
            [150, 100],
        ],
        ["class 3"],
        "#27f17640"
    )
    const _rect2 = new Rectangle(
        undefined,
        [
            [250, 150],
            [300, 150],
            [300, 200],
            [250, 200],
        ],
        ["class 1", "class 2"]
    )
    const _polygon = new Polygon(undefined, [
        [50, 50],
        [50, 100],
        [75, 100],
        [75, 120],
        [90, 120],
        [90, 150],
        [120, 150],
        [120, 50],
    ])
    const _circle = new Circle(undefined, [250, 100], 40, ["class 4"])
    const _ellipse = new Ellipse(undefined, [300, 150], 60, 40, ["class 3"])
    const res = renderHook(useImageAnnotator)
    const { setHandles } = res.result.current

    const Comp = function () {
        const [img, setImg] = React.useState(imageUrl)
        const [shapes, setShapes] = React.useState<Shape[]>([_rect1, _polygon])
        return (
            <>
                <input
                    id="change-img"
                    type="button"
                    onClick={() => {
                        setImg(imageUrl2)
                        setShapes([_rect2, _circle, _ellipse])
                    }}
                />
                <ImageAnnotator
                    setHandles={setHandles}
                    naturalSize={false}
                    imageUrl={img}
                    shapes={shapes}
                    width={700}
                    height={800}
                />
            </>
        )
    }

    const _annotator = render(<Comp />)
    const svg = _annotator.container.querySelector("svg")!
    const container = svg.parentElement as HTMLDivElement
    let imgs = svg.querySelectorAll("image")
    const _img1 = imgs[0] as SVGImageElement
    fireEvent(
        _img1,
        new CustomEvent("testEvent", {
            detail: { testRil: { naturalWidth: 800, naturalHeight: 700 } },
        })
    )

    //#region container, svg, and image
    expect(container.style.width).toBe("700px")
    expect(container.style.height).toBe("800px")
    // The whole URL, not just the file name -- see the annotator's onloaded.
    expect(container).toHaveAttribute("data-img", imageUrl)

    expect(svg).toHaveClass("il-svg")
    expect(svg).toHaveAttribute("height", "612.5")
    expect(svg).toHaveAttribute("width", "700")

    expect(_img1).toHaveAttribute("height", "100%")
    expect(_img1).toHaveAttribute("width", "100%")
    expect(_img1).toHaveAttribute("oncontextmenu", "return false")
    expect(_img1).toHaveAttribute("onmousedown", "return false")
    expect(_img1).toHaveAttribute("href", imageUrl)
    //#endregion

    //#region elements
    let polylines = svg.querySelectorAll("polyline")
    let discs = svg.querySelectorAll('circle[r="2"]')
    let ellipses = svg.querySelectorAll("ellipse")

    expect(polylines.length).toBe(4)
    expect(discs.length).toBe(0)
    expect(ellipses.length).toBe(0)
    //#endregion

    //#region rect1
    const rect1 = polylines[0]
    const rect1Shadow = polylines[1]

    expect(rect1).toHaveAttribute("fill", "#27f17640")
    expect(rect1).toHaveAttribute("points", "131.25,43.75 131.25,87.5 175,87.5 175,43.75 131.25,43.75")
    expect(rect1).toHaveAttribute("stroke", "#27f176")
    expect(rect1).toHaveAttribute("stroke-opacity", "0.7")
    expect(rect1).toHaveAttribute("stroke-width", "2")
    expect(rect1).toHaveAttribute("transform", "rotate(0,153.125,65.625)")
    expect(rect1).not.toHaveClass("il-hid")

    expect(rect1Shadow).toHaveAttribute("fill", "none")
    expect(rect1Shadow).toHaveAttribute("points", "131.25,43.75 131.25,87.5 175,87.5 175,43.75 131.25,43.75")
    expect(rect1Shadow).toHaveAttribute("stroke", "#000000")
    expect(rect1Shadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(rect1Shadow).toHaveAttribute("stroke-width", "4")
    expect(rect1Shadow).toHaveAttribute("transform", "rotate(0,153.125,65.625)")
    expect(rect1Shadow).not.toHaveClass("il-hid")
    //#endregion

    //#region polyline
    const polyline = polylines[2]
    const polylineShadow = polylines[3]

    expect(polyline).toHaveAttribute("fill", "#ffffff00")
    expect(polyline).toHaveAttribute(
        "points",
        "43.75,43.75 43.75,87.5 65.625,87.5 65.625,105 78.75,105 78.75,131.25 105,131.25 105,43.75 43.75,43.75"
    )
    expect(polyline).toHaveAttribute("stroke", "#ff0000")
    expect(polyline).toHaveAttribute("stroke-opacity", "0.7")
    expect(polyline).toHaveAttribute("stroke-width", "2")
    expect(polyline).not.toHaveAttribute("transform")

    expect(polylineShadow).toHaveAttribute("fill", "none")
    expect(polylineShadow).toHaveAttribute(
        "points",
        "43.75,43.75 43.75,87.5 65.625,87.5 65.625,105 78.75,105 78.75,131.25 105,131.25 105,43.75 43.75,43.75"
    )
    expect(polylineShadow).toHaveAttribute("stroke", "#000000")
    expect(polylineShadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(polylineShadow).toHaveAttribute("stroke-width", "4")
    expect(polylineShadow).not.toHaveAttribute("transform")
    //#endregion

    fireEvent(
        _annotator.container.querySelector("#change-img")!,
        new MouseEvent("click", { bubbles: true, cancelable: true })
    )
    imgs = svg.querySelectorAll("image")
    const _img2 = imgs[imgs.length - 1] as SVGImageElement
    fireEvent(
        _img2,
        new CustomEvent("testEvent", {
            detail: { testRil: { naturalWidth: 1400, naturalHeight: 900 } },
        })
    )

    //#region container, svg, and image
    expect(container.style.width).toBe("700px")
    expect(container.style.height).toBe("800px")
    expect(container).toHaveAttribute("data-img", imageUrl2)

    expect(svg).toHaveClass("il-svg")
    expect(svg).toHaveAttribute("height", "450")
    expect(svg).toHaveAttribute("width", "700")

    expect(_img2).toHaveAttribute("height", "100%")
    expect(_img2).toHaveAttribute("width", "100%")
    expect(_img2).toHaveAttribute("oncontextmenu", "return false")
    expect(_img2).toHaveAttribute("onmousedown", "return false")
    expect(_img2).toHaveAttribute("href", imageUrl2)
    //#endregion

    //#region elements
    polylines = svg.querySelectorAll("polyline")
    discs = svg.querySelectorAll('circle[r="2"]')
    ellipses = svg.querySelectorAll("ellipse")

    expect(polylines.length).toBe(2)
    expect(discs.length).toBe(0)
    expect(ellipses.length).toBe(4)
    //#endregion

    //#region rect2
    const rect2 = polylines[0]
    const rect2Shadow = polylines[1]

    expect(rect2).toHaveAttribute("fill", "#ffffff00")
    expect(rect2).toHaveAttribute("points", "125,75 125,100 150,100 150,75 125,75")
    expect(rect2).toHaveAttribute("stroke", "#ffffff")
    expect(rect2).toHaveAttribute("stroke-opacity", "0.7")
    expect(rect2).toHaveAttribute("stroke-width", "2")
    expect(rect2).toHaveAttribute("transform", "rotate(0,137.5,87.5)")

    expect(rect2Shadow).toHaveAttribute("fill", "none")
    expect(rect2Shadow).toHaveAttribute("points", "125,75 125,100 150,100 150,75 125,75")
    expect(rect2Shadow).toHaveAttribute("stroke", "#000000")
    expect(rect2Shadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(rect2Shadow).toHaveAttribute("stroke-width", "4")
    expect(rect2Shadow).toHaveAttribute("transform", "rotate(0,137.5,87.5)")
    //#endregion

    //#region circle
    const circle = ellipses[0]
    const circleShadow = ellipses[1]

    expect(circle).toHaveAttribute("fill", "#ffffff00")
    expect(circle).toHaveAttribute("cx", "125")
    expect(circle).toHaveAttribute("cy", "50")
    expect(circle).toHaveAttribute("rx", "20")
    expect(circle).toHaveAttribute("ry", "20")
    expect(circle).toHaveAttribute("stroke", "#ffffff")
    expect(circle).toHaveAttribute("stroke-opacity", "0.7")
    expect(circle).toHaveAttribute("stroke-width", "2")
    expect(circle).not.toHaveAttribute("transform")

    expect(circleShadow).toHaveAttribute("fill", "none")
    expect(circleShadow).toHaveAttribute("cx", "125")
    expect(circleShadow).toHaveAttribute("cy", "50")
    expect(circleShadow).toHaveAttribute("rx", "20")
    expect(circleShadow).toHaveAttribute("ry", "20")
    expect(circleShadow).toHaveAttribute("stroke", "#000000")
    expect(circleShadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(circleShadow).toHaveAttribute("stroke-width", "4")
    expect(circleShadow).not.toHaveAttribute("transform")
    //#endregion

    //#region ellipse
    const ellipse = ellipses[2]
    const ellipseShadow = ellipses[3]

    expect(ellipse).toHaveAttribute("fill", "#ffffff00")
    expect(ellipse).toHaveAttribute("cx", "150")
    expect(ellipse).toHaveAttribute("cy", "75")
    expect(ellipse).toHaveAttribute("rx", "30")
    expect(ellipse).toHaveAttribute("ry", "20")
    expect(ellipse).toHaveAttribute("stroke", "#ffffff")
    expect(ellipse).toHaveAttribute("stroke-opacity", "0.7")
    expect(ellipse).toHaveAttribute("stroke-width", "2")
    expect(ellipse).toHaveAttribute("transform", "rotate(0,150,75)")

    expect(ellipseShadow).toHaveAttribute("fill", "none")
    expect(ellipseShadow).toHaveAttribute("cx", "150")
    expect(ellipseShadow).toHaveAttribute("cy", "75")
    expect(ellipseShadow).toHaveAttribute("rx", "30")
    expect(ellipseShadow).toHaveAttribute("ry", "20")
    expect(ellipseShadow).toHaveAttribute("stroke", "#000000")
    expect(ellipseShadow).toHaveAttribute("stroke-opacity", "0.4")
    expect(ellipseShadow).toHaveAttribute("stroke-width", "4")
    expect(ellipseShadow).toHaveAttribute("transform", "rotate(0,150,75)")
    //#endregion
})

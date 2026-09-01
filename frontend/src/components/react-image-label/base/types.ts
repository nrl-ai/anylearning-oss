import { ArrayXY, Circle as Circ, Element, PointArray, Polyline, Rect, Text } from "@svgdotjs/svg.js"

import { generateUniqueId } from "@/lib/random"

import Util from "./util"

export type Point = { X: number; Y: number }

export type ShapeColor = string

export type StaticData = {
    width: number
    height: number
    ratio: number
    discRadius: number
    /** Stroke and discs can be hidden when not in edit/draw mode */
    hb: boolean | undefined
}

export abstract class Shape {
    id: string
    /** Neutral inference metadata retained across edit/save/reload. */
    score?: number | null
    group_id?: string | number | null
    attributes?: Record<string, string | number | boolean | null>
    auto_labeling_model?: string
    getCenterWithOffset = (): Point => ({ X: 0, Y: 0 })
    abstract type: string
    abstract labelPosition(): ArrayXY
    abstract getCenter(): ArrayXY
    abstract zoom(factor: number): void
    abstract output(ratio: number): Shape
    abstract centerChanged(newCenter: ArrayXY): void

    labelText(): string {
        return this.categories.join(", ")
    }

    constructor(
        id: string = generateUniqueId(),
        public categories: string[] = [],
        public phi: number = 0,
        public color?: ShapeColor
    ) {
        this.id = id
    }

    getOutput(ratio: number, svg: SVGSVGElement): Shape {
        const obj = this.output(ratio)
        const center = this.getCenter()
        const svgBox = svg.getBoundingClientRect()
        obj.id = this.id
        if (this.color) obj.color = this.color
        obj.phi = Math.round(this.phi)
        obj.score = this.score
        obj.group_id = this.group_id
        obj.attributes = this.attributes ? { ...this.attributes } : undefined
        obj.auto_labeling_model = this.auto_labeling_model
        obj.getCenterWithOffset = () => ({
            X: center[0] + svgBox.x,
            Y: center[1] + svgBox.y,
        })
        return obj
    }

    rotatePosition(): ArrayXY {
        const c = this.getCenter()
        const p = this.labelPosition()
        return [2 * c[0] - p[0], 2 * c[1] - p[1]]
    }
}

export class Dot extends Shape {
    type: string = "dot"

    constructor(
        id?: string,
        public position: ArrayXY = [-100, -100],
        public categories: string[] = [],
        public color?: ShapeColor,
        /** COCO keypoint instance. Null means the image's one implicit subject. */
        public group_id: string | number | null = null,
        /** COCO visibility: 1 occluded, 2 visible. */
        public visible: number = 2
    ) {
        super(id, categories, 0, color)
    }

    labelPosition(): ArrayXY {
        return [this.position[0], this.position[1] - 40]
    }
    labelText(): string {
        const instance = this.group_id === null ? "" : ` · #${this.group_id}`
        const visibility = this.visible === 1 ? " · occluded" : ""
        return `${this.categories.join(", ")}${instance}${visibility}`
    }
    getCenter(): ArrayXY {
        return this.position
    }
    zoom(factor: number): void {
        this.position = [this.position[0] * factor, this.position[1] * factor]
    }
    output(ratio: number) {
        return new Dot(
            undefined,
            [Math.round(this.position[0] / ratio), Math.round(this.position[1] / ratio)],
            this.categories,
            undefined,
            this.group_id,
            this.visible
        )
    }
    centerChanged(newPos: ArrayXY): void {
        this.position = newPos
    }
}

export interface IlElementExtra {
    categoriesPlain?: Text
    categoriesRect?: Rect
    shape: Shape
    shadow: Element
    discs: Circ[]
    editing: boolean
    connector?: Polyline
    categories?: string[]
}

export type ElementWithExtra = Element & IlElementExtra

export abstract class AngledShape extends Shape {
    constructor(
        id?: string,
        public points: ArrayXY[] | PointArray = [],
        public categories: string[] = [],
        public color?: ShapeColor
    ) {
        super(id, categories, 0, color)
    }

    labelPosition(): ArrayXY {
        const x =
            this.points
                .map((p) => p[0])
                .filter((x, i) => i < this.points.length - 1)
                .reduce((sum: number, num) => sum + num, 0) /
            (this.points.length - 1)
        const y = Math.min(...this.points.map((p) => p[1])) - 24
        return [x, y]
    }
    outPoints(ratio: number): ArrayXY[] {
        const center = this.getCenter()
        return this.points
            .filter((p, i) => i < this.points.length - 1)
            .map((p) => {
                const _p = Util.rotate([p[0] / ratio, p[1] / ratio], [center[0] / ratio, center[1] / ratio], this.phi)
                return [Math.round(_p[0]), Math.round(_p[1])]
            })
    }
    getCenter(): ArrayXY {
        if (this.points.length === 0) return [0, 0]
        const x =
            this.points
                .map((p) => p[0])
                .filter((x, i) => i < this.points.length - 1)
                .reduce((sum: number, num) => sum + num, 0) /
            (this.points.length - 1)
        const y = (Math.min(...this.points.map((p) => p[1])) + Math.max(...this.points.map((p) => p[1]))) / 2
        return [x, y]
    }
    centerChanged(newCenter: ArrayXY): void {
        const oldCenter = this.getCenter()
        const dx = newCenter[0] - oldCenter[0],
            dy = newCenter[1] - oldCenter[1]
        this.points.forEach((point) => {
            point[0] += dx
            point[1] += dy
        })
    }

    zoom(factor: number): void {
        this.points = this.points.map((p) => [p[0] * factor, p[1] * factor])
    }
}

export enum Color {
    BlackDisc = "#000",
    GreenDisc = "#009900",
    GrayDisc = "#a6a6a6",
    BlackLine = BlackDisc,
    GreenLine = GreenDisc,
    LightGreenLine = "#ccffcc",
    RedLine = "#f00",
    WhiteLine = "#fff",
    ShapeFill = "#ffffff00",
    Purple = "#663399",
}

export class Rectangle extends AngledShape {
    type: string = "rectangle"
    output(ratio: number) {
        return new Rectangle(undefined, this.outPoints(ratio), this.categories)
    }
}

export class Polygon extends AngledShape {
    type: string = "polygon"
    output(ratio: number) {
        return new Polygon(undefined, this.outPoints(ratio), this.categories)
    }
}

export abstract class RoundShape extends Shape {
    constructor(
        id?: string,
        public centre: ArrayXY = [0, 0],
        public categories: string[] = [],
        public phi: number = 0,
        public color?: ShapeColor
    ) {
        super(id, categories, phi, color)
    }
    abstract get width(): number
    abstract set width(w)
    abstract get height(): number
    abstract set height(h)
    getCenter(): ArrayXY {
        return this.centre
    }
    centerChanged(newCenter: ArrayXY): void {
        this.centre = newCenter
    }
}

export class Circle extends RoundShape {
    type: string = "circle"
    constructor(
        id?: string,
        public centre: ArrayXY = [0, 0],
        public radius: number = 0,
        public categories: string[] = [],
        public color?: ShapeColor
    ) {
        super(id, centre, categories, 0, color)
    }
    get width(): number {
        return 2 * this.radius
    }
    set width(w: number) {
        this.radius = w / 2
    }
    get height(): number {
        return 2 * this.radius
    }
    set height(h: number) {
        this.radius = h / 2
    }

    labelPosition(): ArrayXY {
        return [this.centre[0], this.centre[1] - this.radius - 24]
    }
    zoom(factor: number): void {
        this.centre = [this.centre[0] * factor, this.centre[1] * factor]
        this.radius *= factor
    }
    output = (ratio: number): Shape =>
        new Circle(
            undefined,
            [Math.round(this.centre[0] / ratio), Math.round(this.centre[1] / ratio)],
            Math.round(this.radius / ratio),
            this.categories
        )
}

export class Ellipse extends RoundShape {
    type: string = "ellipse"
    constructor(
        id?: string,
        public centre: ArrayXY = [0, 0],
        public radiusX: number = 0,
        public radiusY: number = 0,
        public categories: string[] = [],
        public phi: number = 0,
        public color?: ShapeColor
    ) {
        super(id, centre, categories, phi, color)
    }
    get width(): number {
        return 2 * this.radiusX
    }
    set width(w: number) {
        this.radiusX = w / 2
    }
    get height(): number {
        return 2 * this.radiusY
    }
    set height(h: number) {
        this.radiusY = h / 2
    }
    labelPosition(): ArrayXY {
        return [this.centre[0], this.centre[1] - this.radiusY - 24]
    }
    zoom(factor: number): void {
        this.centre = [this.centre[0] * factor, this.centre[1] * factor]
        this.radiusX *= factor
        this.radiusY *= factor
    }
    output = (ratio: number): Shape =>
        new Ellipse(
            undefined,
            [Math.round(this.centre[0] / ratio), Math.round(this.centre[1] / ratio)],
            Math.round(this.radiusX / ratio),
            Math.round(this.radiusY / ratio),
            this.categories
        )
}

import { Circle as Circ, Polyline, Rect, Text } from "@svgdotjs/svg.js"

import { ShapeBuilder } from "./ShapeBuilder"
import { AngledShape, Color, IlElementExtra } from "./types"

export class IlPolyline extends Polyline implements IlElementExtra {
    categoriesPlain?: Text
    categoriesRect?: Rect
    shape!: AngledShape
    shadow!: Polyline
    discs!: Circ[]
    hasConnector: boolean = false
    editing: boolean = false
}

export abstract class AngledBuilder<T extends AngledShape> extends ShapeBuilder<T> {
    element?: IlPolyline
    canHB = true
    abstract editShape_mm(event: MouseEvent): void

    createElement(shape: AngledShape): void {
        this.element = Object.assign(this.svg.polyline([]), {
            shape,
            shadow: this.svg.polyline([]),
            discs: [],
            hasConnector: false,
            editing: false,
        })
        this.element.fill(Color.ShapeFill)
        this.element.stroke({ color: Color.RedLine, width: 2, opacity: 0.7 })

        this.element.shadow.fill("none")
        this.element.shadow.stroke({
            color: Color.BlackLine,
            width: 4,
            opacity: 0.4,
        })
    }

    plotShape(): void {
        const shape = this.shape!
        shape.points.push([...shape.points[0]])
        this.processShape()
        shape.zoom(this.sd.ratio)
        this.createElement(shape)
        this.plotAngledShape()
    }

    plotAngledShape(): void {
        if (this.element) this.plot(this.element)
    }

    plot(polyline: IlPolyline): void {
        polyline.shadow.plot(polyline.shape.points)
        polyline.plot(polyline.shape.points)
    }

    editShape(): void {
        const polyline = this.element!
        polyline.discs?.forEach((disc) => disc.remove())
        polyline.discs = []

        if (polyline.shape.points) {
            polyline.shape.points.forEach((point, index, list) => {
                if (index < list.length - 1) {
                    const circle = this.drawDisc(point[0], point[1], 2, Color.BlackDisc)
                    this.element!.discs!.push(circle)
                }
            })
        }

        polyline.discs?.forEach((_disc, index) => {
            _disc
                .fill(Color.GreenDisc)
                .radius(this.sd.discRadius)
                .addClass("seg-point")
                .click((e: MouseEvent) => {
                    e.stopPropagation()
                })
                .mousedown((e: MouseEvent) => {
                    if (e.buttons === 1 && !e.ctrlKey && this.dragIndex === undefined) {
                        this.dragIndex = index
                        ;[this.movePath!, ...this.rotateArr].forEach((item) => item.remove())
                        this.svg.mousemove((e: MouseEvent) => this.editShape_mm(e))
                        e.stopPropagation()
                        // The release almost never lands back on this 3px disc,
                        // so without a window-level terminator dragIndex stayed
                        // set and no vertex could be dragged again.
                        this.captureRelease(() => this.editShape_mu())
                    }
                })
            _disc.mouseup(() => this.editShape_mu())
        })
    }

    editShape_mu() {
        if (this.dragIndex !== undefined) {
            this.dragIndex = undefined
            this.svg.off("mousemove")
            this.freeRelease()
            this.addMoveIcon()
            this.addRotateIcon()
        }
    }

    processShape() {}
}

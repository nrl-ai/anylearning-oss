import { Circle as Circ, Polyline, Rect, Text } from "@svgdotjs/svg.js"
import { ArrayXY } from "@svgdotjs/svg.js"

import { ShapeBuilder } from "../ShapeBuilder"
import { useSettingStore } from "../store"
import { Color, Dot, ElementWithExtra, IlElementExtra, ShapeColor } from "../types"

class IDot extends Polyline implements IlElementExtra {
    discs!: Circ[]
    classNames?: Text
    classNamesWrapper?: Rect
    shape!: Dot
    shadow!: Circ
    editing: boolean = false
    connector?: Polyline
}

export class DotBuilder extends ShapeBuilder<Dot> {
    shape?: Dot
    element?: IDot
    canRotate = false
    canHB = false
    newShape = () => new Dot()
    ofType<T>(shape: T): boolean {
        return shape instanceof Dot
    }

    createAutoLabelingElement(shape: Dot): void {
        const category_name = shape.categories?.[0]
        const is_add_point = category_name === "AUTOLABEL_ADD_POINT"
        const [x, y] = shape.position
        const polyline = this.svg
            .polyline(this.vertices(x, y, 8, 8))
            .fill(is_add_point ? Color.GreenLine : Color.RedLine)
            .stroke({
                color: is_add_point ? Color.GreenLine : Color.RedLine,
                width: 1,
                opacity: 0.0,
                dasharray: "3,3",
            })
            .opacity(0)
        const elem = Object.assign(polyline, {
            shape,
            shadow: this.drawDisc(x, y, 6, is_add_point ? Color.GreenLine : Color.RedLine).opacity(0.6),
            discs: [this.drawDisc(x, y, 4, is_add_point ? Color.GreenLine : Color.RedLine).opacity(0.6)],
            editing: false,
        })
        elem.discs[0].after(elem)
        this.element = elem
    }

    createLabelingElement(shape: Dot): void {
        const [x, y] = shape.position
        const polyline = this.svg
            .polyline(this.vertices(x, y, 16, 13))
            .fill(Color.ShapeFill)
            .stroke({
                color: Color.BlackLine,
                width: 1,
                opacity: 0.8,
                dasharray: "3,3",
            })
        const elem = Object.assign(polyline, {
            shape,
            shadow: this.drawDisc(x, y, 6, Color.BlackLine).opacity(0.4),
            discs: [this.drawDisc(x, y, 4, Color.RedLine).opacity(0.6)],
            editing: false,
        })
        elem.discs[0].after(elem)
        this.element = elem
    }

    createElement(shape: Dot): void {
        const category_name = shape.categories?.[0]
        if (category_name?.startsWith("AUTOLABEL_")) {
            this.createAutoLabelingElement(shape)
        } else {
            this.createLabelingElement(shape)
        }
    }

    applyColors(element: ElementWithExtra, color?: ShapeColor) {
        if (color) {
            element.discs[0].fill(color)
            element.shadow.fill(color)
            element.discs[0].fill(color)
        }
        const occluded =
            element.shape instanceof Dot &&
            element.shape.visible === 1 &&
            useSettingStore.getState().isDimOccludedKeypoints
        element.discs[0].opacity(occluded ? 0.3 : 0.6)
        element.shadow.opacity(occluded ? 0.2 : 0.4)
    }

    labeledStyle(element: ElementWithExtra, labeled: boolean) {
        if (labeled) {
            element.discs[0].fill(Color.WhiteLine)
        }
    }

    plotShape(): void {
        const shape = this.shape!
        shape.zoom(this.sd.ratio)
        this.createElement(shape)
    }

    startDraw(addDot: () => void): void {
        this.svg.click((event: MouseEvent) => this.drawClick(event, () => addDot()))
    }

    drawClick(event: MouseEvent, addDot: () => void) {
        if (event.ctrlKey || event.shiftKey || event.altKey) return
        const elem = this.element!
        if (this.element?.editing) {
            this.stopEdit()
            this.createElement(new Dot())
            this.drawClick(event, addDot)
        } else {
            elem.shape.position = [event.offsetX, event.offsetY]
            elem.discs[0].move(event.offsetX - 4, event.offsetY - 4)
            this.plot(elem)
            addDot()
        }
    }

    plot(elem: IDot): void {
        const [x, y] = elem.shape.position
        elem.shadow.move(x - 6, y - 6)
        elem.plot(this.vertices(x, y, 16, 13))
    }

    stopDraw(): void {
        this.svg.off("click")
    }

    editShape(): void {}

    stopEditShape(elem: IDot): void {
        const shape = elem.shape
        this.setOptions(elem, shape.categories, shape.color)
    }

    vertices(x: number, y: number, w: number, h: number): ArrayXY[] {
        return [
            [x - w, y - h],
            [x + w, y - h],
            [x + w, y + h],
            [x - w, y + h],
            [x - w, y - h],
        ]
    }
}

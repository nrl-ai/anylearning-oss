import { ArrayXY, Element, Path } from "@svgdotjs/svg.js"
import { Svg } from "react-svgdotjs"

import { Color, ElementWithExtra, Point, Shape, ShapeColor, StaticData } from "./types"
import Util, { annotationLabelText, isShowingLabels } from "./util"

/** The part of a MouseEvent that the gesture terminators actually read. */
export type PointerLike = { offsetX: number; offsetY: number }

export abstract class ShapeBuilder<T extends Shape> {
    static _svg: Svg
    static _sd: StaticData
    /**
     * The last pointer position seen over the canvas, in SVG coordinates.
     *
     * A gesture that ends off-canvas has no usable offsetX/offsetY -- the
     * browser reports those against whatever element the pointer happens to be
     * over -- so terminators fall back to the last position that really was on
     * the canvas. Director keeps this up to date.
     */
    static lastPointer: PointerLike = { offsetX: 0, offsetY: 0 }
    /** Fill opacity under the pointer. The resting value is 0.25; see applyColors. */
    static readonly HOVER_FILL_OPACITY = 0.45
    svg: Svg = ShapeBuilder._svg
    sd: StaticData = ShapeBuilder._sd
    private releaseHandler?: () => void
    abstract element?: ElementWithExtra
    abstract shape?: T
    /** Stroke and discs can be hidden when not in edit/draw mode by setting hideBorder=true */
    abstract canHB: boolean
    //#region drag
    private lastPoint?: Point
    private dragOrigin?: Point
    private moveIcon = (center: ArrayXY) => `M${center[0] + 11.3},${
        center[1]
    }l-4.6-4.6v2.4h-4.5v-4.5h2.4l-4.6,-4.6l-4.6,4.6h2.4v4.5h-4.5v-2.4l-4.6,4.6l4.6,4.6v-2.4h4.5v4.5h-2.4l4.6,4.6
  l4.6-4.6h-2.4v-4.5h4.5v2.4l4.6-4.6z`
    protected movePath?: Path
    //#endregion
    private rotateIcon = (center: ArrayXY) =>
        `M${center[0] + 5.2},${center[1] + 4.5}a7,7,0,1,1,0-8l-3,3h9v-9l-3,3a11+11,0,1,0,0+14z`
    protected rotateArr: Element[] = []
    protected canRotate = true
    drawing: boolean = false
    abstract plotShape(): void
    abstract createElement(shape: T): void
    abstract newShape(): T
    abstract startDraw(addShape: () => void): void
    abstract plot(element: ElementWithExtra): void
    abstract stopDraw(): void
    abstract editShape(): void
    dragIndex?: number

    /**
     * Ends the in-flight gesture when the button is released *anywhere*.
     *
     * Every gesture here starts on a mousedown over the canvas and ends on a
     * mouseup bound to the SVG -- or, for vertex drags, to a 3px disc the
     * pointer has usually already left. Release the button off that target
     * (past the edge of the image, over the side panel, outside the window) and
     * the terminator never runs, so the builder keeps its half-finished state.
     * The `!this.rectOrigin` / `!this.lastPoint` / `dragIndex === undefined`
     * guards on mousedown then reject every later gesture, and drawing or
     * selecting silently stops working until the screen is remounted.
     *
     * Terminators all no-op when their state is already clear, so it does not
     * matter whether the SVG handler or this one gets there first.
     */
    protected captureRelease(end: () => void) {
        this.freeRelease()
        const handler = () => {
            this.freeRelease()
            end()
        }
        this.releaseHandler = handler
        window.addEventListener("mouseup", handler)
    }

    /** Drops the window-level release listener. Safe when none is registered. */
    protected freeRelease() {
        if (this.releaseHandler) {
            window.removeEventListener("mouseup", this.releaseHandler)
            this.releaseHandler = undefined
        }
    }

    isAutoLabelingShape(categories: string[]) {
        return categories.some((category) => category.startsWith("AUTOLABEL_"))
    }

    drawDisc(x: number, y: number, radius: number, color: string) {
        return this.svg
            .circle(2 * radius)
            .fill(color)
            .move(x - radius, y - radius)
    }

    rotate(elem: ElementWithExtra = this.element!) {
        if (!this.canRotate) return
        const shape = elem.shape,
            center = shape.getCenter()
        const items = [elem, elem.shadow, elem.connector, ...elem.discs]
        if (elem.editing) items.push(...this.rotateArr)
        items.forEach((el) => el?.node.setAttribute("transform", `rotate(${shape.phi},${center[0]},${center[1]})`))
    }

    abstract ofType<S extends Shape>(shape: S): boolean

    basePlotShape() {
        const shape = this.shape!
        this.plotShape()
        this.rotate()
        this.setOptions(this.element!, shape.categories, shape.color)
    }

    labeledStyle(element: ElementWithExtra, labeled: boolean, color?: string) {
        element.stroke({
            color: color ? Util.removeOpacity(color) : labeled ? Color.WhiteLine : Color.RedLine,
        })
    }

    applyColors(element: ElementWithExtra, color?: ShapeColor, fillOpacity = 0.25) {
        element.fill(Color.ShapeFill)

        if (color) {
            // Translucent fill, solid outline. Filling with the raw category
            // colour made every box opaque, hiding the very thing being
            // labelled; 0.25 keeps the region obvious while the image shows
            // through.
            element.fill(Util.withOpacity(color as string, fillOpacity))
            element.stroke(Util.removeOpacity(color))
        }
    }

    /**
     * Brighten a shape while the pointer is over it.
     *
     * Overlapping masks are the case this exists for: where three polygons
     * cover the same pixels, nothing tells you which one you are about to
     * click, and the borderless display modes (`hb`) leave nothing to read at
     * all. Raising the fill answers that without a click.
     *
     * Not applied while drawing -- during a draw the pointer crosses shapes
     * constantly and flashing each one is noise, not information.
     */
    hoverHighlight(element: ElementWithExtra, labeled: boolean, color?: ShapeColor) {
        const paint = (opacity: number) => {
            if (labeled) this.applyColors(element, color, opacity)
            else element.fill(Color.ShapeFill)
        }

        // Rebound on every setOptions call, so clear the previous pair first:
        // a shape recoloured mid-session would otherwise accumulate handlers
        // closing over the colour it used to have.
        element.off("mouseover.hl").off("mouseout.hl")
        element.on("mouseover.hl", () => {
            if (!this.drawing) paint(ShapeBuilder.HOVER_FILL_OPACITY)
        })
        element.on("mouseout.hl", () => paint(0.25))
    }

    setOptions(element: ElementWithExtra, categories: string[], color?: ShapeColor) {
        const labeled = categories.length > 0 && !this.isAutoLabelingShape(categories)
        this.labeledStyle(element, labeled, color)

        if (labeled) {
            this.applyColors(element, color)
        }

        this.hoverHighlight(element, labeled, color)

        if (this.sd.hb && this.canHB) {
            ;[element.shadow, ...element.discs].forEach((el) => el.addClass("il-hid"))
            element.stroke({ width: 0 })
        }

        element.categories = categories
        element.categoriesPlain?.remove()
        element.categoriesRect?.remove()

        if (isShowingLabels(element.shape)) {
            this.drawLabel(element)
        }
    }

    drawLabel(element: ElementWithExtra) {
        const categories = element.categories
        const labeled = !!categories?.length && !this.isAutoLabelingShape(categories)
        if (element && labeled && categories) {
            const categoriesPlain = annotationLabelText(element.shape)
            if (!categoriesPlain) return
            const pos = element.shape.labelPosition()
            const paddingX = 5
            const paddingY = 2
            const borderColor = element.shape.color ? Util.removeOpacity(element.shape.color) : "#94a3b8"
            element.categoriesPlain = this.svg.plain(categoriesPlain).font({ size: 11, weight: 600 })
            const width = element.categoriesPlain.bbox().width
            const height = element.categoriesPlain.bbox().height
            element.categoriesRect = this.svg
                .rect(width + paddingX * 2, height + paddingY * 2)
                .radius(4)
                .move(pos[0] - width / 2 - paddingX, pos[1] + height / 4 - paddingY)
                .fill("#111827e6")
                .stroke({ color: borderColor, opacity: 0.95, width: 1 })
                .addClass("annotation-label-background")
            element.categoriesPlain.remove()
            element.categoriesPlain = this.svg
                .plain(categoriesPlain)
                .move(pos[0], pos[1])
                .font({ fill: "#f8fafc", size: 11, anchor: "middle", weight: 600 })
                .addClass("class-names annotation-label-text")
        }
    }

    removeLabel(element: ElementWithExtra) {
        if (element.categoriesPlain) element.categoriesPlain.remove()
        if (element.categoriesRect) element.categoriesRect.remove()
        element.categoriesPlain = undefined
        element.categoriesRect = undefined
    }

    removeElement() {
        const elem = this.element!
        ;[
            elem,
            elem.shadow,
            this.movePath,
            elem.categoriesPlain,
            elem.categoriesRect,
            elem.connector,
            ...elem.discs,
            ...this.rotateArr,
        ].forEach((el) => el?.remove())
        if (this.drawing) this.createElement(this.newShape())
    }

    addMoveIcon(): void {
        const str = this.moveIcon(this.element!.shape.getCenter())
        this.movePath = this.svg.path(str)
        this.element!.after(this.movePath)
        this.movePath
            .attr("class", "move-icon grabbable")
            .mousedown((ev: MouseEvent) => this.drag_md(ev))
            .on("contextmenu", (ev: any) => {
                ev.preventDefault()
                this.element!.node.dispatchEvent!(new Event("contextmenu", ev))
            })
    }

    addRotateIcon(): void {
        if (!this.canRotate) return
        const position = this.element!.shape.rotatePosition()
        const str = this.rotateIcon(position)
        const path = this.svg.path(str)
        const bg = this.svg
            .circle(24)
            .move(position[0] - 12, position[1] - 12)
            .fill(Color.ShapeFill)
        this.rotateArr = [path, bg]
        path.attr("class", "rot-icon grabbable")
        bg.attr("class", "grabbable").after(path)
        this.rotateArr.forEach((item) =>
            item.mousedown((ev: MouseEvent) => this.rotate_md(ev)).click((event: MouseEvent) => event.stopPropagation())
        )
        this.rotate()
    }

    initDrag() {
        this.element!.addClass("grabbable")
            .click((event: MouseEvent) => {
                event.stopPropagation()
            })
            .mousedown((event: MouseEvent) => this.drag_md(event))
        this.addMoveIcon()
    }

    drag_md(e: MouseEvent) {
        if (e.buttons === 1 && !e.ctrlKey && !this.lastPoint) {
            this.lastPoint = { X: e.offsetX, Y: e.offsetY }
            this.dragOrigin = { X: e.offsetX, Y: e.offsetY }
            ;[this.movePath!, ...this.rotateArr].forEach((item) => item.remove())
            this.svg.mousemove((e: MouseEvent) => this.drag_mm(e)).mouseup(() => this.drag_mu())
            this.captureRelease(() => this.drag_mu())
            e.stopPropagation()
        }
    }

    drag_mm(e: MouseEvent) {
        if (this.lastPoint) {
            if (e.buttons !== 1) return this.drag_mu()
            if (!this.element) return
            const dx = e.offsetX - this.lastPoint.X,
                dy = e.offsetY - this.lastPoint.Y,
                center = this.element.shape.getCenter()
            ;[this.element, this.element.shadow, this.element.connector, ...this.element.discs].forEach((disc) => {
                disc?.cx(disc.cx() + dx).cy(disc.cy() + dy)
            })
            this.element.shape.centerChanged([center[0] + dx, center[1] + dy])
            this.rotate()
            this.lastPoint = { X: e.offsetX, Y: e.offsetY }
        }
    }

    drag_mu() {
        if (this.lastPoint) {
            // Clear the gesture before the early return: a drag whose element
            // has gone away still has to release the pointer state, or the
            // `!this.lastPoint` guard blocks every later drag.
            this.lastPoint = undefined
            this.dragOrigin = undefined
            this.svg.off("mousemove").off("mouseup")
            this.freeRelease()
            if (!this.element) return
            this.addMoveIcon()
            this.addRotateIcon()
        }
    }

    stopDrag() {
        this.freeRelease()
        if (this.element) {
            this.element.removeClass("grabbable").off("click").off("mousedown")
            if (this.movePath) this.movePath.remove()
            this.rotateArr.forEach((item) => item.remove())
            this.rotateArr = []
            this.lastPoint = undefined
            this.dragOrigin = undefined
            this.movePath = undefined
        }
    }

    rotate_md(e: MouseEvent) {
        if (e.buttons === 1 && !e.ctrlKey) {
            this.svg.mousemove((e: MouseEvent) => this.rotate_mm(e)).mouseup(() => this.rotate_mu())
            this.captureRelease(() => this.rotate_mu())
            e.stopPropagation()
        }
    }

    rotate_mm(e: MouseEvent) {
        if (e.buttons !== 1) return this.rotate_mu()
        if (!this.element) return
        const center = this.element.shape.getCenter(),
            vector: ArrayXY = [e.offsetX - center[0], e.offsetY - center[1]]
        this.element.shape.phi = (Math.atan2(-vector[0], vector[1]) * 180) / Math.PI
        this.rotate()
    }

    rotate_mu() {
        this.svg.off("mousemove").off("mouseup")
        this.freeRelease()
    }

    zoom(elem: ElementWithExtra, factor: number): void {
        elem.shape.zoom(factor)
        this.plot(elem)
        elem.discs?.forEach((_disc) => _disc.cx(_disc.cx() * factor).cy(_disc.cy() * factor))
        elem.connector?.plot(elem.connector.array().map((p) => [p[0] * factor, p[1] * factor] as ArrayXY))
        if (elem.editing) {
            if (this.rotateArr.length > 0) {
                const position = elem.shape.rotatePosition()
                const [path, bg] = this.rotateArr
                ;(path as Path).plot(this.rotateIcon(position))
                bg.move(position[0] - 12, position[1] - 12)
            }
            this.movePath?.plot(this.moveIcon(elem.shape.getCenter()))
        } else this.setOptions(elem, elem.shape.categories, elem.shape.color)
        this.rotate(elem)
    }

    stopEdit() {
        if (this.element && this.element.editing) {
            this.element.editing = false
            this.stopDrag()
            this.stopEditShape(this.element)
            if (this.drawing) this.createElement(this.newShape())
        }
    }

    stopEditShape(elem: ElementWithExtra): void {
        const shape = elem.shape
        elem.discs?.forEach((_disc) => {
            _disc.remove()
        })
        elem.discs = []
        this.setOptions(elem, shape.categories, shape.color)
    }

    edit(): void {
        if (this.isAutoLabelingShape(this.element!.shape.categories)) return
        const elem = this.element!
        elem.editing = true
        if (elem.categoriesPlain) elem.categoriesPlain.clear()
        if (elem.categoriesRect) elem.categoriesRect.remove()
        if (this.canHB) {
            ;[elem.shadow, ...elem.discs].forEach((el) => el.removeClass("il-hid"))
            elem.stroke({ width: 2 })
        }
        this.initDrag()
        this.addRotateIcon()
        this.editShape()
        this.rotate()
    }
}

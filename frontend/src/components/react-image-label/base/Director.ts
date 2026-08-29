import { Svg } from "react-svgdotjs"

import { ShapeBuilder } from "./ShapeBuilder"
import CircleBuilder from "./builders/CircleBuilder"
import { DotBuilder } from "./builders/DotBuilder"
import EllipseBuilder from "./builders/EllipseBuilder"
import PolygonBuilder from "./builders/PolygonBuilder"
import RectangleBuilder from "./builders/RectangleBuilder"
import { useSettingStore } from "./store"
import { Dot, ElementWithExtra, Point, Shape, ShapeColor, StaticData } from "./types"
import { isShowingLabels } from "./util"

export class Director {
    static instance?: Director
    static onAdded: ((shape: Shape) => any) | undefined
    static onContextMenu: ((shape: Shape) => any) | undefined
    static onSelected: ((shape: Shape) => any) | undefined
    builders: ShapeBuilder<Shape>[]
    elements: ElementWithExtra[] = []
    origin?: Point
    panningWithMiddle = false
    private panReleaseHandler?: () => void
    // Wheel events are coalesced into one zoom pass per animation frame; see
    // mousewheel(). pendingZoom accumulates multiplicatively.
    pendingZoom = 1
    zoomAnchor?: Point
    zoomFrame?: number
    editable: boolean

    constructor(
        public svg: Svg,
        public container: HTMLDivElement
    ) {
        this.builders = [
            new PolygonBuilder(),
            new RectangleBuilder(),
            new CircleBuilder(),
            new EllipseBuilder(),
            new DotBuilder(),
        ]
        this.editable = true

        useSettingStore.subscribe(() => this.refreshAnnotationDisplay())
    }

    getBuilder<T extends Shape>(shape: T): ShapeBuilder<T> {
        const builder = this.builders.find((b) => b.ofType(shape))! as ShapeBuilder<T>
        builder.shape = shape
        return builder
    }

    stopEdit = (): void => this.builders.filter((b) => b.element?.editing).forEach((b) => b.stopEdit())

    edit(id: string): void {
        this.stopEdit()
        const elem = this.getElement(id)
        const builder = this.getBuilder(elem.shape)
        builder.element = elem
        builder.edit()
    }

    zoom(factor: number) {
        // TODO: check this
        // const builderInDraw = this.builders.find((b) => b.drawing)
        // if (builderInDraw?.element?.shape.id === 0) builderInDraw.zoom(builderInDraw.element, factor)
        this.elements.forEach((elem) => {
            this.getBuilder(elem.shape).zoom(elem, factor)
        })
    }

    getElement = (id: string) => this.elements.find((p) => p.shape.id === id)!

    setOptions(element: ElementWithExtra, categories: string[], color?: string) {
        this.getBuilder(element.shape).setOptions(element, categories, color)
    }

    plot(shapes: Shape[]): void {
        shapes.forEach((shape) => {
            this.getBuilder(shape).basePlotShape()
            this.addShape(shape, false)
        })
    }

    startDraw(shape: Shape): void {
        const builder = this.getBuilder(shape)
        builder.drawing = true
        builder.createElement(shape)
        builder.startDraw(() => this.addShape(shape))
    }

    stopDraw(): void {
        const builder = this.builders.find((b) => b.drawing)
        if (builder) {
            builder.stopDraw()
            builder.drawing = false
        }
    }

    addShape(shape: Shape, isNew: boolean = true) {
        const builder = this.getBuilder(shape)
        if (!builder.element) return
        const id = builder.element.shape.id
        this.elements.push(builder.element)
        builder.element.node.addEventListener(
            "contextmenu",
            (ev: MouseEvent) => {
                ev.preventDefault()
                const elem = this.elements.find((p) => p.shape.id === id)!
                Director.onContextMenu?.(elem.shape)
                return false
            },
            false
        )
        builder.element.node.onclick = (e: MouseEvent) => {
            if (!this.editable) {
                return
            }

            const elem = this.elements.find((p) => p.shape.id === id)!
            if (!e.ctrlKey && !elem.editing) {
                this.edit(id)
                Director.onSelected?.(builder.element!.shape)
            }
            e.stopPropagation()
        }
        if (isNew) {
            if (!builder.element!.editing) builder.edit()
            Director.onAdded?.(builder.element.shape)
            Director.onSelected?.(builder.element.shape)
        }

        if (isShowingLabels(builder.element.shape)) {
            builder.drawLabel(builder.element)
        }
    }

    updateCategories(id: string, categories: string[], color?: ShapeColor) {
        const elem = this.getElement(id)
        if (!elem) return
        elem.shape.categories = categories
        if (color !== undefined) elem.shape.color = color
        const builder = this.getBuilder(elem.shape)
        if (!elem.editing) builder.setOptions(elem, categories, elem.shape.color)
    }

    updateKeypointMetadata(id: string, groupId: string | number | null, visible: number) {
        const elem = this.getElement(id)
        if (!elem || !(elem.shape instanceof Dot)) return
        elem.shape.group_id = groupId
        elem.shape.visible = visible
        this.getBuilder(elem.shape).setOptions(elem, elem.shape.categories, elem.shape.color)
    }

    removeById(id: string) {
        this.stopEdit()
        const elem = this.getElement(id)
        const builder = this.getBuilder(elem.shape)
        builder.element = elem
        builder.removeElement()
        this.elements.splice(this.elements.indexOf(elem), 1)
    }

    remove() {
        if (this.builders.filter((b) => b.element?.editing).length > 0) {
            const id = this.builders.filter((b) => b.element?.editing)[0].element!.shape!.id
            this.removeById(id)
        }
    }

    /** Ends a pan when the button is released anywhere, not just over the canvas. */
    private capturePanRelease() {
        this.freePanRelease()
        const handler = () => this.drag_mu()
        this.panReleaseHandler = handler
        window.addEventListener("mouseup", handler)
    }

    private freePanRelease() {
        if (this.panReleaseHandler) {
            window.removeEventListener("mouseup", this.panReleaseHandler)
            this.panReleaseHandler = undefined
        }
    }

    drag_md(container: HTMLDivElement, e: MouseEvent) {
        // Pan on Ctrl+drag (as before) or on middle-mouse drag. Middle-drag is
        // the conventional canvas pan and, unlike Ctrl+left, cannot be confused
        // with drawing or with selecting an existing shape.
        const isPanGesture = (e.buttons === 1 && e.ctrlKey) || e.buttons === 4
        if (isPanGesture && !this.origin) {
            e.preventDefault()
            this.origin = { X: e.clientX, Y: e.clientY }
            this.panningWithMiddle = e.buttons === 4
            container.onmousemove = (event: MouseEvent) => this.drag_mm(event)
            container.onmouseup = () => this.drag_mu()
            // Releasing outside the canvas used to leave origin set and
            // onmousemove bound, so the canvas then panned on a plain,
            // button-less mouse move.
            this.capturePanRelease()
        }
    }

    drag_mm(e: MouseEvent) {
        if (this.origin) {
            const parent = this.container
            parent.scrollLeft = parent.scrollLeft - e.clientX + this.origin.X
            parent.scrollTop = parent.scrollTop - e.clientY + this.origin.Y
            this.origin = { X: e.clientX, Y: e.clientY }
            // Middle-drag pans until the button is released; Ctrl+drag ends as
            // soon as Ctrl is let go, which is the previous behaviour.
            if (!this.panningWithMiddle && !e.ctrlKey) this.drag_mu()
        }
    }

    drag_mu() {
        this.freePanRelease()
        if (this.origin) {
            this.container.onmousemove = null
            this.container.onmouseup = null
            this.origin = undefined
            this.panningWithMiddle = false
        }
    }

    getShapes = () => this.elements.map((el) => el.shape.getOutput(ShapeBuilder._sd.ratio, this.svg.node))
    findShape = (id: string) => this.elements.find((el) => el.shape.id === id)!.shape

    static init(svg: Svg, sd: StaticData, container: HTMLDivElement) {
        svg.size(sd.width * sd.ratio, sd.height * sd.ratio)
        ShapeBuilder._svg = svg
        ShapeBuilder._sd = sd
        const instance = (Director.instance = new Director(svg, container))
        // Tracked natively rather than through SVG.js so the builders' routine
        // `off("mousemove")` cannot detach it. Gesture terminators that fire
        // off-canvas read this instead of the release event's own offsets,
        // which would be measured against whatever element is under the cursor.
        svg.node.addEventListener("mousemove", (event: MouseEvent) => {
            ShapeBuilder.lastPointer = { offsetX: event.offsetX, offsetY: event.offsetY }
        })
        container.onmousedown = (event: MouseEvent) => instance.drag_md(container, event)
        container.onwheel = (event: WheelEvent) => instance.mousewheel(event)
        container.onclick = (e: MouseEvent) =>
            !instance.builders.some((b) => b.drawing) && !e.ctrlKey && instance.stopEdit()
    }

    static setActions(
        onAdded?: (shape: Shape) => any,
        onContextMenu?: (shape: Shape) => any,
        onSelected?: (shape: Shape) => any
    ) {
        const hoc = (fun?: (shape: Shape) => any) => (shape: Shape) =>
            fun?.(shape.getOutput(ShapeBuilder._sd.ratio, ShapeBuilder._svg.node))
        Director.onAdded = hoc(onAdded)
        Director.onContextMenu = hoc(onContextMenu)
        Director.onSelected = hoc(onSelected)
    }

    clear() {
        // Drop any zoom queued for the next frame -- it would otherwise run
        // against the cleared SVG after teardown.
        if (this.zoomFrame !== undefined) {
            cancelAnimationFrame(this.zoomFrame)
            this.zoomFrame = undefined
        }
        this.pendingZoom = 1
        this.builders.forEach((b) => {
            b.stopDraw()
            b.stopEdit()
        })
        ShapeBuilder._svg?.clear()
        this.elements = []
        this.builders = []
        Director.instance = undefined
    }

    clearShapes() {
        // Remove all shapes one by one
        while (this.elements.length > 0) {
            this.removeById(this.elements[0].shape.id)
        }
    }

    mousewheel(e: WheelEvent) {
        // Plain wheel zooms. Requiring Ctrl matched a document viewer, but this
        // is a canvas -- every comparable labelling tool zooms on bare wheel,
        // and there is nothing else for the wheel to do here.
        e.preventDefault()

        // Scale with the wheel delta instead of a fixed step, so the small
        // deltas a trackpad emits produce smooth zoom rather than 25% jumps.
        // A full notch (|deltaY| >= 100) still lands on exactly the historical
        // 1.25 / 0.8, so a mouse wheel behaves as it always did.
        const intensity = Math.min(Math.abs(e.deltaY), 100) / 100
        const step = 1 + 0.25 * intensity
        const scale = e.deltaY > 0 ? 1 / step : step

        // Applying the zoom inline made cost scale with *event rate*: zoom()
        // re-plots every shape and every handle, and a trackpad emits 60-120
        // wheel events a second. With a few hundred points that is a full
        // re-plot per event and the canvas stutters badly.
        //
        // Multiply the pending factors together and apply them once per frame
        // instead. Zoom is multiplicative, so collapsing N events into one pass
        // lands on exactly the same scale -- the intermediate steps were never
        // painted anyway, since the browser cannot paint between same-frame
        // events. Work becomes O(shapes) per *frame* rather than per event.
        this.pendingZoom *= scale
        this.zoomAnchor = { X: e.pageX, Y: e.pageY }
        if (this.zoomFrame !== undefined) return
        this.zoomFrame = requestAnimationFrame(() => {
            this.zoomFrame = undefined
            const factor = this.pendingZoom
            this.pendingZoom = 1
            const anchor = this.zoomAnchor
            if (factor === 1 || !anchor) return

            const parent = this.container
            const { scrollLeft, scrollTop } = parent
            this.setSizeAndRatio(factor, true)
            this.zoom(factor)
            // Keep the point under the cursor fixed while scaling.
            parent.scrollLeft = Math.min(
                Math.max(scrollLeft * factor + (factor - 1) * (anchor.X - parent.offsetLeft), 0),
                parent.scrollWidth - parent.clientWidth
            )
            parent.scrollTop = Math.min(
                Math.max(scrollTop * factor + (factor - 1) * (anchor.Y - parent.offsetTop), 0),
                parent.scrollHeight - parent.clientHeight
            )
        })
    }

    setSizeAndRatio(factor: number, relative: boolean) {
        const ratio = relative ? ShapeBuilder._sd.ratio * factor : factor
        factor = ratio / ShapeBuilder._sd.ratio
        ShapeBuilder._sd.ratio = ratio
        ShapeBuilder._svg.size(ShapeBuilder._sd.width * ratio, ShapeBuilder._sd.height * ratio)
        return factor
    }

    showLabels() {
        this.elements.forEach((element) => {
            if (isShowingLabels(element.shape)) {
                this.getBuilder(element.shape).drawLabel(element)
            }
        })
    }

    hideLabels() {
        this.elements.forEach((element) => {
            this.getBuilder(element.shape).removeLabel(element)
        })
    }

    /** Apply label and keypoint visibility preferences to existing shapes. */
    refreshAnnotationDisplay() {
        this.elements.forEach((element) => {
            const builder = this.getBuilder(element.shape)
            builder.setOptions(element, element.shape.categories, element.shape.color)
        })
    }
}

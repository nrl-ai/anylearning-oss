import React, { FC, useCallback, useEffect, useRef } from "react"
import { Svg, SvgContainer, useSvgContainer } from "react-svgdotjs"

import "./index.css"

import { Director } from "../base/Director"
import { Circle, Dot, Ellipse, Polygon, Rectangle, Shape, ShapeColor } from "../base/types"
import Util from "../base/util"
import { AnnotatorHandles } from "./hook"

const getDirector = () => Director.instance!
const normalizeShapeId = (id: unknown): string | undefined => (id === undefined || id === null ? undefined : String(id))
const normalizeVisibility = (value: unknown): number => {
    if (value === 0 || value === "0") return 0
    if (value === 1 || value === false || value === "1") return 1
    if (typeof value === "string" && ["occluded", "hidden", "false"].includes(value.toLowerCase())) return 1
    return 2
}

const retainInferenceMetadata = <T extends Shape>(target: T, source: any): T => {
    target.score = source.score
    target.group_id = source.group_id
    target.attributes = source.attributes ? { ...source.attributes } : undefined
    target.auto_labeling_model = source.auto_labeling_model
    return target
}

const ImageAnnotator: FC<ImageAnnotatorProps> = (props) => {
    const { setHandles, svgContainer } = useSvgContainer()
    const propsRef = useRef(props)
    propsRef.current = props

    const drawShapes = useCallback((shapes?: Shape[] | any[] | null) => {
        const director = getDirector()
        if (!shapes || !director) return
        const rectangles = shapes
            .filter((s) => s instanceof Rectangle || s.type === "rectangle")
            .map((s) =>
                retainInferenceMetadata(new Rectangle(normalizeShapeId(s.id), [...s.points], s.categories, s.color), s)
            )
        const polygons = shapes
            .filter((s) => s instanceof Polygon || s.type === "polygon")
            .map((s) =>
                retainInferenceMetadata(new Polygon(normalizeShapeId(s.id), [...s.points], s.categories, s.color), s)
            )
        const circles = shapes
            .filter((s) => s instanceof Circle || s.type === "circle")
            .map((s) => new Circle(normalizeShapeId(s.id), s.centre, s.radius, s.categories, s.color))
        const ellipses = shapes
            .filter((s) => s instanceof Ellipse || s.type === "ellipse")
            .map(
                (s) =>
                    new Ellipse(
                        normalizeShapeId(s.id),
                        s.centre,
                        s.radiusX,
                        s.radiusY,
                        s.categories,
                        s.phi || 0,
                        s.color
                    )
            )
        const dots = shapes
            .filter((s) => s instanceof Dot || s.type === "dot" || s.type === "point")
            // COCO v=0 means the landmark was not labelled. It is absence,
            // not a dot at the origin or a third visual state.
            .filter((s) => normalizeVisibility(s.visible ?? s.visibility) !== 0)
            .map(
                (s) =>
                    new Dot(
                        normalizeShapeId(s.id),
                        s.position ?? s.points?.[0],
                        s.categories,
                        s.color,
                        s.group_id ?? null,
                        normalizeVisibility(s.visible ?? s.visibility)
                    )
            )
        if (rectangles.length > 0) director.plot(rectangles)
        if (polygons.length > 0) director.plot(polygons)
        if (circles.length > 0) director.plot(circles)
        if (ellipses.length > 0) director.plot(ellipses)
        if (dots.length > 0) director.plot(dots)
    }, [])

    const zoom = useCallback((factor: number, relative: boolean = true) => {
        const director = getDirector()
        factor = director.setSizeAndRatio(factor, relative)
        director.zoom(factor)
    }, [])

    const stopAll = useCallback(() => {
        const director = getDirector()
        director?.stopDraw()
        director?.stopEdit()
    }, [])

    const getHandles = useCallback(
        () => ({
            drawRectangle() {
                stopAll()
                getDirector()?.startDraw(new Rectangle())
            },
            drawPolygon() {
                stopAll()
                getDirector()?.startDraw(new Polygon())
            },
            drawCircle() {
                stopAll()
                getDirector()?.startDraw(new Circle())
            },
            drawEllipse() {
                stopAll()
                getDirector()?.startDraw(new Ellipse())
            },
            drawDot() {
                stopAll()
                getDirector()?.startDraw(new Dot())
            },
            stop: stopAll,
            stopEdit: () => getDirector().stopEdit(),
            edit: (id: string) => getDirector().edit(id),
            delete: (id: string) => getDirector().removeById(id),
            updateCategories: (id: string, categories: string[], color?: ShapeColor) =>
                getDirector().updateCategories(id, categories, color),
            updateKeypointMetadata: (id: string, groupId: string | number | null, visible: number) =>
                getDirector().updateKeypointMetadata(id, groupId, visible),
            zoom,
            getShapes: getDirector().getShapes,
            setShapes: (shapes: Shape[] | null) => {
                getDirector()?.clearShapes()
                if (shapes) {
                    drawShapes(shapes)
                }
            },
            setEditable: (value: boolean) => {
                if (getDirector()?.editable) {
                    getDirector().editable = value
                }
            },
        }),
        [drawShapes, stopAll, zoom]
    )

    const onload = useCallback(
        (svg: Svg, container: HTMLDivElement, imageUrl: string) => {
            const onloaded = (ev: any) => {
                const props = propsRef.current

                if (!ev?.target || !svg.node.innerHTML) return
                const imageNode = ev.target as SVGImageElement,
                    target = ev.detail?.testRil || imageNode

                // Compare the *whole* URL, not Util.fileName(). AnyLearning
                // serves images from /projects/{p}/data_items/{id}/download, so
                // the last path segment is "download" for every image in a
                // project -- fileName() collapsed them all to one string and the
                // guard could not tell two images apart.
                if (container.getAttribute("data-img") !== imageUrl) {
                    // A newer image has been requested since this load started;
                    // drop this stale result and leave the DOM to the newer one.
                    return
                }

                // Keep the most recently added <image> and drop any earlier
                // ones. Deliberately not compared against ev.target: under
                // StrictMode the effect mounts twice, so ev.target can be a
                // detached node from the first pass, and removing "everything
                // that isn't ev.target" deleted the live image -- which is why
                // annotations rendered over a blank canvas.
                const images = Array.from(svg.node.children).filter((child) => child.tagName.toLowerCase() === "image")
                images.slice(0, -1).forEach((stale) => stale.remove())
                const naturalWidth = target.naturalWidth,
                    naturalHeight = target.naturalHeight
                let maxWidth = props.width,
                    maxHeight = props.height,
                    ratio = 1
                svg.addClass("il-svg")

                // Measure the *parent* (the space actually available) rather
                // than trusting props. The container's own width is set from
                // props just below, so measuring the container fed its own
                // value back in and the canvas could end up wider than the
                // area it lives in -- a 1000px image inside a 950px pane, with
                // the right-hand slice clipped by overflow:hidden.
                const parent = container.parentElement
                const availableWidth = parent?.clientWidth || props.width || naturalWidth
                const availableHeight = parent?.clientHeight || props.height || naturalHeight

                Object.assign(container.style, {
                    width: availableWidth + "px",
                    height: availableHeight + "px",
                    overflow: "hidden",
                    backgroundColor: "var(--surface-sunken, #e6e6e6)",
                    // Centre the image in the pane instead of pinning it to the
                    // top-left with dead space beside it.
                    //
                    // "safe" matters once the user zooms in: plain centring
                    // overflows equally in both directions, and the part that
                    // spills past the top/left edge cannot be reached by
                    // scrolling -- so panning could never bring it back into
                    // view. "safe center" centres while the image fits and
                    // falls back to start-alignment once it does not, keeping
                    // every part of a zoomed image reachable.
                    display: "flex",
                    alignItems: "safe center",
                    justifyContent: "safe center",
                })
                if (!props.naturalSize) {
                    maxWidth = availableWidth
                    maxHeight = availableHeight
                    // Scale to *contain*: the smaller of the two ratios keeps the
                    // whole image visible and never overflows either axis.
                    ratio = Math.min(availableWidth / naturalWidth, availableHeight / naturalHeight)
                }
                const statics = {
                    width: naturalWidth,
                    height: naturalHeight,
                    ratio,
                    discRadius: props.discRadius || 3,
                    hb: props.hideBorder,
                }
                Director.init(svg, statics, container)
                drawShapes(props.shapes || null)
                props.setHandles({ ...getHandles(), container } satisfies AnnotatorHandles)
                props.onReady?.({ ...getHandles(), container } satisfies AnnotatorHandles)
            }
            // Store the full URL so the staleness check above is exact.
            container.setAttribute("data-img", imageUrl)
            const image = svg
                .image(imageUrl, onloaded)
                .size("100%", "100%")
                .attr("onmousedown", "return false")
                .attr("oncontextmenu", "return false")
            image.node.addEventListener("testEvent", onloaded)
        },
        [drawShapes, getHandles]
    )

    useEffect(() => {
        Director.setActions(props.onAdded, props.onContextMenu, props.onSelected)
        return () => Director.setActions(undefined, undefined)
    }, [props.onAdded, props.onContextMenu, props.onSelected])

    useEffect(() => {
        if (!svgContainer) {
            return
        }

        const onblur = () => svgContainer?.container.classList.remove("grabbable")
        const onkeydown = (e: KeyboardEvent) =>
            e.key === "Control" && svgContainer?.container.classList.add("grabbable")
        const keyup = (e: KeyboardEvent) => {
            if (e.key === "Control") onblur()
            if (e.key === "Delete") Director.instance?.remove()
            if (e.key === "Escape") Director.instance?.stopEdit()
        }

        window.addEventListener("keydown", onkeydown)
        window.addEventListener("keyup", keyup)
        window.addEventListener("blur", onblur)

        return () => {
            window.removeEventListener("keydown", onkeydown)
            window.removeEventListener("keyup", keyup)
            window.removeEventListener("blur", onblur)
        }
    }, [svgContainer])

    // Held in a ref so the effect below does not re-run when this callback's
    // identity changes. Its cleanup calls Director.clear(), which wipes the
    // entire SVG -- image included -- so an identity change was enough to erase
    // the photo while later drawShapes() calls re-added the annotations,
    // leaving boxes floating over a blank canvas.
    const onloadRef = useRef(onload)
    useEffect(() => {
        onloadRef.current = onload
    }, [onload])

    useEffect(() => {
        if (svgContainer && props.imageUrl) {
            onloadRef.current(svgContainer.svg, svgContainer.container, props.imageUrl)
        }
        return () => {
            Director.instance?.clear()
        }
        // Only a new container or a genuinely different image warrants a reload.
    }, [svgContainer, props.imageUrl])

    return <SvgContainer setHandles={setHandles} />
}

export { ImageAnnotator }

export interface ImageAnnotatorProps {
    onReady?: (annotator: AnnotatorHandles) => any
    onAdded?: (shape: Shape) => any
    onSelected?: (shape: Shape) => any
    onContextMenu?: (shape: Shape) => any
    imageUrl?: string
    shapes?: Shape[] | any[] | null
    naturalSize?: boolean
    width?: number
    height?: number
    discRadius?: number
    hideBorder?: boolean
    showLabels?: boolean
    setHandles: (handles: AnnotatorHandles) => void
}

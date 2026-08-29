import { ArrayXY } from "@svgdotjs/svg.js"

import { useSettingStore } from "./store"
import type { Shape } from "./types"

export default class Util {
    static maxId: number = 0
    static ArrayXYSum = (...array: ArrayXY[]): ArrayXY =>
        array.reduce((sum: ArrayXY, xy) => [sum[0] + xy[0], sum[1] + xy[1]], [0, 0])

    static rotate = (pos: ArrayXY, center: ArrayXY, teta: number): ArrayXY => {
        const dx = pos[0] - center[0],
            dy = pos[1] - center[1],
            fi = (teta * Math.PI) / 180
        return [dx * Math.cos(fi) - dy * Math.sin(fi) + center[0], dx * Math.sin(fi) + dy * Math.cos(fi) + center[1]]
    }

    static fileName = (url: string | null) => url?.substring(url.lastIndexOf("/") + 1) || ""

    static parseColor = (str: string) => {
        let colors: number[]
        if (str[0] === "#") {
            str = str.substring(1, str.length >= 7 ? 7 : 4)
            const collen = str.length / 3
            const fact = [17, 1][collen - 1]
            colors = [
                Math.round(parseInt(str.substring(0, collen), 16) * fact),
                Math.round(parseInt(str.substring(collen, 2 * collen), 16) * fact),
                Math.round(parseInt(str.substring(2 * collen, 3 * collen), 16) * fact),
            ]
        } else
            colors = str
                .split("(")[1]
                .split(")")[0]
                .split(",")
                .map((x) => +x)
        if (colors.length < 4) colors.push(1)
        return colors
    }

    /** Same hue, fixed alpha. Used for shape fills so the image stays visible
        underneath the annotation -- a fully opaque fill hides exactly the pixels
        the user is trying to label. */
    static withOpacity = (color: string, alpha: number) => {
        if (color[0] === "#" || color.startsWith("rgb")) {
            const rgba = Util.parseColor(color)
            // 8-digit hex rather than rgba(): shape colours are stored and
            // compared as hex throughout, so emitting the same notation keeps
            // fills comparable to the colour they were derived from.
            const hex = (n: number) => Math.round(n).toString(16).padStart(2, "0")
            return `#${hex(rgba[0])}${hex(rgba[1])}${hex(rgba[2])}${hex(alpha * 255)}`
        } else return color
    }

    static removeOpacity = (color: string) => {
        if (color[0] === "#" || color.startsWith("rgb")) {
            const rgba = Util.parseColor(color)
            return `rgb(${rgba[0]},${rgba[1]},${rgba[2]})`
        } else return color
    }
}

type KeypointShape = Shape & {
    group_id: string | number | null
    visible: number
}

export function isShowingLabels(shape?: Shape) {
    const settings = useSettingStore.getState()
    if (settings.isShowLabels) return true
    if (shape?.type !== "dot") return false
    const point = shape as KeypointShape
    return (
        (settings.isShowKeypointInstances && point.group_id !== null) ||
        (settings.isShowKeypointVisibility && point.visible === 1)
    )
}

/** Build only the pieces of an annotation label the user asked to see. */
export function annotationLabelText(shape: Shape): string {
    const settings = useSettingStore.getState()
    if (shape.type !== "dot") return settings.isShowLabels ? shape.labelText() : ""

    const point = shape as KeypointShape
    const parts: string[] = []
    if (settings.isShowLabels) parts.push(point.categories.join(", "))
    if (settings.isShowKeypointInstances && point.group_id !== null) parts.push(`#${point.group_id}`)
    if (settings.isShowKeypointVisibility && point.visible === 1) parts.push("occluded")
    return parts.filter(Boolean).join(" · ")
}

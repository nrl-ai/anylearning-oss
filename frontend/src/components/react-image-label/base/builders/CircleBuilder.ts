import { ArrayXY } from "@svgdotjs/svg.js"

import { RoundBuilder } from "../RoundBuilder"
import { Circle } from "../types"

export default class CircleBuilder extends RoundBuilder<Circle> {
    shape?: Circle
    canRotate = false
    newShape = () => new Circle()
    ofType<T>(shape: T): boolean {
        return shape instanceof Circle
    }

    calculateRadius(offset: ArrayXY): ArrayXY {
        const radius = Math.sqrt(Math.pow(this.origin!.X - offset[0], 2) + Math.pow(this.origin!.Y - offset[1], 2)) / 2
        return [radius, radius]
    }

    calculateDifferent(offset: ArrayXY): ArrayXY {
        const xSign = Math.sign(offset[0] - this.origin!.X),
            ySign = Math.sign(offset[1] - this.origin!.Y)
        const diff = Math.min(Math.abs(this.origin!.X - offset[0]), Math.abs(this.origin!.Y - offset[1]))
        return [diff * xSign, diff * ySign]
    }
}

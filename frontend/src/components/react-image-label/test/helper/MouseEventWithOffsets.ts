interface MouseEventWithOffsets extends MouseEventInit {
    pageX?: number
    pageY?: number
    offsetX?: number
    offsetY?: number
    x?: number
    y?: number
    deltaY?: number
}

export class FakeMouseEvent extends MouseEvent {
    constructor(type: string, values: MouseEventWithOffsets) {
        const { pageX, pageY, offsetX, offsetY, x, y, deltaY, ...mouseValues } = values
        super(type, mouseValues)

        Object.defineProperties(this, {
            offsetX: { configurable: true, value: offsetX ?? 0 },
            offsetY: { configurable: true, value: offsetY ?? 0 },
            pageX: { configurable: true, value: pageX ?? 0 },
            pageY: { configurable: true, value: pageY ?? 0 },
            x: { configurable: true, value: x ?? 0 },
            y: { configurable: true, value: y ?? 0 },
            deltaY: { configurable: true, value: deltaY ?? 0 },
        })
    }
}

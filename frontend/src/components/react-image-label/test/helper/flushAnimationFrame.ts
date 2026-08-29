import { act } from "@testing-library/react"

/**
 * Wheel zoom is coalesced into a single pass per animation frame (see
 * Director.mousewheel), so the size change lands one frame after the event
 * rather than during it. Await this between firing a wheel event and asserting
 * on the result.
 *
 * Wrapped in act() because the frame can flush React state updates.
 */
export const flushAnimationFrame = async (): Promise<void> => {
    await act(async () => {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
    })
}

import { useEffect, useState } from "react"

/**
 * False during the prerender and the first client render, true afterwards.
 *
 * The app ships as a static export, so every page is prerendered at build time
 * with no request behind it: `useSearchParams()` is empty, `window` does not
 * exist, and nothing has been fetched. A component that renders real state
 * straight away therefore produces different HTML on the client, which React
 * reports as a hydration mismatch and the user sees as a flash of the wrong
 * content -- a header confidently reading "No project selected" for a moment
 * before the project name replaces it.
 *
 * Gate that content on this hook and render a placeholder until it turns true:
 * the two passes then agree, and the transition is placeholder -> content
 * rather than wrong-answer -> content.
 */
export function useMounted(): boolean {
    const [mounted, setMounted] = useState(false)
    useEffect(() => setMounted(true), [])
    return mounted
}

export default useMounted

import axios, { AxiosRequestConfig } from "axios"

/**
 * The one HTTP client for the backend.
 *
 * Every request needs a bearer token matching `webview.token`, which is
 * generated per window and only exists on `window.pywebview` once the desktop
 * shell has injected it. That header was previously written out by hand at
 * roughly thirty call sites, several of which had quietly forgotten it; an
 * interceptor reads the token at request time so it is always current and
 * always present.
 */
export const api = axios.create()

api.interceptors.request.use((config) => {
    const token = typeof window !== "undefined" ? window.pywebview?.token : undefined
    if (token) config.headers.Authorization = `Bearer ${token}`
    return config
})

/** Appends the token as a query param, for URLs the browser fetches itself. */
export function withToken(url: string): string {
    const token = typeof window !== "undefined" ? window.pywebview?.token : undefined
    if (!token) return url
    return `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`
}

/** Query function body: unwraps `data` so hooks deal in domain types. */
export async function getJson<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const { data } = await api.get<T>(url, config)
    return data
}

/**
 * Like `getJson`, but a 404 resolves to null instead of throwing.
 *
 * Several endpoints use 404 to mean "nothing yet" rather than "error" -- the
 * last training session of a project that has never trained, for instance --
 * and React Query would otherwise retry and surface those as failures.
 */
export async function getJsonOrNull<T>(url: string): Promise<T | null> {
    try {
        return await getJson<T>(url)
    } catch (error) {
        if (axios.isAxiosError(error) && error.response?.status === 404) return null
        throw error
    }
}

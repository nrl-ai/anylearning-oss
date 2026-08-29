import axios, { AxiosInstance } from "axios"

interface AxiosConfig {
    baseURL?: string
    timeout?: number
    headers?: Record<string, string>
}

const DEFAULT_TIMEOUT = 10_000 // 10 seconds

const createPublicHttpClient = (config: AxiosConfig): AxiosInstance => {
    const instance = axios.create({
        baseURL: config.baseURL,
        timeout: config.timeout || DEFAULT_TIMEOUT,
        headers: config.headers || {},
    })
    return instance
}

// TODO: impl privateClient and pass baseURL
export const publicClient = createPublicHttpClient({
    baseURL: "/",
})

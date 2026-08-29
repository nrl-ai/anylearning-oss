"use client"

import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ThemeProvider } from "next-themes"
import React from "react"

import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"

/**
 * Defaults tuned for a local desktop app rather than a website.
 *
 * The backend is a process on the same machine, so latency is negligible and a
 * short staleTime costs nothing -- but refetching on every window focus made
 * the whole UI flicker each time the user came back from a file dialog. Retries
 * are off for the same reason: a request to localhost does not fail
 * intermittently, so a retry only delays showing the real error.
 */
const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: 5_000,
            refetchOnWindowFocus: false,
            retry: false,
        },
    },
})

function AppProvider({ children }: { children?: React.ReactNode }) {
    return (
        <QueryClientProvider client={queryClient}>
            {/* Dark by default: the app is an image-inspection tool, and a
                graphite ground is the right surround for judging photographs.
                enableSystem stays on so "System" remains a choice in Settings —
                it is just no longer what a first run lands on. */}
            <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
                <TooltipProvider delayDuration={250} skipDelayDuration={100}>
                    {children}
                </TooltipProvider>
                <Toaster position="top-right" richColors closeButton />
            </ThemeProvider>
        </QueryClientProvider>
    )
}

export default AppProvider

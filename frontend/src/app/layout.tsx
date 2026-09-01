import type { Metadata } from "next"
import { Inter, JetBrains_Mono, Space_Grotesk } from "next/font/google"
import NextTopLoader from "nextjs-toploader"

import AppProvider from "@/app/AppProvider"
import { DesktopChrome } from "@/components/layout/desktop-chrome"

import "./globals.css"

// next/font downloads these at build time and serves them from the static
// export, so the desktop app never reaches the network for a font at runtime.
// Three faces for three jobs — see frontend/DESIGN.md.
const inter = Inter({
    subsets: ["latin"],
    display: "swap",
    variable: "--font-inter",
})

const spaceGrotesk = Space_Grotesk({
    subsets: ["latin"],
    display: "swap",
    weight: ["500", "600", "700"],
    variable: "--font-space-grotesk",
})

const jetbrainsMono = JetBrains_Mono({
    subsets: ["latin"],
    display: "swap",
    weight: ["400", "500", "600"],
    variable: "--font-jetbrains-mono",
})

export const metadata: Metadata = {
    title: "AnyLearning",
    description: "Build your own AI models!",
    icons: {
        icon: "/anylearning-512x512.png",
        apple: "/anylearning-512x512.png",
    },
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        // suppressHydrationWarning belongs on <html>, not <body>: the theme is
        // applied to the <html> element on the client (className="light" plus a
        // color-scheme style), which the server never renders. React treated the
        // whole tree as mismatched and re-rendered it after hydration, which is
        // visible as a flash on load.
        <html
            lang="en"
            data-scroll-behavior="smooth"
            className={`${inter.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable}`}
            suppressHydrationWarning
        >
            <head>
                <link rel="icon" href="/anylearning-512x512.png" />
                <link rel="apple-touch-icon" href="/anylearning-512x512.png" />
            </head>
            <body className="overflow-hidden">
                <NextTopLoader color="var(--mark)" height={2} showSpinner={false} shadow={false} />
                {/* The window's ground. Frameless fallback renderers also use
                    it for transparent rounded corners; see globals.css. */}
                <div className="window-shell">
                    <AppProvider>{children}</AppProvider>
                </div>
                {/* Custom window controls where the platform frame is replaced.
                    Outside the provider tree and last in the body because they
                    belong to the window rather than to any route. Linux's
                    compositor-owned frame makes this render nothing. */}
                <DesktopChrome />
            </body>
        </html>
    )
}

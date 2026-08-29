const path = require("node:path")

/** @type {import('next').NextConfig} */
const nextConfig = {
    output: process.env.NODE_ENV === "development" ? "standalone" : "export",
    transpilePackages: ["@svgdotjs/svg.js"],
    images: {
        remotePatterns: [
            {
                protocol: "https",
                hostname: "utfs.io",
            },
            {
                protocol: "https",
                hostname: "api.anylearning.nrl.ai",
            },
        ],
        unoptimized: true,
    },
    // Turbopack needs no alias config: it reads `paths` and `baseUrl` straight
    // from tsconfig.json, where "@/*" -> "./src/*" is already declared. Declaring
    // an empty config tells Next the webpack block below is a deliberate
    // fallback rather than an unmigrated leftover -- without it, Next 16 refuses
    // to start under Turbopack when a webpack config is present.
    turbopack: {},
    // The floating dev badge sits on top of the app's own bottom-left chrome
    // and lands in every screenshot taken of a dev build. Compile and runtime
    // errors are still surfaced -- this only hides the indicator.
    devIndicators: false,
    webpack(config) {
        // TS 7's native compiler resolves the path mapping directly. Mirror it
        // for the supported webpack fallback used by restricted CI runners.
        config.resolve.alias["@"] = path.resolve(__dirname, "src")
        return config
    },
}

if (process.env.NODE_ENV === "development") {
    nextConfig.rewrites = async () => {
        return [
            {
                source: "/api/:path*",
                destination: "http://127.0.0.1:5678/api/:path*",
            },
        ]
    }
}
module.exports = nextConfig

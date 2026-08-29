import nextra from "nextra";

const withNextra = nextra({
  theme: "nextra-theme-docs",
  themeConfig: "./theme.config.tsx",
  staticImage: true,
  latex: true,
  flexsearch: {
    codeblocks: false,
  },
  defaultShowCopyCode: true,
});

export default withNextra({
  reactStrictMode: true,
  eslint: {
    // Eslint behaves weirdly in this monorepo.
    ignoreDuringBuilds: true,
  },
  // Docs must also work from the packaged/offline server. Next's on-demand
  // image optimiser can hang there and leave screenshots permanently blank;
  // serve the versioned static assets directly instead.
  images: {
    unoptimized: true,
  },
});

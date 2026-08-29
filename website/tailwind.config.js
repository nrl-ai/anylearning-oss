/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./theme.config.tsx",
  ],
  theme: {
    extend: {
      // Every colour resolves to a token in styles/tokens.css, so the site and
      // the desktop app cannot drift apart. Adding a raw hex here defeats it.
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        surface: "var(--surface)",
        "surface-sunken": "var(--surface-sunken)",
        muted: "var(--muted)",
        "muted-foreground": "var(--muted-foreground)",
        line: "var(--border)",
        mark: {
          DEFAULT: "var(--mark)",
          ink: "var(--mark-ink)",
          strong: "var(--mark-strong)",
          soft: "var(--mark-soft)",
          border: "var(--mark-border)",
        },
        ok: { DEFAULT: "var(--ok)", surface: "var(--ok-surface)" },
        run: { DEFAULT: "var(--run)", surface: "var(--run-surface)" },
        fail: { DEFAULT: "var(--fail)", surface: "var(--fail-surface)" },
        class: {
          1: "var(--class-1)",
          2: "var(--class-2)",
          3: "var(--class-3)",
          4: "var(--class-4)",
          5: "var(--class-5)",
          6: "var(--class-6)",
        },
      },
      fontFamily: {
        display: ["var(--font-display)", "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      borderRadius: {
        DEFAULT: "var(--radius)",
        md: "var(--radius)",
        lg: "calc(var(--radius) + 2px)",
        xl: "calc(var(--radius) + 6px)",
      },
      boxShadow: {
        // Hairline-quiet: in this system elevation is a surface lightness step,
        // and a shadow only separates a true overlay.
        xs: "0 1px 2px oklch(0.2 0.02 255 / 0.05)",
        sm: "0 1px 2px oklch(0.2 0.02 255 / 0.06)",
        md: "0 2px 6px oklch(0.2 0.02 255 / 0.07)",
        lg: "0 8px 24px oklch(0.2 0.02 255 / 0.10)",
      },
    },
  },
  plugins: [],
  darkMode: "class",
};

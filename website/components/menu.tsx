import { AnimatePresence, motion } from "framer-motion";
import React from "react";

import { Wordmark } from "./brand/logo";
import DownloadIcon from "./icons/download";
import GitHubMark from "./icons/github-mark";

// The marketing shell: neutral chrome, the mark used only to point at the page
// you are on and to draw focus rings.
const links = [
  { key: "home", href: "/", label: "Home" },
  { key: "docs", href: "/docs", label: "Documentation" },
  {
    key: "github",
    href: "https://github.com/nrl-ai/anylearning-oss",
    label: "GitHub",
    external: true,
  },
  { key: "download", href: "/download", label: "Download" },
];

const focusRing =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark focus-visible:ring-offset-2 focus-visible:ring-offset-background";

const Menu = ({ activePage }: { activePage?: string }) => {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    // The bar spans the window, but its contents sit in the same max-w-6xl px-6
    // column the page uses. Before this the header rendered raw, so the logo sat
    // hard against the window edge while the headline below it started 88px in.
    <header className="border-b border-line">
      <div className="mx-auto w-full max-w-6xl px-6">
        <div className="flex h-16 items-center justify-between gap-4">
          <a
            href="/"
            aria-label="AnyLearning home"
            className={`rounded text-foreground ${focusRing}`}
          >
            <Wordmark />
          </a>

          <div className="flex items-center gap-2">
            {/* Desktop nav */}
            <nav className="hidden lg:flex lg:items-center lg:gap-1" aria-label="Main">
              {links.map((link) => {
                const isActive = activePage === link.key;
                return (
                  <a
                    key={link.key}
                    href={link.href}
                    target={link.external ? "_blank" : undefined}
                    rel={link.external ? "noopener noreferrer" : undefined}
                    aria-current={isActive ? "page" : undefined}
                    className={`relative rounded px-3 py-2 text-sm transition-colors ${focusRing} ${
                      isActive
                        ? "font-medium text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {link.key === "github" && (
                      <GitHubMark className="mr-1.5 inline-block h-4 w-4 align-[-0.125em]" />
                    )}
                    {link.key === "download" && (
                      <DownloadIcon className="mr-1.5 inline-block h-4 w-4 align-[-0.125em]" />
                    )}
                    {link.label}
                    {isActive && (
                      <span
                        aria-hidden="true"
                        className="absolute inset-x-3 -bottom-[13px] h-0.5 rounded-full bg-mark"
                      />
                    )}
                  </a>
                );
              })}
            </nav>

            {/* Mobile menu button */}
            <motion.button
              type="button"
              onClick={() => setIsOpen(!isOpen)}
              aria-expanded={isOpen}
              aria-label={isOpen ? "Close menu" : "Open menu"}
              className={`rounded-md border border-line p-2 text-foreground lg:hidden ${focusRing}`}
              whileTap={{ scale: 0.95 }}
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                {isOpen ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                ) : (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                )}
              </svg>
            </motion.button>
          </div>
        </div>

        {/* Mobile nav: an overlay, so this is one of the few places a shadow is
            the right tool rather than a surface step. */}
        <AnimatePresence>
          {isOpen && (
            <motion.nav
              className="lg:hidden"
              aria-label="Main"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
            >
              <div className="mb-3 flex flex-col overflow-hidden rounded-lg border border-line bg-surface py-1 shadow-lg">
                {links.map((link) => {
                  const isActive = activePage === link.key;
                  return (
                    <motion.a
                      key={link.key}
                      href={link.href}
                      target={link.external ? "_blank" : undefined}
                      rel={link.external ? "noopener noreferrer" : undefined}
                      aria-current={isActive ? "page" : undefined}
                      className={`relative px-4 py-2.5 text-sm transition-colors hover:bg-muted ${focusRing} ${
                        isActive
                          ? "font-medium text-foreground"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                      whileHover={{ x: 2 }}
                      transition={{ duration: 0.2 }}
                    >
                      {isActive && (
                        <span
                          aria-hidden="true"
                          className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-mark"
                        />
                      )}
                      <span className="inline-flex items-center">
                        {link.key === "github" && <GitHubMark className="mr-2 h-4 w-4" />}
                        {link.key === "download" && <DownloadIcon className="mr-2 h-4 w-4" />}
                        {link.label}
                      </span>
                    </motion.a>
                  );
                })}
              </div>
            </motion.nav>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
};

export default Menu;

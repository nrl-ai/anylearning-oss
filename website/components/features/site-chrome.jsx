import React from "react";

import GitHubMark from "../icons/github-mark";

/** Keep marketing pages aligned with the documentation's fixed light theme. */
export function ThemeSync() {
  React.useEffect(() => {
    document.documentElement.classList.remove("dark");
    document.documentElement.style.colorScheme = "light";
  }, []);

  return null;
}

const footerLink =
  "inline-flex items-center rounded text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark focus-visible:ring-offset-2 focus-visible:ring-offset-background";

export default function SiteFooter() {
  return (
    <footer className="border-t border-line bg-surface-sunken">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-8 text-sm">
        <p className="text-muted-foreground">
          © <span className="font-mono tabular">{new Date().getFullYear()}</span> AnyLearning Team
        </p>
        <div className="flex items-center gap-6">
          <a href="/docs" className={footerLink}>
            <svg
              className="mr-2 h-4 w-4"
              fill="currentColor"
              viewBox="0 0 20 20"
              aria-hidden="true"
            >
              <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z"></path>
            </svg>
            Documentation
          </a>
          <a
            href="https://github.com/nrl-ai/anylearning-oss"
            target="_blank"
            rel="noopener noreferrer"
            className={footerLink}
          >
            <GitHubMark className="mr-2 h-4 w-4" />
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}

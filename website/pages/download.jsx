import React, { useState, useEffect } from "react";
import Head from "next/head";
import { FaApple, FaWindows, FaLinux } from "react-icons/fa";
import Menu from "../components/menu";
import SiteFooter, { ThemeSync } from "../components/features/site-chrome";

const DownloadIcon = () => (
  <svg
    className="mr-2 h-4 w-4"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
    aria-hidden="true"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
    />
  </svg>
);

const CheckIcon = () => (
  <svg
    className="mr-2 mt-0.5 h-4 w-4 shrink-0 text-muted-foreground"
    fill="currentColor"
    viewBox="0 0 20 20"
    aria-hidden="true"
  >
    <path
      fillRule="evenodd"
      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
      clipRule="evenodd"
    />
  </svg>
);

const Badge = ({ tone, children }) => (
  <span
    className={`rounded-full border px-2.5 py-1 text-xs font-medium ${
      tone === "mark"
        ? "border-mark-border bg-mark-soft text-mark"
        : "border-line bg-muted text-muted-foreground"
    }`}
  >
    {children}
  </span>
);

/**
 * One platform. The download itself is the page's whole job, so it is the only
 * thing here that carries the mark.
 */
const PlatformCard = ({ icon, name, requirement, note, warning, version, size, href, alternate, pending }) => {
  const available = Boolean(href);

  return (
  <div className="flex flex-col rounded-lg border border-line bg-surface p-6">
    <div className="mb-5 flex min-h-[1.75rem] items-center justify-between gap-3">
      <div className="flex items-center gap-2">
        <span className="text-lg text-muted-foreground" aria-hidden="true">
          {icon}
        </span>
        <h2 className="font-display text-lg font-semibold tracking-tight">{name}</h2>
      </div>
      {/* The release list is fetched, so no badge is claimed until it lands. */}
      {!pending && (
        <Badge tone={available ? "mark" : "neutral"}>
          {available ? "Latest" : "Coming soon"}
        </Badge>
      )}
    </div>

    <p className="text-sm text-muted-foreground">{requirement}</p>

    {/* The size is worth stating plainly. 26 ships its pretrained weights, so
        the download is gigabytes rather than megabytes, and finding that out
        from a progress bar is a worse way to learn it. */}
    <dl className="mt-4 flex items-baseline gap-x-5 gap-y-1 border-t border-line pt-4 text-sm">
      <div className="flex items-baseline gap-2">
        <dt className="text-muted-foreground">Version</dt>
        <dd className="font-mono tabular text-foreground">{version || "—"}</dd>
      </div>
      {size && (
        <div className="flex items-baseline gap-2">
          <dt className="text-muted-foreground">Size</dt>
          <dd className="font-mono tabular text-foreground">{size}</dd>
        </div>
      )}
    </dl>

    <ul className="mt-4 flex-grow space-y-2 text-sm text-muted-foreground">
      <li className="flex items-start">
        <CheckIcon />
        {note}
      </li>
    </ul>

    {warning && (
      <p className="mt-4 rounded-md border border-line bg-muted px-3 py-2 text-xs leading-relaxed text-foreground">
        {warning}
      </p>
    )}

    {available ? (
      <a
        target="_blank"
        rel="noopener noreferrer"
        href={href}
        className="mt-6 inline-flex w-full items-center justify-center rounded-md bg-mark px-5 py-2.5 text-sm font-medium text-mark-ink transition-colors hover:bg-mark-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      >
        <DownloadIcon />
        Download for {name}
      </a>
    ) : (
      <button
        type="button"
        disabled
        className="mt-6 inline-flex w-full cursor-not-allowed items-center justify-center rounded-md border border-line bg-muted px-5 py-2.5 text-sm font-medium text-muted-foreground"
      >
        <DownloadIcon />
        {pending ? `Download for ${name}` : "Coming soon"}
      </button>
    )}

    {/* A second format where one exists. macOS ships both a disk image and a
        zipped .app: the image is the familiar drag-to-Applications flow, the
        zip is what 24 shipped and what some people would rather have. */}
    {available && alternate && (
      <a
        href={alternate.href}
        target="_blank"
        rel="noopener noreferrer"
        className="mt-3 text-center text-xs text-muted-foreground underline underline-offset-4 hover:text-mark focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark"
      >
        {alternate.label}
        {alternate.size ? ` (${alternate.size})` : ""}
      </a>
    )}
  </div>
  );
};

/**
 * The releases this page offers, newest first.
 *
 * Which of them is actually downloadable is read from `check-for-update.json`,
 * not decided here: a release with no entry in the feed — or an entry whose
 * platform URLs are all null — renders as "Coming soon". Shipping AnyLearning
 * 26 is therefore a change to that file, not to this page.
 */
const RELEASES = [
  {
    key: "anylearning_oss",
    name: "AnyLearning OSS",
    summary:
      "The complete Apache-2.0 application, including local labeling and training, with no account or activation key.",
    whatsNew: { href: "https://github.com/nrl-ai/anylearning-oss/releases", label: "GitHub releases" },
  },
];

const PLATFORMS = [
  {
    id: "macos",
    icon: <FaApple />,
    // 11.0 is what the built binary reports as its minimum, not a guess:
    // "runs on macOS 11.0 (arm64) or higher".
    name: "macOS",
    requirement: "macOS 11 or later",
    note: "Native Apple Silicon support",
    // The .app on its own, for anyone who would rather unzip than mount.
    alternate: { id: "macos_zip", label: "or download the .app as a zip" },
  },
  {
    id: "windows",
    icon: <FaWindows />,
    name: "Windows",
    requirement: "Windows 10 or later, 64-bit",
    note: "Native x64 support",
  },
  {
    id: "linux",
    icon: <FaLinux />,
    name: "Linux",
    requirement: "glibc 2.31 or later (Ubuntu 20.04+)",
    note: "Native x64 support",
  },
];

const ARCHIVE_PLATFORMS = [
  { id: "macos", label: "macOS", icon: <FaApple /> },
  { id: "macos_zip", label: "macOS ZIP", icon: <FaApple /> },
  { id: "windows", label: "Windows", icon: <FaWindows /> },
  { id: "linux", label: "Linux", icon: <FaLinux /> },
];

const compareVersions = (left, right) => {
  const leftParts = left.split(".").map(Number);
  const rightParts = right.split(".").map(Number);
  for (
    let index = 0;
    index < Math.max(leftParts.length, rightParts.length);
    index += 1
  ) {
    const difference = (rightParts[index] || 0) - (leftParts[index] || 0);
    if (difference) return difference;
  }
  return 0;
};

/** Merge histories from update channels without listing the same build twice. */
const getPreviousVersions = (feed, currentVersions) => {
  const byVersion = new Map();

  Object.values(feed || {}).forEach((channel) => {
    (channel?.versions || []).forEach((entry) => {
      if (!entry?.version || currentVersions.has(entry.version)) return;
      const existing = byVersion.get(entry.version) || {
        version: entry.version,
        urls: {},
        sizes: {},
      };
      const urls = Object.fromEntries(
        Object.entries(entry.download_urls || {}).filter(([, value]) => value),
      );
      const sizes = Object.fromEntries(
        Object.entries(entry.download_sizes || {}).filter(([, value]) => value),
      );
      byVersion.set(entry.version, {
        version: entry.version,
        urls: { ...existing.urls, ...urls },
        sizes: { ...existing.sizes, ...sizes },
      });
    });
  });

  return [...byVersion.values()]
    .filter((release) => Object.values(release.urls).some(Boolean))
    .sort((left, right) => compareVersions(left.version, right.version));
};

const PreviousVersions = ({ versions, pending }) => (
  <section
    className="mt-16 border-t border-line pt-10"
    aria-labelledby="previous-versions-heading"
  >
    <p className="t-eyebrow">Release archive</p>
    <h2 id="previous-versions-heading" className="t-display mt-3 text-2xl font-semibold">
      Previous versions
    </h2>
    <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted-foreground">
      Use an older build when you need to reproduce an existing setup or maintain
      compatibility. For new projects, use the latest version above. Select a version
      to see its available installers.
    </p>

    {pending ? (
      <p className="mt-6 text-sm text-muted-foreground" role="status">
        Loading release archive…
      </p>
    ) : versions.length ? (
      <div className="mt-6 overflow-hidden rounded-lg border border-line bg-surface">
        {versions.map((release) => {
          const downloads = ARCHIVE_PLATFORMS.filter(
            (platform) => release.urls[platform.id],
          );

          return (
            <details
              key={release.version}
              className="group border-b border-line last:border-b-0"
            >
              <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 font-medium transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-mark [&::-webkit-details-marker]:hidden">
                <span>AnyLearning {release.version}</span>
                <span className="flex items-center gap-3 text-xs font-normal text-muted-foreground">
                  {downloads.length} {downloads.length === 1 ? "download" : "downloads"}
                  <span className="text-base transition-transform group-open:rotate-180" aria-hidden="true">
                    ↓
                  </span>
                </span>
              </summary>
              <div className="border-t border-line bg-muted/40 px-5 py-5">
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  {downloads.map((platform) => (
                    <a
                      key={platform.id}
                      href={release.urls[platform.id]}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label={`Download AnyLearning ${release.version} for ${platform.label}`}
                      className="flex items-center justify-between gap-3 rounded-md border border-line bg-surface px-4 py-3 text-sm transition-colors hover:border-mark hover:text-mark focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark"
                    >
                      <span className="flex items-center gap-2">
                        <span className="text-muted-foreground" aria-hidden="true">
                          {platform.icon}
                        </span>
                        {platform.label}
                      </span>
                      <span className="font-mono text-xs text-muted-foreground">
                        {release.sizes[platform.id] || "Download"}
                      </span>
                    </a>
                  ))}
                </div>
              </div>
            </details>
          );
        })}
      </div>
    ) : (
      <p className="mt-6 text-sm text-muted-foreground">
        No previous downloads are available yet.
      </p>
    )}
  </section>
);

/** One release: a heading, what it is, and a card per platform. */
const ReleaseSection = ({ release, pending }) => (
  <section className="mt-14 first:mt-12">
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line pb-4">
      <div className="flex items-center gap-3">
        <h2 className="t-display text-2xl font-semibold">{release.name}</h2>
        {!pending && (
          <Badge tone={release.available ? "mark" : "neutral"}>
            {release.available ? `Latest · ${release.version}` : "Coming soon"}
          </Badge>
        )}
      </div>
      {release.whatsNew && (
        <a
          href={release.whatsNew.href}
          className="rounded text-sm text-muted-foreground underline underline-offset-4 hover:text-mark focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark"
        >
          {release.whatsNew.label}
        </a>
      )}
    </div>

    <p className="mt-4 max-w-2xl text-sm leading-relaxed text-muted-foreground">
      {release.summary}
    </p>

    <div className="mt-8 grid gap-4 md:grid-cols-3">
      {PLATFORMS.map((platform) => (
        <PlatformCard
          key={platform.id}
          icon={platform.icon}
          name={platform.name}
          requirement={platform.requirement}
          note={platform.note}
          warning={
            platform.id === "macos" && release.version === "0.26.3"
              ? "Temporary ad-hoc-signed build. Gatekeeper may warn until the signed and notarized package replaces it."
              : null
          }
          version={release.version}
          size={release.sizes?.[platform.id]}
          href={release.urls[platform.id]}
          alternate={
            platform.alternate && release.urls[platform.alternate.id]
              ? {
                    href: release.urls[platform.alternate.id],
                    label: platform.alternate.label,
                    size: release.sizes?.[platform.alternate.id],
                }
              : null
          }
          pending={pending}
        />
      ))}
    </div>
  </section>
);

const DownloadPage = () => {
  const [feed, setFeed] = useState(null);
  // Until the release list is in, no card claims to be available or not.
  const [pending, setPending] = useState(true);

  useEffect(() => {
    const fetchUpdateInfo = async () => {
      try {
        const response = await fetch(`/check-for-update.json?t=${Date.now()}`);
        setFeed(await response.json());
      } catch (error) {
        console.error("Error fetching update info:", error);
      } finally {
        setPending(false);
      }
    };

    fetchUpdateInfo();
  }, []);

  // A release is downloadable when the feed carries it with at least one real
  // platform URL. Available releases lead, so the day 26 ships it moves to the
  // top of the page without anyone editing this file.
  // Duplicate versions are omitted if compatibility aliases are added later.
  const seenVersions = new Set();
  const releases = RELEASES.map((release) => {
    const entry = feed?.[release.key];
    const latest = entry?.versions?.[0];
    const urls = latest?.download_urls ?? {};
    // Optional: a release that does not publish sizes simply does not show
    // them, so the older feed entries keep working untouched.
    const sizes = latest?.download_sizes ?? {};
    return {
      ...release,
      version: entry?.latest_version ?? "",
      urls,
      sizes,
      available: Object.values(urls).some(Boolean),
    };
  })
    .filter((release) => {
      // Entries with no feed data are "coming soon" and are never duplicates.
      if (!release.version) return true;
      if (seenVersions.has(release.version)) return false;
      seenVersions.add(release.version);
      return true;
    })
    .sort((a, b) => Number(b.available) - Number(a.available));
  const currentVersions = new Set(
    releases.map((release) => release.version).filter(Boolean),
  );
  const previousVersions = getPreviousVersions(feed, currentVersions);

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <ThemeSync />
      <Head>
        <title>Download AnyLearning OSS</title>
        <meta
          name="description"
          content="Download AnyLearning to start building powerful AI models locally. Available for macOS, Windows and Linux."
        />
      </Head>

      <div className="mx-auto w-full max-w-5xl flex-1">
        <Menu activePage="download" />

        <main className="pb-20 pt-16">
          <p className="t-eyebrow">Download</p>
          <h1 className="t-display mt-3 text-4xl font-semibold sm:text-5xl">
            Download AnyLearning
          </h1>
          <p className="mt-5 max-w-2xl text-lg text-muted-foreground">
            Label images and train models on your own machine. The app works offline,
            so your data stays private.
          </p>

          {releases.map((release) => (
            <ReleaseSection key={release.key} release={release} pending={pending} />
          ))}

          <PreviousVersions versions={previousVersions} pending={pending} />

          <p className="mt-10 text-sm text-muted-foreground">
            Already downloaded? The{" "}
            <a
              href="/docs/installation"
              className="rounded text-foreground underline underline-offset-4 hover:text-mark focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark"
            >
              installation guide
            </a>{" "}
            walks through setup and your first launch.
          </p>
        </main>
      </div>

      <SiteFooter />
    </div>
  );
};

export default DownloadPage;

/**
 * The AnyLearning mark: an annotation box caught mid-draw.
 *
 * Same geometry as the desktop app's logo — a bounding box with one live
 * corner handle, which is exactly what the canvas draws when you label. The
 * frame takes currentColor and the handle takes --mark, so it is correct in
 * both themes from one asset and needs no baked tile.
 */
export function Logo({ className = "h-8 w-8" }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" role="img" aria-label="AnyLearning" className={className}>
      <path
        d="M7.5 4H16.5A3.5 3.5 0 0 1 20 7.5V14"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M14 20H7.5A3.5 3.5 0 0 1 4 16.5V7.5A3.5 3.5 0 0 1 7.5 4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <rect x="17" y="17" width="5.5" height="5.5" rx="1.5" fill="var(--mark)" />
    </svg>
  );
}

export function Wordmark({ className = "" }) {
  return (
    <span className={`inline-flex items-center gap-2.5 ${className}`}>
      <Logo className="h-7 w-7 shrink-0" />
      <span className="font-display text-[15px] font-semibold tracking-tight">AnyLearning</span>
    </span>
  );
}

export default Logo;

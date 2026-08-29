import sharp from "sharp";

/**
 * Draws tutorial annotations onto a screenshot.
 *
 * Screenshots are captured at 2x, so every coordinate in a shot spec is written
 * in CSS pixels — the same numbers you read off the browser — and scaled here.
 *
 * The palette is the design system's, so a documentation callout looks like it
 * belongs to the product rather than like a red MS Paint circle. `--mark` is
 * the annotation stroke the app itself draws, which makes "look here" and "this
 * is a bounding box" the same visual idea.
 */

const MARK = "#31becc"; // --mark, dark theme
const MARK_INK = "#10262b";
const RUN = "#e8b04b"; // --run, for "careful" callouts
const SHADOW = "rgba(0,0,0,0.45)";

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** A numbered step badge, for "do this, then this". */
function marker({ n, x, y, tone = MARK }) {
  const ink = tone === MARK ? MARK_INK : "#2a1c05";
  return `
    <circle cx="${x}" cy="${y}" r="15" fill="${tone}" stroke="rgba(255,255,255,0.9)" stroke-width="2"/>
    <text x="${x}" y="${y}" fill="${ink}" font-family="JetBrains Mono, ui-monospace, monospace"
          font-size="15" font-weight="700" text-anchor="middle" dominant-baseline="central">${esc(n)}</text>`;
}

/** A ring around the control being described. */
function ring({ x, y, w, h, tone = MARK, radius = 8 }) {
  return `
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}"
          fill="none" stroke="${tone}" stroke-width="3"/>
    <rect x="${x - 3}" y="${y - 3}" width="${w + 6}" height="${h + 6}" rx="${radius + 3}"
          fill="none" stroke="rgba(0,0,0,0.35)" stroke-width="1.5"/>`;
}

/** A short caption pinned near a control. */
function label({ x, y, text, tone = MARK, anchor = "start" }) {
  const pad = 9;
  const charW = 7.1;
  const w = text.length * charW + pad * 2;
  const boxX = anchor === "end" ? x - w : x;
  const ink = tone === MARK ? MARK_INK : "#2a1c05";
  return `
    <rect x="${boxX}" y="${y - 13}" width="${w}" height="26" rx="6" fill="${tone}"/>
    <text x="${boxX + pad}" y="${y}" fill="${ink}" font-family="Inter, system-ui, sans-serif"
          font-size="13" font-weight="600" dominant-baseline="central">${esc(text)}</text>`;
}

/** An arrow from a caption to the thing it names. */
function arrow({ x1, y1, x2, y2, tone = MARK }) {
  const id = `ah${Math.round(x1)}${Math.round(y1)}`;
  return `
    <defs>
      <marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="${tone}"/>
      </marker>
    </defs>
    <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${tone}" stroke-width="3"
          stroke-linecap="round" marker-end="url(#${id})"/>`;
}

/** Dims everything except one region, for a busy screen. */
function spotlight({ x, y, w, h }, width, height) {
  return `
    <defs>
      <mask id="spot">
        <rect width="${width}" height="${height}" fill="white"/>
        <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="black"/>
      </mask>
    </defs>
    <rect width="${width}" height="${height}" fill="rgba(0,0,0,0.5)" mask="url(#spot)"/>`;
}

const RENDERERS = { marker, ring, label, arrow };

/**
 * @param {Buffer} png      screenshot, captured at 2x
 * @param {Array}  notes    annotation specs in CSS pixels
 */
export async function annotate(png, notes = []) {
  if (!notes.length) return png;

  const image = sharp(png);
  const { width, height } = await image.metadata();
  const scale = 2; // matches --force-device-scale-factor

  const spot = notes.find((n) => n.type === "spotlight");
  const body = notes
    .filter((n) => n.type !== "spotlight")
    .map((note) => {
      const render = RENDERERS[note.type];
      if (!render) throw new Error(`Unknown annotation type: ${note.type}`);
      return render(note);
    })
    .join("\n");

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"
       viewBox="0 0 ${width / scale} ${height / scale}">
    <g filter="drop-shadow(0 1px 2px ${SHADOW})">
      ${spot ? spotlight(spot, width / scale, height / scale) : ""}
      ${body}
    </g>
  </svg>`;

  return image
    .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
    .png()
    .toBuffer();
}

export const TONES = { MARK, RUN };

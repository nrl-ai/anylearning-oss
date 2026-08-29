# AnyLearning design system — "Bench"

The rules the UI follows, and why. Tokens live in `src/app/globals.css`; the
primitives that enforce them live in `src/components/ui/` and
`src/components/layout/`.

---

## The thesis

AnyLearning is an **image-inspection instrument that runs on your own machine**,
not a dashboard. Three facts about the product drive every decision here:

1. **The user is judging photographs, X-rays and micrographs.** Chrome that
   carries colour changes how those images read. Every serious image tool —
   Lightroom, Capture One, radiology viewers — surrounds the image with
   chromatically neutral grey for exactly this reason, and so do we.
2. **The pipeline is real and stateful.** Overview → Dataset → Training →
   Models is a genuine sequence with genuine preconditions. You cannot train
   without labelled data; you cannot have a model without a finished run. The
   UI reports where the project actually is, never where you happen to be
   looking.
3. **Nothing leaves the machine.** That is the product's reason to exist. It
   belongs in the product's voice -- the sidebar reads "Local AI training" --
   not in a permanent badge. A "Local" pill lived in the workbench bar briefly
   and was cut: an indicator with exactly one possible value is decoration
   wearing the costume of status.

**The one-line rule:** _the images are the only thing on screen allowed to be
colourful._ Colour in the chrome has to earn its place by meaning something.

---

## Colour

All neutrals are near-zero chroma (≤ 0.008) at hue 255. They are grey on
purpose — they sit next to photographs all day.

> **Note on OKLCH:** its lightness is roughly the cube root of luminance, so
> the numbers run higher than they look. `oklch(0.245)` is a mid-graphite
> (≈ `#1e1f22`), not a mid-grey. Don't port values from an HSL palette by eye.

### Surfaces — elevation is a lightness step, not a shadow

| Token              | Role                                                |
| ------------------ | --------------------------------------------------- |
| `surface-sunken`   | Wells: the canvas surround, log panes, chart cards  |
| `background`       | The app ground                                      |
| `surface` / `card` | Panels, sidebar, bars — one step up from the ground |
| `surface-raised`   | The rare thing that sits above a panel              |

Shadows are hairline-quiet and reserved for true overlays (popovers, dialogs,
the floating tool palette). In dark mode a shadow barely reads at all, which is
precisely why elevation is carried by lightness instead.

### The mark — the annotation stroke

`--mark` is the app's single accent, and it is _the same colour family as an
annotation stroke on the canvas_. It appears on exactly five things:

- the primary action on a screen (one per screen, never one per table row)
- the focus ring
- the current project in the sidebar
- the current stage on the rail
- selection on the canvas

Anything else that wants to be blue is wrong.

### State — never decorative

`idle` · `run` · `ok` · `warn` · `fail`. Each owns a token trio
(`--x`, `--x-surface`, `--x-border`) so a status chip is one lookup rather than
six hand-picked palette classes. `src/lib/status.ts` is the only place that maps
a backend status to a tone and a label; `<StatusBadge>` / `<TrainingStatusBadge>`
are the only ways to render one.

Two conventions worth stating:

- **A user-initiated stop is not a failure.** `terminated` reads as neutral
  "Stopped". Painting a deliberate stop red makes the screen look broken.
- **The state that needs work is the state that gets marked.** Unlabelled
  images are called out; labelled ones recede to a quiet check. A finished
  dataset should look calm, and gaps should pop.

### Label classes

`--class-1` … `--class-8`: eight hues at equal lightness so no class shouts over
another. These are the **only** other saturated colours in the app.

A class is always drawn as an **annotation swatch** — a solid 2px stroke over an
~18% fill — which is exactly how it appears on the canvas. The legend and the
annotation are visibly one system rather than two unrelated colour chips.

### Charts

`--chart-1..3` are one hue at three steps, because training / validation / test
are three parts of one dataset, not three unrelated series. `--chart-4..5` are
contrasting accents for genuinely unrelated series. Chart colours sit a step
below `--mark`: reference data should not outshine the controls.

Single-series charts (the per-metric small multiples) just use `--mark` and put
the metric name in the title — a legend that repeats the only line on the chart
is noise.

---

## Type

Three faces, three jobs. All self-hosted via `next/font` so the desktop app
never reaches the network for a font.

| Face               | Variable         | Used for                                                |
| ------------------ | ---------------- | ------------------------------------------------------- |
| **Space Grotesk**  | `--font-display` | Wordmark, page and panel titles, eyebrows, stage labels |
| **Inter**          | `--font-sans`    | Everything else                                         |
| **JetBrains Mono** | `--font-mono`    | Every number, filename, path, ID and hex value          |

The mono role is functional, not decorative. This app is dense with values you
read by comparison — epochs, loss, mAP, counts, durations, image filenames. In a
proportional face a column of them jitters and can't be scanned. Numbers set in
mono also carry `tabular-nums`.

Use the semantic classes rather than ad-hoc size/weight pairs:

`t-title` · `t-section` · `t-eyebrow` · `t-data` · `t-ident` · `t-meta`

---

## Layout

- **One bar, not two.** `WorkbenchBar` carries project identity, the stage rail
  and the global controls. It replaced an empty 48px header plus a separate
  floating stepper — together ~110px of chrome that said nothing.
- **Fill the height with flex,** never `calc(100vh - <header height>)`. The bar
  changes height (the rail wraps to a second row on narrow windows), and every
  hard-coded offset was already wrong.
- Workspace content is capped at `1600px` and centred.
- Radius scale from `--radius: 0.5rem`. Tighter than the shadcn default —
  a measuring instrument, not a consumer app.

### The stage rail — the signature element

`src/components/layout/stage-rail.tsx`. Each stage shows its **real** state,
derived from live data (label count, image count, run status, model count):

| State     | How it reads                                    |
| --------- | ----------------------------------------------- |
| `empty`   | hairline track only                             |
| `partial` | part-filled neutral bar (e.g. 812/945 labelled) |
| `done`    | filled neutral bar                              |
| `running` | amber, breathing, with live epoch count         |
| `failed`  | red, with a warning glyph                       |

**Completion is shown by fill, not hue** — that's what keeps a healthy project
calm and leaves colour free to mean "look here". The stage you are viewing gets
the mark.

This replaced a stepper that drew a green check on every stage behind the
current tab, so a brand-new project with no images and no models still claimed
three stages complete. The rail cannot make that claim; it reads the data.

---

## Motion

Two durations (`--duration-quick` 120ms, `--duration-settle` 220ms) and one
easing (`--ease-bench`). Motion marks a state change and nothing else. The only
looping animation in the app is `animate-breathe`, and it means one thing: a
machine process is alive. `prefers-reduced-motion` is honoured globally.

---

## Writing

- Sentence case everywhere. Buttons name the action: "Start training", not
  "Submit".
- An action keeps its name through the whole flow.
- Errors say what happened and what to do; they don't apologise and they're
  never vague.
- Empty states are invitations — they carry the next action.
- Confirmations name the thing and the consequence: "Delete 12 images?" beats
  "Are you absolutely sure?".

---

## Adding UI

1. Reach for a primitive first: `Panel` / `PanelHeader` / `PanelBody` /
   `PanelFooter`, `Stat`, `EmptyState`, `StatusBadge`.
2. Use tokens. **No Tailwind palette classes** (`bg-blue-500`, `text-gray-700`,
   …) — they have no dark mode and no meaning. If nothing fits, the token set is
   missing something; add it to `globals.css` rather than working around it.
3. Numbers get `font-mono tabular`.
4. Ask what the colour means. If the answer is "it looked nice", use a neutral.

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

/* --------------------------------------------------------------------------
   LabelDemo — a working bounding-box labeller.

   Drag on the image to draw a box; a class picker appears on release. Click a
   box to select it, backspace (or the × on its chip) removes it.

   Two things worth knowing before editing:

   1. The gesture terminator is bound to `window`, not to the SVG. The desktop
      app shipped the other version once: if the mouse came up outside the
      drawing surface the SVG never saw `pointerup`, `drawing` stayed true and
      labelling silently died until you clicked again. `setPointerCapture`
      already routes the events back to the surface, and the window listener is
      the belt behind that brace — either one alone ends the gesture.

   2. Boxes are stored in viewBox units (VB_W × VB_H), never in screen pixels,
      so the component is resolution-independent: label chips are positioned as
      a percentage of the stage and strokes use `vector-effect:
      non-scaling-stroke`, so a 2px annotation stroke stays 2px at any size.
   -------------------------------------------------------------------------- */

// The stage is a crop of an existing asset, done in the viewBox rather than by
// adding a new file to public/.
const IMAGE = {
  src: "/future-ai.jpeg",
  naturalWidth: 2368,
  naturalHeight: 1792,
  crop: { x: 0, y: 470, width: 2368, height: 1322 },
};

const VB_W = 1000;
const VB_H = Math.round((VB_W * IMAGE.crop.height) / IMAGE.crop.width); // 558

// Placing the untouched image inside the cropped viewBox.
const IMG_SCALE = VB_W / IMAGE.crop.width;
const IMG_X = -IMAGE.crop.x * IMG_SCALE;
const IMG_Y = -IMAGE.crop.y * IMG_SCALE;
const IMG_W = IMAGE.naturalWidth * IMG_SCALE;
const IMG_H = IMAGE.naturalHeight * IMG_SCALE;

const DEFAULT_CLASSES = [
  { id: "person", name: "person", token: 1 },
  { id: "cup", name: "cup", token: 2 },
  { id: "tablet", name: "tablet", token: 3 },
  { id: "chair", name: "chair", token: 4 },
];

// Pre-drawn so the demo looks like work already in progress on first paint.
const DEFAULT_BOXES = [
  { id: "b1", classId: "person", x: 448, y: 66, width: 166, height: 278 },
  { id: "b2", classId: "person", x: 292, y: 98, width: 112, height: 232 },
  { id: "b3", classId: "cup", x: 395, y: 384, width: 78, height: 54 },
];

// Below this (in viewBox units) a drag reads as a mis-click, not a box.
const MIN_SIZE = 12;

// Picker geometry, in CSS px — used only to keep the popover inside the canvas.
const PICKER_W = 176;
const PICKER_HEADER = 62;
const PICKER_ROW = 26;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const classVar = (token) => `var(--class-${token})`;
const tint = (token) => `color-mix(in oklab, ${classVar(token)} 18%, transparent)`;

/** The annotation mark itself: 2px stroke over an 18% fill of the same colour.
 *  Identical to how a box is painted on the canvas — that is the whole point of
 *  a legend swatch. */
function ClassSwatch({ token, className = "" }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block h-3.5 w-5 shrink-0 rounded-[2px] ${className}`}
      style={{
        border: `2px solid ${classVar(token)}`,
        backgroundColor: tint(token),
      }}
    />
  );
}

export default function LabelDemo({
  className = "",
  src = IMAGE.src,
  classes = DEFAULT_CLASSES,
  initialBoxes = DEFAULT_BOXES,
}) {
  const stageRef = useRef(null);
  const startRef = useRef(null);
  const detachRef = useRef(null);
  const seqRef = useRef(0);

  const [boxes, setBoxes] = useState(initialBoxes);
  const [draft, setDraft] = useState(null); // live rectangle while dragging
  const [pending, setPending] = useState(null); // finished rectangle awaiting a class
  const [selectedId, setSelectedId] = useState(null);

  const classById = useMemo(() => {
    const map = {};
    for (const c of classes) map[c.id] = c;
    return map;
  }, [classes]);

  const counts = useMemo(() => {
    const map = {};
    for (const b of boxes) map[b.classId] = (map[b.classId] || 0) + 1;
    return map;
  }, [boxes]);

  /** Client coordinates → viewBox units. The stage keeps the viewBox aspect
   *  ratio exactly, so this is a straight linear map. */
  const toViewBox = useCallback((event) => {
    const node = stageRef.current;
    if (!node) return { x: 0, y: 0 };
    const rect = node.getBoundingClientRect();
    if (!rect.width || !rect.height) return { x: 0, y: 0 };
    return {
      x: clamp(((event.clientX - rect.left) / rect.width) * VB_W, 0, VB_W),
      y: clamp(((event.clientY - rect.top) / rect.height) * VB_H, 0, VB_H),
    };
  }, []);

  const rectFrom = useCallback((a, b) => {
    return {
      x: Math.min(a.x, b.x),
      y: Math.min(a.y, b.y),
      width: Math.abs(a.x - b.x),
      height: Math.abs(a.y - b.y),
    };
  }, []);

  /** Where the picker sits, in stage pixels, clamped so it can never spill out
   *  of the canvas. Measured once when the gesture ends rather than tracked, so
   *  it costs nothing while drawing. */
  const anchorFor = useCallback(
    (rect) => {
      const node = stageRef.current;
      if (!node) return { left: 0, top: 0 };
      const stage = node.getBoundingClientRect();
      const width = PICKER_W;
      const height = PICKER_HEADER + classes.length * PICKER_ROW;
      const gap = 8;
      const left = clamp(
        (rect.x / VB_W) * stage.width,
        gap,
        Math.max(gap, stage.width - width - gap),
      );
      const below = ((rect.y + rect.height) / VB_H) * stage.height + gap;
      const above = (rect.y / VB_H) * stage.height - height - gap;
      let top = below;
      if (below + height > stage.height - gap) {
        top = above >= gap ? above : Math.max(gap, stage.height - height - gap);
      }
      return { left, top };
    },
    [classes.length],
  );

  // Listeners are attached inside the pointerdown handler rather than from an
  // effect, so there is no window between "the gesture started" and "the
  // terminator is listening".
  const detachGesture = useCallback(() => {
    if (!detachRef.current) return;
    detachRef.current();
    detachRef.current = null;
  }, []);

  useEffect(() => detachGesture, [detachGesture]); // unmount safety

  const onSurfacePointerDown = (event) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    if (pending) return; // the picker owns the interaction until it is answered

    setSelectedId(null);
    const start = toViewBox(event);
    startRef.current = start;
    setDraft({ x: start.x, y: start.y, width: 0, height: 0 });

    // Routes every later pointer event of this gesture back to the surface,
    // even when the pointer leaves it.
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      /* capture is best-effort; the window listeners below are the guarantee */
    }

    const finish = () => {
      detachGesture();
      setDraft(null);
      startRef.current = null;
    };

    // The gesture terminator lives on the window, never on the SVG, so it fires
    // even when the pointer comes up over the sidebar, over another window, or
    // off the page entirely.
    const move = (moveEvent) => setDraft(rectFrom(start, toViewBox(moveEvent)));
    const end = (upEvent) => {
      const rect = rectFrom(start, toViewBox(upEvent));
      finish();
      if (rect.width < MIN_SIZE || rect.height < MIN_SIZE) return;
      setPending({ ...rect, anchor: anchorFor(rect) });
    };
    const abort = () => finish();

    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", end);
    window.addEventListener("pointercancel", abort);
    window.addEventListener("blur", abort);
    detachRef.current = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", end);
      window.removeEventListener("pointercancel", abort);
      window.removeEventListener("blur", abort);
    };

    event.preventDefault();
  };

  const assignClass = useCallback(
    (classId) => {
      if (!pending) return;
      seqRef.current += 1;
      const id = `n${seqRef.current}`;
      const { anchor, ...rect } = pending; // the anchor is picker-only state
      void anchor;
      setBoxes((prev) => [...prev, { id, classId, ...rect }]);
      setSelectedId(id);
      setPending(null);
    },
    [pending],
  );

  const removeBox = useCallback((id) => {
    setBoxes((prev) => prev.filter((b) => b.id !== id));
    setSelectedId((current) => (current === id ? null : current));
  }, []);

  // Keyboard shortcuts are live only while something is selected or the picker
  // is open, so the demo never swallows a key the rest of the page wants.
  useEffect(() => {
    if (!pending && !selectedId) return undefined;
    const onKey = (event) => {
      const target = event.target;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (event.key === "Escape") {
        setPending(null);
        setSelectedId(null);
        return;
      }
      if (pending) {
        const index = Number(event.key) - 1;
        if (Number.isInteger(index) && index >= 0 && index < classes.length) {
          event.preventDefault();
          assignClass(classes[index].id);
        }
        return;
      }
      if (event.key === "Backspace" || event.key === "Delete") {
        event.preventDefault();
        removeBox(selectedId);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pending, selectedId, classes, assignClass, removeBox]);

  const reset = () => {
    detachGesture();
    setBoxes(initialBoxes);
    setPending(null);
    setSelectedId(null);
    setDraft(null);
    startRef.current = null;
  };

  const pickerStyle = pending
    ? {
        left: `${pending.anchor?.left ?? 8}px`,
        top: `${pending.anchor?.top ?? 8}px`,
        width: `${PICKER_W}px`,
      }
    : null;

  return (
    <section
      className={`overflow-hidden rounded-xl border border-line bg-surface ${className}`}
    >
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div>
          <p className="t-eyebrow">Object detection</p>
          <h3 className="font-display text-[15px] font-semibold tracking-tight text-foreground">
            Drag to draw a box, then pick a class
          </h3>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono tabular text-xs text-muted-foreground">
            {String(boxes.length).padStart(2, "0")} objects
          </span>
          <button
            type="button"
            onClick={reset}
            className="rounded-md border border-line bg-surface px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-mark"
          >
            Reset
          </button>
        </div>
      </header>

      <div className="grid md:grid-cols-[minmax(0,1fr)_15rem]">
        {/* ---------------------------------------------------------------- */}
        {/* Canvas */}
        <div className="bg-surface-sunken p-3 sm:p-4">
          <div
            ref={stageRef}
            role="group"
            aria-label="Bounding box labelling canvas"
            className="relative w-full touch-none select-none overflow-hidden rounded-lg border border-line bg-background"
            style={{ aspectRatio: `${VB_W} / ${VB_H}` }}
          >
            <svg
              viewBox={`0 0 ${VB_W} ${VB_H}`}
              width="100%"
              height="100%"
              className="block h-full w-full cursor-crosshair"
              onPointerDown={onSurfacePointerDown}
            >
              <image
                href={src}
                x={IMG_X}
                y={IMG_Y}
                width={IMG_W}
                height={IMG_H}
                preserveAspectRatio="xMidYMid slice"
              />

              {boxes.map((box) => {
                const cls = classById[box.classId];
                if (!cls) return null;
                const selected = box.id === selectedId;
                return (
                  <g key={box.id}>
                    <rect
                      x={box.x}
                      y={box.y}
                      width={box.width}
                      height={box.height}
                      fill={classVar(cls.token)}
                      fillOpacity={selected ? 0.28 : 0.18}
                      stroke={classVar(cls.token)}
                      strokeWidth={2}
                      vectorEffect="non-scaling-stroke"
                      className="cursor-pointer"
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        setPending(null);
                        setSelectedId(box.id);
                      }}
                    />
                    {selected ? (
                      <rect
                        x={box.x - 3}
                        y={box.y - 3}
                        width={box.width + 6}
                        height={box.height + 6}
                        fill="none"
                        stroke="var(--foreground)"
                        strokeWidth={1}
                        strokeDasharray="4 3"
                        vectorEffect="non-scaling-stroke"
                        pointerEvents="none"
                      />
                    ) : null}
                  </g>
                );
              })}

              {draft ? (
                <rect
                  x={draft.x}
                  y={draft.y}
                  width={draft.width}
                  height={draft.height}
                  fill="var(--mark)"
                  fillOpacity={0.18}
                  stroke="var(--mark)"
                  strokeWidth={2}
                  strokeDasharray="5 4"
                  vectorEffect="non-scaling-stroke"
                  pointerEvents="none"
                />
              ) : null}

              {pending ? (
                <rect
                  x={pending.x}
                  y={pending.y}
                  width={pending.width}
                  height={pending.height}
                  fill="var(--mark)"
                  fillOpacity={0.18}
                  stroke="var(--mark)"
                  strokeWidth={2}
                  vectorEffect="non-scaling-stroke"
                  pointerEvents="none"
                />
              ) : null}
            </svg>

            {/* Class chips, in HTML so they stay crisp at any stage size. */}
            {boxes.map((box) => {
              const cls = classById[box.classId];
              if (!cls) return null;
              const selected = box.id === selectedId;
              const inside = box.y < 26;
              return (
                <span
                  key={`chip-${box.id}`}
                  className="pointer-events-none absolute z-10 flex items-center gap-1 whitespace-nowrap rounded-[3px] border bg-surface px-1 py-px font-mono text-[10px] leading-4"
                  style={{
                    left: `${(box.x / VB_W) * 100}%`,
                    top: `${((box.y + (inside ? 2 : 0)) / VB_H) * 100}%`,
                    transform: inside ? undefined : "translateY(-115%)",
                    color: classVar(cls.token),
                    borderColor: "currentColor",
                  }}
                >
                  {cls.name}
                  {selected ? (
                    <button
                      type="button"
                      aria-label={`Delete ${cls.name} box`}
                      onClick={() => removeBox(box.id)}
                      className="pointer-events-auto -mr-0.5 px-0.5 leading-none text-current hover:opacity-70"
                    >
                      ×
                    </button>
                  ) : null}
                </span>
              );
            })}

            {/* Class picker */}
            {pending ? (
              <div
                className="absolute z-20 overflow-hidden rounded-lg border border-line bg-surface p-1 shadow-lg"
                style={pickerStyle}
              >
                <p className="t-eyebrow px-1.5 pb-1 pt-1">Assign class</p>
                {classes.map((cls, index) => (
                  <button
                    key={cls.id}
                    type="button"
                    onClick={() => assignClass(cls.id)}
                    className="flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left text-[13px] text-foreground transition-colors hover:bg-muted focus:outline-none focus-visible:bg-muted"
                  >
                    <ClassSwatch token={cls.token} />
                    <span className="truncate">{cls.name}</span>
                    <span className="ml-auto font-mono tabular text-[11px] text-muted-foreground">
                      {index + 1}
                    </span>
                  </button>
                ))}
                <button
                  type="button"
                  onClick={() => setPending(null)}
                  className="mt-1 w-full border-t border-line px-1.5 pb-1 pt-1.5 text-left text-[12px] text-muted-foreground transition-colors hover:text-foreground"
                >
                  Discard box
                </button>
              </div>
            ) : null}
          </div>

          <p className="mt-3 text-xs text-muted-foreground">
            Drag on the image to draw · click a box to select ·{" "}
            <kbd className="rounded border border-line bg-surface px-1 py-px font-mono text-[10px] text-muted-foreground">
              Del
            </kbd>{" "}
            to remove it
          </p>
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* Legend + object list */}
        <aside className="border-t border-line p-4 md:border-l md:border-t-0">
          <p className="t-eyebrow">Classes</p>
          <ul className="mt-2 space-y-1.5">
            {classes.map((cls) => (
              <li key={cls.id} className="flex items-center gap-2 text-[13px]">
                <ClassSwatch token={cls.token} />
                <span className="truncate text-foreground">{cls.name}</span>
                <span className="ml-auto font-mono tabular text-xs text-muted-foreground">
                  {counts[cls.id] || 0}
                </span>
              </li>
            ))}
          </ul>

          <p className="t-eyebrow mt-5">Objects</p>
          {boxes.length === 0 ? (
            <p className="mt-2 text-xs text-muted-foreground">
              None yet. Drag on the image to add one.
            </p>
          ) : (
            <ul className="mt-2 space-y-1">
              {boxes.map((box) => {
                const cls = classById[box.classId];
                if (!cls) return null;
                const selected = box.id === selectedId;
                return (
                  <li key={box.id}>
                    <div
                      className={`flex items-center gap-2 rounded-md border px-1.5 py-1 transition-colors ${
                        selected
                          ? "border-mark-border bg-mark-soft"
                          : "border-transparent hover:bg-muted"
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => setSelectedId(box.id)}
                        className="flex min-w-0 flex-1 items-center gap-2 text-left focus:outline-none focus-visible:underline"
                      >
                        <span
                          aria-hidden="true"
                          className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
                          style={{
                            border: `2px solid ${classVar(cls.token)}`,
                            backgroundColor: tint(cls.token),
                          }}
                        />
                        <span className="truncate text-[13px] text-foreground">
                          {cls.name}
                        </span>
                        <span className="ml-auto shrink-0 font-mono tabular text-[10px] text-muted-foreground">
                          {Math.round(box.width)}×{Math.round(box.height)}
                        </span>
                      </button>
                      <button
                        type="button"
                        aria-label={`Delete ${cls.name} box`}
                        onClick={() => removeBox(box.id)}
                        className="shrink-0 rounded px-1 text-xs leading-none text-muted-foreground transition-colors hover:text-fail focus:outline-none focus-visible:text-fail"
                      >
                        ×
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </aside>
      </div>
    </section>
  );
}

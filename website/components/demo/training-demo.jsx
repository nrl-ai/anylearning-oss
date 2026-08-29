import React, { useEffect, useMemo, useState } from "react";

/* --------------------------------------------------------------------------
   TrainingDemo — a simulated training run.

   Twenty epochs at 600ms each (~12s), with loss and accuracy plotted as inline
   SVG. No chart library, no canvas, no dependencies.

   The curves are a closed-form function of the epoch index — an exponential
   plus a fixed sine ripple — so there is no Math.random anywhere and the run
   looks identical every time it is watched. That matters for a demo people see
   twice.

   Reduced motion: the ticking run *is* the content, so instead of animating it
   anyway we hand the reader the finished state immediately.
   -------------------------------------------------------------------------- */

const TOTAL_EPOCHS = 20;
const STEP_MS = 600; // 20 × 600ms ≈ 12s

const LOSS_DOMAIN = [0, 1];
const ACC_DOMAIN = [0.3, 1];

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const lossAt = (i) =>
  clamp(0.94 * Math.exp(-0.185 * i) + 0.052 + 0.013 * Math.sin(i * 2.1), 0, 1);
const accAt = (i) =>
  clamp(0.945 - 0.63 * Math.exp(-0.21 * i) + 0.009 * Math.sin(i * 1.7), 0, 1);

const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Plot geometry, in viewBox units. */
const PLOT = { w: 300, h: 132, left: 32, right: 294, top: 10, bottom: 104 };

function Curve({ label, values, domain, format, ticks }) {
  const [lo, hi] = domain;
  const xAt = (i) =>
    PLOT.left + (i / (TOTAL_EPOCHS - 1)) * (PLOT.right - PLOT.left);
  const yAt = (v) =>
    PLOT.bottom - ((clamp(v, lo, hi) - lo) / (hi - lo)) * (PLOT.bottom - PLOT.top);

  const points = values.map((v, i) => `${xAt(i).toFixed(2)},${yAt(v).toFixed(2)}`);
  const last = values.length - 1;
  const area =
    values.length > 1
      ? `M ${xAt(0).toFixed(2)},${PLOT.bottom} L ${points.join(" L ")} L ${xAt(
          last,
        ).toFixed(2)},${PLOT.bottom} Z`
      : null;

  return (
    <div className="rounded-lg border border-line bg-surface-sunken p-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="t-eyebrow">{label}</span>
        <span className="font-mono tabular text-[13px] font-medium text-foreground">
          {values.length ? format(values[last]) : "—"}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${PLOT.w} ${PLOT.h}`}
        width="100%"
        className="mt-2 block h-auto w-full overflow-visible"
        role="img"
        aria-label={`${label} over ${TOTAL_EPOCHS} epochs`}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={PLOT.left}
              x2={PLOT.right}
              y1={yAt(t)}
              y2={yAt(t)}
              stroke="var(--border)"
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={PLOT.left - 6}
              y={yAt(t) + 3}
              textAnchor="end"
              className="font-mono tabular"
              fontSize={8}
              fill="var(--muted-foreground)"
            >
              {format(t)}
            </text>
          </g>
        ))}

        {[1, 10, 20].map((e) => (
          <text
            key={e}
            x={xAt(e - 1)}
            y={PLOT.bottom + 14}
            textAnchor="middle"
            className="font-mono tabular"
            fontSize={8}
            fill="var(--muted-foreground)"
          >
            {e}
          </text>
        ))}

        {area ? <path d={area} fill="var(--mark)" fillOpacity={0.12} /> : null}

        {points.length > 1 ? (
          <polyline
            points={points.join(" ")}
            fill="none"
            stroke="var(--mark)"
            strokeWidth={1.75}
            strokeLinejoin="round"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        ) : null}

        {points.length ? (
          <circle
            cx={xAt(last)}
            cy={yAt(values[last])}
            r={2.75}
            fill="var(--mark)"
            stroke="var(--surface-sunken)"
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
          />
        ) : null}
      </svg>
    </div>
  );
}

/** The dot always takes the chip's own text colour, so machine state is the one
 *  thing here carrying hue — amber while alive, green when done, and the same
 *  neutral as its label when idle. */
function StatusChip({ status }) {
  if (status === "running") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-run-surface px-2.5 py-1 text-[11px] font-medium text-run">
        <span className="animate-breathe h-1.5 w-1.5 rounded-full bg-current" />
        running
      </span>
    );
  }
  if (status === "finished") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-ok-surface px-2.5 py-1 text-[11px] font-medium text-ok">
        <span className="h-1.5 w-1.5 rounded-full bg-current" />
        finished
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface-sunken px-2.5 py-1 text-[11px] font-medium text-muted-foreground">
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      idle
    </span>
  );
}

export default function TrainingDemo({
  className = "",
  modelName = "nanodet-plus-m · 320",
}) {
  // Starts on a *finished* run, not an empty one. On a marketing page nobody
  // presses play before deciding whether the thing works, and two blank chart
  // frames read as broken rather than as "not started yet". The button replays
  // the animation from zero for anyone who wants to watch it happen.
  const [status, setStatus] = useState("finished"); // idle | running | finished
  const [epoch, setEpoch] = useState(TOTAL_EPOCHS);

  const series = useMemo(() => {
    const loss = [];
    const acc = [];
    for (let i = 0; i < TOTAL_EPOCHS; i += 1) {
      loss.push(lossAt(i));
      acc.push(accAt(i));
    }
    return { loss, acc };
  }, []);

  useEffect(() => {
    if (status !== "running") return undefined;
    if (prefersReducedMotion()) {
      setEpoch(TOTAL_EPOCHS);
      return undefined;
    }
    const id = setInterval(
      () => setEpoch((e) => Math.min(e + 1, TOTAL_EPOCHS)),
      STEP_MS,
    );
    return () => clearInterval(id);
  }, [status]);

  useEffect(() => {
    if (status === "running" && epoch >= TOTAL_EPOCHS) setStatus("finished");
  }, [status, epoch]);

  const start = () => {
    setEpoch(1);
    setStatus("running");
  };

  const shown = status === "idle" ? 0 : epoch;
  const loss = series.loss.slice(0, shown);
  const acc = series.acc.slice(0, shown);
  const progress = (shown / TOTAL_EPOCHS) * 100;
  const elapsed = ((shown * STEP_MS) / 1000).toFixed(1);
  const lastLoss = loss.length ? loss[loss.length - 1] : null;
  const lastAcc = acc.length ? acc[acc.length - 1] : null;

  return (
    <section
      className={`overflow-hidden rounded-xl border border-line bg-surface ${className}`}
    >
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div>
          <p className="t-eyebrow">Training session</p>
          <h3 className="font-display text-[15px] font-semibold tracking-tight text-foreground">
            {modelName}
          </h3>
        </div>
        <StatusChip status={status} />
      </header>

      <div className="space-y-4 p-4">
        <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
          <div>
            <p className="t-eyebrow">Epoch</p>
            <p className="font-mono tabular text-2xl font-medium leading-tight text-foreground">
              {shown}
              <span className="text-muted-foreground">/{TOTAL_EPOCHS}</span>
            </p>
          </div>
          <dl className="flex gap-6">
            <div>
              <dt className="t-eyebrow">Loss</dt>
              <dd className="font-mono tabular text-sm text-foreground">
                {lastLoss === null ? "—" : lastLoss.toFixed(4)}
              </dd>
            </div>
            <div>
              <dt className="t-eyebrow">Accuracy</dt>
              <dd className="font-mono tabular text-sm text-foreground">
                {lastAcc === null ? "—" : `${(lastAcc * 100).toFixed(1)}%`}
              </dd>
            </div>
            <div>
              <dt className="t-eyebrow">Elapsed</dt>
              <dd className="font-mono tabular text-sm text-foreground">
                {elapsed}s
              </dd>
            </div>
          </dl>
        </div>

        {/* Completion is fill, not hue — the bar never changes colour. */}
        <div
          className="h-1.5 w-full overflow-hidden rounded-full bg-surface-sunken"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={TOTAL_EPOCHS}
          aria-valuenow={shown}
          aria-label="Training progress"
        >
          <div
            className="h-full rounded-full bg-mark transition-[width] duration-500 ease-linear"
            style={{ width: `${progress}%` }}
          />
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <Curve
            label="Loss"
            values={loss}
            domain={LOSS_DOMAIN}
            ticks={[0, 0.5, 1]}
            format={(v) => v.toFixed(2)}
          />
          <Curve
            label="Accuracy"
            values={acc}
            domain={ACC_DOMAIN}
            ticks={[0.3, 0.65, 1]}
            format={(v) => v.toFixed(2)}
          />
        </div>

        <p className="truncate rounded-lg border border-line bg-surface-sunken px-3 py-2 font-mono tabular text-[11px] text-muted-foreground">
          {shown === 0
            ? "waiting for start · logs are written to the project database"
            : `[epoch ${String(shown).padStart(2, "0")}/${TOTAL_EPOCHS}] loss ${lastLoss.toFixed(
                4,
              )}  acc ${lastAcc.toFixed(4)}  lr 0.0010`}
        </p>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={start}
            disabled={status === "running"}
            className={`rounded-md px-3.5 py-1.5 text-[13px] font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-mark ${
              status === "running"
                ? "cursor-not-allowed bg-muted text-muted-foreground"
                : status === "finished"
                  ? "border border-line bg-surface text-foreground hover:bg-muted"
                  : "bg-mark text-mark-ink hover:bg-mark-strong"
            }`}
          >
            {status === "running"
              ? "Training…"
              : status === "finished"
                ? "Replay this run"
                : "Start training"}
          </button>
          <span className="text-xs text-muted-foreground">
            Runs in a separate process on your machine. Nothing leaves it.
          </span>
        </div>
      </div>
    </section>
  );
}

"use client"

import React from "react"
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

/**
 * Small multiples, one metric per chart.
 *
 * Each chart plots a single series, so it does not need its own colour to tell
 * series apart — it gets the mark, and the metric name is the chart's title
 * rather than a legend repeating the one line below it. The previous version
 * drew black lines on a hardcoded white card, which was invisible in the dark
 * theme.
 */
const MetricLogsViewer = ({
    metricLogs,
    shownMetrics,
}: {
    // Values can be null: a diverged epoch records NaN, which has no JSON
    // representation, so the API sends null rather than failing the request.
    metricLogs: Record<string, Record<string, number | null>>
    shownMetrics?: string[] | null
}) => {
    if (!metricLogs || Object.keys(metricLogs).length === 0) return null

    const data: Record<string, number | null>[] = Object.entries(metricLogs).map(([epoch, metrics]) => ({
        epoch: parseInt(epoch) + 1, // Add 1 to start x-axis from 1
        ...Object.fromEntries(
            Object.entries(metrics).map(([key, value]) => [
                key,
                // A diverged epoch is a gap in the line, not a zero and not a
                // crash: plotting NaN as 0 would draw a dip that never happened.
                typeof value === "number" && Number.isFinite(value) ? Number(value.toFixed(2)) : null,
            ])
        ),
    }))

    const metrics = Object.keys(data[0]).filter(
        (key) => key !== "epoch" && (shownMetrics ? shownMetrics.includes(key) : true)
    )

    if (metrics.length === 0) return null

    const axisTick = { fontSize: 10, fill: "var(--muted-foreground)", fontFamily: "var(--font-mono)" }

    return (
        <div className="grid max-h-[320px] grid-cols-1 gap-3 overflow-y-auto sm:grid-cols-2 xl:grid-cols-3">
            {metrics.map((metric) => {
                // The last epoch that actually produced a number, so the
                // headline figure is not "—" merely because the final one
                // diverged.
                const latest = [...data].reverse().find((row) => typeof row[metric] === "number")?.[metric]
                // A diverged run can leave a single surviving point, and a line
                // through one point draws nothing at all -- the chart looked
                // empty while the data was fine. Dots appear only when there
                // are gaps, so a healthy series stays a clean line.
                const hasGaps = data.some((row) => row[metric] === null)
                return (
                    <div key={metric} className="bg-surface-sunken rounded-md border p-2">
                        <div className="mb-1 flex items-baseline justify-between gap-2 px-1">
                            <span className="t-eyebrow truncate">{metric}</span>
                            <span className="tabular font-mono text-xs">
                                {typeof latest === "number" ? latest.toFixed(2) : "—"}
                            </span>
                        </div>
                        <ResponsiveContainer width="100%" height={140}>
                            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
                                <CartesianGrid strokeDasharray="2 3" stroke="var(--border)" vertical={false} />
                                <XAxis
                                    dataKey="epoch"
                                    scale="auto"
                                    padding={{ left: 0, right: 0 }}
                                    tick={axisTick}
                                    stroke="var(--border)"
                                    tickLine={false}
                                />
                                <YAxis tick={axisTick} stroke="var(--border)" tickLine={false} width={44} />
                                <Tooltip
                                    contentStyle={{
                                        fontSize: "11px",
                                        background: "var(--popover)",
                                        border: "1px solid var(--border)",
                                        borderRadius: "var(--radius-sm)",
                                        color: "var(--popover-foreground)",
                                    }}
                                    labelFormatter={(label) => `Epoch ${label}`}
                                    formatter={(value) => [
                                        (typeof value === "number" ? value : Number(value ?? 0)).toFixed(2),
                                        metric,
                                    ]}
                                />
                                <Line
                                    type="monotone"
                                    dataKey={metric}
                                    stroke="var(--mark)"
                                    strokeWidth={1.5}
                                    dot={hasGaps ? { r: 2 } : false}
                                    activeDot={{ r: 3 }}
                                    isAnimationActive={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )
            })}
        </div>
    )
}

export default MetricLogsViewer

import { DatabaseIcon, PieChartIcon } from "lucide-react"
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts"

import { EmptyState } from "@/components/ui/empty-state"
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel"
import useDatasets from "@/lib/use-datasets"

// Training / validation / test are three parts of one dataset, so they read as
// one hue at three steps rather than three unrelated colours.
const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)"]

const DataDistribution: React.FC<{ projectId: number }> = ({ projectId }) => {
    const { datasets } = useDatasets(projectId)
    const totalItems = Object.values(datasets).reduce((sum, dataset) => sum + (dataset.info?.num_total ?? 0), 0)

    if (totalItems === 0) {
        return (
            <Panel>
                <PanelHeader icon={PieChartIcon} title="Dataset split" />
                <EmptyState
                    compact
                    icon={DatabaseIcon}
                    title="No images yet"
                    description="Upload images on the Dataset stage to see how they're split."
                />
            </Panel>
        )
    }

    const renderCustomizedLabel = () => {
        return (
            <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" fill="var(--foreground)">
                <tspan x="50%" dy="-0.4em" fontSize="11" fill="var(--muted-foreground)" letterSpacing="0.08em">
                    TOTAL
                </tspan>
                <tspan
                    x="50%"
                    dy="1.5em"
                    fontSize="20"
                    fontWeight="500"
                    fontFamily="var(--font-mono)"
                    fill="var(--foreground)"
                >
                    {totalItems.toLocaleString()}
                </tspan>
            </text>
        )
    }

    return (
        <Panel>
            <PanelHeader icon={PieChartIcon} title="Dataset split" />
            <PanelBody className="p-3">
                <div className="h-[176px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={Object.entries(datasets).map(([name, { info }]) => ({
                                    name,
                                    value: info?.num_total ?? 0,
                                }))}
                                cx="50%"
                                cy="50%"
                                innerRadius={52}
                                outerRadius={76}
                                paddingAngle={1}
                                dataKey="value"
                                labelLine={false}
                                stroke="var(--card)"
                                strokeWidth={2}
                            >
                                {Object.keys(datasets).map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                                {renderCustomizedLabel()}
                            </Pie>
                            <Tooltip
                                contentStyle={{
                                    fontSize: "12px",
                                    background: "var(--popover)",
                                    border: "1px solid var(--border)",
                                    borderRadius: "var(--radius-sm)",
                                    color: "var(--popover-foreground)",
                                }}
                                itemStyle={{ color: "var(--popover-foreground)" }}
                                formatter={(value, name) => {
                                    const numericValue = typeof value === "number" ? value : Number(value ?? 0)
                                    return [
                                        `${numericValue.toLocaleString()} (${((numericValue / totalItems) * 100).toFixed(1)}%)`,
                                        String(name),
                                    ]
                                }}
                            />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
                <dl className="mt-2 space-y-1">
                    {Object.entries(datasets).map(([name, { info }], index) => (
                        <div key={name} className="flex items-center gap-2 text-xs">
                            <span
                                aria-hidden
                                className="size-2 shrink-0 rounded-full"
                                style={{ backgroundColor: COLORS[index % COLORS.length] }}
                            />
                            <dt className="flex-1">{name}</dt>
                            <dd className="text-muted-foreground tabular font-mono">
                                {(info?.num_total ?? 0).toLocaleString()}
                            </dd>
                            <dd className="tabular w-9 text-right font-mono">
                                {Math.round(((info?.num_total ?? 0) / totalItems) * 100)}%
                            </dd>
                        </div>
                    ))}
                </dl>
            </PanelBody>
        </Panel>
    )
}

export default DataDistribution

"use client"

import { useQuery } from "@tanstack/react-query"
import { AlertTriangleIcon, ChartBarIcon, TagsIcon } from "lucide-react"

import { EmptyState } from "@/components/ui/empty-state"
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { getJson } from "@/lib/api"
import { qk } from "@/lib/query-keys"
import { ClassDistribution as Distribution } from "@/types"

/**
 * How many annotations each class has.
 *
 * The split chart next to this one answers "how much data", which a project can
 * pass while still being unlearnable: 60/20/20 tells you nothing about a class
 * with four examples sitting beside one with four hundred. Bars are drawn
 * relative to the largest class, so imbalance is a shape rather than a number
 * to compare by eye.
 *
 * Two states are called out because they are silent failures otherwise: a class
 * with nothing in the training subset can never be learned, and annotations
 * naming a label the project no longer lists still reach the trainer.
 */
const ClassDistribution: React.FC<{ projectId: number }> = ({ projectId }) => {
    const { data, isPending } = useQuery({
        queryKey: qk.classDistribution(projectId),
        queryFn: () => getJson<Distribution>(`/api/projects/${projectId}/class_distribution`),
        enabled: projectId !== null && projectId !== undefined,
    })

    const classes = [...(data?.classes ?? [])].sort((a, b) => b.total - a.total)

    // Counting annotations means opening the project database and walking it,
    // so this answers later than the panels around it. Until it does, an
    // absent answer is not the same as an empty one: a project with 2,501
    // polygons in it displayed "Nothing labelled yet" for as long as the query
    // took, which reads as data loss.
    if (isPending) {
        return (
            <Panel>
                <PanelHeader icon={ChartBarIcon} title="Class balance" />
                <PanelBody className="space-y-3">
                    {[0, 1, 2].map((row) => (
                        <div key={row} className="space-y-1.5">
                            <Skeleton className="h-3 w-32" />
                            <Skeleton className="h-2 w-full" />
                        </div>
                    ))}
                </PanelBody>
            </Panel>
        )
    }

    if (classes.length === 0) {
        return (
            <Panel>
                <PanelHeader icon={ChartBarIcon} title="Class balance" />
                <EmptyState
                    compact
                    icon={TagsIcon}
                    title="Nothing labelled yet"
                    description="Label a few images, or upload a set that brings its own annotations."
                />
            </Panel>
        )
    }

    const largest = Math.max(...classes.map((row) => row.total), 1)
    const totalAnnotations = classes.reduce((sum, row) => sum + row.total, 0)

    return (
        <Panel>
            <PanelHeader
                icon={ChartBarIcon}
                title="Class balance"
                actions={
                    <span className="text-muted-foreground font-mono text-xs">{totalAnnotations.toLocaleString()}</span>
                }
            />
            <PanelBody className="space-y-2 p-3">
                {classes.map((row) => {
                    const untrained = row.train === 0
                    return (
                        <div key={row.name} className="space-y-1">
                            <div className="flex items-center gap-2 text-xs">
                                <span
                                    aria-hidden
                                    className="size-2 shrink-0 rounded-full"
                                    style={{ backgroundColor: row.color ?? "var(--muted-foreground)" }}
                                />
                                <span className="flex-1 truncate" title={row.name}>
                                    {row.name}
                                </span>
                                {!row.known && (
                                    <Tooltip>
                                        <TooltipTrigger asChild>
                                            <AlertTriangleIcon className="text-muted-foreground size-3.5" />
                                        </TooltipTrigger>
                                        <TooltipContent className="max-w-56 text-xs">
                                            These annotations name a label the project no longer has. Training still
                                            sees them.
                                        </TooltipContent>
                                    </Tooltip>
                                )}
                                <span className="text-muted-foreground tabular font-mono">
                                    {row.total.toLocaleString()}
                                </span>
                            </div>
                            <div className="bg-muted h-1.5 overflow-hidden rounded-full">
                                <div
                                    className="h-full rounded-full"
                                    style={{
                                        width: `${(row.total / largest) * 100}%`,
                                        backgroundColor: row.color ?? "var(--chart-1)",
                                    }}
                                />
                            </div>
                            <div className="text-muted-foreground flex gap-3 font-mono text-[11px]">
                                <span className={untrained ? "text-amber-600 dark:text-amber-500" : undefined}>
                                    train {row.train.toLocaleString()}
                                </span>
                                <span>val {row.validation.toLocaleString()}</span>
                                <span>test {row.test.toLocaleString()}</span>
                                {untrained && <span className="text-amber-600 dark:text-amber-500">never trained</span>}
                            </div>
                        </div>
                    )
                })}
                {data && data.unlabeled.total > 0 && (
                    <p className="text-muted-foreground border-t pt-2 text-xs">
                        {data.unlabeled.total.toLocaleString()} image
                        {data.unlabeled.total === 1 ? "" : "s"} with no annotation
                    </p>
                )}
            </PanelBody>
        </Panel>
    )
}

export default ClassDistribution

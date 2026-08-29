import { ArrowRight, Copy, LineChart } from "lucide-react"
import React from "react"

import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { EmptyState } from "@/components/ui/empty-state"
import { Stat } from "@/components/ui/panel"
import { TrainingStatusBadge } from "@/components/ui/status-badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useToast } from "@/components/ui/use-toast"
import { DetailedTrainingSession } from "@/types"

import MetricLogsViewer from "./metric-logs-viewer"

interface ViewTrainingDetailsProps {
    session: DetailedTrainingSession | null
    isOpen: boolean
    onOpenChange: (open: boolean) => void
    getStatusColor: (status: string) => string
    calculateTrainingTime: (startedAt: string, endedAt: string | null) => string
    goToModelPage: (modelName: string) => void
}

export default function ViewTrainingDetails({
    session,
    isOpen,
    onOpenChange,
    getStatusColor,
    calculateTrainingTime,
    goToModelPage,
}: ViewTrainingDetailsProps) {
    const { toast } = useToast()

    const copyLogs = () => {
        if (session?.training_logs) {
            navigator.clipboard.writeText(session.training_logs)
            toast({
                description: "Training logs copied to the clipboard.",
            })
        }
    }

    const formatLocalTime = (utcTime: string) => {
        // Parse UTC time string and force UTC+0 interpretation
        const date = new Date(utcTime + "Z")
        // Convert to local timezone
        return date.toLocaleString(undefined, {
            year: "numeric",
            month: "numeric",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: true,
            timeZoneName: "short",
        })
    }

    return (
        <Dialog open={isOpen} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-3xl">
                <DialogHeader>
                    <DialogTitle>{session?.name}</DialogTitle>
                </DialogHeader>
                <div className="grid gap-4">
                    {/* Run parameters are numbers you compare between runs, so
                        they use the Stat pattern: eyebrow label, mono value. */}
                    <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
                        <Stat label="Learning rate" value={session?.params.learning_rate ?? "—"} />
                        <Stat label="Batch size" value={session?.params.batch_size ?? "—"} />
                        <Stat label="Epochs" value={session?.params.epochs ?? "—"} />
                        <Stat label="Model variant" value={session?.params.model_variant || "—"} mono={false} />
                        <Stat label="Pretrained model" value={session?.params.pretrained_model || "—"} mono={false} />
                        <div className="min-w-0 space-y-1">
                            <p className="t-eyebrow">Status</p>
                            <TrainingStatusBadge status={session?.status || ""} />
                        </div>
                        <Stat label="Started" value={session?.started_at ? formatLocalTime(session.started_at) : "—"} />
                        {session?.ended_at && <Stat label="Ended" value={formatLocalTime(session.ended_at)} />}
                        {session?.ended_at && (
                            <Stat
                                label="Duration"
                                value={calculateTrainingTime(session.started_at, session.ended_at)}
                            />
                        )}
                    </div>
                    {session?.status.toLowerCase() === "finished" && session?.model?.name && (
                        <div>
                            <Button onClick={() => goToModelPage(session.model.name)} size="sm" variant="outline">
                                Go to the model
                                <ArrowRight />
                            </Button>
                        </div>
                    )}
                    <Tabs defaultValue="metrics" className="relative w-full">
                        <TabsList>
                            <TabsTrigger value="metrics">Metrics</TabsTrigger>
                            {session?.training_logs && <TabsTrigger value="logs">Logs</TabsTrigger>}
                        </TabsList>
                        <TabsContent value="metrics">
                            {session?.metric_logs ? (
                                <MetricLogsViewer metricLogs={session.metric_logs} />
                            ) : (
                                <EmptyState compact icon={LineChart} title="No metrics recorded for this run" />
                            )}
                        </TabsContent>
                        {session?.training_logs && (
                            <TabsContent value="logs" className="relative max-w-full">
                                <div className="relative">
                                    <Button
                                        variant="outline"
                                        size="icon-sm"
                                        aria-label="Copy logs"
                                        className="absolute top-2 right-2 z-10"
                                        onClick={copyLogs}
                                    >
                                        <Copy />
                                    </Button>
                                    <textarea
                                        readOnly
                                        value={session.training_logs}
                                        className="bg-surface-sunken h-[240px] w-full resize-none rounded-md border p-2 font-mono text-xs"
                                    />
                                </div>
                            </TabsContent>
                        )}
                    </Tabs>
                </div>
            </DialogContent>
        </Dialog>
    )
}

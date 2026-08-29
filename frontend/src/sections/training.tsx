"use client"

import { ArrowRight, ChevronRight, CircuitBoard, Play, Square } from "lucide-react"
import { useRouter } from "next/navigation"
import React, { useEffect, useState } from "react"

import { Button } from "@/components/ui/button"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { EmptyState } from "@/components/ui/empty-state"
import { Notices } from "@/components/ui/notices"
import { Panel, PanelBody, PanelFooter, PanelHeader, Stat } from "@/components/ui/panel"
import { TrainingStatusBadge } from "@/components/ui/status-badge"
import { useToast } from "@/components/ui/use-toast"
import { api } from "@/lib/api"
import { isActiveStatus } from "@/lib/status"
import useDatasets from "@/lib/use-datasets"
import useProject from "@/lib/use-project"
import { useTraining } from "@/lib/use-training"
import { getStatusColor } from "@/lib/utils"
import { DetailedTrainingSession, TrainingParams, TrainingSession } from "@/types"

import MetricLogsViewer from "./metric-logs-viewer"
import { NewTrainingDialog } from "./new-training-dialog"
import ViewTrainingDetails from "./view-training-details"

/** Below this, a subset is too small to say much -- but it is advice, not a rule. */
const RECOMMENDED_IMAGES_PER_SUBSET = 10

const formatStartedAt = (startedAt: string) =>
    new Date(startedAt + "Z").toLocaleString(undefined, {
        month: "numeric",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        hour12: true,
    })

function Training({ projectId }: { projectId: number }) {
    const { datasets, invalidateDatasets } = useDatasets(projectId)
    const { project, modelVariants } = useProject(projectId)
    const [isDialogOpen, setIsDialogOpen] = useState<boolean>(false)
    const [isLogsDialogOpen, setIsLogsDialogOpen] = useState<boolean>(false)
    const [isStopDialogOpen, setIsStopDialogOpen] = useState<boolean>(false)
    const [selectedSession, setSelectedSession] = useState<DetailedTrainingSession | null>(null)
    const router = useRouter()
    const { toast } = useToast()

    // A recommendation rather than a gate. Refusing to start left people with
    // a button they could not press and no way to find out why it mattered --
    // and a small test set still trains a perfectly good model, it just
    // measures it less reliably. Training with nothing at all is what actually
    // fails, so only that is blocked.
    const counts = Object.fromEntries(
        Object.entries(datasets).map(([name, dataset]) => [name, dataset.info?.num_total ?? 0])
    )
    const shortSubsets = Object.entries(counts)
        .filter(([, total]) => total < RECOMMENDED_IMAGES_PER_SUBSET)
        .map(([name]) => name)
    const hasTrainingImages = (counts["Training"] ?? 0) > 0
    // Offered when the test set is the only thing that is short and validation
    // has something to copy: the common case of "40 training, 10 validation,
    // no test".
    const canCopyValidationToTest =
        (counts["Test"] ?? 0) < RECOMMENDED_IMAGES_PER_SUBSET && (counts["Validation"] ?? 0) > 0

    const {
        startTraining,
        getTrainingSession,
        terminateTraining,
        trainingSessions,
        lastTrainingSession,
        isLoading,
        error,
    } = useTraining(projectId)

    const [isCopying, setIsCopying] = useState(false)

    const copyValidationToTest = async () => {
        setIsCopying(true)
        try {
            const { data } = await api.post(`/api/projects/${projectId}/data_items/copy_subset`, {
                from_subset: 1,
                to_subset: 2,
            })
            invalidateDatasets()
            toast({
                title: `Copied ${data.copied} images into the test set`,
                description: "They are the same images as validation, so the test score will read high.",
            })
        } catch (error) {
            toast({
                title: "Could not copy the images",
                description:
                    (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
                    "Try again, or upload a test set of your own.",
                variant: "destructive",
            })
        } finally {
            setIsCopying(false)
        }
    }

    const handleStartTraining = async (params: TrainingParams) => {
        if (!project || !modelVariants) return

        const selectedVariant = modelVariants?.find(
            (variant: any) =>
                variant.model_architecture === params.model_architecture && variant.model_size === params.model_size
        )

        if (selectedVariant) {
            const trainingParams = {
                ...params,
                model_architecture: selectedVariant.model_architecture,
                model_size: selectedVariant.model_size,
                model_variant: selectedVariant.name,
            }
            const response = await startTraining(trainingParams)
            if (response) {
                setIsDialogOpen(false)
            }
        }
    }

    const viewSessionDetails = async (sessionId: number) => {
        const session = await getTrainingSession(sessionId)
        if (session) {
            setSelectedSession(session)
            setIsLogsDialogOpen(true)
        }
    }

    const goToModelsPage = () => {
        router.push(`/projects/models?projectId=${projectId}`)
    }

    const goToModelPage = (modelName: string) => {
        window.location.href = `/projects/models?projectId=${projectId}&search=${modelName}`
    }

    const isTraining = lastTrainingSession && isActiveStatus(lastTrainingSession.status)

    const handleStopTraining = async () => {
        if (lastTrainingSession) {
            const response = await terminateTraining(lastTrainingSession.id)
            if (response) {
                setIsStopDialogOpen(false)
            }
        }
    }

    useEffect(() => {
        let interval: NodeJS.Timeout
        if (isLogsDialogOpen && selectedSession) {
            interval = setInterval(async () => {
                const session = await getTrainingSession(selectedSession.id)
                if (session) {
                    setSelectedSession(session)
                }
            }, 5000)
        }
        return () => clearInterval(interval)
    }, [isLogsDialogOpen, selectedSession])

    const calculateTrainingTime = (startedAt: string, endedAt: string | null) => {
        const start = new Date(startedAt)
        const end = endedAt ? new Date(endedAt) : new Date()
        const diff = end.getTime() - start.getTime()
        const minutes = Math.floor(diff / 60000)
        const seconds = Math.floor((diff % 60000) / 1000)
        return `${minutes}m ${seconds}s`
    }

    const epochsDone = lastTrainingSession?.metric_logs ? Object.keys(lastTrainingSession.metric_logs).length : 0
    const epochsTotal = lastTrainingSession?.params?.epochs ?? 0

    const startControl = isTraining ? (
        <Button variant="outline" size="sm" onClick={() => setIsStopDialogOpen(true)}>
            <Square />
            Stop training
        </Button>
    ) : (
        <>
            <Button size="sm" disabled={!hasTrainingImages} onClick={() => setIsDialogOpen(true)}>
                <Play />
                Start training
            </Button>
            <NewTrainingDialog
                projectId={projectId}
                isOpen={isDialogOpen}
                onOpenChange={setIsDialogOpen}
                onStartTraining={handleStartTraining}
            />
        </>
    )

    return (
        <div className="grid gap-4">
            <Panel>
                <PanelHeader
                    icon={CircuitBoard}
                    title={lastTrainingSession ? "Latest run" : "Training"}
                    description={lastTrainingSession?.name}
                    actions={
                        <>
                            {lastTrainingSession && (
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => viewSessionDetails(lastTrainingSession.id)}
                                >
                                    View details
                                    <ChevronRight />
                                </Button>
                            )}
                            {startControl}
                        </>
                    }
                />

                <PanelBody className="space-y-4">
                    {/* A blocked precondition is stated once, next to the
                        control it blocks — not as a standing amber banner that
                        replaces the button entirely. */}

                    <Notices
                        scope={String(projectId)}
                        notices={[
                            ...(shortSubsets.length > 0 && !isTraining
                                ? [
                                      {
                                          key: `short-subsets:${shortSubsets.join(",")}`,
                                          level: "info" as const,
                                          title: `${shortSubsets.join(", ")} ${
                                              shortSubsets.length === 1 ? "has" : "have"
                                          } fewer than ${RECOMMENDED_IMAGES_PER_SUBSET} images`,
                                          detail:
                                              "Training still works; the numbers it reports will be noisy." +
                                              (canCopyValidationToTest
                                                  ? " Copying validation into test gets you a number to look at — but one measured on images the model was tuned against, so treat it as optimistic."
                                                  : ""),
                                          action: canCopyValidationToTest ? (
                                              <Button
                                                  variant="outline"
                                                  size="sm"
                                                  disabled={isCopying}
                                                  onClick={copyValidationToTest}
                                              >
                                                  {isCopying ? "Copying…" : "Copy validation into test"}
                                              </Button>
                                          ) : undefined,
                                      },
                                  ]
                                : []),
                            // What the run's own numbers say to change, from
                            // anylearning/training/diagnostics.py.
                            ...(lastTrainingSession?.advice ?? []).map((item) => ({
                                key: `advice:${lastTrainingSession?.id}:${item.title}`,
                                level: item.level,
                                title: item.title,
                                detail: item.detail,
                            })),
                        ]}
                    />

                    {lastTrainingSession ? (
                        <>
                            <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
                                <div className="space-y-1">
                                    <p className="t-eyebrow">Status</p>
                                    <TrainingStatusBadge status={lastTrainingSession.status} />
                                </div>
                                <Stat label="Started" value={formatStartedAt(lastTrainingSession.started_at)} />
                                <Stat
                                    label={lastTrainingSession.ended_at ? "Duration" : "Running for"}
                                    value={calculateTrainingTime(
                                        lastTrainingSession.started_at,
                                        lastTrainingSession.ended_at
                                    )}
                                />
                                {epochsTotal > 0 && <Stat label="Epochs" value={`${epochsDone}/${epochsTotal}`} />}
                            </div>

                            {lastTrainingSession.metric_logs && (
                                <MetricLogsViewer
                                    metricLogs={lastTrainingSession.metric_logs}
                                    shownMetrics={[
                                        "Validation Loss",
                                        "mAP",
                                        "IoU",
                                        "F1",
                                        "Accuracy",
                                        "Validation IoU",
                                        "Validation F1",
                                        "Validation Accuracy",
                                        "Validation mAP",
                                        "Validation mAP@0.5",
                                    ]}
                                />
                            )}
                        </>
                    ) : (
                        <EmptyState
                            icon={CircuitBoard}
                            title="No training runs yet"
                            description={
                                hasTrainingImages
                                    ? "Start a run to train a model on this project's images. Metrics appear here as it goes."
                                    : "Upload images to the training set, then start a run."
                            }
                        />
                    )}

                    {error && <p className="text-fail text-xs">{error.message || "Something went wrong."}</p>}

                    {trainingSessions && trainingSessions.length > 1 && (
                        <div className="border-t pt-3">
                            <p className="t-eyebrow mb-2">Earlier runs</p>
                            {isLoading ? (
                                <p className="text-muted-foreground text-xs">Loading runs…</p>
                            ) : (
                                <ul className="space-y-0.5">
                                    {trainingSessions.slice(1, 4).map((session: TrainingSession) => (
                                        <li key={session.id}>
                                            <button
                                                type="button"
                                                onClick={() => viewSessionDetails(session.id)}
                                                className="hover:bg-accent focus-visible:ring-ring flex w-full items-center gap-3 rounded-md px-2 py-1.5 text-left focus-visible:ring-2 focus-visible:outline-none"
                                            >
                                                <TrainingStatusBadge status={session.status} />
                                                <span className="min-w-0 flex-1 truncate text-xs">{session.name}</span>
                                                <span className="text-muted-foreground tabular shrink-0 font-mono text-[0.6875rem]">
                                                    {formatStartedAt(session.started_at)}
                                                    {session.ended_at &&
                                                        ` · ${calculateTrainingTime(session.started_at, session.ended_at)}`}
                                                </span>
                                                <ChevronRight className="text-muted-foreground size-3.5 shrink-0" />
                                            </button>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    )}
                </PanelBody>

                <PanelFooter>
                    <Button onClick={goToModelsPage} size="sm" variant="ghost" className="ml-auto">
                        Go to models
                        <ArrowRight />
                    </Button>
                </PanelFooter>
            </Panel>

            <ViewTrainingDetails
                session={selectedSession}
                isOpen={isLogsDialogOpen}
                onOpenChange={setIsLogsDialogOpen}
                getStatusColor={getStatusColor}
                calculateTrainingTime={calculateTrainingTime}
                goToModelPage={goToModelPage}
            />

            <Dialog open={isStopDialogOpen} onOpenChange={setIsStopDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Stop this training run?</DialogTitle>
                        <DialogDescription>
                            The run ends where it is and no model is saved. The images and labels are untouched.
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setIsStopDialogOpen(false)}>
                            Keep training
                        </Button>
                        <Button variant="destructive" onClick={handleStopTraining}>
                            Stop training
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    )
}

export default Training

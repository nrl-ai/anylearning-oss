"use client"

import { Copy, Download, X } from "lucide-react"
import React, { useState } from "react"

import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { toast } from "@/components/ui/use-toast"
import { api } from "@/lib/api"
import { Model } from "@/types"

interface TryModelDialogProps {
    isOpen: boolean
    onOpenChange: (open: boolean) => void
    selectedModel: Model | undefined
    projectId: number
}

function TryModelDialog({ isOpen, onOpenChange, selectedModel, projectId }: TryModelDialogProps) {
    const [inferenceResult, setInferenceResult] = useState<any>(null)
    const [inferenceImage, setInferenceImage] = useState<string | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const [activeTab, setActiveTab] = useState("upload")

    const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0]
        if (file && selectedModel) {
            await runInference(file)
        }
    }

    const handleRandomTestSample = async () => {
        try {
            setIsLoading(true)
            const timestamp = new Date().getTime()
            // A cache-busting param so a second click really re-rolls the sample.
            const response = await api.get(`/api/projects/${projectId}/data_items/random_test_sample`, {
                params: { t: timestamp },
                responseType: "blob",
            })

            const file = new File([response.data], "test_sample.jpg", { type: "image/jpeg" })
            // Plenty of projects never fill the test split, and the server
            // falls back to the validation set rather than refusing. Say so:
            // an image the model was measured on is a weaker check than one it
            // has never seen, and the reader should know which they are looking
            // at.
            if (response.headers?.["x-anylearning-subset"] === "validation") {
                toast({
                    title: "From your validation set",
                    description:
                        "This project has no test images, so this one comes from validation — the model was " +
                        "measured on it during training. Upload an image of your own for a stricter check.",
                })
            }
            await runInference(file)
        } catch (error) {
            const detail =
                (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
                "Could not load an image from this project."
            toast({
                title: "No image to try",
                description: detail,
                variant: "destructive",
            })
        } finally {
            setIsLoading(false)
        }
    }

    const runInference = async (file: File) => {
        if (!selectedModel) return

        try {
            setIsLoading(true)
            const formData = new FormData()
            formData.append("file", file)

            const { data: result } = await api.post(
                `/api/projects/${projectId}/models/${selectedModel.id}/inference`,
                formData
            )
            setInferenceResult(result.results)
            setInferenceImage(result.visualization_image)
        } catch (error) {
            toast({
                title: "Error",
                description: "Failed to perform model inference",
                variant: "destructive",
            })
        } finally {
            setIsLoading(false)
        }
    }

    const handleClearInference = () => {
        setInferenceResult(null)
        setInferenceImage(null)
        const input = document.querySelector('input[type="file"]') as HTMLInputElement
        if (input) {
            input.value = ""
        }
    }

    const handleCopyResults = () => {
        if (inferenceResult) {
            navigator.clipboard.writeText(JSON.stringify(inferenceResult, null, 2))
            toast({
                title: "Copied",
                description: "Results copied to clipboard",
            })
        }
    }

    const handleDownloadImage = () => {
        if (inferenceImage && window?.pywebview?.api?.download_file) {
            // Extract file extension from base64 data
            const fileExtension = inferenceImage.split(";")[0].split("/")[1]
            window.pywebview.api.download_file(inferenceImage, `inference_result.${fileExtension}`)
        }
    }

    return (
        <Dialog open={isOpen} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-5xl">
                <DialogHeader>
                    <DialogTitle>Try {selectedModel?.name}</DialogTitle>
                </DialogHeader>

                <div
                    className={`grid gap-6 ${inferenceResult || inferenceImage ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"}`}
                >
                    <div className="flex flex-col space-y-4">
                        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                            <TabsList className="grid w-full grid-cols-2">
                                <TabsTrigger value="upload">Upload an image</TabsTrigger>
                                <TabsTrigger value="sample">Use a test image</TabsTrigger>
                            </TabsList>

                            <TabsContent value="upload">
                                <Panel>
                                    <PanelBody className="space-y-2">
                                        <Input
                                            type="file"
                                            accept=".png,.jpeg,.jpg,.bmp"
                                            onChange={handleImageUpload}
                                            disabled={isLoading}
                                            className="cursor-pointer"
                                        />
                                        <p className="text-muted-foreground text-xs">PNG, JPEG, JPG or BMP.</p>
                                    </PanelBody>
                                </Panel>
                            </TabsContent>

                            <TabsContent value="sample">
                                <Panel>
                                    <PanelBody>
                                        <Button
                                            onClick={handleRandomTestSample}
                                            className="w-full"
                                            disabled={isLoading}
                                            variant="outline"
                                        >
                                            Pick a random test image
                                        </Button>
                                    </PanelBody>
                                </Panel>
                            </TabsContent>
                        </Tabs>

                        {inferenceResult && (
                            <Panel>
                                <PanelHeader
                                    title="Result"
                                    actions={
                                        <Button size="sm" variant="ghost" onClick={handleCopyResults}>
                                            <Copy />
                                            Copy
                                        </Button>
                                    }
                                />
                                <pre className="bg-surface-sunken max-h-[400px] overflow-auto rounded-b-lg p-3 font-mono text-xs">
                                    {JSON.stringify(inferenceResult, null, 2)}
                                </pre>
                            </Panel>
                        )}
                    </div>

                    {(inferenceResult || inferenceImage) && (
                        <div className="flex flex-col space-y-4">
                            {isLoading && (
                                <div className="bg-surface-sunken rounded-lg border p-8 text-center">
                                    <p className="text-muted-foreground animate-pulse text-sm">Running the model…</p>
                                </div>
                            )}

                            {inferenceImage && (
                                <Panel>
                                    <PanelHeader
                                        title="Prediction"
                                        actions={
                                            <Button size="sm" variant="ghost" onClick={handleDownloadImage}>
                                                <Download />
                                                Download
                                            </Button>
                                        }
                                    />
                                    <div className="bg-surface-sunken rounded-b-lg p-3">
                                        <img
                                            src={inferenceImage}
                                            alt="The model's prediction drawn over the input image"
                                            className="h-auto w-full rounded-md"
                                        />
                                    </div>
                                </Panel>
                            )}

                            {(inferenceResult || inferenceImage) && (
                                <Button variant="outline" onClick={handleClearInference}>
                                    <X />
                                    Try another image
                                </Button>
                            )}
                        </div>
                    )}
                </div>
            </DialogContent>
        </Dialog>
    )
}

export default TryModelDialog

"use client"

import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { StepTabs, StepTabsContent, StepTabsList } from "@/components/ui/step-tabs"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"

const steps = [
    { value: "step1", label: "Step 1" },
    { value: "step2", label: "Step 2" },
    { value: "step3", label: "Step 3" },
    { value: "step4", label: "Complete" },
]

export function StepTabsExample() {
    const [activeStep, setActiveStep] = useState("step1")
    const [variant, setVariant] = useState<"numbered" | "dots">("numbered")
    const [isCompact, setIsCompact] = useState(false)

    const handleNext = () => {
        const currentIndex = steps.findIndex((step) => step.value === activeStep)
        if (currentIndex < steps.length - 1) {
            setActiveStep(steps[currentIndex + 1].value)
        }
    }

    const handlePrevious = () => {
        const currentIndex = steps.findIndex((step) => step.value === activeStep)
        if (currentIndex > 0) {
            setActiveStep(steps[currentIndex - 1].value)
        }
    }

    const toggleVariant = () => {
        setVariant(variant === "numbered" ? "dots" : "numbered")
    }

    const toggleCompact = () => {
        setIsCompact(!isCompact)
    }

    return (
        <div className="mx-auto max-w-3xl p-4">
            <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
                <h2 className="text-xl font-medium text-slate-800">Step Tabs Example</h2>
                <div className="flex items-center gap-3">
                    <div className="flex items-center space-x-2">
                        <Switch id="compact-mode" checked={isCompact} onCheckedChange={toggleCompact} />
                        <Label htmlFor="compact-mode" className="text-xs text-slate-600">
                            Compact
                        </Label>
                    </div>
                    <Button onClick={toggleVariant} variant="outline" size="sm" className="h-8 text-xs">
                        {variant === "numbered" ? "Use Dots" : "Use Numbers"}
                    </Button>
                </div>
            </div>

            <Card className="overflow-hidden border border-slate-100 shadow-none">
                <CardHeader className="border-b border-slate-100 bg-white px-4 py-3">
                    <CardTitle className="text-sm">Stepper Example</CardTitle>
                    <CardDescription className="text-xs">A multi-step process demonstration</CardDescription>
                </CardHeader>
                <CardContent className="p-4">
                    <StepTabs value={activeStep} onValueChange={setActiveStep} compact={isCompact}>
                        <div className="pb-2">
                            <StepTabsList
                                steps={steps}
                                currentValue={activeStep}
                                variant={variant}
                                className={cn("mb-4", isCompact && "py-0.5")}
                                compact={isCompact}
                            />
                        </div>

                        <StepTabsContent value="step1" className="animate-in fade-in duration-300">
                            <div className="space-y-3">
                                <h3 className="text-sm font-medium">Step 1: Getting Started</h3>
                                <p className="text-xs text-slate-600">
                                    This is the content for step 1. Click next to continue.
                                </p>
                            </div>
                        </StepTabsContent>

                        <StepTabsContent value="step2" className="animate-in fade-in duration-300">
                            <div className="space-y-3">
                                <h3 className="text-sm font-medium">Step 2: Configuration</h3>
                                <p className="text-xs text-slate-600">
                                    This is the content for step 2. Click next to continue or previous to go back.
                                </p>
                            </div>
                        </StepTabsContent>

                        <StepTabsContent value="step3" className="animate-in fade-in duration-300">
                            <div className="space-y-3">
                                <h3 className="text-sm font-medium">Step 3: Review</h3>
                                <p className="text-xs text-slate-600">
                                    This is the content for step 3. Click next to complete or previous to go back.
                                </p>
                            </div>
                        </StepTabsContent>

                        <StepTabsContent value="step4" className="animate-in fade-in duration-300">
                            <div className="space-y-3">
                                <h3 className="text-sm font-medium">Completed</h3>
                                <p className="text-xs text-slate-600">You have successfully completed all steps.</p>
                            </div>
                        </StepTabsContent>
                    </StepTabs>
                </CardContent>
                <CardFooter className="flex justify-between border-t border-slate-100 p-3">
                    <Button
                        variant="outline"
                        onClick={handlePrevious}
                        disabled={activeStep === "step1"}
                        size="sm"
                        className="h-7 text-xs"
                    >
                        Previous
                    </Button>
                    <Button
                        variant="default"
                        onClick={handleNext}
                        disabled={activeStep === "step4"}
                        size="sm"
                        className="h-7 text-xs"
                    >
                        {activeStep === "step3" ? "Complete" : "Next"}
                    </Button>
                </CardFooter>
            </Card>
        </div>
    )
}

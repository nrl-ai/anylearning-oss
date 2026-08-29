import React from "react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Step, Steps } from "@/components/ui/steps"

export function TrainingProcess() {
    return (
        <Card>
            <CardHeader>
                <CardTitle>Model Training Process</CardTitle>
                <CardDescription>Follow these steps to train your AI models</CardDescription>
            </CardHeader>
            <CardContent>
                <Steps>
                    <Step title="1. Label Data" description="Use AnyLabeling to label your data" />
                    <Step title="2. Import Data" description="Import labeled data to AnyLearning" />
                    <Step title="3. Train Model" description="Train your AI model" />
                    <Step title="4. Evaluate Model" description="Assess model performance" />
                    <Step title="5. Optimize and Deploy" description="Refine and deploy your model" />
                </Steps>
            </CardContent>
        </Card>
    )
}

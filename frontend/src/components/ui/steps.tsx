import React from "react"

import { cn } from "@/lib/utils"

interface StepProps {
    title: string
    description?: string
    isActive?: boolean
    isCompleted?: boolean
}

const Step: React.FC<StepProps> = ({ title, description, isActive, isCompleted }) => {
    return (
        <li
            className={cn(
                "flex items-center space-x-4",
                isActive && "text-primary",
                isCompleted && "text-muted-foreground"
            )}
        >
            <div
                className={cn(
                    "flex h-8 w-8 items-center justify-center rounded-full border",
                    isActive && "border-primary bg-primary text-primary-foreground",
                    isCompleted && "border-muted-foreground bg-muted-foreground text-background"
                )}
            >
                {isCompleted ? "✓" : ""}
            </div>
            <div>
                <h3 className="font-medium">{title}</h3>
                {description && <p className="text-muted-foreground text-sm">{description}</p>}
            </div>
        </li>
    )
}

interface StepsProps {
    children: React.ReactElement<StepProps> | React.ReactElement<StepProps>[]
    className?: string
    currentStep?: number
}

const Steps: React.FC<StepsProps> = ({ children, className, currentStep = 0 }) => {
    return (
        <ol className={cn("space-y-4", className)}>
            {React.Children.map(children, (child, index) =>
                React.cloneElement(child, {
                    isActive: index === currentStep,
                    isCompleted: index < currentStep,
                })
            )}
        </ol>
    )
}

export { Steps, Step }

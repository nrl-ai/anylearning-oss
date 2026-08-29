"use client"

import { CheckIcon, DotFilledIcon } from "@radix-ui/react-icons"
import { motion } from "framer-motion"
import { Tabs as TabsPrimitive } from "radix-ui"
import * as React from "react"

// Import framer-motion for animations
import { cn } from "@/lib/utils"

// Create a wrapper around TabsPrimitive.Root to handle value changes
interface StepTabsProps extends React.ComponentPropsWithoutRef<typeof TabsPrimitive.Root> {
    value?: string
    onValueChange?: (value: string) => void
    compact?: boolean
}

const StepTabs = React.forwardRef<React.ElementRef<typeof TabsPrimitive.Root>, StepTabsProps>(
    ({ className, value, onValueChange, compact = false, ...props }, ref) => {
        return (
            <TabsPrimitive.Root
                ref={ref}
                value={value}
                onValueChange={onValueChange}
                className={cn(compact && "step-tabs-compact", className)}
                {...props}
            />
        )
    }
)
StepTabs.displayName = "StepTabs"

interface StepTabsListProps extends React.ComponentPropsWithoutRef<typeof TabsPrimitive.List> {
    steps: { value: string; label: string }[]
    currentValue?: string
    variant?: "numbered" | "dots"
    compact?: boolean
}

const StepTabsList = React.forwardRef<React.ElementRef<typeof TabsPrimitive.List>, StepTabsListProps>(
    ({ className, steps, currentValue, variant = "numbered", compact = false, ...props }, ref) => {
        // Find the index of the current step
        const currentIndex = steps.findIndex((step) => step.value === currentValue)

        return (
            <div className={cn("w-full", compact && "mr-auto max-w-xl")}>
                <TabsPrimitive.List
                    ref={ref}
                    className={cn(
                        "relative flex w-full items-center justify-between",
                        compact ? "gap-1" : "gap-1.5",
                        className
                    )}
                    {...props}
                >
                    {steps.map((step, index) => {
                        // Determine if this step is active, completed, or upcoming
                        const isActive = step.value === currentValue
                        const isCompleted = currentIndex > index
                        const isUpcoming = currentIndex < index

                        return (
                            <React.Fragment key={step.value}>
                                {/* Step indicator with TabsPrimitive.Trigger for proper tab selection */}
                                <TabsPrimitive.Trigger
                                    value={step.value}
                                    className="focus-visible:ring-ring/60 m-0 rounded-md border-none bg-transparent p-1 outline-none focus-visible:ring-2"
                                >
                                    <motion.div
                                        className={cn("flex flex-col items-center", compact && "gap-1")}
                                        whileHover={{ scale: 1.03 }}
                                        transition={{ duration: 0.15 }}
                                    >
                                        <motion.div
                                            className={cn(
                                                "relative flex items-center justify-center rounded-full transition-all duration-200",
                                                variant === "numbered"
                                                    ? compact
                                                        ? "h-5 w-5"
                                                        : "h-6 w-6"
                                                    : compact
                                                      ? "h-3 w-3"
                                                      : "h-4 w-4",
                                                isActive && "bg-primary/5 text-primary",
                                                isCompleted && "bg-primary text-primary-foreground",
                                                isUpcoming && "bg-muted text-muted-foreground/60",
                                                !isActive && !isCompleted && !isUpcoming && "bg-muted"
                                            )}
                                            initial={{ opacity: 0, y: 3 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ duration: 0.2, delay: index * 0.03 }}
                                        >
                                            {isCompleted ? (
                                                <CheckIcon
                                                    className={cn(
                                                        variant === "numbered"
                                                            ? compact
                                                                ? "h-3 w-3"
                                                                : "h-3.5 w-3.5"
                                                            : compact
                                                              ? "h-2 w-2"
                                                              : "h-2.5 w-2.5"
                                                    )}
                                                />
                                            ) : variant === "numbered" ? (
                                                <span
                                                    className={cn(
                                                        "font-medium",
                                                        compact ? "text-xs" : "text-sm",
                                                        isActive && "text-primary"
                                                    )}
                                                >
                                                    {index + 1}
                                                </span>
                                            ) : (
                                                <DotFilledIcon
                                                    className={cn(
                                                        compact ? "h-2 w-2" : "h-2.5 w-2.5",
                                                        isActive && "text-primary",
                                                        isUpcoming && "text-muted-foreground/30"
                                                    )}
                                                />
                                            )}
                                        </motion.div>
                                        <motion.span
                                            className={cn(
                                                "text-[11px] font-medium transition-colors duration-200",
                                                isActive && "text-primary",
                                                isCompleted && "text-primary/80",
                                                isUpcoming && "text-muted-foreground/70"
                                            )}
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ duration: 0.2, delay: index * 0.03 + 0.05 }}
                                        >
                                            {step.label}
                                        </motion.span>
                                    </motion.div>
                                </TabsPrimitive.Trigger>

                                {/* Connector line between steps (except after the last step) */}
                                {index < steps.length - 1 && (
                                    <div className="relative flex-1">
                                        <div
                                            className={cn(
                                                "absolute top-1/2 w-full -translate-y-1/2",
                                                compact ? "h-[1px]" : "h-[1px]",
                                                "bg-border"
                                            )}
                                        />
                                        <motion.div
                                            className={cn(
                                                "absolute top-1/2 h-full w-full -translate-y-1/2",
                                                compact ? "h-[1px]" : "h-[1px]",
                                                isCompleted && index + 1 <= currentIndex
                                                    ? "bg-primary/50"
                                                    : "bg-transparent"
                                            )}
                                            initial={{ width: "0%" }}
                                            animate={{
                                                width: isCompleted && index + 1 <= currentIndex ? "100%" : "0%",
                                            }}
                                            transition={{ duration: 0.3 }}
                                        />
                                    </div>
                                )}
                            </React.Fragment>
                        )
                    })}
                </TabsPrimitive.List>
            </div>
        )
    }
)
StepTabsList.displayName = "StepTabsList"

const StepTabsContent = React.forwardRef<
    React.ElementRef<typeof TabsPrimitive.Content>,
    React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
    <TabsPrimitive.Content
        ref={ref}
        className={cn(
            "ring-offset-background focus-visible:ring-ring mt-4 focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-none",
            className
        )}
        {...props}
    />
))
StepTabsContent.displayName = "StepTabsContent"

export { StepTabs, StepTabsList, StepTabsContent }

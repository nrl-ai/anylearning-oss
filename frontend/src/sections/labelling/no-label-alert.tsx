"use client"

import { ArrowRight, TagIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { EmptyState } from "@/components/ui/empty-state"

type NoLabelsAlertProps = {
    projectId: number
}

/**
 * The gate you hit when a project has no classes yet.
 *
 * This is a missing precondition, not a failure, so it reads as an empty state
 * with one clear action rather than the previous warning triangle plus red
 * cross plus amber tip box — three alarm signals for "you haven't set this up".
 */
export default function NoLabelsAlert({ projectId }: NoLabelsAlertProps) {
    return (
        <div className="bg-background fixed inset-0 z-50 flex items-center justify-center p-6">
            <EmptyState
                icon={TagIcon}
                title="Add classes before labelling"
                description="Labelling needs at least one class to assign. Define them in the project overview, then come back."
                action={
                    <Button onClick={() => (window.location.href = `/projects/overview?projectId=${projectId}`)}>
                        Go to project overview
                        <ArrowRight />
                    </Button>
                }
            />
        </div>
    )
}

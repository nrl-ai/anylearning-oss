"use client"

import { useQuery } from "@tanstack/react-query"
import { FileText, Scale } from "lucide-react"
import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Markdown } from "@/components/ui/markdown"
import { getJson } from "@/lib/api"

/**
 * A shipped document, readable from inside the app.
 *
 * Legal documents must remain readable by someone who installed the app and
 * does not have a source checkout.
 */
export function LegalDocument({
    endpoint,
    title,
    description,
    label,
    icon: Icon,
}: {
    endpoint: string
    title: string
    description: string
    label: string
    icon: typeof FileText
}) {
    const [open, setOpen] = useState(false)
    const { data, error, isLoading } = useQuery({
        queryKey: ["legal-document", endpoint],
        queryFn: () => getJson<{ text: string }>(endpoint),
        enabled: open,
        staleTime: Infinity,
    })

    return (
        <>
            <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
                <Icon />
                {label}
            </Button>
            <Dialog open={open} onOpenChange={setOpen}>
                <DialogContent className="flex max-h-[80vh] flex-col sm:max-w-2xl">
                    <DialogHeader>
                        <DialogTitle>{title}</DialogTitle>
                        <DialogDescription>{description}</DialogDescription>
                    </DialogHeader>
                    <div className="min-h-0 flex-1 overflow-y-auto rounded-md border p-4">
                        {isLoading && <p className="text-muted-foreground text-xs">Loading…</p>}
                        {error && (
                            <p className="text-muted-foreground text-xs">This document is missing from this build.</p>
                        )}
                        {data && <Markdown text={data.text} className="text-muted-foreground" />}
                    </div>
                </DialogContent>
            </Dialog>
        </>
    )
}

export function SoftwareLicense() {
    return (
        <LegalDocument
            endpoint="/api/legal/license"
            title="Software license"
            description="AnyLearning is distributed under Apache License 2.0."
            label="Apache-2.0 license"
            icon={FileText}
        />
    )
}

export function ModelLicences() {
    return (
        <LegalDocument
            endpoint="/api/legal/model-policy"
            title="Model licences"
            description="Which model weights ship with AnyLearning, and the licences they carry."
            label="Model licences"
            icon={Scale}
        />
    )
}

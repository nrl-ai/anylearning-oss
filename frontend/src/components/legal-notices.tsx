"use client"

import { useQuery } from "@tanstack/react-query"
import { ChevronDown, Scale } from "lucide-react"
import { useMemo, useState } from "react"

import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { getJson } from "@/lib/api"
import { cn } from "@/lib/utils"

interface Component {
    name: string
    version: string
    text: string
}

/**
 * The third-party licence notices, readable from inside the app.
 *
 * AnyLearning redistributes PyTorch, OpenCV, detectron2 and about 150 other
 * packages. MIT, BSD and Apache 2.0 all require their notice to reach whoever
 * receives the binary, so this is an obligation being discharged, not an
 * About-box nicety.
 *
 * Presented as a list of components rather than one long document. The file is
 * two megabytes; scrolling it to find whether a particular package is in there
 * is not something anyone will do, and rendering its markdown source in a
 * <pre> showed "###" and ``` as literal characters, which reads as a bug.
 *
 * Each licence body stays monospaced and pre-wrapped: licence text is laid out
 * in fixed-width columns and reflowing it changes a legal document.
 */
export function LegalNotices() {
    const [open, setOpen] = useState(false)
    const [query, setQuery] = useState("")
    const [expanded, setExpanded] = useState<string | null>(null)

    const { data, error, isLoading } = useQuery({
        queryKey: ["legal-notices"],
        queryFn: () => getJson<{ components: Component[] }>("/api/legal/notices"),
        // Only fetched when someone opens it: two megabytes most sessions never
        // look at.
        enabled: open,
        staleTime: Infinity,
    })

    const components = useMemo(() => {
        const all = data?.components ?? []
        const needle = query.trim().toLowerCase()
        if (!needle) return all
        return all.filter((component) => component.name.toLowerCase().includes(needle))
    }, [data, query])

    return (
        <>
            <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
                <Scale />
                Third-party licences
            </Button>
            <Dialog open={open} onOpenChange={setOpen}>
                <DialogContent className="flex max-h-[80vh] flex-col sm:max-w-2xl">
                    <DialogHeader>
                        <DialogTitle>Third-party licences</DialogTitle>
                        <DialogDescription>
                            AnyLearning includes these open-source components. Select one to read its licence.
                        </DialogDescription>
                    </DialogHeader>

                    <Input
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                        placeholder="Search components"
                        aria-label="Search components"
                    />

                    <div className="min-h-0 flex-1 overflow-y-auto rounded-md border">
                        {isLoading && <p className="text-muted-foreground p-3 text-xs">Loading…</p>}
                        {error && (
                            <p className="text-muted-foreground p-3 text-xs">
                                The licence notices are missing from this build.
                            </p>
                        )}
                        {data && components.length === 0 && (
                            <p className="text-muted-foreground p-3 text-xs">Nothing matches “{query}”.</p>
                        )}
                        {components.map((component) => {
                            const key = `${component.name} ${component.version}`
                            const isOpen = expanded === key
                            return (
                                <div key={key} className="border-b last:border-b-0">
                                    <button
                                        type="button"
                                        onClick={() => setExpanded(isOpen ? null : key)}
                                        aria-expanded={isOpen}
                                        className="hover:bg-muted/50 flex w-full items-center gap-2 px-3 py-2 text-left text-xs transition-colors"
                                    >
                                        <ChevronDown
                                            className={cn(
                                                "text-muted-foreground size-3.5 shrink-0 transition-transform",
                                                !isOpen && "-rotate-90"
                                            )}
                                        />
                                        <span className="min-w-0 flex-1 truncate">{component.name}</span>
                                        <span className="text-muted-foreground shrink-0 font-mono">
                                            {component.version}
                                        </span>
                                    </button>
                                    {isOpen && (
                                        <pre className="text-muted-foreground max-h-64 overflow-auto px-3 pb-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
                                            {component.text}
                                        </pre>
                                    )}
                                </div>
                            )
                        })}
                    </div>

                    {data && (
                        <p className="text-muted-foreground text-xs">
                            {data.components.length} components. Every licence is reproduced in full above.
                        </p>
                    )}
                </DialogContent>
            </Dialog>
        </>
    )
}

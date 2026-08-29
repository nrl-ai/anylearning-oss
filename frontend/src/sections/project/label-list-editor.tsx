import { CopyIcon, ListIcon, PencilIcon, PlusIcon, TagIcon, TrashIcon } from "lucide-react"
import React from "react"

import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { EmptyState } from "@/components/ui/empty-state"
import { Input } from "@/components/ui/input"
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel"
import { Textarea } from "@/components/ui/textarea"
import { toast } from "@/components/ui/use-toast"
import useLabels from "@/lib/use-labels"

/**
 * The class swatch is drawn the way the class appears on the canvas — a solid
 * stroke over a translucent fill — so the legend and the annotation are
 * visibly the same system rather than two unrelated colour chips.
 */
function ClassSwatch({ color }: { color?: string }) {
    const value = color || "#888888"
    return (
        <span
            aria-hidden
            className="size-4 shrink-0 rounded-[3px] border-2"
            style={{ borderColor: value, backgroundColor: `${value}2e` }}
        />
    )
}

function LabelListEditor({ projectId }: { projectId: number }) {
    const {
        labels,
        isLoading,
        newLabelName,
        editingIndex,
        deletingIndex,
        setNewLabelName,
        setEditingIndex,
        setDeletingIndex,
        addLabel,
        updateLabel,
        removeLabel,
    } = useLabels(projectId)

    const copyClassList = () => {
        const classListText = labels?.map((label) => label.name).join("\n")
        navigator.clipboard.writeText(classListText)
        toast({
            title: "Copied",
            description: "Class list copied to the clipboard.",
        })
    }

    const hasLabels = !isLoading && labels && labels.length > 0

    return (
        <Panel className="flex min-h-0 flex-col">
            <PanelHeader
                icon={TagIcon}
                title="Labels"
                description="The classes you can assign while labelling."
                actions={
                    <Dialog>
                        <DialogTrigger asChild>
                            <Button variant="ghost" size="sm" disabled={!hasLabels}>
                                <ListIcon />
                                Class list
                            </Button>
                        </DialogTrigger>
                        <DialogContent>
                            <DialogHeader>
                                <DialogTitle>Class list</DialogTitle>
                            </DialogHeader>
                            <div className="flex flex-col gap-2">
                                <Textarea
                                    readOnly
                                    value={labels?.map((label) => label.name).join("\n")}
                                    className="min-h-[200px] font-mono text-xs"
                                />
                                <Button variant="outline" size="sm" onClick={copyClassList}>
                                    <CopyIcon />
                                    Copy class list
                                </Button>
                            </div>
                        </DialogContent>
                    </Dialog>
                }
            />
            <PanelBody className="flex min-h-0 flex-1 flex-col gap-3 p-3">
                {hasLabels ? (
                    // Capped rather than flex-sized: this panel sits in a grid
                    // cell with no height of its own, so `flex-1` never had a
                    // bound to scroll against and a 26-class project pushed the
                    // add-label field off the bottom of the page.
                    <ul className="max-h-[22rem] min-h-0 flex-1 space-y-0.5 overflow-y-auto">
                        {labels.map((label, index) => (
                            <li
                                key={index}
                                className="hover:bg-accent/60 group flex items-center gap-2 rounded-md px-2 py-1"
                            >
                                {editingIndex === index ? (
                                    <Input
                                        value={label.name}
                                        onChange={(e) => updateLabel(index, e.target.value, label.color || "#000000")}
                                        onBlur={() => setEditingIndex(null)}
                                        onKeyDown={(e) => e.key === "Enter" && setEditingIndex(null)}
                                        className="h-7 flex-grow"
                                        autoFocus
                                    />
                                ) : (
                                    <>
                                        <ClassSwatch color={label.color} />
                                        <span className="min-w-0 flex-grow truncate text-sm">{label.name}</span>
                                        <span className="text-muted-foreground tabular font-mono text-[0.6875rem] opacity-0 transition-opacity group-hover:opacity-100">
                                            {label.id}
                                        </span>
                                        <Input
                                            type="color"
                                            aria-label={`Colour for ${label.name}`}
                                            value={label.color}
                                            onChange={(e) => updateLabel(index, label.name, e.target.value)}
                                            className="size-6 cursor-pointer rounded border-0 p-0"
                                        />
                                    </>
                                )}
                                <Button
                                    variant="ghost"
                                    size="icon-xs"
                                    aria-label={`Rename ${label.name}`}
                                    onClick={() => setEditingIndex(index)}
                                >
                                    <PencilIcon />
                                </Button>
                                <Button
                                    variant="ghost"
                                    size="icon-xs"
                                    aria-label={`Delete ${label.name}`}
                                    onClick={() => setDeletingIndex(index)}
                                >
                                    <TrashIcon />
                                </Button>
                            </li>
                        ))}
                    </ul>
                ) : (
                    <EmptyState
                        compact
                        icon={TagIcon}
                        title="No labels yet"
                        description="Add the classes you want to detect. You need at least one before you can label images."
                        className="flex-1"
                    />
                )}

                <div className="flex items-center gap-2">
                    <Input
                        value={newLabelName}
                        onChange={(e) => setNewLabelName(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && addLabel()}
                        placeholder="New label name"
                        className="h-8 flex-grow"
                    />
                    <Button variant="outline" size="icon-sm" aria-label="Add label" onClick={addLabel}>
                        <PlusIcon />
                    </Button>
                </div>
            </PanelBody>

            <AlertDialog open={deletingIndex !== null} onOpenChange={() => setDeletingIndex(null)}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>
                            Delete {deletingIndex !== null ? labels?.[deletingIndex]?.name : "this label"}?
                        </AlertDialogTitle>
                        <AlertDialogDescription>
                            This can't be undone. Annotations already using this label keep their shape but lose the
                            class.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={() => deletingIndex !== null && removeLabel(deletingIndex)}>
                            Delete label
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </Panel>
    )
}

export default LabelListEditor

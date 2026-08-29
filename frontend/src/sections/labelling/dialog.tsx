import { Search, Trash2 } from "lucide-react"
import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface DialogProps {
    items: string[]
    itemsChanged: (items: string[]) => void
    onEdit: () => void
    onDelete: () => void
    onClose: () => void
    offset: { X: number; Y: number }
    categories: { name: string; color: string }[]
    keypoint?: {
        groupId: string | number | null
        visible: number
    }
    keypointChanged?: (groupId: string | number | null, visible: number) => void
}

/**
 * Class picker for the shape you just drew.
 *
 * Every row carries the class colour as an annotation swatch — the same
 * stroke-over-fill pair used on the canvas and in the project's label list —
 * instead of tinting the button text, which made low-contrast classes
 * unreadable and gave the panel a different look in each project.
 */
export default function Dialog(props: DialogProps) {
    const [searchTerm, setSearchTerm] = useState("")
    const [selectedItems, setSelectedItems] = useState(props.items)
    const [groupId, setGroupId] = useState(props.keypoint?.groupId?.toString() ?? "")
    const [visibility, setVisibility] = useState(String(props.keypoint?.visible ?? 2))

    const handleSelect = (selectedClass: string) => {
        setSelectedItems([selectedClass])
        props.itemsChanged([selectedClass])
        if (!props.keypoint) props.onClose()
    }

    const updateKeypoint = (nextGroup: string, nextVisibility: string) => {
        const parsed = nextGroup.trim() === "" ? null : Number(nextGroup)
        props.keypointChanged?.(
            parsed === null || Number.isFinite(parsed) ? parsed : nextGroup.trim(),
            Number(nextVisibility)
        )
    }

    const selectedCategory = props.categories.find((c) => selectedItems.includes(c.name))

    const remainingCategories = props.categories
        .filter((c) => c.name !== selectedCategory?.name)
        .filter((c) => c.name.toLowerCase().includes(searchTerm.toLowerCase()))

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={props.onClose}>
            <div
                className="bg-popover text-popover-foreground w-[300px] rounded-lg border p-3 shadow-xl"
                onClick={(e) => e.stopPropagation()}
                role="dialog"
                aria-label={props.keypoint ? "Choose a landmark" : "Choose a class"}
            >
                <div className="mb-2 flex items-center justify-between gap-2">
                    <div className="flex min-w-0 items-center gap-2">
                        {selectedCategory ? (
                            <>
                                <span
                                    aria-hidden
                                    className="size-3.5 shrink-0 rounded-[3px] border-2"
                                    style={{
                                        borderColor: selectedCategory.color,
                                        backgroundColor: `${selectedCategory.color}2e`,
                                    }}
                                />
                                <span className="truncate text-sm font-medium">{selectedCategory.name}</span>
                            </>
                        ) : (
                            <span className="text-muted-foreground text-sm">
                                {props.keypoint ? "No landmark yet" : "No class yet"}
                            </span>
                        )}
                    </div>
                    <Button
                        onClick={props.onDelete}
                        variant="ghost"
                        size="sm"
                        className="text-fail hover:bg-fail-surface hover:text-fail shrink-0"
                    >
                        <Trash2 />
                        Delete
                    </Button>
                </div>

                <div className="relative mb-2">
                    <Search className="text-muted-foreground absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2" />
                    <Input
                        type="text"
                        placeholder={props.keypoint ? "Search landmarks" : "Search classes"}
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="h-8 pl-8"
                        autoFocus
                    />
                </div>

                <div className="max-h-[320px] space-y-0.5 overflow-y-auto">
                    {remainingCategories.length === 0 ? (
                        <p className="text-muted-foreground px-2 py-3 text-center text-xs">
                            No matching {props.keypoint ? "landmarks" : "classes"}.
                        </p>
                    ) : (
                        remainingCategories.map((category) => (
                            <button
                                key={category.name}
                                type="button"
                                onClick={() => handleSelect(category.name)}
                                className="hover:bg-accent focus-visible:ring-ring flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left focus-visible:ring-2 focus-visible:outline-none"
                            >
                                <span
                                    aria-hidden
                                    className="size-3.5 shrink-0 rounded-[3px] border-2"
                                    style={{
                                        borderColor: category.color,
                                        backgroundColor: `${category.color}2e`,
                                    }}
                                />
                                <span className="min-w-0 truncate text-sm">{category.name}</span>
                            </button>
                        ))
                    )}
                </div>

                {props.keypoint && (
                    <div className="mt-3 space-y-3 border-t pt-3">
                        <div className="space-y-1.5">
                            <Label htmlFor="keypoint-instance">Instance</Label>
                            <Input
                                id="keypoint-instance"
                                type="number"
                                min={1}
                                step={1}
                                value={groupId}
                                placeholder="One subject"
                                onChange={(event) => {
                                    const next = event.target.value
                                    setGroupId(next)
                                    updateKeypoint(next, visibility)
                                }}
                            />
                            <p className="text-muted-foreground text-xs">
                                Use the same number for landmarks on the same subject.
                            </p>
                        </div>
                        <div className="space-y-1.5">
                            <Label>Visibility</Label>
                            <Select
                                value={visibility}
                                onValueChange={(next) => {
                                    setVisibility(next)
                                    updateKeypoint(groupId, next)
                                }}
                            >
                                <SelectTrigger className="w-full">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="2">Visible</SelectItem>
                                    <SelectItem value="1">Occluded</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        <Button className="w-full" onClick={props.onClose} disabled={!selectedCategory}>
                            Done
                        </Button>
                    </div>
                )}
            </div>
        </div>
    )
}

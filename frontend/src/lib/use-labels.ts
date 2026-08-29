import { useState } from "react"

import { useToast } from "@/components/ui/use-toast"
import { api } from "@/lib/api"
import { Label } from "@/types"

import useProject from "./use-project"

function generateRandomColor() {
    // Generate random RGB values between 30 and 225 to avoid black/white
    const r = Math.floor(Math.random() * (225 - 30) + 30)
    const g = Math.floor(Math.random() * (225 - 30) + 30)
    const b = Math.floor(Math.random() * (225 - 30) + 30)
    return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`
}

function useLabels(projectId: number) {
    const [newLabelName, setNewLabelName] = useState("")
    const [newLabelColor, setNewLabelColor] = useState(generateRandomColor())
    const [editingIndex, setEditingIndex] = useState<number | null>(null)
    const [deletingIndex, setDeletingIndex] = useState<number | null>(null)
    const { toast } = useToast()
    const { project, isLoading, refetch } = useProject(projectId)
    const labels = project?.labels || []

    const updateLabelsOnServer = async (updatedLabels: Label[]) => {
        try {
            await api.patch(`/api/projects/${projectId}`, {
                labels: JSON.stringify(updatedLabels),
            })
            await refetch()
        } catch (error) {
            console.error("Failed to update labels on server:", error)
        }
    }

    const addLabel = () => {
        if (newLabelName && !labels.some((label) => label.name.toLowerCase() === newLabelName.toLowerCase())) {
            const updatedLabels = [...labels, { name: newLabelName, color: newLabelColor, id: labels.length }]
            updateLabelsOnServer(updatedLabels)
            setNewLabelName("")
            setNewLabelColor(generateRandomColor())
        } else {
            toast({
                title: "Duplicate Label",
                description: "A label with this name already exists.",
                variant: "destructive",
            })
        }
    }

    const updateLabel = (index: number, name: string, color: string) => {
        if (!labels.some((label, i) => i !== index && label.name.toLowerCase() === name.toLowerCase())) {
            const updatedLabels = labels.map((label, i) => (i === index ? { ...label, name, color } : label))
            updateLabelsOnServer(updatedLabels)
            setEditingIndex(null)
        } else {
            toast({
                title: "Duplicate Label",
                description: "A label with this name already exists.",
                variant: "destructive",
            })
        }
    }

    const removeLabel = (index: number) => {
        const updatedLabels = labels.filter((_, i) => i !== index).map((label, i) => ({ ...label, id: i }))
        updateLabelsOnServer(updatedLabels)
        setDeletingIndex(null)
    }

    return {
        labels,
        newLabelName,
        newLabelColor,
        editingIndex,
        deletingIndex,
        setNewLabelName,
        setEditingIndex,
        setDeletingIndex,
        addLabel,
        updateLabel,
        removeLabel,
        isLoading,
    }
}

export default useLabels

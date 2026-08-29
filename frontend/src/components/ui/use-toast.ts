import * as React from "react"
import { toast as sonnerToast } from "sonner"

type ToastOptions = {
    title?: React.ReactNode
    description?: React.ReactNode
    variant?: "default" | "destructive"
}

function toast({ title, description, variant = "default" }: ToastOptions) {
    const message = title ?? description ?? "Notification"
    const options = title ? { description } : undefined
    const id = variant === "destructive" ? sonnerToast.error(message, options) : sonnerToast(message, options)

    return {
        id: String(id),
        dismiss: () => sonnerToast.dismiss(id),
        update: (next: ToastOptions) =>
            sonnerToast(next.title ?? next.description ?? message, {
                id,
                description: next.title ? next.description : undefined,
            }),
    }
}

function useToast() {
    return {
        toast,
        dismiss: (toastId?: string | number) => sonnerToast.dismiss(toastId),
    }
}

export { toast, useToast }

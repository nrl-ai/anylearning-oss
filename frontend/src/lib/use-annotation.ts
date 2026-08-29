/**
 * Annotation reads and writes for the labelling canvas.
 *
 * These stay imperative rather than becoming React Query hooks: the annotator
 * loads and saves on its own gesture lifecycle (image change, explicit save,
 * auto-save on navigate), not on render. They go through the shared client so
 * the auth header is applied in one place; callers keep the cache honest by
 * writing the labelled flag back optimistically.
 */
import { api } from "@/lib/api"

export async function getAnnotation(projectId: number, itemId: number) {
    try {
        const response = await api.get(`/api/projects/${projectId}/data_items/${itemId}/get_annotation`)
        return response.data
    } catch (error) {
        console.error("Error fetching annotation:", error)
        throw error
    }
}

export async function putAnnotation(projectId: number, itemId: number, shapes: any): Promise<boolean> {
    try {
        await api.post(`/api/projects/${projectId}/data_items/${itemId}/set_annotation`, shapes)
        return true
    } catch (error) {
        console.error("Error saving annotation:", error)
        return false
    }
}

export async function putClassId(projectId: number, itemId: number, classId: number): Promise<boolean> {
    try {
        await api.post(`/api/projects/${projectId}/data_items/${itemId}/class_id`, { class_id: classId })
        return true
    } catch (error) {
        console.error("Error saving class ID:", error)
        return false
    }
}

export async function getAutoAnnotation(projectId: number, itemId: number, modelName: string, marks: any) {
    try {
        const response = await api.post(`/api/projects/${projectId}/auto_labeling/inference`, {
            model_name: modelName,
            data_item_id: itemId,
            marks,
            preload_data_item_ids: [itemId + 1, itemId + 2],
        })
        return response.data
    } catch (error) {
        console.error("Error fetching auto annotation:", error)
        throw error
    }
}

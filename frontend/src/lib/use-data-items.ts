import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useCallback, useState } from "react"

import { api, getJson } from "@/lib/api"
import { qk } from "@/lib/query-keys"
import { DataItem } from "@/types"

type Page = { data_items: DataItem[]; total_count: number }

/**
 * A page of data items for one subset.
 *
 * Pagination lives in the query key rather than in imperative fetch calls, so
 * flipping back to a page you have already seen is instant and the grid never
 * blanks out mid-navigation. `fetchDataItems` is kept as the way to move the
 * window, since the dataset screen drives paging from several places.
 */
const useDataItems = (projectId: number | null, subset: number) => {
    const queryClient = useQueryClient()
    const [range, setRange] = useState({ offset: 0, limit: 20 })
    const enabled = projectId !== null

    const queryKey = qk.dataItems(projectId as number, subset, range.offset, range.limit)

    const { data, error, isLoading, refetch } = useQuery({
        queryKey,
        queryFn: () =>
            getJson<Page>(`/api/projects/${projectId}/data_items`, {
                params: { subset, offset: range.offset, limit: range.limit },
            }),
        enabled,
        // Keeps the previous page on screen while the next one loads.
        placeholderData: (previous) => previous,
    })

    /**
     * Optimistic write into the cached page.
     *
     * Marking the current image as labelled has to show immediately -- the user
     * is already moving to the next one -- so it is applied to the cache rather
     * than waiting for a refetch.
     */
    const setDataItems = useCallback(
        (updater: DataItem[] | ((items: DataItem[]) => DataItem[])) => {
            queryClient.setQueryData<Page>(queryKey, (prev) => {
                if (!prev) return prev
                return {
                    ...prev,
                    data_items: typeof updater === "function" ? updater(prev.data_items) : updater,
                }
            })
        },
        [queryClient, queryKey]
    )

    const fetchDataItems = useCallback((offset: number, limit: number) => {
        setRange((prev) => ({
            offset: offset === -1 ? prev.offset : offset,
            limit: limit === -1 ? prev.limit : limit,
        }))
    }, [])

    const invalidateDataItems = useCallback(() => {
        if (projectId === null) return
        queryClient.invalidateQueries({ queryKey: qk.dataItemsAll(projectId) })
        queryClient.invalidateQueries({ queryKey: qk.datasets(projectId) })
    }, [queryClient, projectId])

    const deleteMutation = useMutation({
        mutationFn: (ids: number[]) => api.delete(`/api/projects/${projectId}/data_items`, { data: ids }),
        onSuccess: invalidateDataItems,
    })

    return {
        dataItems: data?.data_items ?? [],
        totalCount: data?.total_count ?? 0,
        setDataItems,
        fetchDataItems,
        invalidateDataItems,
        refetch,
        deleteDataItems: deleteMutation.mutateAsync,
        isLoading: enabled ? isLoading : false,
        isError: enabled && !!error,
        error,
    }
}

export default useDataItems

import { useQueries, useQuery, useQueryClient } from "@tanstack/react-query"

import { getJson } from "@/lib/api"
import { qk } from "@/lib/query-keys"
import { DataItem, Dataset, Datasets } from "@/types"

const SUBSETS = [
    { name: "Training", type: "train", subset: 0 },
    { name: "Validation", type: "validation", subset: 1 },
    { name: "Test", type: "test", subset: 2 },
] as const

const emptyInfo = (type: string, subset: number): Dataset => ({
    type,
    version: "",
    num_total: 0,
    num_labeled: 0,
    num_unlabeled: 0,
    subset,
})

const useDatasets = (projectId: number | null) => {
    const queryClient = useQueryClient()
    const enabled = projectId !== null

    const {
        data: datasetsInfo,
        error: datasetsError,
        isLoading,
    } = useQuery({
        queryKey: qk.datasets(projectId as number),
        queryFn: () => getJson<Dataset[]>(`/api/projects/${projectId}/datasets`),
        enabled,
    })

    // One query per subset, run together. The previous version chained a
    // useState, a useEffect and three manual axios calls to assemble the same
    // shape, which meant the summary counts and the items could disagree for a
    // frame after any change.
    const itemQueries = useQueries({
        queries: SUBSETS.map(({ subset }) => ({
            queryKey: qk.dataItems(projectId as number, subset, 0, -1),
            queryFn: () =>
                getJson<{ data_items: DataItem[] }>(`/api/projects/${projectId}/data_items`, {
                    params: { subset },
                }).then((d) => d.data_items),
            enabled,
        })),
    })

    const datasets: Datasets = Object.fromEntries(
        SUBSETS.map(({ name, type, subset }, index) => [
            name,
            {
                items: itemQueries[index]?.data ?? [],
                info: datasetsInfo?.find((d) => d.subset === subset) ?? emptyInfo(type, subset),
            },
        ])
    )

    const invalidateDatasets = () => {
        if (projectId === null) return
        queryClient.invalidateQueries({ queryKey: qk.datasets(projectId) })
        queryClient.invalidateQueries({ queryKey: qk.dataItemsAll(projectId) })
    }

    return {
        datasets,
        datasetsInfo,
        invalidateDatasets,
        /** Kept for call sites that refresh explicitly after an upload. */
        fetchAllDatasets: invalidateDatasets,
        isLoading: enabled ? isLoading : false,
        isError: enabled && !!datasetsError,
        error: datasetsError,
    }
}

export default useDatasets

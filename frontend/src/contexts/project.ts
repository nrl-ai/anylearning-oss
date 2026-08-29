import { createContext, useContext } from "react"

import { Project } from "@/types"

type ProjectContextValue = Project | undefined
export const ProjectContext = createContext<ProjectContextValue | undefined>(undefined)

export const useProjectContext = (): ProjectContextValue => {
    const value = useContext(ProjectContext)
    if (!value) {
        throw new Error("Invalid use of ProjectContext")
    }

    return value
}

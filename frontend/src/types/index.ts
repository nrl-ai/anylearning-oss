import { Icons } from "@/components/icons"

export type PropsWithClassName<T extends {} = {}> = {
    className?: string
} & T

export type MetaResponse<T = unknown> = {
    statusCode: number
    message?: string
    data?: T | null
    debug?: any
}

export type ExtractMetaResponse<M extends MetaResponse<any>> = M extends MetaResponse<infer T> ? T : never

export type Prettify<T> = {
    [K in keyof T]: T[K]
} & {}

export interface NavItem {
    title: string
    href?: string
    disabled?: boolean
    external?: boolean
    icon?: keyof typeof Icons
    label?: string
    description?: string
    children?: NavItem[]
}

export interface NavItemWithChildren extends NavItem {
    items: NavItemWithChildren[]
}

export interface NavItemWithOptionalChildren extends NavItem {
    items?: NavItemWithChildren[]
}

export interface FooterItem {
    title: string
    items: {
        title: string
        href: string
        external?: boolean
    }[]
}

type Project = {
    id: number
    name: string
    description?: string | null
    createdAt: Date | null
    updatedAt: Date | null
    size: number | null
    numTrainedModels: number | null
    newModelsThisMonth: number | null
    type:
        | "Image Classification"
        | "Object Detection"
        | "Image Segmentation"
        | "Sentiment Analysis"
        | "Text & LLM"
        | "Text AI & LLM Evaluation"
        | "Text AI"
        | "Tabular AI"
        | "Handpose Classification"
        | "Instance Segmentation"
        | "Keypoint Detection"
    labels: Label[]
}

type ProjectCreation = {
    name: string
    description?: string | null | undefined
    type:
        | "Image Classification"
        | "Object Detection"
        | "Image Segmentation"
        | "Sentiment Analysis"
        | "Text AI"
        | "Tabular AI"
        | "Handpose Classification"
        | "Instance Segmentation"
        | "Keypoint Detection"
}

export type { Project, ProjectCreation }

export interface DataItem {
    id: number
    subset: number
    labeled: boolean
    path: string
    original_name: string
    class_id: number
}

export interface Dataset {
    type: string
    version: string
    num_total: number
    num_labeled: number
    num_unlabeled: number
    subset: number
}

export interface Datasets {
    [key: string]: {
        items: DataItem[]
        info: Dataset
    }
}

export type Label = {
    id: number
    name: string
    color?: string
    description?: string
}

export interface DatasetInfo {
    num_total: number
    num_labeled: number
    version: string
}

export interface DatasetWithInfo extends Dataset {
    info: DatasetInfo
}

export interface DatasetsWithInfo {
    [key: string]: DatasetWithInfo
}

export interface Model {
    id: number
    training_session_id: number
    name: string
    description: string
    path: string
    exported_path: string | null
    model_variant: string
    model_architecture: string
    model_size: string
    test_result: { [key: string]: number }
    metric_logs?: { [key: string]: number }[]
    training_session: TrainingSession
}

export type MainNavItem = NavItemWithOptionalChildren

export type SidebarNavItem = NavItemWithChildren

export interface TrainingParams {
    batch_size: number
    epochs: number
    learning_rate: number
    model_architecture: string
    model_size: string
    model_variant: string
    pretrained_model: string
    /**
     * "auto" uses an accelerator when there is one, "cpu" pins the run to the
     * CPU, and an Accelerator.id ("cuda", "mps") asks for that hardware. "gpu"
     * is the older spelling of "whatever accelerator this machine has" and is
     * still accepted. Sessions from before this field default to auto.
     */
    device?: "auto" | "gpu" | "cpu" | "cuda" | "mps"
    /** null keeps the model's own default. Rounded to a multiple of 32 server-side. */
    image_size?: number | null
    /** Keyed by AugmentationOption.key. Omitted keys keep the trainer's default. */
    augmentation?: Record<string, boolean | number> | null
}

/** One augmentation control, as the trainer that supports it describes it. */
export interface AugmentationOption {
    key: string
    label: string
    type: "bool" | "int" | "float"
    default: boolean | number
    help?: string
    minimum?: number
    maximum?: number
    step?: number
}

/** What to change before running again, from anylearning/training/diagnostics.py. */
export interface TrainingAdvice {
    level: "warn" | "info"
    title: string
    detail: string
}

/** One piece of hardware training can run on, besides the CPU. */
export interface Accelerator {
    /** What to send as TrainingParams.device to pick it. */
    id: "cuda" | "mps"
    /** The hardware's own name, e.g. "NVIDIA GeForce RTX 4090" or "Apple M1". */
    name: string | null
    /** Ready-made wording for the menu, e.g. "Apple GPU, Metal (Apple M1)". */
    label: string
    /** Project types this accelerator should not be offered for, by Project.type. */
    excluded_project_types?: string[]
}

export interface TrainingDevices {
    /** Empty on a CPU-only machine. The CPU itself is always a choice. */
    accelerators?: Accelerator[]
    /** True only for a CUDA GPU; a Mac reports false. Kept for older backends. */
    cuda: boolean
    name: string | null
}

export interface TrainingResponse {
    message: string
    session_id: number
}

export interface TrainingSession {
    id: number
    name: string
    description: string
    status: string
    started_at: string
    ended_at: string
    params: TrainingParams
    metric_logs: any
    /** Present when the run's numbers suggest a setting to change. */
    advice?: TrainingAdvice[]
    model: {
        id: number
        name: string
    }
}

export interface DetailedTrainingSession extends TrainingSession {
    training_logs: string
}

declare global {
    interface Window {
        pywebview?: {
            token?: string
            /** The backend doing the rendering: "cocoa", "edgechromium", "gtkwebkit2", … */
            platform?: string
            api?: {
                download_file?: (url: string, filename: string) => void
                /**
                 * The window controls behind the app-drawn title bar, exposed by
                 * `anylearning/window_chrome/`. Every one of them is absent in a
                 * browser, which is why they are all optional -- go through
                 * `@/lib/desktop` rather than reaching for them directly.
                 */
                window_chrome_state?: () => Promise<{ maximized: boolean }>
                window_minimize?: () => Promise<void>
                window_toggle_maximize?: () => Promise<boolean>
                window_close?: () => Promise<void>
                window_begin_drag?: (x: number, y: number) => Promise<boolean>
                window_begin_resize?: (edge: string, x: number, y: number) => Promise<boolean>
                window_set_drag_regions?: (regions: number[][], exclusions: number[][]) => Promise<void>
            }
        }
    }
}

export interface ClassCount {
    name: string
    color: string | null
    /** False when the annotations name a label the project no longer lists. */
    known: boolean
    total: number
    train: number
    validation: number
    test: number
}

export interface ClassDistribution {
    classes: ClassCount[]
    unlabeled: { train: number; validation: number; test: number; total: number }
}

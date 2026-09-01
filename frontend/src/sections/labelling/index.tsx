import { useProjectContext } from "@/contexts/project"
import { Loader2 } from "lucide-react"
import React, { useCallback, useEffect, useReducer, useRef, useState } from "react"

import { Dot, ImageAnnotator, Shape, useImageAnnotator } from "@/components/react-image-label"
import { ShapeColor } from "@/components/react-image-label/base/types"
import { api } from "@/lib/api"
import {
    AutoLabelingPrediction,
    createAutoLabelingPrediction,
    createAutoLabelingPreview,
    persistableAnnotationShapes,
} from "@/lib/auto-labeling-shape"
import { usesCanvasAnnotations } from "@/lib/project-types"
import { generateUniqueId } from "@/lib/random"
import { getAnnotation, putAnnotation } from "@/lib/use-annotation"
import { putClassId } from "@/lib/use-annotation"
import useDataItems from "@/lib/use-data-items"

import AutoLabellingToolbar from "./auto-labeling-toolbar"
import BottomBar from "./bottom-bar"
import Dialog from "./dialog"
import LeftBar from "./left-bar"
import NoLabelsAlert from "./no-label-alert"
import RightBar from "./right-bar"
import { useAutoSaveSettingStore } from "./stores"
import TopBar from "./top-bar"

interface LabelingScreenProps {
    projectId: number
    subset: number
    /** Leaves the labelling screen. Lives in the top bar so it can never sit
        on top of the bar's own controls, which is what the previous absolutely
        positioned "Return" button did. */
    onExit?: () => void
}

interface DialogState {
    show: boolean
    shape: Shape | undefined
}

const imagesPerPage = 20

type LabelingMode = "labeling" | "auto_labeling"

const LabelingScreen: React.FC<LabelingScreenProps> = ({ projectId, subset, onExit }) => {
    const project = useProjectContext()

    const [currentPage, setCurrentPage] = useState<number>(1)
    const [currentImageIndex, setCurrentImageIndex] = useState<number>(0)
    const [dialog, setDialog] = useState<DialogState>({
        show: false,
        shape: undefined,
    })
    const [aiEnabled, setAiEnabled] = useState<boolean>(true)
    const [mode, setMode] = useState<LabelingMode>("labeling")
    const [aiToolSelected, setAiToolSelected] = useState<string>("")
    const [annotatorWidth, setAnnotatorWidth] = useState<number>(0)
    const [annotatorHeight, setAnnotatorHeight] = useState<number>(0)
    const [imageKey, increaseImageKey] = useReducer((x: number) => x + 1, 0)
    const annotatorContainerRef = useRef<HTMLDivElement>(null)
    const [savingStatus, setSavingStatus] = useState<string>("")
    const [selectedTool, setSelectedTool] = useState<string>(() => {
        // Set initial tool based on project type
        if (project?.type === "Object Detection") {
            return "rectangle"
        } else if (project?.type === "Image Segmentation") {
            return "polygon"
        } else if (project?.type === "Keypoint Detection") {
            return "dot"
        }
        return "select"
    })
    const { dataItems, setDataItems, totalCount, fetchDataItems, isLoading, isError } = useDataItems(projectId, subset)
    const [isInferencing, setIsInferencing] = useState<boolean>(false)
    const [selectedModel, setSelectedModel] = useState<string>("")
    const [selectedModelMode, setSelectedModelMode] = useState<"prompted" | "automatic">("prompted")
    const [aiShape, setAiShape] = useState<string>("polygon")
    const [isReady, setReady] = useState(false)
    const autoPredictionIdsRef = useRef<Map<number, Set<string>>>(new Map())
    const [isLoadingImage, setIsLoadingImage] = useState<boolean>(false)
    const [isLoadingAnnotation, setIsLoadingAnnotation] = useState<boolean>(false)
    const [isSavingAnnotation, setSavingAnnotation] = useState<boolean>(false)
    const lastSavedShapesRef = useRef<string>("")
    const autoSaveTimeoutRef = useRef<NodeJS.Timeout | null>(null)
    const isNavigatingRef = useRef(false)
    const loadAnnotationRetryCount = useRef(0)
    const MAX_RETRY_COUNT = 3
    const lastLoadedAnnotationRef = useRef<{ imageId: number; shapes: Shape[] } | undefined>(undefined)

    const currentImage = dataItems[currentImageIndex]
    const totalPages = Math.ceil(totalCount / imagesPerPage)
    const hasCanvasAnnotations = usesCanvasAnnotations(project?.type)

    const { setHandles, annotator } = useImageAnnotator()
    const [showNoLabelsAlert, setShowNoLabelsAlert] = useState<boolean>(false)

    const getCurrentShapesHash = useCallback(() => {
        const shapes = persistableAnnotationShapes(annotator?.getShapes() || [])
        return JSON.stringify(shapes)
    }, [annotator])

    const hasUnsavedChanges = useCallback(() => {
        if (!hasCanvasAnnotations) return false
        const currentShapesHash = getCurrentShapesHash()
        return currentShapesHash !== lastSavedShapesRef.current
    }, [getCurrentShapesHash, hasCanvasAnnotations])

    const saveAnnotation = useCallback(async () => {
        if (
            !hasCanvasAnnotations ||
            !annotator ||
            !dataItems[currentImageIndex] ||
            isSavingAnnotation ||
            !hasUnsavedChanges()
        ) {
            return false
        }

        setSavingAnnotation(true)
        const shapes = persistableAnnotationShapes(annotator.getShapes() || [])
        const currentImageId = dataItems[currentImageIndex].id
        try {
            const success = await putAnnotation(projectId, currentImageId, shapes)
            if (success) {
                lastSavedShapesRef.current = JSON.stringify(shapes)
                setSavingStatus("Annotation saved")
                setTimeout(() => setSavingStatus(""), 5000)
                setDataItems((items) => {
                    const updatedDataItems = [...items]
                    updatedDataItems[currentImageIndex] = {
                        ...updatedDataItems[currentImageIndex],
                        labeled: true,
                    }
                    return updatedDataItems
                })
                return true
            }
        } catch (e) {
            setSavingStatus("Failed to save annotation")
            setTimeout(() => setSavingStatus(""), 5000)
        } finally {
            setSavingAnnotation(false)
        }
        return false
    }, [
        annotator,
        currentImageIndex,
        dataItems,
        hasCanvasAnnotations,
        hasUnsavedChanges,
        isSavingAnnotation,
        projectId,
        setDataItems,
    ])

    // Auto-save every 5 seconds if there are changes
    useEffect(() => {
        if (
            !hasCanvasAnnotations ||
            !useAutoSaveSettingStore.getState().isEnabled ||
            !isReady ||
            isNavigatingRef.current
        ) {
            if (autoSaveTimeoutRef.current) {
                clearTimeout(autoSaveTimeoutRef.current)
            }
            return
        }

        const startAutoSave = () => {
            if (autoSaveTimeoutRef.current) {
                clearTimeout(autoSaveTimeoutRef.current)
            }

            autoSaveTimeoutRef.current = setTimeout(async () => {
                if (hasUnsavedChanges()) {
                    await saveAnnotation()
                }
                startAutoSave()
            }, 5000)
        }

        startAutoSave()

        return () => {
            if (autoSaveTimeoutRef.current) {
                clearTimeout(autoSaveTimeoutRef.current)
            }
        }
    }, [hasCanvasAnnotations, isReady, saveAnnotation, hasUnsavedChanges])

    useEffect(() => {
        if (project && (!project.labels || project.labels.length === 0)) {
            setShowNoLabelsAlert(true)
        }
        if (
            project?.type === "Image Classification" ||
            project?.type === "Handpose Classification" ||
            project?.type === "Keypoint Detection"
        ) {
            setAiEnabled(false)
        }
    }, [project])

    useEffect(() => {
        const offset = (currentPage - 1) * imagesPerPage
        fetchDataItems(offset, imagesPerPage)
    }, [currentPage, fetchDataItems])

    const annotatorRef = useRef(annotator)
    annotatorRef.current = annotator
    useEffect(() => {
        // this effect helps get shapes when user chooses a new image
        let shouldSetShapes = true
        let retryTimeout: ReturnType<typeof setTimeout> | undefined
        const annotator = annotatorRef.current

        if (!currentImage?.id || !annotator || !isReady) return

        // Never send classification metadata through the shape loader/saver.
        // Handpose annotations are landmark dictionaries, not shape arrays.
        if (!hasCanvasAnnotations) {
            annotator.setShapes([])
            lastLoadedAnnotationRef.current = {
                imageId: currentImage.id,
                shapes: [],
            }
            lastSavedShapesRef.current = "[]"
            loadAnnotationRetryCount.current = 0
            setIsLoadingAnnotation(false)
            return
        }

        // Do not leave the previous image's shapes visible while the next
        // request is pending. More importantly, a failed request must never
        // auto-save those stale shapes onto the new image.
        annotator.setShapes([])
        lastLoadedAnnotationRef.current = undefined
        lastSavedShapesRef.current = "[]"

        const loadAnnotationWithRetry = async () => {
            try {
                setIsLoadingAnnotation(true)
                const shapes = await getAnnotation(projectId, currentImage.id)

                if (!shouldSetShapes) return

                if (!Array.isArray(shapes)) throw new TypeError("Annotation response must be a shape array")

                // Store the loaded annotation
                lastLoadedAnnotationRef.current = {
                    imageId: currentImage.id,
                    shapes,
                }

                if (annotator) {
                    // Update shape colors based on categories
                    const shapesWithColors = shapes.map((shape: Shape) => {
                        const category = shape.categories?.[0]
                        if (category) {
                            const label = project?.labels.find((l) => l.name === category)
                            if (label?.color) {
                                return {
                                    ...shape,
                                    color: label.color,
                                }
                            }
                        }
                        return shape
                    })

                    annotator.setShapes(shapesWithColors as Shape[])
                    lastSavedShapesRef.current = JSON.stringify(shapesWithColors)
                }
                loadAnnotationRetryCount.current = 0
            } catch (error) {
                console.error("Failed to load annotation:", error)
                if (loadAnnotationRetryCount.current < MAX_RETRY_COUNT) {
                    loadAnnotationRetryCount.current++
                    retryTimeout = setTimeout(loadAnnotationWithRetry, 1000)
                }
            } finally {
                setIsLoadingAnnotation(false)
            }
        }

        loadAnnotationWithRetry()

        return () => {
            shouldSetShapes = false
            if (retryTimeout) clearTimeout(retryTimeout)
        }
    }, [currentImage?.id, hasCanvasAnnotations, isReady, projectId, project?.labels])

    // Add effect to ensure annotations are loaded into annotator
    useEffect(() => {
        if (
            annotator &&
            lastLoadedAnnotationRef.current &&
            currentImage?.id === lastLoadedAnnotationRef.current.imageId
        ) {
            const currentShapes = annotator.getShapes()
            if (currentShapes && currentShapes.length === 0 && lastLoadedAnnotationRef.current.shapes.length > 0) {
                // Update shape colors based on categories
                const shapesWithColors = lastLoadedAnnotationRef.current.shapes.map((shape) => {
                    const category = shape.categories?.[0]
                    if (category) {
                        const label = project?.labels.find((l) => l.name === category)
                        if (label?.color) {
                            return {
                                ...shape,
                                color: label.color,
                            }
                        }
                    }
                    return shape
                })

                annotator.setShapes(shapesWithColors as Shape[])
                lastSavedShapesRef.current = JSON.stringify(shapesWithColors)
            }
        }
    }, [annotator, currentImage?.id, project?.labels])

    useEffect(() => {
        // Measure the actual container rather than guessing from the window.
        // This used to be `window.innerWidth - 250`, hardcoding a sidebar width
        // that no longer matches the layout (and is 0 when the sidebar is
        // collapsed), so the canvas was consistently the wrong size.
        const element = annotatorContainerRef.current
        if (!element) return

        const applySize = () => {
            const { width, height } = element.getBoundingClientRect()
            const nextWidth = Math.max(0, Math.floor(width))
            const nextHeight = Math.max(0, Math.floor(height))
            if (nextWidth !== annotatorWidth || nextHeight !== annotatorHeight) {
                setAnnotatorWidth(nextWidth)
                setAnnotatorHeight(nextHeight)
                increaseImageKey()
            }
        }

        applySize()

        // ResizeObserver catches sidebar collapse and panel changes too, which a
        // window resize listener never sees.
        const observer = new ResizeObserver(applySize)
        observer.observe(element)
        return () => observer.disconnect()
    }, [annotatorWidth, annotatorHeight])

    const setShapesHandle = useCallback(
        (newShapes: Shape[]) => {
            // Update shape colors based on categories
            const shapesWithColors = newShapes.map((shape) => {
                const category = shape.categories?.[0]
                if (category) {
                    const label = project?.labels.find((l) => l.name === category)
                    if (label?.color) {
                        return {
                            ...shape,
                            color: label.color,
                        }
                    }
                }
                return shape
            })

            annotator?.setShapes(shapesWithColors as Shape[])
        },
        [annotator, project?.labels]
    )

    const clearAutoLabelingShapes = useCallback(
        (excludePrediction = false) => {
            const shapes = annotator?.getShapes() || []
            const newShapes = shapes.filter(
                (shape) =>
                    !shape.categories?.some((category) => category.startsWith("AUTOLABEL_")) ||
                    (excludePrediction && shape.categories?.includes("AUTOLABEL_TMP_SHAPE"))
            )

            // only setShapes if the filtered shapes are different
            if (shapes.length !== newShapes.length) {
                annotator?.setShapes(newShapes)
            }
        },
        [annotator]
    )

    const clearAutomaticPredictions = useCallback(() => {
        const imageId = currentImage?.id
        if (!imageId) return
        const predictionIds = autoPredictionIdsRef.current.get(imageId) ?? new Set<string>()
        const shapes = annotator?.getShapes() || []
        const retained = shapes.filter(
            (shape) => !predictionIds.has(shape.id) && shape.auto_labeling_model !== selectedModel
        )
        if (retained.length !== shapes.length) {
            annotator?.setShapes(retained)
        }
        autoPredictionIdsRef.current.delete(imageId)
    }, [annotator, currentImage?.id, selectedModel])

    const selectAutoLabelingModel = useCallback(
        (name: string, interactionMode: "prompted" | "automatic") => {
            clearAutoLabelingShapes(false)
            setSelectedModel(name)
            setSelectedModelMode(interactionMode)
            setMode("auto_labeling")
            if (interactionMode === "automatic") {
                setAiToolSelected("")
                annotator?.stop()
                annotator?.setEditable(true)
            }
        },
        [annotator, clearAutoLabelingShapes]
    )

    const clearNoCategoryShapes = useCallback(() => {
        const shapes = annotator?.getShapes() || []
        const newShapes = shapes.filter((shape) => shape.categories && shape.categories.length > 0)

        // Only setShapes if there are shapes to remove
        if (shapes.length !== newShapes.length) {
            annotator?.setShapes(newShapes)
            // Reselect the current tool
            handleToolSelect(selectedTool, false)
        }
    }, [annotator, selectedTool])

    const handlePageChange = useCallback(
        (newPage: number) => {
            setCurrentPage(newPage)
            setCurrentImageIndex(newPage < currentPage ? imagesPerPage - 1 : 0)
        },
        [currentPage]
    )

    const handleClassChange = async (classId: number) => {
        if (currentImage) {
            const success = await putClassId(projectId, currentImage.id, classId)
            if (success) {
                setSavingStatus("Class saved")
                setTimeout(() => setSavingStatus(""), 5000)
                setDataItems((items) => {
                    const updatedDataItems = [...items]
                    updatedDataItems[currentImageIndex] = {
                        ...updatedDataItems[currentImageIndex],
                        labeled: classId !== -1,
                        class_id: classId,
                    }
                    return updatedDataItems
                })
            } else {
                setSavingStatus("Failed to save class")
                setTimeout(() => setSavingStatus(""), 5000)
            }
        }
    }

    const hideDialog = () => setDialog({ show: false, shape: undefined })
    const hideAndUpdateCategories = () => {
        if (dialog.show && dialog.shape && annotator) {
            setShapeCategories(dialog.shape, dialog.shape.categories)
            hideDialog()
        }
    }

    const handleToolSelect = (tool: string, isAITool: boolean = false) => {
        if (isLoadingImage || isLoadingAnnotation) return

        setSelectedTool(tool)
        setMode(isAITool ? "auto_labeling" : "labeling")

        if (tool === "select") {
            annotator?.setEditable(true)
        } else {
            annotator?.setEditable(false)
        }

        if (isAITool) {
            setAiToolSelected(tool)
            // Set special category based on AI tool
            const categoryMap = {
                addPoint: "AUTOLABEL_ADD_POINT",
                removePoint: "AUTOLABEL_REM_POINT",
                addRect: "AUTOLABEL_ADD_RECT",
                removeRect: "AUTOLABEL_REM_RECT",
                tmpShape: "AUTOLABEL_TMP_SHAPE",
            }
            const category = categoryMap[tool as keyof typeof categoryMap]

            switch (tool) {
                case "addPoint":
                case "removePoint":
                    annotator?.drawDot()
                    break
                case "addRect":
                case "removeRect":
                    annotator?.drawRectangle()
                    break
            }
        } else {
            clearAutoLabelingShapes(false)
            switch (tool) {
                case "select":
                    annotator?.stop()
                    break
                case "rectangle":
                    annotator?.drawRectangle()
                    break
                case "polygon":
                    annotator?.drawPolygon()
                    break
                case "circle":
                    annotator?.drawCircle()
                    break
                case "dot":
                    annotator?.drawDot()
                    break
            }
        }

        // Close the dialog if it's open
        if (dialog.show) {
            hideDialog()
        }
    }

    useEffect(() => {
        // Re-select current tool when image changes to ensure proper tool state
        setTimeout(() => {
            if (selectedTool) {
                const isAITool = mode === "auto_labeling"
                handleToolSelect(selectedTool, isAITool)
            }
        }, 300)
    }, [currentImage?.path])

    useEffect(() => {
        if (!project?.type) return

        // Detection annotations have one canonical geometry: boxes. SAM still
        // returns a contour (useful for segmentation), but detection projects
        // always reduce it to its bounds before it reaches the canvas or save
        // endpoint.
        if (project.type === "Object Detection") {
            setAiShape("rectangle")
        } else if (project.type === "Image Segmentation") {
            setAiShape("polygon")
        }

        if (!isReady) return

        // Set initial tool based on project type
        if (project.type === "Object Detection") {
            handleToolSelect("rectangle", false)
        } else if (project.type === "Image Segmentation") {
            handleToolSelect("polygon", false)
        } else if (project.type === "Keypoint Detection") {
            handleToolSelect("dot", false)
        } else {
            handleToolSelect("rectangle", false)
        }
    }, [project?.type, isReady])

    const handleFinishAutoLabel = useCallback(() => {
        if (isLoadingImage || isLoadingAnnotation) return

        // Find temporary shape and show dialog for class assignment
        const shapes = annotator?.getShapes()
        const tmpShape = shapes?.find((s) => s.categories?.includes("AUTOLABEL_TMP_SHAPE")) as Shape | undefined
        if (tmpShape) {
            setDialog({ show: true, shape: tmpShape })
        }
    }, [annotator, isLoadingImage, isLoadingAnnotation])

    const inferenceShape = async (newShapes: Shape[]) => {
        if (!currentImage || !selectedModel) return
        setIsInferencing(true)
        // Remove any existing AUTOLABEL_TMP_SHAPE before adding new one
        newShapes = newShapes.filter((shape) => !shape.categories?.includes("AUTOLABEL_TMP_SHAPE"))
        const jsonShapes = JSON.parse(JSON.stringify(newShapes))
        const markShapes = jsonShapes.filter((shape: any) =>
            shape.categories?.some(
                (cat: string) => cat.startsWith("AUTOLABEL_ADD_") || cat.startsWith("AUTOLABEL_REM_")
            )
        )

        // Prepare marks for API request
        const marks = markShapes
            .map((shape: any) => {
                if (shape.type === "dot") {
                    return {
                        type: "point",
                        data: shape.position,
                        label: shape.categories?.[0] === "AUTOLABEL_ADD_POINT" ? 1 : 0,
                    }
                } else if (shape.type === "rectangle") {
                    const points = shape.points
                    const x1 = points[0][0]
                    const y1 = points[0][1]
                    const x2 = points[2][0]
                    const y2 = points[2][1]
                    return {
                        type: "rectangle",
                        data: [x1, y1, x2, y2],
                        label: shape.categories?.[0] === "AUTOLABEL_ADD_RECT" ? 1 : 0,
                    }
                }
                return null
            })
            .filter((mark: any) => mark !== null)

        // Get next 5 images to preload
        const nextImages = dataItems.slice(currentImageIndex + 1, currentImageIndex + 6)
        const nextImageIds = nextImages.map((item) => item.id)

        try {
            const response = await api.post(`/api/projects/${projectId}/auto_labeling/inference`, {
                model_name: selectedModel,
                data_item_id: dataItems[currentImageIndex].id,
                // Prompt geometry belongs to the promptable model that made
                // it. Never carry it into a one-click detector after a picker
                // change, even if React has not painted the cleared canvas yet.
                marks: selectedModelMode === "prompted" ? marks : [],
                preload_data_item_ids: nextImageIds,
                output_shape: project?.type === "Object Detection" ? "rectangle" : aiShape,
            })

            const predictions = (response.data?.result?.shapes || []) as AutoLabelingPrediction[]
            if (selectedModelMode === "automatic") {
                const previousIds = autoPredictionIdsRef.current.get(currentImage.id) ?? new Set<string>()
                const existingShapes = newShapes.filter(
                    (shape) =>
                        !previousIds.has(shape.id) &&
                        shape.auto_labeling_model !== selectedModel &&
                        !shape.categories?.some((category) => category.startsWith("AUTOLABEL_"))
                )
                const projectLabels = new Set(project?.labels.map((label) => label.name) || [])
                const ids = new Set<string>()
                const predictedShapes = predictions
                    .filter(
                        (prediction) =>
                            prediction.points?.length > 0 &&
                            typeof prediction.label === "string" &&
                            projectLabels.has(prediction.label)
                    )
                    .map((prediction) => {
                        const id = generateUniqueId()
                        ids.add(id)
                        return createAutoLabelingPrediction(
                            prediction,
                            project?.type,
                            aiShape,
                            id,
                            selectedModel
                        ) as any as Shape
                    })
                autoPredictionIdsRef.current.set(currentImage.id, ids)
                setShapesHandle([...existingShapes, ...predictedShapes])
                setSavingStatus(
                    predictedShapes.length === 0
                        ? "No predictions matched this project's labels"
                        : `Added ${predictedShapes.length} model prediction${predictedShapes.length === 1 ? "" : "s"}`
                )
                setTimeout(() => setSavingStatus(""), 5000)
                return
            }

            const predictedShape = predictions[0]
            if (!predictedShape?.points?.length) return
            const points = predictedShape.points.map((point) => [point.x, point.y])
            const tmpShape = createAutoLabelingPreview(
                points,
                project?.type,
                aiShape,
                generateUniqueId()
            ) as any as Shape

            newShapes = [...newShapes, tmpShape]
            annotator?.updateCategories(tmpShape.id, tmpShape.categories)
            setShapesHandle(newShapes)
        } catch (error: any) {
            console.error("Auto labeling inference failed:", error)
            setSavingStatus(error?.response?.data?.detail || "Auto-labeling inference failed")
            setTimeout(() => setSavingStatus(""), 5000)
        } finally {
            setIsInferencing(false)
        }
    }

    const setShapeCategories = useCallback(
        (shape: Shape, categories: string[]) => {
            if (categories.length > 1) {
                console.warn("Item has more than 1 category")
            }

            shape.categories = categories

            const label = project?.labels.find((label) => label.name === categories[0])
            let color: ShapeColor | undefined
            if (label?.color) {
                color = label.color
                shape.color = color
            }

            annotator?.updateCategories(shape.id, categories, color)
        },
        [annotator, project?.labels]
    )

    const selectedCategoriesChanged = useCallback(
        (categories: string[]) => {
            if (dialog.shape) {
                setShapeCategories(dialog.shape, categories)
            }
        },
        [dialog.shape, setShapeCategories]
    )

    const keypointMetadataChanged = useCallback(
        (groupId: string | number | null, visible: number) => {
            if (dialog.shape?.type !== "dot") return
            const dot = dialog.shape as Dot
            dot.group_id = groupId
            dot.visible = visible
            annotator?.updateKeypointMetadata(dot.id, groupId, visible)
        },
        [annotator, dialog.shape]
    )

    const navigateImage = useCallback(
        async (direction: number) => {
            if (isLoadingImage || isLoadingAnnotation || isSavingAnnotation) return

            isNavigatingRef.current = true
            if (autoSaveTimeoutRef.current) {
                clearTimeout(autoSaveTimeoutRef.current)
            }

            // Only save if there are unsaved changes and autosave is enabled
            if (hasUnsavedChanges() && useAutoSaveSettingStore.getState().isEnabled) {
                await saveAnnotation()
            }

            setTimeout(() => {
                const newIndex = currentImageIndex + direction
                if (newIndex >= 0 && newIndex < dataItems.length) {
                    setCurrentImageIndex(newIndex)
                } else if (newIndex < 0 && currentPage > 1) {
                    handlePageChange(currentPage - 1)
                } else if (newIndex >= dataItems.length && currentPage < totalPages) {
                    handlePageChange(currentPage + 1)
                }
                isNavigatingRef.current = false
            }, 100)
        },
        [
            isSavingAnnotation,
            currentImageIndex,
            currentPage,
            dataItems.length,
            handlePageChange,
            isLoadingAnnotation,
            isLoadingImage,
            saveAnnotation,
            totalPages,
            hasUnsavedChanges,
        ]
    )

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (isLoadingImage || isLoadingAnnotation) return

            if (event.key === "ArrowLeft") {
                navigateImage(-1)
            } else if (event.key === "ArrowRight") {
                navigateImage(1)
            } else if (event.key === "f" && mode === "auto_labeling") {
                handleFinishAutoLabel()
            }
        }

        window.addEventListener("keydown", handleKeyDown)

        return () => {
            window.removeEventListener("keydown", handleKeyDown)
        }
    }, [navigateImage, mode, handleFinishAutoLabel, isLoadingImage, isLoadingAnnotation])

    if (showNoLabelsAlert) {
        return <NoLabelsAlert projectId={projectId} />
    }

    return (
        <div className="bg-background text-foreground fixed inset-0 flex h-full flex-col">
            <TopBar
                projectName={project?.name.slice(0, 80)}
                subset={subset}
                savingStatus={savingStatus}
                onExit={onExit}
            />
            {aiEnabled && (
                <AutoLabellingToolbar
                    aiShape={aiShape}
                    setAiShape={setAiShape}
                    mode={mode}
                    setMode={setMode}
                    aiToolSelected={aiToolSelected}
                    projectId={projectId}
                    projectType={project?.type || ""}
                    projectLabels={(project?.labels || []).map((label) => label.name)}
                    handleToolSelect={handleToolSelect}
                    model={selectedModel}
                    selectModel={selectAutoLabelingModel}
                    clear={() => {
                        clearAutoLabelingShapes(false)
                        clearAutomaticPredictions()
                    }}
                    finish={handleFinishAutoLabel}
                    run={() => inferenceShape(annotator?.getShapes() || [])}
                    isInferencing={isInferencing}
                />
            )}
            <div className="flex min-h-0 flex-1 overflow-hidden">
                {/* Main content area. `relative` anchors the floating tool
                    palette here — without it the palette positioned against the
                    viewport and covered the top bars. */}
                <div className="relative min-w-0 flex-1 overflow-hidden">
                    {dataItems[currentImageIndex] && (
                        <>
                            {project?.type !== "Image Classification" &&
                                project?.type !== "Handpose Classification" && (
                                    <LeftBar
                                        mode={mode}
                                        projectType={project?.type || ""}
                                        aiEnabled={aiEnabled}
                                        annotator={annotator}
                                        selectedTool={selectedTool}
                                        handleToolSelect={handleToolSelect}
                                        setMode={setMode}
                                        setAiEnabled={setAiEnabled}
                                        saveAnnotation={saveAnnotation}
                                        clearAll={() => {
                                            annotator?.setShapes([])
                                        }}
                                    />
                                )}
                            {/* Positioned rather than `flex-1`: the parent is a
                                block container, so `flex-1` did nothing and this
                                pane's height fell back to its content -- while
                                the annotator sizes its canvas from *this* pane.
                                The two fed each other and settled at the SVG's
                                150px default, rendering a full photo as a
                                thumbnail in a 700px-tall pane. `absolute
                                inset-0` gives a definite size that does not
                                depend on the content, breaking the cycle. */}
                            <div className="bg-surface-sunken absolute inset-0" ref={annotatorContainerRef}>
                                <ImageAnnotator
                                    key={imageKey}
                                    width={annotatorWidth}
                                    height={annotatorHeight}
                                    setHandles={setHandles}
                                    imageUrl={currentImage?.path + "?token=" + window?.pywebview?.token}
                                    onAdded={(shape: Shape) => {
                                        if (mode === "auto_labeling") {
                                            const categoryMap = {
                                                addPoint: "AUTOLABEL_ADD_POINT",
                                                removePoint: "AUTOLABEL_REM_POINT",
                                                addRect: "AUTOLABEL_ADD_RECT",
                                                removeRect: "AUTOLABEL_REM_RECT",
                                                tmpShape: "AUTOLABEL_TMP_SHAPE",
                                            }
                                            const category = categoryMap[aiToolSelected as keyof typeof categoryMap]
                                            if (category) {
                                                shape.categories = [category]
                                                annotator?.updateCategories(shape.id, [category])
                                            }

                                            // Stop editing and adding shapes
                                            const shapes = annotator?.getShapes() || []

                                            // Run inference
                                            setTimeout(() => {
                                                inferenceShape(shapes)
                                            }, 200)
                                        } else {
                                            setDialog({ show: true, shape })
                                        }
                                    }}
                                    onContextMenu={(shape: Shape) => setDialog({ show: true, shape })}
                                    onReady={(annotator) => {
                                        setReady(true)
                                        clearNoCategoryShapes()
                                    }}
                                />
                                {dialog.show && dialog.shape && (
                                    <Dialog
                                        key={dialog.shape.id}
                                        items={dialog.shape.categories}
                                        itemsChanged={selectedCategoriesChanged}
                                        onEdit={() => {
                                            if (dialog.shape?.id !== undefined) {
                                                annotator?.edit(dialog.shape.id)
                                                hideAndUpdateCategories()
                                                clearAutoLabelingShapes(true)
                                            }
                                            clearNoCategoryShapes()
                                        }}
                                        onDelete={() => {
                                            if (dialog.shape?.id !== undefined) {
                                                annotator?.delete(dialog.shape.id)
                                                hideDialog()
                                            }
                                            clearNoCategoryShapes()
                                        }}
                                        onClose={() => {
                                            hideDialog()
                                            clearAutoLabelingShapes(false)
                                            clearNoCategoryShapes()
                                        }}
                                        offset={dialog.shape.getCenterWithOffset()}
                                        categories={
                                            project?.labels?.map((label) => ({
                                                name: label.name,
                                                color: label.color || "#000000",
                                            })) || []
                                        }
                                        keypoint={
                                            project?.type === "Keypoint Detection" && dialog.shape.type === "dot"
                                                ? {
                                                      groupId: (dialog.shape as Dot).group_id,
                                                      visible: (dialog.shape as Dot).visible,
                                                  }
                                                : undefined
                                        }
                                        keypointChanged={keypointMetadataChanged}
                                    />
                                )}
                            </div>
                        </>
                    )}
                </div>
                <RightBar
                    projectId={projectId}
                    project={project}
                    currentImage={currentImage}
                    currentPage={currentPage}
                    currentImageIndex={currentImageIndex}
                    totalPages={totalPages}
                    getShapes={() => annotator?.getShapes() || []}
                    dataItems={dataItems}
                    isLoading={isLoading}
                    isError={isError}
                    setCurrentImageIndex={async (index) => {
                        if (hasUnsavedChanges() && useAutoSaveSettingStore.getState().isEnabled) {
                            await saveAnnotation()
                        }
                        setCurrentImageIndex(index)
                    }}
                    handleClassChange={handleClassChange}
                    handlePageChange={async (dir) => {
                        if (hasUnsavedChanges() && useAutoSaveSettingStore.getState().isEnabled) {
                            await saveAnnotation()
                        }
                        handlePageChange(dir)
                    }}
                />
            </div>

            <BottomBar
                projectType={project?.type || ""}
                navigateImage={navigateImage}
                selectedTool={selectedTool}
                currentImageIndex={currentImageIndex}
                currentPage={currentPage}
                totalPages={totalPages}
                dataItems={dataItems}
            />

            {(isInferencing || isLoadingImage || isLoadingAnnotation || isSavingAnnotation) && (
                <div className="pointer-events-none fixed inset-0 flex items-center justify-center">
                    <span className="bg-popover/90 text-foreground flex items-center gap-2 rounded-md border px-3 py-2 text-xs shadow-lg">
                        <Loader2 className="size-4 animate-spin" />
                        {isSavingAnnotation ? "Saving…" : isInferencing ? "Running the model…" : "Loading…"}
                    </span>
                </div>
            )}
        </div>
    )
}

export default LabelingScreen

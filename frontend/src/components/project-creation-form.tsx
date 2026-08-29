import { zodResolver } from "@hookform/resolvers/zod"
import { useQuery } from "@tanstack/react-query"
import { Box, FileText, Hand, Image, Layers, MapPin, Table2 } from "lucide-react"
import React from "react"
import { useForm } from "react-hook-form"
import * as z from "zod"

import { Button } from "@/components/ui/button"
import { DialogClose } from "@/components/ui/dialog"
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/components/ui/use-toast"
import { getJson } from "@/lib/api"
import { TEXT_AI_PROJECT_TYPE } from "@/lib/project-types"
import { ProjectCreation } from "@/types"

const formSchema = z.object({
    name: z.string().min(1, "Project name is required"),
    type: z.enum([
        "Object Detection",
        "Image Classification",
        "Image Segmentation",
        "Handpose Classification",
        "Sentiment Analysis",
        "Text AI",
        "Tabular AI",
        "Instance Segmentation",
        "Keypoint Detection",
    ]),
    description: z.string().optional(),
})

interface ProjectCreationFormProps {
    onSubmit: (project: ProjectCreation) => void
}

const ProjectCreationForm: React.FC<ProjectCreationFormProps> = ({ onSubmit }) => {
    const { toast } = useToast()
    // What this machine can do. Probing costs one model load on the server, so
    // it is fetched once and kept.
    const { data: capabilities } = useQuery({
        queryKey: ["capabilities"],
        queryFn: () => getJson<{ handpose: boolean; handpose_reason: string | null }>("/api/settings/capabilities"),
        staleTime: Infinity,
    })
    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            name: "",
            type: "Object Detection",
            description: "",
        },
    })

    const handleSubmit = (values: z.infer<typeof formSchema>) => {
        const newProject = {
            ...values,
        }
        onSubmit(newProject)
        form.reset()
        toast({
            title: "Project Created",
            description: `${newProject.name} has been successfully created.`,
        })
    }

    return (
        <Form {...form}>
            <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-8">
                <FormField
                    control={form.control}
                    name="name"
                    render={({ field }) => (
                        <FormItem>
                            <FormLabel>Project Name</FormLabel>
                            <FormControl>
                                <Input placeholder="Enter project name" {...field} />
                            </FormControl>
                            <FormMessage />
                        </FormItem>
                    )}
                />
                <FormField
                    control={form.control}
                    name="type"
                    render={({ field }) => (
                        <FormItem>
                            <FormLabel>Project Type</FormLabel>
                            <Select onValueChange={field.onChange} defaultValue={field.value} value={field.value}>
                                <FormControl>
                                    <SelectTrigger>
                                        <SelectValue>
                                            <div className="flex items-center">
                                                {field.value === "Object Detection" && <Box className="mr-2 h-4 w-4" />}
                                                {field.value === "Image Classification" && (
                                                    <Image className="mr-2 h-4 w-4" />
                                                )}
                                                {field.value === "Image Segmentation" && (
                                                    <Layers className="mr-2 h-4 w-4" />
                                                )}
                                                {field.value === "Handpose Classification" && (
                                                    <Hand className="mr-2 h-4 w-4" />
                                                )}
                                                {field.value === "Sentiment Analysis" && (
                                                    <FileText className="mr-2 h-4 w-4" />
                                                )}
                                                {field.value === TEXT_AI_PROJECT_TYPE && (
                                                    <FileText className="mr-2 h-4 w-4" />
                                                )}
                                                {field.value === "Tabular AI" && <Table2 className="mr-2 h-4 w-4" />}
                                                {field.value === "Instance Segmentation" && (
                                                    <Layers className="mr-2 h-4 w-4" />
                                                )}
                                                {field.value === "Keypoint Detection" && (
                                                    <MapPin className="mr-2 h-4 w-4" />
                                                )}
                                                {field.value}
                                            </div>
                                        </SelectValue>
                                    </SelectTrigger>
                                </FormControl>
                                <SelectContent>
                                    <SelectItem value="Object Detection">
                                        <div className="flex items-center">
                                            <Box className="mr-2 h-4 w-4" />
                                            Object Detection
                                        </div>
                                    </SelectItem>
                                    <SelectItem value="Image Classification">
                                        <div className="flex items-center">
                                            <Image className="mr-2 h-4 w-4" />
                                            Image Classification
                                        </div>
                                    </SelectItem>
                                    <SelectItem value="Image Segmentation">
                                        <div className="flex items-center">
                                            <Layers className="mr-2 h-4 w-4" />
                                            Image Segmentation
                                        </div>
                                    </SelectItem>
                                    {/* Disabled where the hand landmark model
                                        cannot run -- macOS, where mediapipe
                                        aborts. The project would be creatable
                                        and then impossible to fill: every
                                        uploaded image is dropped for want of
                                        landmarks. Better here than after
                                        someone has collected photographs. */}
                                    <SelectItem
                                        value="Handpose Classification"
                                        disabled={capabilities ? !capabilities.handpose : false}
                                    >
                                        <div className="flex items-center">
                                            <Hand className="mr-2 h-4 w-4" />
                                            Handpose Classification
                                            {capabilities && !capabilities.handpose && (
                                                <span className="text-muted-foreground ml-2 text-xs">
                                                    not available on this machine
                                                </span>
                                            )}
                                        </div>
                                    </SelectItem>
                                    <SelectItem value="Instance Segmentation">
                                        <div className="flex items-center">
                                            <Image className="mr-2 h-4 w-4" />
                                            Instance Segmentation
                                        </div>
                                    </SelectItem>
                                    <SelectItem value="Keypoint Detection">
                                        <div className="flex items-center">
                                            <MapPin className="mr-2 h-4 w-4" />
                                            Keypoint Detection
                                        </div>
                                    </SelectItem>
                                    <SelectItem value="Tabular AI">
                                        <div className="flex items-center">
                                            <Table2 className="mr-2 h-4 w-4" />
                                            Tabular AI
                                        </div>
                                    </SelectItem>
                                    <SelectItem value={TEXT_AI_PROJECT_TYPE}>
                                        <div className="flex items-center">
                                            <FileText className="mr-2 h-4 w-4" />
                                            Text AI
                                        </div>
                                    </SelectItem>
                                </SelectContent>
                            </Select>
                            <FormMessage />
                        </FormItem>
                    )}
                />
                <FormField
                    control={form.control}
                    name="description"
                    render={({ field }) => (
                        <FormItem>
                            <FormLabel>Project Description</FormLabel>
                            <FormControl>
                                <Textarea placeholder="Enter project description (optional)" {...field} />
                            </FormControl>
                            <FormMessage />
                        </FormItem>
                    )}
                />
                <DialogClose asChild>
                    <Button type="submit">Create Project</Button>
                </DialogClose>
            </form>
        </Form>
    )
}

export default ProjectCreationForm

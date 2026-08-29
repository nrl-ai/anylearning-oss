import React from "react"

import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

const fakeTrainingSessions = [
    {
        id: 1,
        model: "ResNet50",
        dataset: "ImageNet Subset",
        duration: "3h 45m",
        status: "Completed",
    },
    {
        id: 2,
        model: "VGG16",
        dataset: "CIFAR-10",
        duration: "2h 30m",
        status: "In Progress",
    },
    {
        id: 3,
        model: "Inception-v3",
        dataset: "Places365",
        duration: "4h 15m",
        status: "Completed",
    },
    {
        id: 4,
        model: "EfficientNet-B0",
        dataset: "Stanford Dogs",
        duration: "1h 45m",
        status: "Failed",
    },
    {
        id: 5,
        model: "MobileNet-v2",
        dataset: "Flowers Recognition",
        duration: "2h",
        status: "Completed",
    },
]

export const RecentTrainingSessions: React.FC = () => {
    return (
        <Table>
            <TableHeader>
                <TableRow>
                    <TableHead>Model</TableHead>
                    <TableHead>Dataset</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead>Status</TableHead>
                </TableRow>
            </TableHeader>
            <TableBody>
                {fakeTrainingSessions.map((session) => (
                    <TableRow key={session.id}>
                        <TableCell className="font-medium">{session.model}</TableCell>
                        <TableCell>{session.dataset}</TableCell>
                        <TableCell>{session.duration}</TableCell>
                        <TableCell>
                            <Badge
                                variant={
                                    session.status === "Completed"
                                        ? "outline"
                                        : session.status === "In Progress"
                                          ? "default"
                                          : "destructive"
                                }
                            >
                                {session.status}
                            </Badge>
                        </TableCell>
                    </TableRow>
                ))}
            </TableBody>
        </Table>
    )
}

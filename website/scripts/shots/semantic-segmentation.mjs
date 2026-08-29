import { standardWorkflow } from "./standard-workflow.mjs";

export const project = { name: "EM Particles (CC BY 4.0)", type: "Image Segmentation" };

export const shots = standardWorkflow({
  folder: "semantic_segmentation",
  labelTool: "Use polygons to trace each class region",
  files: {
    create: "01_create_project.png",
    labels: "02_create_labels.png",
    dataset: "03_upload_datasets.png",
    labeler: "04_label_polygons.png",
    trainingDialog: "05_create_training_job.png",
    trainings: "06_view_all_trainings.png",
    details: "07_monitor_training.png",
    tryModel: "08_go_to_inference.png",
    result: "09_make_prediction.png",
    export: "10_export_model.png",
  },
});

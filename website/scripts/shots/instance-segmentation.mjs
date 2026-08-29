import { standardWorkflow } from "./standard-workflow.mjs";

export const project = {
  name: "EM Particles Instances (CC BY 4.0)",
  type: "Instance Segmentation",
};

export const shots = standardWorkflow({
  folder: "instance_segmentation",
  labelTool: "Trace every object as a separate polygon",
  files: {
    create: "01_create_project.png",
    labels: "02_create_labels.png",
    dataset: "03_upload_datasets.png",
    labeler: "04_label_objects.png",
    trainingDialog: "05_create_training_job.png",
    trainings: "06_view_all_trainings.png",
    details: "07_monitor_training.png",
    tryModel: "08_try_model.png",
    result: "09_make_prediction.png",
    export: "10_export_model.png",
  },
});

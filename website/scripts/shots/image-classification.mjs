import { standardWorkflow } from "./standard-workflow.mjs";

export const project = { name: "Chest X-Ray (CC BY 4.0)", type: "Image Classification" };

export const shots = standardWorkflow({
  folder: "image_classification",
  includeLabeler: false,
  files: {
    create: "01_create_project.png",
    labels: "02_create_classes.png",
    dataset: "03_folder_structure.png",
    trainingDialog: "04_create_new_training.png",
    trainings: "05_all_trainings.png",
    details: "06_view_training_detail.png",
    tryModel: "07_try_model.png",
    result: "08_view_model_prediction.png",
    export: "09_export_model.png",
  },
});

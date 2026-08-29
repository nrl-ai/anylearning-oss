import { standardWorkflow } from "./standard-workflow.mjs";

export const project = { name: "ASL Letters (Public Domain)", type: "Handpose Classification" };

export const shots = standardWorkflow({
  folder: "handpose_classification",
  includeLabeler: false,
  files: {
    create: "01_create_project.png",
    labels: "02_edit_class_names.png",
    dataset: "03_folder_structure.png",
    trainingDialog: "04_create_new_training.png",
    trainings: "05_all_trainings.png",
    details: "06_training_details.png",
    tryModel: "07_try_model.png",
    result: "08_try_model_result.png",
  },
});

import { activateTab, clickAndWaitForPrediction, clickText } from "./standard-workflow.mjs";

export const project = { name: "Desert Locust Pose (Apache 2.0)", type: "Keypoint Detection" };

const hideLabelerDebugPanel = {
  eval: `const el=[...document.querySelectorAll('p')].find(e=>/DEV ONLY/i.test(e.textContent));
         if(el){el.closest('div').parentElement.style.display='none'} true`,
  wait: 400,
};

export const shots = [
  {
    file: "keypoint_detection/01_project_overview.png",
    path: "/projects/overview",
    viewport: [1440, 900],
    notes: [
      { type: "label", x: 330, y: 385, text: "One object class, with an ordered landmark schema" },
    ],
  },
  {
    file: "keypoint_detection/02_dataset_imported.png",
    path: "/projects/dataset",
    viewport: [1440, 900],
    notes: [{ type: "label", x: 330, y: 205, text: "Keep a held-out validation split" }],
  },
  {
    file: "keypoint_detection/03_label_instances.png",
    path: "/projects/dataset",
    viewport: [1440, 900],
    steps: [{ eval: clickText("Start labelling|Label now"), wait: 9000 }, hideLabelerDebugPanel],
    notes: [
      { type: "label", x: 70, y: 150, text: "Place named landmarks in schema order" },
      { type: "label", x: 1160, y: 155, text: "Review names and visibility", anchor: "end" },
    ],
  },
  {
    file: "keypoint_detection/04_training_settings.png",
    path: "/projects/training",
    viewport: [1280, 860],
    steps: [{ eval: clickText("Start training"), wait: 1500 }],
    notes: [
      { type: "label", x: 420, y: 310, text: "RF-DETR predicts boxes and landmarks together" },
    ],
  },
  {
    file: "keypoint_detection/05_training_results.png",
    path: "/projects/training",
    viewport: [1440, 900],
    notes: [{ type: "label", x: 310, y: 205, text: "Judge box and keypoint mAP separately" }],
  },
  {
    file: "keypoint_detection/06_inference_result.png",
    path: "/projects/models",
    viewport: [1440, 900],
    requiresModel: true,
    steps: [
      { eval: clickText("Try"), wait: 900 },
      { eval: activateTab("Use a test image"), wait: 600 },
      { eval: clickAndWaitForPrediction, wait: 600 },
    ],
    notes: [],
  },
];

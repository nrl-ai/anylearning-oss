/**
 * Screenshots for pages/docs/object-detection.mdx.
 *
 * Each entry names the file the tutorial already references, so re-running the
 * capture replaces images in place and the prose never has to change.
 *
 * `project` is an id in the local ~/anylearning-data database. The capture
 * script fails loudly if the project is missing rather than quietly shooting
 * the wrong screen — a tutorial illustrated with someone else's data is worse
 * than a stale screenshot.
 */

import { activateTab, clickAndWaitForPrediction } from "./standard-workflow.mjs";

export const project = { name: "Helmet & Jacket (Apache 2.0)", type: "Object Detection" };

const clickText = (text) =>
  `[...document.querySelectorAll('button,a')].find(el => /${text}/i.test(el.textContent))?.click(); true`;

export const shots = [
  {
    file: "object_detection/project-creation-helmet.png",
    caption: "Creating an Object Detection project",
    viewport: [1280, 860],
    path: "/projects/overview",
    steps: [{ eval: clickText("Create project"), wait: 1200 }],
    notes: [
      { type: "label", x: 430, y: 250, text: "Name the project" },
      { type: "label", x: 430, y: 330, text: "Pick Object Detection" },
    ],
  },
  {
    file: "object_detection/edit-class-names.png",
    caption: "Defining the label set",
    viewport: [1440, 900],
    path: "/projects/overview",
    notes: [
      { type: "marker", n: 1, x: 300, y: 386 },
      { type: "label", x: 322, y: 386, text: "Add one class per object you want detected" },
    ],
  },
  {
    file: "object_detection/upload-dataset.png",
    caption: "Uploading images into a split",
    viewport: [1440, 900],
    path: "/projects/dataset",
    notes: [
      { type: "marker", n: 1, x: 300, y: 108 },
      { type: "label", x: 322, y: 108, text: "Choose the split first" },
      { type: "marker", n: 2, x: 300, y: 212 },
      { type: "label", x: 322, y: 212, text: "Upload a .zip of images" },
    ],
  },
  {
    file: "object_detection/labeling-interface.png",
    caption: "The labelling canvas",
    viewport: [1440, 900],
    path: "/projects/dataset",
    steps: [
      { eval: clickText("Start labelling"), wait: 9000 },
      // The shapes debugger is a development-only panel; it does not exist in
      // the shipped app, so it must not appear in the documentation.
      {
        eval: `const el=[...document.querySelectorAll('p')].find(e=>/DEV ONLY/i.test(e.textContent));
               if(el){el.closest('div').parentElement.style.display='none'} true`,
        wait: 400,
      },
    ],
    notes: [
      { type: "ring", x: 12, y: 128, w: 42, h: 300 },
      { type: "label", x: 66, y: 150, text: "Tools — pick the rectangle to draw" },
      { type: "label", x: 1150, y: 140, text: "Your classes", anchor: "end" },
    ],
  },
  {
    file: "object_detection/new-training.png",
    caption: "Starting a training run",
    viewport: [1280, 860],
    path: "/projects/training",
    steps: [{ eval: clickText("Start training"), wait: 1500 }],
    notes: [
      {
        type: "label",
        x: 400,
        y: 300,
        text: "Bigger variants are slower but usually more accurate",
      },
    ],
  },
  {
    file: "object_detection/training.png",
    caption: "A run in progress",
    viewport: [1440, 900],
    path: "/projects/training",
    notes: [{ type: "label", x: 300, y: 200, text: "Metrics update while the run works" }],
  },
  {
    file: "object_detection/training-details.png",
    caption: "Run details and metrics",
    viewport: [1440, 900],
    path: "/projects/training",
    steps: [{ eval: clickText("View details"), wait: 1800 }],
    notes: [],
  },
  {
    file: "object_detection/try-model.png",
    caption: "Trying a trained model",
    viewport: [1440, 900],
    path: "/projects/models",
    requiresModel: true,
    notes: [{ type: "label", x: 900, y: 216, text: "Try runs the model on one image" }],
  },
  {
    file: "object_detection/try-model-result.png",
    caption: "A prediction on a held-out image",
    viewport: [1440, 900],
    path: "/projects/models",
    requiresModel: true,
    steps: [
      { eval: clickText("Try"), wait: 900 },
      { eval: activateTab("Use a test image"), wait: 600 },
      { eval: clickAndWaitForPrediction, wait: 600 },
    ],
    notes: [],
  },
  {
    file: "object_detection/download-models.png",
    caption: "Exporting a model",
    viewport: [1440, 900],
    path: "/projects/models",
    requiresModel: true,
    steps: [
      { eval: `document.querySelector('[aria-label="Download model"]')?.click(); true`, wait: 900 },
    ],
    notes: [{ type: "label", x: 900, y: 260, text: "ONNX for other runtimes" }],
  },
];

import { clickText } from "./standard-workflow.mjs";

export const project = { name: "Helmet & Jacket (Apache 2.0)", type: "Object Detection" };

export const shots = [
  { file: "screenshots/1.png", path: "/projects/overview", viewport: [1600, 1000], notes: [] },
  { file: "screenshots/2.png", path: "/projects/dataset", viewport: [1600, 1000], notes: [] },
  {
    file: "screenshots/3.png",
    path: "/projects/dataset",
    viewport: [1600, 1000],
    steps: [{ eval: clickText("Start labelling|Label now"), wait: 9000 }],
    notes: [],
  },
  { file: "screenshots/4.png", path: "/projects/training", viewport: [1600, 1000], notes: [] },
  {
    file: "screenshots/5.png",
    path: "/projects/models",
    viewport: [1600, 1000],
    requiresModel: true,
    notes: [],
  },
];

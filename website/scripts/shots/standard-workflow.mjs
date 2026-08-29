/** Shared capture specs for the task guides.
 *
 * The guides follow the same product journey—project, dataset, training,
 * models—but keep their own filenames and real demo project. Centralising the
 * interactions means a button rename is fixed once instead of in six scripts.
 */

export const clickText = (text, selector = "button,a") =>
  `[...document.querySelectorAll(${JSON.stringify(selector)})].find(el => ${JSON.stringify(
    text
  )}.toLowerCase().split('|').some(part => el.textContent.toLowerCase().includes(part)))?.click(); true`;

export const activateTab = (label) => `(() => {
  const tab = [...document.querySelectorAll('[role="tab"]')]
    .find(el => el.textContent.trim() === ${JSON.stringify(label)});
  if (!tab) throw new Error('Tab not found: ' + ${JSON.stringify(label)});
  tab.dispatchEvent(new PointerEvent('pointerdown', {
    bubbles: true,
    button: 0,
    pointerId: 1,
    pointerType: 'mouse'
  }));
  tab.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0 }));
  tab.click();
  return true;
})()`;

export const clickAndWaitForPrediction = `(async () => {
  const button = [...document.querySelectorAll('button')]
    .find(el => el.textContent.trim() === 'Pick a random test image');
  if (!button) throw new Error('Random test image button not found');
  button.click();

  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    const prediction = document.querySelector(
      'img[alt="The model\\'s prediction drawn over the input image"]'
    );
    const result = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6,div')]
      .some(el => el.textContent.trim() === 'Result');
    if (prediction || result) return;
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  throw new Error('Inference did not produce a visible result within 90 seconds');
})()`;

const hideLabelerDebugPanel = {
  eval: `const el=[...document.querySelectorAll('p')].find(e=>/DEV ONLY/i.test(e.textContent));
         if(el){el.closest('div').parentElement.style.display='none'} true`,
  wait: 400,
};

const openTryDialog = { eval: clickText("Try"), wait: 900 };
const chooseTestImage = {
  eval: activateTab("Use a test image"),
  wait: 600,
};
const runTestInference = {
  eval: clickAndWaitForPrediction,
  wait: 600,
};

export function standardWorkflow({
  folder,
  files,
  labelTool,
  includeLabeler = true,
  captions = {},
}) {
  const shots = [
    {
      file: `${folder}/${files.create}`,
      caption: captions.create ?? "Create a task-specific project",
      viewport: [1280, 860],
      path: "/projects/overview",
      steps: [{ eval: clickText("Create project"), wait: 1200 }],
      notes: [
        { type: "marker", n: 1, x: 370, y: 255 },
        { type: "label", x: 395, y: 255, text: "Name the project" },
        { type: "marker", n: 2, x: 370, y: 340 },
        { type: "label", x: 395, y: 340, text: "Choose the task" },
      ],
    },
    {
      file: `${folder}/${files.labels}`,
      caption: captions.labels ?? "Review the label set",
      viewport: [1440, 900],
      path: "/projects/overview",
      notes: [
        { type: "marker", n: 1, x: 300, y: 385 },
        { type: "label", x: 324, y: 385, text: "Keep class names consistent" },
      ],
    },
    {
      file: `${folder}/${files.dataset}`,
      caption: captions.dataset ?? "Inspect the imported dataset",
      viewport: [1440, 900],
      path: "/projects/dataset",
      notes: [
        { type: "marker", n: 1, x: 300, y: 108 },
        { type: "label", x: 324, y: 108, text: "Choose training, validation or test" },
      ],
    },
  ];

  if (includeLabeler && files.labeler) {
    shots.push({
      file: `${folder}/${files.labeler}`,
      caption: captions.labeler ?? "Label a real project image",
      viewport: [1440, 900],
      path: "/projects/dataset",
      steps: [{ eval: clickText("Start labelling|Label now"), wait: 9000 }, hideLabelerDebugPanel],
      notes: [
        { type: "ring", x: 12, y: 128, w: 42, h: 300 },
        { type: "label", x: 66, y: 150, text: labelTool },
        { type: "label", x: 1150, y: 140, text: "Classes and annotations", anchor: "end" },
      ],
    });
  }

  shots.push(
    {
      file: `${folder}/${files.trainingDialog}`,
      caption: captions.trainingDialog ?? "Configure a training run",
      viewport: [1280, 860],
      path: "/projects/training",
      steps: [{ eval: clickText("Start training"), wait: 1500 }],
      notes: [
        { type: "label", x: 430, y: 300, text: "Start with the model defaults" },
        { type: "label", x: 430, y: 470, text: "Use Automatic hardware unless you need CPU" },
      ],
    },
    {
      file: `${folder}/${files.trainings}`,
      caption: captions.trainings ?? "Compare completed runs",
      viewport: [1440, 900],
      path: "/projects/training",
      notes: [
        {
          type: "label",
          x: 300,
          y: 205,
          text: "Compare validation metrics, not training loss alone",
        },
      ],
    },
    {
      file: `${folder}/${files.details}`,
      caption: captions.details ?? "Inspect metrics and logs",
      viewport: [1440, 900],
      path: "/projects/training",
      steps: [{ eval: clickText("View details"), wait: 1800 }],
      notes: [],
    },
    {
      file: `${folder}/${files.tryModel}`,
      caption: captions.tryModel ?? "Choose a model to try",
      viewport: [1440, 900],
      path: "/projects/models",
      requiresModel: true,
      steps: [openTryDialog],
      notes: [{ type: "label", x: 850, y: 220, text: "Use a held-out image for an honest check" }],
    },
    {
      file: `${folder}/${files.result}`,
      caption: captions.result ?? "Inspect a real model prediction",
      viewport: [1440, 900],
      path: "/projects/models",
      requiresModel: true,
      steps: [openTryDialog, chooseTestImage, runTestInference],
      notes: [],
    }
  );

  if (files.export) {
    shots.push({
      file: `${folder}/${files.export}`,
      caption: captions.export ?? "Export a trained model",
      viewport: [1440, 900],
      path: "/projects/models",
      requiresModel: true,
      steps: [
        {
          eval: `document.querySelector('[aria-label="Download model"]')?.click(); true`,
          wait: 700,
        },
      ],
      notes: [
        {
          type: "label",
          x: 1120,
          y: 260,
          text: "Download PyTorch or portable ONNX",
          anchor: "end",
        },
      ],
    });
  }

  return shots;
}

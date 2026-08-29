/** Capture deterministic screenshots for the Tabular AI and Text AI guides. */

import fs from "node:fs";
import path from "node:path";

import { launch } from "./lib/browser.mjs";

const app = process.env.ANYLEARNING_APP_URL || "http://localhost:3021";
const output = path.resolve("public/structured_ai");
const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

fs.mkdirSync(output, { recursive: true });

async function evaluate(browser, expression) {
  const result = await browser.call("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
  return result.result?.value;
}

async function waitFor(browser, expression, timeout = 30_000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    if (await evaluate(browser, expression)) return;
    await sleep(250);
  }
  throw new Error(`Timed out waiting for ${expression}`);
}

async function navigate(browser, route, expectedText) {
  await browser.call("Page.navigate", { url: `${app}${route}` });
  await waitFor(browser, `document.body?.innerText.includes(${JSON.stringify(expectedText)})`);
  await sleep(900);
}

async function screenshot(browser, filename) {
  await evaluate(
    browser,
    `(() => { document.querySelector('nextjs-portal')?.remove(); document.documentElement.style.scrollBehavior='auto'; scrollTo(0,0); return true; })()`
  );
  await sleep(250);
  const { data } = await browser.call("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: false,
  });
  fs.writeFileSync(path.join(output, filename), Buffer.from(data, "base64"));
}

const browser = await launch({ width: 1440, height: 1000, scale: 1 });
await browser.setViewport(1440, 1000);
try {
  await navigate(browser, "/projects/dataset?projectId=1", "Import from Hugging Face");
  await screenshot(browser, "import-choices.png");

  await navigate(browser, "/projects/dataset?projectId=2", "45,211");
  await screenshot(browser, "tabular-dataset.png");

  await navigate(browser, "/projects/models?projectId=2", "Structured model report");
  await sleep(1_500);
  await screenshot(browser, "tabular-model.png");

  await navigate(browser, "/projects/dataset?projectId=3", "Import from Hugging Face");
  await evaluate(
    browser,
    `(() => { const input=document.querySelector('input[placeholder="owner/dataset-name"]'); const setter=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set; setter.call(input,'google/IFEval'); input.dispatchEvent(new Event('input',{bubbles:true})); return true; })()`
  );
  await sleep(150);
  await evaluate(
    browser,
    `(() => { [...document.querySelectorAll('button')].find((button)=>button.textContent.trim()==='Inspect')?.click(); return true; })()`
  );
  await waitFor(
    browser,
    `document.body.innerText.includes("Apache-2.0") || document.body.innerText.includes("apache-2.0")`
  );
  await screenshot(browser, "hugging-face.png");

  await navigate(browser, "/projects/dataset?projectId=4", "13,083");
  await screenshot(browser, "text-dataset.png");

  await evaluate(
    browser,
    `(() => { const tab=[...document.querySelectorAll('[role="tab"]')].find((item)=>item.textContent.trim()==='Configure'); tab?.dispatchEvent(new MouseEvent('mousedown',{bubbles:true,button:0})); return Boolean(tab); })()`
  );
  await waitFor(browser, `document.body.innerText.includes("What should this project do?")`);
  await evaluate(
    browser,
    `(() => { const trigger=document.querySelector('[role="combobox"]'); trigger?.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,button:0,pointerType:'mouse'})); return Boolean(trigger); })()`
  );
  await waitFor(browser, `document.body.innerText.includes("Response evaluation")`);
  await screenshot(browser, "text-workflows.png");

  await navigate(browser, "/projects/models?projectId=4", "Structured model report");
  await sleep(1_500);
  await screenshot(browser, "text-model.png");
} finally {
  await browser.close();
}

console.log(`Captured structured workflow screenshots in ${output}`);

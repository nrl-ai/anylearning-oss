/**
 * Record the current auto-labelling workflow from a running AnyLearning app.
 *
 * The script talks directly to Chrome's debugging endpoint, so it adds no
 * browser dependency to the docs project. It is intentionally deterministic:
 * each clip starts from the same project and image and uses visible pointer
 * cues before every action.
 *
 * Usage:
 *   node scripts/record-auto-labeling.mjs --project 3
 *   node scripts/record-auto-labeling.mjs --project 3 --probe
 */

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";

const args = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  args.set(process.argv[index], process.argv[index + 1] ?? true);
}

const app = String(args.get("--app") || "http://127.0.0.1:5688");
const cdp = String(args.get("--cdp") || "http://127.0.0.1:9223");
const projectId = Number(args.get("--project"));
const probeOnly = args.has("--probe");
const output = path.resolve("public/auto_labeling");
const scratch = path.join(os.tmpdir(), "anylearning-auto-labeling-recording");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

if (!Number.isInteger(projectId) || projectId < 1) {
  throw new Error("Pass the numeric demo project id with --project.");
}

async function connect() {
  const targets = await fetch(`${cdp}/json`).then((response) => response.json());
  const target = targets.find((candidate) => candidate.type === "page");
  if (!target) throw new Error(`No page target is available at ${cdp}.`);

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
  });

  let nextId = 0;
  const pending = new Map();
  ws.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (!message.id || !pending.has(message.id)) return;
    const handlers = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) handlers.reject(new Error(JSON.stringify(message.error)));
    else handlers.resolve(message.result);
  };

  const call = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++nextId;
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });

  await call("Page.enable");
  await call("Runtime.enable");
  await call("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });

  return { call, close: () => ws.close() };
}

async function evaluate(browser, expression) {
  const result = await browser.call("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    userGesture: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(`${result.exceptionDetails.text}: ${expression.slice(0, 120)}`);
  }
  return result.result?.value;
}

async function waitFor(browser, expression, timeout = 20_000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    if (await evaluate(browser, expression)) return;
    await sleep(200);
  }
  throw new Error(`Timed out waiting for: ${expression}`);
}

async function screenshot(browser, file) {
  const { data } = await browser.call("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: false,
  });
  fs.writeFileSync(file, Buffer.from(data, "base64"));
}

async function frame(browser, file) {
  const { data } = await browser.call("Page.captureScreenshot", {
    format: "jpeg",
    quality: 82,
    captureBeyondViewport: false,
  });
  fs.writeFileSync(file, Buffer.from(data, "base64"));
}

const buttonWithText = (text) =>
  `(() => [...document.querySelectorAll('button')].find((element) => element.textContent.trim().includes(${JSON.stringify(text)})))()`;

async function openLabeller(browser) {
  await browser.call("Page.navigate", {
    url: `${app}/projects/dataset.html?projectId=${projectId}`,
  });
  await waitFor(browser, `${buttonWithText("Start labelling")} instanceof HTMLElement`);
  await evaluate(browser, `${buttonWithText("Start labelling")}.click(); true`);
  await sleep(2500);
  if (probeOnly) await screenshot(browser, path.join(scratch, "after-start.png"));
  await waitFor(
    browser,
    `document.querySelector('[aria-label="Turn on auto-labelling"],[aria-label="Turn off auto-labelling"]') instanceof HTMLElement`,
  );
  await sleep(1800);
  // This is development-only instrumentation and is never present in a
  // packaged release. Do not let the screenshot app leak it into the guide.
  await evaluate(
    browser,
    `(() => { const p=[...document.querySelectorAll('p')].find((e)=>/DEV ONLY/i.test(e.textContent)); if(p) p.closest('div').parentElement.style.display='none'; return true; })()`,
  );
}

async function installPointer(browser) {
  await evaluate(
    browser,
    `(() => {
      let pointer=document.getElementById('docs-demo-pointer');
      if(!pointer){
        pointer=document.createElement('div');
        pointer.id='docs-demo-pointer';
        pointer.innerHTML='<span></span>';
        Object.assign(pointer.style,{position:'fixed',left:'72px',top:'72px',width:'22px',height:'22px',border:'3px solid white',borderRadius:'999px',background:'#06b6d4',boxShadow:'0 2px 10px #000b',transform:'translate(-50%,-50%)',transition:'left 420ms cubic-bezier(.2,.8,.2,1), top 420ms cubic-bezier(.2,.8,.2,1)',pointerEvents:'none',zIndex:'2147483647'});
        document.body.append(pointer);
        const style=document.createElement('style');
        style.textContent='@keyframes docs-demo-ripple{from{opacity:.8;transform:scale(.4)}to{opacity:0;transform:scale(2.6)}} #docs-demo-pointer span{position:absolute;inset:-7px;border:2px solid #67e8f9;border-radius:999px;opacity:0} #docs-demo-pointer.click span{animation:docs-demo-ripple .55s ease-out}';
        document.head.append(style);
      }
      return true;
    })()`,
  );
}

async function movePointer(browser, x, y, wait = 600) {
  await evaluate(
    browser,
    `(() => { const p=document.getElementById('docs-demo-pointer'); p.style.left='${x}px'; p.style.top='${y}px'; return true; })()`,
  );
  await sleep(wait);
  await browser.call("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
}

async function elementCenter(browser, expression) {
  const center = await evaluate(
    browser,
    `(() => { const element=${expression}; if(!element) return null; const r=element.getBoundingClientRect(); return {x:r.left+r.width/2,y:r.top+r.height/2}; })()`,
  );
  if (!center) throw new Error(`Could not find element: ${expression}`);
  return center;
}

async function clickAt(browser, x, y) {
  await movePointer(browser, x, y);
  await evaluate(
    browser,
    `(() => { const p=document.getElementById('docs-demo-pointer'); p.classList.remove('click'); void p.offsetWidth; p.classList.add('click'); return true; })()`,
  );
  await browser.call("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x,
    y,
    button: "left",
    buttons: 1,
    clickCount: 1,
  });
  await sleep(110);
  await browser.call("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x,
    y,
    button: "left",
    buttons: 0,
    clickCount: 1,
  });
  await sleep(500);
}

async function clickElement(browser, expression) {
  const { x, y } = await elementCenter(browser, expression);
  await clickAt(browser, x, y);
}

async function drag(browser, from, to) {
  await movePointer(browser, from.x, from.y);
  await browser.call("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x: from.x,
    y: from.y,
    button: "left",
    buttons: 1,
    clickCount: 1,
  });
  for (let step = 1; step <= 12; step += 1) {
    const x = from.x + ((to.x - from.x) * step) / 12;
    const y = from.y + ((to.y - from.y) * step) / 12;
    await evaluate(
      browser,
      `(() => { const p=document.getElementById('docs-demo-pointer'); p.style.left='${x}px'; p.style.top='${y}px'; return true; })()`,
    );
    await browser.call("Input.dispatchMouseEvent", { type: "mouseMoved", x, y, button: "left", buttons: 1 });
    await sleep(55);
  }
  await browser.call("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x: to.x,
    y: to.y,
    button: "left",
    buttons: 0,
    clickCount: 1,
  });
  await sleep(500);
}

async function imagePoint(browser, imageX, imageY) {
  return evaluate(
    browser,
    `(() => { const svg=[...document.querySelectorAll('svg')].find((element)=>element.getBoundingClientRect().width>500); if(!svg) return null; const r=svg.getBoundingClientRect(); return {x:r.x+r.width*${imageX}/640,y:r.y+r.height*${imageY}/640}; })()`,
  );
}

async function record(browser, name, action, posterProgress = 0.72) {
  const directory = path.join(scratch, name);
  fs.mkdirSync(directory, { recursive: true });
  let running = true;
  let captureError;
  const capture = (async () => {
    let index = 0;
    while (running) {
      const started = Date.now();
      try {
        await frame(browser, path.join(directory, `${String(index).padStart(5, "0")}.jpg`));
      } catch (error) {
        captureError = error;
        break;
      }
      index += 1;
      await sleep(Math.max(0, 125 - (Date.now() - started)));
    }
    return index;
  })();

  await sleep(900);
  await action();
  await sleep(900);
  running = false;
  const count = await capture;
  if (captureError) throw captureError;
  if (count < 8) throw new Error(`${name} produced too few frames (${count}).`);

  const destination = path.join(output, `${name}.mp4`);
  execFileSync(
    "ffmpeg",
    [
      "-y",
      "-loglevel",
      "error",
      "-framerate",
      "8",
      "-i",
      path.join(directory, "%05d.jpg"),
      "-c:v",
      "libx264",
      "-preset",
      "slow",
      "-crf",
      "24",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      destination,
    ],
    { stdio: "inherit" },
  );

  const poster = path.join(output, `${name}.poster.jpg`);
  fs.copyFileSync(
    path.join(directory, `${String(Math.min(count - 1, Math.floor(count * posterProgress))).padStart(5, "0")}.jpg`),
    poster,
  );
  process.stdout.write(`Recorded ${name}: ${count} frames\n`);
}

async function setAutoLabellingOff(browser) {
  const turnOff = `document.querySelector('[aria-label="Turn off auto-labelling"]')`;
  if (await evaluate(browser, `${turnOff} instanceof HTMLElement`)) {
    await clickElement(browser, turnOff);
    await waitFor(browser, `!${buttonWithText("Include point")}`);
  }
}

async function removeSavedAnnotation() {
  const response = await fetch(`${app}/api/projects/${projectId}/data_items?limit=1`);
  if (!response.ok) throw new Error(`Cannot read demo data item: ${response.status}`);
  const item = (await response.json()).data_items?.[0];
  if (!item) throw new Error("The auto-labelling demo project has no image.");
  const cleared = await fetch(`${app}/api/projects/${projectId}/data_items/${item.id}/set_annotation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "[]",
  });
  if (!cleared.ok) throw new Error(`Cannot reset demo annotation: ${cleared.status}`);
}

function makeThumbnail() {
  const source = path.join(scratch, "thumbnail-source.png");
  const target = path.join(output, "auto-labeling-thumbnail.jpg");
  execFileSync(
    "ffmpeg",
    [
      "-y",
      "-loglevel",
      "error",
      "-i",
      source,
      "-vf",
      "scale=1200:750,crop=1200:630:0:60,drawbox=x=0:y=410:w=1200:h=220:color=0x071116@0.88:t=fill,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Auto-label images faster':fontcolor=white:fontsize=58:x=56:y=455,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='SAM 2 in AnyLearning':fontcolor=0x67e8f9:fontsize=29:x=60:y=540",
      "-frames:v",
      "1",
      "-q:v",
      "2",
      target,
    ],
    { stdio: "inherit" },
  );
}

fs.mkdirSync(output, { recursive: true });
fs.rmSync(scratch, { recursive: true, force: true });
fs.mkdirSync(scratch, { recursive: true });

const browser = await connect();
try {
  if (!probeOnly) await removeSavedAnnotation();
  await openLabeller(browser);
  if (probeOnly) {
    const file = path.join(scratch, "probe.png");
    await screenshot(browser, file);
    const controls = await evaluate(
      browser,
      `[...document.querySelectorAll('button,[role="option"],[role="dialog"]')].map((element) => ({tag: element.tagName, role: element.getAttribute('role'), aria: element.getAttribute('aria-label'), text: element.textContent.trim().replace(/\\s+/g, ' ').slice(0, 100)}))`,
    );
    process.stdout.write(`${JSON.stringify(controls, null, 2)}\n`);
    const drawingSurfaces = await evaluate(
      browser,
      `[...document.querySelectorAll('canvas,svg,img')].map((element) => { const r=element.getBoundingClientRect(); return {tag:element.tagName, width:r.width, height:r.height, x:r.x, y:r.y, src:element.getAttribute('src')?.slice(0,100)} }).filter((item)=>item.width>200 && item.height>200)`,
    );
    process.stdout.write(`${JSON.stringify(drawingSurfaces, null, 2)}\n`);
    process.stdout.write(`${file}\n`);
  } else {
    await installPointer(browser);
    await setAutoLabellingOff(browser);

    await record(browser, "01_enable_and_choose_model", async () => {
      const aiButton = `document.querySelector('[aria-label="Turn on auto-labelling"]')`;
      const ai = await elementCenter(browser, aiButton);
      await movePointer(browser, ai.x, ai.y, 1200);
      await clickAt(browser, ai.x, ai.y);
      await waitFor(browser, `${buttonWithText("Include point")} instanceof HTMLElement`);
      await sleep(1700);

      const modelSelect = `[...document.querySelectorAll('button[role="combobox"]')].find((button)=>button.textContent.includes('Segment Anything'))`;
      await clickElement(browser, modelSelect);
      await waitFor(browser, `document.querySelector('[role="option"]') instanceof HTMLElement`);
      await sleep(1200);
      const sam2Small = `[...document.querySelectorAll('[role="option"]')].find((option)=>option.textContent.includes('Hiera-Small'))`;
      await clickElement(browser, sam2Small);
      await sleep(1400);
    });

    await record(browser, "02_prompt_and_refine", async () => {
      await clickElement(browser, buttonWithText("Include box"));
      const boxStart = await imagePoint(browser, 320, 145);
      // A deliberately loose lower edge shows a realistic correction: the
      // first mask catches a little of the worker's temple, then one negative
      // point removes it while retaining the helmet rim.
      const boxEnd = await imagePoint(browser, 565, 390);
      await drag(browser, boxStart, boxEnd);
      await sleep(3200);

      await clickElement(browser, buttonWithText("Exclude point"));
      const exclude = await imagePoint(browser, 470, 355);
      await clickAt(browser, exclude.x, exclude.y);
      await sleep(3200);
    }, 0.94);

    await screenshot(browser, path.join(scratch, "thumbnail-source.png"));

    await record(browser, "03_assign_and_save", async () => {
      await clickElement(browser, buttonWithText("Finish object"));
      await waitFor(browser, `document.querySelector('[role="dialog"][aria-label="Choose a class"]') instanceof HTMLElement`);
      await sleep(1500);
      const safetyHelmet = `[...document.querySelectorAll('[role="dialog"] button')].find((button)=>button.textContent.trim()==='safety_helmet')`;
      await clickElement(browser, safetyHelmet);
      await waitFor(browser, `!document.querySelector('[role="dialog"][aria-label="Choose a class"]')`);
      await sleep(1200);
      await clickElement(browser, `document.querySelector('[aria-label="Save annotations"]')`);
      await sleep(1700);
    });

    makeThumbnail();
  }
} finally {
  browser.close();
}

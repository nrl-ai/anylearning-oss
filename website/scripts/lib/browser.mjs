import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

/**
 * A very small Chrome DevTools Protocol driver.
 *
 * Deliberately dependency-free: adding Playwright to a docs site just to take
 * screenshots would pull ~300 MB of browsers into a repo whose whole point is
 * being light. Node 22+ has a built-in WebSocket, which is all CDP needs.
 */

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  `${os.homedir()}/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome`,
].filter(Boolean);

function findChrome() {
  for (const candidate of CHROME_CANDIDATES) {
    try {
      if (candidate.includes("*")) continue;
      fs.accessSync(candidate, fs.constants.X_OK);
      return candidate;
    } catch {
      /* try the next one */
    }
  }
  const globbed = fs
    .readdirSync(`${os.homedir()}/.cache/ms-playwright`, { withFileTypes: true })
    .filter((e) => e.isDirectory() && e.name.startsWith("chromium-"))
    .map((e) => `${os.homedir()}/.cache/ms-playwright/${e.name}/chrome-linux64/chrome`)
    .find((p) => fs.existsSync(p));
  if (globbed) return globbed;
  throw new Error("No Chrome found. Set CHROME_PATH to a Chrome or Chromium binary.");
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function launch({ width = 1600, height = 1000 } = {}) {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "anylearning-shots-"));
  const port = 9500 + (process.pid % 400);
  const chrome = spawn(findChrome(), [
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--hide-scrollbars",
    "--force-device-scale-factor=2", // retina-density source images
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    `--window-size=${width},${height}`,
    "about:blank",
  ]);
  chrome.stderr.on("data", () => {});

  let wsUrl;
  for (let i = 0; i < 80 && !wsUrl; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      wsUrl = (await res.json()).webSocketDebuggerUrl;
    } catch {
      await sleep(200);
    }
  }
  if (!wsUrl) throw new Error("Chrome did not expose a debugging endpoint");

  const ws = new WebSocket(wsUrl);
  await new Promise((resolve) => (ws.onopen = resolve));

  let id = 0;
  const pending = new Map();
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
    }
  };
  const send = (method, params = {}, sessionId) =>
    new Promise((resolve, reject) => {
      const mid = ++id;
      pending.set(mid, { resolve, reject });
      ws.send(JSON.stringify({ id: mid, method, params, sessionId }));
    });

  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  const call = (method, params) => send(method, params, sessionId);

  await call("Page.enable");
  await call("Runtime.enable");

  return {
    async setViewport(w, h) {
      await call("Emulation.setDeviceMetricsOverride", {
        width: w,
        height: h,
        deviceScaleFactor: 2,
        mobile: false,
      });
    },
    async goto(url, waitMs = 9000) {
      await call("Page.navigate", { url });
      await sleep(waitMs);
    },
    async evaluate(expression) {
      const result = await call("Runtime.evaluate", {
        expression,
        awaitPromise: true,
        userGesture: true,
        returnByValue: true,
      });
      if (result.exceptionDetails) {
        throw new Error(result.exceptionDetails.text + " :: " + expression.slice(0, 80));
      }
      return result.result?.value;
    },
    async screenshot() {
      const { data } = await call("Page.captureScreenshot", { format: "png" });
      return Buffer.from(data, "base64");
    },
    sleep,
    async close() {
      ws.close();
      chrome.kill();
      // Chrome is still flushing its profile as it exits, so a plain rmSync
      // races it and throws ENOTEMPTY after every successful run.
      await sleep(400);
      try {
        fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
      } catch {
        /* a temp dir left behind is not worth failing a capture over */
      }
    },
  };
}

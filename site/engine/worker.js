// Runs the pipeline off the main thread so the page stays responsive.
// in:  {type: "run", channels: Float32Array[], sr, name, sha256, pads, separator}
// out: {type: "progress", stage, status, value} … then {type: "done", kit, files} or {type: "error", message}

import { makeKit, withPadSplit } from "./pipeline.js";

let configPromise = null;
const loadConfig = () => (configPromise ??= fetch(new URL("./config.json", import.meta.url)).then((r) => r.json()));

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type !== "run") return;
  try {
    const cfg = withPadSplit(await loadConfig(), msg.pads);
    const { kit, files, timings } = await makeKit({
      channels: msg.channels, sr: msg.sr, name: msg.name, sha256: msg.sha256, cfg,
      separator: msg.separator || "quick",
      onProgress: (stage, status, value) => self.postMessage({ type: "progress", stage, status, value }),
    });
    self.postMessage({ type: "done", kit, files, timings }, Object.values(files));
  } catch (err) {
    self.postMessage({ type: "error", message: err?.message || String(err) });
  }
};

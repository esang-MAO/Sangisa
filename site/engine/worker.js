// Runs the pipeline off the main thread so the page stays responsive.
// in:  {type: "run", mode: "kit"|"split", separator: "quick"|"ai", modelBase, channels, sr, name, sha256, pads, includeStems}
// out: {type: "progress", stage, status, value, note} … then {type: "done", kit|split, files} or {type: "error", message}

import { Demucs } from "./demucs.js";
import { loadModelFiles, modelInfo } from "./models.js";
import { makeKit, makeSplit, withPadSplit } from "./pipeline.js";

let configPromise = null;
const loadConfig = () => (configPromise ??= fetch(new URL("./config.json", import.meta.url)).then((r) => r.json()));
const post = (msg, transfer) => self.postMessage(msg, transfer);

/** HT-Demucs on this device: WebGPU when the browser has it, otherwise WebAssembly on the CPU. */
async function aiSeparator(modelBase) {
  const note = (text) => post({ type: "progress", stage: "separate", status: "note", note: text });
  const meta = await modelInfo(modelBase);
  if (!meta) throw new Error("The AI model isn't available on this site yet.");
  return async (stereo, sr, progress) => {
    if (sr !== meta.samplerate) throw new Error(`The AI model needs ${meta.samplerate} Hz audio.`);
    note("Getting the AI model…");
    const files = await loadModelFiles(modelBase, meta, (f, info) => {
      progress(0.15 * f);
      if (!info.cached) note(`Downloading the AI model (${Math.round(info.mb || meta.weights_mb)} MB, only the first time)…`);
    });
    const ort = await import(new URL("../vendor/ort/ort.webgpu.min.mjs", import.meta.url).href);
    ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    let model = null, device = "CPU";
    if (self.navigator?.gpu) {
      try {
        model = await Demucs.create(ort, files.graph, meta, files.weights, ["webgpu"]);
        device = "GPU";
      } catch (e) {
        console.warn("WebGPU didn't start; using WebAssembly", e);
        note("This browser didn't give access to the GPU, so the AI runs on the CPU (slower).");
      }
    }
    model ??= await Demucs.create(ort, files.graph, meta, files.weights, ["wasm"]);
    post({ type: "device", device });
    note(device === "GPU" ? "Separating with the AI model on the GPU…" : "Separating with the AI model on the CPU (slow: keep the screen on)…");
    const stems = await model.separate(stereo, (f) => progress(0.15 + 0.85 * f));
    await model.session.release?.();
    return stems;
  };
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type !== "run") return;
  try {
    const cfg = withPadSplit(await loadConfig(), msg.mode === "split" ? null : msg.pads);
    const onProgress = (stage, status, value) => post({ type: "progress", stage, status, value });
    const separate = msg.separator === "ai" ? await aiSeparator(msg.modelBase) : null;
    if (msg.mode === "split") {
      const { split, files } = await makeSplit({
        channels: msg.channels, sr: msg.sr, name: msg.name, cfg, separate, includeStems: Boolean(msg.includeStems), onProgress,
      });
      post({ type: "done", split, files }, Object.values(files));
      return;
    }
    const { kit, files, timings } = await makeKit({
      channels: msg.channels, sr: msg.sr, name: msg.name, sha256: msg.sha256, cfg, onProgress,
      separator: msg.separator === "ai" ? "quick" : msg.separator || "quick",
      separate, separatorLabel: msg.separator === "ai" ? "htdemucs (on this device)" : null,
    });
    post({ type: "done", kit, files, timings }, Object.values(files));
  } catch (err) {
    post({ type: "error", message: err?.message || String(err) });
  }
};

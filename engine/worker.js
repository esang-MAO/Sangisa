// Runs the pipeline off the main thread so the page stays responsive.
// in:  {type: "run", mode: "kit"|"split", separator: "quick"|"ai", modelBase, channels, sr, name, sha256, pads,
//        includeStems, stems?}
//       or {type: "separate", channels, sr, modelBase, keep} (AI stems only)
// out: {type: "progress", stage, status, value, note}, {type: "device"}, {type: "step"} … then
//      {type: "done", kit|split, files} / {type: "stems", stems, channels}, or {type: "error", message}

import { Demucs } from "./demucs.js";
import { loadModelFiles, modelInfo } from "./models.js";
import { makeKit, makeSplit, withPadSplit } from "./pipeline.js";

let configPromise = null;
const loadConfig = () => (configPromise ??= fetch(new URL("./config.json", import.meta.url)).then((r) => r.json()));
const post = (msg, transfer) => self.postMessage(msg, transfer);

/** HT-Demucs on this device: WebGPU when the browser has it, otherwise WebAssembly on the CPU. */
async function aiSeparator(modelBase, keep) {
  const note = (text) => post({ type: "progress", stage: "separate", status: "note", note: text });
  // Where it got to, so that if the browser stops the page (out of memory) the app can say where.
  const step = (text) => post({ type: "step", step: text });
  const meta = await modelInfo(modelBase);
  if (!meta) throw new Error("The AI model isn't available on this site yet.");
  return async (stereo, sr, progress) => {
    if (sr !== meta.samplerate) throw new Error(`The AI model needs ${meta.samplerate} Hz audio.`);
    note("Getting the AI model…");
    step("loading the AI model");
    let files = await loadModelFiles(modelBase, meta, (f, info) => {
      progress(0.15 * f);
      if (!info.cached) note(`Downloading the AI model (${Math.round(info.mb || meta.weights_mb)} MB, only the first time)…`);
    });
    const ort = await import(new URL("../vendor/ort/ort.webgpu.min.mjs", import.meta.url).href);
    ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    let model = null, device = "CPU";
    if (self.navigator?.gpu) {
      try {
        step("starting the AI model on the GPU");
        model = await Demucs.create(ort, files.graph, meta, files.weights, ["webgpu"]);
        device = "GPU";
      } catch (e) {
        console.warn("WebGPU didn't start; using WebAssembly", e);
        note("This browser didn't give access to the GPU, so the AI runs on the CPU (slower).");
      }
    }
    if (!model) {
      step("starting the AI model on the CPU");
      model = await Demucs.create(ort, files.graph, meta, files.weights, ["wasm"]);
    }
    files = null; // the runtime has its own copy now
    post({ type: "device", device });
    note(device === "GPU" ? "Separating with the AI model on the GPU…" : "Separating with the AI model on the CPU (slow: keep the screen on)…");
    step(`separating on the ${device}, part 1`);
    const stems = await model.separate(stereo, (f, done, total) => {
      progress(0.15 + 0.85 * f);
      step(done < total ? `separating on the ${device}, part ${done + 1} of ${total}` : "finishing the stems");
    }, keep);
    await model.session.release?.();
    return stems;
  };
}

const buffersOf = (arrays) => [...new Set(arrays.map((x) => x.buffer))];

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "separate") {
      // AI separation on its own: the page closes this worker afterwards, which is the only way
      // to give back the WebAssembly memory the model used before the kit is made.
      const stereo = msg.channels.length === 1 ? [msg.channels[0], msg.channels[0]] : msg.channels.slice(0, 2);
      const separate = await aiSeparator(msg.modelBase, msg.keep || undefined);
      const stems = await separate(stereo, msg.sr, (v) => post({ type: "progress", stage: "separate", status: "progress", value: v }));
      const arrays = [...msg.channels, ...Object.values(stems).flat()];
      post({ type: "stems", stems, channels: msg.channels }, buffersOf(arrays));
      return;
    }
    if (msg.type !== "run") return;
    const cfg = withPadSplit(await loadConfig(), msg.mode === "split" ? null : msg.pads);
    const onProgress = (stage, status, value) => post({ type: "progress", stage, status, value });
    // An acapella needs only the vocals (the instrumental is the song minus them): less memory.
    const keep = msg.mode === "split" && !msg.includeStems ? ["vocals"] : undefined;
    const separate = msg.separator === "ai" && !msg.stems ? await aiSeparator(msg.modelBase, keep) : null;
    const label = msg.separator === "ai" ? "htdemucs (on this device)" : null;
    if (msg.mode === "split") {
      const { split, files } = await makeSplit({
        channels: msg.channels, sr: msg.sr, name: msg.name, cfg, separate, stems: msg.stems,
        includeStems: Boolean(msg.includeStems), onProgress,
      });
      post({ type: "done", split, files }, Object.values(files));
      return;
    }
    const { kit, files, timings } = await makeKit({
      channels: msg.channels, sr: msg.sr, name: msg.name, sha256: msg.sha256, cfg, onProgress,
      separator: msg.separator === "ai" ? "quick" : msg.separator || "quick",
      separate, stems: msg.stems, separatorLabel: label,
    });
    post({ type: "done", kit, files, timings }, Object.values(files));
  } catch (err) {
    post({ type: "error", message: err?.message || String(err) });
  }
};

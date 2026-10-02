// The whole pipeline in the browser: stereo audio in, kit.json + slice WAVs out.
// Stage names and labels match the Python pipeline, so the app shows the same progress.

import { analyze } from "./analyze.js";
import { dropCache, toMono } from "./dsp.js";
import { pick } from "./pick.js";
import { encodeWav, renderSlice } from "./render.js";
import { SEPARATORS } from "./separate.js";

export const STAGES = ["ingest", "separate", "analyze", "pick", "render"];
export const LABELS = {
  ingest: "Loading",
  separate: "Separating stems",
  analyze: "Analyzing tempo and key",
  pick: "Finding the best moments",
  render: "Building your kit",
};

export function withPadSplit(cfg, split) {
  if (!split) return cfg;
  const total = Object.values(split).reduce((a, b) => a + b, 0);
  if (![16, 32, 64].includes(total)) throw new Error(`The pad split adds up to ${total}; it must be 16, 32 or 64.`);
  return { ...cfg, kit: { ...cfg.kit, pad_split: { ...split }, pad_count: total } };
}

/**
 * @param {object} o
 * @param {Float32Array[]} o.channels stereo (or mono) audio at o.sr (44.1 kHz expected)
 * @param {string} o.name original file name
 * @param {string} o.sha256
 * @param {object} o.cfg config (site/engine/config.json, optionally with a pad split)
 * @param {string} [o.separator]
 * @param {Object<string, Float32Array[]>} [o.stems] already-separated stems (skips separation)
 * @param {(stage: string, status: "start"|"done"|"progress", value?: number) => void} [o.onProgress]
 */
export async function makeKit({
  channels, sr, name, sha256 = "", cfg, separator = "quick", stems: givenStems = null, onProgress = () => {},
}) {
  const tick = () => new Promise((r) => setTimeout(r, 0));
  const timings = {};
  const run = async (stage, fn) => {
    onProgress(stage, "start");
    const t0 = performance.now();
    await tick();
    const out = await fn((v) => onProgress(stage, "progress", v));
    timings[stage] = Math.round(performance.now() - t0) / 1000;
    onProgress(stage, "done", timings[stage]);
    return out;
  };

  const stereo = channels.length === 1 ? [channels[0], channels[0]] : channels.slice(0, 2);
  const duration = stereo[0].length / sr;
  if (duration > cfg.ingest.max_duration_s) {
    throw new Error(`That song is ${(duration / 60).toFixed(1)} minutes long; the limit is ${cfg.ingest.max_duration_s / 60} minutes.`);
  }
  await run("ingest", async () => {});

  const sep = SEPARATORS[separator];
  if (!sep && !givenStems) throw new Error(`Unknown separator ${separator}`);
  // `stems` lets callers (tests, or stems split elsewhere) skip separation.
  const stems = await run("separate", (p) => givenStems ?? sep.run(stereo, sr, p));
  const analysis = await run("analyze", async () => {
    const mono = toMono(stereo);
    const result = analyze(mono, sr, cfg);
    dropCache(mono);
    return result;
  });
  let monoStems = Object.fromEntries(Object.entries(stems).map(([k, v]) => [k, toMono(v)]));

  const source = { path: name, sha256, duration_s: Math.round(duration * 1000) / 1000, sample_rate: sr, rights_confirmed: true };
  const separation = {
    backend: "browser", model: separator, device: "this device",
    stems: Object.fromEntries(Object.keys(stems).map((k) => [k, `stems/${k}.wav`])),
  };
  const { kit, candidates } = await run("pick", async (p) =>
    pick({ stems: monoStems, analysis, sr, cfg, source, separation, onProgress: p }));

  monoStems = null; // rendering works from the stereo stems
  const files = await run("render", async (p) => {
    const out = {};
    kit.slices.forEach((s, i) => {
      out[s.file] = encodeWav(renderSlice(stems[s.stem], sr, s, cfg), sr);
      if (i % 10 === 0) p(i / kit.slices.length);
    });
    return out;
  });
  return { kit, files, analysis, candidates, timings };
}

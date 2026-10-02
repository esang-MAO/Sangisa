// Stage 4 in the browser: find, score and de-duplicate candidates, assemble the kit.
// Mirrors backend/sangisa/worker/stages/pick.py.

import { chroma, dropCache } from "./dsp.js";
import { PITCH_NAMES, sectionAt } from "./analyze.js";
import { extractorFor } from "./extract.js";
import { scoreAll, stemMfcc } from "./score.js";

const CATEGORY_ORDER = {
  drums: ["kick", "snare", "hat", "perc", "drum_loop"],
  bass: ["bass_loop", "bass_note"],
  vocals: ["vox_chop", "vox_phrase"],
  other: ["other_loop", "stab"],
};

export function findCandidates(stems, ctx, mfccs = {}) {
  const out = [];
  for (const name of Object.keys(stems).sort()) {
    const found = extractorFor(name)(stems[name], ctx);
    // Scoring needs the stem's MFCCs; take them now, while its spectrogram is cached, then let it go.
    mfccs[name] = stemMfcc(stems[name], ctx.sr);
    dropCache(stems[name]);
    found.sort((a, b) => a.start - b.start || a.end - b.end || (a.category < b.category ? -1 : 1));
    found.forEach((c, i) => { c.id = `${name}-${String(i).padStart(4, "0")}`; });
    out.push(...found);
  }
  return out;
}

export function choosePads(cands, cfg) {
  const split = cfg.kit.pad_split;
  const usable = cands.filter((c) => c.score > 0);
  const byScore = [...usable].sort((a, b) => b.score - a.score);
  const used = new Set();
  const pads = [];
  const floor = cfg.kit.min_relative_score;
  for (const [stem, count] of Object.entries(split)) {
    let reps = byScore.filter((c) => c.stem === stem && c.representative);
    if (reps.length) reps = reps.filter((c) => c.score >= floor * reps[0].score);
    const order = CATEGORY_ORDER[stem] || [...new Set(reps.map((c) => c.category))].sort();
    const queues = Object.fromEntries(order.map((cat) => [cat, reps.filter((c) => c.category === cat)]));
    const picked = [];
    while (picked.length < count && order.some((cat) => queues[cat].length)) {
      for (const cat of order) if (queues[cat].length && picked.length < count) picked.push(queues[cat].shift());
    }
    for (const c of byScore) {
      if (picked.length >= count) break;
      if (c.stem === stem && !picked.some((p) => p.id === c.id)) picked.push(c);
    }
    picked.forEach((c) => used.add(c.id));
    pads.push(...picked, ...Array(count - picked.length).fill(null));
  }
  const leftovers = byScore.filter((c) => c.representative && !used.has(c.id));
  pads.forEach((slot, i) => {
    if (slot === null && leftovers.length) { pads[i] = leftovers.shift(); used.add(pads[i].id); }
  });
  const backups = [];
  for (const stem of [...new Set(usable.map((c) => c.stem))].sort()) {
    const pool = byScore.filter((c) => c.stem === stem && !used.has(c.id))
      .sort((a, b) => (a.representative === b.representative ? b.score - a.score : a.representative ? -1 : 1));
    backups.push(...pool.slice(0, cfg.kit.backups_per_stem));
  }
  return { pads, backups };
}

export function letters(n) {
  let out = "";
  while (n > 0) {
    const r = (n - 1) % 26;
    out = String.fromCharCode(65 + r) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

class Labeler {
  constructor(analysis) {
    this.counts = {};
    this.suffix = `_${Math.round(analysis.bpm)}bpm` + (analysis.key_short ? `_${analysis.key_short}` : "");
  }
  name(c) {
    const n = (this.counts[c.category] = (this.counts[c.category] || 0) + 1);
    const stemTitle = c.stem[0].toUpperCase() + c.stem.slice(1);
    const bars = c.bars ? `${c.bars}bar` : null;
    const note = c.note;
    const simple = { kick: "Kick", snare: "Snare", hat: "Hat", perc: "Perc" };
    let label, parts;
    if (simple[c.category]) {
      label = `${simple[c.category]} ${n}`; parts = [stemTitle, simple[c.category], String(n)];
    } else if (c.kind === "loop") {
      const letter = letters(n);
      const noun = { drums: "Drum", vocals: "Vox" }[c.stem] || stemTitle;
      label = `${noun} loop ${letter} (${c.bars} bar)`; parts = [stemTitle, "Loop", letter, bars];
    } else if (c.category === "bass_note") {
      label = `Bass note ${n}` + (note ? ` ${note}` : ""); parts = [stemTitle, "Note", String(n), note];
    } else if (c.category === "vox_chop") {
      label = `Vox chop ${n}` + (note ? ` ${note}` : ""); parts = [stemTitle, "Chop", String(n), note];
    } else if (c.category === "vox_phrase") {
      label = `Vox phrase ${n}`; parts = [stemTitle, "Phrase", String(n)];
    } else {
      const noun = c.category === "stab" ? "Stab" : "Hit";
      label = `${stemTitle} ${noun.toLowerCase()} ${n}` + (note ? ` ${note}` : ""); parts = [stemTitle, noun, String(n), note];
    }
    return [label, parts.filter(Boolean).join("_") + this.suffix];
  }
}

export function noteName(hz) {
  const midi = Math.round(69 + 12 * Math.log2(hz / 440));
  return `${PITCH_NAMES[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

/** YIN pitch on a decimated copy (cheap enough for phones). Returns Hz or null when unpitched. */
export function yinPitch(x, sr, fmin, fmax) {
  const D = Math.max(1, Math.floor(sr / (fmax * 8)));
  const n = Math.floor(x.length / D);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < D; k++) s += x[i * D + k]; y[i] = s / D; }
  const fs = sr / D;
  const tauMax = Math.min(Math.floor(fs / fmin), Math.floor(n / 2) - 1);
  const tauMin = Math.max(2, Math.floor(fs / fmax));
  const W = Math.max(tauMax, 256);
  if (tauMax <= tauMin || n < W + tauMax) return null;
  const f0s = [];
  let voiced = 0, frames = 0;
  for (let start = 0; start + W + tauMax <= n; start += Math.max(1, W >> 1)) {
    frames++;
    const d = new Float64Array(tauMax + 1);
    for (let tau = 1; tau <= tauMax; tau++) {
      let s = 0;
      for (let j = 0; j < W; j++) { const v = y[start + j] - y[start + j + tau]; s += v * v; }
      d[tau] = s;
    }
    let run = 0, found = -1;
    const cm = new Float64Array(tauMax + 1);
    cm[0] = 1;
    for (let tau = 1; tau <= tauMax; tau++) { run += d[tau]; cm[tau] = run > 0 ? (d[tau] * tau) / run : 1; }
    for (let tau = tauMin; tau <= tauMax; tau++) {
      if (cm[tau] < 0.15) {
        while (tau + 1 <= tauMax && cm[tau + 1] < cm[tau]) tau++;
        found = tau;
        break;
      }
    }
    if (found < 0) continue;
    voiced++;
    let tau = found;
    if (tau > 1 && tau < tauMax) {
      const a = cm[tau - 1], b = cm[tau], c = cm[tau + 1];
      const den = a - 2 * b + c;
      if (den) tau += (0.5 * (a - c)) / den;
    }
    f0s.push(fs / tau);
  }
  if (!frames || voiced / frames < 0.4) return null;
  f0s.sort((a, b) => a - b);
  return f0s[f0s.length >> 1];
}

export function chordName(x, sr) {
  const ch = chroma(x, sr, { nFft: x.length >= 4096 ? 4096 : 2048, hop: 512 });
  const m = new Float64Array(12);
  for (let t = 0; t < ch.nFrames; t++) for (let k = 0; k < 12; k++) m[k] += ch.data[t * 12 + k];
  const norm = Math.hypot(...m);
  if (norm < 1e-6) return null;
  let best = -1, name = null;
  for (let root = 0; root < 12; root++) {
    for (const [q, third] of [["maj", 4], ["min", 3]]) {
      const s = (m[root] + m[(root + third) % 12] + m[(root + 7) % 12]) / (norm * Math.sqrt(3));
      if (s > best) { best = s; name = `${PITCH_NAMES[root]}${q}`; }
    }
  }
  return name;
}

function describePitch(c, y, sr) {
  const x = y.subarray(c.start, c.end);
  if ((c.category === "bass_note" || c.category === "vox_chop") && x.length > sr * 0.05) {
    const [fmin, fmax] = c.stem === "bass" ? [30, 400] : [80, 1000];
    const hz = yinPitch(x, sr, fmin, fmax);
    if (hz) c.note = noteName(hz);
  } else if (c.category === "stab" && x.length > 2048) {
    c.note = chordName(x, sr);
  }
}

export function pick({ stems, analysis, sr, cfg, source, separation, onProgress }) {
  const ctx = {
    sr, bpm: analysis.bpm, beatS: 60 / analysis.bpm,
    beats: analysis.beats_s, downbeats: analysis.downbeats_s, cfg, xcfg: cfg.extract,
  };
  const mfccs = {};
  const cands = findCandidates(stems, ctx, mfccs);
  scoreAll(cands, stems, sr, cfg, onProgress, mfccs);
  const { pads, backups } = choosePads(cands, cfg);
  const kept = [...pads.filter(Boolean), ...backups];
  const labeler = new Labeler(analysis);
  const r = cfg.render;
  const r4 = (v) => Math.round(v * 1e4) / 1e4;
  const slices = kept.map((c) => {
    describePitch(c, stems[c.stem], sr);
    const [label, fileStem] = labeler.name(c);
    const loop = c.kind === "loop";
    return {
      id: c.id, stem: c.stem, kind: c.kind, category: c.category, label,
      file: `slices/${fileStem}.wav`,
      source_start_s: r4(c.start / sr), source_end_s: r4(c.end / sr),
      bars: c.bars, beats: c.beats, note: c.note,
      section: sectionAt(analysis.sections || [], c.start / sr),
      score: c.score, score_parts: c.parts, cluster: c.cluster,
      render: {
        fade_in_ms: loop ? r.loop_fade_ms : r.fade_in_ms,
        fade_out_ms: loop ? r.loop_fade_ms : r.fade_out_ms,
        normalize_dbfs: r.normalize ? r.normalize_dbfs : null,
        reverse: false,
        playback: loop ? "loop" : "one_shot",
      },
    };
  });
  const kit = {
    schema_version: 1,
    kit_name: `${source.path.replace(/\.[^.]+$/, "")} Kit`,
    source,
    analysis: Object.fromEntries(
      ["bpm", "key", "key_short", "key_confidence", "time_signature", "beats_s", "downbeats_s", "sections"]
        .map((k) => [k, analysis[k]]),
    ),
    separation,
    layout: { pad_count: cfg.kit.pad_count, pad_split: { ...cfg.kit.pad_split }, origin: "bottom_left" },
    pads: pads.map((c, i) => ({
      pad: i + 1, bank: String.fromCharCode(65 + Math.floor(i / 16)), midi_note: 36 + (i % 16), slice_id: c ? c.id : null,
    })),
    slices,
  };
  return { kit, candidates: cands.length };
}

// Per-stem candidate extractors. Mirrors backend/sangisa/worker/extractors/.

import { fft, nextPow2, rmsFrames } from "./dsp.js";
import { onsetStrength, HOP } from "./analyze.js";

/** A candidate slice; start/end are sample indexes into its stem. */
export function candidate(stem, kind, category, start, end, extra = {}) {
  return {
    stem, kind, category, start, end, bars: null, beats: null, note: null,
    id: "", parts: {}, score: 0, cluster: -1, representative: true, similarity: 0, ...extra,
  };
}

/** librosa.onset.onset_detect(backtrack=True) in samples. */
export function onsets(y, sr) {
  const env = onsetStrength(y, sr);
  const n = env.length;
  let lo = Infinity, hi = -Infinity;
  for (const v of env) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const x = Float64Array.from(env, (v) => (hi > lo ? (v - lo) / (hi - lo) : 0));
  const fr = sr / HOP;
  const preMax = Math.floor(0.03 * fr), postMax = Math.floor(0.0 * fr) + 1;
  const preAvg = Math.floor(0.1 * fr), postAvg = Math.floor(0.1 * fr) + 1;
  const wait = Math.floor(0.03 * fr), delta = 0.07;
  const peaks = [];
  let last = -Infinity;
  for (let i = 0; i < n; i++) {
    let mx = -Infinity;
    for (let j = Math.max(0, i - preMax); j < Math.min(n, i + postMax); j++) mx = Math.max(mx, x[j]);
    if (x[i] !== mx) continue;
    let s = 0, c = 0;
    for (let j = Math.max(0, i - preAvg); j < Math.min(n, i + postAvg); j++) { s += x[j]; c++; }
    if (x[i] < s / c + delta) continue;
    if (i <= last + wait) continue;
    peaks.push(i);
    last = i;
  }
  // backtrack each onset to the preceding local minimum of the onset envelope
  const minima = [0];
  for (let i = 1; i < n - 1; i++) if (env[i] <= env[i - 1] && env[i] < env[i + 1]) minima.push(i);
  const out = new Set();
  for (const p of peaks) {
    let m = 0;
    for (const v of minima) { if (v <= p) m = v; else break; }
    out.add(m * HOP);
  }
  return [...out].sort((a, b) => a - b);
}

export function trimTail(y, start, end, belowDb, hop = 256) {
  const len = end - start;
  if (len < hop * 2) return end;
  const nf = Math.floor(len / hop);
  const frames = new Float64Array(nf);
  let max = 0;
  for (let f = 0; f < nf; f++) {
    let m = 0;
    for (let i = start + f * hop; i < start + (f + 1) * hop; i++) m = Math.max(m, Math.abs(y[i]));
    frames[f] = m;
    max = Math.max(max, m);
  }
  const th = max * 10 ** (-belowDb / 20);
  let last = -1;
  for (let f = 0; f < nf; f++) if (frames[f] > th) last = f;
  if (last < 0) return end;
  return start + Math.min(len, (last + 2) * hop);
}

export function oneShots(y, ctx, stem, kind, category, minS = null) {
  const { sr, xcfg } = ctx;
  const n = y.length;
  const maxLen = Math.floor(xcfg.one_shot_max_s * sr);
  const minLen = Math.floor((minS ?? xcfg.one_shot_min_s) * sr);
  const starts = onsets(y, sr);
  const out = [];
  starts.forEach((s, i) => {
    const nxt = i + 1 < starts.length ? starts[i + 1] : n;
    const end = trimTail(y, s, Math.min(nxt, s + maxLen, n), xcfg.tail_trim_db);
    if (end - s >= minLen) out.push(candidate(stem, kind, category, s, end));
  });
  return out;
}

export function gridLoops(y, ctx, stem, category, barCounts) {
  const { sr } = ctx;
  const n = y.length;
  const barS = 4 * ctx.beatS;
  const out = [];
  for (const bars of barCounts) {
    const length = Math.round(bars * barS * sr);
    for (const t of ctx.downbeats) {
      const s = Math.round(t * sr);
      if (s + length <= n) out.push(candidate(stem, "loop", category, s, s + length, { bars, beats: bars * 4 }));
    }
  }
  return out;
}

export function cap(cands, y, limit) {
  if (cands.length <= limit) return cands;
  const energy = cands.map((c) => {
    let e = 0;
    for (let i = c.start; i < c.end; i++) e += y[i] * y[i];
    return e / (c.end - c.start);
  });
  const keep = energy.map((e, i) => [e, i]).sort((a, b) => b[0] - a[0]).slice(0, limit).map(([, i]) => i);
  return keep.sort((a, b) => a - b).map((i) => cands[i]);
}

/** librosa.effects.split: [start, end] sample ranges louder than -topDb relative to the peak. */
export function nonSilent(y, topDb, frameLength = 2048, hop = 512) {
  const r = rmsFrames(y, frameLength, hop, true);
  let ref = 0;
  for (const v of r) ref = Math.max(ref, v * v);
  const refDb = 10 * Math.log10(Math.max(ref, 1e-10));
  const loud = Array.from(r, (v) => 10 * Math.log10(Math.max(v * v, 1e-10)) - refDb > -topDb);
  const out = [];
  let i = 0;
  while (i < loud.length) {
    if (!loud[i]) { i++; continue; }
    let j = i;
    while (j < loud.length && loud[j]) j++;
    out.push([Math.min(i * hop, y.length), Math.min(j * hop, y.length)]);
    i = j;
  }
  return out;
}

// ---------------------------------------------------------------- per stem

export function bandFeatures(x, sr) {
  const n = nextPow2(Math.max(2048, x.length));
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const L = x.length;
  for (let i = 0; i < L; i++) re[i] = x[i] * (L > 1 ? 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (L - 1)) : 1);
  fft(re, im);
  let total = 0, low = 0, high = 0, cent = 0, logSum = 0, magSum = 0, cnt = 0;
  for (let k = 0; k <= n / 2; k++) {
    const f = (k * sr) / n;
    const p = re[k] * re[k] + im[k] * im[k];
    total += p;
    if (f < 150) low += p;
    if (f > 5000) high += p;
    cent += f * p;
    if (f > 200 && f < 10000) {
      const m = Math.sqrt(p) + 1e-12;
      logSum += Math.log(m); magSum += m; cnt++;
    }
  }
  total += 1e-12;
  return {
    low: low / total, high: high / total, centroid: cent / total,
    flatness: cnt ? Math.exp(logSum / cnt) / (magSum / cnt) : 0,
  };
}

export function classifyDrum(x, sr) {
  if (x.length < 64) return "perc";
  const f = bandFeatures(x, sr);
  if (f.low > 0.45 && f.centroid < 1000) return "kick";
  if (f.high > 0.5 || f.centroid > 6000) return "hat";
  if (f.flatness > 0.15 && f.centroid > 800 && f.centroid <= 6000) return "snare";
  return "perc";
}

function drums(y, ctx) {
  const limit = ctx.xcfg.max_candidates_per_kind;
  const hits = cap(oneShots(y, ctx, "drums", "one_shot", "perc"), y, limit);
  const a = Math.floor(ctx.sr * 0.08);
  for (const c of hits) c.category = classifyDrum(y.subarray(c.start, Math.min(c.end, c.start + a)), ctx.sr);
  const loops = cap(gridLoops(y, ctx, "drums", "drum_loop", ctx.xcfg.drum_loop_bars), y, limit);
  return [...hits, ...loops];
}

function bass(y, ctx) {
  const limit = ctx.xcfg.max_candidates_per_kind;
  return [
    ...cap(oneShots(y, ctx, "bass", "one_shot", "bass_note", ctx.xcfg.note_min_s), y, limit),
    ...cap(gridLoops(y, ctx, "bass", "bass_loop", ctx.xcfg.bass_loop_bars), y, limit),
  ];
}

function vocals(y, ctx) {
  const { sr, xcfg } = ctx;
  const limit = xcfg.max_candidates_per_kind;
  const minLen = Math.floor(xcfg.phrase_min_s * sr), maxLen = Math.floor(xcfg.phrase_max_s * sr);
  const active = nonSilent(y, xcfg.silence_top_db);
  const phrases = [];
  for (const [s, e] of active) {
    if (e - s < minLen) continue;
    const step = e - s > maxLen ? maxLen : e - s;
    for (let a = s; a < e; a += step) {
      const b = Math.min(a + step, e);
      if (b - a >= minLen) phrases.push(candidate("vocals", "phrase", "vox_phrase", a, b));
    }
  }
  const chops = [];
  for (const beats of xcfg.vocal_chop_beats) {
    const length = Math.round(beats * ctx.beatS * sr);
    for (const t of ctx.beats) {
      const s = Math.round(t * sr);
      if (s + length > y.length) continue;
      if (active.some(([a, b]) => a <= s && s < b)) chops.push(candidate("vocals", "chop", "vox_chop", s, s + length, { beats }));
    }
  }
  return [...cap(phrases, y, limit), ...cap(chops, y, limit)];
}

function other(y, ctx, stem = "other") {
  const limit = ctx.xcfg.max_candidates_per_kind;
  return [
    ...cap(oneShots(y, ctx, stem, "one_shot", "stab", ctx.xcfg.note_min_s), y, limit),
    ...cap(gridLoops(y, ctx, stem, `${stem}_loop`, ctx.xcfg.other_loop_bars), y, limit),
  ];
}

export function extractorFor(stem) {
  return { drums, bass, vocals, other }[stem] || ((y, ctx) => other(y, ctx, stem));
}

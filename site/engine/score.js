// Candidate scoring (0-1 per part, weighted) and de-duplication. Mirrors worker/scoring.py.

import { db, mfcc, rms } from "./dsp.js";

const EPS = 1e-10;
const ENV_HOP = 256;

/** Energy of any window of any stem: per-block sums plus the partial blocks at each end. */
class EnergyIndex {
  constructor(stems, block = 1024) {
    this.block = block;
    this.stems = stems;
    this.cums = {};
    for (const [name, y] of Object.entries(stems)) {
      const nb = Math.ceil(y.length / block);
      const c = new Float64Array(nb + 1);
      for (let b = 0; b < nb; b++) {
        let e = 0;
        for (let i = b * block, end = Math.min(y.length, (b + 1) * block); i < end; i++) e += y[i] * y[i];
        c[b + 1] = c[b] + e;
      }
      this.cums[name] = c;
    }
  }
  energy(stem, start, end) {
    const y = this.stems[stem], B = this.block, c = this.cums[stem];
    end = Math.min(end, y.length);
    if (end <= start) return 0;
    const b0 = Math.ceil(start / B), b1 = Math.floor(end / B);
    let e = 0;
    if (b1 <= b0) {
      for (let i = start; i < end; i++) e += y[i] * y[i];
      return e;
    }
    for (let i = start; i < b0 * B; i++) e += y[i] * y[i];
    e += c[b1] - c[b0];
    for (let i = b1 * B; i < end; i++) e += y[i] * y[i];
    return e;
  }
  isolation(stem, start, end) {
    let total = 0;
    for (const s of Object.keys(this.cums)) total += this.energy(s, start, end);
    return total > EPS ? this.energy(stem, start, end) / total : 0;
  }
}

/** Frame RMS without centering (librosa.feature.rms(center=False)). */
function envelope(x) {
  const fl = ENV_HOP * 2;
  if (x.length < fl) return [rms(x)];
  const n = 1 + Math.floor((x.length - fl) / ENV_HOP);
  const out = new Float64Array(n);
  for (let t = 0; t < n; t++) {
    let acc = 0;
    for (let i = t * ENV_HOP; i < t * ENV_HOP + fl; i++) acc += x[i] * x[i];
    out[t] = Math.sqrt(acc / fl);
  }
  return out;
}

const maxOf = (a) => { let m = -Infinity; for (const v of a) if (v > m) m = v; return m; };
const meanOf = (a) => { let s = 0; for (const v of a) s += v; return a.length ? s / a.length : 0; };
const clip01 = (v) => Math.min(1, Math.max(0, v));

export function clarity(c, x, sr) {
  const env = envelope(x);
  if (c.kind === "one_shot") {
    const af = Math.max(1, Math.floor((0.03 * sr) / ENV_HOP));
    const ratio = maxOf(env.slice(0, af)) / (meanOf(env) + EPS);
    const transient = clip01((ratio - 1) / 2);
    const tailN = Math.max(1, Math.floor(env.length / 8));
    const tail = meanOf(env.slice(env.length - tailN)) / (maxOf(env) + EPS);
    return 0.5 * transient + 0.5 * clip01(1 - 2 * tail);
  }
  if (c.kind === "loop") {
    const parts = Math.max(1, Math.round(c.beats || 4));
    const per = [];
    for (let p = 0; p < parts; p++) {
      const a = Math.floor((p * env.length) / parts), b = Math.floor(((p + 1) * env.length) / parts);
      per.push(b > a ? meanOf(env.slice(a, b)) : 0);
    }
    const m = meanOf(per);
    const sd = Math.sqrt(meanOf(per.map((v) => (v - m) ** 2)));
    return clip01(1 - sd / (m + EPS));
  }
  const mx = maxOf(env);
  if (mx <= EPS) return 0;
  let on = 0;
  for (const v of env) if (v > mx * 0.1) on++;
  return on / env.length;
}

export function loudness(x, cfg) {
  const s = cfg.scoring;
  const level = db(rms(x));
  let clipped = 0;
  for (const v of x) if (Math.abs(v) >= 0.999) clipped++;
  if (level < s.min_rms_dbfs || clipped / x.length > s.max_clipped_fraction) return 0;
  return clip01((level - s.min_rms_dbfs) / (-18 - s.min_rms_dbfs));
}

const MFCC_HOP = 512;

/** MFCCs of a whole stem, computed once; candidates read their frames from it. */
export function stemMfcc(y, sr) {
  return mfcc(y, sr, { nMfcc: 20, nFft: 2048, hop: MFCC_HOP });
}

/** Frame range of MFCC frames centred inside [start, end). */
function frameRange(m, start, end) {
  const a = Math.min(m.nFrames - 1, Math.ceil(start / MFCC_HOP));
  const b = Math.max(a + 1, Math.min(m.nFrames, Math.floor(end / MFCC_HOP) + 1));
  return [a, b];
}

function meanCoeffs(m, a, b, k0, k1) {
  const out = new Float64Array(k1 - k0);
  for (let t = a; t < b; t++) for (let k = k0; k < k1; k++) out[k - k0] += m.data[t * 20 + k] / (b - a);
  return out;
}

export function loopability(x, sr, m = null, start = 0) {
  const edge = Math.min(Math.floor(0.05 * sr), Math.floor(x.length / 4));
  if (edge < 256) return 0;
  const head = x.subarray(0, edge), tail = x.subarray(x.length - edge);
  const levelMatch = Math.exp(-Math.abs(Math.log((rms(tail) + EPS) / (rms(head) + EPS))));
  m ??= stemMfcc(x, sr);
  const [ha, hb] = frameRange(m, start, start + edge);
  const [ta, tb] = frameRange(m, start + x.length - edge, start + x.length);
  const a = meanCoeffs(m, ha, hb, 0, 13), b = meanCoeffs(m, ta, tb, 0, 13);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < 13; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  const timbre = dot / (Math.sqrt(na) * Math.sqrt(nb) + EPS);
  return clip01(0.5 * levelMatch + 0.5 * Math.max(timbre, 0));
}

/** MFCC 1-19 mean and spread over the slice (no MFCC 0, so a quieter repeat still matches). */
export function timbreVector(m, start, end) {
  const [a, b] = frameRange(m, start, end);
  const out = new Float64Array(38);
  for (let k = 1; k < 20; k++) {
    let s = 0;
    for (let t = a; t < b; t++) s += m.data[t * 20 + k];
    const mu = s / (b - a);
    let v = 0;
    for (let t = a; t < b; t++) v += (m.data[t * 20 + k] - mu) ** 2;
    out[k - 1] = mu;
    out[18 + k] = Math.sqrt(v / (b - a));
  }
  return out;
}

export function similarity(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += (a[i] - b[i]) ** 2; na += a[i] ** 2; nb += b[i] ** 2; }
  const dist = Math.sqrt(d) / (0.5 * (Math.sqrt(na) + Math.sqrt(nb)) + EPS);
  return Math.max(0, 1 - dist);
}

export function weighted(parts, weights) {
  let total = 0, score = 0;
  for (const [k, v] of Object.entries(parts)) {
    if (v == null || !(weights[k] > 0)) continue;
    total += weights[k];
    score += weights[k] * v;
  }
  if (!total) return 0;
  return parts.loudness === 0 ? 0 : score / total;
}

/** Fill in parts, score, cluster and representative on every candidate. */
export function scoreAll(cands, stems, sr, cfg, onProgress, mfccs = {}) {
  const weights = cfg.scoring.weights;
  const index = new EnergyIndex(stems);
  const vectors = new Array(cands.length);
  for (const c of cands) mfccs[c.stem] ??= stemMfcc(stems[c.stem], sr);
  cands.forEach((c, i) => {
    const x = stems[c.stem].subarray(c.start, c.end);
    const m = mfccs[c.stem];
    c.parts = {
      isolation: index.isolation(c.stem, c.start, c.end),
      clarity: clarity(c, x, sr),
      loudness: loudness(x, cfg),
      loopability: c.kind === "loop" ? loopability(x, sr, m, c.start) : null,
      uniqueness: 1,
    };
    vectors[i] = timbreVector(m, c.start, c.end);
    if (onProgress && i % 50 === 0) onProgress(i / cands.length);
  });

  const prelimW = { ...weights, uniqueness: 0 };
  const prelim = cands.map((c) => weighted({ ...c.parts, uniqueness: null }, prelimW));
  const threshold = cfg.scoring.duplicate_similarity;
  let nextCluster = 0;
  const groups = new Map();
  cands.forEach((c, i) => {
    const key = `${c.stem}|${c.category}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(i);
  });
  for (const members of groups.values()) {
    const reps = [];
    for (const i of [...members].sort((a, b) => prelim[b] - prelim[a])) {
      const c = cands[i];
      let best = -1, bestSim = -1;
      reps.forEach((r, j) => {
        const s = similarity(vectors[i], vectors[r]);
        if (s > bestSim) { bestSim = s; best = j; }
      });
      if (best >= 0 && bestSim >= threshold) {
        const rep = cands[reps[best]];
        c.cluster = rep.cluster; c.representative = false; c.similarity = bestSim;
        c.parts.uniqueness = clip01(1 - bestSim);
        continue;
      }
      c.cluster = nextCluster++;
      c.representative = true;
      reps.push(i);
    }
  }
  const r4 = (v) => (v == null ? null : Math.round(v * 1e4) / 1e4);
  for (const c of cands) {
    for (const k of Object.keys(c.parts)) c.parts[k] = r4(c.parts[k]);
    c.score = r4(weighted(c.parts, weights));
  }
}

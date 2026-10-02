// Stage 3 in the browser: tempo, beat grid, downbeats, key and sections.
// Mirrors backend/sangisa/worker/stages/analyze.py (librosa's algorithms, re-implemented).

import {
  chroma, fft, hann, mean, median, melSpectrogram, mfcc, nextPow2, powerToDb, slope, std,
} from "./dsp.js";

export const HOP = 512;
export const PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

export function analyze(y, sr, cfg) {
  const duration = y.length / sr;
  const { bpm, beats } = detectBeats(y, sr, cfg.analysis.min_bpm, cfg.analysis.max_bpm);
  const downbeats = detectDownbeats(y, sr, beats);
  const key = detectKey(y, sr);
  const r = (v, d) => Math.round(v * 10 ** d) / 10 ** d;
  return {
    duration_s: r(duration, 3),
    bpm: r(bpm, 2),
    key: key.name,
    key_short: key.short,
    key_confidence: r(key.confidence, 3),
    time_signature: "4/4",
    beats_s: Array.from(beats, (b) => r(b, 4)),
    downbeats_s: Array.from(downbeats, (b) => r(b, 4)),
    sections: detectSections(y, sr, beats, duration),
  };
}

// ---------------------------------------------------------------- onset strength

/** librosa.onset.onset_strength on a (dB) mel spectrogram, frame-major [nFrames * nMels]. */
export function onsetStrengthFromDb(S, nFrames, nMels, nFft = 2048, hop = HOP) {
  const lag = 1;
  const env = new Float32Array(nFrames);
  const pad = lag + Math.floor(nFft / (2 * hop));
  for (let t = lag; t < nFrames; t++) {
    let acc = 0;
    for (let m = 0; m < nMels; m++) {
      const d = S[t * nMels + m] - S[(t - lag) * nMels + m];
      if (d > 0) acc += d;
    }
    const idx = t - lag + pad;
    if (idx < nFrames) env[idx] = acc / nMels;
  }
  return env;
}

export function onsetStrength(y, sr) {
  const mel = melSpectrogram(y, sr, { nFft: 2048, hop: HOP, nMels: 128 });
  powerToDb(mel.data);
  return onsetStrengthFromDb(mel.data, mel.nFrames, 128);
}

// ---------------------------------------------------------------- tempo and beats

/** librosa.feature.tempo: autocorrelation tempogram averaged over time, with a log-normal prior at 120 BPM. */
export function estimateTempo(env, sr, hop = HOP, startBpm = 120, stdBpm = 1, maxTempo = 320) {
  const win = Math.floor((8 * sr) / hop);
  const half = win >> 1;
  const n = env.length;
  // linear_ramp padding to 0 at both ends
  const padded = new Float64Array(n + 2 * half);
  for (let i = 0; i < half; i++) {
    padded[i] = (env[0] * i) / half;
    padded[n + half + i] = (env[n - 1] * (half - i)) / half;
  }
  for (let i = 0; i < n; i++) padded[half + i] = env[i];
  const w = hann(win);
  const N = nextPow2(2 * win - 1);
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  const acc = new Float64Array(win);
  for (let t = 0; t < n; t++) {
    re.fill(0); im.fill(0);
    for (let i = 0; i < win; i++) re[i] = padded[t + i] * w[i];
    fft(re, im);
    for (let k = 0; k < N; k++) { re[k] = re[k] * re[k] + im[k] * im[k]; im[k] = 0; }
    fft(re, im, true);
    const peak = Math.abs(re[0]) || 1;
    for (let l = 0; l < win; l++) acc[l] += re[l] / peak;
  }
  let best = -Infinity, bestLag = 1;
  for (let l = 1; l < win; l++) {
    const bpm = (60 * sr) / (hop * l);
    if (bpm >= maxTempo) continue;
    const prior = -0.5 * ((Math.log2(bpm) - Math.log2(startBpm)) / stdBpm) ** 2;
    const score = Math.log1p(1e6 * Math.max(0, acc[l] / n)) + prior;
    if (score > best) { best = score; bestLag = l; }
  }
  return (60 * sr) / (hop * bestLag);
}

/** librosa's dynamic-programming beat tracker (Ellis 2007). Returns beat frames. */
export function beatTrack(env, sr, bpm, hop = HOP, tightness = 100) {
  const n = env.length;
  if (!env.some((v) => v > 0)) return [];
  const fpb = Math.round((sr / hop) * (60 / bpm));
  const sd = std(env, 1) + 1e-10;
  const onsets = Float64Array.from(env, (v) => v / sd);
  // local score: convolve with a Gaussian of width ~ one beat ('same' mode)
  const wl = 2 * fpb + 1;
  const win = Float64Array.from({ length: wl }, (_, i) => Math.exp(-0.5 * (((i - fpb) * 32) / fpb) ** 2));
  const local = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let k = 0; k < wl; k++) {
      const j = i + fpb - k;
      if (j >= 0 && j < n) acc += onsets[j] * win[k];
    }
    local[i] = acc;
  }
  let localMax = 0;
  for (const v of local) if (v > localMax) localMax = v;
  const thresh = 0.01 * localMax;
  const cum = new Float64Array(n);
  const back = new Int32Array(n).fill(-1);
  cum[0] = local[0];
  let firstBeat = true;
  const lfpb = Math.log(fpb);
  for (let i = 1; i < n; i++) {
    let bestScore = -Infinity, loc = -1;
    for (let p = i - Math.round(fpb / 2); p >= i - 2 * fpb; p--) {
      if (p < 0) break;
      const s = cum[p] - tightness * (Math.log(i - p) - lfpb) ** 2;
      if (s > bestScore) { bestScore = s; loc = p; }
    }
    cum[i] = loc >= 0 ? local[i] + bestScore : local[i];
    if (firstBeat && local[i] < thresh) back[i] = -1;
    else { back[i] = loc; firstBeat = false; }
  }
  // last beat: final local max of the cumulative score above half the median local max
  const isMax = (i) => cum[i] > (i > 0 ? cum[i - 1] : cum[i]) && cum[i] >= (i < n - 1 ? cum[i + 1] : cum[i]);
  const maxes = [];
  for (let i = 0; i < n; i++) if (isMax(i)) maxes.push(cum[i]);
  const th = 0.5 * median(maxes.length ? maxes : [0]);
  let tail = n - 1;
  for (let i = n - 1; i >= 0; i--) if (isMax(i) && cum[i] >= th) { tail = i; break; }
  let beats = [];
  for (let i = tail; i >= 0; i = back[i]) { beats.push(i); if (back[i] < 0) break; }
  beats.reverse();
  // trim weak beats at the edges
  const hw = [0, 0.5, 1, 0.5, 0];
  const vals = beats.map((b) => local[b]);
  const smooth = vals.map((_, i) => hw.reduce((s, w, k) => {
    const j = i + 2 - k;
    return j >= 0 && j < vals.length ? s + w * vals[j] : s;
  }, 0));
  const thr = 0.5 * Math.sqrt(mean(smooth.map((v) => v * v)));
  let a = 0, b = beats.length;
  while (a < b && smooth[a] <= thr) a++;
  while (b > a && smooth[b - 1] <= thr) b--;
  beats = beats.slice(a, b);
  return beats;
}

export function detectBeats(y, sr, minBpm, maxBpm) {
  const env = onsetStrength(y, sr);
  let bpm = estimateTempo(env, sr);
  let beats = beatTrack(env, sr, bpm).map((f) => (f * HOP) / sr);
  if (bpm <= 0 || beats.length < 2) {
    return { bpm: 120, beats: Array.from({ length: Math.floor(y.length / sr / 0.5) }, (_, i) => i * 0.5) };
  }
  while (bpm < minBpm) {
    bpm *= 2;
    const mids = beats.slice(1).map((b, i) => (beats[i] + b) / 2);
    beats = [...beats, ...mids].sort((p, q) => p - q);
  }
  while (bpm > maxBpm) {
    bpm /= 2;
    beats = beats.filter((_, i) => i % 2 === 0);
  }
  if (beats.length > 4) {
    const period = slope(beats);
    if (period > 0) bpm = 60 / period;
  }
  return { bpm, beats };
}

/** Average columns of a frame-major feature matrix between boundaries (librosa.util.sync). */
export function syncFeatures(data, nFrames, dim, frames) {
  const bounds = [...new Set([0, ...frames.filter((f) => f >= 0 && f <= nFrames), nFrames])].sort((a, b) => a - b);
  const cols = [];
  for (let i = 0; i < bounds.length - 1; i++) {
    const [s, e] = [bounds[i], bounds[i + 1]];
    const v = new Float64Array(dim);
    if (e > s) {
      for (let t = s; t < e; t++) for (let k = 0; k < dim; k++) v[k] += data[t * dim + k];
      for (let k = 0; k < dim; k++) v[k] /= e - s;
    }
    cols.push(v);
  }
  return cols;
}

export function detectDownbeats(y, sr, beats, beatsPerBar = 4) {
  if (beats.length < beatsPerBar * 2) return beats.slice(0, 1);
  const frames = beats.map((b) => Math.round((b * sr) / HOP));
  const mel = melSpectrogram(y, sr, { nFft: 2048, hop: HOP, nMels: 64, fmax: 8000 });
  const low = new Float32Array(mel.nFrames * 8);
  for (let t = 0; t < mel.nFrames; t++) for (let m = 0; m < 8; m++) low[t * 8 + m] = mel.data[t * 64 + m];
  powerToDb(low);
  const lowEnv = onsetStrengthFromDb(low, mel.nFrames, 8);
  const lowAtBeat = frames.map((f) => lowEnv[Math.min(Math.max(f, 0), lowEnv.length - 1)]);

  const ch = chroma(y, sr, { nFft: 2048, hop: HOP });
  const synced = syncFeatures(ch.data, ch.nFrames, 12, frames).slice(1, beats.length + 1);
  const unit = synced.map((v) => { const n = Math.hypot(...v) + 1e-9; return v.map((x) => x / n); });
  const change = new Float64Array(beats.length);
  for (let i = 1; i < unit.length; i++) {
    let dot = 0;
    for (let k = 0; k < 12; k++) dot += unit[i][k] * unit[i - 1][k];
    change[i] = 1 - dot;
  }
  const norm = (v) => { const m = mean(v), s = std(v) + 1e-9; return Array.from(v, (x) => (x - m) / s); };
  const a = norm(lowAtBeat), b = norm(change);
  const strength = a.map((v, i) => v + b[i]);
  let best = 0, bestScore = -Infinity;
  for (let p = 0; p < beatsPerBar; p++) {
    const vals = strength.filter((_, i) => i % beatsPerBar === p);
    const sc = mean(vals);
    if (sc > bestScore) { bestScore = sc; best = p; }
  }
  return beats.filter((_, i) => i >= best && (i - best) % beatsPerBar === 0);
}

function corr(a, b) {
  const ma = mean(a), mb = mean(b);
  let num = 0, da = 0, dbb = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    dbb += (b[i] - mb) ** 2;
  }
  return da && dbb ? num / Math.sqrt(da * dbb) : 0;
}

export function detectKey(y, sr) {
  const ch = chroma(y, sr, { nFft: 4096, hop: HOP * 4 });
  const profile = new Float64Array(12);
  for (let t = 0; t < ch.nFrames; t++) for (let k = 0; k < 12; k++) profile[k] += ch.data[t * 12 + k] / ch.nFrames;
  if (!profile.some((v) => v > 1e-6)) return { name: null, short: null, confidence: 0 };
  const scores = [];
  for (let tonic = 0; tonic < 12; tonic++) {
    for (const [mode, tpl] of [["major", MAJOR], ["minor", MINOR]]) {
      const rolled = tpl.map((_, i) => tpl[(i - tonic + 12) % 12]);
      scores.push([corr(profile, rolled), tonic, mode]);
    }
  }
  scores.sort((p, q) => q[0] - p[0]);
  const [best, tonic, mode] = scores[0];
  const confidence = Math.min(1, Math.max(0, best - scores[1][0]) + Math.max(0, best) * 0.5);
  const name = PITCH_NAMES[tonic];
  return { name: `${name} ${mode}`, short: `${name}${mode === "major" ? "maj" : "min"}`, confidence };
}

/** Contiguity-constrained Ward clustering into k segments (librosa.segment.agglomerative). */
export function agglomerative(cols, k) {
  let segs = cols.map((v, i) => ({ start: i, n: 1, mean: Float64Array.from(v) }));
  const cost = (a, b) => {
    let d = 0;
    for (let i = 0; i < a.mean.length; i++) d += (a.mean[i] - b.mean[i]) ** 2;
    return ((a.n * b.n) / (a.n + b.n)) * d;
  };
  while (segs.length > k) {
    let bi = 0, bc = Infinity;
    for (let i = 0; i < segs.length - 1; i++) {
      const c = cost(segs[i], segs[i + 1]);
      if (c < bc) { bc = c; bi = i; }
    }
    const [a, b] = [segs[bi], segs[bi + 1]];
    const n = a.n + b.n;
    const m = a.mean.map((v, i) => (v * a.n + b.mean[i] * b.n) / n);
    segs.splice(bi, 2, { start: a.start, n, mean: m });
  }
  return segs.map((s) => s.start);
}

export function detectSections(y, sr, beats, duration) {
  if (beats.length < 16 || duration < 20) return [{ label: "A", start_s: 0, end_s: Math.round(duration * 1000) / 1000 }];
  const frames = beats.map((b) => Math.round((b * sr) / HOP));
  const ch = chroma(y, sr, { nFft: 2048, hop: HOP });
  const mf = mfcc(y, sr, { nMfcc: 13, hop: HOP });
  const cc = syncFeatures(ch.data, ch.nFrames, 12, frames).map((v) => {
    const m = Math.max(...v.map(Math.abs)) || 1;
    return v.map((x) => x / m);
  });
  const mc = syncFeatures(mf.data, mf.nFrames, 13, frames);
  const rowMax = new Float64Array(13);
  for (const v of mc) for (let k = 0; k < 13; k++) rowMax[k] = Math.max(rowMax[k], Math.abs(v[k]));
  const n = Math.min(cc.length, mc.length);
  const feats = [];
  for (let i = 0; i < n; i++) feats.push(Float64Array.from([...cc[i], ...mc[i].map((x, k) => x / (rowMax[k] || 1))]));
  const k = Math.min(Math.max(Math.round(duration / 20), 2), 10);
  const bounds = agglomerative(feats, k);
  const edges = [0, ...beats, duration];
  const starts = bounds.map((b) => edges[b]);
  const ends = [...starts.slice(1), duration];
  const reps = [];
  const letters = [];
  bounds.forEach((b, i) => {
    const e = i + 1 < bounds.length ? bounds[i + 1] : feats.length;
    const m = new Float64Array(feats[0].length);
    for (let t = b; t < e; t++) for (let j = 0; j < m.length; j++) m[j] += feats[t][j] / (e - b);
    const sims = reps.map((r) => {
      let dot = 0, na = 0, nb = 0;
      for (let j = 0; j < m.length; j++) { dot += m[j] * r[j]; na += m[j] ** 2; nb += r[j] ** 2; }
      return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
    });
    const best = sims.length ? sims.indexOf(Math.max(...sims)) : -1;
    if (best >= 0 && sims[best] > 0.9) letters.push(String.fromCharCode(65 + best));
    else { reps.push(m); letters.push(String.fromCharCode(65 + reps.length - 1)); }
  });
  const r = (v) => Math.round(v * 1000) / 1000;
  return letters
    .map((label, i) => ({ label, start_s: r(starts[i]), end_s: r(ends[i]) }))
    .filter((s) => s.end_s - s.start_s > 0.5);
}

export function sectionAt(sections, t) {
  for (const s of sections) if (s.start_s <= t && t < s.end_s) return s.label;
  return null;
}

// Signal-processing building blocks for the in-browser engine.
// They follow librosa's definitions (the Python pipeline uses librosa) closely
// enough that both pipelines agree on tempo, key, onsets and timbre.

const EPS = 1e-10;

// ---------------------------------------------------------------- FFT

const fftCache = new Map();

function fftTables(n) {
  let t = fftCache.get(n);
  if (t) return t;
  if (n & (n - 1)) throw new Error(`FFT size must be a power of two, got ${n}`);
  const bits = Math.log2(n);
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n);
    sin[i] = -Math.sin((2 * Math.PI * i) / n);
  }
  t = { rev, cos, sin };
  fftCache.set(n, t);
  return t;
}

/** In-place complex FFT (radix 2). re/im are Float64Array of length n. */
export function fft(re, im, inverse = false) {
  const n = re.length;
  const { rev, cos, sin } = fftTables(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  const sgn = inverse ? -1 : 1;
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step];
        const wi = sgn * sin[k * step];
        const a = start + k;
        const b = a + half;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

export const nextPow2 = (n) => 1 << Math.ceil(Math.log2(Math.max(2, n)));

/** Periodic Hann window (scipy get_window('hann', n, fftbins=True), as librosa uses). */
export function hann(n, periodic = true) {
  const w = new Float64Array(n);
  const d = periodic ? n : n - 1;
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / d);
  return w;
}

// ---------------------------------------------------------------- STFT

/**
 * Walk the windowed frames of y (centered, zero padded, periodic Hann like librosa) and hand each
 * frame's spectrum (bins 0..nFft/2) to cb(t, re, im). Two real frames share one complex FFT.
 */
function forEachSpectrum(y, nFft, hop, center, cb) {
  const nBins = nFft / 2 + 1;
  const pad = center ? nFft >> 1 : 0;
  const len = y.length + 2 * pad;
  const nFrames = len < nFft ? 1 : 1 + Math.floor((len - nFft) / hop);
  const win = hann(nFft);
  const re = new Float64Array(nFft), im = new Float64Array(nFft);
  const ar = new Float64Array(nBins), ai = new Float64Array(nBins);
  const br = new Float64Array(nBins), bi = new Float64Array(nBins);
  const fill = (buf, t) => {
    const start = t * hop - pad;
    if (start >= 0 && start + nFft <= y.length) {
      for (let i = 0; i < nFft; i++) buf[i] = y[start + i] * win[i];
    } else {
      for (let i = 0; i < nFft; i++) {
        const j = start + i;
        buf[i] = j >= 0 && j < y.length ? y[j] * win[i] : 0;
      }
    }
  };
  for (let t = 0; t < nFrames; t += 2) {
    fill(re, t);
    if (t + 1 < nFrames) fill(im, t + 1); else im.fill(0);
    fft(re, im);
    for (let k = 0; k < nBins; k++) {
      const j = k === 0 ? 0 : nFft - k;
      ar[k] = (re[k] + re[j]) / 2; ai[k] = (im[k] - im[j]) / 2;
      br[k] = (im[k] + im[j]) / 2; bi[k] = (re[j] - re[k]) / 2;
    }
    cb(t, ar, ai);
    if (t + 1 < nFrames) cb(t + 1, br, bi);
  }
  return { nFrames, nBins };
}

/**
 * Power (or magnitude) spectrogram, librosa-style. Returns
 * {data: Float32Array[nFrames * nBins], nFrames, nBins} stored frame-major (data[t * nBins + f]).
 */
export function spectrogram(y, { nFft = 2048, hop = 512, power = 2, center = true } = {}) {
  const nBins = nFft / 2 + 1;
  const pad = center ? nFft >> 1 : 0;
  const len = y.length + 2 * pad;
  const nFrames = len < nFft ? 1 : 1 + Math.floor((len - nFft) / hop);
  const out = new Float32Array(nFrames * nBins);
  forEachSpectrum(y, nFft, hop, center, (t, re, im) => {
    const base = t * nBins;
    for (let f = 0; f < nBins; f++) {
      const p = re[f] * re[f] + im[f] * im[f];
      out[base + f] = power === 2 ? p : power === 1 ? Math.sqrt(p) : Math.pow(p, power / 2);
    }
  });
  return { data: out, nFrames, nBins, nFft, hop };
}

const specCache = new WeakMap();

/** spectrogram() memoised per signal, so analysis and scoring share one STFT per stem. */
export function cachedSpectrogram(y, nFft = 2048, hop = 512) {
  let m = specCache.get(y);
  if (!m) specCache.set(y, (m = new Map()));
  const key = `${nFft}:${hop}`;
  if (!m.has(key)) m.set(key, spectrogram(y, { nFft, hop }));
  return m.get(key);
}

/** Forget cached spectrograms of y (they are large: ~60 MB for a 3-minute signal). */
export function dropCache(y) {
  specCache.delete(y);
}

/** Complex STFT for separation: returns {re, im} Float32Array frame-major plus shape. */
export function stft(y, { nFft = 2048, hop = 512 } = {}) {
  const nBins = nFft / 2 + 1;
  const pad = nFft >> 1;
  const nFrames = 1 + Math.floor((y.length + 2 * pad - nFft) / hop);
  const outRe = new Float32Array(nFrames * nBins);
  const outIm = new Float32Array(nFrames * nBins);
  forEachSpectrum(y, nFft, hop, true, (t, re, im) => {
    outRe.set(re, t * nBins);
    outIm.set(im, t * nBins);
  });
  return { re: outRe, im: outIm, nFrames, nBins, nFft, hop };
}

/** Inverse of stft() by weighted overlap-add, trimmed to `length` samples. */
export function istft({ re, im, nFrames, nBins, nFft, hop }, length) {
  const pad = nFft >> 1;
  const total = nFft + hop * (nFrames - 1);
  const out = new Float32Array(total);
  const norm = new Float32Array(total);
  const win = hann(nFft);
  const fr = new Float64Array(nFft);
  const fi = new Float64Array(nFft);
  for (let t = 0; t < nFrames; t++) {
    const base = t * nBins;
    for (let f = 0; f < nBins; f++) { fr[f] = re[base + f]; fi[f] = im[base + f]; }
    for (let f = 1; f < nFft / 2; f++) { fr[nFft - f] = fr[f]; fi[nFft - f] = -fi[f]; }
    fft(fr, fi, true);
    const start = t * hop;
    for (let i = 0; i < nFft; i++) {
      out[start + i] += fr[i] * win[i];
      norm[start + i] += win[i] * win[i];
    }
  }
  for (let j = 0; j < total; j++) out[j] = norm[j] > 1e-8 ? out[j] / norm[j] : 0;
  const end = Math.min(total, pad + length);
  if (end - pad === length) return out.subarray(pad, end);
  const y = new Float32Array(length);
  y.set(out.subarray(pad, end));
  return y;
}

// ---------------------------------------------------------------- mel / chroma / MFCC

function hzToMel(f) {
  const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27;
  return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fSp;
}

function melToHz(m) {
  const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27;
  return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m;
}

const melCache = new Map();

/** Slaney-style mel filterbank (librosa.filters.mel defaults). Returns Float32Array[nMels * nBins]. */
export function melFilters(sr, nFft, nMels = 128, fmin = 0, fmax = sr / 2) {
  const key = `${sr}:${nFft}:${nMels}:${fmin}:${fmax}`;
  if (melCache.has(key)) return melCache.get(key);
  const nBins = nFft / 2 + 1;
  const fftFreqs = Float64Array.from({ length: nBins }, (_, i) => (i * sr) / nFft);
  const minMel = hzToMel(fmin), maxMel = hzToMel(fmax);
  const melF = Float64Array.from({ length: nMels + 2 }, (_, i) => melToHz(minMel + ((maxMel - minMel) * i) / (nMels + 1)));
  const w = new Float32Array(nMels * nBins);
  for (let m = 0; m < nMels; m++) {
    const lo = melF[m], c = melF[m + 1], hi = melF[m + 2];
    const enorm = 2 / (hi - lo);
    for (let k = 0; k < nBins; k++) {
      const f = fftFreqs[k];
      const lower = (f - lo) / (c - lo);
      const upper = (hi - f) / (hi - c);
      const v = Math.max(0, Math.min(lower, upper));
      w[m * nBins + k] = v * enorm;
    }
  }
  melCache.set(key, w);
  return w;
}

const rangeCache = new WeakMap();

/** Per-row [first, last] bins whose weight is worth multiplying (filterbanks are mostly zeros). */
function filterRanges(filters, nOut, nBins) {
  let r = rangeCache.get(filters);
  if (r) return r;
  r = new Int32Array(nOut * 2);
  for (let m = 0; m < nOut; m++) {
    let max = 0;
    for (let k = 0; k < nBins; k++) max = Math.max(max, Math.abs(filters[m * nBins + k]));
    const th = max * 1e-7;
    let a = nBins, b = -1;
    for (let k = 0; k < nBins; k++) if (Math.abs(filters[m * nBins + k]) > th) { if (k < a) a = k; b = k; }
    r[2 * m] = a; r[2 * m + 1] = b;
  }
  rangeCache.set(filters, r);
  return r;
}

/** Apply a filterbank [nOut * nBins] to a frame-major spectrogram. Returns frame-major [nFrames * nOut]. */
export function applyFilters(spec, filters, nOut) {
  const { data, nFrames, nBins } = spec;
  const ranges = filterRanges(filters, nOut, nBins);
  const out = new Float32Array(nFrames * nOut);
  for (let t = 0; t < nFrames; t++) {
    const s = t * nBins;
    for (let m = 0; m < nOut; m++) {
      const fb = m * nBins;
      let acc = 0;
      for (let k = ranges[2 * m], e = ranges[2 * m + 1]; k <= e; k++) acc += filters[fb + k] * data[s + k];
      out[t * nOut + m] = acc;
    }
  }
  return out;
}

export function melSpectrogram(y, sr, { nFft = 2048, hop = 512, nMels = 128, fmax = sr / 2 } = {}) {
  const spec = cachedSpectrogram(y, nFft, hop);
  return { data: applyFilters(spec, melFilters(sr, nFft, nMels, 0, fmax), nMels), nFrames: spec.nFrames, nMels };
}

/** librosa.power_to_db(S, ref=1.0 or max, amin=1e-10, top_db=80), in place. */
export function powerToDb(data, { refMax = false, topDb = 80 } = {}) {
  let ref = 1;
  if (refMax) { ref = 0; for (const v of data) if (v > ref) ref = v; ref = Math.max(ref, EPS); }
  const refDb = 10 * Math.log10(Math.max(ref, EPS));
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    data[i] = 10 * Math.log10(Math.max(data[i], EPS)) - refDb;
    if (data[i] > max) max = data[i];
  }
  if (topDb != null) for (let i = 0; i < data.length; i++) if (data[i] < max - topDb) data[i] = max - topDb;
  return data;
}

const dctCache = new Map();

/** Orthonormal DCT-II matrix [nOut * n]. */
function dctMatrix(n, nOut) {
  const key = `${n}:${nOut}`;
  if (dctCache.has(key)) return dctCache.get(key);
  const m = new Float64Array(nOut * n);
  for (let k = 0; k < nOut; k++) {
    const s = k === 0 ? Math.sqrt(1 / n) : Math.sqrt(2 / n);
    for (let i = 0; i < n; i++) m[k * n + i] = s * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n));
  }
  dctCache.set(key, m);
  return m;
}

/** MFCC like librosa.feature.mfcc: returns frame-major Float32Array [nFrames * nMfcc]. */
export function mfcc(y, sr, { nMfcc = 20, nFft = 2048, hop = 512, nMels = 128 } = {}) {
  const mel = melSpectrogram(y, sr, { nFft, hop, nMels });
  powerToDb(mel.data);
  const d = dctMatrix(nMels, nMfcc);
  const out = new Float32Array(mel.nFrames * nMfcc);
  for (let t = 0; t < mel.nFrames; t++) {
    for (let k = 0; k < nMfcc; k++) {
      let acc = 0;
      for (let i = 0; i < nMels; i++) acc += d[k * nMels + i] * mel.data[t * nMels + i];
      out[t * nMfcc + k] = acc;
    }
  }
  return { data: out, nFrames: mel.nFrames, n: nMfcc };
}

const chromaCache = new Map();

/** librosa.filters.chroma (tuning 0, ctroct 5, octwidth 2, base_c). Returns Float32Array[12 * nBins]. */
export function chromaFilters(sr, nFft) {
  const key = `${sr}:${nFft}`;
  if (chromaCache.has(key)) return chromaCache.get(key);
  const nChroma = 12, ctroct = 5, octwidth = 2;
  const nBins = nFft / 2 + 1;
  const A440 = 440;
  const frq = [];
  for (let i = 1; i < nFft; i++) {
    const f = (i * sr) / nFft;
    frq.push(nChroma * Math.log2(f / (A440 / 16)));
  }
  frq.unshift(frq[0] - 1.5 * nChroma);
  const width = frq.map((v, i) => (i < frq.length - 1 ? Math.max(frq[i + 1] - v, 1) : 1));
  const half = Math.round(nChroma / 2);
  const wts = Array.from({ length: nChroma }, () => new Float64Array(nFft));
  for (let i = 0; i < nFft; i++) {
    for (let c = 0; c < nChroma; c++) {
      let d = frq[i] - c;
      d = ((((d + half + 10 * nChroma) % nChroma) + nChroma) % nChroma) - half;
      wts[c][i] = Math.exp(-0.5 * ((2 * d) / width[i]) ** 2);
    }
  }
  for (let i = 0; i < nFft; i++) {
    let s = 0;
    for (let c = 0; c < nChroma; c++) s += wts[c][i] ** 2;
    s = Math.sqrt(s) || 1;
    const oct = Math.exp(-0.5 * ((frq[i] / nChroma - ctroct) / octwidth) ** 2);
    for (let c = 0; c < nChroma; c++) wts[c][i] = (wts[c][i] / s) * oct;
  }
  const out = new Float32Array(nChroma * nBins);
  for (let c = 0; c < nChroma; c++) {
    const src = wts[(c + 3) % nChroma]; // base_c: roll by -3 so index 0 is C
    for (let k = 0; k < nBins; k++) out[c * nBins + k] = src[k];
  }
  chromaCache.set(key, out);
  return out;
}

/** chroma_stft: frame-major [nFrames * 12], each frame scaled so its max is 1. */
export function chroma(y, sr, { nFft = 4096, hop = 2048 } = {}) {
  const spec = cachedSpectrogram(y, nFft, hop);
  const c = applyFilters(spec, chromaFilters(sr, nFft), 12);
  for (let t = 0; t < spec.nFrames; t++) {
    let m = 0;
    for (let k = 0; k < 12; k++) m = Math.max(m, c[t * 12 + k]);
    if (m > 1e-10) for (let k = 0; k < 12; k++) c[t * 12 + k] /= m;
  }
  return { data: c, nFrames: spec.nFrames, n: 12, hop };
}

// ---------------------------------------------------------------- small helpers

/** Frame RMS (librosa.feature.rms). */
export function rmsFrames(y, frameLength = 2048, hop = 512, center = true) {
  const pad = center ? frameLength >> 1 : 0;
  const len = y.length + 2 * pad;
  const n = len < frameLength ? 1 : 1 + Math.floor((len - frameLength) / hop);
  const out = new Float32Array(n);
  for (let t = 0; t < n; t++) {
    let acc = 0;
    const start = t * hop - pad;
    for (let i = 0; i < frameLength; i++) {
      const j = start + i;
      if (j >= 0 && j < y.length) acc += y[j] * y[j];
    }
    out[t] = Math.sqrt(acc / frameLength);
  }
  return out;
}

export function rms(x, start = 0, end = x.length) {
  if (end <= start) return 0;
  let acc = 0;
  for (let i = start; i < end; i++) acc += x[i] * x[i];
  return Math.sqrt(acc / (end - start) + EPS);
}

export const db = (v) => 20 * Math.log10(Math.max(v, EPS));
export const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);

export function median(values) {
  const a = Array.from(values).sort((x, y) => x - y);
  if (!a.length) return NaN;
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

export function std(a, ddof = 0) {
  const m = mean(a);
  let s = 0;
  for (const v of a) s += (v - m) ** 2;
  return Math.sqrt(s / Math.max(1, a.length - ddof));
}

export function toMono(channels) {
  if (channels.length === 1) return channels[0];
  const n = channels[0].length;
  const out = new Float32Array(n);
  for (const ch of channels) for (let i = 0; i < n; i++) out[i] += ch[i];
  for (let i = 0; i < n; i++) out[i] /= channels.length;
  return out;
}

/** Least-squares slope of y against 0..n-1. */
export function slope(y) {
  const n = y.length;
  const mx = (n - 1) / 2;
  const my = mean(y);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (i - mx) * (y[i] - my); den += (i - mx) ** 2; }
  return den ? num / den : 0;
}

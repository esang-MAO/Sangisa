// Stage 2 in the browser. Separators take stereo audio ([L, R] Float32Arrays) and
// return {stemName: [L, R]} with every stem the same length as the input.
//
// "quick": harmonic/percussive separation by median filtering (Fitzgerald 2010,
// librosa.decompose.hpss), then a low/high split of the harmonic part.
//   drums = percussive, bass = harmonic below ~180 Hz, other = harmonic above.
// No AI model and no vocal stem, but it runs in seconds on a phone.

import { istft, stft, toMono } from "./dsp.js";

const N_FFT = 2048;
const HOP = 1024;
// Median lengths: 31 frames (~0.7 s) across time, so drum hits (even a kick's ringing body) are
// shorter than half the window; 17 bins (~370 Hz) across frequency.
// A kick's low, ringing body still tends to land in "bass": Quick kicks can sound thin.
const TIME_KERNEL = 31;
const FREQ_KERNEL = 17;

/** Sliding median of `len` values read at src[offset + i*stride], written to dst the same way. */
function medianLine(src, dst, offset, stride, len, k) {
  const h = k >> 1;
  const at = (i) => src[offset + Math.min(len - 1, Math.max(0, i)) * stride];
  const win = [];
  const insert = (v) => {
    let lo = 0, hi = win.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (win[m] < v) lo = m + 1; else hi = m; }
    win.splice(lo, 0, v);
  };
  const remove = (v) => {
    let lo = 0, hi = win.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (win[m] < v) lo = m + 1; else hi = m; }
    win.splice(lo, 1);
  };
  for (let i = -h; i <= h; i++) insert(at(i));
  for (let i = 0; i < len; i++) {
    dst[offset + i * stride] = win[h];
    if (i + 1 < len) { remove(at(i - h)); insert(at(i + h + 1)); }
  }
}

function hpssMasks(mag, nFrames, nBins) {
  const harm = new Float32Array(mag.length);
  for (let f = 0; f < nBins; f++) medianLine(mag, harm, f, nBins, nFrames, TIME_KERNEL); // across time
  // Median across frequency, frame by frame, written over `mag` (it isn't needed afterwards).
  const perc = mag;
  const row = new Float32Array(nBins), med = new Float32Array(nBins);
  for (let t = 0; t < nFrames; t++) {
    row.set(mag.subarray(t * nBins, (t + 1) * nBins));
    medianLine(row, med, 0, 1, nBins, FREQ_KERNEL);
    perc.set(med, t * nBins);
  }
  // soft mask (power 2): share of the harmonic estimate
  for (let i = 0; i < mag.length; i++) {
    const h = harm[i] * harm[i], p = perc[i] * perc[i];
    harm[i] = h + p > 1e-20 ? h / (h + p) : 0.5;
  }
  return harm; // harmonic mask; percussive = 1 - mask
}

export async function quickSeparate(channels, sr, onProgress = () => {}) {
  const n = channels[0].length;
  const mono = toMono(channels);
  let spec = stft(mono, { nFft: N_FFT, hop: HOP });
  const mag = new Float32Array(spec.re.length);
  for (let i = 0; i < mag.length; i++) mag[i] = Math.hypot(spec.re[i], spec.im[i]);
  const { nFrames, nBins } = spec;
  spec = null; // only the magnitudes are needed from here
  onProgress(0.2);
  const hMask = hpssMasks(mag, nFrames, nBins);
  onProgress(0.5);
  // crossover for the harmonic part: fully bass below 120 Hz, fully other above 250 Hz
  const lowShare = Float32Array.from({ length: nBins }, (_, f) => {
    const hz = (f * sr) / N_FFT;
    return hz <= 120 ? 1 : hz >= 250 ? 0 : (250 - hz) / 130;
  });
  const stems = { drums: [], bass: [], other: [] };
  channels.forEach((ch, ci) => {
    const S = stft(ch, { nFft: N_FFT, hop: HOP });
    for (const [name, gain] of [
      ["drums", (i) => 1 - hMask[i]],
      ["bass", (i, f) => hMask[i] * lowShare[f]],
    ]) {
      const re = new Float32Array(S.re.length), im = new Float32Array(S.im.length);
      for (let t = 0; t < S.nFrames; t++) {
        for (let f = 0; f < S.nBins; f++) {
          const i = t * S.nBins + f;
          const g = gain(i, f);
          re[i] = S.re[i] * g; im[i] = S.im[i] * g;
        }
      }
      stems[name].push(istft({ ...S, re, im }, n));
    }
    // The masks add up to one, so the rest of the signal is exactly "other".
    const rest = new Float32Array(n);
    const d = stems.drums[ci], b = stems.bass[ci];
    for (let i = 0; i < n; i++) rest[i] = ch[i] - d[i] - b[i];
    stems.other.push(rest);
    onProgress(0.5 + (0.5 * (ci + 1)) / channels.length);
  });
  return stems;
}

export const SEPARATORS = {
  quick: { label: "Quick split (no AI model)", run: quickSeparate },
};

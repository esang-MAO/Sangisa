// HT-Demucs in the browser, with ONNX Runtime Web running the network
// (exported by scripts/export_demucs_onnx.py).
//
// The network sits between two spectrogram steps that stay here in JavaScript, written to
// match demucs exactly (demucs/htdemucs.py _spec/_ispec, demucs/spec.py, demucs/apply.py):
//   chunk the song (training length, 25% overlap, triangular cross-fade)
//   -> STFT (4096-point, hop 1024, reflect padding, normalised), complex as channels
//   -> ONNX: per-source spectrograms + per-source time-branch output
//   -> inverse STFT of the spectrograms, plus the time branch
// Tests compare this file's output with demucs' own apply_model (tests/engine/demucs.test.mjs).

import { fft, hann } from "./dsp.js";

// ---------------------------------------------------------------- torch-compatible STFT

/** numpy/torch "reflect" padding (the edge sample isn't repeated). */
function reflectPad(x, left, right) {
  const n = x.length;
  const out = new Float32Array(n + left + right);
  out.set(x, left);
  for (let i = 1; i <= left; i++) out[left - i] = x[Math.min(i, n - 1)];
  for (let i = 1; i <= right; i++) out[left + n - 1 + i] = x[Math.max(0, n - 1 - i)];
  return out;
}

/**
 * torch.stft(x, nfft, hop, window=hann(nfft), normalized=True, center=True, pad_mode="reflect").
 * Calls cb(t, re, im) per frame with bins 0..nfft/2.
 */
function torchStft(x, nfft, hop, cb) {
  const half = nfft >> 1;
  const xp = reflectPad(x, half, half);
  const nFrames = 1 + Math.floor((xp.length - nfft) / hop);
  const win = hann(nfft);
  const scale = 1 / Math.sqrt(nfft);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let t = 0; t < nFrames; t++) {
    const s = t * hop;
    for (let i = 0; i < nfft; i++) { re[i] = xp[s + i] * win[i] * scale; im[i] = 0; }
    fft(re, im);
    cb(t, re, im);
  }
  return nFrames;
}

/** HTDemucs._spec + _magnitude(cac): [C*2, F=nfft/2, le] frame data for one chunk, flattened. */
export function demucsSpec(chunk, nfft, hop) {
  const L = chunk[0].length;
  const le = Math.ceil(L / hop);
  const pad = (hop >> 1) * 3;
  const F = nfft / 2;
  const C = chunk.length;
  const out = new Float32Array(C * 2 * F * le);
  chunk.forEach((x, c) => {
    const xp = reflectPad(x, pad, pad + le * hop - L);
    torchStft(xp, nfft, hop, (t, re, im) => {
      const tt = t - 2; // demucs keeps frames 2 .. 2+le
      if (tt < 0 || tt >= le) return;
      for (let f = 0; f < F; f++) {
        out[((c * 2) * F + f) * le + tt] = re[f];
        out[((c * 2 + 1) * F + f) * le + tt] = im[f];
      }
    });
  });
  return { data: out, frames: le };
}

/** HTDemucs._ispec for one source/channel: complex spectrogram [F, le] (re/im planes) -> L samples. */
function demucsIspec(reP, imP, F, le, nfft, hop, L) {
  const T = le + 4; // two zero frames padded on each side
  const pad = (hop >> 1) * 3;
  const total = nfft + hop * (T - 1);
  const ola = new Float64Array(total), env = new Float64Array(total);
  const win = hann(nfft);
  const scale = Math.sqrt(nfft);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let t = 0; t < T; t++) {
    const tt = t - 2;
    const s = t * hop;
    if (tt < 0 || tt >= le) {
      for (let i = 0; i < nfft; i++) env[s + i] += win[i] * win[i];
      continue;
    }
    re.fill(0); im.fill(0);
    for (let f = 0; f < F; f++) { re[f] = reP[f * le + tt]; im[f] = imP[f * le + tt]; }
    // bin F (Nyquist) is zero; mirror for a real signal
    for (let f = 1; f < F; f++) { re[nfft - f] = re[f]; im[nfft - f] = -im[f]; }
    fft(re, im, true);
    for (let i = 0; i < nfft; i++) {
      ola[s + i] += re[i] * scale * win[i];
      env[s + i] += win[i] * win[i];
    }
  }
  const start = (nfft >> 1) + pad; // center=True trim, then demucs' own pad
  const out = new Float32Array(L);
  for (let i = 0; i < L; i++) {
    const j = start + i;
    out[i] = env[j] > 1e-11 ? ola[j] / env[j] : 0;
  }
  return out;
}

// ---------------------------------------------------------------- the model

/** IEEE half -> float. */
function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

/** The weights file as the float32 external-data file the graph expects. */
export function expandWeights(meta, bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (meta.weights_format === "f32") return u8;
  const half = new Uint16Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
  const table = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) table[i] = halfToFloat(i);
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) out[i] = table[half[i]];
  return new Uint8Array(out.buffer);
}

export class Demucs {
  /**
   * @param {object} ort        the onnxruntime-web module
   * @param {ArrayBuffer|Uint8Array} graph  the .onnx file
   * @param {object} meta       the export's .json (sources, segment_samples, nfft, hop, weights)
   * @param {ArrayBuffer|Uint8Array} weights  the weights file (.f16 or .f32)
   * @param {string[]} [executionProviders]  e.g. ["webgpu"] or ["wasm"]
   */
  static async create(ort, graph, meta, weights, executionProviders = ["wasm"]) {
    const session = await ort.InferenceSession.create(graph instanceof Uint8Array ? graph : new Uint8Array(graph), {
      executionProviders,
      externalData: [{ path: meta.external_data_path, data: expandWeights(meta, weights) }],
      graphOptimizationLevel: "all",
    });
    return new Demucs(ort, session, meta);
  }

  constructor(ort, session, meta) {
    this.ort = ort;
    this.session = session;
    this.meta = meta;
  }

  /** One training-length chunk (2 x L) -> sources [S][C] Float32Array(L). */
  async runChunk(chunk) {
    const { nfft, hop, sources, audio_channels: C } = this.meta;
    const L = chunk[0].length;
    const F = nfft / 2;
    const { data: spec, frames: le } = demucsSpec(chunk, nfft, hop);
    const mix = new Float32Array(C * L);
    chunk.forEach((x, c) => mix.set(x, c * L));
    const { Tensor } = this.ort;
    const out = await this.session.run({
      mix: new Tensor("float32", mix, [1, C, L]),
      spec: new Tensor("float32", spec, [1, C * 2, F, le]),
    });
    const so = out.spec_out.data, to = out.time_out.data;
    const S = sources.length;
    const result = [];
    for (let s = 0; s < S; s++) {
      const chans = [];
      for (let c = 0; c < C; c++) {
        const base = ((s * C * 2) + c * 2) * F * le;
        const x = demucsIspec(so.subarray(base, base + F * le), so.subarray(base + F * le, base + 2 * F * le),
          F, le, nfft, hop, L);
        const tb = (s * C + c) * L;
        for (let i = 0; i < L; i++) x[i] += to[tb + i];
        chans.push(x);
      }
      result.push(chans);
    }
    out.spec_out.dispose?.(); out.time_out.dispose?.();
    return result;
  }

  /**
   * demucs.apply.apply_model(model, mix, shifts=0, split=True, overlap=0.25) with demucs' input
   * normalisation. channels: [L, R] Float32Array at meta.samplerate. Returns {source: [L, R]}.
   */
  async separate(channels, onProgress = () => {}) {
    const { sources, segment_samples: L } = this.meta;
    const C = channels.length;
    const n = channels[0].length;
    // normalise by the mono mix's mean and (unbiased) std, as demucs.separate does
    let mean = 0;
    for (let i = 0; i < n; i++) { let m = 0; for (const ch of channels) m += ch[i]; mean += m / C; }
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) { let m = 0; for (const ch of channels) m += ch[i]; v += (m / C - mean) ** 2; }
    const std = Math.sqrt(v / Math.max(1, n - 1)) || 1;

    const stride = Math.floor(0.75 * L);
    const half = Math.floor(L / 2);
    const peak = Math.max(half, L - half);
    const weight = Float32Array.from({ length: L }, (_, i) => (i < half ? i + 1 : L - i) / peak);
    const acc = sources.map(() => channels.map(() => new Float32Array(n)));
    const sumW = new Float32Array(n);
    const offsets = [];
    for (let o = 0; o < n; o += stride) offsets.push(o);

    for (let k = 0; k < offsets.length; k++) {
      const offset = offsets[k];
      const len = Math.min(L, n - offset);
      const delta = L - len;
      const start = offset - Math.floor(delta / 2);
      const chunk = channels.map((ch) => {
        const x = new Float32Array(L);
        for (let i = 0; i < L; i++) {
          const j = start + i;
          if (j >= 0 && j < n) x[i] = (ch[j] - mean) / std;
        }
        return x;
      });
      const out = await this.runChunk(chunk);
      const trim = Math.floor(delta / 2);
      for (let s = 0; s < sources.length; s++) {
        for (let c = 0; c < C; c++) {
          const src = out[s][c], dst = acc[s][c];
          for (let i = 0; i < len; i++) dst[offset + i] += weight[i] * src[trim + i];
        }
      }
      for (let i = 0; i < len; i++) sumW[offset + i] += weight[i];
      onProgress((k + 1) / offsets.length);
    }
    const result = {};
    sources.forEach((name, s) => {
      result[name] = acc[s].map((x) => {
        for (let i = 0; i < n; i++) x[i] = (x[i] / sumW[i]) * std + mean;
        return x;
      });
    });
    return result;
  }
}

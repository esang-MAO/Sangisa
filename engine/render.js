// Stage 5 in the browser: cut each kept slice from its stem and encode it as a WAV.

export function nearestZeroCrossing(mono, index, window) {
  const lo = Math.max(index - window, 1), hi = Math.min(index + window, mono.length - 1);
  if (hi <= lo) return index;
  const neg = (v) => v < 0 || Object.is(v, -0);
  let best = index, bestD = Infinity;
  for (let i = lo; i <= hi; i++) {
    if (neg(mono[i - 1]) !== neg(mono[i])) {
      const d = Math.abs(i - index);
      if (d < bestD) { bestD = d; best = i; }
    }
  }
  return best;
}

/** Mono view of the stereo samples around `index`, for finding zero crossings. */
function monoAround(stem, index, window) {
  const lo = Math.max(0, index - window - 1), hi = Math.min(stem[0].length, index + window + 1);
  const out = new Float32Array(hi - lo);
  for (const ch of stem) for (let i = lo; i < hi; i++) out[i - lo] += ch[i] / stem.length;
  return { lo, mono: out };
}

function zeroCrossing(stem, index, window) {
  const { lo, mono } = monoAround(stem, index, window);
  return lo + nearestZeroCrossing(mono, index - lo, window);
}

/** Render one slice from a stereo stem [L, R]. Returns [L, R] Float32Arrays. */
export function renderSlice(stem, sr, s, cfg) {
  const r = cfg.render;
  const window = Math.floor((sr * r.zero_crossing_window_ms) / 1000);
  let start = Math.round(s.source_start_s * sr);
  let end = Math.round(s.source_end_s * sr);
  if (s.render.playback === "loop") {
    const length = end - start;
    start = zeroCrossing(stem, start, window);
    end = Math.min(start + length, stem[0].length);
  } else {
    start = zeroCrossing(stem, start, window);
    end = zeroCrossing(stem, end, window);
  }
  const out = stem.map((ch) => Float32Array.from(ch.subarray(start, Math.max(start, end))));
  if (s.render.reverse) out.forEach((ch) => ch.reverse());
  const n = out[0].length;
  const fi = Math.min(Math.floor((sr * s.render.fade_in_ms) / 1000), n >> 1);
  const fo = Math.min(Math.floor((sr * s.render.fade_out_ms) / 1000), n >> 1);
  for (const ch of out) {
    for (let i = 0; i < fi; i++) ch[i] *= fi > 1 ? i / (fi - 1) : 0;
    for (let i = 0; i < fo; i++) ch[n - fo + i] *= fo > 1 ? 1 - i / (fo - 1) : 0;
  }
  if (s.render.normalize_dbfs != null) {
    let peak = 0;
    for (const ch of out) for (const v of ch) peak = Math.max(peak, Math.abs(v));
    if (peak > 1e-6) {
      const g = 10 ** (s.render.normalize_dbfs / 20) / peak;
      for (const ch of out) for (let i = 0; i < n; i++) ch[i] *= g;
    }
  }
  return out;
}

/** 24-bit PCM WAV (what the Python pipeline writes). */
export function encodeWav(channels, sr) {
  const nCh = channels.length;
  const n = channels[0].length;
  const bytes = 3;
  const dataLen = n * nCh * bytes;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + dataLen, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, nCh, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * nCh * bytes, true); v.setUint16(32, nCh * bytes, true);
  v.setUint16(34, 24, true); str(36, "data"); v.setUint32(40, dataLen, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nCh; c++) {
      let s = Math.max(-1, Math.min(1, channels[c][i]));
      let x = Math.round(s * 8388607);
      if (x < 0) x += 0x1000000;
      v.setUint8(o, x & 0xff); v.setUint8(o + 1, (x >> 8) & 0xff); v.setUint8(o + 2, (x >> 16) & 0xff);
      o += 3;
    }
  }
  return buf;
}

/** Minimal WAV reader (PCM 16/24/32-bit int, 32-bit float), used by tests and as a decode fallback. */
export function decodeWav(buf) {
  const v = new DataView(buf);
  const tag = (o) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("Not a WAV file");
  let o = 12, fmt = null, data = null;
  while (o + 8 <= buf.byteLength) {
    const id = tag(o), size = v.getUint32(o + 4, true);
    if (id === "fmt ") fmt = { format: v.getUint16(o + 8, true), ch: v.getUint16(o + 10, true), sr: v.getUint32(o + 12, true), bits: v.getUint16(o + 22, true) };
    if (id === "data") data = { offset: o + 8, size };
    o += 8 + size + (size & 1);
  }
  if (!fmt || !data) throw new Error("Malformed WAV file");
  const { ch, sr, bits, format } = fmt;
  const bps = bits / 8;
  const n = Math.floor(data.size / (bps * ch));
  const out = Array.from({ length: ch }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < ch; c++) {
      const p = data.offset + (i * ch + c) * bps;
      let s;
      if (format === 3) s = v.getFloat32(p, true);
      else if (bits === 16) s = v.getInt16(p, true) / 32768;
      else if (bits === 24) { let x = v.getUint8(p) | (v.getUint8(p + 1) << 8) | (v.getUint8(p + 2) << 16); if (x & 0x800000) x -= 0x1000000; s = x / 8388608; }
      else if (bits === 32) s = v.getInt32(p, true) / 2147483648;
      else throw new Error(`Unsupported WAV bit depth ${bits}`);
      out[c][i] = s;
    }
  }
  return { channels: out, sr };
}

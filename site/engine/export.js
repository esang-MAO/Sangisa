// Export a kit as numbered WAVs: the format Koala Sampler (and any DAW or sampler) imports.
// Koala has no preset files: you select a folder's samples and drop them on the first empty pad,
// and they fill the pads in file order. So the zero-padded number *is* the pad assignment.
//
//   <Kit> - 120bpm Fmin/
//     Bank_A/01_Bass_Loop_B_2_bar.wav … 16_Perc_1.wav
//     Bank_B/…                            (32- and 64-pad kits)
//     Extras/Kick_3.wav …                 (optional: the backup sounds)
//     README.txt                          (BPM, key, pad map, import steps)

import { decodeWav } from "./render.js";

export const ORDERS = {
  // Koala and most phone samplers fill left to right from the top row.
  "top-first": "Top row first",
  // Ableton Move/Note and Drum Racks, MPCs: pad 1 is bottom-left.
  "bottom-first": "Bottom row first",
};

const clean = (s) => String(s).replace(/[()]/g, "").replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "");

/** File number (1-16) for a pad, where pad 1 is bottom-left and pads count left to right, bottom to top. */
export function fileNumber(padIndexInBank, order) {
  if (order === "bottom-first") return padIndexInBank + 1;
  const row = Math.floor(padIndexInBank / 4); // 0 = bottom row
  const col = padIndexInBank % 4;
  return (3 - row) * 4 + col + 1;
}

export function folderName(kit) {
  const a = kit.analysis || {};
  const bits = [a.bpm ? `${Math.round(a.bpm)}bpm` : null, a.key_short].filter(Boolean).join(" ");
  return clean(kit.kit_name || "Sangisa Kit").replace(/_/g, " ") + (bits ? ` - ${bits}` : "");
}

/** Which slice goes to which file. Empty pads get a short silent file so numbering stays aligned. */
export function exportPlan(kit, { order = "top-first", extras = false } = {}) {
  const byId = new Map(kit.slices.map((s) => [s.id, s]));
  const entries = [];
  kit.pads.forEach((pad, i) => {
    const bank = String.fromCharCode(65 + Math.floor(i / 16));
    const n = fileNumber(i % 16, order);
    const slice = pad.slice_id ? byId.get(pad.slice_id) : null;
    const name = `${String(n).padStart(2, "0")}_${slice ? clean(slice.label) : "Empty"}.wav`;
    entries.push({ path: `Bank_${bank}/${name}`, bank, number: n, pad: pad.pad, slice });
  });
  entries.sort((a, b) => (a.bank === b.bank ? a.number - b.number : a.bank < b.bank ? -1 : 1));
  if (extras) {
    const onPads = new Set(kit.pads.map((p) => p.slice_id));
    const used = new Set();
    for (const s of kit.slices) {
      if (onPads.has(s.id)) continue;
      let name = clean(s.label);
      while (used.has(name)) name += "_";
      used.add(name);
      entries.push({ path: `Extras/${name}.wav`, bank: null, number: null, pad: null, slice: s });
    }
  }
  return entries;
}

// ---------------------------------------------------------------- audio formats

const gcd = (a, b) => (b ? gcd(b, a % b) : a);

/**
 * Band-limited sample-rate conversion: Blackman-windowed sinc, 24 zero crossings a side,
 * evaluated as a polyphase filter (44.1 -> 48 kHz has 160 phases, all precomputed).
 */
export function resample(channels, fromSr, toSr) {
  if (fromSr === toSr) return channels;
  const g = gcd(fromSr, toSr);
  const L = toSr / g, M = fromSr / g; // output i sits at input position i * M / L
  const zeros = 24;
  const cutoff = Math.min(1, toSr / fromSr) * 0.97;
  const half = Math.ceil(zeros / cutoff);
  const taps = 2 * half;
  const kernel = (d) => {
    if (Math.abs(d) >= half) return 0;
    const a = Math.PI * d * cutoff;
    const sinc = d === 0 ? 1 : Math.sin(a) / a;
    return sinc * (0.42 + 0.5 * Math.cos((Math.PI * d) / half) + 0.08 * Math.cos((2 * Math.PI * d) / half));
  };
  // table[phase][k] for input sample floor(t) - half + 1 + k, normalised to unity gain
  const table = Array.from({ length: L }, (_, phase) => {
    const frac = phase / L;
    const row = new Float64Array(taps);
    let sum = 0;
    for (let k = 0; k < taps; k++) { row[k] = kernel(k - half + 1 - frac); sum += row[k]; }
    for (let k = 0; k < taps; k++) row[k] /= sum;
    return row;
  });
  return channels.map((x) => {
    const n = Math.round((x.length * L) / M);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const pos = i * M;
      const base = Math.floor(pos / L);
      const row = table[pos % L];
      let acc = 0;
      for (let k = 0, j = base - half + 1; k < taps; k++, j++) {
        if (j >= 0 && j < x.length) acc += x[j] * row[k];
      }
      out[i] = acc;
    }
    return out;
  });
}

/** PCM WAV: 16- or 24-bit integer, or 32-bit float. */
export function encodePcm(channels, sr, bits = 24) {
  const nCh = channels.length, n = channels[0].length;
  const float = bits === 32;
  const bps = bits / 8;
  const dataLen = n * nCh * bps;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + dataLen, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, float ? 3 : 1, true); v.setUint16(22, nCh, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * nCh * bps, true); v.setUint16(32, nCh * bps, true);
  v.setUint16(34, bits, true); str(36, "data"); v.setUint32(40, dataLen, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nCh; c++) {
      const s = Math.max(-1, Math.min(1, channels[c][i]));
      if (float) v.setFloat32(o, s, true);
      else if (bits === 16) v.setInt16(o, Math.round(s * 32767), true);
      else {
        let x = Math.round(s * 8388607);
        if (x < 0) x += 0x1000000;
        v.setUint8(o, x & 0xff); v.setUint8(o + 1, (x >> 8) & 0xff); v.setUint8(o + 2, (x >> 16) & 0xff);
      }
      o += bps;
    }
  }
  return buf;
}

const silence = (sr) => [new Float32Array(Math.round(sr * 0.05)), new Float32Array(Math.round(sr * 0.05))];

export function readme(kit, entries, { order, sampleRate, bits }) {
  const a = kit.analysis || {};
  const lines = [
    `${kit.kit_name}`,
    "=".repeat(String(kit.kit_name).length),
    "",
    `Tempo: ${a.bpm ? a.bpm.toFixed(1) + " BPM" : "unknown"}    Key: ${a.key || "unknown"}`,
    `Audio: ${bits === 32 ? "32-bit float" : `${bits}-bit`} WAV, ${sampleRate / 1000} kHz, stereo`,
    `Pad order: ${ORDERS[order]} (file 01 is the ${order === "top-first" ? "top-left" : "bottom-left"} pad)`,
    "",
  ];
  const banks = [...new Set(entries.filter((e) => e.bank).map((e) => e.bank))];
  for (const bank of banks) {
    lines.push(`Bank ${bank} (as laid out on the pads):`, "");
    const byPad = new Map(entries.filter((e) => e.bank === bank).map((e) => [((e.pad - 1) % 16) + 1, e]));
    for (let row = 3; row >= 0; row--) {
      const cells = [];
      for (let col = 0; col < 4; col++) {
        const e = byPad.get(row * 4 + col + 1);
        cells.push(e ? `${String(e.number).padStart(2, "0")} ${e.slice ? e.slice.label : "(empty)"}`.slice(0, 24).padEnd(24) : "".padEnd(24));
      }
      lines.push("  " + cells.join(" | "));
    }
    lines.push("");
  }
  lines.push(
    "Loading into Koala Sampler",
    "--------------------------",
    "1. In the Files app, tap the .zip to unzip it.",
    "2. In Koala, open the sample browser and go to the Bank_A folder (add the folder as a location if needed).",
    "3. Select all the files and drag them onto the first empty pad. They fill the pads in number order.",
    "   Repeat with Bank_B, C, D on the next banks for 32/64-pad kits.",
    "",
    "Other samplers and DAWs: the numbers are the pad order; loops are trimmed to whole bars at the tempo above.",
    "",
    "Made with Sangisa. Releasing music with uncleared samples can require permission from the rights holders.",
  );
  return lines.join("\n") + "\n";
}

/**
 * Build every file of the export. readSlice(slice) returns the slice's rendered WAV (ArrayBuffer).
 * Returns {folder, files: [{path, data: ArrayBuffer|string}]}.
 */
export async function buildExport(kit, readSlice, { order = "top-first", sampleRate = 48000, bits = 24, extras = false, onProgress } = {}) {
  const entries = exportPlan(kit, { order, extras });
  const folder = folderName(kit);
  const files = [];
  let done = 0;
  for (const e of entries) {
    let channels, sr;
    if (e.slice) {
      ({ channels, sr } = decodeWav(await readSlice(e.slice)));
      channels = resample(channels, sr, sampleRate);
    } else {
      channels = silence(sampleRate);
    }
    files.push({ path: `${folder}/${e.path}`, data: encodePcm(channels, sampleRate, bits) });
    onProgress?.(++done / entries.length);
  }
  files.push({ path: `${folder}/README.txt`, data: readme(kit, entries, { order, sampleRate, bits }) });
  return { folder, files };
}

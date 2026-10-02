// Parity tests: the in-browser engine (site/engine) against librosa / the Python pipeline.
// Run through pytest (tests/test_engine_js.py), which writes the reference files first:
//   SANGISA_REF_DIR=<dir> node --test tests/engine/

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { analyze, estimateTempo, beatTrack, onsetStrength } from "../../site/engine/analyze.js";
import { chromaFilters, fft, istft, melFilters, mfcc, stft } from "../../site/engine/dsp.js";
import { classifyDrum, nonSilent, onsets } from "../../site/engine/extract.js";
import { makeKit, withPadSplit } from "../../site/engine/pipeline.js";
import { decodeWav } from "../../site/engine/render.js";

const DIR = process.env.SANGISA_REF_DIR;
if (!DIR) throw new Error("Set SANGISA_REF_DIR (run via pytest tests/test_engine_js.py)");
const ref = JSON.parse(readFileSync(join(DIR, "reference.json"), "utf8"));
const cfg = JSON.parse(readFileSync(new URL("../../site/engine/config.json", import.meta.url), "utf8"));
const load = (name) => {
  const b = readFileSync(join(DIR, `${name}.wav`));
  return decodeWav(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const mix = load("mix");
const SR = mix.sr;
const mono = mix.channels[0];
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (±${tol})`);

test("FFT round trip and STFT reconstruction", () => {
  const re = Float64Array.from({ length: 1024 }, (_, i) => Math.sin(i / 7) + (i % 13) / 13);
  const im = new Float64Array(1024);
  const orig = Float64Array.from(re);
  fft(re, im); fft(re, im, true);
  for (let i = 0; i < 1024; i++) close(re[i], orig[i], 1e-9, "fft");
  const y = mono.subarray(0, SR);
  const back = istft(stft(y, { nFft: 2048, hop: 1024 }), y.length);
  let err = 0;
  for (let i = 0; i < y.length; i++) err = Math.max(err, Math.abs(back[i] - y[i]));
  assert.ok(err < 1e-4, `istft error ${err}`);
});

test("mel and chroma filterbanks match librosa", () => {
  const mel = melFilters(SR, 2048, 128);
  close(mel.reduce((a, b) => a + b, 0), ref.mel_sum, 1e-3 * ref.mel_sum, "mel sum");
  const row = mel.subarray(10 * 1025, 11 * 1025);
  ref.mel_row_10.forEach((v, i) => close(row[i], v, 1e-6, `mel[10][${i}]`));
  const ch = chromaFilters(SR, 4096);
  close(ch.reduce((a, b) => a + b, 0), ref.chroma_sum, 1e-3 * ref.chroma_sum, "chroma sum");
  ref.chroma_row_0_first_400.forEach((v, i) => close(ch[i], v, 1e-5, `chroma[0][${i}]`));
});

test("MFCC and onset strength match librosa", () => {
  const short = mono.subarray(0, SR * 4);
  const m = mfcc(short, SR, { nMfcc: 20 });
  for (let k = 0; k < 20; k++) {
    let s = 0;
    for (let t = 0; t < m.nFrames; t++) s += m.data[t * 20 + k];
    close(s / m.nFrames, ref.mfcc_4s_mean[k], 0.05 * Math.abs(ref.mfcc_4s_mean[k]) + 0.5, `mfcc ${k}`);
  }
  const env = onsetStrength(short, SR);
  assert.equal(env.length, ref.onset_env_4s.length);
  let num = 0, da = 0, db = 0;
  env.forEach((v, i) => { const w = ref.onset_env_4s[i]; num += v * w; da += v * v; db += w * w; });
  assert.ok(num / Math.sqrt(da * db) > 0.999, "onset envelope shape");
});

test("tempo and beats match librosa", () => {
  const env = onsetStrength(mono, SR);
  const tempo = estimateTempo(env, SR);
  close(tempo, ref.tempo, 0.5, "tempo");
  const beats = beatTrack(env, SR, tempo);
  const refBeats = new Set(ref.beat_frames);
  const matched = beats.filter((b) => refBeats.has(b) || refBeats.has(b - 1) || refBeats.has(b + 1)).length;
  assert.ok(matched >= 0.9 * ref.beat_frames.length, `beats matched ${matched}/${ref.beat_frames.length}`);
});

test("onsets and silence split match librosa", () => {
  const drums = load("drums").channels[0];
  const got = onsets(drums, SR);
  const want = new Set(ref.drum_onsets);
  const matched = got.filter((s) => want.has(s) || want.has(s - 512) || want.has(s + 512)).length;
  assert.ok(matched >= 0.9 * ref.drum_onsets.length && got.length <= 1.1 * ref.drum_onsets.length,
    `onsets ${matched}/${ref.drum_onsets.length} (got ${got.length})`);
  assert.deepEqual(nonSilent(drums, 35), ref.split);
});

test("analysis agrees with the Python pipeline", () => {
  const a = analyze(mono, SR, cfg);
  const p = ref.analysis;
  close(a.bpm, p.bpm, 0.5, "bpm");
  assert.equal(a.key, p.key);
  close(a.downbeats_s[0] % 2, p.downbeats_s[0] % 2, 0.05, "downbeat phase");
  assert.equal(a.sections[0].start_s, 0);
});

test("drum classifier", () => {
  const drums = load("drums").channels[0];
  // the synthetic song puts a kick on beat 1 (t = 0) and a snare on beat 2 (t = 0.5 s)
  assert.equal(classifyDrum(drums.subarray(0, Math.floor(SR * 0.08)), SR), "kick");
  assert.equal(classifyDrum(drums.subarray(Math.floor(SR * 0.5), Math.floor(SR * 0.58)), SR), "snare");
});

test("full kit from the real stems", async () => {
  const stems = Object.fromEntries(["drums", "bass", "vocals", "other"].map((n) => [n, load(n).channels]));
  const { kit, files } = await makeKit({ channels: mix.channels, sr: SR, name: "Song.wav", cfg, stems });
  assert.equal(kit.pads.length, 16);
  const byId = new Map(kit.slices.map((s) => [s.id, s]));
  const padSlices = kit.pads.map((p) => byId.get(p.slice_id));
  assert.ok(padSlices.every(Boolean), "every pad filled");
  assert.deepEqual(padSlices.map((s) => s.stem), [...Array(6).fill("drums"), ...Array(4).fill("vocals"), ...Array(3).fill("bass"), ...Array(3).fill("other")]);
  assert.deepEqual(padSlices.slice(0, 3).map((s) => s.category), ["kick", "snare", "hat"]);
  assert.equal(kit.kit_name, "Song Kit");
  assert.ok(kit.slices.length > 16);
  assert.equal(new Set(kit.slices.map((s) => s.file)).size, kit.slices.length);
  for (const s of kit.slices) {
    const wav = decodeWav(files[s.file]);
    assert.equal(wav.sr, SR);
    if (s.render.playback === "loop") close(wav.channels[0].length, s.beats * 60 / kit.analysis.bpm * SR, 3, `loop ${s.label}`);
    assert.ok(s.file.includes("120bpm_Fmin"), s.file);
  }
  const notes = kit.slices.filter((s) => s.category === "bass_note" && s.note).map((s) => s.note);
  assert.ok(notes.length && notes.every((n) => ["F2", "Ab2", "C3", "Eb2"].includes(n)), `bass notes ${notes}`);
});

test("quick mode makes a kit from the mix alone", async () => {
  const { kit } = await makeKit({ channels: mix.channels, sr: SR, name: "Song.wav", cfg: withPadSplit(cfg, { drums: 8, bass: 4, other: 4 }) });
  assert.deepEqual(Object.keys(kit.separation.stems).sort(), ["bass", "drums", "other"]);
  const byId = new Map(kit.slices.map((s) => [s.id, s]));
  const pads = kit.pads.map((p) => byId.get(p.slice_id));
  assert.ok(pads.every(Boolean));
  assert.deepEqual(pads.map((s) => s.stem), [...Array(8).fill("drums"), ...Array(4).fill("bass"), ...Array(4).fill("other")]);
  assert.equal(pads[0].category, "kick");
});

// Numbered-WAV export (Koala Sampler and any DAW).
import assert from "node:assert/strict";
import { test } from "node:test";

import { buildExport, encodePcm, exportPlan, fileNumber, folderName, resample } from "../../site/engine/export.js";
import { decodeWav, encodeWav } from "../../site/engine/render.js";

const LABELS = ["Kick 1", "Snare 1", "Hat 1", "Perc 1", "Drum loop A (2 bar)", "Kick 2", "Vox chop 1 F5", "Vox phrase 1",
  "Vox chop 2", "Vox phrase 2", "Bass loop A (1 bar)", "Bass note 1 Ab2", "Bass loop B (2 bar)", "Other loop A (4 bar)",
  "Other stab 1 Cmin", "Other loop B (2 bar)"];

function fakeKit(nPads = 16) {
  const slices = Array.from({ length: nPads + 3 }, (_, i) => ({
    id: `s${i}`, label: i < 16 ? LABELS[i] : `Extra ${i}`, file: `slices/s${i}.wav`, stem: "drums",
    render: { playback: "one_shot" },
  }));
  return {
    kit_name: "My Song Kit",
    analysis: { bpm: 92.04, key: "F minor", key_short: "Fmin" },
    pads: Array.from({ length: nPads }, (_, i) => ({ pad: i + 1, slice_id: i === 5 ? null : `s${i}` })),
    slices,
  };
}

const tone = (sr, secs, hz) => {
  const x = Float32Array.from({ length: Math.round(sr * secs) }, (_, i) => 0.5 * Math.sin((2 * Math.PI * hz * i) / sr));
  return [x, x];
};

test("file numbers follow the chosen pad order", () => {
  // pad 1 is bottom-left; Koala fills from the top-left
  assert.deepEqual([0, 3, 12, 15].map((p) => fileNumber(p, "top-first")), [13, 16, 1, 4]);
  assert.deepEqual([0, 3, 12, 15].map((p) => fileNumber(p, "bottom-first")), [1, 4, 13, 16]);
});

test("plan names files by number and label, banks in folders, empty pads kept", () => {
  const kit = fakeKit(32);
  const plan = exportPlan(kit, { order: "top-first", extras: true });
  const bankA = plan.filter((e) => e.bank === "A");
  assert.equal(bankA.length, 16);
  assert.deepEqual(bankA.map((e) => e.number), Array.from({ length: 16 }, (_, i) => i + 1));
  assert.equal(bankA.find((e) => e.pad === 1).path, "Bank_A/13_Kick_1.wav");
  assert.equal(bankA.find((e) => e.pad === 13).path, "Bank_A/01_Bass_loop_B_2_bar.wav");
  assert.equal(bankA.find((e) => e.pad === 6).path, "Bank_A/10_Empty.wav");
  assert.equal(plan.filter((e) => e.bank === "B").length, 16);
  assert.ok(plan.some((e) => e.path.startsWith("Extras/")));
  assert.equal(folderName(kit), "My Song Kit - 92bpm Fmin");
});

test("resampling keeps pitch and length", () => {
  const [x] = tone(44100, 1, 440);
  const [y] = resample([x], 44100, 48000);
  assert.equal(y.length, 48000);
  // zero crossings per second stay ~880 (440 Hz)
  let zc = 0;
  for (let i = 1000; i < 47000; i++) if ((y[i - 1] < 0) !== (y[i] < 0)) zc++;
  assert.ok(Math.abs(zc / (46000 / 48000) - 880) < 4, `zero crossings ${zc}`);
  let err = 0;
  for (let i = 1000; i < 47000; i++) err = Math.max(err, Math.abs(y[i] - 0.5 * Math.sin((2 * Math.PI * 440 * i) / 48000)));
  assert.ok(err < 2e-3, `resample error ${err}`);
});

test("PCM encodings round-trip", () => {
  const ch = tone(48000, 0.1, 1000);
  for (const bits of [16, 24, 32]) {
    const { channels, sr } = decodeWav(encodePcm(ch, 48000, bits));
    assert.equal(sr, 48000);
    let err = 0;
    for (let i = 0; i < ch[0].length; i++) err = Math.max(err, Math.abs(channels[0][i] - ch[0][i]));
    assert.ok(err < (bits === 16 ? 1e-4 : 1e-6), `${bits}-bit error ${err}`);
  }
});

test("export builds every file at the requested format", async () => {
  const kit = fakeKit(16);
  const wav = encodeWav(tone(44100, 0.5, 220), 44100);
  const { folder, files } = await buildExport(kit, async () => wav, { order: "top-first", sampleRate: 48000, bits: 24 });
  assert.equal(folder, "My Song Kit - 92bpm Fmin");
  const wavs = files.filter((f) => f.path.endsWith(".wav"));
  assert.equal(wavs.length, 16);
  const one = decodeWav(wavs.find((f) => f.path.endsWith("13_Kick_1.wav")).data);
  assert.equal(one.sr, 48000);
  assert.equal(one.channels[0].length, 24000);
  const readme = files.find((f) => f.path.endsWith("README.txt")).data;
  assert.match(readme, /92\.0 BPM/);
  assert.match(readme, /13 Kick 1/);
  assert.match(readme, /Koala/);
});

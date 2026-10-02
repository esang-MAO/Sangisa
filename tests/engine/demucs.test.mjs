// HT-Demucs in the browser engine vs demucs' own apply_model (PyTorch), on a small model with
// the same structure (tests/engine/fixtures, made by scripts/export_demucs_onnx.py --tiny --reference).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import * as ort from "onnxruntime-web";

import { Demucs, expandWeights } from "../../site/engine/demucs.js";

const dir = new URL("./fixtures/", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("htdemucs_tiny.json", dir), "utf8"));
const f32 = (name) => {
  const b = readFileSync(new URL(name, dir));
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};

test("separation matches demucs.apply.apply_model", async () => {
  const { samples: n, channels: C, sources: S } = meta.reference;
  const input = f32("htdemucs_tiny.ref_in.f32");
  const want = f32("htdemucs_tiny.ref_out.f32");
  const channels = Array.from({ length: C }, (_, c) => input.slice(c * n, (c + 1) * n));
  const model = await Demucs.create(ort, readFileSync(new URL(meta.graph, dir)), meta,
    readFileSync(new URL(meta.weights, dir)));
  const got = await model.separate(channels);
  assert.deepEqual(Object.keys(got), meta.sources);
  let maxErr = 0, maxRef = 0;
  meta.sources.forEach((name, s) => {
    for (let c = 0; c < C; c++) {
      const g = got[name][c];
      const off = (s * C + c) * n;
      for (let i = 0; i < n; i++) {
        maxErr = Math.max(maxErr, Math.abs(g[i] - want[off + i]));
        maxRef = Math.max(maxRef, Math.abs(want[off + i]));
      }
    }
  });
  assert.ok(maxErr / maxRef < 1e-3, `max error ${maxErr} vs peak ${maxRef}`);
  assert.equal(S, meta.sources.length);
});

test("float16 weights widen to float32 exactly", () => {
  const halves = new Uint16Array([0x0000, 0x3c00, 0xc000, 0x3555, 0x7bff, 0x0001, 0x8000]);
  const f = new Float32Array(expandWeights({ weights_format: "f16" }, new Uint8Array(halves.buffer)).buffer);
  assert.deepEqual([...f.slice(0, 3)], [0, 1, -2]);
  assert.ok(Math.abs(f[3] - 0.33325195) < 1e-7);
  assert.equal(f[4], 65504);
  assert.equal(f[5], 2 ** -24);
  assert.ok(Object.is(f[6], -0));
});

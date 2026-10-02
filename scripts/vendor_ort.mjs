// Copy the ONNX Runtime Web files the site needs into site/vendor/ort/ (not committed: ~27 MB).
// Run after `npm ci`: npm run vendor
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules/onnxruntime-web/dist");
const dest = join(root, "site/vendor/ort");
mkdirSync(dest, { recursive: true });
// The WebGPU build; it also runs on the CPU (WebAssembly) when WebGPU isn't available.
for (const f of ["ort.webgpu.min.mjs", "ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"]) {
  copyFileSync(join(src, f), join(dest, f));
  console.log(`site/vendor/ort/${f}`);
}

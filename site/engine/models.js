// Fetch the AI model once and keep it on the device (Cache Storage), with download progress.

const CACHE = "sangisa-models-v1";

async function cached(url) {
  try {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    return hit ? new Uint8Array(await hit.arrayBuffer()) : null;
  } catch {
    return null; // no Cache Storage (private mode, http on a LAN address): just download
  }
}

async function remember(url, bytes, type) {
  try {
    const cache = await caches.open(CACHE);
    await cache.put(url, new Response(bytes, { headers: { "Content-Type": type, "Content-Length": String(bytes.length) } }));
  } catch { /* storage full or unavailable: it'll download again next time */ }
}

/** GET url as bytes, reporting (received, total) as it streams. */
async function download(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Couldn't download the AI model (${res.status} for ${url.split("/").pop()}).`);
  // Content-Length is the compressed size if the server compresses, so it only drives the progress bar.
  const total = Number(res.headers.get("Content-Length")) || 0;
  if (!res.body) {
    const b = new Uint8Array(await res.arrayBuffer());
    onBytes(b.length, b.length);
    return b;
  }
  const chunks = [];
  const reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onBytes(got, Math.max(total, got));
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out;
}

/** Is a model published at base (e.g. "models/htdemucs/")? Returns its meta, or null. */
export async function modelInfo(base, name = "htdemucs") {
  try {
    const res = await fetch(new URL(`${name}.json`, base), { cache: "no-cache" });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

/**
 * The model's graph and weights, from the device cache or downloaded (then cached).
 * onProgress(fraction, {cached, mb}).
 */
export async function loadModelFiles(base, meta, onProgress = () => {}) {
  // The version (a hash of the files) keeps a re-exported model from mixing with a cached one.
  const v = meta.version ? `?v=${meta.version}` : "";
  const urls = [new URL(meta.graph, base).href + v, new URL(meta.weights, base).href + v];
  const found = await Promise.all(urls.map(cached));
  if (found.every(Boolean)) {
    onProgress(1, { cached: true });
    return { graph: found[0], weights: found[1], cached: true };
  }
  const sizes = [0, 0], got = [0, 0];
  const report = () => {
    const total = sizes[0] + sizes[1];
    onProgress(total ? (got[0] + got[1]) / total : 0, { cached: false, mb: total / 1048576 });
  };
  const files = await Promise.all(urls.map((u, i) => found[i] || download(u, (g, t) => { got[i] = g; sizes[i] = t; report(); })));
  await Promise.all(urls.map((u, i) => (found[i] ? null : remember(u, files[i], "application/octet-stream"))));
  return { graph: files[0], weights: files[1], cached: false };
}

/** Forget the cached model (e.g. to free space). */
export async function forgetModels() {
  try { await caches.delete(CACHE); } catch { /* nothing cached */ }
}

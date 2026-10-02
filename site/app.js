// Sangisa kit viewer: loads a kit (kit.json + slices/) and lets you audition, inspect and swap pads.
// Everything runs in the browser; nothing is uploaded.
"use strict";

const STEM_ORDER = ["drums", "vocals", "bass", "other", "guitar", "piano", "instrumental"];
// Rows from the bottom up, so the bottom row of the keyboard plays the bottom row of pads.
const KEY_ROWS = ["zxcv", "asdf", "qwer", "1234"];
const PART_NAMES = ["isolation", "clarity", "loudness", "loopability", "uniqueness"];

const $ = (sel) => document.querySelector(sel);
const state = {
  kit: null,
  read: null,          // async (relativePath) => ArrayBuffer
  bank: 0,
  selectedPad: 0,      // index into kit.pads
  inspect: null,       // slice id shown in the detail panel
  raw: new Map(),      // slice id -> ArrayBuffer (for downloads)
  buffers: new Map(),  // slice id -> AudioBuffer
  voices: new Map(),   // pad index or slice id -> AudioBufferSourceNode
  ctx: null,
};

// ---------------------------------------------------------------- loading

function repoLinks() {
  const m = location.hostname.match(/^([^.]+)\.github\.io$/);
  const repo = location.pathname.split("/").filter(Boolean)[0];
  const base = m && repo ? `https://github.com/${m[1]}/${repo}` : "https://github.com/esang-MAO/Sangisa";
  return { actions: `${base}/actions/workflows/build-kit.yml` };
}

async function loadDemo() {
  const base = "demo/";
  const res = await fetch(base + "kit.json", { cache: "no-cache" });
  if (!res.ok) throw new Error("The demo kit isn't published yet (the Pages workflow builds it).");
  const kit = await res.json();
  openKit(kit, async (p) => {
    const r = await fetch(base + p);
    if (!r.ok) throw new Error(`Missing ${p}`);
    return r.arrayBuffer();
  });
}

async function loadZip(file) {
  if (!window.JSZip) throw new Error("The zip reader didn't load; check your connection.");
  const zip = await JSZip.loadAsync(file);
  const kitPath = Object.keys(zip.files)
    .filter((n) => n.endsWith("kit.json") && !n.startsWith("__MACOSX"))
    .sort((a, b) => a.length - b.length)[0];
  if (!kitPath) throw new Error("No kit.json in that zip.");
  const prefix = kitPath.slice(0, -"kit.json".length);
  const kit = JSON.parse(await zip.file(kitPath).async("string"));
  openKit(kit, async (p) => {
    const f = zip.file(prefix + p);
    if (!f) throw new Error(`Missing ${p} in the zip`);
    return f.async("arraybuffer");
  });
}

async function loadFolder(fileList) {
  const files = [...fileList];
  const kitFile = files
    .filter((f) => f.name === "kit.json")
    .sort((a, b) => a.webkitRelativePath.length - b.webkitRelativePath.length)[0];
  if (!kitFile) throw new Error("No kit.json in that folder.");
  const prefix = kitFile.webkitRelativePath.slice(0, -"kit.json".length);
  const byPath = new Map(files.map((f) => [f.webkitRelativePath, f]));
  const kit = JSON.parse(await kitFile.text());
  openKit(kit, async (p) => {
    const f = byPath.get(prefix + p);
    if (!f) throw new Error(`Missing ${p} in the folder`);
    return f.arrayBuffer();
  });
}

function openKit(kit, read) {
  if (!kit || !Array.isArray(kit.pads) || !Array.isArray(kit.slices)) throw new Error("That isn't a Sangisa kit.json.");
  stopAll();
  Object.assign(state, {
    kit, read, bank: 0, selectedPad: 0, inspect: kit.pads[0]?.slice_id ?? null,
    raw: new Map(), buffers: new Map(),
  });
  state.byId = new Map(kit.slices.map((s) => [s.id, s]));
  showView("kit");
  renderAll();
  preloadBank();
}

// One of: "welcome" (make or open a kit), "progress" (a kit being made), "kit" (the pad grid).
function showView(name) {
  for (const id of ["welcome", "progress", "kit"]) $(`#${id}`).hidden = id !== name;
  if (name !== "kit") stopAll();
  window.scrollTo({ top: 0 });
}

// ---------------------------------------------------------------- audio

function audioCtx() {
  if (!state.ctx) state.ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (state.ctx.state === "suspended") state.ctx.resume();
  return state.ctx;
}

async function rawFor(slice) {
  if (!state.raw.has(slice.id)) state.raw.set(slice.id, await state.read(slice.file));
  return state.raw.get(slice.id);
}

async function bufferFor(slice) {
  if (!state.buffers.has(slice.id)) {
    const data = await rawFor(slice);
    state.buffers.set(slice.id, await audioCtx().decodeAudioData(data.slice(0)));
  }
  return state.buffers.get(slice.id);
}

async function play(key, slice, onEnd) {
  const ctx = audioCtx();
  const buffer = await bufferFor(slice);
  const loop = slice.render?.playback === "loop";
  const current = state.voices.get(key);
  if (current) {
    stopVoice(key);
    if (loop) return false; // tapping a playing loop stops it
  }
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.loop = loop;
  src.connect(ctx.destination);
  src.onended = () => {
    if (state.voices.get(key) === src) state.voices.delete(key);
    onEnd?.();
  };
  src.start();
  state.voices.set(key, src);
  return true;
}

function stopVoice(key) {
  const v = state.voices.get(key);
  if (!v) return;
  state.voices.delete(key);
  try { v.stop(); } catch { /* already stopped */ }
}

function stopAll() {
  for (const key of [...state.voices.keys()]) stopVoice(key);
  document.querySelectorAll(".pad.playing").forEach((p) => p.classList.remove("playing"));
}

async function triggerPad(index) {
  const pad = state.kit.pads[index];
  const slice = pad && state.byId.get(pad.slice_id);
  if (!slice) return;
  selectPad(index, false);
  const el = document.querySelector(`.pad[data-index="${index}"]`);
  el?.classList.add("hit");
  setTimeout(() => el?.classList.remove("hit"), 90);
  try {
    const started = await play(index, slice, () => el?.classList.remove("playing"));
    el?.classList.toggle("playing", started);
  } catch (e) {
    toast(e.message);
  }
}

// ---------------------------------------------------------------- rendering

const stemVar = (stem) => `var(--${STEM_ORDER.includes(stem) ? stem : "other"})`;
const fmt = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function renderAll() {
  renderHeader();
  renderBanks();
  renderGrid();
  renderDetail();
  renderTimeline();
}

function renderHeader() {
  const { kit } = state;
  const a = kit.analysis || {};
  $("#kit-name").textContent = kit.kit_name || "Untitled kit";
  const bits = [
    a.bpm ? `${Math.round(a.bpm * 10) / 10} BPM` : null,
    a.key || "key unknown",
    kit.source?.duration_s ? fmt(kit.source.duration_s) : null,
    `${kit.pads.length} pads`,
    kit.separation?.model ? `stems: ${kit.separation.model}` : null,
  ].filter(Boolean);
  $("#kit-meta").textContent = bits.join(" · ");
}

function renderBanks() {
  const count = Math.ceil(state.kit.pads.length / 16);
  const el = $("#banks");
  el.innerHTML = "";
  if (count < 2) return;
  for (let b = 0; b < count; b++) {
    const btn = document.createElement("button");
    btn.className = "btn small";
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", String(b === state.bank));
    btn.textContent = String.fromCharCode(65 + b);
    btn.onclick = () => { state.bank = b; renderBanks(); renderGrid(); preloadBank(); };
    el.append(btn);
  }
}

function renderGrid() {
  const grid = $("#grid");
  grid.innerHTML = "";
  const start = state.bank * 16;
  // Top row first on screen: pads 13-16, then 9-12, 5-8, 1-4 (pad 1 bottom-left, like Move and Koala).
  for (let row = 3; row >= 0; row--) {
    for (let col = 0; col < 4; col++) {
      const index = start + row * 4 + col;
      const pad = state.kit.pads[index];
      const slice = pad && state.byId.get(pad.slice_id);
      const el = document.createElement("button");
      el.className = "pad" + (slice ? "" : " empty") + (index === state.selectedPad ? " selected" : "");
      el.dataset.index = index;
      if (state.voices.has(index)) el.classList.add("playing");
      const key = KEY_ROWS[row][col].toUpperCase();
      if (slice) {
        el.style.setProperty("--stem", stemVar(slice.stem));
        el.innerHTML = `<span class="num"><span>${pad.pad}</span><span>${key}</span></span>
          <span class="label">${esc(slice.label)}</span><canvas></canvas>`;
        el.setAttribute("aria-label", `Pad ${pad.pad}: ${slice.label}`);
        drawPadWave(el.querySelector("canvas"), slice);
        el.onpointerdown = (e) => { e.preventDefault(); triggerPad(index); };
        el.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); triggerPad(index); } };
      } else {
        el.innerHTML = `<span class="num"><span>${pad ? pad.pad : ""}</span><span>${key}</span></span>`;
        el.onclick = () => selectPad(index);
      }
      grid.append(el);
    }
  }
}

function selectPad(index, rerenderDetail = true) {
  state.selectedPad = index;
  state.inspect = state.kit.pads[index]?.slice_id ?? null;
  document.querySelectorAll(".pad").forEach((p) => p.classList.toggle("selected", Number(p.dataset.index) === index));
  if (rerenderDetail) renderDetail();
  else queueMicrotask(renderDetail);
  highlightTimeline();
}

function renderDetail() {
  const el = $("#detail");
  const pad = state.kit.pads[state.selectedPad];
  const slice = state.inspect && state.byId.get(state.inspect);
  if (!slice) {
    el.innerHTML = `<p class="empty-detail">Pad ${pad?.pad ?? ""} is empty. Pick a candidate from the timeline below
      and choose <b>Put on pad</b>.</p>`;
    return;
  }
  const onPads = state.kit.pads.filter((p) => p.slice_id === slice.id).map((p) => p.pad);
  const isSelected = pad && pad.slice_id === slice.id;
  el.style.setProperty("--stem", stemVar(slice.stem));
  const rows = [
    ["Stem", slice.stem],
    ["Type", `${slice.kind.replace("_", "-")} · ${slice.category.replace("_", " ")}`],
    slice.note ? ["Note", slice.note] : null,
    slice.bars ? ["Length", `${slice.bars} bar${slice.bars === 1 ? "" : "s"}`] : null,
    ["From", `${fmt(slice.source_start_s)} – ${fmt(slice.source_end_s)} (${(slice.source_end_s - slice.source_start_s).toFixed(2)} s)`],
    slice.section ? ["Section", slice.section] : null,
    ["On pads", onPads.length ? onPads.join(", ") : "backup"],
    ["File", slice.file.split("/").pop()],
  ].filter(Boolean);
  const parts = PART_NAMES.map((k) => [k, slice.score_parts?.[k]]).filter(([, v]) => v !== null && v !== undefined);
  el.innerHTML = `
    <h3><span class="swatch"></span>${esc(slice.label)}</h3>
    <p class="meta">Score ${slice.score.toFixed(2)} · cluster ${slice.cluster}</p>
    <canvas class="big-wave"></canvas>
    <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
    <div class="scores">${parts.map(([k, v]) => `<div class="score-row"><span>${k}</span>
      <span class="bar"><i style="width:${Math.round(v * 100)}%"></i></span><span>${v.toFixed(2)}</span></div>`).join("")}</div>
    <div class="row">
      <button class="btn small" data-act="play">Play</button>
      ${isSelected ? "" : `<button class="btn small primary" data-act="swap">Put on pad ${pad.pad}</button>`}
      <button class="btn small" data-act="download">Download WAV</button>
    </div>`;
  drawWave(el.querySelector("canvas"), slice, 70);
  el.querySelector('[data-act="play"]').onclick = () => play(`inspect:${slice.id}`, slice).catch((e) => toast(e.message));
  el.querySelector('[data-act="download"]').onclick = () => downloadSlice(slice);
  const swap = el.querySelector('[data-act="swap"]');
  if (swap) swap.onclick = () => swapOntoPad(slice.id);
}

function renderTimeline() {
  const { kit } = state;
  const duration = kit.source?.duration_s || Math.max(...kit.slices.map((s) => s.source_end_s));
  const tl = $("#timeline");
  tl.innerHTML = "";
  const sections = kit.analysis?.sections || [];
  if (sections.length > 1) {
    const row = document.createElement("div");
    row.className = "sections";
    row.innerHTML = sections.map((s) => `<span style="left:${(s.start_s / duration) * 100}%">${esc(s.label)}</span>`).join("");
    tl.append(row);
  }
  const stems = [...new Set(kit.slices.map((s) => s.stem))].sort(
    (a, b) => (STEM_ORDER.indexOf(a) + 99) % 99 - (STEM_ORDER.indexOf(b) + 99) % 99,
  );
  for (const stem of stems) {
    const lane = document.createElement("div");
    lane.className = "lane";
    lane.style.setProperty("--stem", stemVar(stem));
    lane.innerHTML = `<div class="lane-name"><span class="swatch"></span>${esc(stem)}</div><div class="lane-track"></div>`;
    const track = lane.querySelector(".lane-track");
    const slices = kit.slices.filter((s) => s.stem === stem).sort((a, b) => b.source_end_s - b.source_start_s - (a.source_end_s - a.source_start_s));
    for (const s of slices) {
      const m = document.createElement("button");
      m.className = "mark";
      m.dataset.id = s.id;
      m.title = `${s.label} (${fmt(s.source_start_s)}, score ${s.score.toFixed(2)})`;
      m.setAttribute("aria-label", m.title);
      m.style.left = `${(s.source_start_s / duration) * 100}%`;
      m.style.width = `${((s.source_end_s - s.source_start_s) / duration) * 100}%`;
      m.onclick = () => {
        state.inspect = s.id;
        renderDetail();
        highlightTimeline();
        play(`inspect:${s.id}`, s).catch((e) => toast(e.message));
      };
      track.append(m);
    }
    tl.append(lane);
  }
  highlightTimeline();
}

function highlightTimeline() {
  const onPad = new Set(state.kit.pads.map((p) => p.slice_id));
  document.querySelectorAll(".mark").forEach((m) => {
    m.classList.toggle("on-pad", onPad.has(m.dataset.id));
    m.classList.toggle("current", m.dataset.id === state.inspect);
  });
}

// ---------------------------------------------------------------- waveforms

function peaks(buffer, n) {
  const data = buffer.getChannelData(0);
  const step = Math.max(1, Math.floor(data.length / n));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let max = 0;
    for (let j = i * step, end = Math.min(data.length, j + step); j < end; j++) {
      const v = Math.abs(data[j]);
      if (v > max) max = v;
    }
    out[i] = max;
  }
  return out;
}

async function drawWave(canvas, slice, cssHeight) {
  try {
    const buffer = await bufferFor(slice);
    if (!canvas.isConnected) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 120;
    const h = cssHeight || canvas.clientHeight || 30;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const g = canvas.getContext("2d");
    g.scale(dpr, dpr);
    g.fillStyle = getComputedStyle(canvas).getPropertyValue("--stem").trim() || getComputedStyle(document.documentElement).getPropertyValue(`--${slice.stem}`);
    const cols = Math.max(8, Math.floor(w / 2));
    const p = peaks(buffer, cols);
    const peak = Math.max(...p, 1e-6);
    for (let i = 0; i < cols; i++) {
      const bar = Math.max(1, (p[i] / peak) * (h - 2));
      g.fillRect((i / cols) * w, (h - bar) / 2, Math.max(1, w / cols - 0.5), bar);
    }
  } catch { /* drawn when the audio arrives */ }
}

function drawPadWave(canvas, slice) {
  if (state.buffers.has(slice.id)) requestAnimationFrame(() => drawWave(canvas, slice));
}

async function preloadBank() {
  const start = state.bank * 16;
  for (let i = start; i < Math.min(start + 16, state.kit.pads.length); i++) {
    const slice = state.byId.get(state.kit.pads[i].slice_id);
    if (!slice) continue;
    try {
      await bufferFor(slice);
      const canvas = document.querySelector(`.pad[data-index="${i}"] canvas`);
      if (canvas) drawWave(canvas, slice);
    } catch (e) {
      toast(e.message);
      return;
    }
  }
}

// ---------------------------------------------------------------- editing and saving

function swapOntoPad(sliceId) {
  const pad = state.kit.pads[state.selectedPad];
  stopVoice(state.selectedPad);
  pad.slice_id = sliceId;
  renderGrid();
  renderDetail();
  highlightTimeline();
  preloadBank();
  toast(`Pad ${pad.pad} is now ${state.byId.get(sliceId).label}. Save kit.json to keep it.`);
}

function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function downloadSlice(slice) {
  try {
    saveBlob(new Blob([await rawFor(slice)], { type: "audio/wav" }), slice.file.split("/").pop());
  } catch (e) {
    toast(e.message);
  }
}

/** The kit as Sangisa saves it (kit.json + slices/), to reopen here later. */
async function downloadKitZip() {
  if (!window.JSZip) throw new Error("The zip writer didn't load.");
  const { kit } = state;
  const zip = new JSZip();
  zip.file("kit.json", JSON.stringify(kit, null, 2) + "\n");
  for (const s of kit.slices) zip.file(s.file, await rawFor(s));
  const blob = await zip.generateAsync({ type: "blob", compression: "STORE" });
  saveBlob(blob, `${(kit.kit_name || "Sangisa kit").replace(/[^\w\- ]+/g, "_")}.zip`);
}

// ---------------------------------------------------------------- export (Koala Sampler, any DAW)

const exportState = { file: null, building: 0 };
const prefs = {
  get(k, d) { try { return localStorage.getItem(`sangisa.export.${k}`) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`sangisa.export.${k}`, v); } catch { /* private mode */ } },
};

function exportOptions() {
  return {
    order: document.querySelector('input[name="ex-order"]:checked')?.value || "top-first",
    sampleRate: Number($("#ex-rate").value),
    bits: Number($("#ex-bits").value),
    extras: $("#ex-extras").checked,
  };
}

function openExport() {
  stopAll();
  const order = prefs.get("order", "top-first");
  document.querySelector(`input[name="ex-order"][value="${order}"]`).checked = true;
  $("#ex-rate").value = prefs.get("rate", "48000");
  $("#ex-bits").value = prefs.get("bits", "24");
  $("#ex-extras").checked = prefs.get("extras", "0") === "1";
  $("#export-dialog").showModal();
  prepareExport();
}

async function prepareExport() {
  const build = ++exportState.building;
  const opts = exportOptions();
  prefs.set("order", opts.order); prefs.set("rate", opts.sampleRate); prefs.set("bits", opts.bits);
  prefs.set("extras", opts.extras ? "1" : "0");
  exportState.file = null;
  $("#ex-share").disabled = $("#ex-download").disabled = true;
  const status = $("#ex-status");
  status.textContent = "Preparing…";
  try {
    if (!window.JSZip) throw new Error("The zip writer didn't load.");
    const { buildExport } = await import("./engine/export.js");
    const { folder, files } = await buildExport(state.kit, rawFor, {
      ...opts,
      onProgress: (f) => { if (build === exportState.building) status.textContent = `Preparing… ${Math.round(f * 100)}%`; },
    });
    if (build !== exportState.building) return; // options changed meanwhile
    const zip = new JSZip();
    for (const f of files) zip.file(f.path, f.data);
    const blob = await zip.generateAsync({ type: "blob", compression: "STORE" });
    if (build !== exportState.building) return;
    exportState.file = new File([blob], `${folder}.zip`, { type: "application/zip" });
    const wavs = files.filter((f) => f.path.endsWith(".wav")).length;
    status.textContent = `Ready: ${folder}.zip · ${wavs} WAVs · ${(blob.size / 1048576).toFixed(1)} MB`;
    $("#ex-download").disabled = false;
    const canShare = navigator.canShare?.({ files: [exportState.file] });
    $("#ex-share").disabled = !canShare;
    $("#ex-share").hidden = !canShare;
  } catch (e) {
    if (build === exportState.building) status.textContent = `Couldn't prepare the export: ${e.message}`;
  }
}

$("#export-dialog").addEventListener("change", (e) => { if (e.target.closest(".export-body")) prepareExport(); });
$("#ex-download").onclick = () => exportState.file && saveBlob(exportState.file, exportState.file.name);
$("#ex-share").onclick = guard(async () => {
  if (!exportState.file) return;
  try {
    await navigator.share({ files: [exportState.file], title: exportState.file.name.replace(/\.zip$/, "") });
  } catch (e) {
    if (e.name !== "AbortError") throw e; // closing the share sheet isn't an error
  }
});
$("#ex-sangisa").onclick = (e) => { e.preventDefault(); guard(downloadKitZip)(); };

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 3500);
}

// ---------------------------------------------------------------- wiring

function guard(fn) {
  return async (...args) => {
    try {
      await fn(...args);
    } catch (e) {
      toast(e.message || String(e));
    }
  };
}

$("#action-link").href = repoLinks().actions;
$("#load-demo").onclick = guard(loadDemo);
$("#zip-input").onchange = guard((e) => e.target.files[0] && loadZip(e.target.files[0]));
$("#dir-input").onchange = guard((e) => e.target.files.length && loadFolder(e.target.files));
$("#stop-all").onclick = stopAll;
$("#back-home").onclick = () => { showView("welcome"); refreshRecent(); };
$("#open-export").onclick = () => openExport();

let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; document.body.classList.add("dragging"); });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", guard(async (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  const files = [...(e.dataTransfer?.files || [])];
  const zip = files.find((f) => /\.zip$/i.test(f.name));
  if (zip) return loadZip(zip);
  const song = files.find(isAudioFile);
  if (song) return chooseSong(song); // make.js
  throw new Error("Drop a song to make a kit, or a kit .zip to open one.");
}));

window.addEventListener("keydown", (e) => {
  if (!state.kit || e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
  if (e.target.closest && e.target.closest("input, textarea")) return;
  const k = e.key.toLowerCase();
  if (k === "escape") return stopAll();
  const row = KEY_ROWS.findIndex((r) => r.includes(k));
  if (row < 0) return;
  const index = state.bank * 16 + row * 4 + KEY_ROWS[row].indexOf(k);
  if (index < state.kit.pads.length) {
    e.preventDefault();
    triggerPad(index);
  }
});

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => state.kit && (renderGrid(), renderDetail()), 150);
});

if (new URLSearchParams(location.search).has("demo")) guard(loadDemo)();

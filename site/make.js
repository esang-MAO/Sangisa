// Making a kit: send a song to Sangisa running on the user's computer (`sangisa serve`),
// follow its progress, then open the finished kit in the viewer (app.js).
"use strict";

const AUDIO_EXT = /\.(wav|wave|aif|aiff|flac|mp3|m4a|aac|ogg|opus)$/i;
const DEFAULT_SERVER = "http://localhost:8765";
const isPhone = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

const server = { base: null, info: null, key: null };
const make = { song: null, tracking: null };

const store = {
  get(k) { try { return localStorage.getItem(`sangisa.${k}`); } catch { return null; } },
  set(k, v) { try { v == null ? localStorage.removeItem(`sangisa.${k}`) : localStorage.setItem(`sangisa.${k}`, v); } catch { /* private mode */ } },
};

function isAudioFile(f) {
  return AUDIO_EXT.test(f.name) || (f.type || "").startsWith("audio/");
}

// ---------------------------------------------------------------- talking to the server

function headers() {
  return server.key ? { "X-Sangisa-Key": server.key } : {};
}

async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(server.base + path, { ...opts, headers: { ...headers(), ...(opts.headers || {}) } });
  } catch {
    setConn("off");
    throw new Error("Lost touch with Sangisa. Is it still running on your computer?");
  }
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch { /* not JSON */ }
    throw new Error(detail || `Sangisa answered ${res.status}`);
  }
  return res;
}

async function probe(base) {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 2500);
    const res = await fetch(base + "/api/health", { headers: headers(), signal: ctrl.signal });
    clearTimeout(timer);
    if (res.status === 401) return { needsKey: true };
    if (!res.ok) return null;
    const info = await res.json();
    return info.app === "sangisa" ? info : null;
  } catch {
    return null;
  }
}

function candidates() {
  const list = [];
  const saved = store.get("server");
  if (saved) list.push(saved);
  // Served by `sangisa serve` itself (computer or phone on the same Wi-Fi).
  if (location.protocol.startsWith("http") && !location.hostname.endsWith("github.io")) list.push(location.origin);
  list.push(DEFAULT_SERVER, "http://127.0.0.1:8765");
  return [...new Set(list)];
}

async function connect() {
  setConn("checking");
  for (const base of candidates()) {
    const info = await probe(base);
    if (info?.needsKey) {
      server.base = null;
      setConn("key");
      return false;
    }
    if (info) {
      Object.assign(server, { base, info });
      if (!store.get("route")) setWhere("computer");
      setConn("on");
      refreshRecent();
      return true;
    }
  }
  server.base = null;
  setConn("off");
  return false;
}

function setConn(stateName) {
  const el = $("#conn");
  el.dataset.state = stateName;
  const where = server.base && new URL(server.base).hostname;
  const local = !where || where === "localhost" || where === "127.0.0.1";
  const device = { cuda: "NVIDIA GPU", mps: "Apple Silicon", cpu: "CPU" }[server.info?.device] || "";
  $("#conn-text").textContent = {
    checking: "Looking for Sangisa…",
    on: `Connected to Sangisa ${local ? "on this computer" : `at ${where}`}${device ? ` · ${device}` : ""}`,
    off: "Sangisa isn't running",
    key: "This device needs the link with the access key",
  }[stateName];
  const ready = stateName === "on";
  if (stateName !== "checking") $("#setup").open = !ready && route() === "computer";
  $("#phone-note").hidden = !isPhone;
  updateStart();
  const hint = $("#make-hint");
  if (route() === "device") {
    hint.textContent = "";
  } else if (stateName === "off") {
    hint.innerHTML = isPhone
      ? "Start Sangisa on your computer with <code>sangisa serve --lan</code>, then scan the code it prints."
      : 'Start Sangisa on your computer (steps below). Already running? Open <a href="http://localhost:8765/">localhost:8765</a>.';
  } else if (stateName === "key") {
    hint.textContent = "Scan the code Sangisa printed on your computer, or open the full link it shows (it ends in ?key=…).";
  } else if (ready && server.info?.device === "cpu") {
    hint.textContent = "No GPU found, so stem separation will take about 10–15 minutes per song.";
  } else {
    hint.textContent = "";
  }
}

// ---------------------------------------------------------------- choosing a song

function chooseSong(file) {
  if (!isAudioFile(file)) throw new Error(`${file.name} doesn't look like an audio file.`);
  make.song = file;
  showView("welcome");
  $("#song-title").textContent = file.name;
  $("#song-sub").textContent = `${(file.size / 1048576).toFixed(1)} MB · tap to choose a different song`;
  $("#song-drop").classList.add("chosen");
  updateStart();
}

// "device" runs the in-browser engine; "computer" sends the song to `sangisa serve`.
function route() {
  return document.querySelector('input[name="where"]:checked')?.value || "device";
}

function setWhere(value) {
  const input = document.querySelector(`input[name="where"][value="${value}"]`);
  if (input) input.checked = true;
}

function updateStart() {
  const local = route() === "device";
  const ok = Boolean(make.song && $("#rights").checked && (local || server.base));
  $("#start").disabled = !ok;
  $("#start").title = !make.song ? "Choose a song first" : !$("#rights").checked ? "Confirm you have the rights" :
    !local && !server.base ? "Start Sangisa on your computer first" : "";
  $("#start").textContent = local ? "Make the kit on this device" : "Make the kit";
}

// ---------------------------------------------------------------- making the kit

function upload(form, onProgress) {
  // XHR rather than fetch, for upload progress (big files over Wi-Fi take a while).
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", server.base + "/api/jobs");
    for (const [k, v] of Object.entries(headers())) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* empty */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.detail || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Upload failed: lost touch with Sangisa."));
    xhr.send(form);
  });
}

async function startJob() {
  if (route() === "device") return startLocal(make.song);
  const song = make.song;
  const form = new FormData();
  form.append("file", song, song.name);
  form.append("rights", "true");
  form.append("pads", $("#pad-split").value);
  store.set("pads", $("#pad-split").value);

  showView("progress");
  $("#prog-title").textContent = song.name;
  $("#prog-sub").textContent = "Sending the song to Sangisa…";
  $("#prog-stages").innerHTML = "";
  $("#prog-note").textContent = "";
  try {
    const rec = await upload(form, (f) => { $("#prog-sub").textContent = `Sending the song to Sangisa… ${Math.round(f * 100)}%`; });
    make.song = null;
    resetSongPicker();
    track(rec.id);
  } catch (e) {
    $("#prog-sub").textContent = e.message;
  }
}

// ---------------------------------------------------------------- making it on this device

const DEVICE_MAX_MINUTES = 7; // keeps memory use safe on phones

async function decodeSong(file) {
  const data = await file.arrayBuffer();
  let sha256 = "";
  try {
    const digest = await crypto.subtle.digest("SHA-256", data);
    sha256 = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  } catch { /* crypto.subtle needs https or localhost */ }
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!Ctx) throw new Error("This browser can't decode audio.");
  // An OfflineAudioContext at 44.1 kHz decodes and resamples in one step.
  const ctx = new Ctx(2, 44100, 44100);
  let buffer;
  try {
    buffer = await ctx.decodeAudioData(data);
  } catch {
    throw new Error(`This browser can't read ${file.name}. Try a WAV or MP3.`);
  }
  if (buffer.duration > DEVICE_MAX_MINUTES * 60) {
    throw new Error(`On this device songs can be up to ${DEVICE_MAX_MINUTES} minutes; this one is ${(buffer.duration / 60).toFixed(1)}.`);
  }
  const channels = [];
  for (let c = 0; c < Math.min(2, buffer.numberOfChannels); c++) channels.push(new Float32Array(buffer.getChannelData(c)));
  return { channels, sr: buffer.sampleRate, sha256 };
}

async function keepAwake() {
  try { return await navigator.wakeLock?.request("screen"); } catch { return null; }
}

async function startLocal(song) {
  const pads = Object.fromEntries($("#pad-split").value.split(",").map((p) => p.split("=")).map(([k, v]) => [k, Number(v)]));
  store.set("pads", $("#pad-split").value);
  const now = () => Date.now() / 1000;
  const rec = {
    name: song.name, status: "running", stage: "ingest", stage_label: "Loading", stage_started: now(),
    started: now(), stages: {}, error: null, local: true,
    order: ["ingest", "separate", "analyze", "pick", "render"],
    labels: { ingest: "Loading", separate: "Separating stems", analyze: "Analyzing tempo and key", pick: "Finding the best moments", render: "Building your kit" },
  };
  showView("progress");
  renderProgress(rec);
  const ticker = setInterval(() => renderProgress(rec), 1000);
  const lock = await keepAwake();
  make.tracking = "local";
  const fail = (msg) => {
    rec.status = "failed"; rec.error = msg;
    renderProgress(rec);
  };
  try {
    const decodeStart = now();
    const { channels, sr, sha256 } = await decodeSong(song);
    if (make.tracking !== "local") return; // the user went back while it was decoding
    const worker = new Worker(new URL("engine/worker.js", document.baseURI), { type: "module" });
    make.worker = worker;
    const result = await new Promise((resolve, reject) => {
      make.cancel = () => reject(new Error("cancelled"));
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "progress") {
          if (m.status === "start") Object.assign(rec, { stage: m.stage, stage_label: rec.labels[m.stage], stage_started: now() });
          if (m.status === "done") rec.stages[m.stage] = m.stage === "ingest" ? now() - decodeStart : m.value;
          renderProgress(rec);
        } else if (m.type === "done") resolve(m);
        else if (m.type === "error") reject(new Error(m.message));
      };
      worker.onerror = (e) => reject(new Error(e.message || "The kit maker stopped unexpectedly (out of memory?)."));
      worker.postMessage({ type: "run", channels, sr, name: song.name, sha256, pads, separator: "quick" }, channels.map((c) => c.buffer));
    });
    worker.terminate();
    rec.status = "done";
    renderProgress(rec);
    make.song = null;
    resetSongPicker();
    if (make.tracking !== "local") return; // the user went back
    openKit(result.kit, async (p) => {
      const f = result.files[p];
      if (!f) throw new Error(`Missing ${p}`);
      return f;
    });
    toast("Made on this device. Use Download kit .zip to keep it.");
  } catch (e) {
    if (e.message !== "cancelled") fail(e.message || String(e));
  } finally {
    make.cancel = null;
    clearInterval(ticker);
    make.worker?.terminate();
    make.worker = null;
    if (make.tracking === "local") make.tracking = null;
    try { await lock?.release(); } catch { /* already released */ }
  }
}

function resetSongPicker() {
  $("#song-title").textContent = isPhone ? "Choose a song" : "Drop a song here";
  $("#song-sub").textContent = "or tap to choose one · WAV, AIFF, FLAC, MP3, M4A · up to 10 minutes";
  $("#song-drop").classList.remove("chosen");
  $("#song-input").value = "";
  updateStart();
}

async function track(id) {
  make.tracking = id;
  showView("progress");
  while (make.tracking === id) {
    let rec;
    try {
      rec = await (await api(`/api/jobs/${id}`)).json();
    } catch (e) {
      $("#prog-note").textContent = e.message;
      await sleep(3000);
      continue;
    }
    renderProgress(rec);
    if (rec.status === "done") {
      make.tracking = null;
      await openServerKit(id);
      return;
    }
    if (rec.status === "failed") {
      make.tracking = null;
      return;
    }
    await sleep(1000);
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const mins = (s) => (s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, "0")} s`);

function renderProgress(rec) {
  $("#prog-title").textContent = rec.name;
  const now = Date.now() / 1000;
  const sub = {
    queued: "Waiting for the song before it to finish…",
    running: `Working… ${mins(Math.max(0, now - (rec.started || now)))} so far`,
    done: "Done! Opening your kit…",
    failed: `That didn't work: ${rec.error}`,
  }[rec.status];
  $("#prog-sub").textContent = sub;
  $("#prog-stages").innerHTML = rec.order.map((stage) => {
    const done = stage in rec.stages;
    const active = rec.stage === stage;
    const failed = rec.status === "failed" && !done && (active || (!rec.stage && stage === rec.order.find((s) => !(s in rec.stages))));
    const status = done ? "done" : failed ? "failed" : active ? "active" : "todo";
    const time = done ? mins(rec.stages[stage]) : active ? mins(Math.max(0, now - (rec.stage_started || now))) : "";
    return `<li data-status="${status}"><span class="dot"></span><span>${esc(rec.labels[stage])}</span><span class="t">${time}</span></li>`;
  }).join("");
  const gpu = server.info && server.info.device !== "cpu";
  if (rec.local) {
    $("#prog-note").textContent = rec.status === "running"
      ? "Working on this device. Keep this page open; on a phone, keep the screen on."
      : "";
    return;
  }
  $("#prog-note").textContent = rec.status === "running" && rec.stage === "separate"
    ? `Separating stems is the slow step: about ${gpu ? "1–3" : "10–15"} minutes. You can leave this page; the kit will be under Your kits.`
    : "";
}

async function openServerKit(id) {
  const kit = await (await api(`/api/jobs/${id}/kit.json`)).json();
  const enc = (p) => p.split("/").map(encodeURIComponent).join("/");
  openKit(kit, async (p) => (await api(`/api/jobs/${id}/files/${enc(p)}`)).arrayBuffer());
}

// ---------------------------------------------------------------- your kits

async function refreshRecent() {
  const list = $("#recent");
  if (!server.base) return;
  let jobs;
  try {
    jobs = await (await api("/api/jobs")).json();
  } catch {
    return;
  }
  if (!jobs.length) {
    list.innerHTML = '<li class="meta">Kits you make show up here.</li>';
    return;
  }
  list.innerHTML = jobs.slice(0, 20).map((j) => {
    const when = new Date(j.created * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    const detail = j.status === "done" && j.kit
      ? `${Math.round(j.kit.bpm)} BPM · ${j.kit.key || "key unknown"}`
      : j.status === "failed" ? `Failed: ${j.error}` : j.status === "queued" ? "Waiting…" : `Working: ${j.stage_label || "starting"}`;
    const action = j.status === "done" ? "Open" : j.status === "failed" ? "" : "Progress";
    return `<li data-id="${j.id}" data-status="${j.status}">
      <div><b>${esc(j.kit?.name || j.name)}</b><span class="meta">${esc(detail)} · ${when}</span></div>
      <div class="row">${action ? `<button class="btn small" data-act="open">${action}</button>` : ""}
        ${j.status === "done" || j.status === "failed" ? '<button class="btn small" data-act="delete" aria-label="Delete">✕</button>' : ""}</div>
    </li>`;
  }).join("");
}

$("#recent").addEventListener("click", guard(async (e) => {
  const btn = e.target.closest("button");
  const li = e.target.closest("li[data-id]");
  if (!btn || !li) return;
  const id = li.dataset.id;
  if (btn.dataset.act === "open") {
    if (li.dataset.status === "done") await openServerKit(id);
    else track(id);
  } else if (btn.dataset.act === "delete") {
    if (!confirm("Delete this kit from your computer?")) return;
    await api(`/api/jobs/${id}`, { method: "DELETE" });
    refreshRecent();
  }
}));

// ---------------------------------------------------------------- wiring

(function init() {
  const params = new URLSearchParams(location.search);
  if (params.has("key")) {
    store.set("key", params.get("key"));
    params.delete("key");
    const rest = params.toString();
    history.replaceState(null, "", location.pathname + (rest ? `?${rest}` : "") + location.hash);
  }
  server.key = store.get("key");
  if (isPhone) resetSongPicker();

  const savedPads = store.get("pads");
  if (savedPads && [...$("#pad-split").options].some((o) => o.value === savedPads)) $("#pad-split").value = savedPads;
  $("#server-url").value = store.get("server") || "";

  $("#song-input").onchange = guard((e) => e.target.files[0] && chooseSong(e.target.files[0]));
  $("#rights").onchange = updateStart;
  setWhere(store.get("route") || "device");
  document.querySelectorAll('input[name="where"]').forEach((r) => r.addEventListener("change", () => {
    store.set("route", route());
    setConn($("#conn").dataset.state);
  }));
  $("#start").onclick = guard(startJob);
  $("#prog-back").onclick = () => {
    if (make.tracking === "local") make.cancel?.();
    make.tracking = null;
    showView("welcome");
    refreshRecent();
  };
  $("#server-form").onsubmit = guard(async (e) => {
    e.preventDefault();
    const url = $("#server-url").value.trim().replace(/\/+$/, "");
    store.set("server", url || null);
    if (!(await connect())) toast(url ? `Couldn't reach Sangisa at ${url}.` : "Couldn't find Sangisa.");
  });
  // Check again when the user comes back to the tab (e.g. after starting Sangisa).
  document.addEventListener("visibilitychange", () => document.visibilityState === "visible" && !server.base && connect());
  connect();
})();

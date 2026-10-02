// Making a kit: send a song to Sangisa running on the user's computer (`sangisa serve`),
// follow its progress, then open the finished kit in the viewer (app.js).
"use strict";

const AUDIO_EXT = /\.(wav|wave|aif|aiff|flac|mp3|m4a|aac|ogg|opus)$/i;
const DEFAULT_SERVER = "http://localhost:8765";
const isPhone = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

const MODEL_BASE = "models/htdemucs/";

const server = { base: null, info: null, key: null };
const make = { song: null, tracking: null };
const ai = { meta: null }; // the on-device AI model's description, when this site has it

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

function mode() {
  return document.querySelector('input[name="mode"]:checked')?.value || "kit";
}

function engine() {
  return ai.meta ? document.querySelector('input[name="engine"]:checked')?.value || "ai" : "quick";
}

function setEngine(value) {
  const input = document.querySelector(`input[name="engine"][value="${value}"]`);
  if (input) input.checked = true;
}

/**
 * Show the options for the chosen mode and place. Acapella + instrumental needs a vocal stem, so on
 * this device it needs the AI model (and without the model it runs on the computer).
 */
function updateMode() {
  const split = mode() === "split";
  $("#make-title").textContent = split ? "Split a song" : "Make a kit";
  $("#split-opts").hidden = !split;
  $("#pads-row").hidden = split;
  const device = document.querySelector('input[name="where"][value="device"]');
  const noDevice = split && !ai.meta;
  device.disabled = noDevice;
  device.closest(".choice").classList.toggle("disabled", noDevice);
  if (noDevice && route() === "device") setWhere("computer");
  if (!noDevice && store.get("route") === "device" && route() !== "device") setWhere("device");

  const local = route() === "device";
  const aiInput = document.querySelector('input[name="engine"][value="ai"]');
  aiInput.disabled = !ai.meta;
  $("#engine-ai").classList.toggle("disabled", !ai.meta);
  $("#engine-ai-meta").textContent = ai.meta
    ? `Drums, bass, vocals and the rest, cleanly separated. Downloads the model once (${Math.round(ai.meta.weights_mb)} MB). Fast with a GPU; several minutes on older phones.`
    : "Not set up on this site yet (the model hasn't been published).";
  if (!ai.meta) setEngine("quick");
  else if (split) setEngine("ai");
  $("#engine").hidden = !local || split;
  // On this device the split is always 24-bit WAV of the AI model's vocals.
  for (const id of ["#split-format-row", "#split-normalize-row", "#split-fast-row"]) $(id).hidden = local;
  $("#hq-row").hidden = split || local;
  setConn($("#conn").dataset.state);
}

function updateStart() {
  const local = route() === "device";
  const ok = Boolean(make.song && $("#rights").checked && (local || server.base));
  $("#start").disabled = !ok;
  $("#start").title = !make.song ? "Choose a song first" : !$("#rights").checked ? "Confirm you have the rights" :
    !local && !server.base ? "Start Sangisa on your computer first" : "";
  $("#start").textContent = mode() === "split" ? `Make the acapella + instrumental${local ? " on this device" : ""}`
    : local ? "Make the kit on this device" : "Make the kit";
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
  form.append("mode", mode());
  if (mode() === "split") {
    form.append("format", $("#split-format").value);
    if ($("#split-fast").checked) form.append("fast", "true");
    if ($("#split-normalize").checked) form.append("normalize", "true");
    if ($("#split-stems").checked) form.append("include_stems", "true");
    store.set("split-format", $("#split-format").value);
  } else {
    form.append("pads", $("#pad-split").value);
    if ($("#hq-vocals").checked) form.append("hq_vocals", "true");
    store.set("pads", $("#pad-split").value);
    store.set("hq", $("#hq-vocals").checked ? "1" : null);
  }

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
  const split = mode() === "split";
  const separator = split ? "ai" : engine();
  if (!split) store.set("pads", $("#pad-split").value);
  store.set("engine", ai.meta ? engine() : null);
  const now = () => Date.now() / 1000;
  const rec = {
    name: song.name, status: "running", stage: "ingest", stage_label: "Loading", stage_started: now(),
    started: now(), stages: {}, error: null, local: true, value: null, note: "",
    order: split ? ["ingest", "separate", "analyze", "split"] : ["ingest", "separate", "analyze", "pick", "render"],
    labels: {
      ingest: "Loading", separate: separator === "ai" ? "Separating stems (AI)" : "Separating stems",
      analyze: "Analyzing tempo and key", pick: "Finding the best moments", render: "Building your kit",
      split: "Writing the acapella and instrumental",
    },
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
          if (m.status === "start") Object.assign(rec, { stage: m.stage, stage_label: rec.labels[m.stage], stage_started: now(), value: null });
          if (m.status === "progress") rec.value = m.value;
          if (m.status === "note") rec.note = m.note;
          if (m.status === "done") {
            rec.stages[m.stage] = m.stage === "ingest" ? now() - decodeStart : m.value;
            rec.value = null;
            if (m.stage === "separate") rec.note = "";
          }
          renderProgress(rec);
        } else if (m.type === "device") {
          rec.device = m.device;
        } else if (m.type === "done") resolve(m);
        else if (m.type === "error") reject(new Error(m.message));
      };
      worker.onerror = (e) => reject(new Error(e.message || "The kit maker stopped unexpectedly (out of memory?)."));
      worker.postMessage({
        type: "run", mode: split ? "split" : "kit", channels, sr, name: song.name, sha256, pads, separator,
        modelBase: new URL(MODEL_BASE, document.baseURI).href, includeStems: split && $("#split-stems").checked,
      }, channels.map((c) => c.buffer));
    });
    worker.terminate();
    rec.status = "done";
    renderProgress(rec);
    make.song = null;
    resetSongPicker();
    if (make.tracking !== "local") return; // the user went back
    if (split) {
      openLocalSplit(result.split, result.files, song);
      toast("Made on this device. Download the files to keep them.");
      return;
    }
    openKit(result.kit, async (p) => {
      const f = result.files[p];
      if (!f) throw new Error(`Missing ${p}`);
      return f;
    });
    toast("Made on this device. Use Export for Koala to keep it.");
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
      if (rec.mode === "split") await openSplit(id);
      else await openServerKit(id);
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
    if (rec.status === "running" && rec.value != null && rec.stage === "separate") {
      const li = $(`#prog-stages li[data-status="active"] .t`);
      if (li) li.textContent = `${Math.round(rec.value * 100)}% · ${li.textContent}`;
    }
    $("#prog-note").textContent = rec.status === "running"
      ? [rec.note, "Keep this page open; on a phone, keep the screen on."].filter(Boolean).join(" ")
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

// ---------------------------------------------------------------- acapella + instrumental

const player = { id: null, audios: {}, current: "acapella" };

function fileUrl(id, path, download = false) {
  const q = new URLSearchParams();
  if (server.key) q.set("key", server.key);
  if (download) q.set("download", "1");
  const enc = path.split("/").map(encodeURIComponent).join("/");
  return `${server.base}/api/jobs/${id}/files/${enc}${q.size ? `?${q}` : ""}`;
}

async function openSplit(id) {
  const info = await (await api(`/api/jobs/${id}/split.json`)).json();
  player.id = id;
  const zipQ = server.key ? `?key=${encodeURIComponent(server.key)}` : "";
  showSplit(info, (path, download) => fileUrl(id, path, download), `${server.base}/api/jobs/${id}/split.zip${zipQ}`);
}

/** The acapella + instrumental made on this device: play and download them from memory. */
function openLocalSplit(info, files, original) {
  for (const url of player.urls || []) URL.revokeObjectURL(url);
  player.id = null;
  player.urls = [];
  const urls = {};
  const blobUrl = (blob) => { const u = URL.createObjectURL(blob); player.urls.push(u); return u; };
  for (const [path, bytes] of Object.entries(files)) urls[path] = blobUrl(new Blob([bytes], { type: "audio/wav" }));
  urls["\0original"] = blobUrl(original);
  showSplit({ ...info, original: "\0original" }, (path) => urls[path], null, files);
}

function showSplit(info, urlOf, zipUrl, localFiles = null) {
  showView("split");
  $("#split-title").textContent = info.song;
  const fmt = info.format === "mp3" ? "MP3 320 kbps" : `${info.format.toUpperCase()}${info.bit_depth ? `, ${info.bit_depth}-bit` : ""}`;
  $("#split-meta").textContent = [
    info.bpm ? `${Math.round(info.bpm * 10) / 10} BPM` : null, info.key, `${fmt}, ${info.sample_rate / 1000} kHz`,
    info.normalized ? "normalized" : "adds back up to the original",
  ].filter(Boolean).join(" · ");

  const box = $("#split-audio");
  box.innerHTML = "";
  player.audios = {};
  for (const [role, path] of [["acapella", info.files.acapella], ["instrumental", info.files.instrumental], ["original", info.original]]) {
    const a = new Audio();
    a.preload = "auto";
    a.src = urlOf(path, false);
    a.addEventListener("timeupdate", () => role === player.current && updateClock());
    a.addEventListener("loadedmetadata", () => role === player.current && updateClock());
    a.addEventListener("ended", () => role === player.current && ($("#sp-play").textContent = "▶"));
    box.append(a);
    player.audios[role] = a;
  }
  player.current = "acapella";
  document.querySelector('input[name="sp-src"][value="acapella"]').checked = true;
  $("#sp-play").textContent = "▶";
  updateClock();

  const name = (path) => path.split("/").pop();
  const links = [
    ["Acapella", info.files.acapella], ["Instrumental", info.files.instrumental],
    ...Object.entries(info.stems || {}).map(([k, v]) => [k[0].toUpperCase() + k.slice(1), v]),
  ].map(([label, path]) => `<a class="btn small" href="${urlOf(path, true)}"${localFiles ? ` download="${esc(name(path))}"` : ""}>${esc(label)}</a>`);
  if (zipUrl) links.push(`<a class="btn small primary" href="${zipUrl}">Both (.zip)</a>`);
  else if (localFiles && window.JSZip) links.push('<button class="btn small primary" id="split-zip">All (.zip)</button>');
  $("#split-files").innerHTML = links.join("");
  $("#split-zip")?.addEventListener("click", guard(async () => {
    const zip = new JSZip();
    for (const [path, bytes] of Object.entries(localFiles)) zip.file(name(path), bytes);
    const blob = await zip.generateAsync({ type: "blob", compression: "STORE" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${info.song} - split.zip`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
  }));
  // Making a kit from a split reuses the computer's stems; on the device, start over from the song.
  $("#split-kit").closest(".kit-from").hidden = !player.id;
}

function updateClock() {
  const a = player.audios[player.current];
  if (!a) return;
  const d = Number.isFinite(a.duration) ? a.duration : 0;
  const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  $("#sp-time").textContent = `${fmt(a.currentTime)} / ${fmt(d)}`;
  if (!player.seeking) $("#sp-seek").value = d ? Math.round((a.currentTime / d) * 1000) : 0;
}

function switchSource(role) {
  const from = player.audios[player.current], to = player.audios[role];
  if (!to || from === to) return;
  const playing = !from.paused;
  from.pause();
  to.currentTime = from.currentTime;
  player.current = role;
  if (playing) to.play().catch(() => {});
  updateClock();
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
    const facts = j.kit || j.split;
    const what = [j.kit ? "Kit" : null, j.split ? "Acapella + instrumental" : null].filter(Boolean).join(" · ");
    const detail = j.status === "done" && facts
      ? `${what} · ${facts.bpm ? Math.round(facts.bpm) + " BPM" : ""} · ${facts.key || "key unknown"}`
      : j.status === "failed" ? `Failed: ${j.error}` : j.status === "queued" ? "Waiting…" : `Working: ${j.stage_label || "starting"}`;
    const buttons = [];
    if (j.status === "queued" || j.status === "running") buttons.push('<button class="btn small" data-act="open">Progress</button>');
    if (j.kit && j.status !== "running" && j.status !== "queued") buttons.push('<button class="btn small" data-act="open">Kit</button>');
    if (j.split && j.status !== "running" && j.status !== "queued") buttons.push('<button class="btn small" data-act="split">Acapella</button>');
    return `<li data-id="${j.id}" data-status="${j.status}">
      <div><b>${esc(j.kit?.name || j.name)}</b><span class="meta">${esc(detail)} · ${when}</span></div>
      <div class="row">${buttons.join("")}
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
    if (li.dataset.status === "queued" || li.dataset.status === "running") track(id);
    else await openServerKit(id);
  } else if (btn.dataset.act === "split") {
    await openSplit(id);
  } else if (btn.dataset.act === "delete") {
    if (!confirm("Delete this from your computer?")) return;
    await api(`/api/jobs/${id}`, { method: "DELETE" });
    refreshRecent();
  }
}));

/** Is the AI model published alongside this site? */
async function checkModel() {
  try {
    const res = await fetch(new URL(`${MODEL_BASE}htdemucs.json`, document.baseURI), { cache: "no-cache" });
    ai.meta = res.ok ? await res.json() : null;
  } catch {
    ai.meta = null;
  }
  if (ai.meta && !store.get("engine")) setEngine("ai");
  updateMode();
}

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
  const savedMode = store.get("mode");
  if (savedMode) document.querySelector(`input[name="mode"][value="${savedMode}"]`).checked = true;
  if (store.get("split-format")) $("#split-format").value = store.get("split-format");
  $("#hq-vocals").checked = store.get("hq") === "1";
  document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", () => {
    store.set("mode", mode());
    updateMode();
  }));
  $("#sp-play").onclick = () => {
    const a = player.audios[player.current];
    if (!a) return;
    if (a.paused) { a.play().catch((e) => toast(e.message)); $("#sp-play").textContent = "❚❚"; }
    else { a.pause(); $("#sp-play").textContent = "▶"; }
  };
  $("#sp-seek").addEventListener("input", () => {
    const a = player.audios[player.current];
    player.seeking = true;
    if (a && Number.isFinite(a.duration)) a.currentTime = ($("#sp-seek").value / 1000) * a.duration;
  });
  $("#sp-seek").addEventListener("change", () => { player.seeking = false; });
  document.querySelectorAll('input[name="sp-src"]').forEach((r) => r.addEventListener("change", () => switchSource(r.value)));
  $("#split-back").onclick = () => { showView("welcome"); refreshRecent(); };
  $("#split-kit").onclick = guard(async () => {
    const form = new FormData();
    form.append("pads", $("#pad-split").value);
    await api(`/api/jobs/${player.id}/kit`, { method: "POST", body: form });
    track(player.id);
  });
  setWhere(store.get("route") || "device");
  if (store.get("engine")) setEngine(store.get("engine"));
  document.querySelectorAll('input[name="engine"]').forEach((r) => r.addEventListener("change", () => store.set("engine", engine())));
  checkModel();
  document.querySelectorAll('input[name="where"]').forEach((r) => r.addEventListener("change", () => {
    store.set("route", route());
    updateMode();
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
  updateMode();
  connect();
})();

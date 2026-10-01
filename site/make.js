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
  if (stateName !== "checking") $("#setup").open = !ready;
  $("#phone-note").hidden = !isPhone;
  updateStart();
  const hint = $("#make-hint");
  if (stateName === "off") {
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

function updateStart() {
  const ok = Boolean(make.song && $("#rights").checked && server.base);
  $("#start").disabled = !ok;
  $("#start").title = !make.song ? "Choose a song first" : !$("#rights").checked ? "Confirm you have the rights" :
    !server.base ? "Start Sangisa on your computer first" : "";
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
  $("#start").onclick = guard(startJob);
  $("#prog-back").onclick = () => { make.tracking = null; showView("welcome"); refreshRecent(); };
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

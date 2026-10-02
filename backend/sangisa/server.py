"""``sangisa serve``: a small local web server so the app can take a song directly.

The browser (the GitHub Pages site, or the same UI served from here) uploads a
song; one background worker runs the pipeline on this machine; the browser
polls for progress and then loads the finished kit. Songs and stems never
leave the computer.

Access rules:
- Requests from other web pages are refused unless their origin is allowed
  (this server's own pages and the project's GitHub Pages site by default).
- Requests from other devices (a phone on the same Wi-Fi) must carry the
  access key printed at start-up.
"""

from __future__ import annotations

import ipaddress
import json
import queue
import re
import secrets
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sangisa import __version__
from sangisa.config import Config, load_config
from sangisa.job import MODE_STAGES, Job
from sangisa.pipeline import labels_for, run_pipeline
from sangisa.worker.stages.ingest import AUDIO_EXTENSIONS

DEFAULT_PORT = 8765
DEFAULT_ORIGINS = ("https://esang-mao.github.io",)
MAX_UPLOAD_MB = 300
SITE_DIR = Path(__file__).resolve().parents[2] / "site"
JOB_ID = re.compile(r"^[0-9a-f]{12}$")


@dataclass
class JobRecord:
    id: str
    name: str
    status: str = "queued"          # queued | running | done | failed
    stage: str | None = None
    stage_label: str | None = None
    stage_started: float | None = None
    stages: dict[str, float] = field(default_factory=dict)  # finished stage -> seconds
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    options: dict[str, Any] = field(default_factory=dict)
    mode: str = "kit"                  # kit | split (Acapella + instrumental)
    kit: dict[str, Any] | None = None  # name, bpm, key once a kit is done
    split: dict[str, Any] | None = None  # split.json once an acapella + instrumental is done

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["labels"] = labels_for(self.mode)
        data["order"] = list(MODE_STAGES[self.mode])
        return data


class JobManager:
    """Keeps job records, and runs one job at a time on a background thread."""

    def __init__(self, root: Path, base_config: Config):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_config = base_config
        self.records: dict[str, JobRecord] = {}
        self.lock = threading.Lock()
        self.queue: queue.Queue[tuple[str, Path | None]] = queue.Queue()
        self._load_existing()
        threading.Thread(target=self._worker, name="sangisa-worker", daemon=True).start()

    # Records -------------------------------------------------------------------
    def _status_path(self, job_id: str) -> Path:
        return self.root / job_id / "status.json"

    def _save(self, rec: JobRecord) -> None:
        path = self._status_path(rec.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(rec), indent=2))

    def _load_existing(self) -> None:
        for path in self.root.glob("*/status.json"):
            try:
                rec = JobRecord(**json.loads(path.read_text()))
            except (TypeError, ValueError):
                continue
            if rec.status in ("queued", "running"):  # the server stopped mid-job
                rec.status, rec.error = "failed", "Sangisa was stopped before this finished. Upload it again."
                self._save(rec)
            self.records[rec.id] = rec

    def list(self) -> list[JobRecord]:
        with self.lock:
            return sorted(self.records.values(), key=lambda r: -r.created)

    def get(self, job_id: str) -> JobRecord:
        rec = self.records.get(job_id) if JOB_ID.match(job_id) else None
        if rec is None:
            raise HTTPException(404, "No such job")
        return rec

    def job(self, job_id: str) -> Job:
        self.get(job_id)
        return Job(self.root / job_id)

    def queued_ahead(self) -> int:
        with self.lock:
            return sum(1 for r in self.records.values() if r.status in ("queued", "running"))

    # Work ----------------------------------------------------------------------
    def submit(self, upload: Path, name: str, options: dict[str, Any], mode: str = "kit") -> JobRecord:
        rec = JobRecord(id=uuid.uuid4().hex[:12], name=name, options=options, mode=mode)
        with self.lock:
            self.records[rec.id] = rec
            self._save(rec)
        self.queue.put((rec.id, upload))
        return rec

    def continue_as(self, job_id: str, mode: str, options: dict[str, Any]) -> JobRecord:
        """Run another mode on a finished job (e.g. make a kit from an acapella + instrumental)."""
        rec = self.get(job_id)
        with self.lock:
            if rec.status in ("queued", "running"):
                raise HTTPException(409, "That job is still running")
            rec.mode, rec.status, rec.error = mode, "queued", None
            rec.stages, rec.started, rec.finished = {}, None, None
            rec.options = {**rec.options, **options}
            self._save(rec)
        self.queue.put((job_id, None))
        return rec

    def delete(self, job_id: str) -> None:
        rec = self.get(job_id)
        if rec.status in ("queued", "running"):
            raise HTTPException(409, "That job is still running")
        with self.lock:
            self.records.pop(job_id, None)
        shutil.rmtree(self.root / job_id, ignore_errors=True)

    def _worker(self) -> None:
        while True:
            job_id, upload = self.queue.get()
            try:
                self._run(job_id, upload)
            finally:
                if upload is not None:
                    shutil.rmtree(upload.parent, ignore_errors=True)

    def _config(self, options: dict[str, Any]) -> Config:
        overrides: dict[str, Any] = {"separation": {}, "split": {}}
        if options.get("pads"):
            split = options["pads"]
            overrides["kit"] = {"pad_split": split, "pad_count": sum(split.values())}
        if options.get("model"):
            overrides["separation"]["model"] = options["model"]
        if options.get("hq_vocals"):
            overrides["separation"]["hq_vocals"] = True
        for key in ("format", "fast", "normalize", "include_stems"):
            if key in options:
                overrides["split"][key] = options[key]
        return _merged(self.base_config, overrides)

    def _run(self, job_id: str, upload: Path | None) -> None:
        rec = self.records[job_id]
        rec.status, rec.started = "running", time.time()
        self._save(rec)

        def progress(stage: str, status: str, seconds: float | None) -> None:
            if status == "start":
                rec.stage, rec.stage_label, rec.stage_started = stage, labels_for(rec.mode)[stage], time.time()
            else:
                rec.stages[stage] = round(seconds or 0.0, 1)
            self._save(rec)

        try:
            cfg = self._config(rec.options)
            job = Job(self.root / job_id)
            run_pipeline(job, cfg, source=str(upload) if upload else None, rights_confirmed=True,
                         mode=rec.mode, progress=progress)
            if rec.mode == "split":
                rec.split = json.loads((job.root / "split.json").read_text())
                (job.root / "split.zip").unlink(missing_ok=True)
            else:
                kit = json.loads(job.kit_path.read_text())
                rec.kit = {"name": kit["kit_name"], "bpm": kit["analysis"]["bpm"], "key": kit["analysis"]["key"]}
            rec.status = "done"
        except Exception as exc:  # report any stage failure to the browser
            rec.status, rec.error = "failed", str(exc) or exc.__class__.__name__
        rec.finished = time.time()
        rec.stage, rec.stage_label, rec.stage_started = None, None, None
        self._save(rec)


def _merged(base: Config, overrides: Config) -> Config:
    from sangisa.config import merge, validate

    cfg = merge(base, overrides)
    validate(cfg)
    return cfg


def parse_pads(text: str) -> dict[str, int]:
    split: dict[str, int] = {}
    for part in text.split(","):
        name, _, count = part.partition("=")
        if not name.strip() or not count.strip().isdigit():
            raise ValueError(f"expected stem=count, got {part!r}")
        split[name.strip()] = int(count)
    return split


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host in ("localhost", "testclient"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_app(
    jobs_dir: Path,
    *,
    config: Config | None = None,
    access_key: str | None = None,
    allowed_origins: tuple[str, ...] = DEFAULT_ORIGINS,
    site_dir: Path | None = SITE_DIR,
) -> FastAPI:
    manager = JobManager(jobs_dir, config or load_config())
    key = access_key or secrets.token_urlsafe(12)
    app = FastAPI(title="Sangisa", version=__version__, docs_url=None, redoc_url=None)
    app.state.manager = manager
    app.state.access_key = key

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/"):
            origin = request.headers.get("origin")
            own = f"{request.url.scheme}://{request.url.netloc}"
            if origin and origin != own and origin not in allowed_origins and not _local_origin(origin):
                return JSONResponse({"detail": "This web page isn't allowed to use Sangisa."}, status_code=403)
            if request.method != "OPTIONS" and not _is_loopback(request.client.host if request.client else None):
                given = request.headers.get("x-sangisa-key") or request.query_params.get("key")
                if not given or not secrets.compare_digest(given, key):
                    return JSONResponse({"detail": "Missing or wrong access key."}, status_code=401)
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["x-sangisa-key"],
        # Chrome asks before an https page talks to a server on this machine or the local network.
        allow_private_network=True,
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        from sangisa.worker.stages.separate import detect_device

        return {
            "app": "sangisa",
            "version": __version__,
            "device": detect_device(),
            "busy": manager.queued_ahead(),
            "max_upload_mb": MAX_UPLOAD_MB,
            "pad_split": manager.base_config["kit"]["pad_split"],
        }

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [r.public() for r in manager.list()]

    @app.post("/api/jobs", status_code=201)
    async def create_job(
        file: Annotated[UploadFile, File()],
        rights: Annotated[bool, Form()] = False,
        pads: Annotated[str | None, Form()] = None,
        model: Annotated[str | None, Form()] = None,
        mode: Annotated[str, Form()] = "kit",
        hq_vocals: Annotated[bool, Form()] = False,
        format: Annotated[str | None, Form()] = None,
        fast: Annotated[bool, Form()] = False,
        normalize: Annotated[bool, Form()] = False,
        include_stems: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        if not rights:
            raise HTTPException(400, "Confirm you own this audio or have the rights to sample it.")
        name = Path(file.filename or "song").name
        if Path(name).suffix.lower() not in AUDIO_EXTENSIONS:
            raise HTTPException(415, f"{name} isn't an audio file Sangisa can read (WAV, AIFF, FLAC, MP3, M4A).")
        if mode not in MODE_STAGES:
            raise HTTPException(400, "mode must be kit or split")
        options = _options(manager, pads=pads, model=model, hq_vocals=hq_vocals, format=format, fast=fast,
                           normalize=normalize, include_stems=include_stems)

        upload_dir = manager.root / "_uploads" / uuid.uuid4().hex
        upload_dir.mkdir(parents=True)
        dest = upload_dir / _safe_name(name)
        size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
        with open(dest, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    out.close()
                    shutil.rmtree(upload_dir, ignore_errors=True)
                    raise HTTPException(413, f"That file is over {MAX_UPLOAD_MB} MB.")
                out.write(chunk)
        return manager.submit(dest, name, options, mode).public()

    @app.post("/api/jobs/{job_id}/kit", status_code=202)
    def make_kit(
        job_id: str,
        pads: Annotated[str | None, Form()] = None,
        hq_vocals: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        """Make a kit from a finished job, reusing its separation (e.g. the acapella)."""
        return manager.continue_as(job_id, "kit", _options(manager, pads=pads, hq_vocals=hq_vocals)).public()

    @app.get("/api/jobs/{job_id}/split.json")
    def get_split(job_id: str) -> FileResponse:
        job = manager.job(job_id)
        path = job.root / "split.json"
        if not path.exists() or not job.is_done("split"):
            raise HTTPException(404, "There's no acapella + instrumental for this job yet")
        return FileResponse(path, media_type="application/json")

    @app.get("/api/jobs/{job_id}/split.zip")
    def get_split_zip(job_id: str) -> FileResponse:
        import zipfile

        job = manager.job(job_id)
        if not job.is_done("split"):
            raise HTTPException(404, "There's no acapella + instrumental for this job yet")
        info = json.loads((job.root / "split.json").read_text())
        zpath = job.root / "split.zip"
        if not zpath.exists():
            tmp = zpath.with_suffix(".zip.tmp")
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
                for rel in [*info["files"].values(), *info["stems"].values()]:
                    z.write(job.abs(rel), f"{info['song']}/{Path(rel).relative_to('split')}")
            tmp.replace(zpath)
        name = f"{info['song']} - Acapella + Instrumental.zip"
        return FileResponse(zpath, media_type="application/zip", filename=name)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return manager.get(job_id).public()

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> Response:
        manager.delete(job_id)
        return Response(status_code=204)

    @app.get("/api/jobs/{job_id}/kit.json")
    def get_kit(job_id: str) -> FileResponse:
        job = manager.job(job_id)
        if not job.kit_path.exists() or manager.get(job_id).status != "done":
            raise HTTPException(404, "The kit isn't ready yet")
        return FileResponse(job.kit_path, media_type="application/json")

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def get_file(job_id: str, path: str, download: bool = False) -> FileResponse:
        job = manager.job(job_id)
        target = (job.root / path).resolve()
        split_dir = job.root / "split"
        allowed = (
            target in (job.analysis_path, job.work_wav)  # work.wav: the original, for A/B listening
            or (job.slices_dir in target.parents and target.suffix == ".wav")
            or (split_dir in target.parents and target.suffix in (".wav", ".flac", ".mp3"))
        )
        if not allowed or not target.is_file():
            raise HTTPException(404, "No such file")
        return FileResponse(target, filename=target.name if download else None)

    if site_dir and (site_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=site_dir, html=True), name="site")
    return app


def _options(manager: JobManager, **given: Any) -> dict[str, Any]:
    """Validate the per-job options a form sent; returns only the ones that were set."""
    options: dict[str, Any] = {}
    if given.get("pads"):
        try:
            options["pads"] = parse_pads(given["pads"])
            _merged(manager.base_config, {"kit": {"pad_split": options["pads"],
                                                 "pad_count": sum(options["pads"].values())}})
        except ValueError as exc:
            raise HTTPException(400, f"Pad split: {exc}") from exc
    if given.get("model"):
        if not re.fullmatch(r"[\w.\-]+", given["model"]):
            raise HTTPException(400, "Unknown model name")
        options["model"] = given["model"]
    if given.get("format"):
        if given["format"] not in ("wav", "flac", "mp3"):
            raise HTTPException(400, "format must be wav, flac or mp3")
        options["format"] = given["format"]
    for key in ("hq_vocals", "fast", "normalize", "include_stems"):
        if given.get(key):
            options[key] = True
    return options


def _local_origin(origin: str) -> bool:
    return bool(re.fullmatch(r"https?://(localhost|127\.0\.0\.1)(:\d+)?", origin))


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^\w\- .()]+", "_", Path(name).stem).strip() or "song"
    return f"{stem[:80]}{Path(name).suffix.lower()}"


def lan_address() -> str | None:
    """This machine's address on the local network (no packets are sent)."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        addr = s.getsockname()[0]
        return None if addr.startswith("127.") else addr
    except OSError:
        return None
    finally:
        s.close()


def serve(port: int = DEFAULT_PORT, lan: bool = False, jobs_dir: Path | None = None,
          config: Config | None = None) -> None:
    import uvicorn

    jobs = (jobs_dir or Path("~/.sangisa/jobs")).expanduser()
    jobs.mkdir(parents=True, exist_ok=True)
    # Keep the access key between runs, so a phone that scanned the code once stays connected.
    key_file = jobs / ".access-key"
    if not key_file.exists():
        key_file.write_text(secrets.token_urlsafe(12))
        key_file.chmod(0o600)
    app = create_app(jobs, config=config, access_key=key_file.read_text().strip())
    key = app.state.access_key
    print(f"\nSangisa {__version__} is running. Keep this window open.\n")
    print(f"  On this computer:  http://localhost:{port}/")
    if lan:
        addr = lan_address()
        if addr:
            url = f"http://{addr}:{port}/?key={key}"
            print(f"  On your phone:     {url}")
            print("  (same Wi-Fi; scan the code below with the phone's camera)\n")
            try:
                import segno

                segno.make(url, error="L").terminal(compact=True)
            except Exception:  # the code is a convenience, the URL above is enough
                pass
        else:
            print("  Couldn't find this computer's Wi-Fi address; phones can't connect.")
    else:
        print("  To use it from your phone on the same Wi-Fi, start it with: sangisa serve --lan")
    print(f"\n  Jobs are saved in {jobs}\n")
    uvicorn.run(app, host="0.0.0.0" if lan else "127.0.0.1", port=port, log_level="warning")

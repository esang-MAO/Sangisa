"""Runs the stages in order against a job folder, skipping any that already finished."""

from __future__ import annotations

import time
from typing import Callable

from sangisa.config import Config
from sangisa.job import STAGES, Job
from sangisa.worker.stages import analyze, ingest, pick, render, separate

LABELS = {
    "ingest": "Loading",
    "separate": "Separating stems",
    "analyze": "Analyzing tempo and key",
    "pick": "Finding the best moments",
    "render": "Building your kit",
}

Progress = Callable[[str, str, float | None], None]


def _quiet(stage: str, status: str, seconds: float | None) -> None:
    pass


def run_pipeline(
    job: Job,
    cfg: Config,
    *,
    source: str | None = None,
    rights_confirmed: bool = False,
    only: str | None = None,
    start_at: str | None = None,
    force: bool = False,
    progress: Progress = _quiet,
) -> None:
    job.create()
    previous = job.manifest().get("source", {}).get("input")
    if force or (source is not None and previous is not None and previous != source):
        job.invalidate_from(STAGES[0])
    if only:
        job.invalidate_from(only)
        todo = [only]
    else:
        if start_at:
            job.invalidate_from(start_at)
        todo = [s for s in STAGES if not job.is_done(s)]

    for stage in todo:
        if stage == "ingest" and source is None:
            raise ValueError("No input: pass a song file or URL to start a new job.")
        missing = [s for s in STAGES[: STAGES.index(stage)] if not job.is_done(s)]
        if missing:
            raise ValueError(f"Can't run {stage!r} before {', '.join(missing)} has run.")
        progress(stage, "start", None)
        t0 = time.monotonic()
        job.mark(stage, "running")
        try:
            if stage == "ingest":
                ingest.run(job, cfg, source, rights_confirmed)
            else:
                {"separate": separate.run, "analyze": analyze.run, "pick": pick.run, "render": render.run}[stage](
                    job, cfg
                )
        except Exception as exc:
            job.mark(stage, "failed", error=str(exc))
            raise
        elapsed = time.monotonic() - t0
        job.mark(stage, "done", seconds=round(elapsed, 2))
        progress(stage, "done", elapsed)

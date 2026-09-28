"""Stage 5: cut every kept slice from its stem and write it as a finished WAV."""

from __future__ import annotations

import shutil

import numpy as np

from sangisa import audio
from sangisa.config import Config
from sangisa.job import Job
from sangisa.schema import Kit, Slice


def run(job: Job, cfg: Config) -> None:
    kit = Kit.load(job.kit_path)
    if job.slices_dir.exists():
        shutil.rmtree(job.slices_dir)
    job.slices_dir.mkdir(parents=True)

    stems: dict[str, tuple[np.ndarray, int]] = {}
    for name, rel in kit.separation.stems.items():
        stems[name] = audio.load(job.abs(rel))

    for s in kit.slices:
        data, sr = stems[s.stem]
        audio.save(job.abs(s.file), render_slice(data, sr, s, cfg), sr, cfg["ingest"]["bit_depth"])


def render_slice(stem: np.ndarray, sr: int, s: Slice, cfg: Config) -> np.ndarray:
    rcfg = cfg["render"]
    mono = audio.to_mono(stem)
    window = int(sr * rcfg["zero_crossing_window_ms"] / 1000)
    start = int(round(s.source_start_s * sr))
    end = int(round(s.source_end_s * sr))
    if s.render.playback == "loop":
        # Keep loops exactly on the grid: move the start to a zero crossing and keep the length.
        length = end - start
        start = audio.nearest_zero_crossing(mono, start, window)
        end = min(start + length, stem.shape[1])
    else:
        start = audio.nearest_zero_crossing(mono, start, window)
        end = audio.nearest_zero_crossing(mono, end, window)
    out = stem[:, start:end].copy()
    if s.render.reverse:
        out = out[:, ::-1].copy()
    out = audio.apply_fades(out, sr, s.render.fade_in_ms, s.render.fade_out_ms)
    if s.render.normalize_dbfs is not None:
        out = audio.peak_normalize(out, s.render.normalize_dbfs)
    return out

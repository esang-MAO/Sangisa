"""Acapella + instrumental: write the two full-length files (and optionally the stems).

Both files match the original exactly in length, start point and sample rate, so they drop into a
DAW in sync at bar 1. Without normalizing (the default) they add back up to the original.

    <job>/split/<Song> - Acapella - 92bpm Fmin.wav
    <job>/split/<Song> - Instrumental - 92bpm Fmin.wav
    <job>/split/stems/<Song> - Drums.wav …   (split.include_stems)
    <job>/split.json                          what was written, for the app
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from sangisa import audio
from sangisa.config import Config
from sangisa.job import Job, read_json, write_json


def run(job: Job, cfg: Config) -> None:
    scfg = cfg["split"]
    manifest = job.manifest()
    sep = manifest.get("separation") or {}
    pair = sep.get("pair")
    if not pair:
        raise RuntimeError("Nothing to split: the separation step didn't produce an acapella.")
    analysis = read_json(job.analysis_path) if job.analysis_path.exists() else {}
    src = manifest["source"]
    song = _safe(Path(src["path"]).stem) or "Song"
    bpm = f"{round(analysis['bpm'])}bpm" if analysis.get("bpm") else ""
    tag = " ".join(t for t in (bpm, analysis.get("key_short") or "") if t)  # "92bpm Fmin"
    out_sr = int(src.get("original_sample_rate") or src.get("sample_rate") or 44100)

    out = job.root / "split"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()

    work_sr = sf.info(str(job.work_wav)).samplerate
    files: dict[str, str] = {}
    for role, rel in (("Acapella", pair["vocals"]), ("Instrumental", pair["instrumental"])):
        name = " - ".join(p for p in (song, role, tag) if p)
        files[role.lower()] = job.rel(_write(job.abs(rel), out / name, work_sr, out_sr, scfg))

    stems: dict[str, str] = {}
    if scfg["include_stems"] and sep.get("stems"):
        (out / "stems").mkdir()
        for stem, rel in sorted(sep["stems"].items()):
            name = f"{song} - {stem.capitalize()}"
            stems[stem] = job.rel(_write(job.abs(rel), out / "stems" / name, work_sr, out_sr, scfg))

    write_json(job.root / "split.json", {
        "song": song,
        "bpm": analysis.get("bpm"),
        "key": analysis.get("key"),
        "format": scfg["format"],
        "sample_rate": out_sr,
        "bit_depth": scfg["bit_depth"] if scfg["format"] != "mp3" else None,
        "normalized": scfg["normalize"],
        "vocal_model": sep.get("vocal_model") or sep.get("model"),
        "files": files,
        "stems": stems,
        "original": job.rel(job.work_wav),
    })


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip()


def _write(src: Path, dest_base: Path, work_sr: int, out_sr: int, scfg: Config) -> Path:
    data, _ = audio.load(src)
    if out_sr != work_sr:
        import librosa

        n = int(round(data.shape[1] * out_sr / work_sr))
        data = audio.fit_length(librosa.resample(data, orig_sr=work_sr, target_sr=out_sr, res_type="soxr_vhq"), n)
    if scfg["normalize"]:
        data = audio.peak_normalize(data, -1.0)
    float_out = scfg["format"] == "wav" and scfg["bit_depth"] == 32
    if not float_out and np.max(np.abs(data)) > 1.0:
        data = np.clip(data, -1.0, 1.0)  # integer formats can't hold overs

    fmt = scfg["format"]
    if fmt == "wav":
        dest = dest_base.with_suffix(".wav")
        audio.save(dest, data, out_sr, scfg["bit_depth"])
    elif fmt == "flac":
        dest = dest_base.with_suffix(".flac")
        subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_24"}[scfg["bit_depth"]]
        sf.write(str(dest), data.T, out_sr, subtype=subtype, format="FLAC")
    else:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("MP3 export needs ffmpeg installed.")
        dest = dest_base.with_suffix(".mp3")
        tmp = dest_base.with_suffix(".tmp.wav")
        audio.save(tmp, data, out_sr, 24)
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp),
             "-codec:a", "libmp3lame", "-b:a", "320k", str(dest)],
            capture_output=True, text=True,
        )
        tmp.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg couldn't write the MP3: {result.stderr.strip()}")
    return dest

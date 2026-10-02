"""Stage 1: copy the source into the job, convert it to a 44.1 kHz / 24-bit / stereo working WAV."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from sangisa import audio
from sangisa.config import Config
from sangisa.job import Job

AUDIO_EXTENSIONS = {".wav", ".wave", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}


class IngestError(RuntimeError):
    pass


def run(job: Job, cfg: Config, source: str, rights_confirmed: bool) -> None:
    if not rights_confirmed:
        raise IngestError(
            "Confirm you own this audio or have the rights to sample it "
            "(pass --i-have-rights on the command line)."
        )
    icfg = cfg["ingest"]
    job.create()
    job.input_dir.mkdir(exist_ok=True)
    for old in job.input_dir.iterdir():
        old.unlink()

    original = _fetch(source, job.input_dir, icfg)
    _convert(original, job.work_wav, icfg)

    info = sf.info(str(job.work_wav))
    if info.duration > icfg["max_duration_s"]:
        raise IngestError(
            f"{original.name} is {info.duration / 60:.1f} minutes long; "
            f"the limit is {icfg['max_duration_s'] / 60:.0f} minutes (ingest.max_duration_s)."
        )

    job.update_manifest(
        source={
            "input": source,
            "path": job.rel(original),
            "sha256": _sha256(original),
            "duration_s": round(info.duration, 3),
            "sample_rate": info.samplerate,
            "original_sample_rate": _original_rate(original) or info.samplerate,
            "rights_confirmed": True,
        }
    )


def _fetch(source: str, dest_dir: Path, icfg: Config) -> Path:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in ("http", "https"):
        name = Path(urllib.parse.unquote(parsed.path)).name
        if Path(name).suffix.lower() in AUDIO_EXTENSIONS:
            if not icfg["allow_direct_urls"]:
                raise IngestError("Direct audio URLs are turned off (ingest.allow_direct_urls).")
            dest = dest_dir / name
            with urllib.request.urlopen(source, timeout=60) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
            return dest
        if not icfg["allow_streaming_links"]:
            raise IngestError(
                "That link is not a direct audio file. Streaming-site downloads are off by default "
                "because many platforms prohibit downloading (ingest.allow_streaming_links)."
            )
        return _yt_dlp(source, dest_dir)

    path = Path(source).expanduser()
    if not path.is_file():
        raise IngestError(f"No such file: {source}")
    dest = dest_dir / path.name
    shutil.copy2(path, dest)
    return dest


def _yt_dlp(url: str, dest_dir: Path) -> Path:
    if shutil.which("yt-dlp") is None:
        raise IngestError("yt-dlp is not installed.")
    subprocess.run(
        ["yt-dlp", "-f", "bestaudio", "--no-playlist", "-o", str(dest_dir / "source.%(ext)s"), url],
        check=True,
    )
    return next(dest_dir.glob("source.*"))


def _convert(src: Path, dest: Path, icfg: Config) -> None:
    sr, channels, bits = icfg["sample_rate"], icfg["channels"], icfg["bit_depth"]
    if shutil.which("ffmpeg"):
        codec = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_f32le"}[bits]
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-vn", "-ac", str(channels), "-ar", str(sr), "-c:a", codec, str(dest)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise IngestError(f"ffmpeg could not decode {src.name}: {result.stderr.strip()}")
        return

    # No ffmpeg: fall back to libsndfile (WAV, AIFF, FLAC, and MP3 on recent versions).
    try:
        data, file_sr = audio.load(src)
    except Exception as exc:  # surface any decoder error the same way
        raise IngestError(f"Could not decode {src.name} (install ffmpeg for more formats): {exc}") from exc
    if data.shape[0] == 1:
        data = np.repeat(data, channels, axis=0)
    elif data.shape[0] > channels:
        data = data[:channels]
    if file_sr != sr:
        import librosa

        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    audio.save(dest, data, sr, bits)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _original_rate(path: Path) -> int | None:
    """The source file's own sample rate, so Acapella + instrumental can be delivered at it."""
    try:
        return int(sf.info(str(path)).samplerate)
    except Exception:  # not readable by libsndfile (e.g. M4A): ask ffprobe
        pass
    if shutil.which("ffprobe"):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
    return None

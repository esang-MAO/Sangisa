"""Stage 2: split the working WAV into stems.

Every backend implements ``separate(path, model, out_dir) -> {stem_name: wav_path}``
with lower-case stem names (drums, bass, vocals, other, and guitar/piano for
6-stem models), so models and libraries can be swapped without touching the
stages after this one.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Protocol

from sangisa import audio
from sangisa.config import Config
from sangisa.job import Job


class Separator(Protocol):
    def __call__(self, path: Path, model: str, out_dir: Path, **options) -> dict[str, Path]: ...


_BACKENDS: dict[str, Separator] = {}


def register_separator(name: str) -> Callable[[Separator], Separator]:
    def deco(fn: Separator) -> Separator:
        _BACKENDS[name] = fn
        return fn

    return deco


def get_separator(name: str) -> Separator:
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown separation backend {name!r}; known: {sorted(_BACKENDS)}") from None


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@register_separator("audio-separator")
def audio_separator_backend(
    path: Path, model: str, out_dir: Path, model_dir: str = "~/.cache/sangisa/models", **_: object
) -> dict[str, Path]:
    """HT-Demucs / RoFormer via python-audio-separator. It picks CUDA, MPS or CPU itself."""
    try:
        from audio_separator.separator import Separator as AudioSeparator
    except ImportError as exc:
        raise RuntimeError(
            "Stem separation needs the optional dependency: uv sync --extra separation "
            "(or --extra separation-gpu on NVIDIA machines)"
        ) from exc

    models = Path(model_dir).expanduser()
    models.mkdir(parents=True, exist_ok=True)
    sep = AudioSeparator(
        model_file_dir=str(models),
        output_dir=str(out_dir),
        output_format="WAV",
        sample_rate=44100,
        use_soundfile=True,           # keeps the 24-bit input depth
        normalization_threshold=1.0,  # don't rescale: stems must still sum to the mix
    )
    sep.load_model(model_filename=model)
    names = ["Drums", "Bass", "Vocals", "Other", "Guitar", "Piano", "Instrumental"]
    outputs = sep.separate(str(path), custom_output_names={n: n.lower() for n in names})

    stems: dict[str, Path] = {}
    for out in outputs:
        p = Path(out)
        if not p.is_absolute():
            p = out_dir / p
        stem = p.stem.lower()
        if stem not in {n.lower() for n in names}:
            # Older releases ignore custom names: "work_(Vocals)_htdemucs_ft.wav"
            m = re.search(r"\(([^)]+)\)", p.stem)
            stem = m.group(1).lower() if m else stem
        stems[stem] = p
    return stems


def run(job: Job, cfg: Config) -> None:
    scfg = cfg["separation"]
    separator = get_separator(scfg["backend"])
    mix, sr = audio.load(job.work_wav)
    n = mix.shape[1]

    if job.stems_dir.exists():
        shutil.rmtree(job.stems_dir)
    job.stems_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(dir=job.root) as tmp:
        options = {k: v for k, v in scfg.items() if k not in ("backend", "model")}
        raw = separator(job.work_wav, scfg["model"], Path(tmp), **options)
        if not raw:
            raise RuntimeError("Separation produced no stems")
        stems: dict[str, str] = {}
        for name, path in sorted(raw.items()):
            data, stem_sr = audio.load(path)
            if stem_sr != sr:
                import librosa

                data = librosa.resample(data, orig_sr=stem_sr, target_sr=sr)
            # Stems must line up sample-for-sample with the working copy.
            dest = job.stems_dir / f"{name}.wav"
            audio.save(dest, audio.fit_length(data, n), sr, cfg["ingest"]["bit_depth"])
            stems[name] = job.rel(dest)

    job.update_manifest(
        separation={
            "backend": scfg["backend"],
            "model": scfg["model"],
            "device": detect_device(),
            "stems": stems,
        }
    )

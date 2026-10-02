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


KIT_STEMS = ("drums", "bass", "vocals", "other")


def run(job: Job, cfg: Config, mode: str = "kit") -> None:
    """Produce what `mode` needs, reusing whatever an earlier run already separated.

    - kit: stems/<stem>.wav. With separation.hq_vocals (or when an acapella from the vocal model already
      exists), the vocals come from the vocal model and drums/bass/other from the 4-stem model run on the
      instrumental; otherwise one 4-stem pass over the mix.
    - split: pair/vocals.wav + pair/instrumental.wav (the acapella and the instrumental), from the vocal
      model, or from the 4-stem model with split.fast. split.include_stems also makes the kit stems.
    The instrumental is always the mix minus the acapella, so the pair adds back up to the original exactly.
    """
    scfg = cfg["separation"]
    split = mode == "split"
    use_vocal_model = (split and not cfg["split"]["fast"]) or (not split and scfg["hq_vocals"])
    need_stems = not split or cfg["split"]["include_stems"]

    mix, sr = audio.load(job.work_wav)
    n = mix.shape[1]
    bits = cfg["ingest"]["bit_depth"]
    info = dict(job.manifest().get("separation") or {})
    sep = StemStore(job, cfg, n, sr, bits)
    try:
        _separate(job, cfg, info, sep, mix, split, use_vocal_model, need_stems)
    finally:
        sep.close()


def _separate(job, cfg, info, sep, mix, split, use_vocal_model, need_stems) -> None:
    scfg = cfg["separation"]

    pair = info.get("pair")
    pair_source = info.get("pair_source")
    if pair and not all(job.abs(p).exists() for p in pair.values()):
        pair = None
    if pair and use_vocal_model and pair_source != "vocal_model":
        pair = None  # a fast (4-stem) acapella isn't good enough when the vocal model was asked for

    stems = info.get("stems") if info.get("stems_from") else None
    if stems and not all(job.abs(p).exists() for p in stems.values()):
        stems = None

    if (split or use_vocal_model) and pair is None:
        if use_vocal_model:
            vocals = sep.run(scfg["vocal_model"], job.work_wav).get("vocals")
            if vocals is None:
                raise RuntimeError(f"{scfg['vocal_model']} produced no vocals stem")
            pair, pair_source = sep.save_pair(vocals, mix), "vocal_model"
            stems = None  # stems from before must be rebuilt around the new acapella
        else:
            four = sep.run(scfg["model"], job.work_wav)
            stems, stems_from = sep.save_stems(four), "mix"
            pair, pair_source = sep.save_pair(four["vocals"], mix), "model"
            info["stems_from"] = stems_from

    if need_stems:
        want_from = "instrumental" if pair_source == "vocal_model" else "mix"
        if stems is None or info.get("stems_from") != want_from:
            if want_from == "instrumental":
                rest = sep.run(scfg["model"], job.abs(pair["instrumental"]))
                rest.pop("vocals", None)  # what the 4-stem model hears as vocals here is leftover bleed
                rest["vocals"] = job.abs(pair["vocals"])
            else:
                rest = sep.run(scfg["model"], job.work_wav)
            stems = sep.save_stems(rest)
            info["stems_from"] = want_from

    info.update(
        backend=scfg["backend"],
        model=scfg["model"],
        vocal_model=scfg["vocal_model"] if pair_source == "vocal_model" else None,
        device=detect_device(),
        stems=stems or {},
        pair=pair,
        pair_source=pair_source,
    )
    if not stems:
        info.pop("stems_from", None)
    job.update_manifest(separation=info)


def satisfied(job: Job, cfg: Config, mode: str) -> bool:
    """Whether an earlier separation already covers what `mode` needs (so the stage can be skipped)."""
    info = job.manifest().get("separation") or {}
    split = mode == "split"
    use_vocal_model = (split and not cfg["split"]["fast"]) or (not split and cfg["separation"]["hq_vocals"])
    if (split or use_vocal_model) and not info.get("pair"):
        return False
    if use_vocal_model and info.get("pair_source") != "vocal_model":
        return False
    if (not split or cfg["split"]["include_stems"]) and not info.get("stems"):
        return False
    if not split and use_vocal_model and info.get("stems_from") != "instrumental":
        return False
    return True


class StemStore:
    """Runs models on files and stores the results sample-aligned with work.wav."""

    def __init__(self, job: Job, cfg: Config, n: int, sr: int, bits: int):
        self.job, self.cfg, self.n, self.sr, self.bits = job, cfg, n, sr, bits
        self.tmp = Path(tempfile.mkdtemp(dir=job.root, prefix=".sep-"))

    def close(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def load(self, path: Path):
        data, sr = audio.load(path)
        if sr != self.sr:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=self.sr)
        return audio.fit_length(data, self.n)

    def run(self, model: str, path: Path) -> dict[str, Path]:
        scfg = self.cfg["separation"]
        out = Path(tempfile.mkdtemp(dir=self.tmp))
        raw = get_separator(scfg["backend"])(path, model, out, model_dir=scfg["model_dir"])
        if not raw:
            raise RuntimeError(f"{model} produced no stems")
        return {name: Path(p) for name, p in raw.items()}

    def save_stems(self, raw: dict[str, Path]) -> dict[str, str]:
        d = self.job.stems_dir
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        stems = {}
        for name, path in sorted(raw.items()):
            if name == "instrumental":
                continue
            dest = d / f"{name}.wav"
            audio.save(dest, self.load(path), self.sr, self.bits)
            stems[name] = self.job.rel(dest)
        return stems

    def save_pair(self, vocals_path: Path, mix) -> dict[str, str]:
        d = self.job.root / "pair"
        d.mkdir(exist_ok=True)
        vocals = self.load(vocals_path)
        audio.save(d / "vocals.wav", vocals, self.sr, 32)  # float, so the pair sums back exactly
        audio.save(d / "instrumental.wav", mix - vocals, self.sr, 32)
        return {"vocals": self.job.rel(d / "vocals.wav"), "instrumental": self.job.rel(d / "instrumental.wav")}

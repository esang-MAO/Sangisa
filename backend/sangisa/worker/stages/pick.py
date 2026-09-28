"""Stage 4: find, score and de-duplicate candidates per stem, then assemble the kit and write kit.json."""

from __future__ import annotations

import string
from pathlib import Path

import librosa
import numpy as np

from sangisa import audio
from sangisa.config import Config
from sangisa.job import Job, read_json, write_json
from sangisa.schema import Analysis, Kit, Layout, Pad, RenderSettings, ScoreParts, Separation, Slice, Source
from sangisa.worker.extractors import Candidate, Context, extractor_for
from sangisa.worker.scoring import score_all
from sangisa.worker.stages.analyze import PITCH_NAMES, section_at

# Categories are dealt round-robin per stem so, e.g., the drum pads get a kick,
# a snare, a hat and a perc before any second kick.
CATEGORY_ORDER = {
    "drums": ["kick", "snare", "hat", "perc", "drum_loop"],
    "bass": ["bass_loop", "bass_note"],
    "vocals": ["vox_chop", "vox_phrase"],
    "other": ["other_loop", "stab"],
}


def run(job: Job, cfg: Config) -> None:
    manifest = job.manifest()
    analysis = read_json(job.analysis_path)
    stems: dict[str, np.ndarray] = {}
    sr = cfg["ingest"]["sample_rate"]
    for name, rel in manifest["separation"]["stems"].items():
        data, sr = audio.load(job.abs(rel))
        stems[name] = audio.to_mono(data)

    ctx = Context(
        sr=sr,
        bpm=analysis["bpm"],
        beats=np.asarray(analysis["beats_s"]),
        downbeats=np.asarray(analysis["downbeats_s"]),
        cfg=cfg,
    )
    cands = find_candidates(stems, ctx)
    score_all(cands, stems, sr, cfg)
    write_json(job.candidates_path, [candidate_json(c, sr) for c in cands])

    kit = build_kit(job, cfg, manifest, analysis, stems, cands, sr)
    kit.save(job.kit_path)


def find_candidates(stems: dict[str, np.ndarray], ctx: Context) -> list[Candidate]:
    cands: list[Candidate] = []
    for name in sorted(stems):
        found = extractor_for(name)(stems[name], ctx)
        found.sort(key=lambda c: (c.start, c.end, c.category))
        for i, c in enumerate(found):
            c.id = f"{name}-{i:04d}"
        cands.extend(found)
    return cands


def choose_pads(cands: list[Candidate], cfg: Config) -> tuple[list[Candidate | None], list[Candidate]]:
    """Return (candidate per pad in pad order, backups offered as swaps)."""
    split: dict[str, int] = cfg["kit"]["pad_split"]
    usable = [c for c in cands if c.score > 0]
    by_score = sorted(usable, key=lambda c: -c.score)
    used: set[str] = set()
    pads: list[Candidate | None] = []

    for stem, count in split.items():
        reps = [c for c in by_score if c.stem == stem and c.representative]
        order = CATEGORY_ORDER.get(stem) or sorted({c.category for c in reps})
        queues = {cat: [c for c in reps if c.category == cat] for cat in order}
        picked: list[Candidate] = []
        while len(picked) < count and any(queues.values()):
            for cat in order:
                if queues[cat] and len(picked) < count:
                    picked.append(queues[cat].pop(0))
        # Not enough distinct sounds: fall back to the best remaining, duplicates included.
        for c in by_score:
            if len(picked) >= count:
                break
            if c.stem == stem and c.id not in {p.id for p in picked}:
                picked.append(c)
        used.update(c.id for c in picked)
        pads.extend(picked + [None] * (count - len(picked)))

    # Pads a stem couldn't fill (e.g. an instrumental has no vocals) go to the best leftovers.
    leftovers = [c for c in by_score if c.representative and c.id not in used]
    for i, slot in enumerate(pads):
        if slot is None and leftovers:
            pads[i] = leftovers.pop(0)
            used.add(pads[i].id)

    backups: list[Candidate] = []
    per_stem = cfg["kit"]["backups_per_stem"]
    for stem in sorted({c.stem for c in usable}):
        pool = [c for c in by_score if c.stem == stem and c.id not in used]
        pool.sort(key=lambda c: (not c.representative, -c.score))
        backups.extend(pool[:per_stem])
    return pads, backups


def build_kit(job: Job, cfg: Config, manifest: dict, analysis: dict, stems: dict[str, np.ndarray],
              cands: list[Candidate], sr: int) -> Kit:
    pads, backups = choose_pads(cands, cfg)
    kept: list[Candidate] = [c for c in pads if c is not None] + backups
    labeler = Labeler(analysis, sr)
    rcfg = cfg["render"]

    slices = []
    for c in kept:
        describe_pitch(c, stems[c.stem], sr)
        label, file_stem = labeler.name(c)
        is_loop = c.kind == "loop"
        slices.append(Slice(
            id=c.id,
            stem=c.stem,
            kind=c.kind,
            category=c.category,
            label=label,
            file=f"slices/{file_stem}.wav",
            source_start_s=round(c.start / sr, 4),
            source_end_s=round(c.end / sr, 4),
            bars=c.bars,
            beats=c.beats,
            note=c.note,
            section=section_at(analysis.get("sections", []), c.start / sr),
            score=c.score,
            score_parts=ScoreParts(**c.parts),
            cluster=c.cluster,
            render=RenderSettings(
                fade_in_ms=rcfg["loop_fade_ms"] if is_loop else rcfg["fade_in_ms"],
                fade_out_ms=rcfg["loop_fade_ms"] if is_loop else rcfg["fade_out_ms"],
                normalize_dbfs=rcfg["normalize_dbfs"] if rcfg["normalize"] else None,
                playback="loop" if is_loop else "one_shot",
            ),
        ))

    pad_models = []
    for i, c in enumerate(pads):
        pad_models.append(Pad(
            pad=i + 1,
            bank=string.ascii_uppercase[i // 16],
            midi_note=36 + i % 16,
            slice_id=c.id if c else None,
        ))

    src = manifest["source"]
    sep = manifest["separation"]
    return Kit(
        kit_name=f"{Path(src['path']).stem} Kit",
        source=Source(path=src["path"], sha256=src["sha256"], duration_s=src["duration_s"],
                      sample_rate=src["sample_rate"], rights_confirmed=src["rights_confirmed"]),
        analysis=Analysis(**{k: analysis[k] for k in Analysis.model_fields if k in analysis}),
        separation=Separation(**sep),
        layout=Layout(pad_count=cfg["kit"]["pad_count"], pad_split=dict(cfg["kit"]["pad_split"])),
        pads=pad_models,
        slices=slices,
    )


class Labeler:
    """Human labels ("Kick 1", "Bass loop A") and file names (Bass_Loop_A_2bar_92bpm_Fmin)."""

    def __init__(self, analysis: dict, sr: int):
        self.counts: dict[str, int] = {}
        bpm = int(round(analysis["bpm"]))
        key = analysis.get("key_short")
        self.suffix = f"_{bpm}bpm" + (f"_{key}" if key else "")

    def _next(self, category: str) -> int:
        self.counts[category] = self.counts.get(category, 0) + 1
        return self.counts[category]

    def name(self, c: Candidate) -> tuple[str, str]:
        n = self._next(c.category)
        stem_title = c.stem.capitalize()
        bars = f"{c.bars:g}bar" if c.bars else None
        note = c.note
        simple = {"kick": "Kick", "snare": "Snare", "hat": "Hat", "perc": "Perc"}
        if c.category in simple:
            label, parts = f"{simple[c.category]} {n}", [stem_title, simple[c.category], str(n)]
        elif c.kind == "loop":
            letter = letters(n)
            noun = {"drums": "Drum", "vocals": "Vox"}.get(c.stem, stem_title)
            label = f"{noun} loop {letter} ({c.bars:g} bar)"
            parts = [stem_title, "Loop", letter, bars]
        elif c.category == "bass_note":
            label = f"Bass note {n}" + (f" {note}" if note else "")
            parts = [stem_title, "Note", str(n), note]
        elif c.category == "vox_chop":
            label = f"Vox chop {n}" + (f" {note}" if note else "")
            parts = [stem_title, "Chop", str(n), note]
        elif c.category == "vox_phrase":
            label, parts = f"Vox phrase {n}", [stem_title, "Phrase", str(n)]
        else:  # stabs and one-shots from other / guitar / piano
            noun = "Stab" if c.category == "stab" else "Hit"
            label = f"{stem_title} {noun.lower()} {n}" + (f" {note}" if note else "")
            parts = [stem_title, noun, str(n), note]
        return label, "_".join(p for p in parts if p) + self.suffix


def letters(n: int) -> str:
    out = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = string.ascii_uppercase[r] + out
    return out


def describe_pitch(c: Candidate, y: np.ndarray, sr: int) -> None:
    """Note names for bass notes and vocal chops, chord names for stabs. Only run on kept slices."""
    x = c.audio(y)
    if c.category in ("bass_note", "vox_chop") and len(x) > sr * 0.05:
        fmin, fmax = (30.0, 400.0) if c.stem == "bass" else (80.0, 1000.0)
        target = 22050
        xs = librosa.resample(x, orig_sr=sr, target_sr=target) if sr != target else x
        frame = 4096 if c.stem == "bass" else 2048
        if len(xs) < frame:
            xs = np.pad(xs, (0, frame - len(xs)))
        f0, voiced, _ = librosa.pyin(xs, fmin=fmin, fmax=fmax, sr=target, frame_length=frame)
        if np.mean(voiced) >= 0.4 and np.any(voiced):
            c.note = note_name(float(np.nanmedian(f0[voiced])))
    elif c.category == "stab" and len(x) > 2048:
        c.note = chord_name(x, sr)


def note_name(hz: float) -> str:
    """Note name spelled with flats, e.g. "Ab2" (no "#", which some samplers and file systems dislike)."""
    midi = int(round(librosa.hz_to_midi(hz)))
    return f"{PITCH_NAMES[midi % 12]}{midi // 12 - 1}"


def chord_name(x: np.ndarray, sr: int) -> str | None:
    chroma = librosa.feature.chroma_stft(y=x, sr=sr, n_fft=4096 if len(x) >= 4096 else 2048).mean(axis=1)
    if chroma.max() < 1e-6:
        return None
    best, name = -1.0, None
    for root in range(12):
        for quality, third in (("maj", 4), ("min", 3)):
            template = np.zeros(12)
            template[[root, (root + third) % 12, (root + 7) % 12]] = 1.0
            score = float(np.dot(chroma, template) / (np.linalg.norm(chroma) * np.sqrt(3)))
            if score > best:
                best, name = score, f"{PITCH_NAMES[root]}{quality}"
    return name


def candidate_json(c: Candidate, sr: int) -> dict:
    return {
        "id": c.id,
        "stem": c.stem,
        "kind": c.kind,
        "category": c.category,
        "start_s": round(c.start / sr, 4),
        "end_s": round(c.end / sr, 4),
        "bars": c.bars,
        "beats": c.beats,
        "score": c.score,
        "score_parts": c.parts,
        "cluster": c.cluster,
        "representative": c.representative,
    }

"""Shared pieces for the per-stem candidate extractors."""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from sangisa.config import Config

HOP = 512


@dataclass
class Candidate:
    stem: str
    kind: str       # one_shot | loop | phrase | chop
    category: str   # kick, snare, hat, perc, drum_loop, bass_note, bass_loop, vox_phrase, vox_chop, stab, other_loop
    start: int      # samples into the stem
    end: int
    bars: float | None = None
    beats: float | None = None
    note: str | None = None
    # Filled in by scoring / picking.
    id: str = ""
    parts: dict[str, float | None] = field(default_factory=dict)
    score: float = 0.0
    cluster: int = -1
    representative: bool = True
    similarity: float = 0.0

    def audio(self, y: np.ndarray) -> np.ndarray:
        return y[..., self.start : self.end]


@dataclass
class Context:
    """Everything an extractor may need besides its own stem."""

    sr: int
    bpm: float
    beats: np.ndarray       # seconds
    downbeats: np.ndarray   # seconds
    cfg: Config

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm

    @property
    def xcfg(self) -> Config:
        return self.cfg["extract"]


def onsets(y: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Onset positions in samples, backtracked to the preceding energy minimum."""
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=HOP, backtrack=True, units="frames", **kwargs
    )
    return np.unique(librosa.frames_to_samples(frames, hop_length=HOP))


def one_shots(y: np.ndarray, ctx: Context, stem: str, kind: str, category: str,
              min_s: float | None = None) -> list[Candidate]:
    """Onset to next onset, capped at ``one_shot_max_s``, with trailing silence trimmed off."""
    sr, n = ctx.sr, len(y)
    max_len = int(ctx.xcfg["one_shot_max_s"] * sr)
    min_len = int((ctx.xcfg["one_shot_min_s"] if min_s is None else min_s) * sr)
    starts = onsets(y, sr)
    out = []
    for i, s in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else n
        end = trim_tail(y, int(s), int(min(nxt, s + max_len, n)), ctx.xcfg["tail_trim_db"])
        if end - s >= min_len:
            out.append(Candidate(stem, kind, category, int(s), end))
    return out


def trim_tail(y: np.ndarray, start: int, end: int, below_db: float, hop: int = 256) -> int:
    """Move ``end`` back to where the sound has decayed ``below_db`` under its peak."""
    x = np.abs(y[start:end])
    if len(x) < hop * 2:
        return end
    frames = x[: len(x) // hop * hop].reshape(-1, hop).max(axis=1)
    loud = np.nonzero(frames > frames.max() * 10 ** (-below_db / 20))[0]
    if loud.size == 0:
        return end
    return start + min(len(x), (int(loud[-1]) + 2) * hop)


def grid_loops(y: np.ndarray, ctx: Context, stem: str, category: str, bar_counts: list[int]) -> list[Candidate]:
    """Loops of N bars starting on every downbeat. Lengths come from the tempo so every loop is exact."""
    sr, n = ctx.sr, len(y)
    bar_s = 4 * ctx.beat_s
    out = []
    for bars in bar_counts:
        length = int(round(bars * bar_s * sr))
        for t in ctx.downbeats:
            s = int(round(t * sr))
            if s + length <= n:
                out.append(Candidate(stem, "loop", category, s, s + length, bars=float(bars), beats=bars * 4.0))
    return out


def cap(cands: list[Candidate], y: np.ndarray, limit: int) -> list[Candidate]:
    """Keep the ``limit`` loudest candidates, so a busy hi-hat track doesn't produce thousands."""
    if len(cands) <= limit:
        return cands
    energy = [float(np.mean(np.square(c.audio(y)))) for c in cands]
    keep = np.argsort(energy)[::-1][:limit]
    return [cands[i] for i in sorted(keep)]

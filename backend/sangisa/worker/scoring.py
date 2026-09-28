"""Candidate scoring (0-1 per part, weighted) and de-duplication."""

from __future__ import annotations

import itertools

import librosa
import numpy as np

from sangisa.audio import db, rms
from sangisa.config import Config

from .extractors.base import Candidate

EPS = 1e-10
ENV_HOP = 256
PARTS = ("isolation", "clarity", "loudness", "loopability", "uniqueness")


class EnergyIndex:
    """O(1) energy of any window of any stem, via cumulative sums of squares."""

    def __init__(self, stems: dict[str, np.ndarray]):
        self.cums = {name: np.concatenate([[0.0], np.cumsum(np.square(y, dtype=np.float64))])
                     for name, y in stems.items()}

    def energy(self, stem: str, start: int, end: int) -> float:
        c = self.cums[stem]
        end = min(end, len(c) - 1)
        return float(c[end] - c[start]) if end > start else 0.0

    def isolation(self, stem: str, start: int, end: int) -> float:
        total = sum(self.energy(s, start, end) for s in self.cums)
        return self.energy(stem, start, end) / total if total > EPS else 0.0


def envelope(x: np.ndarray) -> np.ndarray:
    if len(x) < ENV_HOP * 2:
        return np.array([rms(x)])
    return librosa.feature.rms(y=x, frame_length=ENV_HOP * 2, hop_length=ENV_HOP, center=False)[0]


def clarity(c: Candidate, x: np.ndarray, sr: int) -> float:
    env = envelope(x)
    if c.kind == "one_shot":
        # A clean hit starts sharply and dies away inside the slice. A slice that is
        # still loud at its end was cut off mid-sound (false onset, end of file).
        attack_frames = max(1, int(0.03 * sr / ENV_HOP))
        ratio = float(env[:attack_frames].max() / (env.mean() + EPS))
        transient = float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))
        tail = float(env[-max(1, len(env) // 8):].mean() / (env.max() + EPS))
        decay = float(np.clip(1.0 - 2.0 * tail, 0.0, 1.0))
        return 0.5 * transient + 0.5 * decay
    if c.kind == "loop":
        # Steady level: low variation between beats.
        per_beat = [seg.mean() for seg in np.array_split(env, max(1, int(round(c.beats or 4))))]
        per_beat = np.asarray(per_beat)
        cv = float(per_beat.std() / (per_beat.mean() + EPS))
        return float(np.clip(1.0 - cv, 0.0, 1.0))
    # Phrases and chops: mostly sound, not mostly silence.
    return float(np.mean(env > env.max() * 0.1)) if env.max() > EPS else 0.0


def loudness(x: np.ndarray, cfg: Config) -> float:
    scfg = cfg["scoring"]
    level = db(rms(x))
    clipped = float(np.mean(np.abs(x) >= 0.999))
    if level < scfg["min_rms_dbfs"] or clipped > scfg["max_clipped_fraction"]:
        return 0.0
    return float(np.clip((level - scfg["min_rms_dbfs"]) / (-18.0 - scfg["min_rms_dbfs"]), 0.0, 1.0))


def loopability(x: np.ndarray, sr: int) -> float:
    """How seamlessly the end runs back into the start: matching level and timbre at the seam."""
    edge = min(int(0.05 * sr), len(x) // 4)
    if edge < 256:
        return 0.0
    head, tail = x[:edge], x[-edge:]
    level_match = float(np.exp(-abs(np.log((rms(tail) + EPS) / (rms(head) + EPS)))))
    mh = librosa.feature.mfcc(y=head, sr=sr, n_mfcc=13, n_fft=min(2048, edge), hop_length=edge // 4).mean(axis=1)
    mt = librosa.feature.mfcc(y=tail, sr=sr, n_mfcc=13, n_fft=min(2048, edge), hop_length=edge // 4).mean(axis=1)
    timbre = float(np.dot(mh, mt) / (np.linalg.norm(mh) * np.linalg.norm(mt) + EPS))
    return float(np.clip(0.5 * level_match + 0.5 * max(timbre, 0.0), 0.0, 1.0))


def timbre_vector(x: np.ndarray, sr: int) -> np.ndarray:
    """MFCC mean and spread, without MFCC 0 (overall level), so a quieter repeat still matches."""
    n_fft = 2048 if len(x) >= 2048 else 1 << max(8, int(np.log2(max(len(x), 256))))
    m = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=n_fft // 4)[1:]
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 for identical timbre, falling towards 0 as the relative distance grows."""
    dist = np.linalg.norm(a - b) / (0.5 * (np.linalg.norm(a) + np.linalg.norm(b)) + EPS)
    return float(max(0.0, 1.0 - dist))


def weighted(parts: dict[str, float | None], weights: dict[str, float]) -> float:
    used = {k: v for k, v in parts.items() if v is not None and weights.get(k, 0) > 0}
    total_w = sum(weights[k] for k in used)
    if total_w == 0:
        return 0.0
    score = sum(weights[k] * used[k] for k in used) / total_w
    return 0.0 if parts.get("loudness") == 0.0 else float(score)  # silent / clipped slices are out


def score_all(cands: list[Candidate], stems: dict[str, np.ndarray], sr: int, cfg: Config) -> None:
    """Fill in ``parts``, ``score``, ``cluster`` and ``representative`` on every candidate in place."""
    weights = cfg["scoring"]["weights"]
    index = EnergyIndex(stems)
    vectors: dict[int, np.ndarray] = {}

    for i, c in enumerate(cands):
        x = c.audio(stems[c.stem])
        c.parts = {
            "isolation": index.isolation(c.stem, c.start, c.end),
            "clarity": clarity(c, x, sr),
            "loudness": loudness(x, cfg),
            "loopability": loopability(x, sr) if c.kind == "loop" else None,
            "uniqueness": 1.0,
        }
        vectors[i] = timbre_vector(x, sr)

    # Preliminary score without uniqueness decides who represents each cluster.
    prelim_w = {k: v for k, v in weights.items() if k != "uniqueness"}
    prelim = [weighted({k: v for k, v in c.parts.items() if k != "uniqueness"}, prelim_w) for c in cands]

    threshold = cfg["scoring"]["duplicate_similarity"]
    next_cluster = itertools.count()
    groups: dict[tuple[str, str], list[int]] = {}
    for i, c in enumerate(cands):
        groups.setdefault((c.stem, c.category), []).append(i)

    for members in groups.values():
        reps: list[int] = []
        for i in sorted(members, key=lambda i: -prelim[i]):
            c = cands[i]
            sims = [similarity(vectors[i], vectors[r]) for r in reps]
            if sims and max(sims) >= threshold:
                rep = cands[reps[int(np.argmax(sims))]]
                c.cluster, c.representative, c.similarity = rep.cluster, False, max(sims)
                c.parts["uniqueness"] = float(np.clip(1.0 - max(sims), 0.0, 1.0))
                continue
            c.cluster, c.representative = next(next_cluster), True
            reps.append(i)

    for c in cands:
        c.parts = {k: (None if v is None else round(float(v), 4)) for k, v in c.parts.items()}
        c.score = round(weighted(c.parts, weights), 4)

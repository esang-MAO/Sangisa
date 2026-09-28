"""Vocals: phrases split on silence, plus short 1- and 2-beat chops on the beat grid."""

from __future__ import annotations

import librosa
import numpy as np

from .base import Candidate, Context, cap


def extract(y: np.ndarray, ctx: Context) -> list[Candidate]:
    limit = ctx.xcfg["max_candidates_per_kind"]
    return cap(phrases(y, ctx), y, limit) + cap(chops(y, ctx), y, limit)


def phrases(y: np.ndarray, ctx: Context) -> list[Candidate]:
    sr, x = ctx.sr, ctx.xcfg
    min_len, max_len = int(x["phrase_min_s"] * sr), int(x["phrase_max_s"] * sr)
    out = []
    for s, e in librosa.effects.split(y, top_db=x["silence_top_db"], frame_length=2048, hop_length=512):
        if e - s < min_len:
            continue
        # Long stretches without a gap get cut into bar-sized pieces rather than dropped.
        step = max_len if e - s > max_len else e - s
        for a in range(int(s), int(e), step):
            b = min(a + step, int(e))
            if b - a >= min_len:
                out.append(Candidate("vocals", "phrase", "vox_phrase", a, b))
    return out


def chops(y: np.ndarray, ctx: Context) -> list[Candidate]:
    sr, n = ctx.sr, len(y)
    active = librosa.effects.split(y, top_db=ctx.xcfg["silence_top_db"], frame_length=2048, hop_length=512)
    out = []
    for beats in ctx.xcfg["vocal_chop_beats"]:
        length = int(round(beats * ctx.beat_s * sr))
        for t in ctx.beats:
            s = int(round(t * sr))
            if s + length > n:
                continue
            # Only chop where the singer is actually singing at the start of the chop.
            if any(a <= s < b for a, b in active):
                out.append(Candidate("vocals", "chop", "vox_chop", s, s + length, beats=float(beats)))
    return out

"""Bass: single notes at onsets and 1-, 2- and 4-bar loops on the beat grid."""

from __future__ import annotations

import numpy as np

from .base import Candidate, Context, cap, grid_loops, one_shots


def extract(y: np.ndarray, ctx: Context) -> list[Candidate]:
    limit = ctx.xcfg["max_candidates_per_kind"]
    notes = cap(one_shots(y, ctx, "bass", "one_shot", "bass_note", min_s=ctx.xcfg["note_min_s"]), y, limit)
    loops = cap(grid_loops(y, ctx, "bass", "bass_loop", ctx.xcfg["bass_loop_bars"]), y, limit)
    return notes + loops

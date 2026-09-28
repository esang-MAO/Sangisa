"""Other (keys, synths, guitars, anything left): chord stabs at onsets and 2- and 4-bar loops."""

from __future__ import annotations

import numpy as np

from .base import Candidate, Context, cap, grid_loops, one_shots


def extract(y: np.ndarray, ctx: Context, stem: str = "other") -> list[Candidate]:
    limit = ctx.xcfg["max_candidates_per_kind"]
    stabs = cap(one_shots(y, ctx, stem, "one_shot", "stab", min_s=ctx.xcfg["note_min_s"]), y, limit)
    loops = cap(grid_loops(y, ctx, stem, f"{stem}_loop", ctx.xcfg["other_loop_bars"]), y, limit)
    return stabs + loops

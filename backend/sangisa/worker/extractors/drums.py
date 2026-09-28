"""Drums: one-shots cut at onsets and classified as kick / snare / hat / perc, plus 1- and 2-bar loops."""

from __future__ import annotations

import numpy as np

from .base import Candidate, Context, cap, grid_loops, one_shots

ANALYSIS_MS = 80  # classify on the attack, not the tail


def extract(y: np.ndarray, ctx: Context) -> list[Candidate]:
    hits = cap(one_shots(y, ctx, "drums", "one_shot", "perc"), y, ctx.xcfg["max_candidates_per_kind"])
    for c in hits:
        c.category = classify(c.audio(y)[: int(ctx.sr * ANALYSIS_MS / 1000)], ctx.sr)
    loops = cap(grid_loops(y, ctx, "drums", "drum_loop", ctx.xcfg["drum_loop_bars"]), y,
                ctx.xcfg["max_candidates_per_kind"])
    return hits + loops


def band_features(x: np.ndarray, sr: int) -> dict[str, float]:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=max(2048, len(x)))) ** 2
    freqs = np.fft.rfftfreq(max(2048, len(x)), 1 / sr)
    total = float(spec.sum()) + 1e-12
    # Noisiness measured where snares and claps live, so silence elsewhere doesn't mask it.
    mag = np.sqrt(spec[(freqs > 200) & (freqs < 10000)]) + 1e-12
    return {
        "low": float(spec[freqs < 150].sum()) / total,
        "high": float(spec[freqs > 5000].sum()) / total,
        "centroid": float((freqs * spec).sum() / total),
        "flatness": float(np.exp(np.mean(np.log(mag))) / np.mean(mag)),
    }


def classify(x: np.ndarray, sr: int) -> str:
    """Rule-based drum classifier on spectral centroid and band energy."""
    if len(x) < 64:
        return "perc"
    f = band_features(x, sr)
    if f["low"] > 0.45 and f["centroid"] < 1000:
        return "kick"
    if f["high"] > 0.5 or f["centroid"] > 6000:
        return "hat"
    if f["flatness"] > 0.15 and 800 < f["centroid"] <= 6000:
        return "snare"
    return "perc"

"""Small audio helpers shared by the stages. Audio arrays are float32, shaped (channels, samples)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

EPS = 1e-10


def load(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), sr


def save(path: str | Path, audio: np.ndarray, sr: int, bit_depth: int = 24) -> None:
    subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}[bit_depth]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32).T, sr, subtype=subtype)


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + EPS)) if x.size else 0.0


def db(value: float) -> float:
    return float(20.0 * np.log10(max(value, EPS)))


def nearest_zero_crossing(mono: np.ndarray, index: int, window: int) -> int:
    """Index of the sign change closest to ``index`` within ``window`` samples, else ``index``."""
    lo, hi = max(index - window, 1), min(index + window, len(mono) - 1)
    if hi <= lo:
        return index
    seg = mono[lo - 1 : hi + 1]
    crossings = np.nonzero(np.signbit(seg[:-1]) != np.signbit(seg[1:]))[0] + lo
    if crossings.size == 0:
        return index
    return int(crossings[np.argmin(np.abs(crossings - index))])


def apply_fades(audio: np.ndarray, sr: int, fade_in_ms: float, fade_out_ms: float) -> np.ndarray:
    out = audio.copy()
    n = out.shape[-1]
    fi = min(int(sr * fade_in_ms / 1000), n // 2)
    fo = min(int(sr * fade_out_ms / 1000), n // 2)
    if fi > 0:
        out[..., :fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        out[..., n - fo :] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return out


def peak_normalize(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio
    return audio * np.float32(10 ** (target_dbfs / 20) / peak)


def fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Pad with silence or trim so the last axis is exactly ``n`` samples."""
    cur = audio.shape[-1]
    if cur == n:
        return audio
    if cur > n:
        return audio[..., :n]
    pad = [(0, 0)] * (audio.ndim - 1) + [(0, n - cur)]
    return np.pad(audio, pad)

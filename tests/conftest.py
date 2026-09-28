"""Test fixtures: a short synthetic "song" with known tempo, key and stems.

16 bars at 120 BPM in F minor (32 s). Because we build the stems ourselves,
the tests swap in a "fixture" separation backend that hands them back, so no
model download or copyrighted audio is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfilt

from sangisa.config import load_config
from sangisa.worker.stages.separate import register_separator

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM
BARS = 16
DURATION = BARS * 4 * BEAT


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def kick(sr: int = SR) -> np.ndarray:
    t = np.arange(int(0.35 * sr)) / sr
    freq = 45 + 110 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 9)


def snare(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    t = np.arange(int(0.2 * sr)) / sr
    noise = sosfilt(butter(4, [1000, 5000], btype="band", fs=sr, output="sos"), rng.standard_normal(len(t)))
    noise /= np.max(np.abs(noise))
    return (0.8 * noise + 0.3 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 22)


def hat(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    t = np.arange(int(0.06 * sr)) / sr
    noise = rng.standard_normal(len(t))
    for _ in range(3):  # repeated first differences push the energy up high
        noise = np.diff(noise, prepend=0.0)
    return noise / np.max(np.abs(noise)) * np.exp(-t * 70)


def tone(freq: float, dur: float, sr: int = SR, harmonics: int = 3, decay: float = 0.0) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    x = sum(np.sin(2 * np.pi * freq * (h + 1) * t) / (h + 1) for h in range(harmonics))
    env = np.exp(-t * decay) if decay else np.ones_like(t)
    fade = min(len(t) // 2, int(0.005 * sr))
    env[:fade] *= np.linspace(0, 1, fade)
    env[-fade:] *= np.linspace(1, 0, fade)
    return x * env


def place(track: np.ndarray, sound: np.ndarray, at_s: float) -> None:
    i = int(round(at_s * SR))
    n = min(len(sound), len(track) - i)
    if n > 0:
        track[i : i + n] += sound[:n]


def make_stems(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(DURATION * SR)
    drums, bass, vocals, other = (np.zeros(n) for _ in range(4))
    k, s = kick(), snare(rng)
    # F minor: F2, Ab2, C3, Eb2 bass roots; F minor / Db major / Eb major / C minor chords.
    roots = [41, 44, 48, 39]
    chords = [[65, 68, 72], [61, 65, 68], [63, 67, 70], [60, 63, 67]]
    melody = [72, 75, 77, 68]
    for bar in range(BARS):
        t0 = bar * 4 * BEAT
        for beat in range(4):
            tb = t0 + beat * BEAT
            place(drums, k * 0.9 if beat in (0, 2) else s * 0.6, tb)
            place(drums, hat(rng) * 0.25, tb + BEAT / 2)
            place(drums, hat(rng) * 0.2, tb)
            place(bass, tone(midi_hz(roots[bar % 4]), BEAT * 0.9, harmonics=2) * 0.5, tb)
        chord = sum(tone(midi_hz(m), 1.2, decay=3.0) for m in chords[bar % 4]) / 3
        place(other, chord * 0.5, t0)
        place(other, chord * 0.35, t0 + 2.5 * BEAT)
        if bar % 2 == 0:  # a 1.5-bar vocal phrase, then silence
            for i, m in enumerate([melody[bar % 4], melody[(bar + 1) % 4], melody[(bar + 2) % 4]]):
                t = np.arange(int(2 * BEAT * SR)) / SR
                vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t)
                phase = 2 * np.pi * np.cumsum(midi_hz(m) * vib) / SR
                v = (np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase))
                env = np.minimum(1, np.minimum(t / 0.03, (t[-1] - t) / 0.05))
                place(vocals, v * env * 0.3, t0 + i * 2 * BEAT)
    return {"drums": drums, "bass": bass, "vocals": vocals, "other": other}


@pytest.fixture(scope="session")
def song(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Writes song.wav plus each stem; returns {"mix": path, "<stem>": path}."""
    root = tmp_path_factory.mktemp("song")
    stems = make_stems()
    mix = sum(stems.values())
    peak = np.max(np.abs(mix))
    paths = {}
    for name, y in {**stems, "mix": mix}.items():
        stereo = np.stack([y, y]).T / peak * 0.9
        path = root / ("song.wav" if name == "mix" else f"{name}.wav")
        sf.write(path, stereo, SR, subtype="PCM_24")
        paths[name] = path
    return paths


@pytest.fixture(scope="session")
def fixture_separator(song: dict[str, Path]) -> str:
    @register_separator("fixture")
    def _separate(path: Path, model: str, out_dir: Path, **_: object) -> dict[str, Path]:
        return {name: p for name, p in song.items() if name != "mix"}

    return "fixture"


@pytest.fixture
def cfg(fixture_separator: str) -> dict:
    return load_config(overrides={"separation": {"backend": fixture_separator}})

"""Stage 3: tempo, beat grid, downbeats, key and sections, measured on the full mix."""

from __future__ import annotations

import string

import librosa
import numpy as np

from sangisa import audio
from sangisa.config import Config
from sangisa.job import Job, write_json

HOP = 512
PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Kessler key profiles.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def run(job: Job, cfg: Config) -> None:
    mix, sr = audio.load(job.work_wav)
    write_json(job.analysis_path, analyze(audio.to_mono(mix), sr, cfg))


def analyze(y: np.ndarray, sr: int, cfg: Config) -> dict:
    acfg = cfg["analysis"]
    duration = len(y) / sr
    bpm, beats = detect_beats(y, sr, acfg["min_bpm"], acfg["max_bpm"])
    downbeats = detect_downbeats(y, sr, beats)
    key, key_short, key_conf = detect_key(y, sr)
    return {
        "duration_s": round(duration, 3),
        "bpm": round(bpm, 2),
        "key": key,
        "key_short": key_short,
        "key_confidence": round(key_conf, 3),
        "time_signature": "4/4",  # assumed; downbeats are placed every 4 beats
        "beats_s": [round(float(b), 4) for b in beats],
        "downbeats_s": [round(float(b), 4) for b in downbeats],
        "sections": detect_sections(y, sr, beats, duration),
    }


def detect_beats(y: np.ndarray, sr: int, min_bpm: float, max_bpm: float) -> tuple[float, np.ndarray]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    if bpm <= 0 or len(beats) < 2:
        return 120.0, np.arange(0, len(y) / sr, 0.5)

    # Fold into the preferred range, fixing the grid to match.
    while bpm < min_bpm:
        bpm *= 2
        mids = (beats[:-1] + beats[1:]) / 2
        beats = np.sort(np.concatenate([beats, mids]))
    while bpm > max_bpm:
        bpm /= 2
        beats = beats[::2]

    # Report the tempo implied by the grid itself: a straight-line fit over every beat
    # averages out the frame quantisation that a single beat-to-beat gap carries.
    if len(beats) > 4:
        period = float(np.polyfit(np.arange(len(beats)), beats, 1)[0])
        if period > 0:
            bpm = 60.0 / period
    return bpm, beats


def detect_downbeats(y: np.ndarray, sr: int, beats: np.ndarray, beats_per_bar: int = 4) -> np.ndarray:
    """Pick the bar phase where low-end hits land and the harmony changes (both favour the one)."""
    if len(beats) < beats_per_bar * 2:
        return beats[:1]
    frames = librosa.time_to_frames(beats, sr=sr, hop_length=HOP)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=64, fmax=8000)
    low = librosa.onset.onset_strength(S=librosa.power_to_db(mel[:8]), sr=sr, hop_length=HOP)
    low_at_beat = low[np.clip(frames, 0, len(low) - 1)]

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    synced = librosa.util.sync(chroma, frames)[:, 1 : len(beats) + 1]  # column i = beat i to beat i+1
    synced = synced / (np.linalg.norm(synced, axis=0, keepdims=True) + 1e-9)
    change = np.zeros(len(beats))
    change[1 : synced.shape[1]] = 1.0 - np.sum(synced[:, 1:] * synced[:, :-1], axis=0)

    def norm(v: np.ndarray) -> np.ndarray:
        return (v - v.mean()) / (v.std() + 1e-9)

    strength = norm(low_at_beat) + norm(change)
    scores = [strength[p::beats_per_bar].mean() for p in range(beats_per_bar)]
    return beats[int(np.argmax(scores)) :: beats_per_bar]


def detect_key(y: np.ndarray, sr: int) -> tuple[str | None, str | None, float]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP * 4)
    profile = chroma.mean(axis=1)
    if not np.any(profile > 1e-6):
        return None, None, 0.0
    scores = []
    for tonic in range(12):
        for mode, template in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            corr = np.corrcoef(profile, np.roll(template, tonic))[0, 1]
            scores.append((float(np.nan_to_num(corr)), tonic, mode))
    scores.sort(reverse=True)
    best, tonic, mode = scores[0]
    confidence = max(0.0, best - scores[1][0]) + max(0.0, best) * 0.5
    name = PITCH_NAMES[tonic]
    return f"{name} {mode}", f"{name}{'maj' if mode == 'major' else 'min'}", min(confidence, 1.0)


def detect_sections(y: np.ndarray, sr: int, beats: np.ndarray, duration: float) -> list[dict]:
    """Novelty-based segmentation on beat-synchronous chroma + MFCC. Repeated sections share a letter."""
    if len(beats) < 16 or duration < 20:
        return [{"label": "A", "start_s": 0.0, "end_s": round(duration, 3)}]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    frames = librosa.time_to_frames(beats, sr=sr, hop_length=HOP)
    feats = np.vstack([
        librosa.util.normalize(librosa.util.sync(chroma, frames), axis=0),
        librosa.util.normalize(librosa.util.sync(mfcc, frames), axis=1),
    ])
    k = int(np.clip(round(duration / 20), 2, 10))
    bounds = librosa.segment.agglomerative(feats, k)  # indexes into the synced columns
    edges = np.concatenate([[0], beats, [duration]])
    starts = [float(edges[b]) for b in bounds]
    ends = starts[1:] + [duration]

    # Give similar segments the same letter.
    means = [feats[:, bounds[i] : (bounds[i + 1] if i + 1 < len(bounds) else feats.shape[1])].mean(axis=1)
             for i in range(len(bounds))]
    letters: list[str] = []
    reps: list[np.ndarray] = []
    for m in means:
        sims = [float(np.dot(m, r) / (np.linalg.norm(m) * np.linalg.norm(r) + 1e-9)) for r in reps]
        if sims and max(sims) > 0.9:
            letters.append(string.ascii_uppercase[int(np.argmax(sims))])
        else:
            reps.append(m)
            letters.append(string.ascii_uppercase[len(reps) - 1])
    return [
        {"label": letter, "start_s": round(s, 3), "end_s": round(e, 3)}
        for letter, s, e in zip(letters, starts, ends, strict=True)
        if e - s > 0.5
    ]


def section_at(sections: list[dict], t: float) -> str | None:
    for s in sections:
        if s["start_s"] <= t < s["end_s"]:
            return s["label"]
    return None

"""Write librosa reference values for the JavaScript engine's parity tests.

    uv run python tests/engine/reference.py <out_dir>

Writes the synthetic test song (mix + stems) as WAVs and reference.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.conftest import SR, make_stems  # noqa: E402

from sangisa.config import default_config  # noqa: E402
from sangisa.worker.extractors.base import onsets  # noqa: E402
from sangisa.worker.stages.analyze import analyze  # noqa: E402


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stems = make_stems()
    mix = sum(stems.values())
    peak = float(np.max(np.abs(mix)))
    for name, y in {**stems, "mix": mix}.items():
        sf.write(out / f"{name}.wav", np.stack([y, y]).T / peak * 0.9, SR, subtype="FLOAT")

    y = (mix / peak * 0.9).astype(np.float32)
    short = y[: SR * 4]
    env = librosa.onset.onset_strength(y=short, sr=SR, hop_length=512)
    mel = librosa.filters.mel(sr=SR, n_fft=2048, n_mels=128)
    chroma = librosa.filters.chroma(sr=SR, n_fft=4096)
    mf = librosa.feature.mfcc(y=short, sr=SR, n_mfcc=20, n_fft=2048, hop_length=512)
    full_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512)
    tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=full_env, sr=SR, hop_length=512))[0])
    _, beats = librosa.beat.beat_track(onset_envelope=full_env, sr=SR, hop_length=512)
    drums = (stems["drums"] / peak * 0.9).astype(np.float32)
    ref = {
        "sr": SR,
        "mel_sum": float(mel.sum()),
        "mel_row_10": mel[10].tolist(),
        "chroma_sum": float(chroma.sum()),
        "chroma_row_0_first_400": chroma[0, :400].tolist(),
        "onset_env_4s": env.tolist(),
        "mfcc_4s_mean": mf.mean(axis=1).tolist(),
        "tempo": tempo,
        "beat_frames": beats.tolist(),
        "drum_onsets": onsets(drums, SR).tolist(),
        "split": librosa.effects.split(drums, top_db=35).tolist(),
        "analysis": analyze(y, SR, default_config()),
    }
    (out / "reference.json").write_text(json.dumps(ref))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

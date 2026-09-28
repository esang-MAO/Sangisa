import numpy as np
import pytest

from sangisa import audio
from sangisa.config import default_config
from sangisa.schema import RenderSettings, ScoreParts, Slice
from sangisa.worker.stages.render import render_slice

SR = 44100


def make_slice(**render) -> Slice:
    settings = dict(fade_in_ms=2, fade_out_ms=5, normalize_dbfs=-1.0, playback="one_shot") | render
    return Slice(id="x", stem="bass", kind="one_shot", category="bass_note", label="x", file="slices/x.wav",
                 source_start_s=0.5, source_end_s=1.0, score=1, cluster=0,
                 score_parts=ScoreParts(isolation=1, clarity=1, loudness=1, uniqueness=1),
                 render=RenderSettings(**settings))


def test_fades_and_normalize():
    t = np.arange(SR * 2) / SR
    stem = np.stack([np.sin(2 * np.pi * 100 * t) * 0.3] * 2).astype(np.float32)
    out = render_slice(stem, SR, make_slice(), default_config())
    assert out.shape[0] == 2
    assert out.shape[1] == pytest.approx(SR * 0.5, abs=SR * 0.004)
    assert np.max(np.abs(out)) == pytest.approx(10 ** (-1 / 20), abs=1e-3)
    assert abs(out[0, 0]) < 1e-3 and abs(out[0, -1]) < 1e-3


def test_reverse_and_no_normalize():
    t = np.arange(SR * 2) / SR
    stem = np.stack([t * 0.1] * 2).astype(np.float32)
    out = render_slice(stem, SR, make_slice(reverse=True, normalize_dbfs=None, fade_in_ms=0, fade_out_ms=0),
                       default_config())
    assert out[0, 0] > out[0, -1]
    assert np.max(np.abs(out)) < 0.11


def test_zero_crossing():
    x = np.array([-1, -1, -1, 1, 1, 1, 1, -1], dtype=np.float32)
    assert audio.nearest_zero_crossing(x, 1, 3) == 3
    assert audio.nearest_zero_crossing(x, 6, 3) == 7

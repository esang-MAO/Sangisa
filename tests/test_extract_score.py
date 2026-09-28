import numpy as np
import pytest

from sangisa.config import default_config
from sangisa.worker.extractors import Candidate, Context, drums
from sangisa.worker.scoring import loopability, loudness, score_all

from .conftest import SR, hat, kick, snare, tone


@pytest.mark.parametrize("name, expected", [
    ("kick", "kick"), ("snare", "snare"), ("hat", "hat"), ("tom", "perc"), ("block", "perc"),
])
def test_drum_classifier(name, expected):
    rng = np.random.default_rng(0)
    sound = {"kick": kick(), "snare": snare(rng), "hat": hat(rng),
             "tom": tone(180, 0.3, decay=10), "block": tone(900, 0.1, decay=30)}[name]
    assert drums.classify(sound[: int(SR * 0.08)], SR) == expected


def test_silent_and_clipped_slices_score_zero():
    cfg = default_config()
    assert loudness(np.zeros(4410), cfg) == 0.0
    assert loudness(np.ones(4410), cfg) == 0.0
    assert loudness(np.sin(np.linspace(0, 400, 4410)) * 0.3, cfg) > 0.5


def test_seamless_loop_beats_a_broken_one():
    t = np.arange(SR * 2) / SR
    steady = np.sin(2 * np.pi * 110 * t) * 0.5
    broken = steady * np.linspace(1, 0.05, len(t))
    assert loopability(steady, SR) > 0.9
    assert loopability(broken, SR) < loopability(steady, SR) - 0.3


def test_repeated_hits_collapse_into_one_cluster():
    rng = np.random.default_rng(1)
    k = kick()
    s = np.resize(snare(rng), len(k))
    y = np.zeros(SR * 4)
    starts = []
    for i in range(6):
        pos = i * SR // 2
        y[pos : pos + len(k)] += k if i % 2 == 0 else s
        starts.append(pos)
    cands = [Candidate("drums", "one_shot", "kick" if i % 2 == 0 else "snare", p, p + len(k))
             for i, p in enumerate(starts)]
    score_all(cands, {"drums": y}, SR, default_config())
    kicks = [c for c in cands if c.category == "kick"]
    assert sum(c.representative for c in kicks) == 1
    assert len({c.cluster for c in kicks}) == 1
    dup = next(c for c in kicks if not c.representative)
    assert dup.parts["uniqueness"] < 0.1


def test_isolation_penalises_bleed():
    t = np.arange(SR) / SR
    loud = np.sin(2 * np.pi * 220 * t) * 0.5
    cands = [Candidate("bass", "one_shot", "bass_note", 0, SR)]
    score_all(cands, {"bass": loud, "other": loud * 0.01}, SR, default_config())
    clean = cands[0].parts["isolation"]
    cands = [Candidate("bass", "one_shot", "bass_note", 0, SR)]
    score_all(cands, {"bass": loud, "other": loud}, SR, default_config())
    assert clean > 0.99 and cands[0].parts["isolation"] == pytest.approx(0.5, abs=0.01)


def test_grid_loops_have_exact_length():
    from sangisa.worker.extractors.base import grid_loops

    ctx = Context(sr=SR, bpm=90.0, beats=np.arange(0, 20, 60 / 90), downbeats=np.arange(0, 20, 4 * 60 / 90),
                  cfg=default_config())
    loops = grid_loops(np.zeros(SR * 20), ctx, "bass", "bass_loop", [1, 2])
    assert {c.end - c.start for c in loops} == {round(SR * 4 * 60 / 90), round(SR * 8 * 60 / 90)}

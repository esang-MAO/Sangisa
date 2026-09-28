import numpy as np
import pytest

from sangisa.worker.stages.analyze import analyze, section_at

from .conftest import SR, make_stems


@pytest.fixture(scope="module")
def result():
    from sangisa.config import default_config

    mix = sum(make_stems().values())
    return analyze(mix / np.max(np.abs(mix)), SR, default_config())


def test_tempo(result):
    assert result["bpm"] == pytest.approx(120, abs=0.5)
    assert np.median(np.diff(result["beats_s"])) == pytest.approx(0.5, abs=0.02)


def test_downbeats_land_on_the_kick(result):
    downbeats = np.asarray(result["downbeats_s"])
    assert np.median(np.diff(downbeats)) == pytest.approx(2.0, abs=0.05)
    # Bars start every 2 s at 0, 2, 4, ...
    offsets = np.abs(((downbeats + 1.0) % 2.0) - 1.0)
    assert np.median(offsets) < 0.05


def test_key(result):
    assert result["key"] == "F minor"
    assert result["key_short"] == "Fmin"


def test_sections_cover_the_song(result):
    secs = result["sections"]
    assert secs[0]["start_s"] == 0.0
    assert secs[-1]["end_s"] == pytest.approx(32.0, abs=0.1)
    assert section_at(secs, 1.0) == secs[0]["label"]

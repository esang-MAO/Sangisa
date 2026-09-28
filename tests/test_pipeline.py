"""End-to-end: synthetic song in, 16-pad kit out."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sangisa.cli import main
from sangisa.job import Job
from sangisa.pipeline import run_pipeline
from sangisa.schema import Kit

from .conftest import SR


@pytest.fixture(scope="module")
def built_job(tmp_path_factory, song, fixture_separator) -> Job:
    from sangisa.config import load_config

    cfg = load_config(overrides={"separation": {"backend": fixture_separator}})
    job = Job(tmp_path_factory.mktemp("job"))
    run_pipeline(job, cfg, source=str(song["mix"]), rights_confirmed=True)
    return job


def test_every_stage_writes_its_output(built_job: Job):
    m = built_job.manifest()
    assert all(m["stages"][s]["status"] == "done" for s in ("ingest", "separate", "analyze", "pick", "render"))
    assert built_job.work_wav.exists()
    assert sorted(p.stem for p in built_job.stems_dir.glob("*.wav")) == ["bass", "drums", "other", "vocals"]
    assert built_job.analysis_path.exists() and built_job.candidates_path.exists()
    assert m["source"]["rights_confirmed"] is True
    assert len(m["source"]["sha256"]) == 64


def test_kit_json_is_valid_and_complete(built_job: Job):
    kit = Kit.load(built_job.kit_path)
    assert kit.schema_version == 1
    assert len(kit.pads) == 16
    assert [p.pad for p in kit.pads] == list(range(1, 17))
    assert [p.midi_note for p in kit.pads] == list(range(36, 52))
    assert all(p.slice_id for p in kit.pads)
    assert len({p.slice_id for p in kit.pads}) == 16

    stems = [s.stem for _, s in kit.pad_slices()]
    assert stems == ["drums"] * 6 + ["vocals"] * 4 + ["bass"] * 3 + ["other"] * 3
    # Standard drum layout on the bottom row: kick, snare, hat, perc-or-next.
    cats = [s.category for _, s in kit.pad_slices()][:3]
    assert cats == ["kick", "snare", "hat"]
    # Backups are offered as swaps.
    assert len(kit.slices) > 16
    assert kit.analysis.bpm == pytest.approx(120, abs=0.5)


def test_rendered_slices(built_job: Job):
    kit = Kit.load(built_job.kit_path)
    for s in kit.slices:
        path = built_job.abs(s.file)
        info = sf.info(str(path))
        assert info.samplerate == SR and info.subtype == "PCM_24" and info.channels == 2
        assert "120bpm" in path.name
        data, _ = sf.read(str(path))
        if s.render.playback == "loop":
            # Loops are exactly N bars long so they sit on the grid.
            assert info.frames == pytest.approx(s.beats * 60 / kit.analysis.bpm * SR, abs=3)
        if s.render.normalize_dbfs is not None:
            assert 20 * np.log10(np.max(np.abs(data))) == pytest.approx(-1.0, abs=0.05)
        assert np.max(np.abs(data[:2])) < 0.05  # faded in


def test_labels_and_filenames(built_job: Job):
    kit = Kit.load(built_job.kit_path)
    labels = [s.label for s in kit.slices]
    assert "Kick 1" in labels
    assert len(set(labels)) == len(labels)
    assert len({s.file for s in kit.slices}) == len(kit.slices)
    bass_loop = next(s for s in kit.slices if s.category == "bass_loop")
    assert Path(bass_loop.file).name.startswith("Bass_Loop_")
    assert f"_{bass_loop.bars:g}bar_" in bass_loop.file


def test_bass_notes_are_named(built_job: Job):
    kit = Kit.load(built_job.kit_path)
    notes = {s.note for s in kit.slices if s.category == "bass_note" and s.note}
    # The bass line plays F2, Ab2, C3 and Eb2.
    assert notes and notes <= {"F2", "Ab2", "C3", "Eb2"}


def test_completed_stages_are_skipped(built_job: Job, cfg):
    before = json.loads(built_job.manifest_path.read_text())["stages"]
    calls = []
    run_pipeline(built_job, cfg, progress=lambda st, status, _: calls.append(st))
    assert calls == []
    assert json.loads(built_job.manifest_path.read_text())["stages"] == before


def test_single_stage_rerun(tmp_path, song, cfg):
    job = Job(tmp_path / "job")
    run_pipeline(job, cfg, source=str(song["mix"]), rights_confirmed=True)
    split = {"drums": 4, "bass": 4, "vocals": 4, "other": 4}
    cfg2 = {**cfg, "kit": {**cfg["kit"], "pad_split": split}}
    calls = []
    run_pipeline(job, cfg2, start_at="pick", progress=lambda st, status, _: calls.append((st, status)))
    assert [c for c in calls if c[1] == "start"] == [("pick", "start"), ("render", "start")]
    kit = Kit.load(job.kit_path)
    assert [s.stem for _, s in kit.pad_slices()] == ["drums"] * 4 + ["bass"] * 4 + ["vocals"] * 4 + ["other"] * 4


def test_stage_order_is_enforced(tmp_path, cfg):
    with pytest.raises(ValueError, match="before"):
        run_pipeline(Job(tmp_path), cfg, only="pick")


def test_cli(tmp_path, song, fixture_separator, capsys):
    conf = tmp_path / "c.toml"
    conf.write_text(f'[separation]\nbackend = "{fixture_separator}"\n')
    out = tmp_path / "kit"
    assert main([str(song["mix"]), "--out", str(out), "--config", str(conf)]) == 1
    assert "rights" in capsys.readouterr().err

    assert main([str(song["mix"]), "--out", str(out), "--config", str(conf), "--i-have-rights"]) == 0
    printed = capsys.readouterr().out
    assert "Separating stems" in printed and "pad 16" in printed
    assert (out / "kit.json").exists()

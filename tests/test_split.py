"""Acapella + instrumental mode, HQ vocals kits, and reuse between them."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from sangisa.cli import main
from sangisa.config import merge
from sangisa.job import Job
from sangisa.pipeline import run_pipeline
from sangisa.schema import Kit

from . import conftest

ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


@pytest.fixture(autouse=True)
def clear_calls():
    conftest.SEPARATOR_CALLS.clear()


def with_(cfg, **sections):
    return merge(cfg, sections)


def read(path):
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return data, sr


def split_files(job: Job) -> dict:
    return json.loads((job.root / "split.json").read_text())


def test_split_writes_an_acapella_and_instrumental_that_sum_to_the_original(tmp_path, song, cfg):
    job = Job(tmp_path / "job")
    run_pipeline(job, cfg, source=str(song["mix"]), rights_confirmed=True, mode="split")
    info = split_files(job)
    aca, ins = job.abs(info["files"]["acapella"]), job.abs(info["files"]["instrumental"])
    assert aca.name == "song - Acapella - 120bpm Fmin.wav"
    assert ins.name == "song - Instrumental - 120bpm Fmin.wav"
    a, sr = read(aca)
    i, _ = read(ins)
    mix, msr = read(job.work_wav)
    assert sr == msr == 44100 and len(a) == len(i) == len(mix)
    assert sf.info(str(aca)).subtype == "PCM_24"
    assert np.max(np.abs(a + i - mix)) < 1e-5  # adds back up (24-bit rounding only)
    truth, _ = read(song["vocals"])
    assert np.max(np.abs(a - truth[: len(a)])) < 1e-5
    # Only the vocal model ran: no 4-stem pass, no kit stages.
    assert conftest.SEPARATOR_CALLS == [(ROFORMER, "work.wav")]
    assert not job.is_done("pick") and not job.kit_path.exists()


def test_make_a_kit_from_a_split_reuses_the_acapella(tmp_path, song, cfg):
    job = Job(tmp_path / "job")
    run_pipeline(job, cfg, source=str(song["mix"]), rights_confirmed=True, mode="split")
    run_pipeline(job, cfg, mode="kit")
    # The vocal model ran once; the 4-stem model only on the instrumental.
    assert conftest.SEPARATOR_CALLS == [(ROFORMER, "work.wav"), ("htdemucs_ft.yaml", "instrumental.wav")]
    kit = Kit.load(job.kit_path)
    assert kit.separation.vocal_model == ROFORMER
    assert sorted(kit.separation.stems) == ["bass", "drums", "other", "vocals"]
    pair_vocals, _ = read(job.root / "pair/vocals.wav")
    kit_vocals, _ = read(job.abs(kit.separation.stems["vocals"]))
    assert np.max(np.abs(pair_vocals - kit_vocals)) < 1e-5
    assert len(kit.pads) == 16 and all(p.slice_id for p in kit.pads)
    # And the split files are still there.
    assert job.is_done("split")


def test_hq_vocals_kit(tmp_path, song, cfg):
    job = Job(tmp_path / "job")
    run_pipeline(job, with_(cfg, separation={"hq_vocals": True}), source=str(song["mix"]), rights_confirmed=True)
    assert conftest.SEPARATOR_CALLS == [(ROFORMER, "work.wav"), ("htdemucs_ft.yaml", "instrumental.wav")]
    kit = Kit.load(job.kit_path)
    assert kit.separation.vocal_model == ROFORMER
    assert any(s.stem == "vocals" for _, s in kit.pad_slices())


def test_standard_kit_uses_one_four_stem_pass(tmp_path, song, cfg):
    job = Job(tmp_path / "job")
    run_pipeline(job, cfg, source=str(song["mix"]), rights_confirmed=True)
    assert conftest.SEPARATOR_CALLS == [("htdemucs_ft.yaml", "work.wav")]
    assert Kit.load(job.kit_path).separation.vocal_model is None


def test_fast_split_uses_the_four_stem_model(tmp_path, song, cfg):
    job = Job(tmp_path / "job")
    run_pipeline(job, with_(cfg, split={"fast": True, "include_stems": True}), source=str(song["mix"]),
                 rights_confirmed=True, mode="split")
    assert conftest.SEPARATOR_CALLS == [("htdemucs_ft.yaml", "work.wav")]
    info = split_files(job)
    assert sorted(info["stems"]) == ["bass", "drums", "other", "vocals"]
    i, _ = read(job.abs(info["files"]["instrumental"]))
    parts = sum(read(job.abs(info["stems"][s]))[0] for s in ("drums", "bass", "other"))
    assert np.max(np.abs(i - parts)) < 1e-4
    # Asking for the vocal model afterwards replaces the fast acapella.
    run_pipeline(job, cfg, mode="split")
    assert conftest.SEPARATOR_CALLS[-1] == (ROFORMER, "work.wav")


def test_split_matches_the_original_sample_rate_and_length(tmp_path, song, cfg):
    import librosa

    y, sr = read(song["mix"])
    y48 = librosa.resample(y.T, orig_sr=sr, target_sr=48000).T
    src = tmp_path / "song48.wav"
    sf.write(src, y48, 48000, subtype="PCM_24")
    job = Job(tmp_path / "job")
    run_pipeline(job, cfg, source=str(src), rights_confirmed=True, mode="split")
    info = split_files(job)
    for role in ("acapella", "instrumental"):
        f = sf.info(str(job.abs(info["files"][role])))
        assert f.samplerate == 48000
        assert f.frames == len(y48)


@pytest.mark.parametrize("fmt", ["flac", "mp3"])
def test_split_formats(tmp_path, song, cfg, fmt):
    if fmt == "mp3" and shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg isn't installed")
    job = Job(tmp_path / "job")
    run_pipeline(job, with_(cfg, split={"format": fmt, "normalize": True}), source=str(song["mix"]),
                 rights_confirmed=True, mode="split")
    info = split_files(job)
    path = job.abs(info["files"]["acapella"])
    assert path.suffix == f".{fmt}"
    if fmt == "flac":
        a, _ = read(path)
        assert 20 * np.log10(np.max(np.abs(a))) == pytest.approx(-1.0, abs=0.05)


def test_split_from_the_cli(tmp_path, song, fixture_separator, capsys):
    conf = tmp_path / "c.toml"
    conf.write_text(f'[separation]\nbackend = "{fixture_separator}"\n')
    out = tmp_path / "job"
    code = main([str(song["mix"]), "--out", str(out), "--config", str(conf), "--i-have-rights", "--mode", "split"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "Separating the vocals" in printed and "Acapella" in printed
    assert (out / "split.json").exists()

import numpy as np
import pytest
import soundfile as sf

from sangisa.job import Job
from sangisa.worker.stages import ingest


def test_converts_to_working_format(tmp_path, cfg):
    src = tmp_path / "mono.wav"
    sf.write(src, np.sin(np.linspace(0, 2000, 22050 * 2)) * 0.5, 22050, subtype="PCM_16")
    job = Job(tmp_path / "job")
    ingest.run(job, cfg, str(src), rights_confirmed=True)
    info = sf.info(str(job.work_wav))
    assert (info.samplerate, info.channels, info.subtype) == (44100, 2, "PCM_24")
    assert info.duration == pytest.approx(2.0, abs=0.01)
    src_meta = job.manifest()["source"]
    assert src_meta["rights_confirmed"] and src_meta["path"] == "input/mono.wav"
    assert (job.root / "input" / "mono.wav").exists()


def test_requires_rights_confirmation(tmp_path, cfg):
    with pytest.raises(ingest.IngestError, match="rights"):
        ingest.run(Job(tmp_path), cfg, "whatever.wav", rights_confirmed=False)


def test_rejects_long_input(tmp_path, cfg):
    src = tmp_path / "a.wav"
    sf.write(src, np.zeros((44100 * 3, 2)), 44100)
    cfg["ingest"]["max_duration_s"] = 2
    with pytest.raises(ingest.IngestError, match="limit"):
        ingest.run(Job(tmp_path / "job"), cfg, str(src), rights_confirmed=True)


def test_streaming_links_are_off_by_default(tmp_path, cfg):
    with pytest.raises(ingest.IngestError, match="allow_streaming_links"):
        ingest.run(Job(tmp_path), cfg, "https://www.youtube.com/watch?v=x", rights_confirmed=True)


def test_missing_file(tmp_path, cfg):
    with pytest.raises(ingest.IngestError, match="No such file"):
        ingest.run(Job(tmp_path), cfg, str(tmp_path / "nope.wav"), rights_confirmed=True)

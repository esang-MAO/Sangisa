"""The local web server behind `sangisa serve`: upload a song, poll, fetch the kit."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from sangisa.schema import Kit
from sangisa.server import create_app


@pytest.fixture
def app(tmp_path, cfg):
    return create_app(tmp_path / "jobs", config=cfg, access_key="secret-key")


@pytest.fixture
def client(app):
    return TestClient(app)


def wait_for(client: TestClient, job_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = client.get(f"/api/jobs/{job_id}").json()
        if rec["status"] in ("done", "failed"):
            return rec
        time.sleep(0.2)
    raise AssertionError("job didn't finish")


def upload(client: TestClient, song, **form):
    with open(song["mix"], "rb") as f:
        files = {"file": ("My Song.wav", f, "audio/wav")}
        return client.post("/api/jobs", files=files, data={"rights": "true", **form})


def test_health(client):
    body = client.get("/api/health").json()
    assert body["app"] == "sangisa" and body["busy"] == 0


def test_upload_to_kit(client, song):
    res = upload(client, song)
    assert res.status_code == 201, res.text
    rec = wait_for(client, res.json()["id"])
    assert rec["status"] == "done", rec["error"]
    assert set(rec["stages"]) == {"ingest", "separate", "analyze", "pick", "render"}
    assert rec["kit"]["name"] == "My Song Kit"

    kit = Kit.model_validate(client.get(f"/api/jobs/{rec['id']}/kit.json").json())
    assert len(kit.pads) == 16
    wav = client.get(f"/api/jobs/{rec['id']}/files/{kit.slices[0].file}")
    assert wav.status_code == 200 and wav.content[:4] == b"RIFF"
    assert [j["id"] for j in client.get("/api/jobs").json()] == [rec["id"]]

    assert client.delete(f"/api/jobs/{rec['id']}").status_code == 204
    assert client.get(f"/api/jobs/{rec['id']}").status_code == 404


def test_custom_pad_split(client, song):
    rec = wait_for(client, upload(client, song, pads="drums=8,bass=8").json()["id"])
    assert rec["status"] == "done", rec["error"]
    kit = Kit.model_validate(client.get(f"/api/jobs/{rec['id']}/kit.json").json())
    assert [s.stem for _, s in kit.pad_slices()] == ["drums"] * 8 + ["bass"] * 8


def test_rejections(client, song, tmp_path):
    with open(song["mix"], "rb") as f:
        no_rights = client.post("/api/jobs", files={"file": ("a.wav", f)}, data={"rights": "false"})
    assert no_rights.status_code == 400 and "rights" in no_rights.json()["detail"]

    text = tmp_path / "notes.txt"
    text.write_text("hello")
    with open(text, "rb") as f:
        assert client.post("/api/jobs", files={"file": ("notes.txt", f)}, data={"rights": "true"}).status_code == 415

    assert upload(client, song, pads="drums=7").status_code == 400


def test_only_kit_files_are_served(client, song):
    rec = wait_for(client, upload(client, song).json()["id"])
    base = f"/api/jobs/{rec['id']}/files"
    assert client.get(f"{base}/analysis.json").status_code == 200
    assert client.get(f"{base}/work.wav").status_code == 200  # the original, for A/B listening
    for path in ("manifest.json", "stems/drums.wav", "pair/vocals.wav", "../status.json", "input/My Song.wav"):
        assert client.get(f"{base}/{path}").status_code == 404, path
    assert client.get("/api/jobs/not-a-job").status_code == 404


def test_other_websites_are_refused(client):
    assert client.get("/api/health", headers={"origin": "https://evil.example"}).status_code == 403
    ok = client.get("/api/health", headers={"origin": "https://esang-mao.github.io"})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "https://esang-mao.github.io"


def test_private_network_preflight(client):
    res = client.options("/api/jobs", headers={
        "origin": "https://esang-mao.github.io",
        "access-control-request-method": "POST",
        "access-control-request-private-network": "true",
    })
    assert res.status_code == 200
    assert res.headers["access-control-allow-private-network"] == "true"


def test_other_devices_need_the_key(app):
    phone = TestClient(app, client=("192.168.1.23", 50000))
    assert phone.get("/api/health").status_code == 401
    assert phone.get("/api/health", headers={"x-sangisa-key": "wrong"}).status_code == 401
    assert phone.get("/api/health", headers={"x-sangisa-key": "secret-key"}).status_code == 200
    assert phone.get("/api/health?key=secret-key").status_code == 200


def test_acapella_instrumental_then_kit(client, song):
    import io
    import zipfile

    res = upload(client, song, mode="split", format="flac")
    assert res.status_code == 201, res.text
    rec = res.json()
    assert rec["mode"] == "split" and rec["order"] == ["ingest", "separate", "analyze", "split"]
    assert rec["labels"]["separate"] == "Separating the vocals"
    rec = wait_for(client, rec["id"])
    assert rec["status"] == "done", rec["error"]
    assert rec["split"]["files"]["acapella"].endswith("My Song - Acapella - 120bpm Fmin.flac")

    info = client.get(f"/api/jobs/{rec['id']}/split.json").json()
    aca = client.get(f"/api/jobs/{rec['id']}/files/{info['files']['acapella']}")
    assert aca.status_code == 200 and aca.content[:4] == b"fLaC"
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{rec['id']}/split.zip").content))
    assert sorted(z.namelist()) == ["My Song/My Song - Acapella - 120bpm Fmin.flac",
                                    "My Song/My Song - Instrumental - 120bpm Fmin.flac"]
    assert client.get(f"/api/jobs/{rec['id']}/kit.json").status_code == 404

    # Make a kit from it: same job, the acapella is reused.
    res = client.post(f"/api/jobs/{rec['id']}/kit", data={"pads": "drums=4,vocals=4,bass=4,other=4"})
    assert res.status_code == 202, res.text
    rec = wait_for(client, rec["id"])
    assert rec["status"] == "done", rec["error"]
    assert rec["mode"] == "kit" and rec["kit"]["name"] == "My Song Kit" and rec["split"]
    kit = Kit.model_validate(client.get(f"/api/jobs/{rec['id']}/kit.json").json())
    assert kit.separation.vocal_model and len(kit.pads) == 16


def test_bad_split_options(client, song):
    assert upload(client, song, mode="remix").status_code == 400
    assert upload(client, song, mode="split", format="ogg").status_code == 400

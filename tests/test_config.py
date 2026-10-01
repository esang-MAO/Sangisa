import pytest

from sangisa.config import default_config, load_config, merge


def test_defaults_match_spec():
    cfg = default_config()
    assert cfg["kit"]["pad_split"] == {"drums": 6, "vocals": 4, "bass": 3, "other": 3}
    assert cfg["ingest"]["sample_rate"] == 44100 and cfg["ingest"]["bit_depth"] == 24
    assert cfg["ingest"]["allow_streaming_links"] is False
    assert cfg["separation"]["model"] == "htdemucs_ft.yaml"


def test_user_file_overrides_only_what_it_sets(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("[scoring.weights]\nisolation = 0.9\n")
    cfg = load_config(f)
    assert cfg["scoring"]["weights"]["isolation"] == 0.9
    assert cfg["scoring"]["weights"]["clarity"] == default_config()["scoring"]["weights"]["clarity"]


def test_pad_split_must_fill_the_kit():
    with pytest.raises(ValueError, match="adds up to"):
        load_config(overrides={"kit": {"pad_split": {"drums": 10}}})


def test_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


def test_pad_split_override_replaces_the_default():
    cfg = load_config(overrides={"kit": {"pad_split": {"drums": 8, "bass": 8}}})
    assert cfg["kit"]["pad_split"] == {"drums": 8, "bass": 8}

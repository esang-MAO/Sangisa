import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from sangisa.worker.stages.separate import audio_separator_backend


def test_audio_separator_backend_maps_stem_names(tmp_path, monkeypatch):
    """Exercises our wrapper against a stand-in for python-audio-separator."""
    seen = {}

    class FakeSeparator:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def load_model(self, model_filename):
            seen["model"] = model_filename

        def separate(self, path, custom_output_names=None):
            out = Path(seen["init"]["output_dir"])
            names = []
            for stem in ("Drums", "Bass", "Vocals", "Other"):
                # Mimic older releases that ignore custom names and return bare file names.
                name = f"work_({stem})_htdemucs_ft.wav"
                sf.write(out / name, np.zeros((100, 2)), 44100)
                names.append(name)
            return names

    module = types.ModuleType("audio_separator.separator")
    module.Separator = FakeSeparator
    monkeypatch.setitem(sys.modules, "audio_separator", types.ModuleType("audio_separator"))
    monkeypatch.setitem(sys.modules, "audio_separator.separator", module)

    stems = audio_separator_backend(tmp_path / "work.wav", "htdemucs_ft.yaml", tmp_path, model_dir=str(tmp_path / "m"))
    assert sorted(stems) == ["bass", "drums", "other", "vocals"]
    assert all(p.exists() for p in stems.values())
    assert seen["model"] == "htdemucs_ft.yaml"
    assert seen["init"]["normalization_threshold"] == 1.0

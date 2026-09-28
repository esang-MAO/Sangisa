"""Build the demo kit shown on the GitHub Pages viewer.

Runs the real pipeline on the synthetic test song (tests/conftest.py), with a
stand-in separation backend that returns the song's own stems, so it needs no
PyTorch, no model download and no copyrighted audio.

    uv run python scripts/build_demo.py site/demo
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.conftest import SR, make_stems  # noqa: E402

from sangisa.config import load_config  # noqa: E402
from sangisa.job import Job  # noqa: E402
from sangisa.pipeline import LABELS, run_pipeline  # noqa: E402
from sangisa.worker.stages.separate import register_separator  # noqa: E402


def main(out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stems = make_stems()
        peak = float(np.max(np.abs(sum(stems.values()))))
        paths = {}
        for name, y in {**stems, "Sangisa Demo": sum(stems.values())}.items():
            paths[name] = tmp / f"{name}.wav"
            sf.write(paths[name], np.stack([y, y]).T / peak * 0.9, SR, subtype="PCM_24")

        @register_separator("demo")
        def _separate(path, model, out_dir, **_):
            return {n: p for n, p in paths.items() if n in stems}

        cfg = load_config(overrides={"separation": {"backend": "demo", "model": "demo (pre-split stems)"},
                                     "kit": {"backups_per_stem": 6}})
        job = Job(tmp / "job")
        run_pipeline(job, cfg, source=str(paths["Sangisa Demo"]), rights_confirmed=True,
                     progress=lambda st, status, _: status == "start" and print(f"→ {LABELS[st]}…"))

        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        shutil.copy(job.kit_path, out / "kit.json")
        shutil.copytree(job.slices_dir, out / "slices")
    print(f"Demo kit written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "site/demo"))

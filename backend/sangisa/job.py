"""The job folder: every stage reads its inputs from here and writes its outputs here.

Layout::

    <job>/
      manifest.json      source, rights confirmation, per-stage status
      input/             untouched copy of the original file
      work.wav           44.1 kHz / 24-bit / stereo working copy
      stems/<stem>.wav   separated stems, same length as work.wav
      analysis.json      tempo, beat grid, key, sections
      candidates.json    every scored candidate (feeds re-roll and swaps)
      kit.json           the kit contract (see schema.py)
      slices/*.wav       rendered slices
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

STAGES = ("ingest", "separate", "analyze", "pick", "render")


class Job:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    # Paths -----------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def work_wav(self) -> Path:
        return self.root / "work.wav"

    @property
    def stems_dir(self) -> Path:
        return self.root / "stems"

    @property
    def analysis_path(self) -> Path:
        return self.root / "analysis.json"

    @property
    def candidates_path(self) -> Path:
        return self.root / "candidates.json"

    @property
    def kit_path(self) -> Path:
        return self.root / "kit.json"

    @property
    def slices_dir(self) -> Path:
        return self.root / "slices"

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def abs(self, rel_path: str) -> Path:
        return self.root / rel_path

    # Manifest --------------------------------------------------------------
    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self.write_manifest({"stages": {}})

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"stages": {}}
        return json.loads(self.manifest_path.read_text())

    def write_manifest(self, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.manifest_path, data)

    def update_manifest(self, **fields: Any) -> None:
        data = self.manifest()
        data.update(fields)
        self.write_manifest(data)

    def is_done(self, stage: str) -> bool:
        return self.manifest().get("stages", {}).get(stage, {}).get("status") == "done"

    def mark(self, stage: str, status: str, **extra: Any) -> None:
        data = self.manifest()
        entry = data.setdefault("stages", {}).setdefault(stage, {})
        entry.update(status=status, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        self.write_manifest(data)

    def invalidate_from(self, stage: str) -> None:
        """Forget the status of ``stage`` and every stage after it."""
        data = self.manifest()
        stages = data.setdefault("stages", {})
        for name in STAGES[STAGES.index(stage):]:
            stages.pop(name, None)
        self.write_manifest(data)


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())

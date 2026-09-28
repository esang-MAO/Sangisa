"""The kit.json contract.

Every exporter (Ableton Live .adg, Move .ablpresetbundle, numbered WAVs) and,
later, the API and review UI read this file. ``slices`` holds every kept
candidate (the ones on pads plus the backups offered as swaps); ``pads`` only
points at slices by id, so swapping or re-rolling a pad rewrites ``pads`` and
nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

SliceKind = Literal["one_shot", "loop", "phrase", "chop"]
Playback = Literal["one_shot", "loop"]


class Source(BaseModel):
    path: str
    sha256: str
    duration_s: float
    sample_rate: int
    rights_confirmed: bool


class Section(BaseModel):
    label: str
    start_s: float
    end_s: float


class Analysis(BaseModel):
    bpm: float
    key: str | None = Field(description='Full name, e.g. "F minor"')
    key_short: str | None = Field(description='Filename form, e.g. "Fmin"')
    key_confidence: float = 0.0
    time_signature: str = "4/4"
    beats_s: list[float] = []
    downbeats_s: list[float] = []
    sections: list[Section] = []


class Separation(BaseModel):
    backend: str
    model: str
    device: str | None = None
    stems: dict[str, str] = Field(description="stem name -> path relative to the job folder")


class Layout(BaseModel):
    pad_count: int
    pad_split: dict[str, int]
    origin: Literal["bottom_left"] = "bottom_left"


class Pad(BaseModel):
    pad: int = Field(ge=1, description="1-based; pad 1 is bottom-left")
    bank: str = Field(description='"A" for pads 1-16, "B" for 17-32, ...')
    midi_note: int = Field(description="36 (C1) for pad 1 of each bank")
    slice_id: str | None


class ScoreParts(BaseModel):
    isolation: float
    clarity: float
    loudness: float
    loopability: float | None = None
    uniqueness: float


class RenderSettings(BaseModel):
    fade_in_ms: float
    fade_out_ms: float
    normalize_dbfs: float | None
    reverse: bool = False
    playback: Playback


class Slice(BaseModel):
    id: str
    stem: str
    kind: SliceKind
    category: str
    label: str
    file: str = Field(description="Rendered WAV, relative to the job folder")
    source_start_s: float
    source_end_s: float
    bars: float | None = None
    beats: float | None = None
    note: str | None = Field(default=None, description='Pitch ("F2") or chord ("Fmin")')
    section: str | None = None
    score: float
    score_parts: ScoreParts
    cluster: int
    render: RenderSettings

    @property
    def duration_s(self) -> float:
        return self.source_end_s - self.source_start_s


class Kit(BaseModel):
    schema_version: int = SCHEMA_VERSION
    kit_name: str
    source: Source
    analysis: Analysis
    separation: Separation
    layout: Layout
    pads: list[Pad]
    slices: list[Slice]

    def slice(self, slice_id: str) -> Slice:
        for s in self.slices:
            if s.id == slice_id:
                return s
        raise KeyError(slice_id)

    def pad_slices(self) -> list[tuple[Pad, Slice | None]]:
        return [(p, self.slice(p.slice_id) if p.slice_id else None) for p in self.pads]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Kit":
        return cls.model_validate(json.loads(Path(path).read_text()))

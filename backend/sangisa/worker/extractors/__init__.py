"""Per-stem candidate extractors. Any stem without its own extractor (guitar, piano) is treated like "other"."""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np

from . import bass, drums, other, vocals
from .base import Candidate, Context

Extractor = Callable[[np.ndarray, Context], list[Candidate]]

EXTRACTORS: dict[str, Extractor] = {
    "drums": drums.extract,
    "bass": bass.extract,
    "vocals": vocals.extract,
    "other": other.extract,
}


def extractor_for(stem: str) -> Extractor:
    return EXTRACTORS.get(stem, partial(other.extract, stem=stem))


__all__ = ["Candidate", "Context", "EXTRACTORS", "extractor_for"]

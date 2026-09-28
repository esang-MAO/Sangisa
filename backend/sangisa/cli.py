"""``sangisa song.wav --out kit/``: song in, sample pack out."""

from __future__ import annotations

import argparse
import sys

from sangisa import __version__
from sangisa.config import load_config
from sangisa.job import STAGES, Job
from sangisa.pipeline import LABELS, run_pipeline
from sangisa.schema import Kit


def parse_split(text: str) -> dict[str, int]:
    split = {}
    for part in text.split(","):
        name, _, count = part.partition("=")
        if not count.strip().isdigit():
            raise argparse.ArgumentTypeError(f"expected stem=count, got {part!r}")
        split[name.strip()] = int(count)
    return split


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sangisa",
        description="Take a song apart and turn it into a sample pack.",
    )
    p.add_argument("input", nargs="?", help="audio file (WAV, AIFF, FLAC, MP3, M4A) or direct audio URL")
    p.add_argument("--out", "-o", required=True, help="job folder to write into (reused on re-runs)")
    p.add_argument("--config", "-c", help="TOML file overriding the defaults")
    p.add_argument("--i-have-rights", action="store_true",
                   help="confirm you own this audio or have the rights to sample it")
    p.add_argument("--model", help="separation model, e.g. htdemucs_ft.yaml or htdemucs_6s.yaml")
    p.add_argument("--pads", type=parse_split, metavar="STEM=N,...",
                   help="pad split, e.g. drums=6,vocals=4,bass=3,other=3")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--stage", choices=STAGES, help="re-run just this stage")
    g.add_argument("--from-stage", choices=STAGES, help="re-run from this stage onward")
    g.add_argument("--force", action="store_true", help="re-run every stage")
    p.add_argument("--version", action="version", version=f"sangisa {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict = {}
    if args.model:
        overrides["separation"] = {"model": args.model}
    if args.pads:
        overrides["kit"] = {"pad_split": args.pads, "pad_count": sum(args.pads.values())}
    try:
        cfg = load_config(args.config, overrides)
    except (OSError, ValueError) as exc:
        print(f"sangisa: {exc}", file=sys.stderr)
        return 2

    job = Job(args.out)

    def progress(stage: str, status: str, seconds: float | None) -> None:
        if status == "start":
            note = " (the slow step: about 1-2 min on a GPU, 10-15 min on CPU)" if stage == "separate" else ""
            print(f"→ {LABELS[stage]}…{note}", flush=True)
        else:
            print(f"  done in {seconds:.1f}s", flush=True)

    try:
        run_pipeline(
            job, cfg,
            source=args.input,
            rights_confirmed=args.i_have_rights,
            only=args.stage,
            start_at=args.from_stage,
            force=args.force,
            progress=progress,
        )
    except Exception as exc:  # noqa: BLE001 - report any stage failure cleanly
        print(f"sangisa: {exc}", file=sys.stderr)
        return 1

    if job.kit_path.exists() and job.is_done("render"):
        print_summary(Kit.load(job.kit_path), job)
    return 0


def print_summary(kit: Kit, job: Job) -> None:
    a = kit.analysis
    print(f"\n{kit.kit_name}: {a.bpm:.0f} BPM, {a.key or 'key unknown'}")
    for pad, s in kit.pad_slices():
        label = f"{s.label:<28} {s.duration_s:5.2f}s  score {s.score:.2f}" if s else "(empty)"
        print(f"  pad {pad.pad:>2}  {label}")
    print(f"\nkit.json and {len(kit.slices)} slices written to {job.root}")


if __name__ == "__main__":
    sys.exit(main())

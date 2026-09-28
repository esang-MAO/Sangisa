# Sangisa

*Sangisa* is Lingala for "to combine". The app takes a song apart and hands the pieces back so you can build something new with them.

Drop in a song and Sangisa separates the stems, finds the most usable moments, and returns a labeled, trimmed sample pack tagged with tempo and key. It exports to:

- an Ableton Live Drum Rack (`.adg`)
- an Ableton Move kit (`.ablpresetbundle`)
- numbered WAVs for any DAW or Koala Sampler

A second mode returns a full-length acapella and instrumental instead.

The app is local-first: songs and stems stay on your machine.

See [docs/SPEC.md](docs/SPEC.md) for the full build spec.

## Status

Milestone 1 (the pipeline CLI) is in place: song in, 16 trimmed and labeled slices plus `kit.json` out.
The exporters, API and web UI come next (see the build order in the spec).

## Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and ffmpeg (recommended; without it only WAV, AIFF and FLAC are read).

```sh
uv sync --extra separation          # CPU / Apple Silicon (pulls in PyTorch)
uv sync --extra separation-gpu      # NVIDIA GPU
uv run sangisa song.mp3 --out kit/ --i-have-rights
```

`--i-have-rights` confirms you own the audio or have permission to sample it; the job won't start without it.
The first run downloads the HT-Demucs model (about 300 MB) to `~/.cache/sangisa/models`.
Stem separation is the slow step: under 90 seconds on a GPU, 10-15 minutes on a CPU.

```text
→ Loading…
→ Separating stems…
→ Analyzing tempo and key…
→ Finding the best moments…
→ Building your kit…

song Kit: 120 BPM, F minor
  pad  1  Kick 1                        0.26s  score 0.69
  pad  2  Snare 1                       0.10s  score 0.37
  ...
  pad 16  Other loop B (2 bar)          4.00s  score 0.52
```

### Useful options

| Option | What it does |
| --- | --- |
| `--pads drums=8,bass=4,vocals=4` | Change the pad split (must add up to 16, 32 or 64) |
| `--model htdemucs_6s.yaml` | Use a different separation model |
| `--config my.toml` | Override any default in [`default_config.toml`](backend/sangisa/default_config.toml): scoring weights, pad split, limits, fades |
| `--from-stage pick` | Re-run from a stage onward, e.g. after changing the pad split or scoring weights |
| `--stage render` | Re-run a single stage |
| `--force` | Start over |

Stages that already finished are skipped, so re-running the same command is cheap.

## The job folder

```text
kit/
  manifest.json      source file, hash, rights confirmation, per-stage status and timings
  input/             untouched copy of the original
  work.wav           44.1 kHz / 24-bit / stereo working copy
  stems/             drums.wav bass.wav vocals.wav other.wav (sample-aligned with work.wav)
  analysis.json      BPM, beat grid, downbeats, key, sections
  candidates.json    every scored candidate (feeds swaps and re-rolls)
  kit.json           the kit: pads -> slices, with per-slice metadata
  slices/            Drums_Kick_1_120bpm_Fmin.wav, Bass_Loop_A_2bar_120bpm_Fmin.wav, ...
```

`kit.json` is the contract every exporter reads. It is defined in [`backend/sangisa/schema.py`](backend/sangisa/schema.py).
`slices` lists every kept candidate (the ones on pads plus up to 20 backups per stem). `pads` points at slices by id, so a swap only rewrites `pads`.

## Layout

```text
backend/sangisa/
  cli.py  pipeline.py  job.py  config.py  schema.py  audio.py
  worker/stages/       ingest, separate, analyze, pick, render (one module each)
  worker/extractors/   per-stem candidate finders: drums, bass, vocals, other
  worker/scoring.py    isolation, clarity, loudness, loopability, uniqueness
  exporters/           Milestones 2-4
tests/                 pytest, on a generated 16-bar song (no copyrighted audio)
```

## Development

```sh
uv sync
uv run pytest
uv run ruff check backend tests
```

The tests build a synthetic song with known tempo, key and stems, and swap in a stand-in separation backend, so they run in about a minute without PyTorch or model downloads.

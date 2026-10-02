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
The web app makes kits right in the browser (Quick split, no install) or hands the song to Sangisa on your
computer (`sangisa serve`) for the full AI stem split.
The exporters come next (see the build order in the spec).

## Make kits on your phone (no computer)

Open <https://esang-mao.github.io/Sangisa/>, choose a song, pick **Make it on: This device**, tick the rights box and
tap **Make the kit on this device**. Everything runs inside the browser: the song is never uploaded. A 3-minute song
takes well under a minute on a laptop and a minute or two on a recent phone. Keep the page open while it works.

On-device kits currently use a **Quick split** (harmonic/percussive separation, no AI model) into drums, bass and
everything else. There's no vocal stem yet, and kicks can sound thin because their low end lands in the bass stem.
Running the full AI stem split on the phone is the next step. Songs can be up to 7 minutes on a device, and the kit
isn't saved there yet, so use **Export for Koala** to keep it.

The in-browser engine (`site/engine/`) is a JavaScript port of the Python pipeline. Its tests check it against
librosa and the Python pipeline on the same song: filterbanks, MFCC, onsets, tempo, beats, key and the finished kit.

## Export to Koala Sampler (or any sampler / DAW)

In the kit view, **Export for Koala** builds a zip of numbered WAVs:

```text
My Song Kit - 92bpm Fmin/
  Bank_A/01_Bass_loop_B_2_bar.wav … 13_Kick_1.wav … 16_Perc_1.wav
  Extras/…            (optional: the backup sounds)
  README.txt          (BPM, key, the pad map, import steps)
```

- **Pad order:** *Top row first* (Koala fills pads left to right from the top) keeps the layout you see in Sangisa,
  so the kick stays bottom-left. *Bottom row first* suits Move, MPC and Ableton, where pad 1 is bottom-left.
- **Audio:** 48 kHz (what the iPhone and Koala run at) or 44.1 kHz, as 24-bit, 32-bit float or 16-bit WAV.
  The sample-rate conversion is band-limited, and loops stay whole bars.
- On iPhone, **Share…** opens the share sheet. Choose **Save to Files** and tap the zip there to unzip it. In Koala,
  browse to `Bank_A`, select all the files and drag them onto the first empty pad.

Your choices are remembered for next time.

## Make kits on your computer (full stem split)

Songs are processed by Sangisa running on your own computer, so they never leave your devices.

1. Once: install [uv](https://docs.astral.sh/uv/getting-started/installation/) (and ffmpeg for MP3/M4A), then:
   ```sh
   git clone https://github.com/esang-MAO/Sangisa
   cd Sangisa
   uv sync --extra separation        # or --extra separation-gpu on an NVIDIA machine
   ```
2. Each time, in the `Sangisa` folder:
   ```sh
   uv run sangisa serve
   ```
   Open <http://localhost:8765>, drop in a song, tick the rights box, and choose **Make the kit**. The app shows each
   stage as it runs, then opens the pad grid. **Your kits** keeps every kit you've made, and **Export for Koala** saves
   one. The <https://esang-mao.github.io/Sangisa/> page also connects to it in Chrome, Edge and Firefox.
3. From a phone or tablet on the same Wi-Fi, start it with `uv run sangisa serve --lan` and scan the QR code it
   prints. Other devices need the access key in that link; other websites can't use the server.

Stem separation takes about 1–3 minutes per song on an Apple Silicon Mac or NVIDIA GPU, 10–15 minutes on a CPU.
Kits are kept in `~/.sangisa/jobs` (change it with `--jobs-dir`). Making kits on a phone without a computer isn't
supported yet.

## Kit viewer on GitHub Pages

<https://esang-mao.github.io/Sangisa/> also shows a demo kit built by the real pipeline from a generated song, and opens
any kit `.zip`. Play the pads (mouse, touch or keys `Z X C V` … `1 2 3 4`), inspect scores, and swap backups onto pads.

No computer handy? **Actions → Build a kit → Run workflow** runs the pipeline on GitHub's CPUs (about 15 minutes) from
a direct link to an audio file. Download the **sangisa-kit** artifact and drop the zip onto the viewer. The repo is
public, so the song link shows in the run log and any signed-in GitHub user can download the artifact until it expires
after a day. Only use audio you own or have permission to sample.

On every push to `main`, the `Pages` workflow builds the demo kit and publishes `site/` to the `gh-pages` branch.
One-time setup: **Settings → Pages → Build and deployment → Source: Deploy from a branch**, then pick `gh-pages` and
`/ (root)`.

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
  server.py            `sangisa serve`: upload API, background worker, serves site/
  worker/stages/       ingest, separate, analyze, pick, render (one module each)
  worker/extractors/   per-stem candidate finders: drums, bass, vocals, other
  worker/scoring.py    isolation, clarity, loudness, loopability, uniqueness
  exporters/           Milestones 2-4
site/                  the web app (published to GitHub Pages)
site/engine/           the pipeline in JavaScript, run in a Web Worker for on-device kits; export.js
                       writes the numbered-WAV (Koala) export
scripts/build_demo.py  builds the viewer's demo kit
tests/                 pytest, on a generated 16-bar song (no copyrighted audio)
```

## Development

```sh
uv sync
uv run pytest
uv run ruff check backend tests
```

The tests build a synthetic song with known tempo, key and stems, and swap in a stand-in separation backend, so they run in about a minute without PyTorch or model downloads.

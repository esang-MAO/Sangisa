# Sangisa — Build Spec

Sep 27, 2026 · @Ken Ndiang

## Overview

The app turns one song into a ready-to-play sample pack in a few minutes. The user drops in an audio file or pastes a link; the app separates stems, finds the most usable moments, and exports them in the format their gear expects.

**Name:** Sangisa, from the Lingala word meaning "to combine". The app takes songs apart so they can be put back together as something new.

**Who it's for:** beat makers and finger drummers who sample, especially people on Ableton Live, Ableton Move, or the Koala Sampler phone app.

**The core promise:** input a song, get back 16 (or 32/64) labeled, trimmed, tempo- and key-tagged one-shots and loops, grouped by stem, loaded onto pads, with no manual chopping required. The user can still audition, swap, and re-trim any pad before exporting.

**Three export targets for v1:**

- Ableton Live Drum Rack preset (.adg), with each slice on its own pad
- Ableton Move drum kit preset, ready to transfer to the device
- A zip of numbered WAV files (01–16 etc.) that any DAW or Koala Sampler can import

**Second mode, Acapella + Instrumental:** the user can instead get two full-length files, the isolated vocal (acapella) and everything else (instrumental or backing track), for remixes, DJ edits, karaoke, or practice. Same input, no slicing, no kit.

## User flow

The whole flow is four screens, and the default path needs only two clicks: drop the song, then pick an export.

1. **Drop** — a single drop zone that accepts an audio file (WAV, AIFF, FLAC, MP3, M4A) or a pasted URL. Show file name, duration, and a waveform once loaded. A mode switch sits above the drop zone: Sample pack (default) or Acapella + instrumental. Include a checkbox: "I own this audio or have the rights to sample it" (see Legal guardrails).
2. **Processing** — a progress screen with named stages: Loading → Separating stems → Analyzing tempo and key → Finding the best moments → Building your kit. Show an honest time estimate; stem separation is the slow step.
3. **Review the kit** — the main screen.
   - A 4×4 pad grid (toggle to 8×8 for 64 pads) laid out bottom-left = pad 1, matching Move and Koala.
   - Each pad shows stem color, a label ("Kick 1", "Vox chop 3", "Bass loop A"), and a mini waveform. Tap to audition.
   - A side panel shows the full song with every candidate slice marked, grouped by stem. Drag a candidate onto a pad to swap it.
   - Per-pad controls: trim start and end, fade in and out, normalize, reverse, one-shot vs. loop.
   - Global controls: kit name, pads per stem (e.g. 6 drums, 4 vocals, 3 bass, 3 other), "Re-roll" to pick a different set of candidates.
4. **Export** — three big buttons: Ableton Live Drum Rack, Ableton Move, Numbered WAVs (for any DAW or Koala). Optional toggles: include full stems, include a README with BPM and key, sample rate and bit depth.

**In Acapella + instrumental mode,** steps 2–4 collapse into one screen: the same progress view, then two players (Acapella, Instrumental) with solo, an A/B toggle against the original, and a download button for each file or both as a zip. A "Make a kit from this" button hands the already-separated audio to sample-pack mode without starting over.

The app remembers the last export target and pad split so repeat users can go drop → export.

## Processing pipeline

The pipeline runs in five stages; each writes its output to a job folder so any stage can be re-run without redoing the ones before it.

### 1. Ingestion

- **Files:** decode anything with ffmpeg, then convert to a working copy at 44.1 kHz, 24-bit stereo WAV. Move plays back at 44.1 kHz, so standardizing early avoids surprises ([Ableton](https://help.ableton.com/hc/en-us/articles/15616909965596-Presets-on-Move-and-Note)).
- **Links:** use yt-dlp to fetch best-quality audio only. Put this behind the rights checkbox and a config flag so it can be turned off entirely (see Legal guardrails).
- Reject inputs over a configurable length (default 10 minutes) and log the source, duration, and hash in a job manifest.

### 2. Stem separation

- Wrap separation behind one interface, `separate(path, model) -> {stem_name: wav_path}`, so models can be swapped.
- **Default:** HT-Demucs fine-tuned (`htdemucs_ft`) for four stems: drums, bass, vocals, other. It's the most widely used open model and handles dense mixes well ([StemSplit explainer](https://stemsplit.io/blog/stem-separation-explained)).
- **Optional "HQ vocals" mode:** a BS-RoFormer model, currently the top performer for vocal isolation but heavier to run ([StemSplit explainer](https://stemsplit.io/blog/stem-separation-explained)).
- **Optional 6-stem mode:** `htdemucs_6s` adds guitar and piano.
- Use [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), which runs Demucs and RoFormer models from one package on CUDA, Apple Silicon (MPS), or CPU.
- Expect CPU-only runs of HT-Demucs to take 10–15 minutes per song versus under 90 seconds on a GPU ([DEV Community](https://dev.to/stevecase430/the-best-resources-for-audio-stem-separation-in-python-2026-i5j)). Design the UI and job queue around that.

**Acapella + instrumental mode:**

- Use a dedicated two-stem vocal model rather than the four-stem default. BS-RoFormer and MelBand-RoFormer vocal models give the cleanest acapellas with the least metallic artifacting ([StemSplit explainer](https://stemsplit.io/blog/stem-separation-explained)), and python-audio-separator runs them out of the box ([python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)).
- Save both of the model's outputs: the vocal stem and its paired instrumental.
- Fast fallback: HT-Demucs vocals as the acapella, and drums + bass + other summed as the instrumental.
- Optional cleanup toggle: run a dereverb model on the acapella; a dereverb-tuned BS-RoFormer is available ([Hugging Face toolkit](https://huggingface.co/collections/StemSplitio/music-source-separation-toolkit-2026)).
- Cache the result: if the user then asks for a kit, reuse this vocal stem and only split the instrumental into drums, bass, and other.

### 3. Musical analysis (on the full mix)

- Tempo and beat grid (librosa `beat_track`, or madmom for better accuracy), downbeats, and time signature.
- Musical key (Essentia `KeyExtractor` or librosa chroma).
- Song sections via novelty or self-similarity (intro, verse, chorus) so loops can be labeled by section.

### 4. Finding the good bits (per stem)

Each stem gets its own candidate extractor, then every candidate is scored.

| Stem | Candidates | Labeling |
| --- | --- | --- |
| Drums | One-shots cut at onsets (onset to next onset or 1.5 s, whichever is shorter); also 1- and 2-bar loops | Classify as kick, snare/clap, hat, perc using spectral centroid and low-band energy, or a small classifier |
| Bass | Single notes at onsets; 1-, 2-, and 4-bar loops on the beat grid | Note name from pitch tracking (e.g. pYIN) |
| Vocals | Phrases split on silence gaps; short chops of 1–2 beats | "Vox phrase", "Vox chop", plus note name when pitched |
| Other | Chord stabs at onsets; 2- and 4-bar melodic loops | Chord or key label from chroma |

**Scoring each candidate (0–1, weighted and configurable):**

- **Isolation:** energy of this stem versus the other stems in the same window. Low bleed scores high.
- **Clarity:** sharp transient for one-shots; steady level for loops.
- **Loudness:** reject near-silent or clipped slices.
- **Loopability:** for loops, low discontinuity between the end and the start.
- **Uniqueness:** cluster candidates by an audio embedding (MFCC mean or CLAP) and keep the best of each cluster so the kit isn't eight copies of the same kick.

**Kit assembly:** fill the pad split (default 6 drums / 4 vocals / 3 bass / 3 other) with the highest-scoring unique candidates. Drums go on the bottom rows in a standard layout (kick, snare, hats, perc). Keep the next-best 20 or so per stem as swap candidates for the review screen.

### 5. Slice rendering

- Cut at zero crossings, add 2–5 ms fades, optional peak-normalize to −1 dBFS.
- Name files with stem, label, BPM, and key, e.g. `Bass_Loop_A_2bar_92bpm_Fmin.wav`.
- Store per-slice metadata (source time range, score, BPM, key) in `kit.json`, which every exporter reads.

## Export formats

All three exporters read the same `kit.json` plus rendered WAVs, so adding a fourth target later (MPC, SP-404) is one new module.

| Target | File produced | How the user loads it | Key constraints |
| --- | --- | --- | --- |
| Ableton Live | `.adg` Drum Rack preset + `Samples/` folder, zipped | Unzip into the Live User Library, drag the rack onto a MIDI track | Pad 1 = MIDI note 36 (C1), bottom-left |
| Ableton Move (and Note) | `.ablpresetbundle` | Move Manager at move.local → Presets tab → drag in | 16 pads, Drum Sampler per pad, 400 MB max |
| Any DAW / Koala Sampler | Folder of `01_…wav` to `16_…wav` (or to 64), zipped | Koala: add the folder as a Location, multi-select all, drag onto the first empty pad | No preset file; order is the only structure |

### Ableton Live Drum Rack (.adg)

- An `.adg` is a gzip-compressed XML file. Don't write it from scratch: load a known-good template rack, clone one pad's chain per slice, and rewrite the sample path, name, and MIDI note.
- Use Ableton's free **Move & Note Drum Rack Template.adg** as the base ([Ableton](https://help.ableton.com/hc/en-us/articles/15616909965596-Presets-on-Move-and-Note)). Its structure (Instrument Rack → Drum Rack → Drum Sampler on each pad) also works in Live 12.1+, and the user can export it to Move from Live if they prefer.
- Offer a "Live 11 compatible" toggle that uses a Simpler-based template instead, since Drum Sampler first appeared in Live 12.1 ([Move manual](https://cdn-resources.ableton.com/resources/pdfs/move-manual/1/2024-10-04/move1-manual-en.pdf)).
- Reference implementation: [Ableton Device Creator](https://github.com/ben-juodvalkis/Ableton-Device-Creator), a Python library that builds drum racks from a sample folder using a template.
- Use relative sample paths so the rack still finds its samples after the zip is moved.

### Ableton Move preset (.ablpresetbundle)

- The bundle is a zip holding a `Preset.ablpreset` JSON file (the drum rack definition) and a `Samples/` folder ([move-kit-builder](https://github.com/tevinprince/move-kit-builder)).
- The JSON layout was reverse-engineered by the [extending-move](https://github.com/charlesvestal/extending-move) project; it is not official Ableton documentation and may change with Move firmware. Keep the writer isolated and versioned, and add a test that validates output against a real bundle exported from Live.
- Required structure: Instrument Rack → Drum Rack with Drum Samplers on the pads, at most one return chain with one effect, and one insert effect ([Ableton](https://help.ableton.com/hc/en-us/articles/15616909965596-Presets-on-Move-and-Note)).
- Set each pad to one-shot playback so long slices play through instead of being cut short by the default envelope ([move-kit-builder](https://github.com/tevinprince/move-kit-builder)).
- Offer Move's three kit styles as an option: normal, choke (next pad cuts the last one, good for vocal chops), and gate (plays while held, good for loops) ([Move manual](https://www.ableton.com/en/move/manual/)).
- Bundles over 400 MB are rejected by Move Manager ([Ableton](https://help.ableton.com/hc/en-us/articles/15616909965596-Presets-on-Move-and-Note)); warn before export.
- Only 16 pads per kit: if the user built 32 or 64, export one bundle per bank.

### Numbered WAVs (any DAW, Koala Sampler)

- Koala has no preset files; kits load by selecting a folder's contents and dragging them onto the first empty pad, which fills the pads in order ([Kit Maker](https://www.kit-maker.com/import-koala-samples/)).
- So the zero-padded number is the pad assignment: `01_Kick_1.wav`, `02_Snare_1.wav` … `16_Vox_Chop_4.wav`. Koala holds 64 pads across four banks of 16 ([Sound On Sound](https://www.soundonsound.com/reviews/elf-audio-koala-sampler)), so support 16, 32, or 64.
- Put each bank in its own subfolder (`Bank_A/`, `Bank_B/`…) so a bank drops in with one drag.
- Include a `README.txt` with song BPM, key, and a pad map.
- Open question: confirm on a device whether Koala fills pads from top-left or bottom-left, and match the numbering so the kick lands where the Move and Live layouts put it.

### Acapella + instrumental

- A zip with two full-length files, `<Song> - Acapella.wav` and `<Song> - Instrumental.wav`, with BPM and key appended when detected (e.g. `<Song> - Acapella - 92bpm Fmin.wav`).
- Both files match the original exactly in length, start point, and sample rate, so they drop into a DAW in sync at bar 1.
- Format choice: 24-bit WAV (default), FLAC, or 320 kbps MP3.
- No normalization by default, so acapella plus instrumental still adds back up to the original; offer a normalize toggle for people who want louder files.
- Optional: include the four individual stems in the same zip.

## Architecture and tech stack

Build v1 as a local-first web app: a Python backend and a browser UI that run on the user's own computer at localhost. That avoids GPU hosting bills, keeps songs on the user's machine, and uses Apple Silicon or an NVIDIA GPU when present.

```text
Browser UI (React)  ──HTTP──▶  API (FastAPI)  ──enqueue──▶  Job queue (RQ / in-process)
        ▲                          │                               │
        └──── poll progress ◀──────┘                               ▼
                                                         Worker: ingest → separate → analyze
                                                                 → pick → render → export
                                                                   (writes to job folder)
```

The API only queues work and serves files; the worker does all the audio processing, so the same code later runs on a hosted GPU server unchanged. The UI polls the API for stage progress and loads finished slices from the job folder.

| Layer | Recommendation |
| --- | --- |
| Frontend | React + Vite + TypeScript; wavesurfer.js for waveforms; Web Audio API for pad audition |
| API | Python 3.11+, FastAPI, Pydantic models shared with the worker |
| Jobs | Redis + RQ; a simple in-process queue in "local mode" so Redis is optional |
| Audio I/O | ffmpeg, soundfile, yt-dlp |
| Separation | python-audio-separator (HT-Demucs, BS-RoFormer), PyTorch with CUDA or MPS |
| Analysis | librosa, Essentia (key), optional madmom (beats) |
| Exporters | gzip + lxml for .adg; zipfile + json for .ablpresetbundle and WAV zips |
| Packaging | One `docker compose up` for GPU servers; a `make dev` / single script for local Mac and Windows |
| Tests | pytest with golden files: a short royalty-free test song, snapshot the kit.json, and validate every export opens |

Suggested repo layout: `frontend/`, `backend/api/`, `backend/worker/stages/` (one module per stage), `backend/exporters/` (one module per target), `templates/` (base .adg and reference .ablpresetbundle), `tests/fixtures/`.

## Legal guardrails

The tool itself is legal to build, but most songs people will drop in are copyrighted, and many streaming sites forbid downloading in their terms of service. Build these guardrails in from day one (this is product guidance, not legal advice):

- **Rights checkbox** on every job: the user confirms they own the audio or have permission to sample it. Store the confirmation in the job manifest.
- **Local-first processing:** songs and stems stay on the user's machine. If a hosted version comes later, auto-delete uploads and stems after a short window (e.g. 24 hours) and never make them shareable.
- **Link input in two tiers:** direct audio file URLs (a WAV or MP3 link, the user's own cloud storage) are always on. Streaming-site downloads via yt-dlp sit behind a config flag that is off by default, with a notice that many platforms prohibit downloading.
- **No sharing features for source-derived kits in v1:** no public kit gallery or marketplace, since that would redistribute copyrighted audio.
- **Clear copy in the UI:** a one-line note that releasing music with uncleared samples can require permission from the rights holders.

## MVP scope and build order

The MVP is file upload → 4 stems → 16 auto-picked pads → all three exports, running locally. Everything else waits.

**In v1:** file upload and direct audio URLs, HT-Demucs 4-stem separation, BPM and key detection, per-stem slice picking with scoring and dedupe, 16-pad review grid with audition, swap, and trim, the three exporters, and Acapella + instrumental mode using a RoFormer vocal model.

**Later:** streaming-link ingestion (behind the flag), BS-RoFormer HQ mode for kit stems, dereverb cleanup, and 6-stem mode, 32/64-pad kits, MIDI clip export that replays the original groove from the slices, extra targets (MPC, SP-404), a hosted version.

**Build order (each milestone ends with something runnable):**

1. **Pipeline CLI:** `sangisa song.wav --out kit/` runs ingest → separate → analyze → pick → render and writes WAVs plus `kit.json`. No UI.
2. **Numbered WAV exporter:** the simplest target; verify by loading into Koala and any DAW. Add split mode here too (a --mode split flag) that writes the acapella and instrumental pair.
3. **Ableton Live .adg exporter:** template-based; verify the rack opens in Live with every pad loaded.
4. **Move .ablpresetbundle exporter:** verify by uploading through Move Manager (or opening in Ableton Note).
5. **API and job queue:** wrap the CLI in FastAPI with progress reporting.
6. **Web UI:** drop zone, progress, pad grid review, export buttons.
7. **Polish:** re-roll, per-pad trim, kit presets, error handling, packaging.

**Acceptance test:** a 3-minute royalty-free song produces a 16-pad kit in under 3 minutes on a GPU machine, and all three exports load without errors in Live 12, Move, and Koala. In split mode, the acapella and instrumental line up sample-for-sample with the original and sum back to it with little audible difference.

### Kickoff prompt for Claude Code

Paste this into Claude Code along with this document:

```text
I'm building Sangisa, a local-first app that turns a song into a sample pack. The full spec is attached. Read it end to end first.

Start with Milestone 1 only: a Python CLI (`sangisa song.wav --out kit/`) that
1. converts input to 44.1 kHz 24-bit stereo WAV with ffmpeg,
2. separates 4 stems with python-audio-separator using htdemucs_ft (auto-select CUDA, MPS, or CPU),
3. detects BPM, beat grid, and key,
4. extracts and scores candidate slices per stem as described in the spec, dedupes them, and picks 16 pads (6 drums, 4 vocals, 3 bass, 3 other),
5. renders trimmed, faded WAVs and writes kit.json with per-slice metadata.

Requirements:
- Python 3.11+, managed with uv, one module per pipeline stage under backend/worker/stages/.
- Each stage reads and writes a job folder so stages can be re-run individually.
- Put scoring weights and pad split in a config file.
- Add pytest tests using a short generated test signal so tests don't need copyrighted audio.
- Keep exporters out of this milestone, but design kit.json so the .adg, .ablpresetbundle, and numbered-WAV exporters can be written against it next.

Before writing code, propose the kit.json schema and the module layout and wait for my OK.
```

## Sources

- [Presets on Move and Note — Ableton Help](https://help.ableton.com/hc/en-us/articles/15616909965596-Presets-on-Move-and-Note)
- [Ableton Move manual](https://www.ableton.com/en/move/manual/) and [PDF edition](https://cdn-resources.ableton.com/resources/pdfs/move-manual/1/2024-10-04/move1-manual-en.pdf)
- [move-kit-builder (GitHub)](https://github.com/tevinprince/move-kit-builder)
- [extending-move (GitHub)](https://github.com/charlesvestal/extending-move)
- [Ableton Device Creator (GitHub)](https://github.com/ben-juodvalkis/Ableton-Device-Creator)
- [How to import samples into Koala Sampler — Kit Maker](https://www.kit-maker.com/import-koala-samples/)
- [Koala Sampler review — Sound On Sound](https://www.soundonsound.com/reviews/elf-audio-koala-sampler)
- [python-audio-separator (GitHub)](https://github.com/nomadkaraoke/python-audio-separator)
- [Stem separation explained — StemSplit](https://stemsplit.io/blog/stem-separation-explained)
- [Stem separation resources in Python — DEV Community](https://dev.to/stevecase430/the-best-resources-for-audio-stem-separation-in-python-2026-i5j)

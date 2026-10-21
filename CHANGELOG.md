# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Until 1.0, the stdout keys, exit codes and output file
names may still change between minor versions; every such change is listed here.

## [Unreleased]

### Apple Silicon

- Whisper now runs on the GPU of Apple Silicon Macs through mlx-whisper (MLX). `device: auto` picks
  it on an arm64 Mac; `metal` selects it explicitly, `cpu` still means faster-whisper on the CPU.
  Word timestamps, the VAD (faster-whisper's Silero VAD, with the speech chunks joined and the times
  mapped back), the automatic no-VAD retry, language detection with its probability, `SPEECH:` and
  frame pairing work as on CUDA. Differences: greedy decoding with temperature fallback instead of
  beam search 5 (transcript header `beam 1`), no batching, MLX weights (`mlx-community/whisper-*`,
  the same sizes) stored in `model_dir` next to the CTranslate2 ones.
- Measured on an Apple M5 Pro (24 GB), 18:40 lecture end to end: `small` RTF 0.026, `large-v3-turbo`
  0.031, `large-v3` 0.095, against 0.063 for `small` on the CPU (int8); peak memory 1.2, 2.1 and
  3.9 GB. Comparison with whisper.cpp and the CPU in [`docs/DESIGN.md`](docs/DESIGN.md#apple-silicon-engine).
- Failures of the Apple GPU engine raise the same GPU errors as CUDA and never switch to the CPU:
  a busy or full GPU falls back to YouTube's auto-captions or exits 3; an engine that cannot start
  (mlx-whisper missing, no Metal) exits 3 with `ERROR: Whisper could not run on the GPU (…)` and a
  `HINT:` about CPU mode, also on YouTube.
- `doctor` has new `METAL:` and `ENGINE:` lines, reports the chip and its unified memory under
  `GPU:` and `VRAM_TOTAL_MB:`, recommends `RECOMMEND_DEVICE: metal` with a model by memory (16 GB or
  more `large-v3`, 8 GB `large-v3-turbo`, otherwise `small`), and checks `MODEL_CACHED:` for the
  engine in use. `setup` downloads the MLX weights with their sizes; `--device metal` is a new
  value of the `device` setting and `--device`.
- The script's dependencies gain `mlx-whisper>=0.4.3,<0.5` on Apple Silicon only (about 0.7 GB in
  uv's cache); Windows and Linux installs do not change. mlx-whisper's PyTorch dependency is
  overridden away in the script header (only its conversion tool imports it).
- macOS CPU mode was run on an Apple M5 Pro (`doctor`, `setup`, `fetch`, frames) and works.

### Changed

- `doctor`'s `CUDA:` line on macOS says `macOS has no CUDA (Whisper runs on the Apple GPU instead)`
  on Apple Silicon and `(CPU mode only)` on an Intel Mac.
- The sound map still needs an NVIDIA GPU.

## [0.1.0] - 2026-10-21

The first public release. video-watch was developed and used privately before this release; this
version collects that work under a new name and makes it portable.

### Transcripts

- `fetch` for YouTube, any site yt-dlp supports, and local video files.
- Source order: manual subtitles in the original language (or a matching embedded subtitle stream
  in a local file), otherwise local Whisper; YouTube's original-language auto-captions only when
  Whisper cannot run on the GPU. Translated tracks are never used, and an unknown language means
  Whisper. Every run states its source (`SOURCE:`).
- faster-whisper with `large-v3` by default; `large-v3-turbo`, `large-v2`, `medium`, `small`,
  `base` and `tiny` selectable. Batched inference, `float16` on CUDA, VAD, word timestamps.
- The batch size adapts to the free VRAM (8, 4, 2 or 1), so large-v3 also runs on 6-8 GB cards,
  more slowly. The `ESTIMATE:` line accounts for the model, device and batch.
- On a busy or full GPU, YouTube falls back to its original-language auto-captions; other sources
  exit 3. Without any usable CUDA, `fetch` exits 3 with a `HINT:` about CPU mode.
- Automatic retry without VAD when a run finds almost no speech (music videos), and a `SPEECH:`
  hint (words per minute, longest gap) for the cases in between.
- Description and chapters in the transcript header; per-segment stamps, a paragraph version and
  precise `segments.json`.

### Sound map (optional)

- A local sound-event map with PE-A-Frame base: music, speech and singing coverage plus 42 event
  labels (laughter, applause, explosions, sirens, ...) with times, and loudness peaks and silences.
  Calibrated on synthetic mixtures; silence kept out of the model's input.
- `sounds` command: the map's summary, or open-text queries (`--query "car horn, glass breaking"`).
- Runs in a separate helper process after Whisper has released the GPU; NVIDIA GPU only. Off until
  enabled in `setup` or the config; never downloaded by `fetch` or `sounds` (`fetch --sounds`
  without it is a `NOTE:`, `sounds` exits 3).

### Frames

- `frames --at` (time-exact stills from a stream seek, no full download), `--overview` (labeled
  contact sheets from scene changes) and `--range` (dense frames over a short window).
- Every frame or tile is paired with what is said (`SAID:`) and heard (`HEARD:`) at that moment.
- CPU only, with a token estimate per call and a safety ceiling (`--max-frames`).

### Sources and lifecycle

- `playlist` lists playlists and channels without downloading, with markers for unavailable,
  private, live and upcoming entries and a processing estimate.
- Offline work keys, cache reuse, a 24-hour purge, exclusive per-video locks, and `clean` by URL,
  path or work key. Audio is deleted right after use; local originals are only read.
- Login walls give `NEEDS_COOKIES` (exit 2); Firefox cookies are used only after the user agrees.

### Setup and packaging

- `doctor`: a read-only environment report (always exit 0) with model, device and sound-map
  recommendations by GPU memory and a `READY:` verdict.
- `setup`: applies the chosen device, model and sound-map option and pre-downloads what they need;
  `--dry-run` shows sizes first, and downloads need `--yes`.
- `doctor`, `config` and `setup --dry-run` also run with the standard library alone
  (`uv run --no-project --python ">=3.11" python video_watch.py …`), before any package download.
- `config`: show, path, get, set and unset settings; a per-OS config file and `VIDEO_WATCH_*`
  environment variables. A configured work directory must be new, empty or marked as a video-watch
  work directory.
- Explicit CPU mode (`device: cpu`, int8); no silent CPU fallback.
- Linux: CUDA libraries from the pip wheels are found automatically (verified on WSL).
- `doctor` reports `SOUND_HELPER_READY:`; `setup --sounds on` records the prepared sound-map
  environment in a marker file, so `fetch` and `sounds` never start a large install unasked.
- An ffmpeg or ffprobe that an agent sandbox blocks is reported as such (`blocked:` in `doctor`, an
  `ERROR:` with a full-access `HINT:`), not as missing. Copyable `HINT:` commands use the script's
  absolute path.
- A portable Agent Skill installable with `npx skills add AAyberkU/video-watch`, a Claude Code
  plugin and marketplace, and an agent-guided install guide (`INSTALL.md`). Frame reading verified
  in Claude Code, Codex CLI and Kilo CLI.
- Unit tests on Windows, Linux and macOS in CI.

### Known limitations

- CPU-only Windows and Linux machines still download about 1.2 GB of CUDA wheels.
- macOS runs in CPU mode only and has not been tested yet; the sound map needs an NVIDIA GPU.
- Codex on Windows needs full access for video-watch commands (its default sandbox cannot run
  per-user ffmpeg and uv installs and has no network).
- Frame reading in OpenCode and in IDE extensions is not verified yet.

[0.1.0]: https://github.com/AAyberkU/video-watch/releases/tag/v0.1.0

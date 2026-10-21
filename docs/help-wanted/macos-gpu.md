# macOS GPU support (Apple Silicon)

**Labels:** `help wanted`, `platform: macos`, `enhancement`

## Context

video-watch transcribes with faster-whisper (CTranslate2), which runs on NVIDIA GPUs (CUDA) and on
the CPU, but not on Apple GPUs. On a Mac, video-watch can only run in explicit CPU mode
(`setup --device cpu --model small`), which is slow, so most Mac users would need a small model and
accept lower accuracy. CPU mode on macOS is expected to work (the CUDA wheels are skipped there) but
has not been run on a Mac yet; a `doctor` report and one CPU-mode `fetch` with its timing are
already a useful first contribution. Frames work on every platform (ffmpeg on the CPU); the sound map is a separate issue
([sound map on CPU / Apple Silicon](sound-map-cpu-apple-silicon.md)).

The maintainer has no Mac, so this needs someone who does.

## Suggested approach

Add a second Whisper engine that is selected on macOS with Apple Silicon. Candidates (unverified
which fits best):

- **MLX Whisper** (`mlx-whisper`, on Apple's MLX framework): Python, installable from PyPI, runs the
  Whisper large-v3 and turbo weights on the Apple GPU. Check whether it provides word timestamps and
  a VAD option.
- **whisper.cpp with Metal**: a native binary or Python bindings; mature Metal support, word-level
  timestamps, a built-in VAD in recent versions. Needs a plan for how `setup` obtains the binary.

Where the code lives and what an engine must provide (segments with word timestamps, VAD and no-VAD
passes, language handling, errors as `GpuUnavailable`, model storage under `model_dir`) is
described in [CONTRIBUTING.md](../../CONTRIBUTING.md#adding-a-whisper-engine-or-backend).

Steps:

1. Prototype both candidates outside video-watch on the same 3–4 public videos (speech with
   subtitles to compare against, music, a non-English video); note speed, memory and output quality.
2. Propose the choice in this issue with the numbers.
3. Implement it behind the existing `device` setting (for example `auto` picking the Apple GPU on
   Apple Silicon), with `doctor` reporting the engine and a sensible `RECOMMEND_MODEL`.

## Acceptance criteria

- [ ] On an Apple Silicon Mac, `doctor` reports the GPU engine and `READY: yes` after `setup`.
- [ ] `fetch` transcribes with the Apple GPU engine; `SOURCE: whisper`, `SPEECH:` and the no-VAD
      auto-retry behave as on CUDA; segments carry word timestamps.
- [ ] `frames` `SAID:` pairing works with the new engine's segments.
- [ ] No silent fallback: if the engine cannot run, the user gets an `ERROR:` and a `HINT:`.
- [ ] The CUDA path and its tests are unchanged; new logic has unit tests that run without a GPU
      (CI includes macOS).
- [ ] README platform table, `docs/REFERENCE.md` and `CHANGELOG.md` updated; measured speed and
      memory for at least `large-v3-turbo` and one smaller model on the tested Mac are in the PR.

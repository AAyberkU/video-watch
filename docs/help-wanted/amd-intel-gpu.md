# AMD and Intel GPU support

**Labels:** `help wanted`, `platform: amd-intel`, `enhancement`

## Context

video-watch's Whisper engine, faster-whisper (CTranslate2), uses CUDA on NVIDIA GPUs or runs on the
CPU. AMD Radeon and Intel Arc or integrated GPUs are not used, so on those machines video-watch
works only in explicit CPU mode (`setup --device cpu --model small`), which is slow (and still
downloads about 1.2 GB of NVIDIA CUDA wheels it does not use; see the roadmap). The maintainer has
only NVIDIA hardware.

## Suggested approach

Add a second Whisper engine for non-NVIDIA GPUs. Candidates (unverified; please test before
committing to one):

- **whisper.cpp with Vulkan**: one backend for AMD and Intel on Windows and Linux, word-level
  timestamps, a built-in VAD in recent versions. Needs a plan for how `setup` obtains the right
  binary or bindings.
- **ROCm** (AMD on Linux), for example a CTranslate2 or PyTorch build with ROCm, if one is
  maintained and installable without system-wide changes.
- **OpenVINO** (Intel) Whisper pipelines, if they provide word timestamps.

Where the code lives and what an engine must provide (segments with word timestamps, VAD and no-VAD
passes, language handling, errors as `GpuUnavailable`, model storage under `model_dir`) is
described in [CONTRIBUTING.md](../../CONTRIBUTING.md#adding-a-whisper-engine-or-backend).

Steps:

1. Run `doctor` on your machine and post it here, with your GPU, driver and OS.
2. Prototype the engine outside video-watch on 3–4 public videos; note speed, VRAM and quality.
3. Propose the engine and its selection rule (how `device: auto` finds it, what `doctor` reports).
4. Implement it without changing the CUDA path.

## Acceptance criteria

- [ ] On at least one AMD or Intel GPU, `doctor` reports the engine and the GPU memory, and
      `READY: yes` after `setup`.
- [ ] `fetch` transcribes on the GPU with word timestamps; `SPEECH:`, the no-VAD auto-retry and
      `SAID:` pairing work.
- [ ] Errors (driver missing, out of memory) give `ERROR:` and `HINT:`, never a silent CPU run.
- [ ] Engine installation is handled by `setup` (with sizes shown before downloading) or clearly
      documented; no global installs behind the user's back.
- [ ] Unit tests for the new logic run without a GPU; the CUDA path and its tests are unchanged.
- [ ] README platform table, `docs/REFERENCE.md` and `CHANGELOG.md` updated; measured timings in the
      PR.

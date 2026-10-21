# Roadmap

Where video-watch is and where it could go. Items under "help wanted" are the biggest gaps; ideas
are not decided and may never happen. Open an issue before starting large work so we can agree on
the approach.

## Shipped (0.1.0)

- [x] Transcripts from manual subtitles or local Whisper (`large-v3` by default, other models
      selectable), with a strict source policy and the source always stated.
- [x] YouTube, any yt-dlp site, local files; playlist and channel listing.
- [x] Automatic no-VAD retry for music videos and a `SPEECH:` hint for partial misses.
- [x] Description and chapters in the transcript header.
- [x] Frames: time-exact stills, scene-detected contact sheets, dense ranges; `SAID:` and `HEARD:`
      pairing.
- [x] Optional local sound map (PE-A-Frame) with open-text sound queries and loudness.
- [x] `doctor`, `setup` and `config`; per-OS config file and environment variables; explicit CPU
      mode; a Whisper batch size that adapts to the free VRAM (large-v3 on 6-8 GB cards).
- [x] Portable Agent Skill, `npx skills add`, Claude Code plugin, agent-guided install.
- [x] Unit tests in CI on Windows, Linux and macOS.

## Not supported yet: help wanted

These need people with the hardware. Each has an issue draft with context, a suggested approach and
acceptance criteria in [`docs/help-wanted/`](docs/help-wanted/).

- [ ] **macOS GPU (Apple Silicon / Metal).** faster-whisper's CTranslate2 backend runs on CUDA or
      the CPU only, so Macs run in slow CPU mode today. A second Whisper engine is needed, for
      example MLX Whisper or whisper.cpp with Metal (unverified which fits best).
      [Draft](docs/help-wanted/macos-gpu.md)
- [ ] **AMD and Intel GPUs.** Same gap on Windows and Linux. Candidates: whisper.cpp with Vulkan
      (both vendors), ROCm builds for AMD on Linux (unverified). [Draft](docs/help-wanted/amd-intel-gpu.md)
- [ ] **Sound map on the CPU or Apple Silicon.** The helper uses PyTorch with CUDA only. A CPU or
      MPS path needs measured speed and a check that results match the CUDA calibration.
      [Draft](docs/help-wanted/sound-map-cpu-apple-silicon.md)
- [ ] **Verify frame reading in OpenCode and IDE extensions.** Verified in Claude Code, Codex CLI
      and Kilo CLI; OpenCode with a vision model and the IDE extensions are not checked yet.
      [Draft](docs/help-wanted/agent-image-reading.md)
- [ ] **Verify CPU mode on macOS and CUDA on native Linux.** macOS CPU mode is expected to work but
      has not been run on a Mac; Linux CUDA was verified on WSL (Arch Linux) with `small` only.
      A [platform report](https://github.com/AAyberkU/video-watch/issues/new?template=platform_support.yml)
      with `doctor` output and timings is enough.

## Known limitations, planned

- [ ] **Split the CUDA dependencies.** On Windows and Linux the script's inline dependencies always
      include the NVIDIA CUDA wheels (about 1.2 GB), so CPU-only machines download them too. A
      split (for example a separate CPU entry point, or installing the wheels only when `setup`
      chooses a GPU) would make CPU mode a much smaller download.

## Ideas, not decided

None of these are committed. Several are unverified estimates from the
[research notes](docs/research/).

- **Speaker diarization** (who speaks when). The best open candidate is pyannote
  `speaker-diarization-community-1` (CC-BY-4.0 weights, language-independent, an "exclusive" mode for
  aligning with word timestamps). Its download is gated: each user needs a Hugging Face token and
  must accept the terms, so it would be opt-in. VRAM and speed are unmeasured (estimated 1–2 GB).
- **OCR for on-screen text** (slides, code, burned-in captions) as text output, which could cost
  fewer tokens than sending frames. Engine not chosen.
- **A unified timeline** file merging transcript lines, sound events and frame stamps, readable by
  time range.
- **An optional Gemini engine** for motion-heavy questions (sports, dance, gameplay feel), strictly
  opt-in and clearly labeled as uploading the video. It would break the "nothing leaves your
  machine" default, so it needs discussion first.
- **A Codex plugin** (`plugin.json` plus a marketplace file). The portable skill already works in
  Codex; a plugin would add versioned installs.
- **A Kilo marketplace listing** (a pull request to Kilo's marketplace repository).
- **Frame deduplication and budget presets** for overviews of long, static videos.
- **Shorter sound-map start-up** (about 15 s of environment, import and model load per map today).
- **Reproducible environments** with `uv lock --script`, and revisiting the `faster-whisper<2` and
  `av<16` pins.
- **Better paragraph splitting** for long stretches without punctuation.

## Not planned

- **An MCP server.** A CLI plus a skill already works in every agent that can run commands and read
  images, with no running process and no per-agent configuration, and image results from MCP tools
  are not reliably shown to the model in every agent. An MCP server would still need a skill to
  explain its output. This can be revisited if an agent appears that cannot run commands.
- **Speech emotion recognition.** Near chance on natural and non-English speech; the sound map
  reports laughter, crying or shouting as sounds instead.
- **Batch transcription of whole playlists.** List first, then fetch the videos a question needs.
- **DRM-protected content** and content that is live right now.

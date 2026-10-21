# Contributing to video-watch

Thank you for helping. Bug reports, platform reports, docs fixes and code are all welcome.

**Own a Mac, an AMD GPU or an Intel GPU?** You are exactly who this project needs. video-watch only
runs at full speed on NVIDIA GPUs today, and the maintainer cannot test other hardware. Even a
`doctor` report and a timing from your machine helps; a second Whisper engine would be a big
contribution. See the [help-wanted drafts](docs/help-wanted/) and the [roadmap](ROADMAP.md).

## Before you start

- For anything larger than a small fix, open an issue first so we can agree on the approach.
- Read [docs/DESIGN.md](docs/DESIGN.md) for why things are the way they are, and
  [docs/REFERENCE.md](docs/REFERENCE.md) for the exact interface.
- Be kind and constructive in issues and reviews.

## Development setup

You need [uv](https://docs.astral.sh/uv/), ffmpeg and ffprobe on `PATH`, and Git. Nothing is
installed into your system Python: both scripts declare their dependencies inline (PEP 723), and
`uv run` builds isolated environments.

```bash
git clone https://github.com/AAyberkU/video-watch
cd video-watch
uv run --no-project --python ">=3.11" python skills/video-watch/scripts/video_watch.py doctor
```

`doctor`, `config` and `setup --dry-run` use only the standard library and run in this light form
without installing the script's packages; every other command runs with plain `uv run`.

Run the CLI straight from the repository while developing:

```bash
uv run skills/video-watch/scripts/video_watch.py fetch "https://youtu.be/aircAruvnKk"
```

To try your working copy through an agent, copy (or symlink) `skills/video-watch` into the agent's
skills folder (see the [README](README.md#manually)).

## Running the tests

```bash
uv run --with pytest pytest -q
```

The unit tests need no GPU and no network: Whisper, the sound model, yt-dlp and ffmpeg are faked
where a flow needs them. CI runs them on Windows, Linux and macOS
(`.github/workflows/tests.yml`). Every pull request must keep them green; new logic needs new tests.

## Code layout

| Path | Content |
| --- | --- |
| `skills/video-watch/SKILL.md` | The Agent Skill: when to run, how to read the output, how to answer. |
| `skills/video-watch/scripts/video_watch.py` | The whole CLI in one file: inputs, yt-dlp, Whisper, frames, playlists, cache, `doctor`, `setup`, `config`. |
| `skills/video-watch/scripts/sound_events.py` | The sound-map helper: its own PEP 723 environment with PyTorch, run as a subprocess; talks to the CLI through JSON files. |
| `tests/test_video_watch.py` | Unit tests for the pure logic and the command flows (with fakes). |
| `INSTALL.md` | The guide an agent follows when a user says "install this". |
| `.claude-plugin/` | Claude Code plugin and marketplace manifests. |
| `docs/` | Reference, design notes, research, help-wanted drafts. |

The CLI is deliberately a single file: `npx skills add` copies the skill folder, and one file with
an inline dependency header runs anywhere `uv` does without a build step. Keep it that way unless an
issue agrees otherwise. Pure logic (parsing, span rules, formatting) stays separate from I/O so it can
be unit-tested.

## Interface stability

The skill, `INSTALL.md` and agents in the wild parse the CLI's output, so the following is a
contract:

- commands and flags;
- stdout keys (`SOURCE:`, `FRAME:`, `SAID:`, `HEARD:`, `SOUNDS:`, `ERROR:`, `HINT:`,
  `NEEDS_COOKIES`, `READY:`, ...), their order and the stamp formats (`[mm:ss]`, `[mm:ss.s]`,
  `[h:mm:ss]`);
- the `ERROR:` prefixes the skill keys on (`invalid frames request:`, `invalid sounds request:`,
  `ffmpeg`, `another run is using this video`, `no usable CUDA GPU for Whisper`,
  `Whisper could not run on the GPU`, `sound map unavailable:`);
- exit codes 0, 1, 2 and 3;
- output file names and the `transcript.md` and `sounds.md` section headings;
- config keys and `VIDEO_WATCH_*` environment variables.

If you change any of it, update `skills/video-watch/SKILL.md`, `README.md`, `docs/REFERENCE.md` and
`CHANGELOG.md` in the same pull request, and explain why in the description. Adding a new key or an
optional `NOTE:` is usually fine; renaming or removing one needs discussion first.

Some behavior is also policy and needs an issue before it changes: the transcript source order
(never a translated track), no silent CPU fallback, cookies only after the user agrees, frames only
when the visuals matter, local files only read, and labels as acoustic cues rather than emotions.

## Adding a Whisper engine or backend

faster-whisper (CTranslate2) runs on CUDA and the CPU only. Supporting Apple Silicon, AMD or Intel
GPUs means adding a second engine, for example MLX Whisper or whisper.cpp (Metal, Vulkan).

**Where the Whisper code lives:** in `video_watch.py`, around `load_whisper()` (loads a model for a
device), `transcribe()` (one transcription pass, with or without VAD), `run_whisper()` (the flow:
estimate, load, download or extract audio, transcribe, the no-VAD retry, cleanup) and
`whisper_language()` (maps a language tag to one the engine accepts). GPU failures are raised as
`GpuUnavailable`: with `no_cuda=True` (no usable device at all) `fetch` exits 3 with the CPU-mode
hint; otherwise (busy or out of memory) it drives the YouTube auto-captions fallback or the exit-3
retry path. The batch size comes from `choose_batch_size()` and the measured `WHISPER_VRAM_MB`
table; `whisper_estimate_min()` gives the `ESTIMATE:`. Device and model choices come from the config
(`device`, `model`, `model_dir`); `doctor` (`cuda_status()`, `recommend()`) reports them.

**What an engine must provide:**

- **Segments with word timestamps.** A list of `Segment(start, end, text, words)` in seconds, with
  `words` as `{"start", "end", "word", "p"}` entries. Word times feed the no-VAD retry's word count,
  `SPEECH:` and frame pairing; without them those features degrade.
- **VAD behavior.** A VAD pass that skips silence and music (the default, `min_silence_duration_ms`
  500) and a no-VAD pass for `--no-vad` and the automatic retry. If the engine's VAD differs, say
  how in the pull request and check a music video and a speech video.
- **Language.** Accept the language from the metadata or `--lang` (two- or three-letter codes,
  normalized), auto-detect otherwise, and report the detected language.
- **Errors.** Out of memory, missing drivers or an unusable device must raise `GpuUnavailable` with
  a short reason, never fall back to the CPU silently.
- **Models.** Map the config's model names (`large-v3`, `large-v3-turbo`, ...) to the engine's
  weights, store them under `model_dir`, and report `MODEL_CACHED:` in `doctor`. Downloads happen in
  `setup` (with sizes), not unexpectedly during a `fetch`.
- **Selection.** Choose the engine from the device and platform; `doctor` must report it and
  recommend sensibly. The CUDA path must keep working unchanged.
- **Dependencies.** Prefer engines installable through the PEP 723 header. If a native binary is
  needed, document how `setup` gets it and keep it out of the CUDA path.

A comparison on a few public videos (same model size, both engines: word error eyeballed or
measured, real-time factor, peak memory) belongs in the pull request.

## Testing on real hardware

Unit tests cannot cover GPUs, sites or codecs. For changes to the fetch, Whisper, sound-map or
frames paths, also run real videos and look at the actual output files:

- one video with manual subtitles and one without (Whisper), for example `aircAruvnKk` (3Blue1Brown,
  English subtitles) and a short clip without subtitles;
- a music video for the no-VAD retry, and a non-English video for language handling;
- for frames, open the JPEGs and check the moments; for the sound map, read `sounds.md`;
- `clean` afterwards, and check nothing is left in the work directory.

In the pull request, include your OS, GPU or CPU, driver version, the `doctor` output, and timings.
Never commit downloaded media, cookies, model files or personal paths.

## Pull request checklist

- [ ] `uv run --with pytest pytest -q` passes; new logic has tests.
- [ ] Real-video check for changes to fetch, Whisper, sound map or frames (what you ran and saw).
- [ ] Interface changes are reflected in `SKILL.md`, `README.md`, `docs/REFERENCE.md` and
      `CHANGELOG.md`.
- [ ] No new hard-coded paths, credentials, personal data or media files.
- [ ] Dependencies stay in the PEP 723 headers, with pins explained.
- [ ] Docs are in English.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).

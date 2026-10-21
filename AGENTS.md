# Instructions for AI agents working on this repository

This file is for coding agents (Claude Code, Codex, OpenCode, Kilo, ...) that change video-watch
itself. To *use* video-watch, follow `skills/video-watch/SKILL.md`; to install it for a user, follow
`INSTALL.md`.

## What this is

video-watch lets agents that read text and images "watch" videos: a transcript (subtitles or local
Whisper), an optional non-speech sound map, and timestamped frames, from YouTube, any yt-dlp site
and local files. Everything runs locally. Read [README.md](README.md),
[docs/REFERENCE.md](docs/REFERENCE.md) (the interface) and [docs/DESIGN.md](docs/DESIGN.md) (the
reasons) before changing behavior.

## Layout

| Path | Content |
| --- | --- |
| `skills/video-watch/SKILL.md` | The portable Agent Skill (`name: video-watch`, equal to the folder name). |
| `skills/video-watch/scripts/video_watch.py` | The whole CLI, one file with a PEP 723 header, run with `uv run`. |
| `skills/video-watch/scripts/sound_events.py` | The sound-map helper (PyTorch, its own PEP 723 header), run as a subprocess. |
| `tests/test_video_watch.py` | Unit tests (no GPU, no network). |
| `INSTALL.md` | Agent-guided install guide. |
| `.claude-plugin/` | Claude Code plugin and marketplace manifests. |
| `.github/` | CI workflow, issue and pull request templates. |
| `docs/` | Reference, design notes, research, help-wanted drafts. |

## Commands

Run from the repository root:

```bash
uv run --with pytest pytest -q                                   # unit tests
uv run --no-project --python ">=3.11" python skills/video-watch/scripts/video_watch.py doctor   # read-only report, no package install
uv run skills/video-watch/scripts/video_watch.py fetch "<url-or-path>"
uv run skills/video-watch/scripts/video_watch.py frames "<url-or-path>" --at 0:10
uv run skills/video-watch/scripts/video_watch.py clean --all
```

## Rules

- **Run the unit tests after every code change.** For changes to fetch, Whisper, the sound map or
  frames, also run one short public video and inspect the real output files (open the JPEGs, read
  `sounds.md`), then `clean`.
- **The interface is a contract.** Commands, flags, stdout keys and their order, stamp formats,
  `ERROR:` prefixes, exit codes, output file names, config keys and `VIDEO_WATCH_*` variables are
  parsed by the skill and by agents. Change them only when the task asks for it, and then update
  `SKILL.md`, `README.md`, `docs/REFERENCE.md` and `CHANGELOG.md` in the same change.
- **Do not change these without a discussion in an issue:**
  - the transcript source order (manual subtitles in the original language, otherwise Whisper;
    never a translated track; unknown language means Whisper; YouTube auto-captions only as the
    GPU fallback; `SOURCE:` always first);
  - no silent CPU fallback: without usable CUDA, stop with `ERROR:` and `HINT:` unless the user
    chose CPU mode;
  - cookies only after the user agrees, Firefox only, never on the first attempt;
  - local input files are only read; cleanup removes only direct children of the work directory and
    never follows links;
  - frames run on the CPU only, never trigger a fetch, and are taken only when the visuals matter;
  - frames must be time-exact: decode from the seek point, never from stream-copied sections;
  - the transcript is primary: a sound map that fails becomes a `NOTE:` and never changes `fetch`'s
    exit code; Whisper and the sound model never hold the GPU at the same time;
  - sound labels are acoustic cues, not emotions; no emotion model;
  - sound-map thresholds change only with a new calibration documented in `docs/DESIGN.md`.
- **Dependencies** belong in the PEP 723 headers. No global installs (`pip install` into system
  Python, `uv tool install`, system package managers) during development. Keep the `av<16` pin,
  the CUDA 12.8 PyTorch index, and the `mlx-whisper<0.5` pin with its `torch` override (the Apple
  GPU engine) unless you have tested a replacement on real hardware.
- **No hard-coded machine paths.** Use the config, platform directories and `tempfile`.
- **No personal data.** Never commit cookies, tokens, downloaded media, model files, transcripts of
  private videos, or user names and paths from your machine. Use neutral public test videos.
- **GPU tests are serialized and short.** Whisper large-v3 peaks at about 7.2 GB of VRAM at batch 8
  and the sound model at about 4.4 GB; never run two GPU jobs at once.
- **Keep `doctor`, `setup` and `config` standard-library only at import time**, so they keep working
  in the light form (`uv run --no-project --python ">=3.11" python …`) before any package download.
- **Git:** do not push, force-push, rewrite history or publish releases unless the human you work
  for explicitly asks.
- Documentation is in English.

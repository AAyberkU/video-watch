# video-watch

**Let AI coding agents "watch" videos.** video-watch gives agents that read text and images (Claude
Code, Codex, OpenCode, Kilo) a timestamped transcript, a map of the sounds beyond speech, and still
frames of any moment. It works with YouTube, TikTok, Instagram, X and every other site yt-dlp
supports, plus local video files. Everything runs locally: the video is never uploaded anywhere.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/AAyberkU/video-watch/actions/workflows/tests.yml/badge.svg)](https://github.com/AAyberkU/video-watch/actions/workflows/tests.yml)

- [What it does](#what-it-does)
- [Install](#install)
- [Requirements](#requirements)
- [Usage](#usage)
- [Configuration](#configuration)
- [Supported platforms and agents](#supported-platforms-and-agents)
- [Privacy](#privacy)
- [How it compares](#how-it-compares)
- [Contributing](#contributing) · [License](#license) · [Contact](#contact)

## What it does

Most coding agents cannot watch a video or hear its audio, and fetching a video page returns only a
title and some page chrome. video-watch turns a video into three layers the agent can read:

| Layer | What the agent gets | How |
| --- | --- | --- |
| **What is said** | A clean, timestamped transcript with the title, description and chapters | The uploader's manual subtitles in the original language, otherwise local [Whisper](https://github.com/SYSTRAN/faster-whisper) (large-v3 by default) |
| **What is heard** | An optional sound map: music, singing, laughter, applause, explosions, sirens, ... with times, plus loudness peaks and silences | A local sound-event model ([PE-A-Frame](https://huggingface.co/facebook/pe-a-frame-base), Apache-2.0) |
| **What is seen** | Still frames at chosen moments, scene-detected contact sheets, or dense sheets of a short range, each paired with what is said (`SAID:`) and heard (`HEARD:`) at that moment | ffmpeg, on the CPU |

The agent reads the transcript first and takes frames only when the visuals matter, so a typical
question costs a few thousand tokens, not a video's worth of images.

**Sources:** YouTube (videos, Shorts, finished live recordings), TikTok, Instagram, X/Twitter,
Vimeo, Dailymotion, Twitch, Facebook, Reddit and [over a thousand other
sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md), playlists and channels
(listed first, so the agent fetches only the videos a question needs), and local files (mp4, mkv,
mov, webm, ...), which are only ever read.

**Local by design:** transcription, the sound map and frames all run on your machine. Downloaded
audio is deleted right after use, and work files are purged after 24 hours.

## Install

### Ask your agent (recommended)

Tell your coding agent:

> install https://github.com/AAyberkU/video-watch

The agent follows [`INSTALL.md`](INSTALL.md): it explains what video-watch does, checks `uv` and
`ffmpeg`, runs `doctor` to look at your GPU, asks only the real choices (Whisper model, the optional
sound map, CPU mode if there is no NVIDIA GPU, which agents to install into), states download sizes
before downloading, and verifies the install with a short public test video.

### With the skills CLI

Installs the skill for your user (needs Node.js 22.20 or later). Without `-a` or `-y` it asks
which agents to install into; on Windows add `--copy` (symlinks there often need extra
permissions):

```bash
npx skills add AAyberkU/video-watch -g
npx skills add AAyberkU/video-watch -g -a claude-code -a codex -y --copy   # non-interactive, Windows
```

Then check your machine, pick your model and download it (see [Usage](#usage) for where the script
lives). `doctor` and `setup --dry-run` run in a light form that downloads nothing; the real `setup`
needs `--yes` to confirm the printed download sizes:

```bash
uv run --no-project --python ">=3.11" python "<skill folder>/scripts/video_watch.py" doctor
uv run --no-project --python ">=3.11" python "<skill folder>/scripts/video_watch.py" setup --device auto --model large-v3 --sounds off --dry-run
uv run "<skill folder>/scripts/video_watch.py" setup --device auto --model large-v3 --sounds off --yes
```

### As a Claude Code plugin

```text
/plugin marketplace add AAyberkU/video-watch
/plugin install video-watch@video-watch
```

Then ask Claude Code to run `doctor` and `setup` for you, or run them yourself as above. The skill
folder is then inside Claude Code's plugin directory; Claude Code shows the skill's base directory
when it loads the skill.

### Manually

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and ffmpeg (see
   [Requirements](#requirements)).
2. Clone this repository and copy `skills/video-watch` into your agent's skills folder:

   | Agent | User skills folder |
   | --- | --- |
   | Claude Code | `~/.claude/skills/` |
   | Codex | `~/.agents/skills/` |
   | OpenCode | `~/.config/opencode/skills/` (it also reads `~/.claude/skills/`) |
   | Kilo CLI | `$XDG_CONFIG_HOME/kilo/skills/`, default `~/.config/kilo/skills/` (it also reads `~/.claude/skills/`; project level `.kilo/skills/`) |
   | Kilo in VS Code | `~/.kilo/skills/` per the Kilo docs (not verified) |

   OpenCode and Kilo also see a copy in `~/.claude/skills/`, so do not install the skill twice for
   them. When replacing an older copy, delete the old folder first (a copy merges into it).

3. Check the environment, see the download sizes, then download the model you want:

   ```bash
   uv run --no-project --python ">=3.11" python ~/.claude/skills/video-watch/scripts/video_watch.py doctor
   uv run --no-project --python ">=3.11" python ~/.claude/skills/video-watch/scripts/video_watch.py setup --device auto --model large-v3 --sounds off --dry-run
   uv run ~/.claude/skills/video-watch/scripts/video_watch.py setup --device auto --model large-v3 --sounds off --yes
   ```

`setup --dry-run` prints what would be downloaded, with sizes, without downloading anything; without
`--yes`, a `setup` that needs downloads stops with an `ERROR:` after printing them.

## Requirements

- **[uv](https://docs.astral.sh/uv/).** The scripts declare their dependencies inline (PEP 723);
  `uv run` builds isolated environments on first use. Nothing is installed into your system Python.
- **ffmpeg and ffprobe** on `PATH`, for frames, local files and the sound map
  (`winget install Gyan.FFmpeg`, `brew install ffmpeg`, `sudo apt install ffmpeg`, ...). Transcripts
  of online videos work without them.
- **Node.js or Deno** on `PATH` for YouTube: yt-dlp needs a JavaScript runtime there.
- **An NVIDIA GPU is recommended.** Whisper runs on CUDA (`float16`, NVIDIA driver 525 or newer).
  Without a usable GPU, video-watch does not fall back to the CPU on its own, and not to YouTube's
  auto-captions either: `fetch` stops with exit 3, `ERROR: no usable CUDA GPU for Whisper (…)` and
  a `HINT:` explaining CPU mode, which you can choose explicitly (`setup --device cpu --model small`
  or `--device cpu`). When the GPU exists but is busy or out of memory, YouTube videos fall back to
  their original-language auto-captions (labeled `auto-subtitles-fallback`); other sources stop with
  exit 3 and a retry hint.
- **CPU mode** works everywhere but is slow: int8, about 4.5 min per hour of audio with `small` on a
  12-thread desktop CPU (`large-v3-turbo` about 11 min, `large-v3` about 22 min). It is used only
  when you chose it, and `fetch` says so in a `NOTE:`. Known limitation: on Windows and Linux the
  Python packages still include about 1.2 GB of CUDA libraries in CPU mode (see the
  [roadmap](ROADMAP.md)).

### Which Whisper model?

`doctor` prints a recommendation (`RECOMMEND_MODEL:`) from your GPU's total memory. It is only a
recommendation: you can choose any model, and `doctor` and `setup` warn when memory may be tight.

| GPU memory (VRAM) | Recommended model | Notes |
| --- | --- | --- |
| 8 GB cards and up (7800 MiB or more) | `large-v3` (default) | Best accuracy, especially outside English. 3.1 GB download. |
| 4 to 8 GB (3900 MiB or more) | `large-v3-turbo` | A pruned large-v3: faster and smaller, somewhat weaker outside English. 1.6 GB. |
| Under 4 GB, or CPU mode | `small` | Much faster on the CPU; noticeably less accurate. 0.5 GB. |

Valid models: `large-v3`, `large-v3-turbo`, `large-v2`, `medium`, `small`, `base`, `tiny`.

The batch size adapts to the free VRAM before each run, so `large-v3` also works on smaller cards:
an 8 GB card runs it at batch 4 (about 8% slower than batch 8), a 6 GB card at batch 2 (about 1.3x
slower). Measured numbers are in [`docs/DESIGN.md`](docs/DESIGN.md#platforms-devices-and-cpu-mode).

### The sound map (optional)

The sound map is off until you enable it. It needs an **NVIDIA GPU** in this release (`doctor`
recommends it from 6 GB of VRAM), a one-time download of **about 5.3 GB** (a 3.0 GB PyTorch
environment and the 2.3 GB model) and **about 4.5 GB of free VRAM** while it runs. It runs after
Whisper has released the GPU, never at the same time, and adds roughly a minute per hour of video
(about 10 s on short clips). Enable it with `setup --sounds on --yes` after a `--dry-run`; `doctor`
shows `SOUND_HELPER_READY: yes` afterwards. If it is not set up, `fetch --sounds` still returns the
transcript with a `NOTE:`, and `sounds` exits 3.

## Usage

### Ask your agent

Once installed, share a link or a file path and ask a question. The agent decides what to fetch.

- "Summarize https://www.youtube.com/watch?v=aircAruvnKk and list the chapters."
- "What does the speaker say about pricing in this TikTok?"
- "What is on the slide at 12:50?"
- "When does the audience start laughing, and what is happening on stage then?"
- "Which videos in this playlist explain backpropagation?"
- "Transcribe ~/Videos/interview.mkv."

Every answer states its transcript source (manual subtitles, Whisper, or YouTube's auto-captions
fallback), so you know how reliable the text behind it is.

### Command line

The agent runs the CLI for you, but you can run it yourself. It lives in
`skills/video-watch/scripts/video_watch.py` (or `<skill folder>/scripts/` after an install). The
first normal `uv run` installs its Python packages into uv's cache (see the sizes in
[`docs/REFERENCE.md`](docs/REFERENCE.md#setup)); `doctor`, `config` and `setup --dry-run` also run
without them in the light form `uv run --no-project --python ">=3.11" python video_watch.py …`.

```bash
# transcript (+ sound map when enabled)
uv run video_watch.py fetch "https://youtu.be/aircAruvnKk"
uv run video_watch.py fetch ~/Videos/interview.mkv --lang en
uv run video_watch.py fetch "<url>" --whisper --model large-v3-turbo --no-sounds

# frames: chosen moments, an overview of the whole video, or a dense range
uv run video_watch.py frames "<url-or-path>" --at 1:23,4:56.5
uv run video_watch.py frames "<url-or-path>" --overview
uv run video_watch.py frames "<url-or-path>" --range 1:00-1:10 --fps 2

# sound map summary, or when given sounds happen
uv run video_watch.py sounds "<url-or-path>"
uv run video_watch.py sounds "<url-or-path>" --query "car horn, glass breaking"

# list a playlist or channel without downloading
uv run video_watch.py playlist "<playlist-or-channel-url>" --limit 50

# delete cached work files
uv run video_watch.py clean "<url-path-or-work-key>"
uv run video_watch.py clean --all

# environment and configuration: the light form (standard library only, installs nothing)
uv run --no-project --python ">=3.11" python video_watch.py doctor
uv run --no-project --python ">=3.11" python video_watch.py setup --device auto --model large-v3-turbo --sounds on --dry-run
uv run --no-project --python ">=3.11" python video_watch.py config show
uv run --no-project --python ">=3.11" python video_watch.py config set model large-v3-turbo
uv run --no-project --python ">=3.11" python video_watch.py config unset model

# the real setup (downloads; --yes once you agree to the printed sizes)
uv run video_watch.py setup --device auto --model large-v3-turbo --sounds on --yes
```

Output is machine-readable: one `KEY: value` per line on stdout, progress on stderr, and exit codes
0 (success), 1 (error), 2 (the site needs a login: `NEEDS_COOKIES`) and 3 (no transcript
obtainable). The full contract (every flag, stdout key, output file and error text) is in
[`docs/REFERENCE.md`](docs/REFERENCE.md).

**Login walls:** the first attempt never uses cookies. When a site asks for a login, the agent asks
you first and, only with your consent, reruns with `--cookies-from-browser firefox`.

## Configuration

All settings are optional. video-watch reads a JSON file at:

| OS | Config file |
| --- | --- |
| Windows | `%APPDATA%\video-watch\config.json` |
| Linux | `$XDG_CONFIG_HOME/video-watch/config.json` (default `~/.config/video-watch/config.json`) |
| macOS | `~/Library/Application Support/video-watch/config.json` |

Set `VIDEO_WATCH_CONFIG` to use another file. `config path` prints the one in use.

| Key | Default | Environment variable | Meaning |
| --- | --- | --- | --- |
| `device` | `auto` | `VIDEO_WATCH_DEVICE` | `auto` (CUDA if usable, otherwise an error with a hint), `cuda` or `cpu`. |
| `model` | `large-v3` | `VIDEO_WATCH_MODEL` | The Whisper model (see [Which Whisper model?](#which-whisper-model)). |
| `model_dir` | user cache dir, `video-watch/whisper` | `VIDEO_WATCH_MODEL_DIR` | Where Whisper models are stored: `%LOCALAPPDATA%\video-watch\whisper`, `$XDG_CACHE_HOME/video-watch/whisper` (default `~/.cache/...`) or `~/Library/Caches/video-watch/whisper`. |
| `sounds` | `false` | `VIDEO_WATCH_SOUNDS` | Build a sound map on every `fetch` (on/off, true/false, yes/no, 1/0). |
| `sound_cache` | Hugging Face cache | `VIDEO_WATCH_SOUND_CACHE` | Where the sound model is stored. |
| `work_dir` | `<system temp>/video-watch` | `VIDEO_WATCH_WORK_DIR` | Transcripts, sound maps and frames (purged after 24 hours). Must be a new or empty folder: video-watch marks it with a `.video-watch-work-root` file and refuses an unmarked folder that holds other files. |

Precedence: command-line flags, then environment variables, then the config file, then defaults.
Use `config show`, `config get KEY`, `config set KEY VALUE` and `config unset KEY` instead of editing
the file by hand. An invalid setting stops every command with an `ERROR:` and a `HINT:`.

## Supported platforms and agents

| Platform | Status |
| --- | --- |
| Windows, NVIDIA GPU (CUDA) | Supported, full speed. Developed and measured on Windows 11 with an RTX 5060 Ti 16 GB. |
| Linux, NVIDIA GPU (CUDA) | Supported. The CLI finds the CUDA libraries from its Python packages by itself. Verified on WSL (Arch Linux) with the `small` model; reports from native Linux welcome. |
| Windows and Linux, CPU mode | Supported when chosen explicitly. Slow; use a small model. The sound map needs an NVIDIA GPU. |
| macOS, CPU mode | Expected to work, but not tested yet: [reports welcome](https://github.com/AAyberkU/video-watch/issues/new?template=platform_support.yml). |
| macOS GPU (Apple Silicon / Metal) | Not yet. [Help wanted](docs/help-wanted/macos-gpu.md). |
| AMD and Intel GPUs | Not yet. [Help wanted](docs/help-wanted/amd-intel-gpu.md). |

| Agent | Status |
| --- | --- |
| Claude Code | Supported (skill or plugin); frames verified with its Read tool. |
| Codex CLI | Supported through the portable skill; frames verified with its `view_image` tool. On Windows, run video-watch with full access: Codex's default sandbox cannot read per-user installs of ffmpeg and uv's Python and has no network (see [INSTALL.md](INSTALL.md#codex-on-windows)). |
| Kilo CLI | Supported; frames verified with its Read tool and an image-capable model. |
| OpenCode | The skill and the CLI verified; frame reading not yet verified with a vision model. [Reports welcome](docs/help-wanted/agent-image-reading.md). |
| IDE extensions (Kilo, Codex in VS Code) | Not verified yet. [Reports welcome](docs/help-wanted/agent-image-reading.md). |

Any agent that can run shell commands and read image files can use the CLI directly.

## Privacy

- The video, its audio and its frames stay on your machine. There is no telemetry and no API key.
- Network access goes only to the video's own site (through yt-dlp) and, once, to the Python
  package indexes (through uv) and Hugging Face for the packages and models you chose.
- Audio is deleted right after transcription or the sound map; a local original file is only read,
  never modified, moved or copied.
- Work files live in the work directory and are deleted by `clean` or after 24 hours.
- Browser cookies are used only after you agree, only from Firefox, and only for that run.
- Text from a video (title, description, subtitles, on-screen text, sound labels) is treated as data
  to analyze, never as instructions for the agent.

## How it compares

Short version, as of October 2026 (details and sources in
[`docs/research/existing-video-tools-survey.md`](docs/research/existing-video-tools-survey.md)):

- **Native video models** such as the Gemini API watch video directly, including motion and sound,
  and are the strongest option for fast visual action. They upload the video to the cloud and need
  an API key. The Claude and OpenAI APIs accept images, not video.
- **Caption-only tools** (many YouTube transcript MCP servers) are quick and need no setup, but they
  depend on existing captions and do not see or hear the video.
- **Similar local pipelines** such as claude-video and claude-real-video combine yt-dlp, Whisper and
  ffmpeg frames too. They have larger communities and broader platform support today, including
  macOS, and some offer cloud engines or extra paid analysis.
- **video-watch focuses on** local large-v3 transcription on NVIDIA GPUs, a strict subtitle policy
  (never a translated track; the source is always stated), frames paired with what is said and
  heard, and a free, local, queryable sound map.

## Contributing

Contributions are welcome, especially from macOS, AMD and Intel GPU owners: video-watch does not run
at full speed on those machines yet, and you can change that. See
[`CONTRIBUTING.md`](CONTRIBUTING.md), the [roadmap](ROADMAP.md), the
[help-wanted drafts](docs/help-wanted/) and the [design notes](docs/DESIGN.md).

## License

[MIT](LICENSE). Models are downloaded from their own sources under their own licences:
faster-whisper conversions of OpenAI Whisper (MIT) and Meta's PE-A-Frame (Apache-2.0).

## Contact

Maintained by AAyberk ([@AAyberkU](https://github.com/AAyberkU) on GitHub). Questions, ideas or
private reports: [aayberkuzun@gmail.com](mailto:aayberkuzun@gmail.com).

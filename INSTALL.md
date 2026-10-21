# Installing video-watch (guide for AI agents)

This guide is for an AI coding agent (Claude Code, Codex, OpenCode, Kilo, or similar) whose user
asked something like "install https://github.com/AAyberkU/video-watch". Follow the steps in order.
People who prefer to install by hand can use the manual section of [README.md](README.md).

## Ground rules

- Talk to the user in **their** language, in plain words. Keep technical output out of your
  messages unless they ask for it.
- Blocks marked **SAY** are what to tell the user (adapt the wording, keep the facts). Blocks marked
  **RUN** are commands for you to run. Use the variant for the user's OS and shell.
- **Ask before** installing any program, before any download larger than about 100 MB, and before
  replacing an existing file or folder. State the size when you ask.
- Ask only the decisions listed in step 5. Everything else uses the defaults silently.
- Do only what this guide describes. No hidden or unrelated actions, no system-wide Python
  packages, no changes to other tools' settings. video-watch has no telemetry and sends nothing
  anywhere; keep it that way.
- If a step fails, explain the error simply, suggest the fix from this guide, and stop when there
  is no clear fix. Never work around a failure by guessing.
- Commands must never wait for input: keep the non-interactive flags shown here (`--yes`, `-y`).

### Codex on Windows

Codex's default sandbox on Windows runs commands as a separate user that cannot read per-user
installs (a winget ffmpeg under `%LOCALAPPDATA%`, uv's Python under `%APPDATA%\uv\python`) and has
no network, so video-watch fails there in confusing ways. Run its commands with full access: ask
the user to switch with `/permissions` → Full access (or start `codex exec -s danger-full-access`),
or approve the escalation when Codex asks. A machine-wide ffmpeg install may also help (untested).

## Step 1: Introduce video-watch

**SAY** (in the user's language, in your own words):

> video-watch lets me "watch" videos for you. When you send me a video link (YouTube, TikTok,
> Instagram, X, Vimeo and many other sites) or a video file on your computer and ask about it, I
> get what is said (subtitles, or a speech-to-text model running on your computer), optionally
> which sounds are heard (music, laughter, applause, ...), and still pictures from the moments
> that matter. Everything runs on your computer; the video is never uploaded anywhere. I will check
> your computer first, ask you a few short questions, and then set it up.

## Step 2: Check the prerequisites

**RUN** these checks (each prints a version, or fails when the program is missing):

```
uv --version
ffmpeg -version
ffprobe -version
nvidia-smi
node --version
git --version
```

Note the OS too (Windows, macOS or Linux, and the CPU architecture).

- `uv` (required) runs the tool and fetches Python and its packages by itself. If missing, ask the
  user, then install it with the official installer:
  - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - macOS and Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Alternatives if the user prefers a package manager: `winget install --id=astral-sh.uv -e`
    (Windows), `brew install uv` (macOS).
- `ffmpeg` and `ffprobe` (required for still frames, sounds and local files) come together. If
  missing, ask the user, then offer the command for their system:
  - Windows: `winget install --id=Gyan.FFmpeg -e` (or `choco install ffmpeg`, or `scoop install ffmpeg`)
  - macOS: `brew install ffmpeg`
  - Debian or Ubuntu: `sudo apt install ffmpeg`
  - Fedora: `sudo dnf install ffmpeg` (may need the RPM Fusion repository); Arch: `sudo pacman -S ffmpeg`

  `sudo` asks for the user's password; let the user run that command themselves if your shell
  cannot prompt.
- `nvidia-smi` working means an NVIDIA GPU with a driver is present. Missing is not an error;
  step 5 explains the options.
- `node` (version 22.20 or later) and `git` are optional for the install itself; they decide how
  step 6 installs the skill. For YouTube, though, yt-dlp needs a JavaScript runtime (Node.js or
  Deno) on `PATH`. If neither exists, tell the user that YouTube videos may fail until one is
  installed, and offer the official Node.js LTS installer (ask first).

After installing anything, a new terminal may be needed before the program is found on `PATH`. Run
the check again before going on.

## Step 3: Get a local copy of the repository

Use a new, empty temporary folder, for example `<temp>/video-watch-install`
(`$env:TEMP\video-watch-install` on Windows, `/tmp/video-watch-install` elsewhere).

**RUN**, when `git` exists:

```
git clone --depth 1 https://github.com/AAyberkU/video-watch "<temp>/video-watch-install"
```

Otherwise download and unpack the ZIP (about 1 MB):

- Windows (PowerShell):
  ```
  Invoke-WebRequest https://github.com/AAyberkU/video-watch/archive/refs/heads/main.zip -OutFile "$env:TEMP\video-watch.zip"
  Expand-Archive "$env:TEMP\video-watch.zip" "$env:TEMP\video-watch-install"
  ```
- macOS and Linux:
  ```
  curl -L -o /tmp/video-watch.zip https://github.com/AAyberkU/video-watch/archive/refs/heads/main.zip
  unzip /tmp/video-watch.zip -d /tmp/video-watch-install
  ```

On every OS the ZIP unpacks into a `video-watch-main` subfolder
(`<temp>/video-watch-install/video-watch-main`).

Below, `<repo>` means that local copy (the folder that contains `INSTALL.md`). The CLI is
`<repo>/skills/video-watch/scripts/video_watch.py`.

## Step 4: Run the environment check (`doctor`)

`doctor` only reads: it downloads no models and no packages. Run it in the **light form**, which
uses only Python's standard library (uv fetches a small Python of about 30 MB once if it has none):

**RUN:**

```
uv run --no-project --python ">=3.11" python "<repo>/skills/video-watch/scripts/video_watch.py" doctor
```

It always exits 0 and prints `KEY: value` lines. What they mean:

| Key | Meaning |
| --- | --- |
| `OS:`, `PYTHON:`, `UV:` | System and tool versions (`UV: missing` when uv is not on `PATH`). |
| `FFMPEG:`, `FFPROBE:` | Version or path; `missing` (step 2); or `blocked: <path> (<reason>)` when they are installed but this process may not run them (an agent sandbox, see [Codex on Windows](#codex-on-windows)). |
| `GPU:`, `VRAM_TOTAL_MB:`, `VRAM_FREE_MB:` | The NVIDIA GPU with its driver (or `none`) and its memory. |
| `CUDA:` | `ok (…)`, or `unavailable: <reason>` (no NVIDIA GPU, a driver older than 525, or macOS). |
| `DISK_FREE_GB:` | Free space on the drive of the model folder. |
| `CONFIG:` | The settings file (`(not created yet)` on a new install). |
| `DEVICE:`, `MODEL:`, `MODEL_DIR:`, `SOUNDS:`, `WORK_DIR:` | Current settings (defaults on a new install). |
| `MODEL_CACHED:`, `SOUND_MODEL_CACHED:` | Whether the models are already downloaded. |
| `SOUND_HELPER_READY:` | Whether `setup --sounds on` has prepared the sound map's PyTorch environment. `no` until that setup ran once, also on machines that already have the sound model. |
| `RECOMMEND_MODEL:`, `RECOMMEND_DEVICE:`, `RECOMMEND_SOUNDS:` | What the tool suggests for this machine; `RECOMMEND_SOUNDS:` gives its reason in brackets. |
| `WARN:` | Something to tell the user (for example a missing model, low disk space or tight GPU memory). |
| `READY:` | `yes` when transcription can run now: a usable device, the model downloaded, ffmpeg and ffprobe found and runnable, an acceptable work folder, and (if the sound map is on) the sound model downloaded, `SOUND_HELPER_READY: yes` and an NVIDIA GPU present. `no` is normal before step 7. |

**SAY** a short summary in plain words, for example: "Your computer has an NVIDIA RTX ... with 8 GB
of graphics memory, so the most accurate speech model fits. ffmpeg is installed." Mention each
`WARN:` simply, except the "not downloaded yet" ones, which step 7 takes care of. If `CUDA:` is
`unavailable` although `nvidia-smi` worked, say the graphics driver may need an update (version 525
or newer) and that this can be fixed later.

## Step 5: Ask the decisions

Ask these together in one short message, each with your recommendation, so the user can answer
"OK" to take all recommendations. Ask (c) only when it applies. Do not ask anything else.

**(a) Speech model** (turns speech into text when a video has no subtitles).
Recommend `RECOMMEND_MODEL:` (`large-v3` from about 8 GB of graphics memory, `large-v3-turbo` from
about 4 GB, otherwise and in CPU mode `small`). The user may pick any of `large-v3`,
`large-v3-turbo`, `large-v2`, `medium`, `small`, `base`, `tiny`. In plain words: bigger models are
more accurate but need more graphics memory and download space; `large-v3` is the most accurate
(about 3.1 GB download), `large-v3-turbo` is faster and lighter with a little less accuracy (about
1.6 GB), `small` suits weak GPUs and CPU mode (about 0.5 GB). On a 6 GB card `large-v3` still works,
about 1.3 times slower, because the tool picks a smaller batch automatically. If the user picks a
model bigger than recommended, pass on the `WARN:` that `setup` prints (memory may be tight, or it
will likely run out of memory) and respect their choice.

**(b) Sound map** (optional; offer it only when there is an NVIDIA GPU).
Explain: "Besides speech, I can also tell which sounds are heard and when: music, singing,
laughter, applause, explosions, sirens, and so on, with timestamps. It needs an extra download of
about 5 GB, about 4.5 GB of free graphics memory while it runs, works only with an NVIDIA GPU, and
adds about one minute per hour of video. You can switch it on or off later."
Recommend **yes** when `RECOMMEND_SOUNDS:` says yes (an NVIDIA GPU with 6 GB or more); otherwise
recommend no, and do not offer it at all when there is no NVIDIA GPU.

**(c) CPU mode** (only when `CUDA:` is unavailable or `GPU:` is `none`, including every Mac).
Explain: "Your computer has no usable NVIDIA graphics card, so speech recognition would run on the
processor. That works but is much slower: with the `small` model roughly 5 minutes per hour of
video on a typical desktop processor, longer on laptops; `large-v3-turbo` takes about 2.5 times
and `large-v3` about 5 times as long. Videos with subtitles are unaffected. Do you want CPU mode?" Recommend CPU mode with the
`small` model, unless the user would rather fix the GPU driver first. The sound map is not
available without an NVIDIA GPU. On Windows and Linux the tool's packages still include about
1.2 GB of NVIDIA libraries even in CPU mode (a known limitation); Macs do not download them.
macOS CPU mode is expected to work but has not been tested yet; say so honestly.
Never choose CPU mode without the user's yes; the tool never falls back to the CPU by itself.

**(d) Which agents get the skill.**
Detect installed agents and offer those, recommending all detected ones plus the agent you are
running in:

| Agent | Signs it is installed |
| --- | --- |
| Claude Code | `claude` on `PATH`, or a `~/.claude` folder |
| Codex | `codex` on `PATH`, or a `~/.codex` folder |
| OpenCode | `opencode` on `PATH`, or a `~/.config/opencode` folder |
| Kilo | `kilo` on `PATH`, a `~/.kilo` or `~/.config/kilo` folder, or the Kilo extension in VS Code |

(`~` is the user's home folder, `%USERPROFILE%` on Windows.)

Defaults that are not asked: device `auto` (GPU), the standard model folder, and the standard work
folder in the system temp directory.

## Step 6: Install the skill into the chosen agents

The skill is the folder `skills/video-watch` (instructions plus the CLI). Install it globally
(for all projects). If a `video-watch` skill folder already exists in a target, ask before
replacing it.

- **One copy per agent.** OpenCode and Kilo also read `~/.claude/skills`, so a Claude Code copy
  there is already seen by them (Kilo logs a harmless duplicate-name warning when it finds the
  skill twice). Do not install the same skill twice for them.
- **Other video skills.** If another installed skill has an overlapping description (another
  video, YouTube or transcript skill), ask the user whether to remove it, so the right skill
  triggers.

### Route A: `npx skills` (when Node 22.20 or later exists)

**RUN**, with one `-a` per chosen agent (`claude-code`, `codex`, `opencode`, `kilo`). `--yes` lets
`npx` fetch the small `skills` package without a prompt, and `-y` answers the installer's own
questions:

```
npx --yes skills add AAyberkU/video-watch -g -a claude-code -a codex -a opencode -a kilo -y
```

On Windows add `--copy` (symlinks there often need extra permissions). Afterwards
`npx --yes skills list -g` shows where the skill went.

### Route B: copy the folder by hand

Copy `<repo>/skills/video-watch` (the whole folder, including `scripts/`) into each target:

| Agent | Windows | macOS and Linux |
| --- | --- | --- |
| Claude Code | `%USERPROFILE%\.claude\skills\video-watch` | `~/.claude/skills/video-watch` |
| Codex | `%USERPROFILE%\.agents\skills\video-watch` | `~/.agents/skills/video-watch` |
| OpenCode | `%USERPROFILE%\.config\opencode\skills\video-watch` | `~/.config/opencode/skills/video-watch` |
| Kilo CLI | `%USERPROFILE%\.config\kilo\skills\video-watch` | `$XDG_CONFIG_HOME/kilo/skills/video-watch` (default `~/.config/kilo/skills/video-watch`) |

- OpenCode and Kilo also read `~/.claude/skills`: when Claude Code already has a copy there, skip
  their own folders.
- Kilo: the CLI reads `$XDG_CONFIG_HOME/kilo/skills` (default `~/.config/kilo/skills`) for the user
  and `.kilo/skills` inside a project. The Kilo docs list `~/.kilo/skills` for the VS Code
  extension (not verified); use it only for the extension.
- **Replacing an existing copy:** delete the old `video-watch` folder first (after the user agreed);
  `Copy-Item` and `cp` merge into an existing folder and would keep stale files.

**RUN** (example for one target; repeat per target):

- Windows (PowerShell):
  ```
  New-Item -ItemType Directory -Force "$HOME\.agents\skills" | Out-Null
  Copy-Item -Recurse "<repo>\skills\video-watch" "$HOME\.agents\skills\"
  ```
- macOS and Linux:
  ```
  mkdir -p ~/.agents/skills
  cp -R "<repo>/skills/video-watch" ~/.agents/skills/
  ```

### Claude Code plugin (alternative for Claude Code only)

If the user prefers Claude Code's plugin system (updates through `/plugin`), they can type in
Claude Code `/plugin marketplace add AAyberkU/video-watch` and then
`/plugin install video-watch@video-watch` (or run `claude plugin marketplace add AAyberkU/video-watch`
and `claude plugin install video-watch@video-watch` in a terminal). Use either the plugin or a
`~/.claude/skills` copy for Claude Code, not both.

Below, `<skill-dir>` means one installed copy of the skill folder, for example
`~/.claude/skills/video-watch` or `~/.agents/skills/video-watch`. With the plugin only, the skill
folder is inside Claude Code's plugin directory (under `~/.claude/plugins/`); Claude Code shows the
skill's base directory when it loads the skill, or use `<repo>/skills/video-watch` from step 3 for
steps 7 and 8.

## Step 7: Apply the choices (`setup`)

First see what would be downloaded. **RUN** the dry run in the light form, with the user's choices
(it downloads and writes nothing):

```
uv run --no-project --python ">=3.11" python "<skill-dir>/scripts/video_watch.py" setup --device auto --model large-v3 --sounds on --dry-run
```

(`--device cpu --model small` for CPU mode; `--sounds off` when the user declined the sound map.)

It prints `DEVICE:`, `MODEL:`, `SOUNDS:`, `MODEL_DIR:`, one `DOWNLOAD: <what>, about X.X GB` line
per download, `DOWNLOAD_TOTAL_GB:`, any `WARN:` lines, `CONFIG: <path> (would write {...})` and
`SETUP: dry-run`. The possible downloads:

| Download | Size |
| --- | --- |
| The tool's Python packages (once, into uv's cache) | about 1.4 GB on Windows, 1.5 GB on Linux (mostly NVIDIA CUDA libraries, also in CPU mode), 0.2 GB on macOS |
| Whisper model | `large-v3` 3.1 GB, `large-v3-turbo` 1.6 GB, `large-v2` 3.1 GB, `medium` 1.5 GB, `small` 0.5 GB, `base` 0.15 GB, `tiny` 0.08 GB |
| Sound map (only with `--sounds on`) | PyTorch environment about 3.0 GB plus the model about 2.3 GB |

With `--sounds on`, the plan lists `DOWNLOAD: sound map environment (PyTorch with CUDA 12.8,
transformers) into the uv cache (less if uv already has it), about 3.0 GB` until a `setup --sounds on
--yes` has completed once, even when the sound model is already cached (`SOUND_HELPER_READY: no`).

**SAY** the `DOWNLOAD_TOTAL_GB:` and what it is made of, plus any `WARN:`, and ask for a go-ahead
(for example "This downloads about 4.5 GB: the tool's components and the speech model. Start
now?"). Continue only with a yes.

Then **RUN** `setup` in the normal form (it needs the tool's packages and installs them itself),
with the same choices, without `--dry-run` and with `--yes`, which records that the user agreed:

```
uv run "<skill-dir>/scripts/video_watch.py" setup --device auto --model large-v3 --sounds on --yes
```

It prints the same plan, then `MODEL_CACHED: yes`, `SOUND_MODEL_CACHED: yes` (with the sound map),
`CONFIG: <path>` and `SETUP: done` (exit 0). The settings file is written only after every download
succeeded. Downloads can take several minutes; tell the user. On `ERROR:` (exit 1), explain it and
the `HINT:` simply. Common ones:

- `ERROR: setup needs to download about X GB`: `--yes` was missing; ask the user, then add it.
- `ERROR: CUDA is unavailable for device auto: …`: there is no usable NVIDIA GPU; go back to (c).
- `ERROR: the sound map needs an NVIDIA GPU, and none was found`: rerun with `--sounds off`.

## Step 8: Verify with a short public video

Use a short public video, for example `https://www.youtube.com/watch?v=jNQXAC9IVRw` (19 seconds).
`--whisper` makes sure the speech model is tested, not just subtitles.

**RUN:**

```
uv run "<skill-dir>/scripts/video_watch.py" fetch "https://www.youtube.com/watch?v=jNQXAC9IVRw" --whisper
uv run "<skill-dir>/scripts/video_watch.py" frames "https://www.youtube.com/watch?v=jNQXAC9IVRw" --at 0:05
```

Check: `fetch` prints `SOURCE: whisper` and a `TRANSCRIPT:` path whose file contains text (and,
with the sound map on, a `SOUNDS:` line; a `NOTE: sound map skipped: …` there is not an install
failure, explain it); `frames` prints one `FRAME:` path; open that image with your image-reading
tool and confirm you can see it. Then **RUN**:

```
uv run "<skill-dir>/scripts/video_watch.py" clean "https://www.youtube.com/watch?v=jNQXAC9IVRw"
```

If YouTube asks for a login (`NEEDS_COOKIES`), try another short public video instead of using
cookies.

Finally delete the temporary copy from step 3 (and the ZIP, if any).

## Step 9: Tell the user it is ready

**SAY** a short summary, for example:

> video-watch is installed for Claude Code and Codex. It uses the `large-v3` speech model on your
> NVIDIA GPU, and the sound map is on. Settings are in `<CONFIG: path>`. To use it, just send me a
> video link or a video file and ask about it, for example "Summarize this video" or "What happens
> at 2:10?". Agents that were already open may need a restart to see the new skill.

## Changing choices later

When the user asks to change something, read the current settings first (light form, nothing is
downloaded):

```
uv run --no-project --python ">=3.11" python "<skill-dir>/scripts/video_watch.py" config show
uv run --no-project --python ">=3.11" python "<skill-dir>/scripts/video_watch.py" doctor
```

`config show` prints `CONFIG: <path>` and then one `KEY: value (flag|env|config|default)` line per
setting. Use the current values for everything the user does not change. Below, "light" means the
light form above and "normal" means `uv run "<skill-dir>/scripts/video_watch.py" …`.

- **Turn the sound map on:** explain it as in step 5 (b) (about 5 GB, about 4.5 GB of free graphics
  memory, NVIDIA only), run `setup --device <current> --model <current> --sounds on --dry-run`
  (light), ask with the printed size, then the same `setup` with `--yes` (normal).
- **Turn the sound map off:** `config set sounds off` (light; prints `SET: sounds = false`). Nothing
  is downloaded. Already downloaded sound components stay in the caches until removed (see
  uninstall).
- **Change the speech model:** explain the trade-off as in step 5 (a), run
  `setup --device <current> --model <name> --sounds <current> --dry-run` (light), ask about the
  download size, then the same `setup` with `--yes` (normal).
- **Switch between GPU and CPU:** the same way with `setup --device auto …` or
  `setup --device cpu --model small …` (CPU mode is slow; explain it as in step 5 (c)).
- **Other settings:** `config set KEY VALUE` and `config unset KEY` (light; keys `device`, `model`,
  `model_dir`, `sounds`, `sound_cache`, `work_dir`; `sounds` accepts on/off, true/false, yes/no,
  1/0). `config unset` returns a key to its default. A `work_dir` must be a new or empty folder: the
  tool deletes old work files inside it.
- **Existing users with the sound map on:** after an update, `doctor` shows
  `SOUND_HELPER_READY: no` and `READY: no` until `setup --sounds on --yes` (normal form) runs once;
  the sound map is skipped until then. Use the current device and model, and state the printed size
  first (often much less than 3 GB when uv already has the packages).
- **Update video-watch:** `npx --yes skills update -g -y` (route A), `/plugin` in Claude Code (plugin
  route), or repeat steps 3 and 6 to replace the copied folders (route B; ask before replacing, and
  delete the old folder first).
- **Uninstall:** ask which parts to remove, then:
  - the skill: `npx --yes skills remove video-watch -g -a <agents> -y` (route A), delete the copied
    `video-watch` folders (route B), or `/plugin uninstall video-watch@video-watch` (plugin);
  - the settings: the folder of the `CONFIG:` path (`%APPDATA%\video-watch` on Windows,
    `~/.config/video-watch` on Linux, `~/Library/Application Support/video-watch` on macOS);
  - the speech models: the `model_dir` folder (`%LOCALAPPDATA%\video-watch\whisper` on Windows,
    `~/.cache/video-watch/whisper` on Linux, `~/Library/Caches/video-watch/whisper` on macOS);
  - the sound model: `models--facebook--pe-a-frame-base` in the Hugging Face cache
    (`~/.cache/huggingface/hub`, or the configured `sound_cache`);
  - the sound map's marker: `sound-helper.json` in the cache folder above the model folder
    (`%LOCALAPPDATA%\video-watch`, `~/.cache/video-watch` or `~/Library/Caches/video-watch`);
  - the work files: `uv run "<skill-dir>/scripts/video_watch.py" clean --all` before removing the
    skill;
  - the Python packages live in uv's shared cache; `uv cache prune` removes unused entries.
    `uv cache clean` frees everything, also for other tools that use uv, so ask before suggesting it.

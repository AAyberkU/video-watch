# video-watch reference

The full command-line contract: commands, flags, stdout keys, exit codes, output files and
behavior. The skill ([`skills/video-watch/SKILL.md`](../skills/video-watch/SKILL.md)) relies on it,
so changes to anything here are interface changes (see [CONTRIBUTING](../CONTRIBUTING.md#interface-stability)).
For the reasons behind the design, see [DESIGN.md](DESIGN.md).

In the examples, `video_watch.py` stands for `skills/video-watch/scripts/video_watch.py` (or the
`scripts/` folder of an installed skill), always run with `uv run`. The first normal `uv run`
installs the script's Python packages into uv's cache (sizes under [`setup`](#setup)).

`doctor`, `config` and `setup --dry-run` use only the standard library, so they also run in a
**light form** that installs no packages (uv fetches a small Python once if it has none):

```text
uv run --no-project --python ">=3.11" python video_watch.py doctor
```

A real `setup` (with downloads) needs the packages; started in the light form, it reruns itself as
`uv run --script video_watch.py setup …` with the same arguments.

- [Commands](#commands)
- [Accepted inputs](#accepted-inputs)
- [Transcripts (`fetch`)](#transcripts-fetch)
- [Output files](#output-files)
- [Exit codes and errors](#exit-codes-and-errors)
- [Cache and lifecycle](#cache-and-lifecycle)
- [Sound map](#sound-map)
- [Frames](#frames)
- [Playlists](#playlists)
- [Environment: `doctor`, `setup`, `config`](#environment-doctor-setup-config)
- [Devices: GPU and CPU](#devices-gpu-and-cpu)
- [Cookies](#cookies)
- [Whisper settings](#whisper-settings)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

## Commands

```text
uv run video_watch.py fetch <url-id-or-path> [--whisper] [--lang CODE] [--no-vad] [--model NAME] [--device auto|cuda|metal|cpu] [--sounds | --no-sounds] [--cookies-from-browser BROWSER]
uv run video_watch.py frames <url-id-or-path> --at 1:23,4:56[,...] [--max-frames 200] [--cookies-from-browser BROWSER]
uv run video_watch.py frames <url-id-or-path> --overview [--interval SECONDS] [--scene-threshold 0.3] [--max-frames 200] [--cookies-from-browser BROWSER]
uv run video_watch.py frames <url-id-or-path> --range 1:00-1:10 [--fps 2] [--max-frames 200] [--cookies-from-browser BROWSER]
uv run video_watch.py sounds <url-id-or-path> [--query "car horn, glass breaking"] [--threshold 4] [--cookies-from-browser BROWSER]
uv run video_watch.py playlist <url> [--limit 200] [--cookies-from-browser BROWSER]
uv run video_watch.py clean <video-id-url-path-or-work-key>
uv run video_watch.py clean --all
uv run video_watch.py doctor
uv run video_watch.py setup [--device auto|cuda|metal|cpu] [--model NAME] [--sounds on|off] [--model-dir PATH] [--yes] [--dry-run]
uv run video_watch.py config show | path | get KEY | set KEY VALUE | unset KEY
```

### `fetch` flags

| Flag | Meaning |
| --- | --- |
| `--whisper` | Force Whisper even when manual or embedded subtitles exist. |
| `--lang CODE` | Override the video's spoken language (for example `en`, `de`, `tr`). On sites that do not state a language, it also lets a manual subtitle track in that language be used. |
| `--no-vad` | Disable voice activity detection, for music-heavy videos where speech gets dropped. Uses the sequential decoder, about 1.6x slower. Rarely needed by hand: a VAD run that finds almost no speech is retried without VAD automatically (see [No-VAD auto-retry](#no-vad-auto-retry)). |
| `--model NAME` | Whisper model for this run: `large-v3` (default), `large-v3-turbo`, `large-v2`, `medium`, `small`, `base`, `tiny`. `large-v2` helps on videos where v3 falls into repetition loops. |
| `--device auto\|cuda\|metal\|cpu` | Device for this run. `auto` means the GPU of the machine: the NVIDIA GPU, or the Apple GPU on an Apple Silicon Mac. `cuda` is the NVIDIA GPU and `metal` the Apple GPU. Without a usable GPU, `fetch` stops with exit 3 (see [Devices](#devices-gpu-and-cpu)); it never switches to the CPU silently. `cpu` runs Whisper on the CPU (int8, slow). |
| `--sounds` | Build the [sound map](#sound-map) for this run even when it is disabled in the configuration. If it is not set up, the transcript is still printed with `NOTE: sound map skipped: the sound map is not set up yet …` and exit 0; nothing is downloaded. `--sounds` and `--no-sounds` together are a usage error. |
| `--no-sounds` | Skip the sound map and its back-fill for an older cached transcript. A sound map already cached is still reported. |
| `--cookies-from-browser firefox` | Use the browser's cookies for the site. Never on the first attempt: only after `NEEDS_COOKIES` (exit 2) and only with the user's consent. See [Cookies](#cookies). |

### `frames` flags

Exactly one of `--at`, `--overview` and `--range`:

| Flag | Meaning |
| --- | --- |
| `--at TIMES` | Comma-separated moments: `ss`, `ss.s`, `m:ss` or `h:mm:ss`. Transcript stamps such as `[01:23]` are accepted too. |
| `--overview` | Contact sheets of the whole video, picked by scene changes plus interval fills. |
| `--range START-END` | Dense frames over a window (`1:00-1:10`) on labeled contact sheets. |
| `--fps N` | Range only: frames per second, default 2, at most 10. |
| `--interval SECONDS` | Overview only: the longest stretch without a pick. Default: about 1/30 of the duration, between 10 and 60 s. |
| `--scene-threshold X` | Overview only: ffmpeg scene score threshold between 0 and 1. Default 0.3; lower finds more changes. |
| `--max-frames N` | Safety ceiling per call, default 200. Too many `--at` moments or range frames is an error; an overview above it is thinned evenly with a `NOTE:`. |
| `--cookies-from-browser BROWSER` | As for `fetch`. |

## Accepted inputs

`fetch`, `frames` and `sounds` accept the same inputs, classified without any network request:

- **YouTube:**
  - a bare 11-character video ID (an existing file of that name in the current folder is taken as
    the local file instead);
  - `youtube.com/watch?v=`, `m.youtube.com`, `music.youtube.com`, `youtu.be/`, `/shorts/`,
    `/live/`, `/embed/` and `youtube-nocookie.com` URLs. Extra parameters (`t=`, `si=`, `list=`)
    are ignored; only the single video is processed.
- **Any other http(s) URL:** handed to yt-dlp. A URL without a scheme (`vimeo.com/123`) gets
  `https://`.
- **Local files:** an absolute or relative path (quotes and spaces are fine) or a `file:` URI, to
  anything ffprobe sees with a video or audio stream. The original is only ever read.
- **Rejected with exit 3:**
  - playlist, channel and album URLs (the `HINT:` points to [`playlist`](#playlists));
  - content that is live now, upcoming, or still processing after a stream. Finished, processed
    recordings of live streams are ordinary videos and work;
  - DRM-protected content;
  - pages without a video, unavailable videos, missing local files, and (for `fetch`) videos
    without an audio track.

## Transcripts (`fetch`)

### Transcript sources

| Source | First choice | Otherwise | GPU busy or out of memory |
| --- | --- | --- | --- |
| YouTube | Manual subtitles in the original language; the metadata must state it | Whisper | Original-language auto-captions (`-orig`), labeled `auto-subtitles-fallback` |
| Other sites | Manual subtitles in the original language; the metadata must state it | Whisper | Exit 3, retry later |
| Local files | An embedded text subtitle stream whose language tag matches the audio's | Whisper | Exit 3, retry later |

No usable CUDA at all (no NVIDIA GPU, driver or CUDA libraries) is different: every source exits 3
with no fallback, also on YouTube (see [Devices](#devices-gpu-and-cpu)).

- **Unknown language means Whisper** for every source: a track whose language cannot be compared
  with the spoken language could be a translation. `--lang` supplies the language when the metadata
  does not. Translated tracks are never used.
- **Not subtitles:** chat replays and comment overlays (`live_chat`, `danmaku`, `rechat`,
  `comments`), track names marking machine captions (`auto`, `-orig`) on other sites, and all of
  TikTok's `subtitles` (machine captions, often machine-translated).
- **Embedded subtitles:** text codecs only (subrip, ass/ssa, webvtt, mov_text). Bitmap subtitles
  (PGS, VobSub, DVB) and forced tracks are skipped. Language tags compare by base language across
  two- and three-letter codes (`eng` matches `en`). The stream is converted to SRT through a pipe.
  `SOURCE:` stays `manual-subtitles`; the header and a `NOTE:` name the stream.

### Local files

- The original file is **only read**: never modified, moved, copied or deleted. Every output goes
  into the work directory, and files inside the work root are rejected as input.
- **Metadata:** title from the container's title tag, else the file name; duration and chapters via
  ffprobe; description from a `description`, `comment` or `synopsis` tag. The transcript header
  shows `- File: <absolute path>` instead of `- URL:`.
- **Audio for Whisper:** ffmpeg decodes the default audio stream into a 16 kHz mono WAV in the work
  directory, deleted right after use. The audio's language tag is passed to Whisper.
- **Work key:** `file-<hash>` of the resolved path, size and modification time, so an edited file
  gets a fresh transcript.

### stdout

One `KEY: value` per line; the first line on success is always `SOURCE:`.

```text
SOURCE: manual-subtitles | whisper | auto-subtitles-fallback   (manual-subtitles includes embedded ones)
TITLE: …
CHANNEL: …
DURATION: mm:ss | h:mm:ss
LANGUAGE: <subtitle track key or Whisper language code>
TRANSCRIPT: <absolute path>
PARAGRAPHS: <absolute path>
SEGMENTS_JSON: <absolute path>
CHARS: <characters in transcript.md>
CHAPTERS: <number of chapters; 0 when the video has none>
SPEECH: <n> words/min; longest gap mm:ss (mm:ss-mm:ss)   (Whisper sources only)
SOUNDS: music 39%, speech 61%, singing 0%; laughter 3, applause 2 (+3 more labels)   (when a sound map exists)
SOUNDS_FILE: <absolute path of sounds.md>   (with SOUNDS:)
CACHED: yes | no
NOTE: … (optional, may repeat)
```

On failure stdout carries `ERROR: <message>` and optionally `HINT: <text>`. Progress goes to
stderr. Before Whisper starts, an `ESTIMATE:` line on stderr gives the expected minutes:

```text
ESTIMATE: ~2.5 min for 1:03:30 of audio (whisper large-v3, batch 8)
ESTIMATE: ~5.0 min for 1:03:30 of audio (whisper small, on the CPU, slow)
```

- **GPU:** `0.2 + 0.036 × speed × slowdown × audio minutes`. `speed` is 1.0 for `large-v3` and
  `large-v2`, 0.8 `medium`, 0.68 `large-v3-turbo`, 0.64 `small`, 0.55 `base`, 0.5 `tiny`.
  `slowdown` applies to the large models and `medium` only: 1.08 at batch 4, 1.31 at batch 2, 1.77
  at batch 1. `--no-vad` replaces it with 1.6 (the sequential decoder).
- **CPU:** `0.2 + RTF × audio minutes`, with RTF 0.36 for `large-v3` and `large-v2`, 0.25 `medium`,
  0.19 `large-v3-turbo`, 0.075 `small`, 0.035 `base`, 0.02 `tiny`; times 1.2 with `--no-vad`.
- **Apple GPU:** `0.2 + RTF × audio minutes`, with RTF 0.095 for `large-v3` and `large-v2`, 0.055
  `medium`, 0.031 `large-v3-turbo`, 0.026 `small`, 0.02 `base`, 0.018 `tiny`; the same with
  `--no-vad` (no batches).
- The first use of a model adds up to 1.5 min for its download (`incl. first-time model download`).
- Calibrated on an RTX 5060 Ti, a 12-thread desktop CPU (6 threads) and an Apple M5 Pro (24 GB);
  `medium`, `base` and `tiny` are estimates. Slower hardware takes proportionally longer.

When a Whisper run used the CPU, stdout has
`NOTE: transcribed on the CPU (int8, device cpu): slower than a GPU, same model`. A run on the Apple
GPU has `NOTE: transcribed on the Apple GPU (mlx-whisper, float16, greedy decoding instead of beam
search)`, and its transcript header says `whisper (<model>, float16, metal, …, beam 1, sequential)`.

### No-VAD auto-retry

VAD (Silero, inside faster-whisper) drops singing over music, so a music video can come back empty.
When a default (VAD) Whisper run finds fewer than 5 words per minute of audio, `fetch` retries once
with VAD off on the same audio:

- Words are counted from Whisper's word timestamps, so scripts without spaces count correctly.
- The retry result is kept unless it found fewer words. If both are empty, exit 3
  (`transcript is empty`).
- stdout gets `NOTE: VAD found little speech (<n> words in <duration>); retried with VAD off`, and
  the transcript's source line says `VAD off (auto-retry)`.
- Without VAD, Whisper can hallucinate filler lines over music or silence, such as subtitle credits
  ("Subtitles by the Amara.org community"). VAD-off transcripts carry a header note saying so.
- Never retried: an explicit `--no-vad` run, manual subtitles and the auto-captions fallback.

Between "nearly empty" and "complete" there is a gray zone the script cannot judge (some speech plus
a long sung stretch). Every Whisper transcript therefore prints a `SPEECH:` hint: words per minute
and the longest stretch without transcribed speech. The agent decides from context whether a
`--no-vad` rerun is worth it.

## Output files

Work directory: `<work_dir>/<work key>/` (default `work_dir`: `<system temp>/video-watch`). The work
key is the 11-character video ID for YouTube; `<extractor>-<id>` for other sites (for example
`vimeo-76979871`, derived offline by yt-dlp's extractor matching; the extractor part lowercased, at
most 80 characters, a URL hash when the URL carries no id); and `file-<hash>` for local files.

| File | Content |
| --- | --- |
| `transcript.md` | A header (title, channel, URL or `- File:`, upload date, duration, language, source, Whisper settings, generated-at), a `## Description` section (the uploader's description verbatim) and a `## Chapters` list, then one `[mm:ss] text` line per segment (`[h:mm:ss]` from one hour). Empty sections are omitted. The file the agent reads. |
| `transcript-paragraphs.md` | The same header with the chapters but without the description, text grouped into paragraphs of roughly 15–30 seconds. Used when the user asks to save a transcript. |
| `segments.json` | Precise segment start and end times and text. Whisper segments carry a `words` list with word-level times. |
| `sounds.md` | The sound map for reading: header, `## Summary`, `## Timeline`, `## Loudness`. |
| `sounds.json` | The sound map for programs: every span (`label`, `start`, `end`, `score`), coverage, counts, loudness, model, threshold, version fingerprint; `duration` (analysed audio) and `media_duration` (the video's). |
| `meta.json` | Source, settings, timings, versions, `description`, `chapters`, and source fields (`source_kind`: `youtube`, `site`, `file`; `url`, `input_url`, `extractor`, `site_id`, `source_path`). `video_id` holds the work key. `sounds` records the sound map, or `{"skipped": reason}`. |
| `frames/` | Written by `frames` only. |

## Exit codes and errors

| Code | Meaning | Stdout |
| --- | --- | --- |
| 0 | Success. | `SOURCE:` … (or the command's own keys) |
| 1 | Other error, including usage errors, invalid settings, `setup` and `config` errors, and missing ffmpeg for `frames` or local files. | `ERROR:` + `HINT:` |
| 2 | A login wall on any site: YouTube's bot check, sign-in required, age-restricted, private or members-only videos, login-required posts. | `NEEDS_COOKIES`, then `ERROR:` + `HINT:` saying to ask the user, then retry with `--cookies-from-browser firefox`. |
| 3 | No transcript obtainable: a rejected input, an unavailable video, a video without audio, no usable CUDA (any source), or a busy GPU for a source other than YouTube. | `ERROR:`, with a `HINT:` for playlists (use `playlist`) and for the GPU (CPU mode, or retry later). |

`frames`, `sounds` and `playlist` use the same codes; `sounds` exits 3 when the sound map cannot be
built (`ERROR: sound map unavailable: …`). For `fetch`, a sound map that cannot be built is only a
`NOTE:`. `doctor` always exits 0. The GPU errors of `fetch`:

| Exit-3 `ERROR:` | Cause | `HINT:` |
| --- | --- | --- |
| `no usable CUDA GPU for Whisper (<reason>)` | No NVIDIA GPU, driver or CUDA libraries (device `auto` or `cuda`). | `no usable NVIDIA GPU (CUDA) was found. CPU mode works everywhere but is slow: rerun with --device cpu --model small, or make it the default with: setup --device cpu --model small --yes (downloads the small model, about 0.5 GB)` |
| `Whisper could not run on the GPU (<reason>)` | The GPU is busy or out of memory (not YouTube; YouTube uses its auto-captions), or the Apple GPU engine cannot start at all (`device` `metal` or `auto` on a Mac: mlx-whisper missing or no Metal; never the auto-captions, also on YouTube). | Retry later (close games or other GPU apps) or choose a smaller model; the auto-captions fallback exists only for YouTube. For an Apple GPU engine that cannot start: `the Apple GPU engine (mlx-whisper) could not start. Rerun the command (uv reinstalls the packages of video_watch.py), or use CPU mode: --device cpu --model small, or … setup --device cpu --model small --yes`. |
| `Whisper could not run on the GPU (<reason>) and no original-language auto-captions exist` | YouTube, GPU busy, and no `-orig` track. | As above. |

Exit-1 causes are told apart by their `ERROR:` prefix, so a yt-dlp upgrade is suggested only for
real extraction failures:

| Exit-1 `ERROR:` starts with | Cause | `HINT:` |
| --- | --- | --- |
| `invalid frames request:` (or an argparse usage message) | Bad or conflicting `frames` arguments, or too many frames for `--max-frames`. | Fix the arguments. |
| `invalid sounds request:` | Bad `sounds` arguments (empty `--query`, more than 20 prompts, `--threshold` without `--query`). | Fix the arguments. |
| `ffmpeg was not found on PATH` | ffmpeg or ffprobe is missing. | Install ffmpeg; `fetch` of a URL works without it. |
| `ffmpeg was found but cannot be run by this process (<path>: <reason>)` | ffmpeg or ffprobe (the path names which) is on `PATH` but this process may not run it, typically an agent sandbox running as another user. | `the agent's sandbox may block access; allow full access for video-watch commands (for Codex: /permissions → Full access, or codex exec -s danger-full-access)` |
| `ffmpeg …` | ffmpeg failed to decode, timed out, or could not build a sheet or read a local file. | Rerun once, otherwise report it; not a yt-dlp problem. |
| `another run is using this video` | A `fetch`, `frames` or `sounds` run for the same video is still active. | Retry when it finishes; never run them in parallel. |
| `no frame could be extracted: the video stream failed` | The site refused every stream format. | Rerun once, then upgrade yt-dlp. |
| `invalid config request:`, `config file …`, `VIDEO_WATCH_…:` | A bad `config` command, an invalid config file or environment value. Every command stops on an invalid setting before doing any work. | `fix the value with: uv run video_watch.py config set KEY VALUE (or edit or delete the config file)` or the valid keys. |
| `work dir … already holds other files and is not a video-watch work dir` | `work_dir` points to a folder with other content (see [Cache and lifecycle](#cache-and-lifecycle)). | Choose a new or empty folder, or `config unset work_dir`. |
| `invalid setup request:`, `setup needs to download about X GB`, `CUDA is unavailable for device …`, `the sound map needs an NVIDIA GPU, and none was found`, `the sound map needs an NVIDIA GPU with CUDA 12: NVIDIA driver <ver> is too old for CUDA 12 (needs 525 or newer)` | See [`setup`](#setup). | As printed. |
| Anything else | yt-dlp extraction failed. | `uv run --upgrade-package yt-dlp video_watch.py fetch <url>` |

## Cache and lifecycle

- **Cache:** a repeated `fetch` reuses the existing transcript (`CACHED: yes`) when it satisfies the
  request: a plain request accepts a manual-subtitles or Whisper transcript; `--whisper` only a
  Whisper one; a different explicit `--model`, `--no-vad` or `--lang` reruns; a cached
  auto-captions fallback is always retried. A cache hit needs no network request.
- **Purge:** every work command (`fetch`, `frames`, `sounds`, `playlist`, `clean`) first deletes
  work directories older than 24 hours. `doctor`, `setup` and `config` never touch the work
  directory.
- **Work root marker:** the work root holds a `.video-watch-work-root` marker file. A configured
  `work_dir` is adopted only when it is new, empty or already marked; a non-empty folder without the
  marker is refused with exit 1 (`config set work_dir` checks this too), and the purge and
  `clean --all` do nothing in an unmarked folder. So a mistyped `work_dir` (a home or project
  folder) can never be emptied.
- **Lock:** a run that writes to a work directory creates `.lock` exclusively. If another live
  process holds it, the run stops with exit 1 and touches nothing. A stale lock is taken over.
- **Audio** is deleted right after transcription (after the sound map when one is built), also on
  failure. The overview's low-resolution video is deleted right after use.
- **`clean`** accepts a YouTube ID or URL, any other URL, a local path or `file:` URI, a work key,
  or a path inside a work directory, all resolved without network. It also finds work directories
  whose stored URL or source path matches (earlier versions of an edited file, a deleted file). It
  only removes direct children of the work root, skips links and junctions, never touches a source
  file, and prints `REMOVED: <n>`.

## Sound map

When enabled (`sounds: true` in the config, `VIDEO_WATCH_SOUNDS`, or `fetch --sounds`), `fetch`
builds a sound map after the transcript: a local sound-event model
([PE-A-Frame base](https://huggingface.co/facebook/pe-a-frame-base), Apache-2.0) scores 45 curated
sound descriptions against every 40 ms of audio, and ffmpeg measures loudness.

```text
SOUNDS: music 90%, speech 18%, singing 5%; laughter 9, applause 7, cheering 6, crowd 4, wind 2, beeping 1 (+3 more labels)
SOUNDS_FILE: <work dir>/RCzK4z2AIrU/sounds.md
```

- The summary gives coverage (% of the audio) for music, speech and singing, then the most frequent
  discrete events with counts (at most 6 named).
- **Off by default.** With the map off (and no `--sounds`), `fetch` prints no `SOUNDS:` line and
  no sound `NOTE:`, unless an earlier `sounds` run left a current map in the work directory.
- **The transcript is primary.** When the map cannot be built (not set up, no NVIDIA GPU, GPU busy
  or out of memory, no audio, ffmpeg or uv missing, any unexpected error), `fetch` prints
  `NOTE: sound map skipped: <reason>` and keeps its output and exit code exactly as without a map.
- **Checks first, nothing downloaded:** before any audio is fetched for the map, ffmpeg and uv must
  run, `nvidia-smi` must exist, `CUDA_VISIBLE_DEVICES` must not hide the GPU (empty, `-1` or not an
  index or GPU id), the sound map must be set up (the sound model cached and the
  `sound-helper.json` marker of `setup --sounds on` present, see [`doctor`](#doctor); a `fetch` or
  `sounds` run never starts the helper's 3 GB environment install or the model download), and at
  least 4600 MB of VRAM must be free; otherwise the map is skipped at once. Cached fetches wait 15
  minutes before retrying after a busy GPU; `sounds <input>` tries at once.
- A blocked ffmpeg during a fetch's map gives `NOTE: sound map skipped: ffmpeg was found but cannot
  be run by this process (…)`; the transcript is unaffected.
- A cached transcript without a current map gets one on the next `fetch` (audio fetched again,
  Whisper never rerun) unless `--no-sounds` is given.

### `sounds.md`

```text
# Sound map: <title>

- Model: facebook/pe-a-frame-base (logit threshold 4; …)
- Duration: 03:58
- Labels name sounds the model hears (acoustic cues such as laughter or crying), not emotions …

## Summary

- Coverage: music 90%, speech 18%, singing 5%
- Events: laughter 9, applause 7, cheering 6, crowd 4, …

## Timeline

[00:01-00:09] crowd
[00:02-00:09] applause
[00:20-02:11] music
[01:28-01:32] laughter

## Loudness

- Integrated loudness: -6.9 LUFS; loudness range 7 LU
- Loudest moments (short-term loudness, 3 s): [00:37] -6 LUFS, [01:05] -5.3 LUFS, …
- Silences (below -50 dB for 3 s or more): none
```

- **Timeline:** one line per span, in the transcript's stamp form; labels with the same span share a
  line; speech is left out (the transcript covers it). Above 400 spans, the lowest-scoring ones are
  left out of `sounds.md`; `sounds.json` keeps all of them.
- **Labels:** music, speech, singing; laughter, applause, cheering, crowd, shouting, screaming,
  crying, whispering, sighing, coughing, sneezing, booing; explosion, gunshot, glass breaking,
  crash, punch; car engine, car horn, siren, motorcycle, airplane, helicopter, train; dog barking,
  cat meowing, birds chirping; door, knocking, footsteps, keyboard typing, phone ringing, alarm,
  bell; rain, thunder, wind, waves, fire crackling; video game sound effects, beeping, whoosh.
- **Labels are acoustic cues.** "laughter" or "crying" means that sound is heard, not how anyone
  feels. In games and edited videos a label can name a sound effect rather than a real event.

What to expect, from calibration on synthetic mixtures (see [DESIGN.md](DESIGN.md#sound-map)):
most clear effects over speech are found (72–88% depending on loudness) within about 0.05 s of
their start, at well under one false alarm per minute; over loud music, fewer (35–75%). Short or
quiet sounds can be missed, sounds lasting longer than about 45 s under music are mostly not
reported, and related labels can co-occur. Treat a single label as evidence, not proof.

### `sounds` command

- Without `--query`: prints the cached map's summary, or builds the map when it is missing or
  outdated. No transcript is needed.

  ```text
  VIDEO_ID: <work key>
  TITLE: …
  DURATION: mm:ss
  SOUNDS: …
  SOUNDS_FILE: <absolute path>
  CACHED: yes | no
  ```

- `--query`: open-text sound descriptions (comma-separated, at most 20). The audio is fetched again,
  scored and deleted; the map is not changed. Descriptive phrases work best ("a man shouting",
  "machine gun fire").

  ```text
  VIDEO_ID: UTI1KfmKIoY
  TITLE: …
  DURATION: 04:41
  QUERY: machine gun fire, a rocket launcher firing, a dog barking
  EVENT: [00:05.7-00:07] machine gun fire 5.5
  EVENT: [00:42.9-00:45.7] a rocket launcher firing 4.4
  NOTE: 'a dog barking' not found (threshold 4, plus the margin under music); strongest at [01:29.4] with score 1.0
  ```

  `EVENT:` lines are sorted by time and end with the peak logit (4 and above counts). `--threshold`
  (a logit from -10 to 15, default 4) moves the bar for the query.

- When the map cannot be built, `sounds` exits 3 with `ERROR: sound map unavailable: <reason>`. Not
  set up yet: `ERROR: sound map unavailable: the sound map is not set up yet (setup --sounds on
  downloads about 5 GB once)` with `HINT: enable the sound map once with: uv run "<path>"
  setup --sounds on --yes (NVIDIA GPU; about 5 GB download: PyTorch and the model); ask the user
  first`. A hidden GPU gives `ERROR: sound map unavailable: no usable GPU: CUDA_VISIBLE_DEVICES=…
  hides the GPU from CUDA` with a `HINT:` to unset it; a busy GPU gives a retry `HINT:` (see
  [Hints](#hints)).
- A missing or blocked ffmpeg is checked before anything else and exits 1 like `frames` (see
  [Exit codes and errors](#exit-codes-and-errors)).

### Model and performance

- The model runs in `sound_events.py`, a separate uv script with its own PyTorch environment,
  started as a subprocess after Whisper has released the GPU. The two exchange JSON files in the
  work directory; all GPU memory is freed when the subprocess ends. `--no-sounds` never imports
  PyTorch.
- NVIDIA GPU only in this release, fp16, about 4.4 GB of VRAM at peak (4600 MB free are required
  before it starts). Measured on an RTX 5060 Ti:
  34 s of model time per hour of audio; a 1-hour video's whole map took about 55 s, a 4-minute video
  about 11 s.
- `setup --sounds on` prepares everything once: the helper's environment (PyTorch for CUDA 12.8 and
  `transformers`, about 3.0 GB in uv's cache) and the model (2.3 GB) in the Hugging Face cache, or
  in `sound_cache` / `VIDEO_WATCH_SOUND_CACHE` when set. After that it loads without network access.
- Swapping the model: `sound_events.py` reads a request (audio, prompts, model id, chunks) and
  writes per-frame logits; another model only needs another script with the same contract. Changing
  the model, prompts or thresholds changes the map's version fingerprint, so cached maps are
  rebuilt.

## Frames

`frames` extracts JPEG stills with ffmpeg on the CPU; it never loads Whisper or the sound model.
All sources work; a local file is read in place.

### Targeted frames (`--at`)

- Moments are clamped into the video, rounded to 0.1 s, deduplicated and sorted.
- **No full download:** yt-dlp resolves the stream URL of one plain-HTTPS format (short side at
  most 1080 px, preferring the largest size, then the smallest file), and ffmpeg seeks with `-ss`
  before `-i`, decoding from the seek point, so frames are time-exact. Each frame downloads only a
  few seconds of video.
- **Fallback:** if no such format exists or a seek fails, ffmpeg seeks the same way in other formats
  (the best HLS format with the same codec first). A `NOTE:` names the fallback formats.
- Encrypted HLS (DRM) is detected and exits 3.
- **Output:** JPEG, long side fitted to 1280 px, never upscaled. File names encode the moment
  (`frames/frame-0h01m23.5s.jpg`), so repeats are reused and marked `(cached)`.

### Overview (`--overview`)

1. Download one low-resolution video-only format (short side 360 px, else the smallest above 360 up
   to 480, else the closest); deleted right after use.
2. Pick the start plus ffmpeg scene changes, skip black stretches, keep a minimum gap (a quarter of
   the interval, 2 to 15 s), and fill long stretches without cuts. Every moment is a whole second,
   so a tile's stamp passed to `--at` returns the same frame in full detail.
3. Above `--max-frames`, thin evenly with a `NOTE:`.
4. Build 3x3 contact sheets of 640 px tiles, each labeled with its stamp (`frames/overview-01.jpg`,
   …). The label font is discovered per OS, falling back to ffmpeg's default; if labeling fails,
   labels are dropped with a `NOTE:` and the `TIMES:` lines still map the tiles.

### Range (`--range`)

For fast action where one frame per scene is not enough. From START to END every 1/fps seconds
(default 2 fps, at most 10), in one ffmpeg pass decoded from the seek point, onto the overview's
labeled sheets (`frames/range-01.jpg`, …). More frames than `--max-frames` is an error, checked
before any network access. An END past the video is clamped with a `NOTE:`.

### Frames stdout

```text
VIDEO_ID: <work key>
TITLE: …
DURATION: mm:ss | h:mm:ss
MODE: at | overview | range
FRAME: [mm:ss] | [mm:ss.s] <absolute path> [(cached)]   (at; one per frame, in time order)
SAID: [mm:ss] | [mm:ss.s] <transcript text>      (follows its FRAME when there is speech)
HEARD: [mm:ss] | [mm:ss.s] label, label          (follows the SAID position when sounds are heard)
PICKS: start + N scene changes (threshold X) + M interval fills (no gap over I s), min gap G s   (overview)
SHEET: <absolute path>                           (overview and range; one per sheet)
TIMES: [mm:ss] [mm:ss] …                         (follows its SHEET, tiles in reading order)
SAID: [mm:ss] <transcript text>                  (after its TIMES, per tile with speech, each followed by its HEARD)
COUNT: <frames, or tiles over all sheets>
TOKENS_ESTIMATE: ~<image tokens for all listed files>
NOTE: … (optional, may repeat)
```

- **SAID:** the transcript segment containing the moment, otherwise the nearest one within 2 s, on
  one line, cut at 300 characters. Only from a transcript already cached in the work directory;
  without one, a single `NOTE:` says to run `fetch` first. `frames` never fetches a transcript.
- **HEARD:** sound-map spans containing the moment or within 1 s, events first, then music and
  singing, at most 6 labels. Only from a cached sound map; without one there are no `HEARD:` lines.
- **Stamps** use the transcript format, adding tenths when a moment has them (`[00:05.5]`).
- **Token estimate:** each JPEG fitted to a 1568 px long side, then width × height / 750 (a
  1280x720 frame is about 1.2k tokens, a full 3x3 sheet about 1.85k).

## Playlists

`fetch` and `frames` take single videos only. `playlist` lists entries without downloading, so the
agent can pick the videos a question needs:

```text
PLAYLIST: <title>
CHANNEL: <uploader, empty when unknown>
COUNT: <entries listed>
TOTAL: <entries in the playlist | unknown>
DURATION_TOTAL: <sum of the known durations | unknown>
ESTIMATE: ~<n> min if every listed video needed Whisper
ENTRY: <index> [<duration | ?>] <url> [<marker>] <title>    (one per entry, in playlist order)
NOTE: … (optional, may repeat)
```

- **Inputs:** YouTube playlists (a `watch?v=...&list=...` URL lists the list), YouTube channels (the
  videos tab, newest first), and other sites' playlist, album, channel or user pages.
- **`--limit N`** (default 200): one more entry is requested to tell whether the list is longer; a
  `NOTE:` reports truncation.
- **Markers:** `(unavailable: private)`, `(unavailable: deleted)`, `(unavailable: needs login)`,
  `(unavailable: members only)`, `(unavailable: premium only)`, `(live now)`, `(upcoming)`,
  `(processing)`. Such entries are left out of `ESTIMATE`.
- **`ESTIMATE`** sums the per-video Whisper estimate (see [stdout](#stdout)) for the configured
  model and device, at batch 8 on the GPU.

## Environment: `doctor`, `setup`, `config`

### `doctor`

A read-only report: it never downloads or writes anything and always exits 0. One `KEY: value` per
line:

```text
OS: <system release (machine)>
PYTHON: <version>
UV: <version | path | missing>
FFMPEG: <version | blocked: <path> (<reason>) | missing>
FFPROBE: <path | blocked: <path> (<reason>) | missing>
GPU: <name (driver X) | name (Apple GPU, unified memory) | none>
VRAM_TOTAL_MB: <MiB (unified memory on a Mac) | unknown>
VRAM_FREE_MB: <MiB | unknown>
CUDA: ok (<details>) | unavailable: <reason>
METAL: ok (<details>) | unavailable: <reason>
DISK_FREE_GB: <free GB on the model directory's drive | unknown>
CONFIG: <path> [(not created yet)]
DEVICE: auto | cuda | metal | cpu
ENGINE: faster-whisper (CUDA, float16) | mlx-whisper (Apple GPU, float16) | faster-whisper (CPU, int8)
MODEL: <name>
MODEL_DIR: <path>
MODEL_CACHED: yes | no
SOUNDS: on | off
SOUND_MODEL_CACHED: yes | no
SOUND_HELPER_READY: yes | no
WORK_DIR: <path>
RECOMMEND_MODEL: <name>
RECOMMEND_DEVICE: cuda | metal | cpu
RECOMMEND_SOUNDS: yes | no (<reason>)
WARN: … (optional, may repeat)
READY: yes | no
```

- **`CUDA:`** In the light form, CUDA is judged from `nvidia-smi` and the driver version (525 or
  newer for CUDA 12): `ok (driver X; CTranslate2 is checked on the first run)`. When the script's
  packages are installed, CTranslate2's own device check is added. macOS is always
  `unavailable: macOS has no CUDA (Whisper runs on the Apple GPU instead)` on Apple Silicon and
  `… (CPU mode only)` on an Intel Mac.
- **`METAL:`** `ok (mlx-whisper is checked on the first run)` on an Apple Silicon Mac in the light
  form; with the script's packages installed, MLX's own check is added (`ok (mlx-whisper 0.4.3,
  Metal)` or `unavailable: MLX sees no Metal GPU`). Every other machine says
  `unavailable: needs macOS on Apple Silicon (an arm64 Mac)`. On a Mac, `GPU:` names the chip and
  `VRAM_TOTAL_MB:` is its unified memory (the GPU shares it with the system); `VRAM_FREE_MB:` is
  `unknown`.
- **`ENGINE:`** What `DEVICE` resolves to on this machine: `auto` is the Apple GPU (mlx-whisper) on
  Apple Silicon and CUDA (faster-whisper) elsewhere. `MODEL_CACHED:` looks for the weights of that
  engine: the MLX conversion for the Apple GPU, the CTranslate2 conversion for CUDA and CPU mode.
- **`SOUND_HELPER_READY:`** `yes` when `setup --sounds on` has prepared the sound map's environment
  (PyTorch with CUDA 12.8, `transformers`) for the current `sound_events.py`. It is recorded in a
  marker file, `<user cache>/video-watch/sound-helper.json` (`%LOCALAPPDATA%\video-watch`,
  `$XDG_CACHE_HOME/video-watch` or `~/.cache/video-watch`, `~/Library/Caches/video-watch`), holding
  a fingerprint of the helper's dependency header and model; a changed header needs a new setup.
  Machines that set up the sound map before the marker existed show `no` until
  `setup --sounds on --yes` runs once.
- **`READY: yes`** means transcription can run now: the device is usable (`cpu`, CUDA ok or Metal ok), the
  configured model is downloaded, ffmpeg and ffprobe are found, the configured work directory would
  be accepted, and, when the sound map is on, the sound model is downloaded, `SOUND_HELPER_READY` is
  `yes` and an NVIDIA GPU is present.
- **Recommendations** use the GPU's total VRAM (MiB as `nvidia-smi` reports it): 7800 or more →
  `large-v3`; 3900 or more → `large-v3-turbo`; below that, or without usable CUDA → `small` with
  `RECOMMEND_DEVICE: cpu`. On an Apple Silicon Mac with a working Metal engine the unified memory
  decides, with `RECOMMEND_DEVICE: metal`: 16 GB or more → `large-v3`, 8 GB or more →
  `large-v3-turbo`, otherwise `small`. The sound map is recommended from 6000 MiB with an NVIDIA
  GPU (never on a Mac yet). These are recommendations only; any model can be chosen.
- **Fit warnings** (also printed by `setup`): when even batch 1 of the model plus a 400 MB margin
  exceeds the total VRAM, `WARN: <model> needs about X GB of VRAM even at batch 1; … it will likely
  run out of memory …`; when only batch 8 does not fit, `WARN: memory may be tight for <model> on X
  GB of VRAM: it runs with a smaller batch (slower) and fails while other apps hold VRAM`. On a
  Mac the warning is `WARN: <model> needs about X GB of memory on the Apple GPU; this Mac has Y GB
  shared with the system, so it may run out of memory …` when the model's peak plus 3 GB exceeds
  the unified memory. Other
  `WARN:` lines cover a missing uv or ffmpeg, a blocked ffmpeg or ffprobe (`WARN: ffmpeg was found
  but cannot be run by this process; <sandbox hint>`), a device that cannot work, a model not
  downloaded yet,
  a sound map that is on but not set up (`setup --sounds on`), little free VRAM right now, less than
  10 GB of free disk space, an invalid setting (the default is shown), and a configured work
  directory that would be refused (`WARN: work dir … already holds other files and is not a
  video-watch work dir; fetch, frames and sounds would stop with an error (…)`, with
  `READY: no`).

### `setup`

Non-interactive: applies the user's choices and pre-downloads what they need. Choices not given are
taken from the current settings.

```bash
uv run --no-project --python ">=3.11" python video_watch.py setup --device cpu --model small --sounds off --dry-run
uv run video_watch.py setup --device auto --model large-v3 --sounds off --yes
uv run video_watch.py setup --device cuda --model large-v3-turbo --sounds on --model-dir /data/whisper --yes
```

stdout:

```text
DEVICE: auto | cuda | metal | cpu
MODEL: <name>
SOUNDS: on | off
MODEL_DIR: <path>
DOWNLOAD: <what>, about X.X GB              (one per download still needed)
DOWNLOAD_TOTAL_GB: X.X
WARN: …                                     (optional: fit warning, CPU mode is slow, low disk space)
CONFIG: <path> (would write {...})          (--dry-run only)
SETUP: dry-run                              (--dry-run only, exit 0)
MODEL_CACHED: yes
SOUND_MODEL_CACHED: yes                     (with --sounds on)
NOTE: the sound map is off; its model stays in the Hugging Face cache … (shared with other tools) and is used again if the sound map is turned back on   (--sounds off with a cached sound model)
CONFIG: <path>
SETUP: done                                 (exit 0)
```

| Download | Size |
| --- | --- |
| The script's Python packages, once, into uv's cache (yt-dlp, faster-whisper and, except on macOS, the CUDA libraries; on Apple Silicon also MLX and mlx-whisper) | about 1.4 GB on Windows, 1.5 GB on Linux, 0.7 GB on Apple Silicon, 0.2 GB on an Intel Mac |
| Whisper model into `model_dir` (the CTranslate2 conversion for CUDA and CPU mode, the MLX conversion for the Apple GPU: the same sizes) | `large-v3` 3.1 GB, `large-v3-turbo` 1.6 GB, `large-v2` 3.1 GB, `medium` 1.5 GB, `small` 0.5 GB, `base` 0.15 GB, `tiny` 0.08 GB |
| Sound map environment (PyTorch for CUDA 12.8, `transformers`) into uv's cache, with `--sounds on` while `SOUND_HELPER_READY` is `no` (listed as `… into the uv cache (less if uv already has it)`) | about 3.0 GB |
| Sound model `facebook/pe-a-frame-base` into the sound cache, with `--sounds on` | about 2.3 GB |

- `--dry-run` prints the plan and changes nothing; it runs in the light form.
- Downloads need `--yes`, which states that the user agreed to the printed sizes. Without it, a
  setup that needs downloads ends with `ERROR: setup needs to download about X GB` and exit 1.
- In the light form a real setup reruns itself with the script's packages (`uv run --script`).
- The config file is written only after every download succeeded (`device`, `model`, `sounds`, and
  `model_dir` when given). A successful sound-map preparation also writes the
  `sound-helper.json` marker (see [`doctor`](#doctor)).
- Exit 1 with `ERROR:` for: an invalid choice or config file; `--device auto` or `cuda` without
  usable CUDA on a machine that is not Apple Silicon (`CUDA is unavailable for device auto:
  <reason>`, with a `HINT:` to use `--device cpu --model small --sounds off`, on an Apple Silicon
  Mac also `--device metal`); `--device metal` where the Apple GPU engine cannot run (`the Apple GPU
  engine is unavailable for device metal: <reason>`); `--sounds on` without an NVIDIA GPU (`the sound map
  needs an NVIDIA GPU, and none was found`) or with an NVIDIA driver older than 525 (`the sound map
  needs an NVIDIA GPU with CUDA 12: …`, HINT: update the driver or rerun with `--sounds off`); a
  failed download.
- `--device cpu` with `--sounds on` is accepted when an NVIDIA GPU is present: Whisper runs on the
  CPU and the sound map on the GPU.

### `config`

```bash
uv run video_watch.py config show          # CONFIG: <path> [(not created yet)], then KEY: value (flag|env|config|default)
uv run video_watch.py config path          # CONFIG: <path>
uv run video_watch.py config get model     # the bare value, e.g. large-v3
uv run video_watch.py config set model large-v3-turbo   # SET: model = large-v3-turbo, then CONFIG: <path>
uv run video_watch.py config unset model   # UNSET: model, then CONFIG: <path>
```

- Keys: `device`, `model`, `model_dir`, `sounds`, `sound_cache`, `work_dir`. See the
  [README](../README.md#configuration) for defaults, file locations and environment variables.
- `sounds` accepts `on`, `off`, `true`, `false`, `yes`, `no`, `1` and `0`; `config` prints it as
  `true`/`false`, `doctor` and `setup` as `on`/`off`. Paths are stored absolute.
- `config set work_dir` refuses a non-empty folder without the work-root marker.
- `config set KEY VALUE` adds `WARN: VIDEO_WATCH_<KEY>=<value> is set and overrides this value;
  unset it to use the config file` when that environment variable is set.
- Errors (an unknown key, an invalid value, an unreadable config file) print `ERROR:` and `HINT:`
  and exit 1. Unknown keys in the file (for example from a newer version) are kept and ignored.

### Hints

`HINT:` lines are meant to be copied: commands in them use the script's absolute path, in the
normal form (`uv run "<path>/video_watch.py" …`) or, for `config`, the light form. The ones that
change setup:

| Situation | `HINT:` |
| --- | --- |
| Invalid config value | `fix the value with: uv run --no-project --python ">=3.11" python "<path>" config set KEY VALUE (or edit or delete the config file)` |
| Invalid `VIDEO_WATCH_*` variable (work commands) | `fix or unset the environment variable VIDEO_WATCH_<KEY>; it overrides the config file` |
| Refused `work_dir` | `set work_dir to a new or empty folder (… config set work_dir PATH) or remove the setting (… config unset work_dir) to use the default in the system temp folder; if VIDEO_WATCH_WORK_DIR is set, change or unset it instead` |
| No usable CUDA (`fetch`) | `no usable NVIDIA GPU (CUDA) was found. CPU mode works everywhere but is slow: rerun with --device cpu --model small, or make it the default with: uv run "<path>" setup --device cpu --model small --yes (downloads the small model, about 0.5 GB)` |
| Sound map not set up (`sounds`) | `enable the sound map once with: uv run "<path>" setup --sounds on --yes (NVIDIA GPU; about 5 GB download: PyTorch and the model); ask the user first` |
| `CUDA_VISIBLE_DEVICES` hides the GPU (`sounds`) | `the sound map needs a visible NVIDIA GPU: unset CUDA_VISIBLE_DEVICES or set it to a GPU index such as 0` |
| Busy GPU (`sounds`) | `the sound model needs about 4.5 GB of free VRAM; retry later (close games or other GPU apps)` |
| Bad `frames` or `sounds` arguments | `fix the … arguments (see https://github.com/AAyberkU/video-watch/blob/main/docs/REFERENCE.md); not a yt-dlp problem` |

## Devices: GPU and CPU

- **`auto`** means the GPU of the machine: the Apple GPU on an Apple Silicon Mac (below), the
  NVIDIA GPU everywhere else. **`cuda`** is always the NVIDIA GPU (`float16`). On Linux the CLI
  re-executes itself once with the CUDA wheels' library folders on `LD_LIBRARY_PATH` (the loader
  reads it only at start); on Windows it adds their DLL folders.
- **`metal`** (also what `auto` means on Apple Silicon): Whisper on the Apple GPU through
  mlx-whisper with `float16`. The same fetch contract as on CUDA (`SOURCE: whisper`, `SPEECH:`,
  word timestamps in `segments.json`, VAD with the automatic no-VAD retry, the language from the
  metadata, `--lang` or auto-detection with its probability in a `NOTE:`). Differences: greedy
  decoding with temperature fallback instead of beam search (mlx-whisper has none, header `beam 1`),
  no batching, and VAD is Silero (faster-whisper's own) cutting the silence out before the Apple
  GPU decodes the speech chunks joined end to end (times are mapped back). The model weights are
  MLX conversions (`mlx-community/whisper-*`) stored under `model_dir` next to the CTranslate2 ones,
  so switching between `metal` and `cpu` downloads each once. Memory is the Mac's unified memory
  (peak resident memory 1.2 GB for `small`, 2.1 GB for `large-v3-turbo`, 3.9 GB for `large-v3`).
  Failures raise the same `GpuUnavailable` handling as CUDA: an out-of-memory or busy GPU falls
  back on YouTube to its auto-captions or exits 3 (`ERROR: Whisper could not run on the GPU
  (<reason>)`); an engine that cannot start at all (mlx-whisper missing, no Metal, `device: metal`
  on a machine that is not Apple Silicon) exits 3 with that error and the Apple GPU `HINT:`, also
  on YouTube, and never switches to the CPU. The sound map still needs an NVIDIA GPU.
- **No usable CUDA:** `fetch` exits 3 with
  `ERROR: no usable CUDA GPU for Whisper (<reason>)` and the CPU-mode `HINT:` shown under
  [Exit codes and errors](#exit-codes-and-errors). There is no fallback, also not to YouTube's
  auto-captions.
- **GPU present but busy or out of memory:** YouTube falls back to its original-language
  auto-captions (`SOURCE: auto-subtitles-fallback`); other sources exit 3 with
  `ERROR: Whisper could not run on the GPU (<reason>)` and a retry `HINT:`.
- **Batch size** (batched GPU runs): the free VRAM is read from `nvidia-smi` before the model loads,
  and the largest batch of 8, 4, 2 and 1 whose measured peak plus 400 MB fits is used; without
  `nvidia-smi`, batch 8. For `large-v3`: 7556 MB free or more → 8; 5679 → 4; 4909 → 2; otherwise 1.
  The batch changes speed, not the transcript; the transcript header names it (`batched x4`).
- **`cpu`** (only when chosen): `int8`, batch 4, threads = half the logical CPUs on x86 (1 to 16),
  every core on Apple Silicon (no simultaneous multithreading, at most 16). The
  transcript header says `whisper (<model>, int8, cpu, …)` and stdout has
  `NOTE: transcribed on the CPU (int8, device cpu): slower than a GPU, same model`.
- **Model downloads:** `fetch --model NAME` accepts all seven models; a model that `setup` did not
  download is downloaded on first use (the `ESTIMATE:` says so).

## Cookies

- The first attempt never uses cookies.
- A login wall on any site gives exit 2 with `NEEDS_COOKIES`. The agent asks the user and, with
  approval, reruns with `--cookies-from-browser firefox`. Cookies tie requests to the user's account
  on that site, so they are never used silently.
- Firefox only by policy (Chrome cookies on Windows use app-bound encryption). No `cookies.txt`
  export.
- For frame seeks, cookies are passed only to that run's ffmpeg through its domain-scoped
  `-cookies` option (like any command-line argument, visible in the process list while it runs).
- If the run with cookies still fails, there is nothing more to try.

## Whisper settings

| Setting | Value |
| --- | --- |
| Runtime | faster-whisper (CTranslate2), `BatchedInferencePipeline`; on the Apple GPU mlx-whisper (see [Devices](#devices-gpu-and-cpu)) |
| Model | `large-v3` by default; configurable |
| Device and precision | CUDA with `float16`; the Apple GPU with `float16`; CPU (`int8`) only when chosen explicitly |
| Batch size | 8, 4, 2 or 1 by free VRAM on the GPU (see [Devices](#devices-gpu-and-cpu)); 4 on the CPU |
| Language | From the video's metadata or `--lang`, otherwise auto-detect |
| VAD | On, `min_silence_duration_ms=500`; auto-retry without VAD under 5 words per minute |
| `condition_on_previous_text` | Off |
| Beam size | 5 (1, greedy, on the Apple GPU) |
| Word timestamps | On (stored in `segments.json`) |

The model loads before the audio download, so GPU problems surface early.

## Dependencies

Both scripts declare their dependencies in a PEP 723 header; `uv run` resolves them into isolated
environments. Notable pins:

- `video_watch.py`: `yt-dlp[default,curl-cffi]` (browser impersonation, which Vimeo requires),
  `faster-whisper>=1.2.1,<2`, `ctranslate2>=4.5,<5`, `nvidia-cublas-cu12` and
  `nvidia-cudnn-cu12>=9,<10` (except on macOS), `mlx-whisper>=0.4.3,<0.5` (only on Apple Silicon:
  `sys_platform == 'darwin' and platform_machine == 'arm64'`), and `av<16` (PyAV 16+ breaks
  faster-whisper 1.2.1 with a `metadata_errors` error). mlx-whisper declares PyTorch as a
  dependency, but only its model-conversion tool imports it, so the header overrides it away
  (`[tool.uv] override-dependencies`, about 0.5 GB saved). yt-dlp's JavaScript runtimes are set to
  deno and node.
- Known limitation: the CUDA wheels (about 1.2 GB) are installed on Windows and Linux even for CPU
  mode, because the inline dependencies cannot depend on the chosen device; splitting them out is
  on the [roadmap](../ROADMAP.md).
- `sound_events.py`: `torch` from the PyTorch CUDA 12.8 index (needed for RTX 50-series / sm_120)
  and an exact `transformers` version with `PeAudioFrameLevelModel`. No `torch.compile`.

## Troubleshooting

- **Extraction fails (exit 1, a yt-dlp error):** sites change often; update yt-dlp inside the script
  environment with `uv run --upgrade-package yt-dlp video_watch.py fetch <url>`.
- **`CUDA: unavailable` in `doctor`, or `ERROR: no usable CUDA GPU for Whisper`:** check the
  NVIDIA driver (`nvidia-smi`; version 525 or newer). Without a usable NVIDIA GPU, choose CPU mode
  explicitly with a small model (`setup --device cpu --model small`). On an Apple Silicon Mac,
  `METAL:` in `doctor` should say `ok`; `device: auto` then runs Whisper on the Apple GPU.
- **A Mac stays on the CPU or says `no usable CUDA`:** `device` may still be `cpu` from an earlier
  setup (`setup --device auto --yes` switches to the Apple GPU and downloads the MLX weights), and
  an Intel Mac, or Python running under Rosetta, has no Apple GPU engine.
- **`CUBLAS_STATUS_NOT_SUPPORTED` on RTX 50-series:** the compute type must be `float16`, which the
  script uses on CUDA.
- **`NEEDS_COOKIES` (exit 2):** the site wants a login. Ask the user before rerunning with
  `--cookies-from-browser firefox`.
- **Vimeo:** `vimeo.com/<id>` pages may ask for a login; the player URL
  `https://player.vimeo.com/video/<id>` often works. Vimeo's streams are DRM-encrypted, so only its
  subtitles can be read; Whisper and frames exit 3.
- **`the video has no audio track` (exit 3):** the site states the video is silent. Frames still
  work.
- **`SOURCE: auto-subtitles-fallback`:** the GPU was busy or out of memory. Rerun when it has room;
  a cached fallback is retried automatically.
- **`metadata_errors` error:** PyAV is too new. Keep the `av<16` pin.
- **Repetition loops in the transcript:** rerun with `--model large-v2`.
- **Speech missing in music-heavy videos:** rerun with `--no-vad` (an almost empty VAD run is retried
  automatically).
- **`NOTE: sound map skipped: …`:** the transcript is fine; only the sound map is missing. "the GPU
  is busy" means less than 4600 MB of free VRAM; "not set up yet" means `setup --sounds on --yes`
  has not run (about 5 GB once), or has not run since the `sound-helper.json` marker was introduced
  (`SOUND_HELPER_READY: no` in `doctor`; run it once).
- **`ffmpeg was found but cannot be run by this process`:** an agent sandbox blocks it (for example
  Codex on Windows, whose sandbox user cannot read per-user installs under `%LOCALAPPDATA%`). Run
  video-watch with full access.

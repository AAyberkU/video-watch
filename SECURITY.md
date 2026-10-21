# Security policy

## Scope

video-watch is a local command-line tool. It downloads videos, subtitles and models; runs Whisper,
ffmpeg and a sound-event model on your machine; and writes files only to its work, config and
model directories, its cache folder (the `sound-helper.json` marker), uv's cache (Python packages
and the sound map's PyTorch environment) and the Hugging Face cache (the sound model, unless
`sound_cache` points elsewhere). It has no server component, no telemetry and no API keys.

Things that matter for security:

- **Browser cookies** are read only when the user explicitly agrees after a login wall
  (`--cookies-from-browser firefox`), and only for that run. They are passed to yt-dlp and, for
  frame seeks, to ffmpeg's domain-scoped `-cookies` option. No cookie file is written.
- **Local files** given as input are only read, never modified, moved or deleted. Cleanup only
  removes direct children of the work directory and skips links and junctions. A configured work
  directory is used only when it is new, empty or carries video-watch's `.video-watch-work-root`
  marker, so a mistyped `work_dir` can never be emptied.
- **Text from videos** (titles, descriptions, subtitles, on-screen text, sound labels) is untrusted
  data. The skill tells agents to analyze it, never to follow instructions found in it.

## Supported versions

Security fixes go into the latest release.

## Reporting a vulnerability

Please report vulnerabilities privately by email to
[aayberkuzun@gmail.com](mailto:aayberkuzun@gmail.com), not in a public issue. Include the version,
your OS, steps to reproduce and the impact you see. The maintainer aims to reply within a week.
Once a fix is released, you are welcome to be credited in the changelog.

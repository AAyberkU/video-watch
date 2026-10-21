## What and why

<!-- What does this change, and which issue does it address? -->

## How it was tested

<!-- Unit tests, plus real videos for changes to fetch, Whisper, the sound map or frames:
     what you ran, on which OS/GPU, and what you checked in the output files. -->

## Checklist

- [ ] `uv run --with pytest pytest -q` passes; new logic has tests.
- [ ] Real-video check done (or not needed: docs or pure logic only).
- [ ] Interface changes (commands, flags, stdout keys, exit codes, file names, config keys) are
      reflected in `skills/video-watch/SKILL.md`, `README.md`, `docs/REFERENCE.md` and `CHANGELOG.md`.
- [ ] No hard-coded paths, credentials, personal data or media files.
- [ ] Dependencies stay in the PEP 723 headers.

# Verify frame reading in OpenCode and IDE extensions

**Labels:** `help wanted`, `good first issue`, `documentation`

## Context

video-watch's frames are JPEG files; the skill tells the agent to open them with its image or
file-reading tool. Verified so far:

- **Claude Code:** its Read tool shows the frames to the model.
- **Codex CLI:** its `view_image` tool works (with full access on Windows, where the default
  sandbox cannot run per-user ffmpeg and uv installs).
- **Kilo CLI:** its Read tool works with an image-capable model.

Not verified yet:

- **OpenCode:** the skill installs and the CLI runs, but frame reading has not been checked with a
  vision model. OpenCode's documentation says its read tool detects PNG, JPEG, GIF and WebP (up to
  20 MiB) when the model supports vision; an open OpenCode issue reports problems with some custom
  OpenAI-compatible providers.
- **IDE extensions** (Kilo in VS Code, Codex in an IDE): not tested.

## Suggested approach

1. Install video-watch in OpenCode, or in an IDE extension (Kilo or Codex), with a vision-capable
   model.
2. Ask: "What is on screen at 5:00 in https://www.youtube.com/watch?v=aircAruvnKk?" (a public
   3Blue1Brown video). The agent should run `fetch`, then `frames --at 5:00`, then open the JPEG.
3. Check whether the answer describes the actual frame (a neural network diagram), and note how the
   agent opened the file.
4. Also try an overview: "Give me a visual overview of that video."

## Acceptance criteria

- [ ] A report per agent (and per surface: CLI or IDE) with the agent version, model, OS, whether
      the image reached the model, and the tool the agent used.
- [ ] If an agent needs different wording or a specific tool name, a small pull request to
      `skills/video-watch/SKILL.md` (keeping it portable) or `INSTALL.md`.
- [ ] README's agent table updated from "help wanted" to the verified status.

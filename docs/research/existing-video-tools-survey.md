# Existing ways to let LLM agents "watch" video: a survey

> Research from 2026-10-21, written while video-watch was still a private tool, before its sound map
> and public packaging shipped. Star counts, prices and features were current then and change fast.

Date: 2026-10-21. Method: web search and direct fetches of GitHub pages, registries (Glama, PulseMCP,
mcp.so, SkillsLLM), vendor docs, blogs. Star counts were read from GitHub pages on 2026-10-21 unless
marked otherwise. **[unverified]** = came only from a third-party or SEO/vendor page, not from the
project or official docs. GitHub pages did not show last-commit dates; dates given are from
changelogs, PyPI, or articles.

Core question: *are we wasting effort because something better already exists?*
Short answer: **no, but the core (yt-dlp + captions/Whisper + ffmpeg frames) is now a commodity**,
and one project (claude-video, ~18.9k stars) does most of it. Our value is in the specific
combination: fully local, high-quality GPU Whisper on Windows, strict transcript-source policy,
frame/speech pairing, and (then in development) a local non-speech sound map, which no free tool has.

---

## 1. Native / first-party options (no tool to install)

| Option | What it really does | Video visuals? | Audio beyond speech? | Limits / cost | Source |
| --- | --- | --- | --- | --- | --- |
| **Claude apps + Anthropic API** | No video input. Images only (JPEG/PNG/GIF/WebP, first frame of animations). | No | No | — | Third-party guides Sept–Oct 2026; no official announcement of video found **[unverified official]** |
| **ChatGPT app** | No official video upload in the help center (audio-only MP4 is accepted for audio). Anecdotal reports of mp4 analysis in some clients; June 2026 community feature request for native video upload. | Inconsistent | No | — | help.openai.com/en/articles/8555545; community.openai.com/t/1384775 |
| **OpenAI API (GPT-5 family)** | Model pages list Video input as "Not supported". Feature request for Responses API video input closed as not planned; cookbook says send ffmpeg frames as images + Whisper. | Only via your own frames | No | — | developers.openai.com model pages; github.com/openai/openai-node/issues/1778; OpenAI Cookbook "Introduction to GPT-4o" |
| **Gemini API (native video)** | True video input: frames at 1 fps (custom fps, clipping offsets) plus audio track at 32 tok/s; YouTube URLs accepted directly (public only); "agentic mode" timeline navigation on newest models. | Yes, 1 fps default | Yes, it hears the soundtrack (not documented how well for non-speech) | ~300 tok/s high res, ~100 tok/s low res; up to 1 h (high) / 3 h (low) on 1M-context models; YouTube URL is a free preview, free tier 8 h/day. Flash input ≈ $0.75–1.50 / M tokens **[unverified]** → roughly $0.30–1.60 per hour of video. Cloud upload. | ai.google.dev/gemini-api/docs/video-understanding (updated 2026-10-13) |
| **Gemini app "Ask about this video" (YouTube extension)** | Answers from YouTube captions; reported not to analyze the video itself. | No **[unverified]** | No | Videos with captions only | androidauthority.com/gemini-ask-this-video-youtube-feature-3570650 |
| **YouTube "Ask" (watch page) / Ask YouTube** | Gemini Q&A on select videos, select languages/regions, signed-in adults; Ask YouTube (I/O 2026) is search, US Premium Labs. | Unclear | Unclear | YouTube only; not callable by an agent | support.google.com/youtube/answer/14110396; blog.youtube I/O 2026 |
| **NotebookLM** | YouTube as a source = existing captions only; fails without captions; public videos only. | No **[unverified]** | No | Not an agent tool | Tom's Guide; BibiGPT comparison (competitor) |
| **Claude in Chrome / ChatGPT Atlas / browser agents** | Read the YouTube page and its transcript panel; screenshot-based, slow. | Only a screenshot of the current frame | No | Needs the browser; per-video manual | tygartmedia.com/what-is-claude-in-chrome; Atlas guides **[unverified]** |
| **Zapier/Make-style connectors** | Found nothing beyond generic YouTube Data API / transcript actions. | No | No | — | zapier.com/blog/claude-connectors |

**Takeaway:** Only Gemini actually "watches" (frames + sound). Claude and GPT still need an external
pipeline in October 2026. Every native "ask about this video" feature we found works from captions.

---

## 2. Top third-party tools (comparison table)

| # | Tool | Type | Popularity (2026-10-21) | Sources | Transcript | Visuals | Audio beyond speech | Local / cloud | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **claude-video** `/watch` — github.com/bradautomates/claude-video | Claude Code plugin / Agent Skill (Codex, Cursor, Copilot, Gemini CLI…) | ~18.9k stars, 2.0k forks; v0.3.2 on 2026-09-25 | yt-dlp sites + local files | Captions first (incl. auto), then local WhisperX (default `small`; verified only on Apple Silicon) or Groq large-v3 / OpenAI whisper-1 | Detail dial: transcript / efficient (keyframes, ≤50) / balanced (scenes, ≤100) / token-burner; 16×16 dedup; `--timestamps`, `--start/--end` focus ≤2 fps; 512 px | Only through optional **Gemini engine** (default when GEMINI_API_KEY is set) | Local or cloud | Free; Whisper API ≈ $0.005/5 min; Gemini per use |
| 2 | **Agent Reach** — github.com/Panniantong/agent-reach | CLI giving agents web reach | ~100k stars (very broad tool) | YouTube subtitles + search via yt-dlp, Bilibili, X, Reddit… | Subtitles only (Whisper only for one podcast site) | None | None | Local | Free |
| 3 | **claude-real-video (crv)** — github.com/HUANGCHIHHUNGLeo/claude-real-video | Python CLI + MCP (`crv-mcp`), Codex edition | ~2.2k stars; v0.10.x; HN front page claimed **[unverified]** | URL via yt-dlp + local | Sidecar/embedded subs first, then local whisper / faster-whisper (VAD) / mlx; default `base`; `--speakers` diarization | ffmpeg scene changes + 1 fps density floor, `--adaptive`, multi-channel dedup (global, settled-local, action), `--text-anchors` (frames at subtitle cues), `frames.json` with selection reason | **Pro ($29 one-time):** audio events, voice emotion, tone curves, camera/shot labels | Local | Free / $29 Pro |
| 4 | **claude-video-vision** — github.com/jordanrendric/claude-video-vision | Claude Code plugin + npm MCP (6 tools) | ~1.4k stars; v1.4.0 | YouTube (yt-dlp) + local | Manual subs → auto-captions → backend (Gemini / local whisper.cpp or openai-whisper / OpenAI API) | Claude picks fps, range, resolution; ≤100 frames, 512 px; no scene detection | Only with Gemini backend (non-speech events) | Local or cloud | Free; Gemini/OpenAI per use |
| 5 | **claude-watch** — github.com/taoufik123-collab/claude-watch | Claude skill | ~1.0k stars | yt-dlp + local | Captions first, then Groq/OpenAI Whisper API (25 MB cap) | Scene-change frames, ≤100, 2 fps cap; **"hook microscope"**: first 10 s at 2 fps + word-level Whisper | None | Mostly local, ASR in cloud | Free + API cents |
| 6 | **watch-skill / DeepWatch** — github.com/oxbshw/watch-skill | Agent skill + Python pkg (27+ hosts) | v1.4.3, 2026-09-08 (stars not checked) | yt-dlp + local | Captions, local Whisper extra, `diarize` extra | Scene sampling + **OCR of on-screen text**; **persistent index** of watched sources | None documented | Local | Free |
| 7 | **video-use** — github.com/browser-use/video-use | Agent video *editing* kit (not Q&A) | ~29.4k stars | Local footage | ElevenLabs Scribe (cloud): word timestamps, diarization, **audio events (laughter, applause)** | LLM "reads" a packed `takes_packed.md`; `timeline_view` PNG = filmstrip + waveform + word labels, only at decision points | Yes, via Scribe | Cloud ASR | ElevenLabs per use |
| 8 | **kimtaeyoon83/mcp-server-youtube-transcript** | MCP server | 603 stars; ~734 weekly visitors on PulseMCP | YouTube | Captions only; ad/sponsor filtering via chapters | Optional `analyze_video` via TwelveLabs Pegasus (direct file URL only) | No | Cloud for visuals | Free / TwelveLabs |
| 9 | **anaisbetts/mcp-youtube** | MCP server | 551 stars | YouTube | yt-dlp subtitles | None | No | Local | Free |
| 10 | **jkawamoto/mcp-youtube-transcript** | MCP server | 490 stars | YouTube | youtube-transcript-api (captions); paginated at 50k chars | None | No | Local | Free |
| 11 | **kevinwatt/yt-dlp-mcp** | MCP server | 291 stars | YouTube search; downloads any yt-dlp site | Subtitles/auto-captions, plus comments, metadata | None | No | Local | Free |
| 12 | **Video Intelligence MCP** — github.com/DugboTek/video-intelligence-mcp | MCP server (Gemini) | Glama "stale"; v0.1.1 2026-08-04 | yt-dlp sites (YouTube, IG, TikTok, Vimeo, X) | Gemini | Gemini ~1 fps; timestamped production breakdown (shots, motion, editing) | Gemini: music, SFX | Cloud | Gemini per use |
| 13 | **yt-analysis-mcp** / **gemini-mcp** family (Legorobotdude, rlabs-inc, salviz) | MCP servers delegating to Gemini | small | YouTube URL passed straight to Gemini | Gemini | Gemini | Gemini | Cloud | Gemini (YouTube URL preview free) |
| 14 | **TwelveLabs MCP** — twelvelabs.io/blog/twelve-labs-mcp-server | Hosted MCP (Marengo search, Pegasus analysis) | Commercial | Uploaded/indexed videos | Platform | Native video model, semantic clip search across a library | Partially (not documented) | Cloud | Free tier 3 GB; paid plans |
| 15 | **VideoDB Director MCP** — github.com/video-db/agent-toolkit | Hosted MCP | 49 stars | Uploaded videos | Platform indexing | Scene index, search, edits | — | Cloud | Paid API |
| 16 | **VideoContext MCP** — github.com/sakshamjain2301/videocontext-mcp | MCP server | 1 star (design reference only) | YouTube + allowlisted local | **Creator captions only, rejects auto-captions**, else local faster-whisper | Profiles / interval / fps / scene / hybrid / timestamps; **low-res previews first, then chosen full frames**; unified TSV timeline | No | Local | Free |

Also seen: OpenAI Cookbook GPT-4o notebook (fixed-interval frames + Whisper; the canonical recipe);
LangChain `YoutubeLoader` / LlamaIndex YouTube reader (captions only); byjlw/video-analyzer,
vidlizer, OpenSceneSense-Ollama (local vision-LLM describes each keyframe, Whisper for audio —
useful for local LLMs without vision tool calls); JCodesMore/youtube-for-ai-agents (55 stars,
search + transcripts + clips); Speechmatics audio events and `audient` MCP (audio-event APIs).

---

## 3. Short notes per strong alternative (better / worse than ours)

**claude-video** — the market leader, closest to us.
Better: huge adoption and community, runs on ~50 agent hosts, one-command install, detail dial with
duration-based frame budgets and dedup, opt-in Gemini engine that truly sees motion and hears audio,
cross-platform. Worse: local Whisper is WhisperX `small` by default and verified only on Apple
Silicon (Windows/NVIDIA untested), otherwise sends audio to Groq/OpenAI; uses auto-captions as a
first-class source; no frame↔speech pairing lines; no non-speech audio without Gemini; >10 min videos
get a sparse scan; Windows needs manual installs; long-video token spend is its top complaint.

**Gemini (API or any Gemini-delegating MCP)** — the strongest *quality* alternative for visuals.
Better: real temporal understanding (motion, order of events) at 1 fps, hears music/SFX, handles
1–3 h, YouTube URLs free in preview, zero local GPU. Worse: cloud upload (privacy), public YouTube
only for URL mode (other sites need download + Files API), 1 fps misses fast cuts, answers are
Gemini's interpretation not inspectable evidence, needs an API key, costs ≈ $0.3–1.6/h beyond free
tier **[unverified price]**, hallucinated timestamps are a known risk.

**claude-real-video** — the most technically advanced free local pipeline.
Better: richer frame selection (scene + density floor + adaptive + three dedup channels + frames at
subtitle cues), diarization, MCP + CLI + Codex edition, cross-platform. Worse: default Whisper
`base`; audio events, emotion and camera labels are paid (Pro); no strict caption policy.

**claude-video-vision / claude-watch / watch-skill** — mid-size variants.
Notable: Claude-chosen fps/range (vision), hook microscope + pacing metrics + Obsidian save
(watch), OCR + persistent index (watch-skill). All weaker than us on ASR quality or locality.

**Caption-only MCP servers (kimtaeyoon83, anaisbetts, jkawamoto, kevinwatt, Agent Reach,
NotebookLM, Gemini app, Atlas, Claude in Chrome)** — cheapest and most popular per query, but blind:
no visuals, fail or degrade without captions, YouTube-centric. Ours is strictly better for content
understanding; they are better only at search/discovery and zero setup.

**TwelveLabs / VideoDB** — hosted video-native platforms. Better for searching *across* a large video
library and semantic clip retrieval. Worse: paid, cloud, upload/index step, overkill for "watch this
one video".

**video-use** — different job (editing), but its LLM-facing design is the best reference for
audio events and compact "reading" of video.

---

## 4. Feature-by-feature: video-watch vs. the field

| Feature | Ours | Best alternative | Verdict |
| --- | --- | --- | --- |
| Sources (yt-dlp sites, local files, playlists) | All yt-dlp sites, local files untouched, playlist listing then pick | claude-video / crv: same yt-dlp base; YouTube MCPs add search | **Equal**; we lack search |
| Transcript quality | Manual subs (original language) else local **large-v3** on GPU (~2–2.5 min/h) | Most use captions incl. auto, then `small`/`base` local or cloud large-v3 | **Ahead** (locally); Groq large-v3 equals quality but is cloud |
| Transcript source policy | Never translated tracks; auto-captions only as fallback; always states source | VideoContext (rejects auto-captions) | **Ahead** of popular tools, matches best design |
| Music / non-speech-heavy speech handling | Auto VAD-off retry; `SPEECH:` density hint | crv VAD gating only; claude-video "missing speech" report (v0.3.0) | **Ahead / equal** |
| Context (description, chapters) | In header | kimtaeyoon83 uses chapters for ad filtering; most include metadata | **Equal**, sponsor filtering is a small idea |
| Frame selection | Targeted time-exact stills, scene-detected contact sheets, "fewest that suffice" | crv (scene+floor+adaptive+dedup), claude-video (budget dial + dedup), VideoContext (preview→full) | **Behind slightly** on automation (no dedup, no budget presets); **ahead** on targeted/minimal use |
| Frame ↔ speech pairing | `SAID:` per frame | crv `--text-anchors`, video-use timeline_view, VideoContext unified timeline | **Ahead / equal** (few do it explicitly) |
| Motion / temporal understanding | Planned dense `frames --range` | Gemini native video; claude-video focus ≤2 fps | **Behind** Gemini; equal to frame tools once range lands |
| On-screen text | Model reads frames | watch-skill OCR, crv Codex edition | **Behind** (no OCR text output) |
| Non-speech audio map | In progress: local PE-A-Frame, `HEARD:`, open-text sound queries, loudness peaks/silences | Gemini (cloud), ElevenLabs Scribe (cloud), crv Pro (paid) | **Unique if shipped**: only free, local, timestamped, queryable option found |
| Diarization / emotion | Not planned | crv (`--speakers` free, emotion Pro), Scribe | **Behind** (by choice) |
| Long videos | Transcript-first, frames on demand | Gemini 1–3 h natively; claude-video sparse >10 min | **Equal** for speech-heavy; Gemini better for visual long-form |
| Privacy / cost | Fully local, $0 marginal, no API keys; cookies only after asking | crv/VideoContext also local; most others need API keys for fallback | **Ahead** of popular tools |
| Ops (cache, purge, locks, tests) | Cache, 24 h purge, locks, extensive unit tests | crv cache; VideoContext TTL + tests; claude-video ~33 commits | **Ahead** in rigor |
| Knowledge capture | Saves a transcript on request | claude-watch Obsidian save; watch-skill persistent index | **Equal**; index/search across watched videos is missing |
| Distribution / hosts | Claude Code + Kilo skill, Windows + NVIDIA (at the time) | Agent Skills on ~50 hosts, MCP for any client, macOS/Linux | **Behind** at the time; the public release adds a portable skill, but macOS and AMD/Intel GPUs still run only in CPU mode |
| Prompt-injection hygiene | Not stated | VideoContext labels video-derived text/images as untrusted data | **Behind** (cheap to add) |

---

## 5. What we could adopt (ranked by value / effort)

1. **Optional Gemini "second opinion" engine, opt-in per question** (claude-video v0.3, claude-video-vision,
   Video Intelligence MCP). Use only when visuals/motion dominate (fast action, sports, dance, gameplay)
   or as a verifier; YouTube URLs are free in preview. Keep local as default; state clearly that it uploads.
2. **Frame dedup + duration-based budget presets** (claude-video detail dial, 16×16 diff ≤2.0; crv density
   floor + adaptive + text-anchors). Cheap, cuts tokens, fits "fewest frames that suffice".
3. **Unified timeline file** merging transcript, frames (`SAID:`), and sound events (`HEARD:`) in one
   compact TSV/Markdown readable by time range (VideoContext timeline; video-use `takes_packed.md`).
   Phase 5 is the natural moment.
4. **Filmstrip + waveform + word-label strip image** for a time range (video-use `timeline_view`) — pairs
   directly with phase-5 loudness peaks/silences and `frames --range`.
5. **OCR of on-screen text** as text output (watch-skill, ~80 MB model) for slides, code, captions burned
   into Shorts — fewer image tokens than sending frames.
6. **Short-form "hook" mode + pacing metrics** (claude-watch: first 10 s at 2 fps + word-level timing;
   cuts/min, mean shot length), useful for short-form content analysis.
7. **Treat video-derived text/images as untrusted data** in SKILL instructions (VideoContext, crv
   reviewer comment) — titles, descriptions, captions, and OCR can carry prompt injection.
8. Small: search via `ytsearch` (Agent Reach, kevinwatt), sponsor-segment skipping from chapters
   (kimtaeyoon83; SponsorBlock would be the stronger variant), preview-then-full-resolution frames.

---

## 6. Verdict

- **Not wasted effort.** Neither Claude nor GPT accepts video natively in October 2026; every native
  "ask this video" feature except Gemini works from captions. Some pipeline like ours is required.
- **But the basic pipeline is a commodity.** If the goal were only "transcript + a few frames", installing
  claude-video (18.9k stars) would give ~70–80% of our current function with far less effort.
- **Where ours is genuinely better:** fully local large-v3 on an NVIDIA GPU under Windows (the popular
  tools default to small/base locally or upload audio), strict transcript-source policy with stated
  source, music/VAD handling, explicit `SAID:` pairing, playlist flow, test rigor,
  zero cost and no keys. Phase 5's local, queryable sound map would be unique among free tools.
- **Where ours is behind:** no native-video option for motion-heavy content (Gemini wins there), no
  dedup/budget presets, no OCR, no search, a single platform and host at the time, no community yet.
- **Use ours for:** private/local videos, non-YouTube sites, non-English or caption-less speech, music-heavy
  or noisy content, "what is said/heard at moment X" questions, playlist research.
  **Use Gemini (or add it as an opt-in engine) for:** fast visual action, choreography, gameplay feel,
  long visual-only videos. **Use caption MCPs** only for quick text summaries of captioned YouTube.

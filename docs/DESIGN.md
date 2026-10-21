# Design notes

Why video-watch works the way it does: the main decisions, the reasons behind them, and the
alternatives that were considered. The exact behavior is in [REFERENCE.md](REFERENCE.md). Research
that informed these choices is in [`research/`](research/); its figures were current on the dates
given there.

- [Goals](#goals)
- [Access layer: yt-dlp](#access-layer-yt-dlp)
- [Speech to text: Whisper](#speech-to-text-whisper)
- [Transcript source order](#transcript-source-order)
- [Timestamps, context and the transcript file](#timestamps-context-and-the-transcript-file)
- [VAD, music and the no-VAD retry](#vad-music-and-the-no-vad-retry)
- [Frames](#frames)
- [Any video source](#any-video-source)
- [Playlists](#playlists)
- [Login walls and cookies](#login-walls-and-cookies)
- [Sound map](#sound-map)
- [Storage and lifecycle](#storage-and-lifecycle)
- [Platforms, devices and CPU mode](#platforms-devices-and-cpu-mode)
- [Packaging](#packaging)
- [Not pursued](#not-pursued)

## Goals

- Let agents that read text and images answer questions about a video: what is said, what is
  heard, and what is seen at a given moment.
- Run locally, with no API keys, no telemetry and nothing uploaded.
- Spend tokens only where the answer needs them: text first, images only when the visuals matter.
- Be honest about uncertainty: always state the transcript source, label sounds as acoustic cues,
  never use a possibly translated subtitle track.
- Keep a small, stable, machine-readable interface (`KEY: value` lines and exit codes) that a skill
  can rely on.

## Access layer: yt-dlp

- **Choice:** yt-dlp fetches metadata, subtitles, audio and stream URLs.
- **Why:** it is the most actively maintained extractor, supports over a thousand sites, provides
  the audio needed for Whisper, and has the most countermeasures when sites change.
- **Alternatives:** `youtube-transcript-api` (captions only, YouTube only); third-party YouTube MCP
  servers (uncertain maintenance); the official YouTube Data API (serves captions only for the
  caller's own videos).

## Speech to text: Whisper

- **Choice:** OpenAI Whisper models through faster-whisper (CTranslate2), with
  `BatchedInferencePipeline` (batch 8, smaller when free VRAM is short), `float16` on CUDA, VAD on
  (`min_silence_duration_ms=500`),
  `condition_on_previous_text=False`, beam 5, the language from the metadata, and word timestamps.
- **Why Whisper:** multilingual accuracy. Many strong non-Whisper models do not support languages
  such as Turkish (NVIDIA Parakeet and Canary, Voxtral, Granite Speech, Phi-4-multimodal, Kyutai
  STT), distil-whisper is English-only, and the multilingual alternatives that exist were weaker in
  the evidence found.
- **Why these settings:**
  - `float16` is required on RTX 50-series (Blackwell, sm_120): `int8`, `int8_float16` and `auto`
    fail there with `CUBLAS_STATUS_NOT_SUPPORTED`. It halves memory against fp32 with no meaningful
    accuracy loss.
  - VAD skips silence and music, the main triggers of large-v3 hallucinations; turning off
    `condition_on_previous_text` prevents repetition loops; beam 5 is the standard default.
  - Passing the language explicitly avoids wrong detection on intros.
  - Batch 8 by default: large-v3 peaks at about 7.2 GB of VRAM. Batch 16 was not faster and used
    about 2.5 GB more. Smaller GPUs get a smaller batch automatically (see
    [Platforms, devices and CPU mode](#platforms-devices-and-cpu-mode)).
  - Word timestamps cost nothing measurable and let frames and speech line up.
  - The model loads before the audio download, so GPU problems surface early.
- **Default model `large-v3`.** `large-v3-turbo` is a pruned large-v3 (same encoder, 4 decoder
  layers instead of 32), faster and smaller but at most as accurate and weaker outside English per
  its model card, so it is the recommendation for 4–8 GB GPUs rather than the default (an adaptive
  batch lets large-v3 itself run on 6 GB cards, more slowly). `large-v2`
  stays available for videos where v3 falls into repetition loops.
- **Alternatives:** the second engine for Apple Silicon is mlx-whisper (see
  [Apple Silicon engine](#apple-silicon-engine)); whisper.cpp (Vulkan) and ffmpeg's `whisper` filter
  are candidates for AMD and Intel GPUs (see the [roadmap](../ROADMAP.md)). WhisperX and HF
  transformers were not chosen. Paid cloud APIs were never the default.

## Transcript source order

- **Choice:**
  1. Manual (uploader) subtitles in the video's original spoken language, or for local files an
     embedded text subtitle stream whose language tag matches the audio stream.
  2. Otherwise always Whisper.
  3. YouTube's original-language auto-captions (`-orig`) only when Whisper cannot run on the GPU at
     that moment, labeled `auto-subtitles-fallback`.
  - Translated tracks are never used, and an unknown original language means Whisper.
  - `--whisper` forces Whisper; a user asking for Whisper is always honored.
  - Every answer states its source, and `fetch` stdout always starts with `SOURCE:`.
- **Why:**
  - Human subtitles are usually more accurate than any ASR and arrive in seconds.
  - YouTube's auto-captions are generally weaker than large-v3 (missing punctuation, misspelled
    names, rolling duplicate lines), and Whisper takes only minutes.
  - Most sites other than YouTube do not state a language. An unmatched track could be a
    translation, and a wrong-language transcript is worse than a few minutes of Whisper.
  - The auto-captions fallback stays YouTube-only: YouTube marks its original-language ASR track
    with `-orig`, so a translation can be ruled out; other sites' automatic captions carry no such
    marker (TikTok's are often machine-translated).
- **Alternatives:** auto-captions first and Whisper only on poor quality; using a lone manual track
  when the language is unknown. Both rejected for accuracy and translation risk.

## Timestamps, context and the transcript file

- **One stamp per segment** in `transcript.md` (roughly every 3–6 s). Fixed 1–30 s grids would cut
  sentences and fake precision that segment times (about ±1 s) do not have. A paragraph version
  (`transcript-paragraphs.md`, about 15–30 s per stamp) exists for saving a readable copy, and
  `segments.json` keeps precise and word-level times.
- **Description and chapters in the header.** Descriptions often hold context the speech lacks
  (the premise of a joke, names, credits, timestamps). YouTube caps descriptions at 5000 characters
  (about 1.5k tokens), so they are included verbatim in the file the agent reads anyway, rather than
  in a second file the agent could forget, or on stdout, which would break the one-line contract.
- Old cached transcripts get description and chapters on their next `fetch` from one metadata
  request; Whisper never runs again.

## VAD, music and the no-VAD retry

- **Problem:** VAD drops singing over music, so a music video can come back empty.
- **Choice:** a hybrid.
  - Obvious case, automatic: a VAD run with fewer than 5 words per minute of audio is retried once
    with VAD off on the same audio and loaded model. Normal speech is about 120–160 words per
    minute, so 5 means VAD missed nearly everything, and sparse-speech videos do not pay for a
    retry (about 1.6x the first transcription).
  - Gray zone, the agent decides: every Whisper transcript prints a `SPEECH:` hint (words per
    minute and the longest stretch without speech). The agent has the context the script lacks
    (title, description, chapters, the question) and reruns with `--no-vad` only when the gap
    likely holds what matters.
- **Alternatives:** asking the user every time (a round trip, and an empty result looks like a dead
  end); always running without VAD on music (no reliable music signal, and VAD-off output
  hallucinates more); retrying on any long gap (gaps are often legitimate); running both decoders
  and merging (double cost, fragile merging).

## Frames

- **When to look:** only when the question is about the visuals, the speaker points at the screen
  without describing it, or the video has little speech. A frame costs about 1.2k tokens and a
  contact sheet about 1.85k, so frames are worth it only when the transcript is not enough.
- **No fixed counts.** The agent takes the fewest frames that suffice, says which moments it looked
  at and why, and asks first only for very large scans (roughly 100k+ tokens). `--max-frames`
  (default 200) is a safety ceiling against runaway output, not a product limit. A fixed small cap
  would make visual questions on gameplay or slide-heavy videos fail arbitrarily.
- **Three modes:** `--at` (full-detail stills), `--overview` (3x3 contact sheets from scene changes
  plus interval fills, 640 px tiles that stay legible after the ~1568 px downscale vision models
  apply), and `--range` (dense, time-exact frames over a short window for fast action).
- **Time-exact by construction.** ffmpeg seeks with `-ss` before `-i` in a stream URL and decodes
  from the seek point; nothing is extracted from stream-copied sections. An early design used
  yt-dlp's section downloads, but a copied section starts at the previous keyframe, so frames were
  seconds off; verified by PSNR against reference frames, the direct-seek fallback in another HLS
  format with the same codec is bit-identical. Overview picks are whole seconds, so a tile's stamp
  passed to `--at` returns the same frame. Black stretches (fade-ins) are skipped.
- **No full download for `--at`:** a plain-HTTPS format with a short side up to 1080 px; among equal
  sizes the smallest file wins (on a 1080p Short, vp9 seeks were about 3x faster than avc1). The
  overview downloads one 360p video-only format and deletes it right after.
- **Pairing:** each frame or tile carries `SAID:` (the transcript segment at that moment, within
  2 s) and `HEARD:` (sound-map labels within 1 s), from what is already cached. The most common
  visual question ties a moment to speech or sound, and plain text lines are exact and cheap.
  Burning text into the images was rejected: harder to read reliably, covers the frame, and makes
  cached frames depend on the transcript.
- **CPU only:** frames never load Whisper or the sound model, and never trigger a fetch.
- **Rejected:** YouTube storyboards (too low resolution for on-screen text), fixed-interval sampling
  only (misses short scenes), Pillow for sheets (ffmpeg's `tile` and `drawtext` suffice).

## Any video source

- **Choice:** YouTube, any other http(s) URL through yt-dlp's own extractor matching (no allowlist),
  and local files through ffprobe and ffmpeg. Playlists passed to `fetch` or `frames`, live,
  upcoming and processing content, and DRM-protected content are rejected with exit 3.
- **Work keys are computed offline:** the YouTube ID; `<extractor>-<id>` from yt-dlp's URL matching
  for other sites; `file-<hash>` of path, size and modification time for local files. A cache hit
  and `clean` then need no network request, and an edited file gets a fresh transcript without
  hashing a multi-GB file.
- **Local files are only read.** Audio is decoded into a 16 kHz mono WAV in the work directory (what
  Whisper resamples to anyway), selecting the default audio stream; frames seek in the file in
  place. Outputs are refused outside the work root or at the source path, and files inside the work
  root are rejected as input, so cleanup can never remove a source.
- **Findings that shaped the code:** sites without audio-only formats get a small muxed format;
  formats without size or codec still work; TikTok needs the session cookies from extraction for
  stream seeks (passed through ffmpeg's domain-scoped `-cookies`, never a raw header that would
  follow redirects); Vimeo's player streams carry DRM keys only in their media playlists, which are
  checked so the run exits cleanly; silent videos stop before Whisper loads.

## Playlists

- **Choice:** `playlist` lists entries with yt-dlp's flat extraction (no download), with markers for
  unavailable, private, live or upcoming entries and a Whisper time estimate. The agent lists first,
  picks only the videos a question needs, asks before large selections (more than 10 videos or about
  30 minutes of processing), and fetches one by one.
- **Why:** questions about a playlist usually need a few of its videos. A batch fetch would spend
  GPU time and context on videos the answer does not use, and the script cannot judge relevance.

## Login walls and cookies

- **Choice:** the first attempt never uses cookies. Login walls on any site map to exit 2
  (`NEEDS_COOKIES`); the agent asks the user and, with approval, reruns with
  `--cookies-from-browser firefox`. If that fails, it stops and explains.
- **Why:** cookies tie requests to the user's account on that site, so heavy or silent use could flag
  the account. yt-dlp can read Firefox's cookies directly, while Chrome's use app-bound encryption
  on Windows. A `cookies.txt` export was rejected as a long-lived credential file on disk.

## Sound map

Speech is not all a video says: music, laughter, applause or an explosion change what a moment
means. The sound map turns them into a timeline the agent reads as text. Research:
[phase5-audio-models.md](research/phase5-audio-models.md) and
[pe-a-frame-base-vs-large.md](research/pe-a-frame-base-vs-large.md).

### Model

- **Choice:** [`facebook/pe-a-frame-base`](https://huggingface.co/facebook/pe-a-frame-base) (Meta,
  Apache-2.0, not gated; 560M parameters, 2.24 GB; 48 kHz mono input, one embedding per 40 ms). It
  scores any text prompt against every frame, so it gives real onsets and offsets and also answers
  open-text queries (`sounds --query`).
- **Why:** frame-level, open-vocabulary sound event detection in `transformers` with a clean
  licence. Base is within rounding of large on AudioSet-Strong (PSDS1A/PSDS1T 0.42/0.60 vs
  0.43/0.61), better on DESED, and 2.6x smaller. Clip taggers (CED, BEATs, AST) give only
  window-level onsets; the closed-set PretrainedSED has no open vocabulary and no stated weight
  licence.

### Isolation

- **Choice:** the model runs in `sound_events.py`, a second PEP 723 script with its own environment
  (PyTorch from the CUDA 12.8 index, a pinned `transformers`), started after Whisper has released the
  GPU. The two exchange JSON files in the work directory, tied together by a request id.
- **Why:** faster-whisper's environment stays untouched; `--no-sounds` and the unit tests never
  import PyTorch; process exit frees all VRAM, so Whisper (about 7 GB with large-v3) and the sound
  model (about 4.4 GB) never hold the GPU together; the JSON contract is the swap point for another
  model; all span logic stays in plain, unit-tested Python.
- The helper never outlives its run: on Windows it runs in a kill-on-close Job Object (a process
  group elsewhere), the whole tree is stopped on a timeout, and it exits by itself when the
  requesting process is gone.
- **Batch 1:** the codec front end takes 93% of the time, so batching does not help (34 s per hour
  at batch 1, 2 and 4) while peak VRAM rises from about 4.4 to 7.2 GB.
- **Alternatives:** loading the model in the Whisper process (DLL conflicts, import cost on every
  run); ONNX Runtime (a Blackwell regression in onnxruntime-gpu 1.23, and no export exists).

### Chunks and silence

- 30 s chunks (the paper's clip length) overlapping by 4 s; each frame comes from the chunk where it
  has context on both sides, so a sound across a chunk border is one span.
- The model hears digital silence as music, wind or engines, and silence inside a chunk skews the
  scores next to it. Stretches below -70 dBFS of 1 s or more are kept out of the model's input, and
  every silent 40 ms frame is muted for all labels.

### Calibration

- **Set-up:** effect clips from ESC-50 (30 classes mapped to the labels; used locally for testing
  only, never committed) mixed at known times into 10-minute beds of public-domain speech, CC0 music,
  and that music under the speech at -12 dB; 60 events per bed at +6, 0 and -6 dB SNR, plus clean
  copies for false alarms and copies with sustained 20–90 s events.
- **Finding:** one global probability threshold cannot work. Separation is good (frame AUROC
  0.95–0.99 for most labels), but on plain music many prompts sit at a median probability of
  0.5–0.98. The model card's 0.2 gave over a hundred false-alarm units per minute.
- **Rules chosen:**
  - One global logit threshold, 4.0 (probability 0.982), with hysteresis down to 3.0.
  - Under music only (the music logit's 5 s median at 0 or above), an event must also rise 6 above
    its own 90 s rolling median.
  - Look-alike pruning: of overlapping event spans, keep a span only within 4 of the best peak and
    among the 4 best labels.
  - Events last at least 0.2 s and merge across gaps up to 0.5 s; music, speech and singing at least
    1 s with gaps up to 2 s.
- **Result:**

  | Bed | Hit rate +6 / 0 / -6 dB | Onset error (median) | False alarms per minute, clean bed |
  | --- | --- | --- | --- |
  | Speech | 0.88 / 0.80 / 0.72 | 0.03–0.04 s | 0.7 |
  | Speech over music | 0.85 / 0.83 / 0.60 | 0.03–0.06 s | 0.6 |
  | Music | 0.75 / 0.70 / 0.35 | 0.03–0.04 s | 0.4 |

  The price of the music margin: a sound lasting longer than about 45 s under music (rain behind a
  song) is mostly not reported. Sustained sounds kept 86% of their time over speech, 55% over speech
  with music and 31% over music.
- **Rejected:** per-video percentile baselines (7–40 false alarms per minute under music), a rolling
  median everywhere (sustained sounds lost even without music), per-prompt thresholds (would overfit
  30 classes and say nothing about the others), sentence prompts instead of labels (no consistent
  gain).
- **Prompt set:** 45 labels covering music and speech coverage, human reactions as sounds,
  action and impacts, vehicles, animals, household and nature sounds, and edited-video and game
  sounds. Ambiguous prompt texts are reworded ("punch or hit impact", not the drink); prompts that
  co-fired or false-alarmed without adding value were removed. Effects with no calibration data
  (air horns, record scratches) are left to `sounds --query`.

### Defaults and failure

- **Off until the user enables it.** It needs an NVIDIA GPU, about 5.3 GB of downloads (a 3.0 GB
  PyTorch environment and the 2.3 GB model) and about 4.4 GB of VRAM at peak (4600 MB must be free
  before it starts), so the user decides during setup; `doctor` recommends it from 6 GB of total
  VRAM. When enabled it runs on every `fetch`: about 55 s per hour of audio in all, about 10 s on
  short videos (mostly start-up).
- **Never an unasked download:** a `fetch` or `sounds` run never installs the sound map. If it is
  not set up, `fetch --sounds` notes that it was skipped (exit 0), and `sounds` exits 3 with a hint
  to run `setup --sounds on`.
- **The transcript is primary:** a map that cannot be built is a `NOTE:` and never changes the exit
  code. A busy GPU is checked before any audio is fetched.
- **Labels are acoustic cues,** never emotions. Speech emotion recognition was rejected: on natural
  speech the best in-domain systems reach about 0.45 macro-F1, and zero-shot results on non-English
  speech are near chance. Paralinguistic sounds (laughter, crying, screaming) come from the sound map
  instead.

## Storage and lifecycle

- Work files live under the work directory (default `<system temp>/video-watch/<work key>/`), so
  follow-up questions do not re-download.
- Audio is deleted right after use, also on failure; the overview's video right after the tiles.
- The agent runs `clean` when the conversation about a video is over; every run also purges work
  directories older than 24 hours.
- An exclusive `.lock` per work directory keeps a `frames` run from deleting files under a running
  `fetch`, or the reverse.
- Nothing is saved elsewhere unless the user asks; a saved transcript is text only, and frames are
  never saved.

## Platforms, devices and CPU mode

- **Full speed on NVIDIA GPUs (Windows and Linux)** through CUDA. Developed and measured on Windows
  11 with an RTX 5060 Ti 16 GB (Blackwell, sm_120). The CUDA libraries come from pip wheels
  (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`), so no system CUDA toolkit is needed. On Linux the
  dynamic loader reads `LD_LIBRARY_PATH` only at process start, so the CLI re-executes itself once
  with the wheels' library folders on it (the approach faster-whisper documents); verified on WSL
  (Arch Linux) with the `small` model. On Windows the DLL folders are added at run time.
- **No silent CPU fallback.** large-v3 on the CPU takes about 22 minutes per hour of audio, and a
  silent CPU run would tie up the machine without warning. With `device: auto` or `cuda` and no
  usable CUDA, `fetch` exits 3 with `ERROR: no usable CUDA GPU for Whisper (…)` and a `HINT:`
  explaining CPU mode. YouTube's auto-captions are not used in that case either: the machine cannot
  run Whisper at all, and the user should decide about CPU mode rather than silently get weaker
  captions every time.
- **The same on Apple Silicon.** An Apple GPU engine that cannot start (mlx-whisper missing, no
  Metal) is an error with a hint about CPU mode, never a CPU run, and never YouTube's auto-captions
  either (they replace Whisper only for a GPU that is busy or full).
- **A busy GPU is different.** When a GPU exists but is out of memory or busy (a game, another
  model), retrying later helps, so YouTube keeps its auto-captions fallback and other sources exit 3
  with a retry hint.
- **Measured VRAM and speed** (RTX 5060 Ti 16 GB, 18.7 minutes of English speech, `float16`, beam 5,
  word timestamps; peak VRAM above what was in use before):

  | Run | Peak VRAM | Transcription RTF |
  | --- | --- | --- |
  | large-v3, batch 8 | 7156 MB | 0.032 |
  | large-v3, batch 4 | 5279 MB | 0.0345 |
  | large-v3, batch 2 | 4509 MB | 0.042 |
  | large-v3, batch 1 | 4221 MB | 0.057 |
  | large-v3-turbo, batch 8 | 3325 MB | 0.022 |
  | small, batch 8 | 1437 MB | 0.020 |

  The decoder's beam search and word timestamps dominate the small models, so turbo and small are
  only modestly faster than large-v3 on a fast GPU. large-v2 shares large-v3's architecture; the
  other batch sizes of turbo and small, and all of medium, base and tiny, are estimates.
- **Adaptive batch.** Before the model loads, the free VRAM is read from `nvidia-smi`, and the
  largest batch of 8, 4, 2 and 1 whose measured peak plus a 400 MB margin fits is used (batch 8
  when `nvidia-smi` is unavailable). For large-v3 that means 8 from 7556 MB free, 4 from 5679 MB, 2
  from 4909 MB, otherwise 1. An 8 GB card therefore runs large-v3 at batch 4 (about 8% slower) and a
  6 GB card at batch 2 (about 1.3x slower), with the same transcript. The `ESTIMATE:` line accounts
  for the batch.
- **Model recommendation by total VRAM** (`doctor`, MiB as `nvidia-smi` reports them, so an "8 GB"
  card counts): 7800 or more → `large-v3`, 3900 or more → `large-v3-turbo`, otherwise or without
  usable CUDA → `small`. The sound map is recommended from 6000 MiB. These are recommendations only;
  the user can choose any model. `doctor` and `setup` warn when even batch 1 plus the margin exceeds
  the total VRAM ("will likely run out of memory"), or when only batch 8 does not fit ("memory may
  be tight … runs with a smaller batch").
- **CPU mode, only when chosen** (`device: cpu`): `int8` (the fast CPU type), batch 4 (as fast as 8
  and faster than sequential decoding), threads set to every core on Apple Silicon and otherwise
  to half the logical CPUs (at most 16;
  hyper-threads slow int8 matrix work down). Measured with 6 threads on a 12-thread desktop CPU:
  RTF 0.074 for `small`, 0.19 for `large-v3-turbo`, 0.36 for `large-v3` (12 threads were no faster).
  The transcript header says `int8, cpu`, and `fetch` prints a `NOTE:`, so a CPU transcript is never
  mistaken for a GPU one.
- **Known limitation:** the CUDA wheels (about 1.2 GB) are part of the script's inline dependencies
  on Windows and Linux, so CPU-only machines download them too (macOS does not). Splitting them out
  is on the [roadmap](../ROADMAP.md).
- **macOS:** no CUDA. On Apple Silicon `auto` runs Whisper on the Apple GPU (next section); CPU mode
  (the CUDA wheels are skipped there) works on every Mac when chosen. Verified on an Apple M5 Pro
  (24 GB, macOS 27): `small` int8 on the CPU transcribed the 18:40 English lecture in 71 s (RTF
  0.063).
- **Not yet supported:** AMD/Intel GPUs, which need another engine (for example whisper.cpp with
  Vulkan), and the sound map on anything but an NVIDIA GPU (its PyTorch path is CUDA-only in this
  release). See the [help-wanted drafts](help-wanted/).

### Apple Silicon engine

- **Choice:** mlx-whisper (Apple's MLX framework) on the Apple GPU, selected by `device: auto` on an
  arm64 Mac (`metal` explicitly; `cpu` still means faster-whisper on the CPU). Its weights are
  `mlx-community/whisper-*` conversions in float16, the same sizes as the CTranslate2 ones.
- **Why not whisper.cpp with Metal** (pywhispercpp 1.5.1, also installable through the PEP 723
  header, prebuilt Metal wheel, native Silero VAD, word timestamps): the speed is similar, but the
  no-VAD pass, which is the music-video retry path, went wrong with `large-v3-turbo` and the
  binding's default greedy settings (458 words for a 318-word song, word error 0.49 against 0.04
  for mlx-whisper on the same song), and its language detection is a separate call. mlx-whisper
  has the temperature fallback and compression-ratio check of OpenAI's reference decoder, word
  timestamps from the cross-attention alignment, and its transcription path needs no binary. The
  cost of mlx-whisper is a bigger environment (MLX, numba and SciPy: about 0.7 GB in uv's cache,
  with its PyTorch dependency overridden away because only the conversion tool imports it) than
  whisper.cpp's wheel, and it has no VAD and no beam search.
- **VAD:** mlx-whisper has none, so faster-whisper's Silero VAD (already a dependency, the same
  settings, `min_silence_duration_ms` 500) cuts the audio, the speech chunks are joined end to end
  and decoded in one call, and every segment and word time is mapped back to the original timeline.
  This was measured against passing the chunks as `clip_timestamps`, which decodes each chunk in
  its own 30 s window and was 2.4 to 4.4 times slower with `small` (RTF 0.041 to 0.074 against
  0.017 without VAD). An empty
  VAD result returns no segments, so the automatic no-VAD retry behaves as on CUDA.
- **Decoding differences:** greedy decoding with temperature fallback instead of beam search 5
  (mlx-whisper has no beam search; the header says `beam 1`), no batching, and the
  language probability comes from `detect_language` on the first 30 s of speech.
- **Measured** on an Apple M5 Pro (15 cores, 24 GB, macOS 27) with three 3-minute public clips,
  `fetch`'s Whisper settings (word timestamps on, no conditioning on the previous text, language
  auto-detected): an English lecture with manual subtitles (3Blue1Brown, 0:30-3:30), a Turkish talk
  with manual Turkish subtitles (TEDxIstanbul, 1:00-4:00) and a pop song (0:00-3:00, English
  subtitles). RTF is transcription time over audio time (model already loaded, VAD included); WER is
  the word error against the subtitle text (subtitles are not verbatim, so only the comparison
  between engines is meaningful); the music clip is the no-VAD pass, because Silero VAD drops all of
  a song with every engine; memory is the peak resident set size of the process.

  | Engine and model | RTF en / tr / music | WER en / tr / music | Peak memory |
  | --- | --- | --- | --- |
  | faster-whisper CPU int8, beam 5, `large-v3-turbo` | 0.108 / 0.099 / 0.142 | 0.060 / 0.100 / 0.094 | 3.8 GB |
  | **mlx-whisper Apple GPU, `large-v3-turbo`** | **0.026 / 0.025 / 0.032** | **0.043 / 0.034 / 0.044** | 2.1 GB |
  | whisper.cpp Metal, `large-v3-turbo` | 0.031 / 0.031 / 0.040 | 0.043 / 0.047 / 0.487 | 2.0 GB |
  | faster-whisper CPU int8, beam 5, `small` | 0.069 / 0.103 / 0.146 | 0.028 / 0.169 / 0.167 | 2.4 GB |
  | **mlx-whisper Apple GPU, `small`** | **0.019 / 0.019 / 0.013** | **0.055 / 0.087 / 0.189** | 1.2 GB |
  | whisper.cpp Metal, `small` | 0.017 / 0.017 / 0.012 | 0.052 / 0.116 / 0.063 | 0.9 GB |
  | mlx-whisper Apple GPU, `large-v3` | 0.064 / 0.065 / not run | 0.050 / 0.025 / not run | 3.9 GB |

  The CPU baseline's batched VAD run dropped words in the Turkish clip (301 of 320 with turbo; its
  no-VAD run scores 0.034), which is why its VAD numbers are worse than the engines' without it. The
  Apple GPU is about four times faster than this Mac's CPU with `large-v3-turbo` and three to five
  times faster with `small` on these clips (2.4 times end to end on the full lecture, where audio
  decoding and VAD weigh more), at similar accuracy on English and Turkish speech (`small` is worse
  on the song: 0.19 against 0.17 for the CPU run and 0.06 for whisper.cpp). End to end on the 18:40 lecture
  through `fetch` (audio decoding, VAD and transcription): `small` RTF 0.026 (CPU int8: 0.063),
  `large-v3-turbo` 0.031 and `large-v3` 0.095 (model loading takes 1 to 2 s once cached; the first
  load after installing took 11 to 17 s). These are the numbers behind the
  `ESTIMATE:` of the Apple GPU engine. Every engine ran with the same decoding options except the
  beam size, so the table is a comparison of engines on one Mac, not a benchmark of the models.
- **Memory** is unified memory shared with the system, so `doctor` reads the Mac's total memory
  (`sysctl hw.memsize`) in place of VRAM. The recommendation is `large-v3` from 16 GB,
  `large-v3-turbo` from 8 GB and `small` below; a warning appears when a model's peak plus 3 GB
  exceeds the memory.
- **Not done here:** the sound map on the Apple GPU (separate item in the roadmap), and a batched
  decoder (mlx-whisper decodes one 30 s window at a time and was already several times faster than
  the CPU).
- **No hard-coded machine paths:** models, caches, the work directory and the label font are
  discovered per OS or configured. A configured work directory must be new, empty or already
  carry the `.video-watch-work-root` marker, because the purge and `clean --all` delete inside it.

## Packaging

- **One portable Agent Skill** (`skills/video-watch/SKILL.md`, name equal to the folder name,
  description under 1024 characters) that runs the CLI from its own `scripts/` folder. The CLI lives
  inside the skill folder so one copy carries everything an agent needs.
- **Distribution:** `npx skills add AAyberkU/video-watch` (installs into Claude Code, Codex, OpenCode,
  Kilo and other agents), a Claude Code plugin (this repository is its own marketplace), and an
  agent-guided install (`INSTALL.md`) that asks only the real decisions.
- **No MCP server:** a CLI plus a skill works in every agent that can run commands and read image
  files, needs no running process or per-agent configuration, and image results from MCP tools are
  not reliably shown to the model in every agent. An MCP server would still need a skill to explain
  its output. See [harness-packaging-research.md](research/harness-packaging-research.md).
- **Prompt-injection hygiene:** the skill treats text from the video (title, description,
  subtitles, on-screen text, sound labels) as data to analyze, never as instructions.

## Not pursued

| Option | Reason |
| --- | --- |
| YouTube auto-captions in the normal path | Generally weaker than large-v3; used only as a GPU-busy fallback for YouTube. |
| Batch `fetch` of whole playlists | The script cannot judge which videos matter; list, then fetch what is needed. |
| `cookies.txt` export, Chrome cookies | A long-lived credential file on disk; Chrome cookies use app-bound encryption on Windows. |
| DRM platforms | Out of scope; DRM-protected content exits 3. |
| Whisper batch size 16 | Not faster than 8 and about 2.5 GB more VRAM. |
| Speech emotion recognition | Near chance on natural and non-English speech. |
| Audio-language models for the sound timeline | 10–15x worse at multi-label tagging than a sound-event model, 10–36 GB of VRAM. |
| PE-A-Frame large | No measurable gain for 2.6x the size. |

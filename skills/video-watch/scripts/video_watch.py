# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # curl-cffi: browser impersonation, which some sites (Vimeo) now require.
#     "yt-dlp[default,curl-cffi]",
#     "faster-whisper>=1.2.1,<2",
#     "ctranslate2>=4.5,<5",
#     # CUDA libraries for CTranslate2 (Windows and Linux; macOS has no CUDA and runs on the CPU).
#     "nvidia-cublas-cu12; sys_platform != 'darwin'",
#     "nvidia-cudnn-cu12>=9,<10; sys_platform != 'darwin'",
#     # faster-whisper 1.2.1 passes `metadata_errors` to av.open(), which PyAV 16+ removed.
#     "av>=11,<16",
# ]
# ///
"""video-watch: transcripts (subtitles or local Whisper), a sound map and frames of any video.

Sources: YouTube, any other site yt-dlp supports, and local video files. Everything runs locally.
The sound map (sounds beyond speech, PE-A-Frame) runs in sound_events.py next to this file, a
separate uv script with its own PyTorch environment.

Usage:
    uv run video_watch.py fetch <url-or-path> [--whisper] [--lang CODE] [--no-vad]
                                              [--model NAME] [--device auto|cuda|cpu]
                                              [--sounds | --no-sounds]
                                              [--cookies-from-browser BROWSER]
    uv run video_watch.py frames <url-or-path> --at 1:23,4:56 | --overview [--interval S]
                                               [--scene-threshold 0.3] | --range 1:00-1:10
                                               [--fps 2] [--max-frames 200]
                                               [--cookies-from-browser BROWSER]
    uv run video_watch.py sounds <url-or-path> [--query "car horn, glass breaking"]
                                               [--threshold 4] [--cookies-from-browser BROWSER]
    uv run video_watch.py playlist <url> [--limit 200] [--cookies-from-browser BROWSER]
    uv run video_watch.py clean <video_id-url-path-or-key>
    uv run video_watch.py clean --all
    uv run video_watch.py doctor
    uv run video_watch.py setup [--device auto|cuda|cpu] [--model NAME] [--sounds on|off]
                                [--model-dir PATH] [--yes] [--dry-run]
    uv run video_watch.py config show | path | get KEY | set KEY VALUE | unset KEY

Models: large-v3 (default), large-v3-turbo, large-v2, medium, small, base, tiny.
Machine-readable results go to stdout as `KEY: value` lines; progress goes to stderr.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import ctypes
import gc
import glob
import hashlib
import heapq
import html
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from array import array
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NoReturn, Sequence
from urllib.parse import parse_qs, urlparse
from urllib.request import url2pathname

APP_NAME = "video-watch"
# Work dirs live under the work root: `<system temp>/video-watch` unless `work_dir` is configured.
DEFAULT_WORK_ROOT = Path(tempfile.gettempdir()) / APP_NAME
WORK_ROOT = DEFAULT_WORK_ROOT
# A configured work root must be new, empty or already marked: the purge and `clean --all` delete
# its subdirectories, so an existing folder with other content is never adopted.
WORK_ROOT_MARKER = ".video-watch-work-root"
# PATH before Whisper's CUDA DLL directories are added; the sound helper gets this one.
PATH_AT_START = os.environ.get("PATH", "")
PURGE_AGE_S = 24 * 3600
# Transient downloads and intermediates, deleted after each run and by the purge.
DOWNLOAD_PATTERNS = (
    "audio.*", "video.*", "*.part", "*.ytdl", "sound-*.json", "frames/tile-*", "frames/*.tmp.jpg",
)

# Configuration: a JSON file in the user config dir (VIDEO_WATCH_CONFIG overrides its path).
# Precedence per key: command-line flag, then environment variable, then config file, then default.
CONFIG_ENV = "VIDEO_WATCH_CONFIG"
CONFIG_KEYS = ("device", "model", "model_dir", "sounds", "sound_cache", "work_dir")
CONFIG_ENV_VARS = {key: f"VIDEO_WATCH_{key.upper()}" for key in CONFIG_KEYS}
DEVICES = ("auto", "cuda", "cpu")
DEFAULT_DEVICE = "auto"
DEFAULT_MODEL = "large-v3"

# Whisper models: faster-whisper (CTranslate2) conversions on the Hugging Face Hub, their download
# size in MB, and the VRAM in MB a batched GPU run needs at batch 8 (for warnings and the batch
# size). Sizes are the model.bin files plus tokenizer and config.
WHISPER_MODELS: dict[str, dict[str, Any]] = {
    "large-v3": {"repo": "Systran/faster-whisper-large-v3", "size_mb": 3090},
    "large-v3-turbo": {"repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "size_mb": 1620},
    "large-v2": {"repo": "Systran/faster-whisper-large-v2", "size_mb": 3090},
    "medium": {"repo": "Systran/faster-whisper-medium", "size_mb": 1530},
    "small": {"repo": "Systran/faster-whisper-small", "size_mb": 485},
    "base": {"repo": "Systran/faster-whisper-base", "size_mb": 145},
    "tiny": {"repo": "Systran/faster-whisper-tiny", "size_mb": 76},
}
MODEL_NAMES = tuple(WHISPER_MODELS)

# Whisper settings (fixed by design; see docs/DESIGN.md).
COMPUTE_TYPE = "float16"  # int8/auto fail with CUBLAS_STATUS_NOT_SUPPORTED on RTX 50-series.
CPU_COMPUTE_TYPE = "int8"  # the fast CPU type (no cuBLAS involved)
BEAM_SIZE = 5
BATCH_SIZE = 8
VAD_MIN_SILENCE_MS = 500
# Batched GPU runs pick the largest batch whose peak VRAM (below, plus a margin) fits the free VRAM.
BATCH_CHOICES = (8, 4, 2, 1)
VRAM_MARGIN_MB = 400
# Peak VRAM in MB of a GPU run (float16, beam 5, word timestamps) by batch size, on top of what
# was in use before: measured on an RTX 5060 Ti 16 GB with 18.7 min of English speech (batch 1
# also covers the sequential decoder of --no-vad: 4221 MB for large-v3). large-v2 has large-v3's
# architecture; turbo and small at batch 1, and medium, base and tiny, are estimates.
WHISPER_VRAM_MB: dict[str, dict[int, int]] = {
    "large-v3": {1: 4221, 2: 4509, 4: 5279, 8: 7156},
    "large-v2": {1: 4221, 2: 4509, 4: 5279, 8: 7156},
    "large-v3-turbo": {1: 2400, 2: 2525, 8: 3325},
    "medium": {1: 2100, 2: 2300, 8: 3600},
    "small": {1: 850, 2: 925, 8: 1437},
    "base": {1: 450, 2: 500, 8: 750},
    "tiny": {1: 350, 2: 380, 8: 550},
}
# GPU transcription time relative to batched large-v3 (measured RTF: large-v3 0.032 at batch 8,
# turbo 0.022, small 0.020; the decoder's beam search and word timestamps dominate the small
# models), and the slowdown of smaller batches for the large models (turbo and small: none).
GPU_SPEED = {"large-v3": 1.0, "large-v2": 1.0, "large-v3-turbo": 0.68, "medium": 0.8, "small": 0.64, "base": 0.55, "tiny": 0.5}
BATCH_SLOWDOWN = {8: 1.0, 4: 1.08, 2: 1.31, 1: 1.77}
BATCH_SENSITIVE_MODELS = ("large-v3", "large-v2", "medium")
# CPU mode (only when chosen): int8, batched like the GPU path (batch 4: as fast as 8, faster than
# sequential). Minutes per minute of audio, measured with 6 threads on a 12-thread desktop CPU
# (small 0.074, turbo 0.19, large-v3 0.36; 12 threads were no faster); medium, base and tiny
# are estimates. Slower CPUs take proportionally longer.
CPU_BATCH_SIZE = 4
CPU_RTF = {"large-v3": 0.36, "large-v2": 0.36, "large-v3-turbo": 0.19, "medium": 0.25, "small": 0.075, "base": 0.035, "tiny": 0.02}
CPU_SEQUENTIAL_SLOWDOWN = 1.2
# Expected wall-clock minutes per minute of audio for batched large-v3 on the RTX 5060 Ti,
# plus a fixed overhead for model load and audio download. Calibrated from live runs
# (3.6-63 min videos: transcription RTF 0.031-0.037, end-to-end 0.61 min for 12.6 min of
# audio, 2.36 min for 63.5 min).
ESTIMATE_FACTOR = 0.036
ESTIMATE_OVERHEAD_MIN = 0.2
MODEL_DOWNLOAD_EXTRA_MIN = 1.5  # first use of a model downloads up to ~3 GB
SEQUENTIAL_SLOWDOWN = 1.6  # --no-vad runs the non-batched decoder (measured RTF 0.053 vs 0.031)
# VAD drops singing over music. A VAD run below this many words per minute of audio is retried
# once with VAD off. Normal speech is ~120-160 wpm and sparse commentary still gives ~15+ wpm,
# so only a run that missed nearly everything (e.g. a music video) pays for the retry.
AUTO_RETRY_MAX_WPM = 5.0
VAD_OFF_NOTE = (
    "without VAD, Whisper can hallucinate filler lines over music or silence "
    "(e.g. subtitle credits such as 'Subtitles by the Amara.org community'); treat them as noise"
)

# Frames (ffmpeg, CPU only). The ceiling only stops runaway output; it is not a product limit.
MAX_FRAMES_DEFAULT = 200
FRAME_LONG_SIDE = 1280  # targeted frames; never upscaled
TILE_LONG_SIDE = 640  # contact-sheet tiles, still legible after the ~1568 px image downscale
SHEET_COLS, SHEET_ROWS = 3, 3
SHEET_PADDING = 4
TARGET_MAX_SHORT_SIDE = 1080  # stream format for targeted frames
# Downloaded format for overview scans: 360p already fills a 640 px tile; otherwise the smallest
# size up to 480p, otherwise the size closest to 360p.
OVERVIEW_SHORT_SIDE, OVERVIEW_MAX_SHORT_SIDE = 360, 480
SCENE_THRESHOLD_DEFAULT = 0.3
END_MARGIN_S = 0.5  # a moment at or after the end becomes a frame this far before it
# blackdetect: a frame is black when 98% of its pixels are below 10% luminance. Picks that land
# in black stretches (fade-ins, fades to black) move to the first frame after them.
BLACK_PIXEL_THRESHOLD = 0.10
# Tile label fonts, per OS, the first existing one wins; without any, ffmpeg's default font is tried.
LABEL_FONT_CANDIDATES = {
    "win32": ("arial.ttf", "segoeui.ttf", "tahoma.ttf", "verdana.ttf"),  # in %WINDIR%\Fonts
    "darwin": (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ),
    "linux": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ),
}
# Claude downscales images to a ~1568 px long side and bills about width * height / 750 tokens.
IMAGE_MAX_SIDE = 1568
PIXELS_PER_TOKEN = 750
# SAID lines pair a frame with the cached transcript: the segment containing the moment,
# otherwise the nearest one within this window, cut to one line of at most SAID_MAX_CHARS.
SAID_WINDOW_S = 2.0
SAID_MAX_CHARS = 300
NO_TRANSCRIPT_NOTE = "no transcript cached; run fetch first to pair frames with speech"

# Sound map (phase 5): PE-A-Frame scores text prompts against 40 ms audio frames. The model runs
# in sound_events.py, a separate uv script with its own torch environment (see docs/DESIGN.md).
SOUND_HELPER = Path(__file__).resolve().with_name("sound_events.py")
SOUND_MODEL = "facebook/pe-a-frame-base"
SOUND_SAMPLE_RATE = 48_000
SOUND_HOP = 1920  # samples per model frame
SOUND_FRAME_RATE = SOUND_SAMPLE_RATE / SOUND_HOP  # 25 frames per second
SOUND_LOGIT_SCALE = 100  # logits (and frame levels in dBFS) travel from the backend as int16 hundredths
SOUND_FILL = -32700  # the lowest score: frames without a score, and muted (silent) frames
# Silence gate: the model scores digital silence as music, wind or engines. Frames whose level
# (RMS over 40 ms, from the backend) is below this are muted for every prompt.
SOUND_SILENCE_DBFS = -70.0
# Silence also distorts how the model scores the sounds around it within the same 30 s chunk, so
# silent stretches of at least this long (ffmpeg silencedetect at SOUND_SILENCE_DBFS) are kept out
# of the model's input: each non-silent segment is chunked on its own.
SOUND_SILENCE_SPLIT_S = 1.0
# GPU pre-check before any audio is fetched for the map: nvidia-smi's free memory must reach this
# (the model peaked at about 4.4 GB). Without nvidia-smi the model run itself decides.
SOUND_MIN_FREE_VRAM_MB = 4600
# After the GPU was busy, cached fetches do not retry the back-fill for this long (`sounds`
# always tries).
SOUND_GPU_RETRY_S = 900
SOUND_THRESHOLD_RANGE = (-10.0, 15.0)  # accepted `sounds --threshold` logits
# The paper evaluates 30 s clips; chunks overlap so every frame has context on both sides.
# Batch 1: the same speed as batch 2 or 4 (the DAC front end is the bottleneck) and the least
# VRAM (about 4.3 GB at peak).
SOUND_CHUNK_S = 30.0
SOUND_OVERLAP_S = 4.0
SOUND_BATCH_SIZE = 1
# Calibrated on ESC-50 effects mixed into speech, music and speech-over-music beds (docs/DESIGN.md): one
# global logit threshold (sigmoid 0.982) for every label, with hysteresis. Under music many event
# prompts score high throughout, so there an event must also rise SOUND_MUSIC_MARGIN above its
# own rolling median over SOUND_BASELINE_WINDOW_S; music is "heard" when the music logit's
# SOUND_MUSIC_GATE_S median reaches SOUND_MUSIC_GATE.
SOUND_THRESHOLD = 4.0
SOUND_HYSTERESIS = 1.0
SOUND_MUSIC_MARGIN = 6.0
SOUND_BASELINE_WINDOW_S = 90
SOUND_MUSIC_GATE = 0.0
SOUND_MUSIC_GATE_S = 5
# Discrete events: at least 0.2 s, merged across gaps up to 0.5 s. Coverage labels (music,
# speech, singing): at least 1 s, merged across gaps up to 2 s (pauses between phrases).
SOUND_EVENT_MIN_S, SOUND_EVENT_GAP_S = 0.2, 0.5
SOUND_COVERAGE_MIN_S, SOUND_COVERAGE_GAP_S = 1.0, 2.0
# Look-alike labels fire together (waves also as water, wind and rain): of overlapping events,
# a span stays only within SOUND_COLABEL_MARGIN of the best one and among the SOUND_COLABEL_TOP
# best labels. On the calibration set this cut extra labels per event from about 5 to 1.6.
SOUND_COLABEL_MARGIN = 4.0
SOUND_COLABEL_TOP = 4
# Curated prompts: (label shown in the output, text sent to the model). Chosen for video content:
# coverage of music, speech and singing; human non-speech sounds (reactions, not emotions);
# action and impact sounds; vehicles, animals, household and nature sounds; edited-video and game
# sounds. Labels stay English; the assistant answers in the user's language.
SOUND_PROMPTS: tuple[tuple[str, str], ...] = (
    ("music", "music"),
    ("speech", "speech"),
    ("singing", "singing"),
    ("laughter", "laughter"),
    ("applause", "applause"),
    ("cheering", "cheering"),
    ("crowd", "crowd"),
    ("shouting", "shouting"),
    ("screaming", "screaming"),
    ("crying", "crying"),
    ("whispering", "whispering"),
    ("sighing", "sighing"),
    ("coughing", "coughing"),
    ("sneezing", "sneezing"),
    ("booing", "booing"),
    ("explosion", "explosion"),
    ("gunshot", "gunshot"),
    ("glass breaking", "glass breaking"),
    ("crash", "crash, collision impact"),
    ("punch", "punch or hit impact"),
    ("car engine", "car engine"),
    ("car horn", "car horn"),
    ("siren", "siren"),
    ("motorcycle", "motorcycle"),
    ("airplane", "airplane"),
    ("helicopter", "helicopter"),
    ("train", "train"),
    ("dog barking", "dog barking"),
    ("cat meowing", "cat meowing"),
    ("birds chirping", "birds chirping"),
    ("door", "door slamming or opening"),
    ("knocking", "knocking"),
    ("footsteps", "footsteps"),
    ("keyboard typing", "keyboard typing"),
    ("phone ringing", "phone ringing"),
    ("alarm", "alarm"),
    ("bell", "bell"),
    ("rain", "rain"),
    ("thunder", "thunder"),
    ("wind", "wind"),
    ("waves", "waves"),
    ("fire crackling", "fire crackling"),
    ("video game sound effects", "video game sound effects"),
    ("beeping", "beeping"),
    ("whoosh", "whoosh"),
)
# Labels summarized as coverage (% of the audio) rather than counted as discrete events.
SOUND_COVERAGE_LABELS = ("music", "speech", "singing")
SOUND_TIMELINE_EXCLUDED = ("speech",)  # the transcript already covers speech (timeline, HEARD)
SOUND_SUMMARY_MAX_EVENTS = 6  # event labels named in the SOUNDS: line
SOUND_TIMELINE_MAX = 400  # noise cap: sounds.md keeps the highest-scoring spans; sounds.json all
SOUND_QUERY_MAX = 20  # prompts per `sounds --query`
SOUND_TIMEOUT_S = 1800  # plus half the audio length; a first run installs PyTorch and the model
SOUND_CACHE_ENV = "VIDEO_WATCH_SOUND_CACHE"  # the sound model cache dir (else the HF cache)
# What `setup --sounds on` downloads once: the helper's PyTorch (CUDA 12.8) environment and the model.
SOUND_ENV_SIZE_MB = 3000
SOUND_MODEL_SIZE_MB = 2300
SOUND_PREPARE_TIMEOUT_S = 3 * 3600
# Expected sound-map time: a fixed start-up (uv, PyTorch import, model load) plus 34 s of
# inference per hour of audio on the RTX 5060 Ti; a first run downloads about 5 GB.
SOUND_ESTIMATE_OVERHEAD_S = 15.0
SOUND_ESTIMATE_PER_AUDIO_S = 34.0 / 3600
SOUND_FIRST_RUN_EXTRA_MIN = 5.0
# HEARD lines: sounds whose span contains the frame's moment or is within this window.
HEARD_WINDOW_S = 1.0
HEARD_MAX_LABELS = 6
# Loudness (ffmpeg ebur128 and silencedetect on the same audio).
LOUD_WINDOW_S = 3.0  # ebur128 short-term loudness window
LOUD_PEAK_COUNT = 5
LOUD_PEAK_MIN_SEP_S, LOUD_PEAK_MAX_SEP_S = 5.0, 60.0  # a tenth of the video, clamped
SILENCE_THRESHOLD_DB = -50.0
SILENCE_MIN_S = 3.0
SILENCE_MAX_LISTED = 20  # in sounds.md; sounds.json has all
# The sound map is off until the user enables it (config `sounds`, VIDEO_WATCH_SOUNDS or
# `setup --sounds on`); `--sounds` forces it for one fetch and `--no-sounds` skips it.
SOUND_MAP_DEFAULT = False
# Commands an assistant can copy: the full CLI (`uv run`, installs the transcription packages on
# first use) and the light form for doctor, config and setup --dry-run (standard library only).
SCRIPT_PATH = Path(__file__).resolve()
RUN_CMD = f'uv run "{SCRIPT_PATH}"'
LIGHT_CMD = f'uv run --no-project --python ">=3.11" python "{SCRIPT_PATH}"'
REFERENCE_URL = "https://github.com/AAyberkU/video-watch/blob/main/docs/REFERENCE.md"
SOUNDS_HINT = (
    "the sound model needs about 4.5 GB of free VRAM; retry later (close games or other GPU apps)"
)
SOUNDS_NO_GPU_HINT = (
    "the sound map needs a visible NVIDIA GPU: unset CUDA_VISIBLE_DEVICES or set it to a GPU index such as 0"
)
SOUNDS_SETUP_HINT = (
    f"enable the sound map once with: {RUN_CMD} setup --sounds on --yes "
    "(NVIDIA GPU; about 5 GB download: PyTorch and the model); ask the user first"
)
SOUNDS_ARGS_HINT = f"fix the sounds arguments (see {REFERENCE_URL}); not a yt-dlp problem"



def ffmpeg_install_hint(platform: str | None = None) -> str:
    """How to install ffmpeg (a system dependency the tool never installs) on this OS."""
    platform = platform or sys.platform
    if platform == "win32":
        return "winget install Gyan.FFmpeg"
    if platform == "darwin":
        return "brew install ffmpeg"
    return "the package manager, e.g. sudo apt install ffmpeg"


FFMPEG_INSTALL = ffmpeg_install_hint()
SOUNDS_FFMPEG_HINT = (
    f"sounds need ffmpeg on PATH (system dependency: {FFMPEG_INSTALL}); "
    "transcripts of URLs (fetch) work without it"
)
# frames --range: dense frames over a window, on contact sheets like --overview.
RANGE_FPS_DEFAULT = 2.0  # a 10 s range gives 21 tiles (3 sheets, ~5.5k tokens)
RANGE_MAX_FPS = 10.0  # moments are kept to 0.1 s
RANGE_MAX_SHORT_SIDE = 480  # stream format for range tiles (640 px tiles need no more)
FFMPEG_HINT = (
    f"frames need ffmpeg on PATH (system dependency: {FFMPEG_INSTALL}); "
    "transcripts (fetch) work without it"
)
LOCAL_FFMPEG_HINT = (
    f"local video files need ffmpeg and ffprobe on PATH (system dependency: {FFMPEG_INSTALL}); "
    "transcripts of URLs (fetch) work without it"
)

EXIT_OK, EXIT_ERROR, EXIT_NEEDS_COOKIES, EXIT_NO_TRANSCRIPT = 0, 1, 2, 3
SANDBOX_HINT = (
    "the agent's sandbox may block access; allow full access for video-watch commands "
    "(for Codex: /permissions → Full access, or codex exec -s danger-full-access)"
)
WORK_DIR_HINT = (
    f"set work_dir to a new or empty folder ({LIGHT_CMD} config set work_dir PATH) or remove the setting "
    f"({LIGHT_CMD} config unset work_dir) to use the default in the system temp folder; "
    "if VIDEO_WATCH_WORK_DIR is set, change or unset it instead"
)
CONFIG_HINT = f"fix the value with: {LIGHT_CMD} config set KEY VALUE (or edit or delete the config file)"
UPGRADE_HINT = f'uv run --upgrade-package yt-dlp "{SCRIPT_PATH}" fetch <url>'
FFMPEG_FAIL_HINT = "ffmpeg could not process this video; rerun once, otherwise report the error (not a yt-dlp problem)"
STREAM_FAIL_HINT = f"the site refused the video stream; rerun once, then try: {UPGRADE_HINT}"
ARGS_HINT = f"fix the frames arguments (see {REFERENCE_URL}); not a yt-dlp problem"
BUSY_HINT = "fetch and frames for the same video must not run in parallel; retry when the other run has finished"
PLAYLIST_HINT = f"list its videos with: {RUN_CMD} playlist <url>, then fetch the needed ones one by one"
# Cookies: the first attempt never uses them. After a login wall the assistant asks the user and
# reruns with Firefox cookies only (no cookies.txt export, no Chrome; see docs/DESIGN.md).
COOKIES_HINT = (
    "the site requires a login; ask the user first, then retry with --cookies-from-browser firefox "
    "(signed in to the site in Firefox)"
)
COOKIES_USED_HINT = (
    "cookies were used but the site still refused; check that Firefox is signed in to the site, "
    "otherwise this video cannot be read"
)
GPU_RETRY_HINT = (
    "Whisper ran out of GPU memory or the GPU is busy; retry fetch later (close games or other GPU "
    "apps) or choose a smaller model (--model large-v3-turbo or small). "
    "The auto-captions fallback exists only for YouTube"
)
NO_CUDA_HINT = (
    "no usable NVIDIA GPU (CUDA) was found. CPU mode works everywhere but is slow: rerun with "
    f"--device cpu --model small, or make it the default with: {RUN_CMD} setup --device cpu --model small --yes "
    "(downloads the small model, about 0.5 GB)"
)

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}

# Work-dir keys: the YouTube id, `<extractor>-<id>` for other sites, `file-<hash>` for local files.
WORK_KEY_MAX_LEN = 80
WORK_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHORT_HASH_LEN = 12
# Inputs without a URL scheme that are not existing files: a media extension marks a missing file.
MEDIA_EXTENSIONS = {
    ".mp4", ".m4v", ".mkv", ".mov", ".webm", ".avi", ".wmv", ".flv", ".ts", ".mts", ".m2ts", ".mpg",
    ".mpeg", ".3gp", ".ogv", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma",
}

# yt-dlp result types that hold several videos (multi_video: bilibili parts, twitcasting, ...).
PLAYLIST_TYPES = ("playlist", "multi_video")
# Playlist listing ceiling (entries printed per call); longer playlists get a truncation NOTE.
PLAYLIST_LIMIT_DEFAULT = 200

# Subtitle tracks that are not subtitles (chat replays, comment overlays).
IGNORED_SUB_TRACKS = {"live_chat", "danmaku", "rechat", "comments", "chat"}
# Site track names that mark machine-generated captions, and sites whose `subtitles` are all
# machine-generated (TikTok's captions are ASR, often machine-translated).
_AUTO_TRACK_RE = re.compile(r"auto|-orig$", re.IGNORECASE)
MACHINE_SUBTITLE_EXTRACTORS = ("tiktok",)
# Embedded subtitle codecs that are text; bitmap ones (PGS, VobSub, DVB) cannot be read.
TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}

# ISO 639-2 (bibliographic and terminology) to ISO 639-1 for the languages Whisper knows; local
# files and some sites tag streams with three-letter codes.
_ISO639_2 = dict(
    pair.split("=")
    for pair in """
    eng=en chi=zh zho=zh ger=de deu=de spa=es rus=ru kor=ko fre=fr fra=fr jpn=ja por=pt tur=tr
    pol=pl cat=ca dut=nl nld=nl ara=ar swe=sv ita=it ind=id hin=hi fin=fi vie=vi heb=he ukr=uk
    gre=el ell=el may=ms msa=ms cze=cs ces=cs rum=ro ron=ro dan=da hun=hu tam=ta nor=no nob=no
    nno=nn tha=th urd=ur hrv=hr bul=bg lit=lt lat=la mao=mi mri=mi mal=ml wel=cy cym=cy slo=sk
    slk=sk tel=te per=fa fas=fa lav=lv ben=bn srp=sr aze=az slv=sl kan=kn est=et mac=mk mkd=mk
    bre=br baq=eu eus=eu ice=is isl=is arm=hy hye=hy nep=ne mon=mn bos=bs kaz=kk alb=sq sqi=sq
    swa=sw glg=gl mar=mr pan=pa sin=si khm=km sna=sn yor=yo som=so afr=af oci=oc geo=ka kat=ka
    bel=be tgk=tg snd=sd guj=gu amh=am yid=yi lao=lo uzb=uz fao=fo hat=ht pus=ps tuk=tk mlt=mt
    san=sa ltz=lb bur=my mya=my tib=bo bod=bo tgl=tl fil=tl mlg=mg asm=as tat=tt lin=ln hau=ha
    bak=ba jav=jw sun=su
    """.split()
)
_NO_LANGUAGE = {"", "und", "mul", "zxx", "mis", "unknown"}


class FetchError(Exception):
    def __init__(self, message: str, code: int = EXIT_ERROR, hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.hint = hint


class FfmpegError(FetchError):
    """ffmpeg failed on a local file or a decode step; the message always starts with "ffmpeg"."""

    def __init__(self, message: str):
        super().__init__(message, EXIT_ERROR, FFMPEG_FAIL_HINT)


class GpuUnavailable(Exception):
    """Whisper could not run on the GPU (CUDA init, out of memory, missing CUDA libraries).
    `no_cuda` marks a machine without a usable CUDA device, where retrying later cannot help."""

    def __init__(self, message: str, no_cuda: bool = False):
        super().__init__(message)
        self.no_cuda = no_cuda

    @property
    def hint(self) -> str:
        return NO_CUDA_HINT if self.no_cuda else GPU_RETRY_HINT


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[dict[str, Any]] | None = None


@dataclass
class Result:
    source: str
    language: str | None
    segments: list[Segment]
    notes: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Platform directories and configuration


def platform_key(platform: str | None = None) -> str:
    """`win32`, `darwin` or `linux` (every other POSIX system is treated like Linux)."""
    platform = platform or sys.platform
    return platform if platform in ("win32", "darwin") else "linux"


def user_config_dir(platform: str | None = None, env: dict[str, str] | None = None, home: Path | None = None) -> Path:
    """The per-user config dir of video-watch: `%APPDATA%\\video-watch` on Windows,
    `~/Library/Application Support/video-watch` on macOS, `$XDG_CONFIG_HOME/video-watch`
    (default `~/.config/video-watch`) elsewhere."""
    env = os.environ if env is None else env
    home = home or Path.home()
    kind = platform_key(platform)
    if kind == "win32":
        base = env.get("APPDATA") or str(home / "AppData" / "Roaming")
    elif kind == "darwin":
        base = str(home / "Library" / "Application Support")
    else:
        xdg = env.get("XDG_CONFIG_HOME", "")
        base = xdg if os.path.isabs(xdg) else str(home / ".config")  # XDG: relative paths are ignored
    return Path(base) / APP_NAME


def user_cache_dir(platform: str | None = None, env: dict[str, str] | None = None, home: Path | None = None) -> Path:
    """The per-user cache dir of video-watch: `%LOCALAPPDATA%\\video-watch` on Windows,
    `~/Library/Caches/video-watch` on macOS, `$XDG_CACHE_HOME/video-watch` (default
    `~/.cache/video-watch`) elsewhere."""
    env = os.environ if env is None else env
    home = home or Path.home()
    kind = platform_key(platform)
    if kind == "win32":
        base = env.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
    elif kind == "darwin":
        base = str(home / "Library" / "Caches")
    else:
        xdg = env.get("XDG_CACHE_HOME", "")
        base = xdg if os.path.isabs(xdg) else str(home / ".cache")
    return Path(base) / APP_NAME


def hf_cache_dir(env: dict[str, str] | None = None) -> Path:
    """The Hugging Face hub cache that huggingface_hub uses by default."""
    env = os.environ if env is None else env
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"])
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    xdg = env.get("XDG_CACHE_HOME", "")
    return Path(xdg if os.path.isabs(xdg) else Path.home() / ".cache") / "huggingface" / "hub"


def config_path() -> Path:
    override = os.environ.get(CONFIG_ENV, "").strip()
    return Path(override).expanduser() if override else user_config_dir() / "config.json"


class ConfigError(Exception):
    """An invalid config file, environment variable or `config set` value; `hint` says how to fix it."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


def parse_bool(text: str) -> bool:
    low = str(text).strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"'{text}' is not on/off (use on, off, true, false, yes, no, 1 or 0)")


def parse_config_value(key: str, value: Any) -> Any:
    """Validate and normalize one setting (a JSON value or text from env/CLI). Raises ValueError."""
    if key not in CONFIG_KEYS:
        raise ValueError(f"unknown key '{key}' (known: {', '.join(CONFIG_KEYS)})")
    if key == "device":
        text = str(value).strip().lower()
        if text not in DEVICES:
            raise ValueError(f"device must be one of {', '.join(DEVICES)}, not '{value}'")
        return text
    if key == "model":
        text = str(value).strip()
        if text not in WHISPER_MODELS:
            raise ValueError(f"model must be one of {', '.join(MODEL_NAMES)}, not '{value}'")
        return text
    if key == "sounds":
        return value if isinstance(value, bool) else parse_bool(str(value))
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{key} needs a directory path")
    return str(Path(text).expanduser().absolute())


def load_config(path: Path | None = None) -> dict[str, Any]:
    """The validated config file (empty when it does not exist). Raises ConfigError."""
    path = path or config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"config file {path} could not be read ({exc})") from None
    try:
        data = json.loads(text) if text.strip() else {}
    except ValueError as exc:
        raise ConfigError(f"config file {path} is not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must hold a JSON object")
    out = {}
    for key, value in data.items():
        if key not in CONFIG_KEYS:
            continue  # unknown keys (e.g. from a newer version) are kept on disk but ignored
        try:
            out[key] = parse_config_value(key, value)
        except ValueError as exc:
            raise ConfigError(f"config file {path}: {exc}") from None
    return out


def save_config(updates: dict[str, Any], remove: Sequence[str] = (), path: Path | None = None) -> Path:
    """Merge `updates` into the config file (keeping unknown keys), drop `remove`, write atomically."""
    path = path or config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        raise ConfigError(f"config file {path} is not valid JSON; fix or delete it first") from None
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must hold a JSON object; fix or delete it first")
    data.update(updates)
    for key in remove:
        data.pop(key, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def default_setting(key: str) -> Any:
    return {
        "device": DEFAULT_DEVICE,
        "model": DEFAULT_MODEL,
        "model_dir": str(user_cache_dir() / "whisper"),
        "sounds": SOUND_MAP_DEFAULT,
        "sound_cache": None,  # the Hugging Face cache
        "work_dir": str(DEFAULT_WORK_ROOT),
    }[key]


def resolve_setting(key: str, args: argparse.Namespace | None = None, config: dict[str, Any] | None = None) -> tuple[Any, str]:
    """The effective value of a setting and where it came from (`flag`, `env`, `config` or
    `default`): a command-line flag wins over the environment, the environment over the config
    file, the config file over the default. Raises ConfigError for an invalid env value."""
    flag = getattr(args, key, None) if args is not None else None
    if key == "sounds" and args is not None:
        flag = True if getattr(args, "sounds", None) is True else False if getattr(args, "no_sounds", False) else None
    if flag is not None:
        return parse_config_value(key, flag), "flag"
    env_value = os.environ.get(CONFIG_ENV_VARS[key], "").strip()
    if env_value:
        try:
            return parse_config_value(key, env_value), "env"
        except ValueError as exc:
            var = CONFIG_ENV_VARS[key]
            raise ConfigError(f"{var}: {exc}", f"fix or unset the environment variable {var}; it overrides the config file") from None
    config = load_config() if config is None else config
    if key in config:
        return config[key], "config"
    return default_setting(key), "default"


def setting(key: str, args: argparse.Namespace | None = None) -> Any:
    return resolve_setting(key, args)[0]


def model_dir(args: argparse.Namespace | None = None) -> Path:
    return Path(setting("model_dir", args))


def sound_cache_dir() -> Path:
    """Where the sound model is (or will be) cached: `sound_cache`, else the Hugging Face cache."""
    value = setting("sound_cache")
    return Path(value) if value else hf_cache_dir()


# ---------------------------------------------------------------------------
# URL handling


def parse_video_id(url: str) -> str:
    """Return the 11-character video id of a single-video URL (or bare id).

    Raises ValueError for pure playlist URLs and anything that is not a YouTube video.
    """
    url = url.strip()
    if VIDEO_ID_RE.match(url):
        return url
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path_parts = [p for p in parsed.path.split("/") if p]
    query = parse_qs(parsed.query)

    candidate: str | None = None
    if host == "youtu.be":
        candidate = path_parts[0] if path_parts else None
    elif host in YOUTUBE_HOSTS:
        if path_parts[:1] == ["watch"]:
            candidate = (query.get("v") or [None])[0]
        elif len(path_parts) >= 2 and path_parts[0] in {"shorts", "live", "embed", "v", "e"}:
            candidate = path_parts[1]
        elif path_parts[:1] == ["playlist"] or ("list" in query and not query.get("v")):
            raise ValueError("Playlist URLs are not supported; pass a single video URL.")
    else:
        raise ValueError(f"Not a YouTube URL: {url}")

    if not candidate or not VIDEO_ID_RE.match(candidate):
        raise ValueError(f"Could not find a video id in URL: {url}")
    return candidate


@dataclass
class Source:
    """What a command works on: a YouTube video, a video on another yt-dlp site, or a local file."""

    kind: str  # "youtube", "site" or "file"
    key: str  # work-dir name: the YouTube id, `<extractor>-<id>` or `file-<hash>`
    url: str | None = None  # the URL yt-dlp extracts; None for local files
    path: Path | None = None  # the local file, resolved; it is only ever read


_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")
_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_DOMAIN_RE = re.compile(r"^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?:[/?#:]|$)")
_YOUTUBE_LIST_PATHS = {"playlist", "channel", "c", "user", "feed"}


def strip_input(text: str) -> str:
    """The argument without surrounding whitespace and quotes (a pasted, quoted path with spaces)."""
    return text.strip().strip("\"'").strip()


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:SHORT_HASH_LEN]


def work_key(extractor: str, video_id: str | None, url: str) -> str:
    """`<extractor>-<id>` for a non-YouTube site: lowercased, filesystem-safe on Windows and at
    most WORK_KEY_MAX_LEN characters. Without a usable id, a short hash of the URL stands in."""
    site = re.sub(r"[^a-z0-9]+", "", extractor.lower()) or "site"
    part = re.sub(r"[^A-Za-z0-9._-]+", "_", str(video_id or "")).strip("._-")
    key = f"{site}-{part or short_hash(url)}"
    if len(key) > WORK_KEY_MAX_LEN:  # keep it unique: cut and append a hash of the full key
        key = key[: WORK_KEY_MAX_LEN - SHORT_HASH_LEN - 1].rstrip("._-") + "-" + short_hash(key)
    return key


def site_key(url: str) -> str:
    """Work key of a non-YouTube URL without any network request: yt-dlp's extractor matching
    (the same first-match order yt-dlp uses) and the id in the URL. Generic pages hash the URL."""
    from yt_dlp.extractor import gen_extractor_classes

    for ie in gen_extractor_classes():
        if ie.suitable(url):
            name = ie.ie_key()
            return work_key(name, None if name == "Generic" else ie.get_temp_id(url), url)
    return work_key("generic", None, url)


def file_key(path: Path) -> str:
    """`file-<hash>` of the resolved path, size and modification time, so an edited file gets a
    fresh transcript. Case-insensitive paths on Windows map to one key."""
    st = path.stat()
    return f"file-{short_hash(f'{os.path.normcase(str(path))}|{st.st_size}|{st.st_mtime_ns}')}"


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def local_source(path: Path) -> Source:
    """A local file; it must exist and must not live inside the work root (which is purged)."""
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError:
        raise FetchError(f"local file not found: {path.expanduser().absolute()}", EXIT_NO_TRANSCRIPT) from None
    if not resolved.is_file():
        raise FetchError(f"not a file: {resolved}", EXIT_NO_TRANSCRIPT)
    if is_inside(resolved, WORK_ROOT):
        raise FetchError(f"files inside the work root ({WORK_ROOT}) are not accepted; it is purged", EXIT_NO_TRANSCRIPT)
    return Source("file", file_key(resolved), None, resolved)


def url_source(url: str) -> Source:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if host == "youtu.be" or host in YOUTUBE_HOSTS:
        try:
            video_id = parse_video_id(url)
        except ValueError as exc:
            path = urlparse(url).path.strip("/")
            listing = "list=" in url or path.split("/")[0] in _YOUTUBE_LIST_PATHS or path.startswith("@")
            raise FetchError(str(exc), EXIT_NO_TRANSCRIPT, PLAYLIST_HINT if listing else None) from None
        return Source("youtube", video_id, f"https://www.youtube.com/watch?v={video_id}")
    return Source("site", site_key(url), url)


def file_uri_path(uri: str) -> Path:
    """Local path of a `file:` URI. A host other than `localhost` is a UNC share
    (`file://server/share/a.mp4` -> `\\\\server\\share\\a.mp4`)."""
    if not uri.lower().startswith("file://"):
        return Path(url2pathname(uri[5:]))  # `file:` + a drive path, without slashes after the colon
    parsed = urlparse(uri)
    path = url2pathname(parsed.path)
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        return Path("\\\\" + parsed.netloc + "\\" + path.replace("/", "\\").lstrip("\\"))
    return Path(path)


def resolve_source(text: str) -> Source:
    """Classify a command argument without network access.

    In order: a bare 11-character YouTube id (unless a file of that name exists); a `file:` URI; an http(s) URL (YouTube forms are
    parsed exactly as in v1, anything else goes to yt-dlp); an existing local path; a path-like
    input that does not exist (an error); a schemeless domain (`vimeo.com/...`).
    Raises FetchError (exit 3) for anything else.
    """
    raw = strip_input(text)
    if not raw:
        raise FetchError("empty input; pass a video URL, a YouTube video id or a local file path", EXIT_NO_TRANSCRIPT)
    if VIDEO_ID_RE.match(raw):
        # An existing extensionless file with an id-like name is meant as the file.
        return local_source(Path(raw)) if Path(raw).is_file() else Source("youtube", raw, f"https://www.youtube.com/watch?v={raw}")
    if raw.lower().startswith("file:"):
        return local_source(file_uri_path(raw))
    scheme = _SCHEME_RE.match(raw)
    if scheme and not _DRIVE_RE.match(raw):
        if scheme.group(1).lower() not in ("http", "https"):
            raise FetchError(f"unsupported URL scheme '{scheme.group(1)}'; pass an http(s) URL or a local file path", EXIT_NO_TRANSCRIPT)
        return url_source(raw)
    path = Path(raw).expanduser()
    try:
        exists = path.exists()
    except (OSError, ValueError):  # characters such as `?` are invalid in Windows paths
        exists = False
    if exists or _DRIVE_RE.match(raw) or "\\" in raw or raw.startswith((".", "~", "/")):
        return local_source(path)
    if path.suffix.lower() in MEDIA_EXTENSIONS:
        return local_source(path)  # a missing file such as `clip.mp4`
    if _DOMAIN_RE.match(raw):
        return url_source("https://" + raw)
    raise FetchError(f"not a video URL, a YouTube video id or an existing file: {raw}", EXIT_NO_TRANSCRIPT)


# ---------------------------------------------------------------------------
# Timestamps and subtitle parsing


def format_timestamp(seconds: float, long_form: bool) -> str:
    """`mm:ss` for videos under an hour, `h:mm:ss` otherwise (long_form)."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if long_form:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m + 60 * h:02d}:{s:02d}"


def format_duration(seconds: float | None) -> str:
    if not seconds:
        return "unknown"
    return format_timestamp(seconds, seconds >= 3600)


_CUE_TIME_RE = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)
_TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]*\}")  # HTML-style tags and ASS overrides such as {\an8}


def _cue_seconds(h: str | None, m: str, s: str, frac: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(frac.ljust(3, "0")) / 1000


def parse_subtitles(text: str) -> list[Segment]:
    """Parse WebVTT or SRT text into cues; a cue's text lines are joined by newlines.

    A cue runs from its timing line to the next strictly empty line. YouTube's auto
    captions put a whitespace-only line inside cues, so such lines do not end a cue.
    """
    cues: list[Segment] = []
    current: list[str] | None = None
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = _CUE_TIME_RE.search(raw)
        if match:
            g = match.groups()
            current = []
            cues.append(Segment(_cue_seconds(*g[0:4]), _cue_seconds(*g[4:8]), ""))
        elif raw == "":
            current = None
        elif current is not None:
            clean = html.unescape(_TAG_RE.sub("", raw)).replace("\u200b", "").strip()
            if clean:
                current.append(clean)
                cues[-1].text = "\n".join(current)
    return [c for c in cues if c.text]


def clean_rolling_cues(cues: list[Segment]) -> list[Segment]:
    """Collapse YouTube auto-caption "rolling" cues into unique lines.

    Auto-generated VTT repeats the previous line above each new one and adds tiny
    transition cues; a line may also grow word by word. Each distinct line is kept once,
    starting when it first appears and ending when it was last shown.
    """
    out: list[Segment] = []
    for cue in cues:
        for line in cue.text.split("\n"):
            line = " ".join(line.split())
            if not line:
                continue
            recent = out[-3:]
            same = next((seg for seg in reversed(recent) if seg.text == line), None)
            if same is not None:
                same.end = max(same.end, cue.end)
                continue
            if out and line.startswith(out[-1].text + " "):
                out[-1].text = line
                out[-1].end = max(out[-1].end, cue.end)
                continue
            out.append(Segment(cue.start, cue.end, line))
    for prev, nxt in zip(out, out[1:]):
        if prev.end > nxt.start:
            prev.end = max(prev.start, nxt.start)
    return out


def flatten_cues(cues: list[Segment]) -> list[Segment]:
    """Turn multi-line manual cues into single-line segments."""
    return [Segment(c.start, c.end, " ".join(c.text.split())) for c in cues if c.text.strip()]


# ---------------------------------------------------------------------------
# Paragraphs


_SENTENCE_END = (".", "!", "?", "…", "。", "！", "？", '."', '?"', '!"')


def group_paragraphs(
    segments: list[Segment], min_s: float = 15.0, max_s: float = 30.0, pause_s: float = 1.0
) -> list[tuple[float, str]]:
    """Group segments into (start, text) paragraphs of roughly min_s..max_s seconds.

    A paragraph closes after min_s once a sentence ends or a pause follows; it closes
    before a segment that would push it past max_s.
    """
    paragraphs: list[tuple[float, str]] = []
    current: list[Segment] = []

    def flush() -> None:
        if current:
            text = " ".join(" ".join(s.text.split()) for s in current).strip()
            if text:
                paragraphs.append((current[0].start, text))
            current.clear()

    for i, seg in enumerate(segments):
        if current and seg.end - current[0].start > max_s:
            flush()
        current.append(seg)
        length = seg.end - current[0].start
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        pause = nxt is not None and nxt.start - seg.end >= pause_s
        if length >= min_s and (seg.text.rstrip().endswith(_SENTENCE_END) or pause):
            flush()
    flush()
    return paragraphs


# ---------------------------------------------------------------------------
# Source selection


def _lang_base(code: str) -> str:
    return re.split(r"[-_]", code.lower(), maxsplit=1)[0]


def normalize_lang(code: str | None) -> str | None:
    """Base language as ISO 639-1 where known (`eng`, `en-US` -> `en`); None when untagged."""
    base = _lang_base(code or "")
    if base in _NO_LANGUAGE:
        return None
    return _ISO639_2.get(base, base)


def lang_matches(a: str | None, b: str | None) -> bool:
    """True when both codes name the same known language (`eng` and `en-GB` match)."""
    na, nb = normalize_lang(a), normalize_lang(b)
    return na is not None and na == nb


def choose_manual_track(
    subtitles: dict[str, Any], original_lang: str | None, strict: bool = False
) -> tuple[str | None, str]:
    """Pick the uploader-provided subtitle track in the original spoken language.

    `subtitles` is yt-dlp's `subtitles` mapping (manual tracks only). Returns the track
    key (or None) and a note explaining the decision. The metadata (or --lang) must state the
    language on every site: with an unknown language a track could be a translation, so Whisper
    runs instead. strict (sites other than YouTube) also skips track names that mark machine
    captions.
    """
    tracks = [k for k in subtitles if k not in IGNORED_SUB_TRACKS]
    if strict:
        tracks = [k for k in tracks if not _AUTO_TRACK_RE.search(k)]
    if not tracks:
        return None, "no manual subtitles"
    if not normalize_lang(original_lang):
        return None, (
            f"original language unknown; manual subtitles ({', '.join(sorted(tracks))}) "
            "not used, since they could be a translation"
        )
    lang = original_lang.lower()
    exact = [k for k in tracks if k.lower() == lang]
    pick = (exact or [k for k in tracks if lang_matches(k, lang)] or [None])[0]
    if pick:
        return pick, f"manual subtitle track '{pick}' matches original language '{original_lang}'"
    return None, (
        f"manual subtitles ({', '.join(sorted(tracks))}) are not in the original "
        f"language '{original_lang}'"
    )


def choose_auto_track(automatic_captions: dict[str, Any], original_lang: str | None) -> str | None:
    """Pick YouTube's original-language ASR track (`<lang>-orig`), never a translation."""
    orig = [k for k in automatic_captions if k.endswith("-orig")]
    if original_lang:
        base = _lang_base(original_lang)
        exact = [k for k in orig if k.lower() == f"{original_lang.lower()}-orig"]
        same_base = [k for k in orig if _lang_base(k) == base]
        return (exact or same_base or [None])[0]
    return orig[0] if len(orig) == 1 else None


def _tag(tags: dict[str, Any] | None, *names: str) -> str:
    """First non-empty tag among `names` (ffprobe tag keys vary in case across containers)."""
    low = {str(k).lower(): v for k, v in (tags or {}).items()}
    for name in names:
        value = low.get(name)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _disposition(stream: dict[str, Any], name: str) -> bool:
    return bool((stream.get("disposition") or {}).get(name))


def stream_language(stream: dict[str, Any]) -> str | None:
    lang = _tag(stream.get("tags"), "language")
    return lang if normalize_lang(lang) else None


def primary_stream(streams: list[dict[str, Any]], codec_type: str) -> dict[str, Any] | None:
    """The default stream of a type (else the first); cover art is not a video stream."""
    candidates = [s for s in streams if s.get("codec_type") == codec_type and not _disposition(s, "attached_pic")]
    return min(candidates, key=lambda s: (not _disposition(s, "default"), s.get("index", 0)), default=None)


def choose_embedded_subtitle(
    streams: list[dict[str, Any]], original_lang: str | None
) -> tuple[dict[str, Any] | None, str]:
    """Pick an embedded text subtitle stream of a local file in the spoken language.

    `streams` are ffprobe streams. Only text codecs count (bitmap PGS, VobSub and DVB cannot be
    read), forced tracks are skipped (they cover only foreign-language parts), and the stream's
    language tag must match the audio's (or `--lang`). An unknown language means Whisper, so a
    translated track is never used. Returns the stream (or None) and a note.
    """
    subs = [s for s in streams if s.get("codec_type") == "subtitle"]
    if not subs:
        return None, "no embedded subtitles"
    text = [s for s in subs if s.get("codec_name") in TEXT_SUB_CODECS and not _disposition(s, "forced")]
    if not text:
        codecs = ", ".join(sorted({str(s.get("codec_name")) for s in subs}))
        return None, f"no embedded text subtitles (found {codecs}; bitmap or forced tracks are not used)"
    described = ", ".join(f"#{s.get('index')} {stream_language(s) or 'untagged'}" for s in text)
    if not normalize_lang(original_lang):
        return None, f"audio language unknown; embedded subtitles ({described}) not used, since they could be a translation"
    matches = [s for s in text if lang_matches(stream_language(s), original_lang)]
    if not matches:
        return None, f"embedded subtitles ({described}) do not match the audio language '{original_lang}'"
    pick = min(matches, key=lambda s: (not _disposition(s, "default"), s.get("index", 0)))
    return pick, (
        f"embedded subtitle stream #{pick.get('index')} ({pick.get('codec_name')}, '{stream_language(pick)}') "
        f"matches the audio language '{original_lang}'"
    )


def file_info(probe: dict[str, Any], path: Path) -> dict[str, Any]:
    """An info dict for a local file from ffprobe JSON, shaped like yt-dlp's where it overlaps
    (title, duration, description, chapters with start_time/title, language, width, height)."""
    fmt = probe.get("format") or {}
    streams = probe.get("streams") or []
    video, audio = primary_stream(streams, "video"), primary_stream(streams, "audio")
    if video is None and audio is None:
        raise FetchError(f"no video or audio stream in {path}", EXIT_NO_TRANSCRIPT)
    tags = fmt.get("tags") or {}
    try:
        duration = float(fmt.get("duration") or 0) or None
    except ValueError:
        duration = None
    chapters = []
    for ch in probe.get("chapters") or []:
        try:
            chapters.append({"start_time": float(ch.get("start_time")), "title": _tag(ch.get("tags"), "title")})
        except (TypeError, ValueError):
            continue
    width = height = None
    if video is not None:
        width, height = video.get("width"), video.get("height")
        rotation = next((sd.get("rotation") for sd in video.get("side_data_list") or [] if "rotation" in sd), None)
        rotation = rotation if rotation is not None else _tag(video.get("tags"), "rotate")
        try:
            if width and height and abs(int(float(rotation or 0))) % 180 == 90:
                width, height = height, width  # ffmpeg autorotates frames of phone videos
        except ValueError:
            pass
    return {
        "title": _tag(tags, "title") or path.name,
        "duration": duration,
        "description": _tag(tags, "description", "comment", "synopsis"),
        "chapters": chapters,
        "language": stream_language(audio) if audio is not None else None,
        "audio_index": audio.get("index") if audio is not None else None,
        "width": width,
        "height": height,
        "vcodec": video.get("codec_name") if video is not None else None,
    }


# ---------------------------------------------------------------------------
# Work directory, cache and purge


def work_dir(video_id: str) -> Path:
    return WORK_ROOT / video_id


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _locked(directory: Path) -> bool:
    try:
        return _pid_alive(int((directory / ".lock").read_text().strip()))
    except (OSError, ValueError):
        return False


def acquire_lock(directory: Path) -> None:
    """Create the dir's `.lock` exclusively with this process ID.

    A lock held by another live process stops the run with exit 1, so `fetch` and `frames`
    for the same video never delete each other's files. A stale lock is taken over.
    """
    lock = directory / ".lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _locked(directory):
                raise FetchError("another run is using this video; retry when it finishes", EXIT_ERROR, BUSY_HINT) from None
            lock.unlink(missing_ok=True)  # stale lock of a finished or crashed run
            continue
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return
    raise FetchError("another run is using this video; retry when it finishes", EXIT_ERROR, BUSY_HINT)


def release_lock(directory: Path) -> None:
    """Remove `.lock` only if this process holds it."""
    lock = directory / ".lock"
    try:
        if lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass


def remove_downloads(directory: Path, retries: int = 5) -> None:
    """Delete audio, video, partial downloads, sound-map request files and frame intermediates.
    A file still held by a just-stopped process is retried a few times."""
    for pattern in DOWNLOAD_PATTERNS:
        for path in directory.glob(pattern):
            for attempt in range(retries):
                try:
                    path.unlink()
                    break
                except FileNotFoundError:
                    break
                except PermissionError as exc:
                    if attempt + 1 == retries:
                        log(f"warning: could not delete {path}: {exc}")
                    else:
                        time.sleep(0.4)
                except OSError as exc:
                    log(f"warning: could not delete {path}: {exc}")
                    break


def is_link(path: Path) -> bool:
    """True for symlinks and other reparse points (Windows junctions). purge and clean never
    follow them, so nothing outside the work root is touched."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    reparse = getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(reparse)


def work_dirs(root: Path) -> list[Path]:
    """Real directories directly under the work root; links and junctions are skipped."""
    if not root.is_dir():
        return []
    return [d for d in root.iterdir() if not is_link(d) and d.is_dir()]


def work_root_problem(root: Path) -> str | None:
    """Why `root` cannot become the work root (read-only check), or None."""
    if (root / WORK_ROOT_MARKER).is_file() or not root.exists():
        return None
    if not root.is_dir() or is_link(root):
        return f"work dir {root} is not a plain directory"
    try:
        if any(root.iterdir()):
            return f"work dir {root} already holds other files and is not a video-watch work dir"
    except OSError as exc:
        return f"work dir {root} cannot be read ({exc})"
    return None


def prepare_work_root(root: Path | None = None) -> Path:
    """Create the work root with its marker file. An existing directory is adopted only when it
    is empty or already marked: the purge and `clean --all` delete its subdirectories, so a
    misconfigured `work_dir` (a home or project folder) must never become the work root."""
    root = root or WORK_ROOT
    marker = root / WORK_ROOT_MARKER
    if marker.is_file():
        return root
    problem = work_root_problem(root)
    if problem:
        raise FetchError(problem, EXIT_ERROR, WORK_DIR_HINT)
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text("video-watch work dir: temporary files, deleted automatically after 24 hours.\n", encoding="utf-8")
    return root


def purge(root: Path | None = None, now: float | None = None, max_age_s: float = PURGE_AGE_S) -> list[Path]:
    """Delete work dirs older than max_age_s and leftover downloads in dirs not in use.

    Returns the removed work dirs. Dirs locked by a live process are left alone, and nothing
    happens in a directory without the work-root marker.
    """
    root = root or WORK_ROOT
    if not root.is_dir() or not (root / WORK_ROOT_MARKER).is_file():
        return []
    now = time.time() if now is None else now
    removed = []
    for directory in root.iterdir():
        if not directory.is_dir() or is_link(directory) or _locked(directory):
            continue
        if now - directory.stat().st_mtime > max_age_s:
            shutil.rmtree(directory, ignore_errors=True)
            removed.append(directory)
        else:
            remove_downloads(directory)
            (directory / ".lock").unlink(missing_ok=True)
    return removed


def has_results(directory: Path) -> bool:
    """True when the dir holds a transcript, a sound map or frames worth keeping after a failed run."""
    frames = directory / "frames"
    if (directory / "meta.json").exists() or (directory / "sounds.json").exists():
        return True
    return frames.is_dir() and any(frames.glob("*.jpg"))


def cache_satisfies(meta: dict[str, Any], args: argparse.Namespace) -> bool:
    source = meta.get("source")
    settings = meta.get("settings", {})
    if source not in {"manual-subtitles", "whisper"}:
        return False  # GPU fallbacks are retried.
    if args.whisper and source != "whisper":
        return False
    if args.model and (source != "whisper" or settings.get("model") != args.model):
        return False
    if args.no_vad and (source != "whisper" or settings.get("vad_filter") is not False):
        return False
    if args.lang and normalize_lang(meta.get("language")) != normalize_lang(args.lang):
        return False
    return True


def load_cache(directory: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    needed = [directory / n for n in ("meta.json", "transcript.md", "transcript-paragraphs.md", "segments.json")]
    if not all(p.is_file() for p in needed):
        return None
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if cache_satisfies(meta, args) else None


# ---------------------------------------------------------------------------
# Output


def video_context(info: dict[str, Any]) -> dict[str, Any]:
    """The uploader's description (verbatim; YouTube caps it at 5000 chars) and chapters
    (`start` seconds, `title`) from yt-dlp's info dict. Empty values when there are none."""
    description = (info.get("description") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    chapters = []
    for ch in info.get("chapters") or []:
        title = " ".join(str(ch.get("title") or "").split())
        if ch.get("start_time") is not None and title:
            chapters.append({"start": round(float(ch["start_time"]), 3), "title": title})
    return {"description": description, "chapters": sorted(chapters, key=lambda c: c["start"])}


def has_context(meta: dict[str, Any]) -> bool:
    """False for a work dir cached before description and chapters were stored."""
    return "description" in meta and "chapters" in meta


def chapter_lines(chapters: list[dict[str, Any]], long_form: bool) -> list[str]:
    return [f"- [{format_timestamp(c['start'], long_form)}] {c['title']}" for c in chapters]


def header_lines(meta: dict[str, Any], long_form: bool = False, description: bool = True) -> list[str]:
    """The header block of both transcript files.

    The description goes only into transcript.md (description=False for the paragraphs file,
    the compact one to save); chapters go into both. Empty sections are omitted.
    """
    source = meta["source"]
    s = meta.get("settings", {})
    vad_off = source == "whisper" and not s.get("vad_filter")
    if source == "whisper":
        if s.get("vad_filter"):
            vad = f"VAD on (min silence {s.get('vad_min_silence_ms')} ms)"
        else:
            vad = "VAD off (auto-retry)" if s.get("vad_auto_retry") else "VAD off"
        mode = f"batched x{s.get('batch_size')}" if s.get("batched") else "sequential"
        cpu = ", cpu" if s.get("device") == "cpu" else ""
        source_desc = (
            f"whisper ({s.get('model')}, {s.get('compute_type')}{cpu}, {vad}, beam {s.get('beam_size')}, {mode})"
        )
    else:
        source_desc = f"{source} (track {s.get('track')})"
    upload = meta.get("upload_date") or ""
    if re.fullmatch(r"\d{8}", upload):
        upload = f"{upload[:4]}-{upload[4:6]}-{upload[6:]}"
    lines = [
        f"# {meta.get('title') or meta['video_id']}",
        "",
        f"- Channel: {meta.get('channel') or 'unknown'}",
        f"- File: {meta['source_path']}" if meta.get("source_path") else f"- URL: {meta['url']}",
        f"- Uploaded: {upload or 'unknown'}",
        f"- Duration: {format_duration(meta.get('duration'))}",
        f"- Language: {meta.get('language') or 'unknown'}",
        f"- Source: {source_desc}",
        *([f"- Note: {VAD_OFF_NOTE}"] if vad_off else []),
        f"- Generated: {meta['generated_at']}",
        "",
    ]
    if description and meta.get("description"):
        lines += ["## Description", "", meta["description"], ""]
    if meta.get("chapters"):
        lines += ["## Chapters", "", *chapter_lines(meta["chapters"], long_form), ""]
    return lines + ["---", ""]


def write_transcripts(directory: Path, meta: dict[str, Any], segments: list[Segment]) -> None:
    """Write transcript.md, transcript-paragraphs.md and meta.json (not segments.json)."""
    long_form = (meta.get("duration") or (segments[-1].end if segments else 0)) >= 3600
    lines = [f"[{format_timestamp(s.start, long_form)}] {s.text}" for s in segments]
    head = header_lines(meta, long_form)
    (directory / "transcript.md").write_text("\n".join(head + lines) + "\n", encoding="utf-8")

    paras = [f"[{format_timestamp(t, long_form)}] {text}\n" for t, text in group_paragraphs(segments)]
    head = header_lines(meta, long_form, description=False)
    (directory / "transcript-paragraphs.md").write_text("\n".join(head) + "\n".join(paras), encoding="utf-8")
    (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_segments(directory: Path) -> list[Segment] | None:
    """Segments from the work dir's segments.json, or None when there is no readable one."""
    try:
        data = json.loads((directory / "segments.json").read_text(encoding="utf-8"))
        return [Segment(float(s["start"]), float(s["end"]), s["text"], s.get("words")) for s in data["segments"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_outputs(directory: Path, meta: dict[str, Any], segments: list[Segment]) -> None:

    data = {
        "video_id": meta["video_id"],
        "source": meta["source"],
        "language": meta.get("language"),
        "word_timestamps": any(s.words for s in segments),
        "segments": [
            {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
            | ({"words": s.words} if s.words else {})
            for s in segments
        ],
    }
    (directory / "segments.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    write_transcripts(directory, meta, segments)


def print_summary(directory: Path, meta: dict[str, Any], cached: bool, extra_notes: list[str] | None = None) -> None:
    transcript = directory / "transcript.md"
    out = [
        f"SOURCE: {meta['source']}",
        f"TITLE: {meta.get('title') or ''}",
        f"CHANNEL: {meta.get('channel') or ''}",
        f"DURATION: {format_duration(meta.get('duration'))}",
        f"LANGUAGE: {meta.get('language') or 'unknown'}",
        f"TRANSCRIPT: {transcript.resolve()}",
        f"PARAGRAPHS: {(directory / 'transcript-paragraphs.md').resolve()}",
        f"SEGMENTS_JSON: {(directory / 'segments.json').resolve()}",
        f"CHARS: {len(transcript.read_text(encoding='utf-8'))}",
        f"CHAPTERS: {len(meta.get('chapters') or [])}",
    ]
    if meta["source"] == "whisper":
        segments = load_segments(directory) or []
        audio_s = meta.get("duration") or (segments[-1].end if segments else 0)
        out.append(speech_line(segments, audio_s, audio_s >= 3600))
    sounds = load_sounds(directory)
    if sounds:
        out += sound_lines(directory, sounds)
    out.append(f"CACHED: {'yes' if cached else 'no'}")
    out += [f"NOTE: {n}" for n in [*meta.get("notes", []), *(extra_notes or [])]]
    print("\n".join(out), flush=True)


def versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for pkg in ("yt-dlp", "yt-dlp-ejs", "faster-whisper", "ctranslate2", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"):
        try:
            result[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


# ---------------------------------------------------------------------------
# yt-dlp


class _StderrLogger:
    """Route yt-dlp output to stderr so stdout stays machine-readable."""

    def debug(self, msg: str) -> None:
        if not msg.startswith("[debug] "):
            log(msg)

    info = debug

    def warning(self, msg: str) -> None:
        log(f"WARNING: {msg}")

    def error(self, msg: str) -> None:
        log(msg)


def ydl_options(args: argparse.Namespace, directory: Path) -> dict[str, Any]:
    last_pct = {"value": -10}

    def progress(d: dict[str, Any]) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                pct = int(100 * d.get("downloaded_bytes", 0) / total)
                if pct >= last_pct["value"] + 10:
                    last_pct["value"] = pct
                    log(f"download: {pct}% of {total / 1e6:.1f} MB")

    opts: dict[str, Any] = {
        "logger": _StderrLogger(),
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
        # A URL that still resolves to a playlist is rejected, so its entries are not resolved
        # one by one; two entries are enough to tell a one-video playlist from a real one.
        "extract_flat": "in_playlist",
        "playlistend": 2,
        # Audio for Whisper. Sites without audio-only formats (TikTok, X) get a small muxed format.
        "format": "bestaudio/best[height<=480]/best",
        "outtmpl": str(directory / "audio.%(ext)s"),
        "progress_hooks": [progress],
        # YouTube needs a JS runtime for its player challenges; yt-dlp only enables deno by
        # default, so allow node as well (the challenge solver ships as yt-dlp-ejs).
        "js_runtimes": {"deno": {}, "node": {}},
        "retries": 3,
        "color": "no_color",
    }
    if args.cookies_from_browser:
        name, _, profile = args.cookies_from_browser.partition(":")
        opts["cookiesfrombrowser"] = (name.lower(), profile or None, None, None)
    return opts


_DRM_PATTERNS = ("drm protected", "drm-protected", "protected by drm")
# Login walls on any site: yt-dlp's own login hint ("Use --cookies-from-browser or --cookies ...",
# "Use --username and --password ..."), YouTube's bot-check, age and members-only gates, and the
# wording Instagram, X and Facebook use.
_COOKIE_PATTERNS = (
    "not a bot",
    "sign in to confirm",
    "confirm your age",
    "use --cookies",
    "cookies-from-browser",
    "use --username",
    "login required",
    "log in to",
    "you need to log in",
    "requires authentication",
    "registered users",
    "members-only",
    "join this channel",
    "private video",
)
_LIVE_PATTERNS = (
    "live event will begin",
    "premieres in",
    "premiere will begin",
    "not currently live",
    "live stream recording is not available",
)
_UNAVAILABLE_PATTERNS = (
    "video unavailable",
    "video is unavailable",
    "has been removed",
    "does not exist",
    "is not available",
    "account associated with this video has been terminated",
)
_NO_VIDEO_PATTERNS = ("unsupported url", "no video could be found", "there's no video", "no video in this", "no media found")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LIVE_REASONS = {
    "is_live": "it is live right now",
    "is_upcoming": "it has not started yet",
    "post_live": "the stream has just ended and is still being processed",
}


def classify_ytdlp_error(message: str, used_cookies: bool) -> FetchError:
    message = _ANSI_RE.sub("", message).strip().removeprefix("ERROR:").strip()
    low = message.lower()
    if any(p in low for p in _DRM_PATTERNS):
        return FetchError(f"DRM-protected content is not supported ({message})", EXIT_NO_TRANSCRIPT)
    if any(p in low for p in _COOKIE_PATTERNS):
        return FetchError(message, EXIT_NEEDS_COOKIES, COOKIES_USED_HINT if used_cookies else COOKIES_HINT)
    if any(p in low for p in _LIVE_PATTERNS):
        return FetchError(f"live or upcoming stream is not supported ({message})", EXIT_NO_TRANSCRIPT)
    if any(p in low for p in _UNAVAILABLE_PATTERNS):
        return FetchError(message, EXIT_NO_TRANSCRIPT)
    if any(p in low for p in _NO_VIDEO_PATTERNS):
        return FetchError(
            f"no video found at this URL ({message})",
            EXIT_NO_TRANSCRIPT,
            f"check the link; if the site should be supported, try: {UPGRADE_HINT}",
        )
    return FetchError(message, EXIT_ERROR, f"yt-dlp may be outdated; try: {UPGRADE_HINT}")


def check_playable(info: dict[str, Any]) -> None:
    """Reject content that is live now, upcoming or still processing, and DRM-only content, on
    every site. Finished, processed recordings of live streams are ordinary videos."""
    status = info.get("live_status") or ("is_live" if info.get("is_live") else None)
    if status in LIVE_REASONS:
        raise FetchError(
            f"live status '{status}' is not supported ({LIVE_REASONS[status]}); "
            "finished, processed recordings work: retry after it has ended and been processed",
            EXIT_NO_TRANSCRIPT,
        )
    formats = info.get("formats") or []
    if info.get("_has_drm") and not [f for f in formats if not f.get("has_drm")]:
        raise FetchError("DRM-protected content is not supported", EXIT_NO_TRANSCRIPT)


def extract_info(ydl: Any, url: str, used_cookies: bool) -> dict[str, Any]:
    from yt_dlp.utils import DownloadError

    try:
        info = ydl.extract_info(url, download=False)
        if info and info.get("_type") in PLAYLIST_TYPES:
            entries = list(info.get("entries") or [])
            # A one-entry "playlist" (some posts wrap their only video) is that video.
            if len(entries) == 1 and entries[0].get("url") and (info.get("playlist_count") or 1) <= 1:
                info = ydl.extract_info(entries[0]["url"], ie_key=entries[0].get("ie_key"), download=False)
    except DownloadError as exc:
        raise classify_ytdlp_error(str(exc), used_cookies) from None
    if not info:
        raise FetchError("yt-dlp returned no metadata", EXIT_ERROR, f"try: {UPGRADE_HINT}")
    if info.get("_type") in PLAYLIST_TYPES:
        raise FetchError("URL resolved to a playlist; pass a single video URL", EXIT_NO_TRANSCRIPT, PLAYLIST_HINT)
    check_playable(info)
    return info


def has_audio(info: dict[str, Any]) -> bool:
    """False only when every format states it has no audio (silent clips, e.g. some reels)."""
    formats = info.get("formats") or []
    return not formats or any(f.get("acodec") != "none" for f in formats)


def download_subtitle(ydl: Any, formats: list[dict[str, Any]]) -> list[Segment]:
    by_ext = {f.get("ext"): f for f in formats if f.get("url") or f.get("data")}
    fmt = by_ext.get("vtt") or by_ext.get("srt")
    if fmt is None:
        raise FetchError(f"no VTT/SRT subtitle format among {sorted(map(str, by_ext))}")
    if fmt.get("data"):  # some sites embed the subtitle text in the metadata
        return parse_subtitles(fmt["data"])
    with ydl.urlopen(fmt["url"]) as response:
        text = response.read().decode("utf-8", errors="replace")
    return parse_subtitles(text)


# ---------------------------------------------------------------------------
# Whisper


def nvidia_lib_dirs() -> list[str]:
    """Directories of the pip-installed CUDA libraries (nvidia-cublas-cu12, nvidia-cudnn-cu12) in
    this environment: `nvidia/*/bin` (DLLs) on Windows, `nvidia/*/lib` (shared objects) on Linux."""
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    sub = "bin" if os.name == "nt" else "lib"
    return sorted(d for root in spec.submodule_search_locations for d in glob.glob(os.path.join(root, "*", sub)))


def setup_cuda_libs() -> None:
    """Make the pip-installed CUDA DLLs loadable by CTranslate2 on Windows. On Linux the dynamic
    loader reads LD_LIBRARY_PATH only at process start, so main() re-executes the script with
    these directories on it (cuda_reexec_env); macOS has no CUDA."""
    if os.name != "nt":
        return
    for bin_dir in nvidia_lib_dirs():
        os.add_dll_directory(bin_dir)
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


# Linux: set when main() re-executed itself with the CUDA library dirs on LD_LIBRARY_PATH; holds
# the original value so the sound helper (its own PyTorch CUDA libraries) gets that one back.
REEXEC_ENV = "VIDEO_WATCH_REEXEC_LD_LIBRARY_PATH"


def cuda_reexec_env(environ: dict[str, str] | None = None, lib_dirs: list[str] | None = None, platform: str | None = None) -> dict[str, str] | None:
    """The environment to re-execute with on Linux so CTranslate2 finds cuBLAS and cuDNN from the
    nvidia-* wheels (the approach faster-whisper documents), or None when nothing is needed:
    not Linux, no wheel libraries, already on LD_LIBRARY_PATH, or already re-executed."""
    environ = dict(os.environ if environ is None else environ)
    if platform_key(platform) != "linux" or REEXEC_ENV in environ:
        return None
    dirs = nvidia_lib_dirs() if lib_dirs is None else lib_dirs
    current = environ.get("LD_LIBRARY_PATH", "")
    have = [p for p in current.split(":") if p]
    if not dirs or all(d in have for d in dirs):
        return None
    environ[REEXEC_ENV] = current
    environ["LD_LIBRARY_PATH"] = ":".join([*dirs, *have])
    return environ


_GPU_ERROR_PATTERNS = (
    "cuda",
    "cudnn",
    "cublas",
    "out of memory",
    "no cuda",
    "cannot be loaded",
    "is not found or cannot",
    ".dll",
)
# GPU errors that mean "no usable CUDA here" (no device, driver or libraries) rather than a busy
# or full GPU: retrying later cannot help, CPU mode can.
_NO_CUDA_PATTERNS = ("no cuda", "driver", "cannot be loaded", "is not found", "cannot open shared object", ".dll", "no kernel image")


def is_gpu_error(exc: BaseException) -> bool:
    low = str(exc).lower()
    return any(p in low for p in _GPU_ERROR_PATTERNS)


def gpu_unavailable(exc: BaseException) -> GpuUnavailable:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    low = message.lower()
    no_cuda = "out of memory" not in low and any(p in low for p in _NO_CUDA_PATTERNS)
    return GpuUnavailable(message, no_cuda=no_cuda)


def model_cache_path(name: str, root: Path) -> Path:
    """The Hugging Face cache folder of a Whisper model under the model dir."""
    return root / ("models--" + WHISPER_MODELS[name]["repo"].replace("/", "--"))


def model_cached(name: str, root: Path) -> bool:
    """True when a complete snapshot (with model.bin) of the model is in the model dir."""
    snapshots = model_cache_path(name, root) / "snapshots"
    try:
        return any((d / "model.bin").is_file() for d in snapshots.iterdir())
    except OSError:
        return False


def cpu_threads(platform: str | None = None, machine: str | None = None, logical: int | None = None) -> int:
    """CTranslate2 threads on the CPU, at least 1 and at most 16: every core on Apple Silicon (no
    simultaneous multithreading), otherwise half the logical CPUs (x86 CPUs usually run two
    threads per core, and hyper-threads add nothing to int8 matrix work)."""
    import platform as platform_mod

    logical = logical or os.cpu_count() or 2
    machine = (machine or platform_mod.machine()).lower()
    if platform_key(platform) == "darwin" and machine in ("arm64", "aarch64"):
        return max(1, min(16, logical))
    return max(1, min(16, logical // 2))


def resolve_device(args: argparse.Namespace | None = None) -> str:
    """`cpu` only when the user chose it; `auto` and `cuda` both mean the GPU (never a silent CPU run)."""
    return "cpu" if setting("device", args) == "cpu" else "cuda"


def load_whisper(model_name: str, device: str = "cuda", root: Path | None = None) -> Any:
    """The Whisper model on the GPU (float16) or, only when the user chose it, the CPU (int8).
    A GPU problem raises GpuUnavailable; there is no CPU fallback. Cached models load without a
    network round trip."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    root = root or model_dir()
    local = model_cached(model_name, root)
    if device == "cpu":
        from faster_whisper import WhisperModel

        return WhisperModel(
            model_name, device="cpu", compute_type=CPU_COMPUTE_TYPE, cpu_threads=cpu_threads(),
            download_root=str(root), local_files_only=local,
        )
    setup_cuda_libs()
    try:
        import ctranslate2
        from faster_whisper import WhisperModel

        if ctranslate2.get_cuda_device_count() < 1:
            raise GpuUnavailable("no CUDA device visible to CTranslate2", no_cuda=True)
        return WhisperModel(
            model_name, device="cuda", compute_type=COMPUTE_TYPE, download_root=str(root), local_files_only=local
        )
    except GpuUnavailable:
        raise
    except (RuntimeError, OSError, ValueError) as exc:
        if is_gpu_error(exc):
            raise gpu_unavailable(exc) from None
        raise


def whisper_language(code: str | None) -> str | None:
    if not code:
        return None
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    base = normalize_lang(code)  # local files tag streams with ISO 639-2 (`eng`)
    return base if base in _LANGUAGE_CODES else None


def transcribe(
    model: Any, audio: Path, language: str | None, vad: bool, duration: float | None, batch_size: int = BATCH_SIZE
) -> tuple[list[Segment], Any, dict[str, Any]]:
    from faster_whisper import BatchedInferencePipeline

    common = dict(
        language=language,
        beam_size=BEAM_SIZE,
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    if vad:
        # The batched pipeline needs VAD segmentation; timestamps are re-enabled so each
        # VAD chunk (up to 30 s) is split into Whisper's natural sub-segments.
        pipeline = BatchedInferencePipeline(model)
        segments_iter, info = pipeline.transcribe(
            str(audio),
            batch_size=batch_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": VAD_MIN_SILENCE_MS},
            without_timestamps=False,
            **common,
        )
        settings = {"batched": True, "batch_size": batch_size}
    else:
        # Without VAD the batched pipeline would need fixed 30 s clips that cut words, so
        # fall back to the sequential decoder (slower, same model and settings).
        segments_iter, info = model.transcribe(str(audio), vad_filter=False, **common)
        settings = {"batched": False}

    total = duration or info.duration or 0
    segments: list[Segment] = []
    next_report = 10
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            words = [
                {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word, "p": round(w.probability, 3)}
                for w in (seg.words or [])
            ]
            segments.append(Segment(seg.start, seg.end, text, words or None))
        if total and 100 * seg.end / total >= next_report:
            log(f"transcribe: {int(100 * seg.end / total)}% ({format_duration(seg.end)} / {format_duration(total)})")
            next_report = (int(100 * seg.end / total) // 10 + 1) * 10
    return segments, info, settings


def count_words(segments: list[Segment]) -> int:
    """Words in the segments: Whisper's word timestamps when present (they also split scripts
    written without spaces), otherwise whitespace-separated tokens."""
    return sum(len(s.words) if s.words else len(s.text.split()) for s in segments)


def longest_gap(segments: list[Segment], audio_s: float | None) -> tuple[float, float]:
    """(start, end) of the longest stretch without transcribed speech, counting the stretch before
    the first segment and, when the audio length is known, the one after the last."""
    best = (0.0, 0.0)
    covered = 0.0  # end of speech so far; segments may overlap
    for s in sorted(segments, key=lambda s: s.start):
        if s.start - covered > best[1] - best[0]:
            best = (covered, s.start)
        covered = max(covered, s.end)
    if audio_s and audio_s - covered > best[1] - best[0]:
        best = (covered, audio_s)
    return best


def speech_line(segments: list[Segment], audio_s: float | None, long_form: bool) -> str:
    """`SPEECH:` hint for Whisper transcripts: words per minute of audio and the longest gap
    without speech. A long gap can be speech or singing that VAD dropped (the assistant decides
    whether a `--no-vad` rerun is worth it); a uniform average would hide it."""
    wpm = count_words(segments) / (audio_s / 60) if audio_s else 0.0
    start, end = longest_gap(segments, audio_s)
    span = f"{format_timestamp(start, long_form)}-{format_timestamp(end, long_form)}"
    return f"SPEECH: {wpm:.0f} words/min; longest gap {format_timestamp(end - start, long_form)} ({span})"


def nearly_empty(segments: list[Segment], audio_s: float | None) -> bool:
    """True when a VAD run found (almost) no speech: no segments, or fewer than
    AUTO_RETRY_MAX_WPM words per minute of audio. Unknown length: only an empty run counts."""
    if not segments:
        return True
    if not audio_s:
        return False
    return count_words(segments) / (audio_s / 60) < AUTO_RETRY_MAX_WPM


def prefer_retry(first: list[Segment], retry: list[Segment]) -> bool:
    """Whether the no-VAD retry replaces the VAD result: yes unless it found fewer words, so an
    empty retry never replaces a non-empty VAD result."""
    return count_words(retry) >= count_words(first)


def _transcribe_on_device(
    model: Any, audio: Path, language: str | None, vad: bool, duration: float | None, batch_size: int, device: str
) -> tuple[list[Segment], Any, dict[str, Any]]:
    """transcribe(); on the GPU, CUDA errors and running out of memory become GpuUnavailable."""
    if device == "cpu":
        return transcribe(model, audio, language, vad, duration, batch_size)
    try:
        return transcribe(model, audio, language, vad, duration, batch_size)
    except (RuntimeError, OSError, MemoryError) as exc:
        if is_gpu_error(exc) or isinstance(exc, MemoryError):
            raise gpu_unavailable(exc) from None
        raise


def model_vram_mb(name: str, batch: int) -> int:
    """Peak VRAM (MB) a GPU run of the model needs at this batch size (batch 1 also stands for the
    sequential decoder of --no-vad); sizes between the measured ones are interpolated."""
    table = WHISPER_VRAM_MB.get(name) or WHISPER_VRAM_MB[DEFAULT_MODEL]
    if batch in table:
        return table[batch]
    sizes = sorted(table)
    lo = max([b for b in sizes if b <= batch], default=sizes[0])
    hi = min([b for b in sizes if b >= batch], default=sizes[-1])
    if lo == hi:
        return table[lo]
    return round(table[lo] + (table[hi] - table[lo]) * (batch - lo) / (hi - lo))


def choose_batch_size(name: str, free_mb: int | None) -> int:
    """The batch size for a batched GPU run: the largest of 8, 4, 2 and 1 whose measured peak VRAM
    plus a safety margin fits the free VRAM, so 6-8 GB cards run large-v3 at a smaller batch
    (slower, same transcript). Unknown free VRAM (no nvidia-smi) keeps the default of 8."""
    if free_mb is None:
        return BATCH_SIZE
    for batch in BATCH_CHOICES:
        if model_vram_mb(name, batch) + VRAM_MARGIN_MB <= free_mb:
            return batch
    return BATCH_CHOICES[-1]


def whisper_estimate_min(
    duration_s: float | None, no_vad: bool = False, model: str = DEFAULT_MODEL, device: str = "cuda", batch: int = BATCH_SIZE
) -> float:
    """Expected wall-clock minutes of a Whisper fetch (without a first-time model download)."""
    if device == "cpu":
        factor = CPU_RTF.get(model, CPU_RTF[DEFAULT_MODEL]) * (CPU_SEQUENTIAL_SLOWDOWN if no_vad else 1)
    else:
        slowdown = BATCH_SLOWDOWN.get(batch, 1.0) if model in BATCH_SENSITIVE_MODELS else 1.0
        speed = GPU_SPEED.get(model, 1.0) * (SEQUENTIAL_SLOWDOWN if no_vad else slowdown)
        factor = ESTIMATE_FACTOR * speed
    return ESTIMATE_OVERHEAD_MIN + factor * (duration_s or 0) / 60


def download_audio(ydl: Any, info: dict[str, Any], directory: Path, args: argparse.Namespace) -> Path:
    """Download the audio of a site video with yt-dlp into the work dir."""
    try:
        result = ydl.process_ie_result(dict(info), download=True)
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises many types
        from yt_dlp.utils import DownloadError

        if isinstance(exc, DownloadError):
            raise classify_ytdlp_error(str(exc), bool(args.cookies_from_browser)) from None
        raise
    downloads = (result or {}).get("requested_downloads") or []
    audio = Path(downloads[0]["filepath"]) if downloads else next(iter(directory.glob("audio.*")), None)
    if audio is None or not audio.is_file():
        raise FetchError("audio download produced no file", EXIT_ERROR, f"try: {UPGRADE_HINT}")
    return audio


def run_whisper(
    ydl: Any,
    info: dict[str, Any],
    directory: Path,
    args: argparse.Namespace,
    language: str | None,
    get_audio: Callable[[], Path] | None = None,
    keep_audio: bool = False,
) -> Result:
    """Transcribe with Whisper. The audio comes from yt-dlp (`ydl`), or from `get_audio` for a
    local file; either way it lands in the work dir and is deleted right after transcription,
    unless `keep_audio` (the sound map runs next on the same download; fetch deletes it after).
    The model is released before returning, so the sound model never shares the GPU with it."""
    model_name = setting("model", args)
    device = resolve_device(args)
    root = model_dir(args)
    duration = info.get("duration") or 0
    batch = CPU_BATCH_SIZE if device == "cpu" else choose_batch_size(model_name, gpu_free_mb())
    estimate = whisper_estimate_min(duration, args.no_vad, model_name, device, batch)
    first_use = not model_cached(model_name, root)
    size_gb = WHISPER_MODELS[model_name]["size_mb"] / 1000
    extra = f" incl. first-time model download (~{size_gb:.1f} GB)" if first_use else ""
    estimate += MODEL_DOWNLOAD_EXTRA_MIN * min(1.0, size_gb / 3) if first_use else 0
    where = "on the CPU, slow" if device == "cpu" else f"batch {batch}"
    log(f"ESTIMATE: ~{estimate:.1f} min for {format_duration(duration)} of audio (whisper {model_name}, {where}{extra})")

    t0 = time.monotonic()
    model = load_whisper(model_name, device, root)  # raises GpuUnavailable before any audio is downloaded
    t_load = time.monotonic() - t0
    log(f"model {model_name} loaded in {t_load:.1f}s")

    try:
        t1 = time.monotonic()
        audio = get_audio() if get_audio else download_audio(ydl, info, directory, args)
        t_download = time.monotonic() - t1
        log(f"audio ready in {t_download:.1f}s ({audio.stat().st_size / 1e6:.1f} MB)")

        t2 = time.monotonic()
        lang = whisper_language(args.lang or language)
        vad = not args.no_vad
        segments, winfo, mode = _transcribe_on_device(model, audio, lang, vad, duration, batch, device)
        t_transcribe = time.monotonic() - t2

        # VAD drops singing over music: retry once without it when the run found almost nothing.
        # Only for the default VAD run; an explicit --no-vad run is never retried.
        notes = []
        retried, t_retry = False, 0.0
        audio_s = duration or winfo.duration or 0
        if vad and nearly_empty(segments, audio_s):
            retried = True
            found = f"VAD found little speech ({count_words(segments)} words in {format_duration(audio_s)})"
            log(f"{found}; retrying with VAD off")
            retry_min = whisper_estimate_min(audio_s, True, model_name, device) - ESTIMATE_OVERHEAD_MIN
            log(f"ESTIMATE: ~{retry_min:.1f} min for the no-VAD retry of {format_duration(audio_s)} of audio (whisper {model_name}, sequential)")
            t3 = time.monotonic()
            try:
                retry = _transcribe_on_device(model, audio, lang, False, duration, batch, device)
            except GpuUnavailable as exc:
                if not segments:
                    raise  # nothing to keep: the usual GPU fallback applies
                retry = None
                notes.append(f"{found}; the retry with VAD off failed on the GPU ({exc}); kept the VAD result")
            t_retry = time.monotonic() - t3
            if retry is not None and prefer_retry(segments, retry[0]):
                segments, winfo, mode = retry
                vad = False
                notes.append(f"{found}; retried with VAD off")
            elif retry is not None:
                notes.append(f"{found}; retried with VAD off, which found less; kept the VAD result")
            log(f"no-VAD retry in {t_retry:.1f}s")
            t_transcribe += t_retry  # transcribe_s and the RTF cover both runs
    finally:
        del model  # free the GPU before anything else (the sound map) runs
        gc.collect()
        if not keep_audio:
            remove_downloads(directory)

    if not lang:
        notes.append(f"language auto-detected as '{winfo.language}' (p={winfo.language_probability:.2f})")
    if device == "cpu":
        notes.append(f"transcribed on the CPU ({CPU_COMPUTE_TYPE}, device cpu): slower than a GPU, same model")
    settings = {
        "model": model_name,
        "device": device,
        "compute_type": CPU_COMPUTE_TYPE if device == "cpu" else COMPUTE_TYPE,
        "beam_size": BEAM_SIZE,
        "vad_filter": vad,
        "vad_min_silence_ms": VAD_MIN_SILENCE_MS if vad else None,
        "vad_auto_retry": retried,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "language_forced": lang,
        **mode,
    }
    rtf = t_transcribe / duration if duration else None
    timings = {
        "model_load_s": round(t_load, 2),
        "download_s": round(t_download, 2),
        "transcribe_s": round(t_transcribe, 2),
        "rtf": round(rtf, 4) if rtf else None,
    } | ({"retry_transcribe_s": round(t_retry, 2)} if retried else {})
    log(f"transcribed in {t_transcribe:.1f}s" + (f" (RTF {rtf:.3f})" if rtf else ""))
    return Result("whisper", winfo.language, segments, notes, settings, timings)


# ---------------------------------------------------------------------------
# Sound map: pure logic (chunk plan, stitching, spans, summaries). The model runs in
# sound_events.py; tracks here are per-frame logits in hundredths (int16 from the backend).


def chunk_plan(total_frames: int, chunk_frames: int, overlap_frames: int) -> list[tuple[int, int]]:
    """(start_frame, n_frames) chunks covering the audio: equal-length chunks overlapping by
    `overlap_frames`, the last one aligned to the end (so every chunk has the same length and
    can be batched). Audio shorter than one chunk is one shorter chunk."""
    if total_frames <= 0:
        return []
    if total_frames <= chunk_frames:
        return [(0, total_frames)]
    hop = max(1, chunk_frames - overlap_frames)
    starts = list(range(0, total_frames - chunk_frames, hop)) + [total_frames - chunk_frames]
    return [(s, chunk_frames) for s in sorted(set(starts))]


def segment_plan(
    total_frames: int, silences: Sequence[tuple[float, float]], chunk_frames: int, overlap_frames: int,
    min_silence_s: float = SOUND_SILENCE_SPLIT_S,
) -> list[tuple[int, int]]:
    """The chunk plan without long silences: each stretch between silences of at least
    `min_silence_s` is chunked on its own (chunk_plan), so no chunk mixes sound with a long
    silence. Frames inside the silences get no chunk (stitch_tracks fills them as unscored)."""
    cuts = sorted(
        (math.ceil(a * SOUND_FRAME_RATE), math.floor(b * SOUND_FRAME_RATE))
        for a, b in silences if b - a >= min_silence_s
    )
    plan: list[tuple[int, int]] = []
    pos = 0
    for a, b in [*cuts, (total_frames, total_frames)]:
        a, b = max(pos, min(a, total_frames)), max(pos, min(b, total_frames))
        plan += [(pos + s, n) for s, n in chunk_plan(a - pos, chunk_frames, overlap_frames)]
        pos = b
    return plan


def stitch_tracks(chunks: list[tuple[int, Sequence[int]]], total_frames: int, fill: int = SOUND_FILL) -> array:
    """One track (int16 array) from overlapping chunks: inside an overlap, frames before its
    midpoint come from the earlier chunk and the rest from the later one, so every frame is scored
    with context on both sides and nothing is counted twice. A sound that crosses a chunk boundary
    is therefore one continuous run. Frames no chunk covers get `fill`."""
    out = array("h", [fill]) * total_frames
    chunks = sorted(chunks, key=lambda c: c[0])
    for i, (start, values) in enumerate(chunks):
        end = start + len(values)
        lo = start if i == 0 else max(start, (start + chunks[i - 1][0] + len(chunks[i - 1][1])) // 2)
        hi = end if i + 1 == len(chunks) else min(end, (chunks[i + 1][0] + end) // 2)
        lo, hi = max(0, lo), min(total_frames, hi)
        if hi > lo:
            part = values[lo - start : hi - start]
            out[lo:hi] = part if isinstance(part, array) and part.typecode == "h" else array("h", part)
    return out


def silent_runs(levels: Sequence[int], floor_db: float = SOUND_SILENCE_DBFS) -> list[tuple[int, int]]:
    """Frame runs [start, end) whose level (dBFS in hundredths, from the backend) is below `floor_db`."""
    q = floor_db * SOUND_LOGIT_SCALE
    runs: list[list[int]] = []
    for i, v in enumerate(levels):
        if v < q:
            if runs and runs[-1][1] == i:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
    return [(a, b) for a, b in runs]


def silence_gate(tracks: dict[str, Sequence[int]], levels: Sequence[int] | None) -> dict[str, Sequence[int]]:
    """Mute every prompt where the audio is (digitally) silent. The model scores pure silence as
    music, wind or engines; nothing can be heard below SOUND_SILENCE_DBFS, so those frames get
    the lowest score before any span is found. Without levels (another backend), unchanged."""
    runs = silent_runs(levels) if levels is not None else []
    if not runs:
        return tracks
    out: dict[str, Sequence[int]] = {}
    for label, track in tracks.items():
        muted = array("h", track)
        for a, b in runs:
            b = min(b, len(muted))
            if b > a:
                muted[a:b] = array("h", [SOUND_FILL]) * (b - a)
        out[label] = muted
    return out


def second_medians(track: Sequence[int], frame_rate: float = SOUND_FRAME_RATE) -> list[float]:
    """Median of each whole second of a track (the last second may be partial)."""
    fr = round(frame_rate)
    out = []
    for i in range(0, len(track), fr):
        part = sorted(track[i : i + fr])
        n = len(part)
        out.append(part[n // 2] if n % 2 else (part[n // 2 - 1] + part[n // 2]) / 2)
    return out


def rolling_median(values: Sequence[float], half_window: int) -> list[float]:
    """Median of each value's centred window of 2 * half_window + 1 values (shorter at the ends)."""
    out = []
    for i in range(len(values)):
        part = sorted(values[max(0, i - half_window) : i + half_window + 1])
        n = len(part)
        out.append(part[n // 2] if n % 2 else (part[n // 2 - 1] + part[n // 2]) / 2)
    return out


def music_gate(music_track: Sequence[int]) -> list[bool]:
    """Per second: does the model hear music (its music logit, as a SOUND_MUSIC_GATE_S median,
    at or above SOUND_MUSIC_GATE)? Under music many event prompts score high throughout, so
    events there must also stand out from their own rolling median."""
    smooth = rolling_median(second_medians(music_track), SOUND_MUSIC_GATE_S // 2)
    return [v >= SOUND_MUSIC_GATE * SOUND_LOGIT_SCALE for v in smooth]


def detect_spans(
    track: Sequence[int],
    threshold: float,
    min_frames: int,
    gap_frames: int,
    frame_rate: float = SOUND_FRAME_RATE,
    hysteresis: float = SOUND_HYSTERESIS,
    baseline: Sequence[float] | None = None,
    margin: float = 0.0,
    gate: Sequence[bool] | None = None,
) -> list[tuple[float, float, float]]:
    """Spans (start_s, end_s, peak logit) of one prompt's logit track (hundredths).

    - A frame is active when its logit reaches `threshold - hysteresis`; a run of active frames
      counts only when one of its frames reaches `threshold` (hysteresis: a sound is not cut
      into pieces where its score dips briefly).
    - With a `baseline` (per-second rolling median), frames in seconds where `gate` is true
      (music is heard) must also exceed the baseline by `margin` (by `margin - hysteresis` to
      stay active). Without a gate the margin applies everywhere.
    - Runs at most `gap_frames` apart are merged, then spans shorter than `min_frames` are
      dropped (merging first, so a burst of short hits such as clapping survives as one span).
    """
    fr = round(frame_rate)
    hi_q, lo_q = threshold * SOUND_LOGIT_SCALE, (threshold - hysteresis) * SOUND_LOGIT_SCALE
    hi_m, lo_m = margin * SOUND_LOGIT_SCALE, (margin - hysteresis) * SOUND_LOGIT_SCALE

    def stands_out(i: int, v: int, need: float) -> bool:
        if baseline is None:
            return True
        sec = min(i // fr, len(baseline) - 1)
        if gate is not None and not gate[min(sec, len(gate) - 1)]:
            return True
        return v - baseline[sec] >= need

    runs: list[list[int]] = []
    for sec_start in range(0, len(track), fr):
        part = track[sec_start : sec_start + fr]
        if max(part) < lo_q:  # most seconds of most prompts: skip without a Python loop
            continue
        for i, v in enumerate(part, sec_start):
            if v >= lo_q and stands_out(i, v, lo_m):
                if runs and runs[-1][1] == i:
                    runs[-1][1] = i + 1
                else:
                    runs.append([i, i + 1])
    kept = [r for r in runs if any(track[i] >= hi_q and stands_out(i, track[i], hi_m) for i in range(r[0], r[1]))]
    merged: list[list[int]] = []
    for a, b in kept:
        if merged and a - merged[-1][1] <= gap_frames:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [
        (round(a / frame_rate, 2), round(b / frame_rate, 2), round(max(track[a:b]) / SOUND_LOGIT_SCALE, 2))
        for a, b in merged
        if b - a >= min_frames
    ]


def event_spans(
    label: str, track: Sequence[int], gate: Sequence[bool] | None, threshold: float = SOUND_THRESHOLD
) -> list[dict[str, Any]]:
    """Spans of a discrete-event prompt: threshold plus, under music, the rolling-median margin."""
    base = rolling_median(second_medians(track), SOUND_BASELINE_WINDOW_S // 2)
    spans = detect_spans(
        track, threshold, round(SOUND_EVENT_MIN_S * SOUND_FRAME_RATE), round(SOUND_EVENT_GAP_S * SOUND_FRAME_RATE),
        baseline=base, margin=SOUND_MUSIC_MARGIN, gate=gate,
    )
    return [{"label": label, "start": a, "end": b, "score": p} for a, b, p in spans]


def coverage_spans(label: str, track: Sequence[int], threshold: float = SOUND_THRESHOLD) -> list[dict[str, Any]]:
    """Spans of a coverage label (music, speech, singing): the threshold alone (no music margin:
    music must not be measured against itself), longer minimum and gap."""
    spans = detect_spans(
        track, threshold, round(SOUND_COVERAGE_MIN_S * SOUND_FRAME_RATE), round(SOUND_COVERAGE_GAP_S * SOUND_FRAME_RATE)
    )
    return [{"label": label, "start": a, "end": b, "score": p} for a, b, p in spans]


def prune_colabels(spans: list[dict[str, Any]], margin: float = SOUND_COLABEL_MARGIN, top: int = SOUND_COLABEL_TOP) -> list[dict[str, Any]]:
    """Drop look-alike labels that fire together with a clearer one (waves also scoring as water,
    wind and rain). A span stays when no overlapping span of another label peaks more than
    `margin` above it and its label is among the `top` best labels overlapping it.

    A sweep in start order: the spans overlapping one are those still open when it starts plus
    those starting before it ends, so long event-heavy audio stays fast."""
    order = sorted(range(len(spans)), key=lambda i: (spans[i]["start"], i))
    starts = [spans[i]["start"] for i in order]
    open_spans: list[tuple[float, int]] = []  # heap of (end, index) of spans that started earlier
    keep: set[int] = set()
    for pos, i in enumerate(order):
        s = spans[i]
        while open_spans and open_spans[0][0] <= s["start"]:
            heapq.heappop(open_spans)
        overlapping = [j for _, j in open_spans]
        overlapping += order[pos + 1 : bisect.bisect_left(starts, s["end"], lo=pos + 1)]
        best = {s["label"]: s["score"]}
        for j in overlapping:
            o = spans[j]
            best[o["label"]] = max(best.get(o["label"], -1e9), o["score"])
        ranked = sorted(best, key=lambda label: (-best[label], label))
        if s["score"] >= best[ranked[0]] - margin and s["label"] in ranked[:top]:
            keep.add(i)
        heapq.heappush(open_spans, (s["end"], i))
    return [s for i, s in enumerate(spans) if i in keep]


def sound_spans(
    tracks: dict[str, Sequence[int]],
    threshold: float = SOUND_THRESHOLD,
    levels: Sequence[int] | None = None,
    duration: float | None = None,
) -> list[dict[str, Any]]:
    """All spans of a sound map, sorted by start: silence muted first; coverage labels (music,
    speech, singing) with the threshold alone and longer minimum and gap; the rest as discrete
    events, pruned of look-alike co-labels. Ends are clamped to the audio's `duration`."""
    tracks = silence_gate(tracks, levels)
    gate = music_gate(tracks["music"]) if "music" in tracks else None
    cover: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for label, track in tracks.items():
        if label in SOUND_COVERAGE_LABELS:
            cover += coverage_spans(label, track, threshold)
        else:
            events += event_spans(label, track, gate, threshold)
    spans = cover + prune_colabels(events)
    if duration:
        spans = [s | {"end": min(s["end"], round(duration, 2))} for s in spans if s["start"] < duration]
    order = {label: i for i, label in enumerate(tracks)}
    return sorted(spans, key=lambda s: (s["start"], order.get(s["label"], 0)))


def coverage(spans: list[dict[str, Any]], duration: float) -> dict[str, float]:
    """Fraction of the audio covered by each coverage label (music, speech, singing)."""
    out = {}
    for label in SOUND_COVERAGE_LABELS:
        covered = sum(s["end"] - s["start"] for s in spans if s["label"] == label)
        out[label] = round(min(1.0, covered / duration), 3) if duration else 0.0
    return out


def event_counts(spans: list[dict[str, Any]]) -> dict[str, int]:
    """Spans per discrete-event label, most frequent first (ties by label)."""
    counts: dict[str, int] = {}
    for s in spans:
        if s["label"] not in SOUND_COVERAGE_LABELS:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def sounds_summary(cover: dict[str, float], counts: dict[str, int], limit: int = SOUND_SUMMARY_MAX_EVENTS) -> str:
    """One line: coverage percentages, then the most frequent events with their counts."""
    head = ", ".join(f"{label} {round(100 * cover.get(label, 0))}%" for label in SOUND_COVERAGE_LABELS)
    if not counts:
        return f"{head}; no other sounds detected"
    shown = list(counts.items())[:limit]
    more = len(counts) - len(shown)
    tail = ", ".join(f"{label} {n}" for label, n in shown) + (f" (+{more} more labels)" if more else "")
    return f"{head}; {tail}"


def span_stamp(start: float, end: float, long_form: bool) -> str:
    """`[mm:ss-mm:ss]`, or one stamp when both fall in the same second (`h:mm:ss` from one hour)."""
    a, b = format_timestamp(start, long_form), format_timestamp(end, long_form)
    return f"[{a}]" if a == b else f"[{a}-{b}]"


def timeline_lines(spans: list[dict[str, Any]], long_form: bool, limit: int = SOUND_TIMELINE_MAX) -> tuple[list[str], int]:
    """`[mm:ss-mm:ss] label, label` lines sorted by time; labels with the same rendered span share a
    line. Speech is left out (the transcript covers it). Above `limit` spans, the lowest-scoring
    ones are dropped; returns the lines and the number dropped."""
    shown = [s for s in spans if s["label"] not in SOUND_TIMELINE_EXCLUDED]
    dropped = 0
    if len(shown) > limit:
        keep = sorted(shown, key=lambda s: -s["score"])[:limit]
        dropped = len(shown) - limit
        keep_ids = {id(s) for s in keep}
        shown = [s for s in shown if id(s) in keep_ids]
    lines: list[str] = []
    labels: dict[str, list[str]] = {}
    for s in shown:  # spans are sorted by start
        stamp = span_stamp(s["start"], s["end"], long_form)
        if stamp not in labels:
            labels[stamp] = []
            lines.append(stamp)
        if s["label"] not in labels[stamp]:
            labels[stamp].append(s["label"])
    return [f"{stamp} {', '.join(labels[stamp])}" for stamp in lines], dropped


_EBUR_RE = re.compile(r"\bt:\s*(-?[\d.]+)\s+TARGET:.*?\bM:\s*(-?[\d.]+|-?inf|nan)\s+S:\s*(-?[\d.]+|-?inf|nan)")
_EBUR_I_RE = re.compile(r"Integrated loudness:\s*I:\s*(-?[\d.]+)\s*LUFS", re.DOTALL)
_EBUR_LRA_RE = re.compile(r"Loudness range:\s*LRA:\s*(-?[\d.]+)\s*LU", re.DOTALL)
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


def parse_ebur128(text: str) -> tuple[list[tuple[float, float]], float | None, float | None]:
    """(t, short-term loudness) samples from ffmpeg `ebur128` frame lines, plus the summary's
    integrated loudness (LUFS) and loudness range (LU). Silent samples (-inf, below -70) are dropped."""
    samples = []
    for line in text.splitlines():
        m = _EBUR_RE.search(line)
        if m:
            try:
                st = float(m.group(3))
            except ValueError:
                continue
            if math.isfinite(st) and st > -70:
                samples.append((float(m.group(1)), st))
    summary = text[text.rfind("Summary:") :] if "Summary:" in text else ""
    i = _EBUR_I_RE.search(summary)
    lra = _EBUR_LRA_RE.search(summary)
    return samples, float(i.group(1)) if i else None, float(lra.group(1)) if lra else None


def parse_silencedetect(text: str, duration: float | None) -> list[tuple[float, float]]:
    """(start, end) silent stretches from ffmpeg `silencedetect` lines; one still open at the end
    of the audio ends there."""
    out: list[tuple[float, float]] = []
    start: float | None = None
    for line in text.splitlines():
        m = _SILENCE_START_RE.search(line)
        if m:
            start = max(0.0, float(m.group(1)))
            continue
        m = _SILENCE_END_RE.search(line)
        if m and start is not None:
            out.append((round(start, 2), round(float(m.group(1)), 2)))
            start = None
    if start is not None and duration and duration > start:
        out.append((round(start, 2), round(duration, 2)))
    return out


def loudness_peaks(samples: list[tuple[float, float]], count: int, min_separation: float) -> list[tuple[float, float]]:
    """The `count` loudest moments, at least `min_separation` s apart, in time order. A sample
    at t is the short-term loudness of the 3 s ending at t, so its moment is the window centre."""
    picked: list[tuple[float, float]] = []
    for t, value in sorted(samples, key=lambda s: (-s[1], s[0])):
        moment = max(0.0, t - LOUD_WINDOW_S / 2)
        if all(abs(moment - p[0]) >= min_separation for p in picked):
            picked.append((round(moment, 1), value))
            if len(picked) == count:
                break
    return sorted(picked)


def peak_separation(duration: float | None) -> float:
    """Minimum distance between reported loud moments: a tenth of the video, 5 to 60 s."""
    return min(LOUD_PEAK_MAX_SEP_S, max(LOUD_PEAK_MIN_SEP_S, (duration or 0) / 10))


def loudness_lines(loud: dict[str, Any] | None, long_form: bool) -> list[str]:
    """The Loudness section of sounds.md."""
    if not loud:
        return ["- Not measured."]
    lines = []
    if loud.get("integrated_lufs") is not None:
        lra = f"; loudness range {loud['range_lu']:g} LU" if loud.get("range_lu") is not None else ""
        lines.append(f"- Integrated loudness: {loud['integrated_lufs']:g} LUFS{lra}")
    peaks = loud.get("peaks") or []
    if peaks:
        text = ", ".join(f"[{format_timestamp(p['time'], long_form)}] {p['lufs']:g} LUFS" for p in peaks)
        lines.append(f"- Loudest moments (short-term loudness, 3 s): {text}")
    silences = loud.get("silences") or []
    rule = f"below {SILENCE_THRESHOLD_DB:g} dB for {SILENCE_MIN_S:g} s or more"
    if silences:
        text = ", ".join(span_stamp(s["start"], s["end"], long_form) for s in silences[:SILENCE_MAX_LISTED])
        more = len(silences) - SILENCE_MAX_LISTED
        lines.append(f"- Silences ({rule}): {text}" + (f" (+{more} more in sounds.json)" if more > 0 else ""))
    else:
        lines.append(f"- Silences ({rule}): none")
    return lines


def sounds_markdown(data: dict[str, Any], title: str | None) -> str:
    """sounds.md: header, summary, timeline and loudness."""
    duration = data.get("duration") or 0
    long_form = duration >= 3600
    timeline, dropped = timeline_lines(data["spans"], long_form)
    counts = data.get("counts") or {}
    cover = data.get("coverage") or {}
    lines = [
        f"# Sound map: {title or 'untitled'}",
        "",
        (
            f"- Model: {data['model']} (logit threshold {data['threshold']:g}; under music an event must also rise "
            f"{SOUND_MUSIC_MARGIN:g} above its {SOUND_BASELINE_WINDOW_S} s median)"
        ),
        f"- Duration: {format_duration(duration)}",
        (
            "- Labels name sounds the model hears (acoustic cues such as laughter or crying), not emotions or "
            "intentions. They are English; scores are model logits (higher is surer)."
        ),
        f"- Generated: {data['generated_at']}",
        "",
        "## Summary",
        "",
        "- Coverage: " + ", ".join(f"{label} {round(100 * cover.get(label, 0))}%" for label in SOUND_COVERAGE_LABELS),
        "- Events: " + (", ".join(f"{label} {n}" for label, n in counts.items()) if counts else "none detected"),
        "",
        "## Timeline",
        "",
        *(timeline or ["(no sounds detected)"]),
    ]
    if dropped:
        lines += ["", f"({dropped} lower-scoring spans are left out here; sounds.json has all of them.)"]
    lines += ["", "## Loudness", "", *loudness_lines(data.get("loudness"), long_form), ""]
    return "\n".join(lines)


def heard_labels(spans: list[dict[str, Any]], t: float, window: float = HEARD_WINDOW_S) -> list[str]:
    """Labels heard at moment t: spans containing it or within `window` s, speech left out.
    Discrete events come first (highest score first), then music and singing; at most
    HEARD_MAX_LABELS."""
    best: dict[str, float] = {}
    for s in spans:
        if s["label"] in SOUND_TIMELINE_EXCLUDED:
            continue
        if s["start"] - window <= t <= s["end"] + window:
            best[s["label"]] = max(best.get(s["label"], -1e9), s["score"])
    ranked = sorted(best.items(), key=lambda kv: (kv[0] in SOUND_COVERAGE_LABELS, -kv[1], kv[0]))
    return [label for label, _ in ranked][:HEARD_MAX_LABELS]


def heard_lines(spans: list[dict[str, Any]] | None, moments: list[float], long_form: bool) -> list[str]:
    """`HEARD: [stamp] label, label` for each moment with sounds; the stamp is the frame's moment."""
    lines = []
    for t in moments:
        labels = heard_labels(spans or [], t)
        if labels:
            lines.append(f"HEARD: [{format_moment(t, long_form)}] {', '.join(labels)}")
    return lines


def parse_query(text: str) -> list[str]:
    """Comma-separated prompts of `sounds --query`, trimmed and deduplicated in order."""
    prompts = list(dict.fromkeys(" ".join(p.split()) for p in text.split(",") if p.strip()))
    if not prompts:
        raise ValueError("--query needs at least one sound description")
    if len(prompts) > SOUND_QUERY_MAX:
        raise ValueError(f"--query takes at most {SOUND_QUERY_MAX} sound descriptions")
    return prompts


def event_lines(spans: list[dict[str, Any]], long_form: bool) -> list[str]:
    """`EVENT: [start-end] label score` lines in time order (tenths when a moment has them)."""
    return [
        f"EVENT: [{format_moment(s['start'], long_form)}-{format_moment(s['end'], long_form)}] {s['label']} {s['score']:.1f}"
        for s in sorted(spans, key=lambda s: (s["start"], s["label"]))
    ]


def query_miss_note(prompt: str, track: Sequence[int], threshold: float, long_form: bool) -> str:
    """NOTE for a query prompt without any span: its strongest moment, as weak evidence."""
    if not track:
        return f"'{prompt}': no audio frames"
    i = max(range(len(track)), key=track.__getitem__)
    return (
        f"'{prompt}' not found (threshold {threshold:g}, plus the margin under music); strongest at "
        f"[{format_moment(round(i / SOUND_FRAME_RATE, 1), long_form)}] with score {track[i] / SOUND_LOGIT_SCALE:.1f}"
    )


def parse_range(text: str) -> tuple[float, float]:
    """`START-END` with moments as for --at (`1:00-1:10`, `[01:00]-[01:10]`, an en dash works too)."""
    parts = re.split(r"\s*[-\u2013]\s*", text.strip(), maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"invalid range '{text.strip()}'; use START-END, e.g. 1:00-1:10")
    start, end = parse_moment(parts[0]), parse_moment(parts[1])
    if end <= start:
        raise ValueError(f"invalid range '{text.strip()}'; the end must be after the start")
    return start, end


def range_times(start: float, end: float, fps: float) -> list[float]:
    """Moments from start to end (inclusive when it falls on the grid) every 1/fps s, in tenths."""
    count = math.floor((end - start) * fps + 1e-6) + 1
    return sorted({round(start + k / fps, 1) for k in range(count)})


# ---------------------------------------------------------------------------
# Sound map: audio, loudness, the model subprocess and the output files


class SoundUnavailable(Exception):
    """The sound map could not be built. fetch reports it as a NOTE and keeps its exit code;
    `permanent` marks a reason that a later fetch cannot fix (no audio at all)."""

    def __init__(self, message: str, gpu: bool = False, permanent: bool = False, setup: bool = False, no_gpu: bool = False):
        super().__init__(message)
        self.gpu = gpu
        self.permanent = permanent
        self.setup = setup  # not set up yet: `setup --sounds on` downloads what it needs
        self.no_gpu = no_gpu  # no usable GPU here (hidden or absent): retrying later cannot help


def sound_version() -> str:
    """Fingerprint of everything that shapes a sound map; a cached map with another one is rebuilt."""
    parts = [
        SOUND_MODEL, SOUND_PROMPTS, SOUND_COVERAGE_LABELS, SOUND_TIMELINE_EXCLUDED, SOUND_THRESHOLD,
        SOUND_HYSTERESIS, SOUND_MUSIC_MARGIN, SOUND_BASELINE_WINDOW_S, SOUND_MUSIC_GATE, SOUND_MUSIC_GATE_S,
        SOUND_EVENT_MIN_S, SOUND_EVENT_GAP_S, SOUND_COVERAGE_MIN_S, SOUND_COVERAGE_GAP_S, SOUND_COLABEL_MARGIN,
        SOUND_COLABEL_TOP, SOUND_CHUNK_S, SOUND_OVERLAP_S, SOUND_SILENCE_DBFS, SOUND_SILENCE_SPLIT_S,
    ]
    return short_hash(json.dumps(parts))


def sounds_wanted(args: argparse.Namespace) -> bool:
    """Whether fetch builds the sound map: `--no-sounds` never, `--sounds` always, otherwise the
    `sounds` setting (VIDEO_WATCH_SOUNDS, then the config file; off by default)."""
    if getattr(args, "no_sounds", False):
        return False
    if getattr(args, "sounds", None) is True:
        return True
    try:
        return bool(setting("sounds"))
    except ConfigError:
        return False  # main() already reported an invalid setting before any command ran


def find_uv() -> str | None:
    """The uv executable: `uv run` exports its own path as UV; otherwise PATH."""
    return os.environ.get("UV") or shutil.which("uv")


def sound_helper_marker() -> Path:
    """Written after `setup --sounds on` prepared the helper's environment (PyTorch, transformers)."""
    return user_cache_dir() / "sound-helper.json"


def sound_helper_fingerprint() -> str:
    """The helper's dependency header and model: a new header means a new environment to install."""
    try:
        text = SOUND_HELPER.read_text(encoding="utf-8")
    except OSError:
        text = ""
    header = text.split("# ///", 2)[1] if text.count("# ///") >= 2 else text
    return short_hash(header + SOUND_MODEL)


def sound_helper_ready() -> bool:
    """True when setup prepared the helper environment for the current helper header. (uv may
    still have pruned its cache since; then the next run installs it again.)"""
    try:
        data = json.loads(sound_helper_marker().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("fingerprint") == sound_helper_fingerprint()


def mark_sound_helper_ready() -> None:
    marker = sound_helper_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"fingerprint": sound_helper_fingerprint(), "model": SOUND_MODEL}) + "\n", encoding="utf-8")


def sound_setup_done() -> bool:
    """The sound map is set up: its environment prepared and its model cached."""
    return sound_helper_ready() and sound_model_cached()


def sound_model_cached() -> bool:
    """Whether the sound model is in its cache (`sound_cache`, else the Hugging Face cache): a
    snapshot with its weights. Without it the sound map is not set up (`setup --sounds on`)."""
    try:
        return sound_model_cached_in(sound_cache_dir())
    except ConfigError:
        return False


def sound_setup_size_gb() -> float:
    return (SOUND_ENV_SIZE_MB + SOUND_MODEL_SIZE_MB) / 1000


def sound_estimate_min(duration_s: float | None, first_run: bool = False) -> float:
    minutes = (SOUND_ESTIMATE_OVERHEAD_S + SOUND_ESTIMATE_PER_AUDIO_S * (duration_s or 0)) / 60
    return minutes + (SOUND_FIRST_RUN_EXTRA_MIN if first_run else 0)


def nvidia_smi_query(fields: str) -> list[str] | None:
    """One `nvidia-smi --query-gpu` row (the comma-separated `fields`) of the GPU CUDA would use
    (the first of CUDA_VISIBLE_DEVICES, else GPU 0); None without nvidia-smi or when it fails."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    cmd = [smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    if visible.isdigit():
        cmd += ["-i", visible]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (proc.stdout or "").strip().splitlines() if proc.returncode == 0 else []
    return [v.strip() for v in lines[0].split(",")] if lines else None


def gpu_free_mb() -> int | None:
    """Free VRAM of the GPU the model would use, from nvidia-smi; None when it cannot be read
    (then the model run itself decides)."""
    row = nvidia_smi_query("memory.free")
    try:
        return int(row[0]) if row else None
    except ValueError:
        return None


def cuda_visible_devices_hides_gpu(value: str | None = None) -> str | None:
    """Why CUDA_VISIBLE_DEVICES hides every GPU from CUDA (empty, -1 or not an index or GPU id),
    or None. nvidia-smi still lists the GPU then, but no CUDA program can use it."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES") if value is None else value
    if value is None:
        return None
    first = value.split(",")[0].strip()
    if first.isdigit() or first.startswith(("GPU-", "MIG-")):
        return None
    return f"CUDA_VISIBLE_DEVICES={value!r} hides the GPU from CUDA"


def check_gpu_free() -> None:
    """Fail fast (before any audio is fetched): without an NVIDIA GPU, before `setup --sounds on`
    has downloaded the model (a fetch never starts a 5 GB download unasked), or with too little
    free VRAM."""
    if not shutil.which("nvidia-smi"):
        raise SoundUnavailable("the sound map needs an NVIDIA GPU (nvidia-smi was not found)")
    hidden = cuda_visible_devices_hides_gpu()
    if hidden:
        raise SoundUnavailable(f"no usable GPU: {hidden}", no_gpu=True)
    if not sound_setup_done():
        raise SoundUnavailable(
            f"the sound map is not set up yet (setup --sounds on downloads about {sound_setup_size_gb():.0f} GB once)",
            setup=True,
        )
    free = gpu_free_mb()
    if free is not None and free < SOUND_MIN_FREE_VRAM_MB:
        raise SoundUnavailable(f"the GPU is busy ({free} MB of VRAM free, about {SOUND_MIN_FREE_VRAM_MB} MB needed)", gpu=True)


def to_sound_pcm(ffmpeg: str, src: Path, directory: Path, stream_index: int | None = None) -> Path:
    """Decode audio (a download in the work dir, or a local original that is only read) into the
    raw 48 kHz mono 16-bit PCM the sound model reads, in the work dir. Raw PCM has no header, so
    there is no 4 GiB limit (a WAV of this format would end at about 12.4 hours)."""
    out = _work_output(directory, "audio.48k.pcm", src)
    pick = ["-map", f"0:{stream_index}"] if stream_index is not None else ["-map", "0:a:0"]
    cmd = [
        "-i", str(src), *pick, "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SOUND_SAMPLE_RATE),
        "-c:a", "pcm_s16le", "-f", "s16le", str(out),
    ]
    try:
        proc = run_ffmpeg(ffmpeg, cmd, timeout=3 * 3600)
    except subprocess.TimeoutExpired:
        raise SoundUnavailable("ffmpeg timed out converting the audio") from None
    if proc.returncode != 0 or not out.is_file():
        lines = proc.stderr.strip().splitlines()
        raise SoundUnavailable(f"ffmpeg could not convert the audio ({lines[-1] if lines else proc.returncode})")
    return out


def pcm_duration(pcm: Path) -> float:
    """Seconds of 48 kHz mono 16-bit raw PCM."""
    return pcm.stat().st_size / 2 / SOUND_SAMPLE_RATE


def analyze_loudness(ffmpeg: str, pcm: Path, duration: float) -> tuple[dict[str, Any] | None, list[str]]:
    """Integrated loudness, loudness range, loudest moments and long silences (ffmpeg ebur128 and
    silencedetect in one pass over the raw PCM). A failure only loses this section, with a note."""
    af = f"ebur128=framelog=info,silencedetect=noise={SILENCE_THRESHOLD_DB:g}dB:d={SILENCE_MIN_S:g}"
    raw = ["-f", "s16le", "-ar", str(SOUND_SAMPLE_RATE), "-ac", "1"]
    try:
        proc = run_ffmpeg(ffmpeg, [*raw, "-i", str(pcm), "-af", af, "-f", "null", "-"], timeout=3 * 3600, loglevel="info")
    except subprocess.TimeoutExpired:
        return None, ["loudness not measured: ffmpeg timed out"]
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        return None, [f"loudness not measured: ffmpeg failed ({lines[-1] if lines else proc.returncode})"]
    samples, integrated, lra = parse_ebur128(proc.stderr)
    peaks = loudness_peaks(samples, LOUD_PEAK_COUNT, peak_separation(duration))
    return {
        "integrated_lufs": integrated,
        "range_lu": lra,
        "peaks": [{"time": t, "lufs": v} for t, v in peaks],
        "silences": [{"start": a, "end": b} for a, b in parse_silencedetect(proc.stderr, duration)],
    }, []


def silent_stretches(ffmpeg: str, pcm: Path, duration: float) -> list[tuple[float, float]]:
    """Silences (below SOUND_SILENCE_DBFS for SOUND_SILENCE_SPLIT_S or more) of the raw PCM, to keep
    them out of the model's input. A failure only means no split (the frame gate still mutes)."""
    af = f"silencedetect=noise={SOUND_SILENCE_DBFS:g}dB:d={SOUND_SILENCE_SPLIT_S:g}"
    raw = ["-f", "s16le", "-ar", str(SOUND_SAMPLE_RATE), "-ac", "1"]
    try:
        proc = run_ffmpeg(ffmpeg, [*raw, "-i", str(pcm), "-af", af, "-f", "null", "-"], timeout=3 * 3600, loglevel="info")
    except subprocess.TimeoutExpired:
        return []
    return parse_silencedetect(proc.stderr, duration) if proc.returncode == 0 else []


def decode_logits(text: str) -> array:
    """A backend track (logits or levels): base64 of little-endian int16 hundredths."""
    values = array("h")
    values.frombytes(base64.b64decode(text))
    if sys.byteorder == "big":
        values.byteswap()
    return values


def helper_env() -> dict[str, str]:
    """Environment for the helper: the current one without the Whisper environment's traces
    (VIRTUAL_ENV, its Scripts or bin directory, the CUDA DLL directories added to PATH, on Linux
    the CUDA library dirs put on LD_LIBRARY_PATH), plus the configured sound-model cache."""
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("VIRTUAL_ENV", REEXEC_ENV)}
    venv_bin = os.path.normcase(str(Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")))
    path = [p for p in PATH_AT_START.split(os.pathsep) if p and os.path.normcase(p) != venv_bin]
    env["PATH"] = os.pathsep.join(path)
    if REEXEC_ENV in os.environ:  # PyTorch brings its own CUDA libraries
        original = os.environ[REEXEC_ENV]
        if original:
            env["LD_LIBRARY_PATH"] = original
        else:
            env.pop("LD_LIBRARY_PATH", None)
    try:
        cache = setting("sound_cache")
    except ConfigError:
        cache = None
    if cache:
        env[SOUND_CACHE_ENV] = str(cache)
    return env


class _KillOnCloseJob:
    """Windows Job Object that kills every process in it when terminated or closed, so a timed-out
    helper never survives as uv's grandchild (the python process holding VRAM and the audio)."""

    def __init__(self) -> None:
        self.handle = None
        if os.name != "nt":
            return
        k32 = ctypes.windll.kernel32
        k32.CreateJobObjectW.restype = ctypes.c_void_p
        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            return

        class _Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32),
            ]

        class _Io(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class _Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _Basic), ("IoInfo", _Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = _Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        if k32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.handle = handle
        else:
            k32.CloseHandle(ctypes.c_void_p(handle))

    def add(self, proc: subprocess.Popen) -> bool:
        if not self.handle:
            return False
        k32 = ctypes.windll.kernel32
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        return bool(k32.AssignProcessToJobObject(self.handle, ctypes.c_void_p(int(proc._handle))))

    def close(self) -> None:
        """Closing the last handle kills whatever is still running in the job."""
        if self.handle:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self.handle))
            self.handle = None


def kill_tree(proc: subprocess.Popen, job: _KillOnCloseJob | None = None) -> None:
    """Stop the helper and everything it started (uv, the venv trampoline, python): on Windows
    the Job Object and taskkill /T, on POSIX the helper's own process group (it runs in a new
    session, so its group id is its pid)."""
    if os.name == "nt":
        if job is not None and job.handle:
            ctypes.windll.kernel32.TerminateJobObject(ctypes.c_void_p(job.handle), 1)
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        try:
            os.killpg(proc.pid, getattr(signal, "SIGKILL", 9))
        except OSError:
            pass
    try:
        proc.kill()
        proc.wait(timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pass


def helper_popen_options(os_name: str | None = None) -> dict[str, Any]:
    """Popen options that give the helper its own process tree: a new session (and process group)
    on POSIX, so kill_tree can stop the group; Windows uses a Job Object instead."""
    return {} if (os_name or os.name) == "nt" else {"start_new_session": True}


def run_helper(cmd: list[str], env: dict[str, str], timeout: float) -> int:
    """Run the helper command in its own process tree; on timeout kill the whole tree and raise
    SoundUnavailable. Returns the exit code. The helper also stops by itself when this process
    is gone (it watches the parent pid) or its deadline has passed."""
    job = _KillOnCloseJob()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, env=env, **helper_popen_options())
    job.add(proc)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc, job)
        raise SoundUnavailable(f"the sound model timed out after {timeout / 60:.0f} min and was stopped") from None
    except BaseException:
        kill_tree(proc, job)  # Ctrl+C or any error: never leave the model running
        raise
    finally:
        job.close()


def run_sound_backend(
    pcm: Path, prompts: list[str], directory: Path, silences: Sequence[tuple[float, float]] = ()
) -> tuple[dict[str, array], array | None, dict[str, Any]]:
    """Run sound_events.py (`uv run`, its own PyTorch environment) on 48 kHz raw PCM and return one
    stitched logit track per prompt, the frame levels (dBFS hundredths, None if the backend gives
    none) and the response metadata. Request and response travel as JSON files in the work dir; a
    request id ties them together, so a late response of a killed run is never taken for this
    one. Long `silences` are left out of the model's input. Raises SoundUnavailable."""
    uv = find_uv()
    if not uv:
        raise SoundUnavailable("uv was not found on PATH")
    samples = pcm.stat().st_size // 2
    total = math.ceil(samples / SOUND_HOP)
    if total == 0:
        raise SoundUnavailable("the audio is empty", permanent=True)
    plan = segment_plan(total, silences, round(SOUND_CHUNK_S * SOUND_FRAME_RATE), round(SOUND_OVERLAP_S * SOUND_FRAME_RATE))
    if not plan:  # all silent: nothing to hear, no model run
        return {p: stitch_tracks([], total) for p in prompts}, None, {"timings": {}, "silent": True}
    request, response = directory / "sound-request.json", directory / "sound-response.json"
    response.unlink(missing_ok=True)
    request_id = os.urandom(8).hex()
    timeout = SOUND_TIMEOUT_S + samples / SOUND_SAMPLE_RATE / 2
    request.write_text(json.dumps({
        "request_id": request_id, "audio": str(pcm.resolve()), "format": "s16le", "sample_rate": SOUND_SAMPLE_RATE,
        "prompts": prompts, "model": SOUND_MODEL, "chunks": plan, "batch_size": SOUND_BATCH_SIZE,
        "out": str(response.resolve()), "parent_pid": os.getpid(), "deadline": time.time() + timeout,
    }), encoding="utf-8")
    cmd = [uv, "run", "--quiet", "--script", str(SOUND_HELPER), str(request.resolve())]
    try:
        code = run_helper(cmd, helper_env(), timeout)
    except OSError as exc:
        raise SoundUnavailable(f"uv could not start the sound model ({exc})") from None
    try:
        resp = json.loads(response.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SoundUnavailable(f"the sound model gave no result (exit {code}; see stderr)") from None
    if not isinstance(resp, dict) or resp.get("request_id") != request_id:
        raise SoundUnavailable("the sound model answered another request (a stale response)")
    if not resp.get("ok"):
        message = resp.get("message") or "unknown error"
        low = message.lower()
        if any(p in low for p in ("invalid device id", "no cuda device", "no cuda gpus", "cuda driver initialization failed")):
            raise SoundUnavailable(f"no usable GPU ({message})", no_gpu=True)
        if resp.get("kind") == "gpu":
            raise SoundUnavailable(f"the GPU is busy or unavailable ({message})", gpu=True)
        if resp.get("kind") == "model":
            raise SoundUnavailable(f"the sound model could not be loaded ({message})")
        raise SoundUnavailable(f"the sound model failed ({message})")
    try:
        chunks = resp["chunks"]
        tracks = {p: stitch_tracks([(c["start"], decode_logits(c["logits"][i])) for c in chunks], total) for i, p in enumerate(resp["prompts"])}
        levels = stitch_tracks([(c["start"], decode_logits(c["level"])) for c in chunks], total) if all("level" in c for c in chunks) else None
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SoundUnavailable(f"the sound model's response is malformed ({type(exc).__name__}: {exc})") from None
    return tracks, levels, resp


def load_sounds(directory: Path) -> dict[str, Any] | None:
    """The work dir's sounds.json, or None when there is no readable one."""
    try:
        data = json.loads((directory / "sounds.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("spans"), list) else None


def current_sounds(directory: Path) -> dict[str, Any] | None:
    """The work dir's sound map when it was built with the current model and rules."""
    data = load_sounds(directory)
    return data if data and data.get("version") == sound_version() else None


def sounds_duration(data: dict[str, Any]) -> float | None:
    """The duration to print for a sound map: the video's (metadata or ffprobe) when known, as
    fetch and frames print it; otherwise the analysed audio's."""
    return data.get("media_duration") or data.get("duration")


def sound_lines(directory: Path, data: dict[str, Any]) -> list[str]:
    """The `SOUNDS:` and `SOUNDS_FILE:` lines of a sound map."""
    return [
        f"SOUNDS: {sounds_summary(data.get('coverage') or {}, data.get('counts') or {})}",
        f"SOUNDS_FILE: {(directory / 'sounds.md').resolve()}",
    ]


def sound_record(data: dict[str, Any]) -> dict[str, Any]:
    """What meta.json keeps about a sound map."""
    return {
        "model": data["model"], "threshold": data["threshold"], "version": data["version"],
        "generated_at": data["generated_at"], "spans": len(data["spans"]),
    }


def build_sound_map(
    directory: Path,
    acquire: Callable[[], tuple[Path, int | None]],
    title: str | None,
    source_fields: dict[str, Any] | None = None,
    media_duration: float | None = None,
) -> dict[str, Any]:
    """Build sounds.json and sounds.md: a GPU check first (nothing is downloaded for a busy GPU),
    audio from `acquire` (a download, or a local original and its audio stream) converted to 48
    kHz mono, loudness via ffmpeg, then the sound model. Every audio file is deleted afterwards.
    Raises SoundUnavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        blocked = blocked_error("ffmpeg")
        raise SoundUnavailable(str(blocked) if blocked else "ffmpeg was not found on PATH")
    if not find_uv():
        raise SoundUnavailable("uv was not found on PATH")
    check_gpu_free()
    try:
        try:
            src, stream = acquire()
        except FetchError as exc:
            raise SoundUnavailable(f"the audio could not be fetched ({exc})") from None
        t0 = time.monotonic()
        pcm = to_sound_pcm(ffmpeg, src, directory, stream)
        if is_inside(src, directory) and src.resolve() != pcm.resolve():
            src.unlink(missing_ok=True)  # a download: only the 48 kHz copy is needed now
        duration = pcm_duration(pcm)
        if duration <= 0:
            raise SoundUnavailable("the audio is empty", permanent=True)
        log(f"ESTIMATE: ~{sound_estimate_min(duration):.1f} min for the sound map of {format_duration(duration)} of audio")
        loudness, notes = analyze_loudness(ffmpeg, pcm, duration)
        prompts = [prompt for _, prompt in SOUND_PROMPTS]
        tracks, levels, resp = run_sound_backend(pcm, prompts, directory, silent_stretches(ffmpeg, pcm, duration))
        spans = sound_spans({label: tracks[prompt] for label, prompt in SOUND_PROMPTS}, levels=levels, duration=duration)
        elapsed = time.monotonic() - t0
        data = {
            "version": sound_version(),
            "title": title,
            **(source_fields or {}),  # lets clean find a dir that holds only a sound map
            "model": SOUND_MODEL,
            "threshold": SOUND_THRESHOLD,
            "frame_rate": SOUND_FRAME_RATE,
            "duration": round(duration, 2),
            "media_duration": media_duration,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "coverage": coverage(spans, duration),
            "counts": event_counts(spans),
            "spans": spans,
            "loudness": loudness,
            "notes": notes,
            "timings": {"total_s": round(elapsed, 2), **(resp.get("timings") or {})},
            "peak_vram_mb": resp.get("peak_vram_mb"),
            "versions": resp.get("versions"),
        }
        (directory / "sounds.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        (directory / "sounds.md").write_text(sounds_markdown(data, title), encoding="utf-8")
        log(f"sound map in {elapsed:.1f}s: {len(spans)} spans")
        return data
    finally:
        remove_downloads(directory)  # the download, the 48 kHz copy and the request files


def write_meta(directory: Path, meta: dict[str, Any]) -> None:
    (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def make_sound_map(
    directory: Path,
    meta: dict[str, Any],
    acquire: Callable[[], tuple[Path, int | None]] | None,
    skip: str | None = None,
) -> list[str]:
    """Build the sound map for a fetch and record it in meta.json. Never fails the fetch: any
    problem becomes a `NOTE:` (the transcript is primary; its output and exit code stay exactly
    as without a sound map), with the traceback on stderr for unexpected errors. A current map
    from an earlier `sounds` run is reused. Returns the notes."""
    try:
        existing = current_sounds(directory)
        if existing:
            meta["sounds"] = sound_record(existing)
            write_meta(directory, meta)
            return []
        if skip or acquire is None:
            return [f"sound map skipped: {skip or 'no audio to analyse'}"]
        try:
            data = build_sound_map(directory, acquire, meta.get("title"), media_duration=meta.get("duration"))
        except SoundUnavailable as exc:
            if exc.permanent:
                meta["sounds"] = {"skipped": str(exc)}
            elif exc.gpu:
                meta["sounds"] = {"gpu_busy_at": time.time(), "reason": str(exc)}
            if exc.permanent or exc.gpu:
                write_meta(directory, meta)
            return [f"sound map skipped: {exc}"]
        meta["sounds"] = sound_record(data)
        write_meta(directory, meta)
        return [f"sound map: {n}" for n in data.get("notes") or []]
    except Exception as exc:  # noqa: BLE001 - the sound map must never fail the transcript
        log("sound map failed unexpectedly:\n" + traceback.format_exc().rstrip())
        return [f"sound map skipped: unexpected error ({type(exc).__name__}: {exc})"]


def needs_sound_map(directory: Path, meta: dict[str, Any]) -> bool:
    """A cached transcript without a current sound map gets one (unless there is no audio)."""
    if (meta.get("sounds") or {}).get("skipped"):
        return False
    return current_sounds(directory) is None


def gpu_backoff_note(meta: dict[str, Any], now: float | None = None) -> str | None:
    """While a recent GPU failure is within SOUND_GPU_RETRY_S, cached fetches do not retry the
    back-fill (no audio download per fetch on a busy GPU); this is the note saying so."""
    busy = (meta.get("sounds") or {}).get("gpu_busy_at")
    now = time.time() if now is None else now
    if not isinstance(busy, (int, float)) or now - busy >= SOUND_GPU_RETRY_S:
        return None
    retry = datetime.fromtimestamp(busy + SOUND_GPU_RETRY_S).astimezone().strftime("%H:%M")
    return f"sound map skipped: the GPU was busy {round((now - busy) / 60)} min ago; a fetch retries after {retry} (`sounds` tries now)"


def site_audio(info: dict[str, Any], directory: Path, args: argparse.Namespace) -> Callable[[], tuple[Path, None]]:
    """Audio for the sound map of a site video: the download Whisper kept, else a new download."""

    def acquire() -> tuple[Path, None]:
        kept = [p for p in sorted(directory.glob("audio.*")) if not p.name.startswith("audio.48k.") and p.suffix not in (".part", ".ytdl")]
        if kept:
            return kept[0], None
        if not has_audio(info):
            raise SoundUnavailable("the video has no audio track", permanent=True)
        import yt_dlp

        with yt_dlp.YoutubeDL(ydl_options(args, directory)) as ydl:
            return download_audio(ydl, info, directory, args), None

    return acquire


def file_audio(path: Path, audio_index: int | None) -> Callable[[], tuple[Path, int]]:
    """Audio for the sound map of a local file: its default audio stream, read in place."""

    def acquire() -> tuple[Path, int]:
        if audio_index is None:
            raise SoundUnavailable("the file has no audio stream", permanent=True)
        return path, audio_index

    return acquire


def refetch_audio(source: Source, directory: Path, args: argparse.Namespace) -> Callable[[], tuple[Path, int | None]]:
    """Audio for a sound map without the transcript run: a new download for sites (one metadata
    request), the original for local files."""
    if source.kind == "file":

        def acquire_file() -> tuple[Path, int]:
            ffmpeg = find_ffmpeg(LOCAL_FFMPEG_HINT)
            info = file_info(probe_file(find_ffprobe(ffmpeg), source.path), source.path)
            return file_audio(source.path, info.get("audio_index"))()

        return acquire_file

    def acquire_site() -> tuple[Path, None]:
        import yt_dlp

        with yt_dlp.YoutubeDL(ydl_options(args, directory)) as ydl:
            info = extract_info(ydl, source.url, bool(args.cookies_from_browser))
        return site_audio(info, directory, args)()

    return acquire_site


def backfill_sounds(source: Source, directory: Path, meta: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Add a sound map to a cached transcript (an old cache, or one whose map was skipped or is
    outdated) without running Whisper: the audio is fetched again (downloaded for sites, read
    from the original for local files) and deleted afterwards. Never fails the fetch."""
    note = gpu_backoff_note(meta)
    if note:
        return [note]
    try:
        acquire_lock(directory)
    except FetchError as exc:
        return [f"sound map not added: {exc}"]
    try:
        log("adding a sound map to the cached transcript")
        return make_sound_map(directory, meta, refetch_audio(source, directory, args))
    except Exception as exc:  # noqa: BLE001 - make_sound_map already catches; belt and braces
        log("sound map back-fill failed unexpectedly:\n" + traceback.format_exc().rstrip())
        return [f"sound map skipped: unexpected error ({type(exc).__name__}: {exc})"]
    finally:
        remove_downloads(directory)
        release_lock(directory)


# ---------------------------------------------------------------------------
# Frames: moments, picks and sheets


_MOMENT_RE = re.compile(r"^(?:(?:(\d+):)?(\d+):)?(\d+(?:\.\d+)?)$")


def parse_moment(text: str) -> float:
    """Parse `ss`, `ss.s`, `m:ss` or `h:mm:ss` (brackets allowed, as in transcript stamps)."""
    raw = text.strip().strip("[]").strip()
    match = _MOMENT_RE.match(raw)
    if not match:
        raise ValueError(f"invalid timestamp '{text.strip()}'; use ss, ss.s, m:ss or h:mm:ss")
    h, m, s = match.groups()
    if (m is not None and float(s) >= 60) or (h is not None and int(m) >= 60):
        raise ValueError(f"invalid timestamp '{text.strip()}'; minutes and seconds must be below 60")
    return int(h or 0) * 3600 + int(m or 0) * 60 + float(s)


def parse_moments(spec: str) -> list[float]:
    parts = [p for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("--at needs at least one timestamp")
    return [parse_moment(p) for p in parts]


def format_moment(t: float, long_form: bool) -> str:
    """Transcript-style stamp, plus a tenth of a second when the moment has one (`01:23.5`)."""
    tenths = round(t * 10)
    stamp = format_timestamp(tenths // 10, long_form)
    return f"{stamp}.{tenths % 10}" if tenths % 10 else stamp


def normalize_moments(moments: list[float], duration: float | None) -> tuple[list[float], list[str]]:
    """Clamp moments into the video, round to 0.1 s, dedupe and sort.

    A moment at or after the end becomes a frame END_MARGIN_S before the end. Returns the
    moments and notes about clamped ones.
    """
    long_form = (duration or 0) >= 3600
    notes: list[str] = []
    out: set[float] = set()
    for t in moments:
        if duration and t > duration - END_MARGIN_S:
            clamped = max(0.0, duration - END_MARGIN_S)
            notes.append(
                f"{format_moment(t, long_form)} is at or past the end ({format_duration(duration)}); "
                f"used {format_moment(clamped, long_form)}"
            )
            t = clamped
        out.add(round(t, 1))
    return sorted(out), notes


def frame_name(t: float) -> str:
    """File name that encodes the moment to 0.1 s, so a repeated request reuses the file."""
    tenths = round(t * 10)
    h, rem = divmod(tenths, 36000)
    m, rem = divmod(rem, 600)
    s, d = divmod(rem, 10)
    return f"frame-{h}h{m:02d}m{s:02d}.{d}s.jpg"


_CODEC_RANK = (("avc1", 3), ("vp09", 2), ("vp9", 2), ("av01", 1))


def _codec_rank(vcodec: str) -> int:
    return next((rank for prefix, rank in _CODEC_RANK if vcodec.startswith(prefix)), 0)


_AUDIO_EXTS = {"m4a", "mp3", "aac", "opus", "ogg", "oga", "wav", "flac", "weba", "mka"}


def _is_video_format(f: dict[str, Any]) -> bool:
    """True for formats with a video stream. YouTube always names the codec; other sites may
    leave it unknown (None), so dimensions or a non-audio extension count as video then."""
    vcodec = f.get("vcodec")
    if vcodec == "none" or not f.get("url") or f.get("has_drm"):
        return False
    if vcodec is None:
        return bool(f.get("width") or f.get("height")) or (
            f.get("ext") not in _AUDIO_EXTS and f.get("resolution") != "audio only"
        )
    return True


def _short_side(f: dict[str, Any]) -> int:
    """Short side in pixels; the height when only that is known; 0 when unknown."""
    w, h = f.get("width"), f.get("height")
    return min(w, h) if w and h else (h or w or 0)


def choose_video_format(formats: list[dict[str, Any]], max_short_side: int, stream: bool = True) -> dict[str, Any] | None:
    """Pick a video format whose short side is at most max_short_side (Shorts are vertical).

    stream=True keeps only plain HTTP(S) formats, which ffmpeg can seek in with range
    requests (no HLS or DASH fragments). Preference: the largest short side, video-only over
    muxed, then the smallest file (less data per seek or download; measured 3x faster seeks
    on a 1080p Short), then avc1 > vp9 > av1 when sizes are unknown. Formats of unknown size
    (some sites) rank below all others.
    """
    candidates = []
    for f in formats:
        if not _is_video_format(f) or _short_side(f) > max_short_side:
            continue
        if stream and f.get("protocol") not in {"https", "http"}:
            continue
        candidates.append(f)

    def key(f: dict[str, Any]) -> tuple[int, bool, float, int]:
        size = f.get("filesize") or f.get("filesize_approx") or float("inf")
        return _short_side(f), f.get("acodec") == "none", -size, _codec_rank(f.get("vcodec") or "")

    return max(candidates, key=key, default=None)


def choose_overview_format(formats: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the low-res format an overview downloads.

    Short side exactly 360 (it fills a 640 px tile), otherwise the smallest size above 360 up
    to 480, otherwise the size closest to 360. Within a size: video-only, then smallest file.
    Without any known size (some sites), the smallest video format by bitrate or file size.
    """
    sides = {_short_side(f) for f in formats if _is_video_format(f) and _short_side(f)}
    if not sides:
        videos = [f for f in formats if _is_video_format(f)]
        return min(videos, key=lambda f: f.get("tbr") or f.get("filesize") or f.get("filesize_approx") or float("inf"), default=None)

    def rank(side: int) -> tuple[int, int, int]:
        tier = 0 if side == OVERVIEW_SHORT_SIDE else 1 if OVERVIEW_SHORT_SIDE < side <= OVERVIEW_MAX_SHORT_SIDE else 2
        return tier, abs(side - OVERVIEW_SHORT_SIDE), -side

    side = min(sides, key=rank)
    same = [f for f in formats if _is_video_format(f) and _short_side(f) == side]
    return choose_video_format(same, side, stream=False)


def default_interval(duration: float | None) -> float:
    """Longest overview stretch without a pick: ~1/30 of the video, 10 to 60 s, whole seconds."""
    return float(round(min(60.0, max(10.0, (duration or 0) / 30))))


def min_gap_for(interval: float) -> float:
    """Smallest distance between overview picks, so bursts of cuts give one frame."""
    return round(min(15.0, max(2.0, interval / 4)), 1)


def pick_overview_times(
    scene_times: list[float],
    duration: float | None,
    interval: float,
    min_gap: float,
    black: list[tuple[float, float]] = (),
) -> tuple[list[float], int, int]:
    """Choose overview moments: the start, scene changes at least min_gap apart, and evenly
    spaced fills so no stretch is longer than interval.

    Every moment is a whole second (a scene change is rounded up into its new scene), so a
    tile's `[mm:ss]` stamp passed to `--at` returns the same frame in full detail. A moment
    inside a black stretch (`black`: (start, end) pairs, e.g. a fade-in from black) moves to
    the first whole second after it, so the start is not a black tile.
    Returns (sorted moments, scene picks, interval fills).
    """
    end = duration - END_MARGIN_S if duration else max(scene_times, default=0.0)

    def unblack(t: float) -> float:
        for a, b in black:
            if a - 0.01 <= t < b:
                return float(math.ceil(b))
        return t

    start = unblack(0.0)
    picks = [start if start <= end else 0.0]
    for t in sorted(unblack(float(math.ceil(t))) for t in scene_times):
        if t <= end and t - picks[-1] >= min_gap:
            picks.append(t)
    bounds = picks + ([duration] if duration else [])
    fills: set[float] = set()
    for a, b in zip(bounds, bounds[1:]):
        if b - a > interval:
            n = math.ceil((b - a) / interval)
            fills |= {unblack(float(round(a + k * (b - a) / n))) for k in range(1, n)}
    fills -= set(picks)
    fills = {t for t in fills if not duration or t <= end}
    return sorted(set(picks) | fills), len(picks) - 1, len(fills)


def thin_evenly(times: list[float], limit: int) -> list[float]:
    """Keep `limit` moments spread evenly over the list, always including the first."""
    if len(times) <= limit:
        return times
    if limit <= 1:
        return times[:limit]
    step = (len(times) - 1) / (limit - 1)
    return [times[round(i * step)] for i in range(limit)]


def sheet_groups(times: list[float], per_sheet: int = SHEET_COLS * SHEET_ROWS) -> list[list[float]]:
    return [times[i : i + per_sheet] for i in range(0, len(times), per_sheet)]


def sheet_layout(count: int) -> tuple[int, int]:
    """Columns and rows for a sheet of `count` tiles; a short last sheet gets fewer rows."""
    cols = min(SHEET_COLS, count)
    return cols, math.ceil(count / SHEET_COLS)


def scaled_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Fit the long side to long_side keeping aspect, never upscaling; even dimensions."""
    scale = min(1.0, long_side / max(width, height))
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def image_tokens(width: int, height: int) -> int:
    """Approximate Claude image tokens after its downscale to a ~1568 px long side."""
    scale = min(1.0, IMAGE_MAX_SIDE / max(width, height))
    return round(width * scale * height * scale / PIXELS_PER_TOKEN)


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Width and height from a JPEG's start-of-frame marker, or None."""
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if 0xC0 <= marker <= 0xCF and marker not in {0xC4, 0xC8, 0xCC}:
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            return width, height
        i += 2 + int.from_bytes(data[i + 2 : i + 4], "big")
    return None


def file_tokens(path: Path) -> int:
    size = jpeg_size(path.read_bytes()) if path.is_file() else None
    return image_tokens(*size) if size else 0


_PTS_TIME_RE = re.compile(r"\bpts_time:\s*(-?\d+(?:\.\d+)?)")


def parse_showinfo_times(text: str) -> list[float]:
    """Frame times from ffmpeg `showinfo` log lines (`... pts_time:12.345 ...`)."""
    times = []
    for line in text.splitlines():
        match = _PTS_TIME_RE.search(line) if "showinfo" in line else None
        if match:
            times.append(float(match.group(1)))
    return times


_BLACK_RE = re.compile(r"black_start:\s*(-?\d+(?:\.\d+)?)\s+black_end:\s*(-?\d+(?:\.\d+)?)")


def parse_blackdetect(text: str) -> list[tuple[float, float]]:
    """Black stretches from ffmpeg `blackdetect` log lines (`black_start:0 black_end:1.2 ...`)."""
    return [(float(a), float(b)) for a, b in _BLACK_RE.findall(text)]


def find_label_font(
    platform: str | None = None, env: dict[str, str] | None = None, fc_match: Callable[[], str | None] | None = None
) -> Path | None:
    """A TrueType font for tile labels: per OS the first existing candidate (Windows: in
    %WINDIR%\\Fonts; Linux: fontconfig's sans-serif first). None means ffmpeg's default font."""
    env = os.environ if env is None else env
    kind = platform_key(platform)
    candidates: list[Path] = []
    if kind == "win32":
        windir = env.get("WINDIR") or env.get("SystemRoot")
        if windir:
            candidates = [Path(windir) / "Fonts" / name for name in LABEL_FONT_CANDIDATES["win32"]]
    else:
        if kind == "linux":
            found = (fc_match or _fc_match_sans)()
            if found:
                candidates.append(Path(found))
        candidates += [Path(c) for c in LABEL_FONT_CANDIDATES[kind]]
    return next((c for c in candidates if c.suffix.lower() in (".ttf", ".ttc", ".otf") and c.is_file()), None)


def _fc_match_sans() -> str | None:
    fc = shutil.which("fc-match")
    if not fc:
        return None
    try:
        proc = subprocess.run([fc, "-f", "%{file}", "sans-serif"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


_LABEL_FONT: list[Path | None] = []


def label_font() -> Path | None:
    """The tile label font, looked up once per run."""
    if not _LABEL_FONT:
        _LABEL_FONT.append(find_label_font())
    return _LABEL_FONT[0]


def drawtext_filter(label: str, font: Path | None, size: int) -> str:
    """ffmpeg drawtext for a tile label; colons are escaped for the filtergraph. Without a font
    file ffmpeg uses its default font (fontconfig builds)."""
    text = label.replace("\\", "").replace("'", "").replace(":", "\\:")
    escaped = font.as_posix().replace(":", "\\:") if font else ""
    fontfile = f"fontfile='{escaped}':" if font else ""
    return (
        f"drawtext={fontfile}text='{text}':x=8:y=8:fontsize={size}"
        ":fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6"
    )


# ---------------------------------------------------------------------------
# Frames: ffmpeg and yt-dlp


# Fit the long side to FRAME_LONG_SIDE without upscaling; quoted so the commas stay inside.
FRAME_VF = (
    f"scale='if(gte(iw,ih),min({FRAME_LONG_SIDE},iw),-2)':'if(gte(iw,ih),-2,min({FRAME_LONG_SIDE},ih))'"
)


def blocked_executable(
    name: str, env: dict[str, str] | None = None, stat_fn: Callable[[str], os.stat_result] = os.stat
) -> tuple[str, str] | None:
    """An executable `name` on PATH that exists but this process may not run, as (path, reason):
    a sandboxed agent running as another user can see ffmpeg's folder on PATH without being
    allowed to read it, and shutil.which then reports nothing. A permission error on a folder
    counts only when the folder is named after the tool (an unreadable folder may hold anything)."""
    env = os.environ if env is None else env
    exts = [""]
    if os.name == "nt":
        exts = [e for e in env.get("PATHEXT", ".EXE;.BAT;.CMD").split(";") if e] or [".EXE"]
    for folder in env.get("PATH", "").split(os.pathsep):
        folder = folder.strip().strip('"')
        if not folder:
            continue
        for ext in exts:
            candidate = os.path.join(folder, name + ext.lower())
            try:
                st = stat_fn(candidate)
            except PermissionError as exc:
                if name.lower() in folder.lower():
                    return candidate, exc.strerror or "permission denied"
                continue
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if not os.access(candidate, os.X_OK):
                return candidate, "not executable by this user"
            try:
                with open(candidate, "rb"):
                    pass
            except OSError as exc:
                return candidate, exc.strerror or str(exc)
    return None


def blocked_error(name: str) -> FetchError | None:
    """`ERROR: ffmpeg was found but cannot be run …` for an ffmpeg or ffprobe the sandbox blocks."""
    found = blocked_executable(name)
    if not found:
        return None
    return FetchError(f"ffmpeg was found but cannot be run by this process ({found[0]}: {found[1]})", EXIT_ERROR, SANDBOX_HINT)


def find_ffmpeg(hint: str = FFMPEG_HINT) -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise blocked_error("ffmpeg") or FetchError("ffmpeg was not found on PATH", EXIT_ERROR, hint)
    return path


def find_ffprobe(ffmpeg: str) -> str:
    """ffprobe from PATH, else next to ffmpeg (both ship together)."""
    path = shutil.which("ffprobe")
    if path:
        return path
    sibling = Path(ffmpeg).with_name("ffprobe" + Path(ffmpeg).suffix)
    if sibling.is_file():
        return str(sibling)
    raise blocked_error("ffprobe") or FetchError("ffmpeg was not found on PATH (ffprobe is missing)", EXIT_ERROR, LOCAL_FFMPEG_HINT)


def probe_file(ffprobe: str, path: Path) -> dict[str, Any]:
    """ffprobe JSON (format, streams, chapters) of a local file; it is only read."""
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-show_chapters", str(path)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise FfmpegError(f"ffmpeg (ffprobe) timed out reading {path}") from None
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        raise FetchError(f"not a readable video or audio file: {path} ({lines[-1] if lines else proc.returncode})", EXIT_NO_TRANSCRIPT)
    try:
        return json.loads(proc.stdout or "{}")
    except ValueError:
        raise FetchError(f"not a readable video or audio file: {path}", EXIT_NO_TRANSCRIPT) from None


def _work_output(directory: Path, name: str, original: Path) -> Path:
    """An output path in the work dir; never the original file (local originals are read-only)."""
    out = directory / name
    if not is_inside(out, WORK_ROOT) or out.resolve() == original.resolve():
        raise FetchError(f"refusing to write {out}: outputs go only into the work root", EXIT_ERROR)
    return out


def extract_audio(ffmpeg: str, path: Path, stream_index: int, directory: Path) -> Path:
    """Decode one audio stream of a local file into a 16 kHz mono WAV in the work dir (what
    Whisper resamples to anyway). The original is only read."""
    out = _work_output(directory, "audio.wav", path)
    cmd = ["-i", str(path), "-map", f"0:{stream_index}", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)]
    try:
        proc = run_ffmpeg(ffmpeg, cmd, timeout=3600)
    except subprocess.TimeoutExpired:
        raise FfmpegError("ffmpeg timed out extracting the audio") from None
    if proc.returncode != 0 or not out.is_file():
        lines = proc.stderr.strip().splitlines()
        raise FfmpegError(f"ffmpeg could not extract the audio: {lines[-1] if lines else proc.returncode}")
    return out


def extract_embedded_subtitle(ffmpeg: str, path: Path, stream_index: int) -> list[Segment]:
    """Read one embedded text subtitle stream as SRT through a pipe (no file is written)."""
    try:
        proc = run_ffmpeg(ffmpeg, ["-i", str(path), "-map", f"0:{stream_index}", "-c:s", "srt", "-f", "srt", "pipe:1"], timeout=600)
    except subprocess.TimeoutExpired:
        raise FfmpegError("ffmpeg timed out reading the subtitle stream") from None
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        raise FfmpegError(f"ffmpeg could not read the subtitle stream: {lines[-1] if lines else proc.returncode}")
    return parse_subtitles(proc.stdout)


def run_ffmpeg(ffmpeg: str, args: list[str], timeout: float = 120, loglevel: str = "error") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", loglevel, "-y", *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def extract_frame(
    ffmpeg: str, source: str, at: float, out: Path, vf: str, input_args: tuple[str, ...] = (), quality: int = 2
) -> str | None:
    """Write the frame of `source` at `at` seconds to `out` with input seeking.

    The JPEG is written to a temporary name first, so a partial file never counts as cached.
    Returns an error message, or None on success.
    """
    tmp = out.with_name(out.name.removesuffix(".jpg") + ".tmp.jpg")
    cmd = [*input_args, "-ss", f"{at:.3f}", "-i", source, "-frames:v", "1", "-an", "-sn", "-vf", vf]
    try:
        proc = run_ffmpeg(ffmpeg, cmd + ["-q:v", str(quality), str(tmp)])
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return "ffmpeg timed out"
    if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        lines = proc.stderr.strip().splitlines()
        return lines[-1] if lines else "ffmpeg produced no frame"
    os.replace(tmp, out)
    return None


_COOKIE_ATTRS = {"domain", "path", "secure", "expires", "version", "max-age", "httponly", "samesite", "comment"}


def parse_format_cookies(cookies: str | None) -> list[dict[str, str]]:
    """Cookies from yt-dlp's per-format `cookies` field (`name=value; Domain=...; Path=...;
    Secure; ...` for each cookie, already selected for the format URL). Values are unquoted."""
    from http.cookies import SimpleCookie

    decode = SimpleCookie().value_decode  # yt-dlp quotes values with special characters
    out: list[dict[str, str]] = []
    for part in (cookies or "").split(";"):  # quoting escapes `;` inside values
        name, sep, value = part.strip().partition("=")
        if not name:
            continue
        if name.lower() in _COOKIE_ATTRS:
            if out and name.lower() in ("domain", "path") and sep:
                out[-1][name.lower()] = value
        elif sep:
            out.append({"name": name, "value": decode(value)[0], "domain": "", "path": ""})
    return out


def ffmpeg_cookies(cookies: str | None, url: str) -> str:
    """ffmpeg `-cookies` value, one `name=value; path=...; domain=...;` line per cookie as yt-dlp's
    ffmpeg downloader builds it. ffmpeg sends each cookie only to its domain, also after
    redirects, unlike a raw `Cookie:` header."""
    host = urlparse(url).hostname or ""
    return "".join(
        f"{c['name']}={c['value']}; path={c['path'] or '/'}; domain={c['domain'] or host};\r\n"
        for c in parse_format_cookies(cookies)
    )


def stream_input_args(fmt: dict[str, Any]) -> tuple[str, ...]:
    """ffmpeg input options for a format URL: yt-dlp's HTTP headers, the format's cookies
    (domain-scoped) and a 30 s I/O timeout. Local files need none.

    Without --cookies-from-browser the cookies are the session cookies yt-dlp received while
    extracting (TikTok refuses its streams without them); with it, they include the consented
    browser cookies for that domain.
    """
    if fmt.get("local"):
        return ()
    headers = {k: v for k, v in (fmt.get("http_headers") or {}).items() if k.lower() != "cookie"}
    text = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    cookies = ffmpeg_cookies(fmt.get("cookies"), fmt.get("url") or "")
    return ("-rw_timeout", "30000000") + (("-headers", text) if text else ()) + (("-cookies", cookies) if cookies else ())


def download_video(info: dict[str, Any], directory: Path, args: argparse.Namespace, opts: dict[str, Any]) -> list[dict[str, Any]]:
    """Download with yt-dlp using the already extracted info; returns `requested_downloads`."""
    import yt_dlp
    from yt_dlp.utils import DownloadError

    with yt_dlp.YoutubeDL(ydl_options(args, directory) | opts) as ydl:
        try:
            result = ydl.process_ie_result(dict(info), download=True)
        except DownloadError as exc:
            raise classify_ytdlp_error(str(exc), bool(args.cookies_from_browser)) from None
    return (result or {}).get("requested_downloads") or []


def seek_formats(formats: list[dict[str, Any]], max_short_side: int = TARGET_MAX_SHORT_SIDE) -> list[dict[str, Any]]:
    """Formats to seek in for targeted frames, in order of preference.

    First the plain-HTTPS stream format; then, as fallbacks, the best HLS format (another
    protocol and host) and the best remaining format. ffmpeg seeks accurately in all of them,
    so a fallback frame shows exactly the requested moment.
    """
    usable = [f for f in formats if f.get("url")]
    hls = [f for f in usable if str(f.get("protocol", "")).startswith("m3u8")]
    order: list[dict[str, Any]] = []
    stream_fmt = choose_video_format(usable, max_short_side)
    if stream_fmt is not None:
        order.append(stream_fmt)
        # The same codec over HLS usually carries the identical encode, so its frames match.
        same_codec = [f for f in hls if _codec_family(f) == _codec_family(stream_fmt)]
        hls = same_codec or hls
    for pool in (hls, usable):
        pick = choose_video_format([f for f in pool if all(f is not o for o in order)], max_short_side, stream=False)
        if pick is not None:
            order.append(pick)
    return order


def _codec_family(fmt: dict[str, Any]) -> str:
    vcodec = (fmt.get("vcodec") or "").lower()
    return "vp9" if vcodec.startswith(("vp9", "vp09")) else vcodec[:4]


# yt-dlp's HLS DRM markers (downloader/hls.py). Some sites (Vimeo's player) put the key only in
# the media playlists, so yt-dlp does not flag the format and ffmpeg fails with "Invalid data".
_HLS_DRM_RE = re.compile(
    r'#EXT-X-(?:SESSION-)?KEY:.*?(?:URI="skd://|KEYFORMAT="com\.apple\.streamingkeydelivery"'
    r'|KEYFORMAT="com\.microsoft\.playready")|#EXT-X-FAXS-CM:'
)


def hls_has_drm(playlist: str) -> bool:
    return bool(_HLS_DRM_RE.search(playlist))


def hls_format_has_drm(fmt: dict[str, Any]) -> bool:
    """Read an HLS format's media playlist (a few KB) and check it for DRM keys."""
    import urllib.request

    request = urllib.request.Request(fmt["url"], headers=dict(fmt.get("http_headers") or {}))
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return hls_has_drm(response.read(512 * 1024).decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        return False


def is_stream_error(message: str) -> bool:
    """True for network or HTTP failures (the site refused the stream), not decoding problems."""
    low = message.lower()
    return any(p in low for p in ("http error", "server returned", "403", "404", "timed out", "connection", "i/o error"))


def detect_scenes(ffmpeg: str, video: Path, threshold: float) -> tuple[list[float], list[tuple[float, float]]]:
    """Scene-change times and black stretches of a local video, in one decoding pass."""
    vf = f"blackdetect=d=0:pix_th={BLACK_PIXEL_THRESHOLD},select='gt(scene,{threshold})',showinfo"
    try:
        proc = run_ffmpeg(ffmpeg, ["-i", str(video), "-an", "-sn", "-vf", vf, "-f", "null", "-"], 3600, "info")
    except subprocess.TimeoutExpired:
        raise FfmpegError("ffmpeg scene detection timed out") from None
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        raise FfmpegError(f"ffmpeg scene detection failed: {lines[-1] if lines else proc.returncode}")
    return parse_showinfo_times(proc.stderr), parse_blackdetect(proc.stderr)


def said_segment(segments: list[Segment], t: float, window: float = SAID_WINDOW_S) -> Segment | None:
    """The transcript segment spoken at moment t: the one containing it (the latest-starting
    one if segments overlap), otherwise the nearest one within `window` seconds, else None."""
    containing = [s for s in segments if s.start <= t <= s.end]
    if containing:
        return max(containing, key=lambda s: s.start)
    near = [(s.start - t if t < s.start else t - s.end, s.start, s) for s in segments]
    near = [n for n in near if n[0] <= window]
    return min(near, key=lambda n: n[:2])[2] if near else None


def said_text(text: str, limit: int = SAID_MAX_CHARS) -> str:
    """Segment text on one line, cut at a word boundary with an ellipsis beyond `limit` chars."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut[limit // 2 :]:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "…"


def said_lines(segments: list[Segment] | None, moments: list[float], long_form: bool) -> list[str]:
    """`SAID: [stamp] text` for each moment with speech; the stamp is the frame's moment."""
    lines = []
    for t in moments:
        seg = said_segment(segments or [], t)
        text = said_text(seg.text) if seg is not None else ""
        if text:
            lines.append(f"SAID: [{format_moment(t, long_form)}] {text}")
    return lines


def cached_sound_spans(directory: Path) -> list[dict[str, Any]] | None:
    """Spans of the work dir's sound map for HEARD lines; None without one (frames never build
    it: they stay CPU-only)."""
    data = load_sounds(directory)
    return data["spans"] if data else None


def load_frames_info(frames_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((frames_dir / "info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def print_frames_summary(
    key: str, info: dict[str, Any], mode: str, lines: list[str], count: int, tokens: int, notes: list[str]
) -> None:
    """`VIDEO_ID:` carries the work key: the YouTube id, `<extractor>-<id>` or `file-<hash>`."""
    out = [
        f"VIDEO_ID: {key}",
        f"TITLE: {info.get('title') or ''}",
        f"DURATION: {format_duration(info.get('duration'))}",
        f"MODE: {mode}",
        *lines,
        f"COUNT: {count}",
        f"TOKENS_ESTIMATE: ~{tokens}",
    ]
    out += [f"NOTE: {n}" for n in notes]
    print("\n".join(out), flush=True)


# ---------------------------------------------------------------------------
# Commands


def source_meta(source: Source, info: dict[str, Any], input_text: str) -> dict[str, Any]:
    """Source fields for meta.json and frames/info.json. `clean` matches them without network
    access. A local file's `url` is its `file:` URI and `source_path` its absolute path."""
    if source.kind == "file":
        return {"source_kind": "file", "url": source.path.as_uri(), "source_path": str(source.path)}
    url = source.url if source.kind == "youtube" else info.get("webpage_url") or source.url
    fields = {"source_kind": source.kind, "url": url, "input_url": strip_input(input_text)}
    if source.kind == "site":
        fields |= {"extractor": info.get("extractor_key"), "site_id": info.get("id")}
    return fields


def cmd_fetch(args: argparse.Namespace) -> int:
    source = resolve_source(args.url)
    directory = work_dir(source.key)
    cached = load_cache(directory, args)
    if cached:
        refresh = source.kind == "youtube" and not has_context(cached)  # only v1 YouTube caches lack them
        notes = refresh_context(source.url, directory, cached, args) if refresh else []
        if sounds_wanted(args) and needs_sound_map(directory, cached):
            notes += backfill_sounds(source, directory, cached, args)  # never runs Whisper
        os.utime(directory)
        print_summary(directory, cached, cached=True, extra_notes=notes)
        return EXIT_OK

    tools: tuple[str, str] | None = None
    if source.kind == "file":  # before any work dir, like frames; URLs never need ffmpeg
        ffmpeg = find_ffmpeg(LOCAL_FFMPEG_HINT)
        tools = (ffmpeg, find_ffprobe(ffmpeg))
    directory.mkdir(parents=True, exist_ok=True)
    acquire_lock(directory)  # exit 1 if a frames run is using this video
    try:
        if tools:
            return _fetch_file(source, directory, args, *tools)
        return _fetch(source, directory, args)
    finally:
        remove_downloads(directory)
        release_lock(directory)
        if not has_results(directory):
            shutil.rmtree(directory, ignore_errors=True)  # failed first run: leave nothing behind


def refresh_context(url: str, directory: Path, meta: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Add description and chapters to a transcript cached before they were stored.

    One metadata request, then both transcript files are rewritten from segments.json;
    Whisper never runs again. `meta` is updated in place. On failure the cached transcript is
    still served and a note says why (meta stays old-style, so the next fetch tries again).
    """
    import yt_dlp

    segments = load_segments(directory)
    if not segments:
        return ["description and chapters unavailable: segments.json could not be read"]
    try:
        acquire_lock(directory)
    except FetchError as exc:
        return [f"description and chapters not added: {exc}"]
    try:
        with yt_dlp.YoutubeDL(ydl_options(args, directory)) as ydl:
            info = extract_info(ydl, url, bool(args.cookies_from_browser))
        meta.update(video_context(info))
        write_transcripts(directory, meta, segments)
        log("added description and chapters to the cached transcript")
        return []
    except FetchError as exc:
        return [f"description and chapters unavailable: {exc}"]
    finally:
        release_lock(directory)


def _fetch(source: Source, directory: Path, args: argparse.Namespace) -> int:
    """fetch for YouTube and other yt-dlp sites."""
    import yt_dlp

    started = time.monotonic()
    youtube = source.kind == "youtube"
    used_cookies = bool(args.cookies_from_browser)
    with yt_dlp.YoutubeDL(ydl_options(args, directory)) as ydl:
        t0 = time.monotonic()
        info = extract_info(ydl, source.url, used_cookies)
        t_info = time.monotonic() - t0
        original_lang = args.lang or info.get("language")
        notes: list[str] = []
        result: Result | None = None

        if args.whisper:
            notes.append("whisper forced with --whisper")
        else:
            extractor = str(info.get("extractor_key") or "")
            if not youtube and extractor.lower().startswith(MACHINE_SUBTITLE_EXTRACTORS) and info.get("subtitles"):
                track, why = None, f"{extractor} subtitles are machine-generated captions"
            else:
                # Only with a stated original language (never a possible translation).
                track, why = choose_manual_track(info.get("subtitles") or {}, original_lang, strict=not youtube)
            if track:
                try:
                    cues = download_subtitle(ydl, info["subtitles"][track])
                    segments = flatten_cues(cues)
                    if segments:
                        result = Result(
                            "manual-subtitles", track, segments, [why], {"track": track}, {}
                        )
                    else:
                        notes.append(f"manual track '{track}' was empty; using whisper")
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"manual track '{track}' could not be downloaded ({exc}); using whisper")
            else:
                notes.append(f"{why}; using whisper")

        sound_skip = None
        if result is None:
            if not has_audio(info):
                raise FetchError("the video has no audio track; there is nothing to transcribe", EXIT_NO_TRANSCRIPT)
            try:
                result = run_whisper(ydl, info, directory, args, original_lang, keep_audio=sounds_wanted(args))
            except GpuUnavailable as exc:
                sound_skip = f"the GPU is unavailable (Whisper could not run: {exc})"
                log(f"whisper unavailable on GPU: {exc}")
                if exc.no_cuda:  # no usable CUDA at all: never a silent fallback, CPU mode is the user's choice
                    raise FetchError(f"no usable CUDA GPU for Whisper ({exc})", EXIT_NO_TRANSCRIPT, exc.hint) from None
                if not youtube:  # the auto-captions fallback is YouTube-only (its `-orig` tracks)
                    raise FetchError(f"Whisper could not run on the GPU ({exc})", EXIT_NO_TRANSCRIPT, exc.hint) from None
                track = choose_auto_track(info.get("automatic_captions") or {}, original_lang)
                if not track:
                    raise FetchError(
                        f"Whisper could not run on the GPU ({exc}) and no original-language "
                        "auto-captions exist",
                        EXIT_NO_TRANSCRIPT,
                        exc.hint,
                    ) from None
                cues = download_subtitle(ydl, info["automatic_captions"][track])
                segments = clean_rolling_cues(cues)
                if not segments:
                    raise FetchError(f"auto-caption track '{track}' was empty", EXIT_NO_TRANSCRIPT) from None
                result = Result(
                    "auto-subtitles-fallback",
                    track.removesuffix("-orig"),
                    segments,
                    [f"Whisper unavailable on GPU ({exc}); used YouTube auto-captions '{track}'"],
                    {"track": track},
                    {},
                )

    acquire = None if sound_skip else site_audio(info, directory, args)
    return _finish_fetch(source, directory, args, info, result, notes, t_info, started, acquire, sound_skip)


def _fetch_file(source: Source, directory: Path, args: argparse.Namespace, ffmpeg: str, ffprobe: str) -> int:
    """fetch for a local file: an embedded text subtitle in the audio's language, else Whisper on
    audio that ffmpeg extracts into the work dir. The original file is only read."""
    started = time.monotonic()
    t0 = time.monotonic()
    probe = probe_file(ffprobe, source.path)
    info = file_info(probe, source.path)
    t_info = time.monotonic() - t0
    original_lang = args.lang or info.get("language")
    notes: list[str] = []
    result: Result | None = None

    if args.whisper:
        notes.append("whisper forced with --whisper")
    else:
        stream, why = choose_embedded_subtitle(probe.get("streams") or [], original_lang)
        if stream is not None:
            label = f"embedded #{stream.get('index')} {stream.get('codec_name')} '{stream_language(stream)}'"
            try:
                segments = flatten_cues(extract_embedded_subtitle(ffmpeg, source.path, stream["index"]))
                if segments:
                    result = Result("manual-subtitles", stream_language(stream), segments, [why], {"track": label}, {})
                else:
                    notes.append(f"subtitle stream {label} was empty; using whisper")
            except FfmpegError as exc:
                notes.append(f"subtitle stream {label} could not be read ({exc}); using whisper")
        else:
            notes.append(f"{why}; using whisper")

    if result is None:
        if info.get("audio_index") is None:
            raise FetchError("the file has no audio stream; there is nothing to transcribe", EXIT_NO_TRANSCRIPT)
        try:
            result = run_whisper(
                None, info, directory, args, original_lang,
                get_audio=lambda: extract_audio(ffmpeg, source.path, info["audio_index"], directory),
            )
        except GpuUnavailable as exc:
            log(f"whisper unavailable on GPU: {exc}")
            if exc.no_cuda:
                raise FetchError(f"no usable CUDA GPU for Whisper ({exc})", EXIT_NO_TRANSCRIPT, exc.hint) from None
            raise FetchError(f"Whisper could not run on the GPU ({exc})", EXIT_NO_TRANSCRIPT, exc.hint) from None

    acquire = file_audio(source.path, info.get("audio_index"))
    return _finish_fetch(source, directory, args, info, result, notes, t_info, started, acquire)


def _finish_fetch(
    source: Source,
    directory: Path,
    args: argparse.Namespace,
    info: dict[str, Any],
    result: Result,
    notes: list[str],
    t_info: float,
    started: float,
    sound_audio: Callable[[], tuple[Path, int | None]] | None = None,
    sound_skip: str | None = None,
) -> int:
    """Write the transcript, then (by default) the sound map on the same audio, then print."""
    if not result.segments:
        raise FetchError("transcript is empty (no speech detected)", EXIT_NO_TRANSCRIPT)

    result.timings = {"info_s": round(t_info, 2), **result.timings, "total_s": round(time.monotonic() - started, 2)}
    meta = {
        "video_id": source.key,
        **source_meta(source, info, args.url),
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "language": result.language,
        "metadata_language": info.get("language"),
        "source": result.source,
        "notes": notes + result.notes,
        "settings": result.settings,
        "timings": result.timings,
        "versions": versions(),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **video_context(info),
    }
    write_outputs(directory, meta, result.segments)  # the transcript is safe before the sound map runs
    sound_notes = make_sound_map(directory, meta, sound_audio, sound_skip) if sounds_wanted(args) else []
    print_summary(directory, meta, cached=False, extra_notes=sound_notes)
    return EXIT_OK


def cmd_frames(args: argparse.Namespace) -> int:
    source = resolve_source(args.url)
    if args.max_frames < 1:
        raise FetchError("invalid frames request: --max-frames must be at least 1", EXIT_ERROR, ARGS_HINT)
    if not args.overview and (args.interval is not None or args.scene_threshold is not None):
        raise FetchError(
            "invalid frames request: --interval and --scene-threshold only apply to --overview", EXIT_ERROR, ARGS_HINT
        )
    if args.fps is not None and args.range is None:
        raise FetchError("invalid frames request: --fps only applies to --range", EXIT_ERROR, ARGS_HINT)
    if args.range is not None:
        try:
            args.range_bounds = parse_range(args.range)
        except ValueError as exc:
            raise FetchError(f"invalid frames request: {exc}", EXIT_ERROR, ARGS_HINT) from None
        fps = args.fps if args.fps is not None else RANGE_FPS_DEFAULT
        if not 0 < fps <= RANGE_MAX_FPS:
            raise FetchError(f"invalid frames request: --fps must be above 0 and at most {RANGE_MAX_FPS:g}", EXIT_ERROR, ARGS_HINT)
        count = len(range_times(*args.range_bounds, fps))
        if count > args.max_frames:
            raise FetchError(
                f"invalid frames request: {count} frames ({args.range} at {fps:g} fps) exceed the safety ceiling of "
                f"{args.max_frames} frames",
                EXIT_ERROR,
                "lower --fps or narrow the range; pass a higher --max-frames only if this many frames are really needed",
            )
    if args.at is not None:
        try:
            args.moments = parse_moments(args.at)
        except ValueError as exc:
            raise FetchError(f"invalid frames request: {exc}", EXIT_ERROR, ARGS_HINT) from None
        requested = len({round(t, 1) for t in args.moments})
        if requested > args.max_frames:
            raise FetchError(
                f"invalid frames request: {requested} moments exceed the safety ceiling of {args.max_frames} frames",
                EXIT_ERROR,
                "pass a higher --max-frames if this many frames are really needed",
            )
    if args.interval is not None and args.interval <= 0:
        raise FetchError("invalid frames request: --interval must be positive", EXIT_ERROR, ARGS_HINT)
    if args.scene_threshold is not None and not 0 < args.scene_threshold < 1:
        raise FetchError("invalid frames request: --scene-threshold must be between 0 and 1", EXIT_ERROR, ARGS_HINT)
    ffmpeg = find_ffmpeg()  # before any network work; fetch of a URL never needs ffmpeg
    if source.kind == "file":
        find_ffprobe(ffmpeg)

    directory = work_dir(source.key)
    directory.mkdir(parents=True, exist_ok=True)
    acquire_lock(directory)  # exit 1 if a fetch or another frames run is using this video
    try:
        (directory / "frames").mkdir(exist_ok=True)
        if args.at is not None:
            return _frames_at(source, directory, args, ffmpeg)
        if args.range is not None:
            return _frames_range(source, directory, args, ffmpeg)
        return _frames_overview(source, directory, args, ffmpeg)
    finally:
        remove_downloads(directory)
        release_lock(directory)
        if has_results(directory):
            os.utime(directory)
        else:
            shutil.rmtree(directory, ignore_errors=True)  # failed first run: leave nothing behind


def _video_info(source: Source, directory: Path, args: argparse.Namespace, ffmpeg: str) -> dict[str, Any]:
    if source.kind == "file":
        info = file_info(probe_file(find_ffprobe(ffmpeg), source.path), source.path)
    else:
        import yt_dlp

        with yt_dlp.YoutubeDL(ydl_options(args, directory)) as ydl:
            t0 = time.monotonic()
            info = extract_info(ydl, source.url, bool(args.cookies_from_browser))
            log(f"metadata in {time.monotonic() - t0:.1f}s")
    # Title and duration let a fully cached --at request skip the network next time; the source
    # fields let clean find the dir offline.
    known = {"title": info.get("title"), "duration": info.get("duration"), **source_meta(source, info, args.url)}
    (directory / "frames" / "info.json").write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")
    return info


def _local_format(source: Source, info: dict[str, Any]) -> dict[str, Any]:
    """The local file as a pseudo-format for the frame code: ffmpeg seeks in it directly."""
    if not info.get("width") or not info.get("height"):
        raise FetchError(f"no video stream in {source.path}; frames need video", EXIT_NO_TRANSCRIPT)
    return {
        "format_id": "file",
        "url": str(source.path),
        "width": info["width"],
        "height": info["height"],
        "vcodec": info.get("vcodec"),
        "protocol": "file",
        "local": True,
    }


def _describe(fmt: dict[str, Any]) -> str:
    size = f"{fmt.get('width') or '?'}x{fmt.get('height') or '?'}"
    return f"{fmt.get('format_id')} ({size} {fmt.get('vcodec')}, {fmt.get('protocol')})"


def _frames_at(source: Source, directory: Path, args: argparse.Namespace, ffmpeg: str) -> int:
    frames_dir = directory / "frames"
    info = load_frames_info(frames_dir)
    if info:
        moments, notes = normalize_moments(args.moments, info.get("duration"))
        if not all((frames_dir / frame_name(t)).is_file() for t in moments):
            info = None
    if info is None:
        info = _video_info(source, directory, args, ffmpeg)
        moments, notes = normalize_moments(args.moments, info.get("duration"))
    long_form = (info.get("duration") or 0) >= 3600
    cached = {t for t in moments if (frames_dir / frame_name(t)).is_file()}
    todo = [t for t in moments if t not in cached]

    errors: dict[float, str] = {}
    if todo:
        t0 = time.monotonic()
        # A local file is read in place (time-exact input seeking); sites seek in a stream URL.
        candidates = [_local_format(source, info)] if source.kind == "file" else seek_formats(info.get("formats") or [])
        if not candidates:
            raise FetchError("no video format to take frames from (the source may be audio only)", EXIT_NO_TRANSCRIPT)
        log(f"seeking in {'the local file,' if source.kind == 'file' else 'format'} {_describe(candidates[0])}")
        failed: list[dict[str, Any]] = []  # formats that failed once are tried last
        fallback_used: set[str] = set()
        for t in todo:
            err = "no usable video format"
            for fmt in [f for f in candidates if f not in failed] + failed:
                err = extract_frame(ffmpeg, fmt["url"], t, frames_dir / frame_name(t), FRAME_VF, stream_input_args(fmt))
                if err is None:
                    if fmt is not candidates[0]:
                        fallback_used.add(fmt["format_id"])
                    break
                log(f"frame [{format_moment(t, long_form)}]: seek in format {_describe(fmt)} failed ({err})")
                if fmt not in failed:
                    failed.append(fmt)
            if err:
                errors[t] = err
        if fallback_used:
            notes.append(f"stream seek failed for some frames; used fallback format(s) {', '.join(sorted(fallback_used))}")
        log(f"{len(todo) - len(errors)} frame(s) extracted in {time.monotonic() - t0:.1f}s")

    segments = load_segments(directory)  # the cached transcript, if fetch ran; frames never fetch
    sounds = cached_sound_spans(directory)  # the cached sound map, if fetch built one
    lines, tokens = [], 0
    for t in moments:
        if t in errors:
            notes.append(f"no frame at {format_moment(t, long_form)}: {errors[t]}")
            continue
        path = frames_dir / frame_name(t)
        cached_mark = " (cached)" if t in cached else ""
        lines.append(f"FRAME: [{format_moment(t, long_form)}] {path.resolve()}{cached_mark}")
        lines += said_lines(segments, [t], long_form)
        lines += heard_lines(sounds, [t], long_form)
        tokens += file_tokens(path)
    if segments is None:
        notes.append(NO_TRANSCRIPT_NOTE)
    if not lines:
        first = next(iter(errors.values()))
        hls = next((f for f in candidates if str(f.get("protocol", "")).startswith("m3u8")), None)
        if source.kind != "file" and hls is not None and hls_format_has_drm(hls):
            raise FetchError("DRM-protected content is not supported (the video stream is encrypted)", EXIT_NO_TRANSCRIPT)
        if source.kind != "file" and is_stream_error(first):
            raise FetchError(f"no frame could be extracted: the video stream failed ({first})", EXIT_ERROR, STREAM_FAIL_HINT)
        raise FfmpegError(f"ffmpeg could not extract any frame ({first})")
    count = sum(line.startswith("FRAME: ") for line in lines)
    print_frames_summary(source.key, info, "at", lines, count, tokens, notes)
    return EXIT_OK


def _frames_overview(source: Source, directory: Path, args: argparse.Namespace, ffmpeg: str) -> int:
    info = _video_info(source, directory, args, ffmpeg)
    if source.kind == "file":
        # Scanned in place: no download and no copy, and the original is never deleted.
        local = _local_format(source, info)
        log("overview: scanning the local file directly")
        return _overview_scan(source.key, info, source.path, (local["width"], local["height"]), directory, args, ffmpeg, [])

    formats = info.get("formats") or []
    fmt = choose_overview_format(formats)
    if fmt is None:
        raise FetchError("no downloadable video format", EXIT_ERROR, f"yt-dlp may be outdated; try: {UPGRADE_HINT}")
    attempts = [fmt]
    # If ffmpeg cannot decode the chosen format (the smallest file is often AV1), retry once
    # with a non-AV1 format.
    alternate = choose_overview_format([f for f in formats if f is not fmt and not (f.get("vcodec") or "").startswith("av01")])
    if alternate is not None:
        attempts.append(alternate)
    for i, fmt in enumerate(attempts):
        try:
            video = _download_overview_video(info, fmt, directory, args)
            try:
                size = (fmt.get("width"), fmt.get("height"))
                if not all(size):  # some sites do not report the size of a format
                    probed = file_info(probe_file(find_ffprobe(ffmpeg), video), video)
                    size = (probed.get("width"), probed.get("height"))
                if not all(size):
                    raise FfmpegError("ffmpeg found no video stream in the downloaded format")
                notes = [f"decoding failed with the first format; used format {fmt['format_id']} ({fmt.get('vcodec')})"] if i else []
                return _overview_scan(source.key, info, video, size, directory, args, ffmpeg, notes)
            finally:
                video.unlink(missing_ok=True)  # the downloaded low-res copy only
        except FfmpegError as exc:
            remove_downloads(directory)
            if i + 1 == len(attempts):
                raise
            log(f"{exc}; retrying with format {_describe(attempts[i + 1])}")
    raise AssertionError("unreachable")


def _download_overview_video(info: dict[str, Any], fmt: dict[str, Any], directory: Path, args: argparse.Namespace) -> Path:
    t0 = time.monotonic()
    log(f"overview: downloading format {_describe(fmt)}")
    downloads = download_video(info, directory, args, {"format": fmt["format_id"], "outtmpl": str(directory / "video.%(ext)s")})
    video = Path(downloads[0]["filepath"]) if downloads else next(iter(directory.glob("video.*")), None)
    if video is None or not video.is_file() or not is_inside(video, directory):
        raise FetchError("video download produced no file", EXIT_ERROR, f"try: {UPGRADE_HINT}")
    log(f"video downloaded in {time.monotonic() - t0:.1f}s ({video.stat().st_size / 1e6:.1f} MB)")
    return video


def _overview_scan(
    key: str,
    info: dict[str, Any],
    video: Path,
    size: tuple[int, int],
    directory: Path,
    args: argparse.Namespace,
    ffmpeg: str,
    notes: list[str],
) -> int:
    """Scene detection, tiles and contact sheets from `video`, which is only read (a downloaded
    copy, or a local original)."""
    frames_dir = directory / "frames"
    duration = info.get("duration")
    long_form = (duration or 0) >= 3600

    t1 = time.monotonic()
    threshold = args.scene_threshold or SCENE_THRESHOLD_DEFAULT
    interval = args.interval or default_interval(duration)
    gap = min_gap_for(interval)
    scenes, black = detect_scenes(ffmpeg, video, threshold)
    times, n_scene, n_fill = pick_overview_times(scenes, duration, interval, gap, black)
    log(f"scene detection in {time.monotonic() - t1:.1f}s: {len(scenes)} scene changes, {len(black)} black stretches")
    picks = (
        f"start + {n_scene} scene changes (threshold {threshold:g}) + {n_fill} interval fills "
        f"(no gap over {interval:g} s), min gap {gap:g} s"
    )
    if len(times) > args.max_frames:
        notes.append(
            f"{len(times)} moments exceeded the safety ceiling; thinned evenly to {args.max_frames} "
            "(raise it with --max-frames)"
        )
        times = thin_evenly(times, args.max_frames)

    t2 = time.monotonic()
    tw, th = scaled_size(size[0], size[1], TILE_LONG_SIDE)
    base_vf = f"scale={tw}:{th}:force_original_aspect_ratio=decrease,pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2"
    labels = True  # with the found font, else ffmpeg's default; a failure disables labels
    font = label_font()
    font_size = max(14, min(tw, th) // 13)
    done: list[float] = []
    for t in times:
        stamp = format_moment(t, long_form)
        tile = frames_dir / f"tile-{len(done):04d}.jpg"
        vf = f"{base_vf},{drawtext_filter(stamp, font, font_size)}" if labels else base_vf
        err = extract_frame(ffmpeg, str(video), t, tile, vf, quality=3)
        if err and labels:
            labels = False
            notes.append(f"tile labels disabled ({err}); map tiles with the TIMES lines")
            err = extract_frame(ffmpeg, str(video), t, tile, base_vf, quality=3)
        if err:
            notes.append(f"no frame at {stamp}: {err}")
            continue
        done.append(t)
    if not done:
        raise FfmpegError("ffmpeg could not extract any overview frame")

    lines, tokens = build_sheets(ffmpeg, directory, done, "overview", long_form, notes)
    log(f"{len(done)} tiles on {sum(line.startswith('SHEET: ') for line in lines)} sheet(s) in {time.monotonic() - t2:.1f}s")
    print_frames_summary(key, info, "overview", [f"PICKS: {picks}", *lines], len(done), tokens, notes)
    return EXIT_OK


def build_sheets(
    ffmpeg: str, directory: Path, times: list[float], prefix: str, long_form: bool, notes: list[str]
) -> tuple[list[str], int]:
    """3x3 contact sheets `frames/<prefix>-NN.jpg` from the tiles `frames/tile-NNNN.jpg` (one per
    moment in `times`, in order), replacing earlier sheets of that prefix. Returns the
    `SHEET:`/`TIMES:`/`SAID:`/`HEARD:` lines and the token estimate."""
    frames_dir = directory / "frames"
    for old in frames_dir.glob(f"{prefix}-*.jpg"):
        old.unlink()
    segments = load_segments(directory)  # the cached transcript, if fetch ran; frames never fetch
    if segments is None:
        notes.append(NO_TRANSCRIPT_NOTE)
    sounds = cached_sound_spans(directory)
    lines, tokens = [], 0
    per_sheet = SHEET_COLS * SHEET_ROWS
    for k, group in enumerate(sheet_groups(times, per_sheet)):
        cols, rows = sheet_layout(len(group))
        sheet = frames_dir / f"{prefix}-{k + 1:02d}.jpg"
        tmp = sheet.with_name(sheet.stem + ".tmp.jpg")
        vf = f"tile={cols}x{rows}:nb_frames={len(group)}:padding={SHEET_PADDING}:color=white"
        cmd = ["-start_number", str(k * per_sheet), "-i", str(frames_dir / "tile-%04d.jpg")]
        try:
            proc = run_ffmpeg(ffmpeg, cmd + ["-frames:v", "1", "-vf", vf, "-q:v", "3", str(tmp)])
        except subprocess.TimeoutExpired:
            raise FfmpegError("ffmpeg timed out building a contact sheet") from None
        if proc.returncode != 0 or not tmp.is_file():
            lines_err = proc.stderr.strip().splitlines()
            raise FfmpegError(f"ffmpeg could not build a contact sheet: {lines_err[-1] if lines_err else proc.returncode}")
        os.replace(tmp, sheet)
        lines.append(f"SHEET: {sheet.resolve()}")
        lines.append("TIMES: " + " ".join(f"[{format_moment(t, long_form)}]" for t in group))
        for t in group:  # per tile: what is said, then what is heard
            lines += said_lines(segments, [t], long_form) + heard_lines(sounds, [t], long_form)
        tokens += file_tokens(sheet)
    return lines, tokens


def extract_range_tiles(
    ffmpeg: str, fmt: dict[str, Any], start: float, count: int, fps: float, frames_dir: Path, size: tuple[int, int]
) -> tuple[str | None, int]:
    """Decode `count` tiles at `fps` from `start` in one ffmpeg pass into frames/tile-NNNN.jpg.

    Input seeking (`-ss` before `-i`) decodes from the seek point, never from a stream copy, and
    the fps filter takes the frame nearest each moment (within half a source frame), so the
    tiles are time-exact like `--at` frames. Returns an error message (or None) and the number
    of tiles written."""
    for old in frames_dir.glob("tile-*"):
        old.unlink()
    tw, th = scaled_size(size[0], size[1], TILE_LONG_SIDE)
    pad = f"scale={tw}:{th}:force_original_aspect_ratio=decrease,pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2"
    cmd = [
        *stream_input_args(fmt), "-ss", f"{start:.3f}", "-i", fmt["url"], "-t", f"{count / fps:.3f}",
        "-an", "-sn", "-vf", f"fps={fps:g}:round=near,{pad}", "-frames:v", str(count), "-start_number", "0",
        "-q:v", "3", str(frames_dir / "tile-%04d.jpg"),
    ]
    try:
        proc = run_ffmpeg(ffmpeg, cmd, timeout=600)
    except subprocess.TimeoutExpired:
        return "ffmpeg timed out", 0
    produced = len(list(frames_dir.glob("tile-*.jpg")))
    if proc.returncode != 0 or not produced:
        lines = proc.stderr.strip().splitlines()
        return (lines[-1] if lines else "ffmpeg produced no frame"), 0
    return None, produced


def label_tiles(ffmpeg: str, frames_dir: Path, times: list[float], long_form: bool, notes: list[str]) -> None:
    """Stamp each range tile with its moment, as overview tiles are; without a usable font the
    labels are dropped with a note (the TIMES lines still map the tiles)."""
    font = label_font()
    for i, t in enumerate(times):
        tile = frames_dir / f"tile-{i:04d}.jpg"
        size = jpeg_size(tile.read_bytes()) or (TILE_LONG_SIDE, TILE_LONG_SIDE)
        tmp = tile.with_name(tile.stem + ".tmp.jpg")
        vf = drawtext_filter(format_moment(t, long_form), font, max(14, min(size) // 13))
        try:
            proc = run_ffmpeg(ffmpeg, ["-i", str(tile), "-vf", vf, "-q:v", "3", str(tmp)])
        except subprocess.TimeoutExpired:
            proc = None
        if proc is None or proc.returncode != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            notes.append("tile labels disabled (drawtext failed); map tiles with the TIMES lines")
            return
        os.replace(tmp, tile)


def _frames_range(source: Source, directory: Path, args: argparse.Namespace, ffmpeg: str) -> int:
    """Dense frames over START-END at --fps, on labeled contact sheets like --overview. Sites
    seek in a stream URL (no download; other formats as fallbacks), local files in place."""
    info = _video_info(source, directory, args, ffmpeg)
    frames_dir = directory / "frames"
    duration = info.get("duration")
    long_form = (duration or 0) >= 3600
    start, end = args.range_bounds
    notes: list[str] = []
    if duration:
        last = max(0.0, duration - END_MARGIN_S)
        if start > last:
            raise FetchError(
                f"invalid frames request: the range starts at or after the end ({format_duration(duration)})",
                EXIT_ERROR,
                ARGS_HINT,
            )
        if end > last:
            notes.append(
                f"range end {format_moment(end, long_form)} is at or past the end ({format_duration(duration)}); "
                f"used {format_moment(last, long_form)}"
            )
            end = last
    fps = args.fps or RANGE_FPS_DEFAULT
    times = range_times(start, end, fps)
    if source.kind == "file":
        candidates = [_local_format(source, info)]
    else:
        candidates = seek_formats(info.get("formats") or [], RANGE_MAX_SHORT_SIDE)
    if not candidates:
        raise FetchError("no video format to take frames from (the source may be audio only)", EXIT_NO_TRANSCRIPT)
    t0 = time.monotonic()
    err, produced = "no usable video format", 0
    for i, fmt in enumerate(candidates):
        size = (fmt.get("width") or 1280, fmt.get("height") or 720)
        where = "the local file" if source.kind == "file" else f"format {_describe(fmt)}"
        log(f"range: decoding {len(times)} frames from {where}")
        err, produced = extract_range_tiles(ffmpeg, fmt, start, len(times), fps, frames_dir, size)
        if err is None:
            if i:
                notes.append(f"stream seek failed; used fallback format {fmt['format_id']}")
            break
        log(f"range: {where} failed ({err})")
    if err is not None:
        hls = next((f for f in candidates if str(f.get("protocol", "")).startswith("m3u8")), None)
        if source.kind != "file" and hls is not None and hls_format_has_drm(hls):
            raise FetchError("DRM-protected content is not supported (the video stream is encrypted)", EXIT_NO_TRANSCRIPT)
        if source.kind != "file" and is_stream_error(err):
            raise FetchError(f"no frame could be extracted: the video stream failed ({err})", EXIT_ERROR, STREAM_FAIL_HINT)
        raise FfmpegError(f"ffmpeg could not extract the range ({err})")
    if produced < len(times):
        notes.append(f"only {produced} of {len(times)} frames could be decoded (the video may end earlier)")
        times = times[:produced]
    label_tiles(ffmpeg, frames_dir, times, long_form, notes)
    lines, tokens = build_sheets(ffmpeg, directory, times, "range", long_form, notes)
    log(f"{len(times)} range tiles in {time.monotonic() - t0:.1f}s")
    print_frames_summary(source.key, info, "range", lines, len(times), tokens, notes)
    return EXIT_OK


# ---------------------------------------------------------------------------
# sounds: the sound map on its own, and open-text sound queries


def _sound_source(
    source: Source, directory: Path, args: argparse.Namespace, ffmpeg: str
) -> tuple[Callable[[], tuple[Path, int | None]], dict[str, Any]]:
    """Metadata and an audio getter for `sounds`: a local file is probed and read in place; a
    site video is extracted with yt-dlp and its audio downloaded when needed."""
    if source.kind == "file":
        info = file_info(probe_file(find_ffprobe(ffmpeg), source.path), source.path)
        return file_audio(source.path, info.get("audio_index")), info
    import yt_dlp

    with yt_dlp.YoutubeDL(ydl_options(args, directory)) as ydl:
        info = extract_info(ydl, source.url, bool(args.cookies_from_browser))
    return site_audio(info, directory, args), info


def print_sounds_header(key: str, title: str | None, duration: float | None, lines: list[str], notes: list[str]) -> None:
    out = [f"VIDEO_ID: {key}", f"TITLE: {title or ''}", f"DURATION: {format_duration(duration)}", *lines]
    print("\n".join(out + [f"NOTE: {n}" for n in notes]), flush=True)


def cmd_sounds(args: argparse.Namespace) -> int:
    """`sounds <input>`: the cached sound map's summary (built when missing or outdated).
    `sounds <input> --query "a, b"`: when those sounds happen (open-text prompts), without
    changing the sound map. Both re-acquire the audio when needed and delete it afterwards."""
    source = resolve_source(args.url)
    query = None
    if args.query is not None:
        try:
            query = parse_query(args.query)
        except ValueError as exc:
            raise FetchError(f"invalid sounds request: {exc}", EXIT_ERROR, SOUNDS_ARGS_HINT) from None
    elif args.threshold is not None:
        raise FetchError("invalid sounds request: --threshold only applies to --query", EXIT_ERROR, SOUNDS_ARGS_HINT)
    low, high = SOUND_THRESHOLD_RANGE
    if args.threshold is not None and not (math.isfinite(args.threshold) and low <= args.threshold <= high):
        raise FetchError(
            f"invalid sounds request: --threshold must be a logit from {low:g} to {high:g}", EXIT_ERROR, SOUNDS_ARGS_HINT
        )
    ffmpeg = find_ffmpeg(SOUNDS_FFMPEG_HINT)  # before any network work
    if source.kind == "file":
        find_ffprobe(ffmpeg)
    directory = work_dir(source.key)
    directory.mkdir(parents=True, exist_ok=True)
    acquire_lock(directory)  # exit 1 if a fetch or frames run is using this video
    try:
        if query is not None:
            return _sounds_query(source, directory, args, ffmpeg, query)
        return _sounds_map(source, directory, args, ffmpeg)
    finally:
        remove_downloads(directory)
        release_lock(directory)
        if has_results(directory):
            os.utime(directory)
        else:
            shutil.rmtree(directory, ignore_errors=True)  # failed first run: leave nothing behind


def _sound_failure(exc: SoundUnavailable) -> FetchError:
    hint = SOUNDS_SETUP_HINT if exc.setup else SOUNDS_NO_GPU_HINT if exc.no_gpu else SOUNDS_HINT if exc.gpu else None
    return FetchError(f"sound map unavailable: {exc}", EXIT_NO_TRANSCRIPT, hint)


def _sounds_map(source: Source, directory: Path, args: argparse.Namespace, ffmpeg: str) -> int:
    data = current_sounds(directory)
    cached = data is not None
    notes: list[str] = []
    if not cached:
        acquire, info = _sound_source(source, directory, args, ffmpeg)
        try:
            data = build_sound_map(
                directory, acquire, info.get("title"), source_meta(source, info, args.url), info.get("duration")
            )
        except SoundUnavailable as exc:
            raise _sound_failure(exc) from None
        notes = [f"sound map: {n}" for n in data.get("notes") or []]
        meta_path = directory / "meta.json"
        if meta_path.is_file():  # keep a cached transcript's record in step
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["sounds"] = sound_record(data)
                write_meta(directory, meta)
            except (OSError, ValueError):
                pass
    lines = [*sound_lines(directory, data), f"CACHED: {'yes' if cached else 'no'}"]
    print_sounds_header(source.key, data.get("title"), sounds_duration(data), lines, notes)
    return EXIT_OK


def _sounds_query(source: Source, directory: Path, args: argparse.Namespace, ffmpeg: str, query: list[str]) -> int:
    acquire, info = _sound_source(source, directory, args, ffmpeg)
    threshold = args.threshold if args.threshold is not None else SOUND_THRESHOLD
    try:
        check_gpu_free()  # before any download
        try:
            src, stream = acquire()
        except FetchError as exc:
            raise SoundUnavailable(f"the audio could not be fetched ({exc})") from None
        pcm = to_sound_pcm(ffmpeg, src, directory, stream)
        if is_inside(src, directory):
            src.unlink(missing_ok=True)
        duration = pcm_duration(pcm)
        prompts = list(dict.fromkeys(["music", *query]))  # music decides where the music margin applies
        tracks, levels, _ = run_sound_backend(pcm, prompts, directory, silent_stretches(ffmpeg, pcm, duration))
    except SoundUnavailable as exc:
        raise _sound_failure(exc) from None
    tracks = silence_gate(tracks, levels)
    gate = music_gate(tracks["music"])
    spans = []
    for q in query:
        # The coverage labels are measured as in the map: music never against its own baseline.
        found = coverage_spans(q, tracks[q], threshold) if q.lower() in SOUND_COVERAGE_LABELS else event_spans(q, tracks[q], gate, threshold)
        spans += [s | {"end": min(s["end"], round(duration, 2))} for s in found if s["start"] < duration]
    long_form = (info.get("duration") or duration) >= 3600
    notes = [
        query_miss_note(q, tracks[q], threshold, long_form) for q in query if not any(s["label"] == q for s in spans)
    ]
    lines = [f"QUERY: {', '.join(query)}", *event_lines(spans, long_form)]
    print_sounds_header(source.key, info.get("title"), info.get("duration") or duration, lines, notes)
    return EXIT_OK


# ---------------------------------------------------------------------------
# Playlists: list entries without downloading


class _CollectingLogger(_StderrLogger):
    """Also keep yt-dlp's messages, so hidden unavailable entries become a NOTE."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, msg: str) -> None:
        super().debug(msg)
        self.messages.append(msg)

    info = debug

    def warning(self, msg: str) -> None:
        super().warning(msg)
        self.messages.append(msg)


def playlist_url(text: str) -> str:
    """The URL to list. A YouTube `watch?v=...&list=...` URL lists its `list`; a YouTube channel
    URL without a tab lists its videos tab (the channel root lists only its tabs)."""
    raw = strip_input(text)
    if not _SCHEME_RE.match(raw) and _DOMAIN_RE.match(raw):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in ("http", "https"):
        raise FetchError(f"playlist needs an http(s) URL: {raw}", EXIT_NO_TRANSCRIPT)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host == "youtu.be" or host in YOUTUBE_HOSTS:
        list_id = (parse_qs(parsed.query).get("list") or [None])[0]
        if list_id:
            return f"https://www.youtube.com/playlist?list={list_id}"
        parts = [p for p in parsed.path.split("/") if p]
        handle = len(parts) == 1 and parts[0].startswith("@")
        if handle or (len(parts) == 2 and parts[0] in {"channel", "c", "user"}):
            return f"https://www.youtube.com/{'/'.join(parts)}/videos"
    return raw


_UNAVAILABLE_TITLES = {"[private video]": "private", "[deleted video]": "deleted", "[unavailable video]": "unavailable"}
_UNAVAILABLE_AVAILABILITY = {
    "private": "private",
    "needs_auth": "needs login",
    "premium_only": "premium only",
    "subscriber_only": "members only",
}
_ENTRY_LIVE_MARKERS = {"is_live": "(live now)", "is_upcoming": "(upcoming)", "post_live": "(processing)"}


def entry_marker(entry: dict[str, Any]) -> str:
    """Marker for an entry that fetch cannot read: unavailable, private, live or upcoming."""
    title = " ".join(str(entry.get("title") or "").split()).lower()
    reason = _UNAVAILABLE_TITLES.get(title) or _UNAVAILABLE_AVAILABILITY.get(entry.get("availability") or "")
    if reason:
        return f"(unavailable: {reason})"
    return _ENTRY_LIVE_MARKERS.get(entry.get("live_status") or "", "")


def entry_line(index: int, entry: dict[str, Any]) -> str:
    """`ENTRY: <index> [<duration or ?>] <url> [marker] <title>`."""
    duration = entry.get("duration")
    url = entry.get("url") or entry.get("webpage_url") or entry.get("original_url") or "?"
    title = " ".join(str(entry.get("title") or "").split())
    parts = [f"ENTRY: {index}", f"[{format_duration(duration) if duration else '?'}]", url, entry_marker(entry), title]
    return " ".join(p for p in parts if p)


def playlist_lines(
    info: dict[str, Any], entries: list[dict[str, Any] | None], limit: int, notes: list[str] = (),
    model: str = DEFAULT_MODEL, device: str = "cuda",
) -> list[str]:
    """The `playlist` stdout. `entries` may hold up to limit + 1 items (one more than listed
    tells that the playlist was truncated)."""
    listed = entries[:limit]
    truncated = len(entries) > limit
    total = info.get("playlist_count")
    if total is None and not truncated:
        total = len(entries)
    real = [e for e in listed if e]
    readable = [e for e in real if not entry_marker(e)]
    known = [float(e["duration"]) for e in real if e.get("duration")]
    estimate = sum(whisper_estimate_min(e.get("duration"), model=model, device=device) for e in readable)
    out = [
        f"PLAYLIST: {' '.join(str(info.get('title') or info.get('id') or '').split())}",
        f"CHANNEL: {info.get('uploader') or info.get('channel') or ''}",
        f"COUNT: {len(listed)}",
        f"TOTAL: {total if total is not None else 'unknown'}",
        f"DURATION_TOTAL: {format_duration(sum(known))}",
        f"ESTIMATE: ~{estimate:.1f} min if every listed video needed Whisper",
    ]
    out += [entry_line(e.get("playlist_index") or i + 1, e) for i, e in enumerate(listed) if e]
    notes = list(notes)
    if truncated:
        notes.append(f"truncated to the first {limit} entries (--limit {limit}); the playlist has {total or 'more'}")
    if len(real) < len(listed):
        notes.append(f"{len(listed) - len(real)} entries returned no data and are not listed")
    unreadable = len(real) - len(readable)
    if unreadable:
        notes.append(f"{unreadable} entries are unavailable, live or upcoming; ESTIMATE leaves them out")
    unknown = sum(1 for e in readable if not e.get("duration"))
    if unknown:
        notes.append(f"{unknown} entries have no known duration; DURATION_TOTAL and ESTIMATE leave their audio out")
    return out + [f"NOTE: {n}" for n in notes]


def cmd_playlist(args: argparse.Namespace) -> int:
    if args.limit < 1:
        raise FetchError("invalid playlist request: --limit must be at least 1", EXIT_ERROR, "pass a positive --limit")
    import yt_dlp
    from yt_dlp.utils import DownloadError

    url = playlist_url(args.url)
    logger = _CollectingLogger()
    opts = ydl_options(args, WORK_ROOT) | {
        "logger": logger,
        "noplaylist": False,
        "extract_flat": "in_playlist",  # titles, durations and URLs only; nothing is downloaded
        "playlistend": args.limit + 1,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except DownloadError as exc:
            raise classify_ytdlp_error(str(exc), bool(args.cookies_from_browser)) from None
    if not info:
        raise FetchError("yt-dlp returned no metadata", EXIT_ERROR, f"try: {UPGRADE_HINT}")
    if info.get("_type") not in PLAYLIST_TYPES:
        raise FetchError("not a playlist: this URL is a single video", EXIT_NO_TRANSCRIPT, "use fetch for a single video")
    hidden = [m.strip() for m in logger.messages if "unavailable" in m.lower() and "hidden" in m.lower()]
    notes = [f"yt-dlp: {m}" for m in hidden]
    lines = playlist_lines(info, list(info.get("entries") or []), args.limit, notes, setting("model"), resolve_device())
    print("\n".join(lines), flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------
# clean


def _looks_like_path(raw: str) -> bool:
    return bool(_DRIVE_RE.match(raw)) or "\\" in raw or "/" in raw or Path(raw).suffix.lower() in MEDIA_EXTENSIONS


def _stored_sources(directory: Path) -> list[dict[str, Any]]:
    out = []
    for name in ("meta.json", "frames/info.json", "sounds.json"):
        try:
            out.append(json.loads((directory / name).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return [d for d in out if isinstance(d, dict)]


def _work_dir_of(raw: str, root: Path) -> Path | None:
    """The work dir (a direct child of the root) that a path inside the work root belongs to."""
    if not raw.lower().startswith("file:") and not _looks_like_path(raw):
        return None
    try:
        path = (file_uri_path(raw) if raw.lower().startswith("file:") else Path(raw).expanduser()).resolve()
        parts = path.relative_to(root.resolve()).parts
    except (OSError, ValueError):
        return None
    return root / parts[0] if parts else None


def matching_work_dirs(text: str, root: Path | None = None) -> list[Path] | None:
    """Work dirs a `clean` argument names, found without network access.

    Accepts a work key (dir name), a work dir's path or a file inside it (the `TRANSCRIPT:`
    path), a YouTube id or URL, any other URL (its offline key, plus dirs whose stored URL
    matches) or a local path (its key, plus dirs whose stored path matches, so earlier versions
    of an edited or deleted file are found too). Returns None when the argument is none of
    these. Links and junctions under the root are never returned.
    """
    root = root or WORK_ROOT
    raw = strip_input(text)
    if WORK_KEY_RE.match(raw) and (root / raw).is_dir() and not is_link(root / raw):
        return [root / raw]
    inside = _work_dir_of(raw, root)
    if inside is not None:
        return [inside] if inside.is_dir() and not is_link(inside) else []
    found: set[Path] = set()
    urls: set[str] = set()
    path: str | None = None
    try:
        source = resolve_source(raw)
    except FetchError:
        source = None
        if _SCHEME_RE.match(raw) and not _DRIVE_RE.match(raw) and not raw.lower().startswith("file:"):
            urls = {raw}
        elif raw.lower().startswith("file:") or _looks_like_path(raw):
            local = file_uri_path(raw) if raw.lower().startswith("file:") else Path(raw).expanduser()
            path = os.path.normcase(str(local.resolve()))
        else:
            return None
    if source is not None:
        if (root / source.key).is_dir() and not is_link(root / source.key):
            found.add(root / source.key)
        if source.kind == "file":
            path = os.path.normcase(str(source.path))
        else:
            urls = {raw, source.url}
    for directory in work_dirs(root):
        for stored in _stored_sources(directory):
            if stored.get("url") in urls or stored.get("input_url") in urls:
                found.add(directory)
            elif path and stored.get("source_path") and os.path.normcase(stored["source_path"]) == path:
                found.add(directory)
    return sorted(found)


def cmd_clean(args: argparse.Namespace) -> int:
    if args.all:
        targets = work_dirs(WORK_ROOT)
    else:
        targets = matching_work_dirs(args.video_id or "")
        if targets is None:
            print(f"ERROR: invalid video id: {args.video_id} (not a video id, URL, local file or work key)")
            return EXIT_ERROR
    removed = 0
    root = WORK_ROOT.resolve()
    for target in targets:
        # Only direct children of the work root are ever removed; local originals live elsewhere.
        if target.is_dir() and not is_link(target) and target.resolve().parent == root:
            shutil.rmtree(target, ignore_errors=True)
            removed += not target.exists()
    print(f"REMOVED: {removed}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# doctor, setup and config. These three run with the standard library alone, so an installer can
# call them before the heavy environment (yt-dlp, faster-whisper, CUDA libraries) exists:
#     uv run --no-project --python ">=3.11" python video_watch.py doctor
# `setup` without --dry-run needs that environment and starts `uv run video_watch.py setup ...`
# itself when it is missing.

# Recommendation thresholds on total VRAM (MiB as nvidia-smi reports it; an "8 GB" card shows
# about 8188, a "4 GB" card about 4096).
RECOMMEND_LARGE_MB = 7800
RECOMMEND_TURBO_MB = 3900
RECOMMEND_SOUNDS_MB = 6000
MIN_CUDA_DRIVER = 525  # CUDA 12 (CTranslate2 4.5+ wheels) needs driver 525 or newer
WHISPER_ENV_SIZE_MB = {"win32": 1400, "linux": 1500, "darwin": 200}  # uv env of video_watch.py


def heavy_env_present() -> bool:
    """True when this interpreter has the transcription packages (run as `uv run video_watch.py`)."""
    return all(importlib.util.find_spec(name) is not None for name in ("faster_whisper", "ctranslate2", "yt_dlp"))


@dataclass
class GpuInfo:
    name: str | None = None
    total_mb: int | None = None
    free_mb: int | None = None
    driver: str | None = None


def gpu_info() -> GpuInfo:
    row = nvidia_smi_query("name,memory.total,memory.free,driver_version")
    if not row or len(row) < 4:
        return GpuInfo()

    def num(text: str) -> int | None:
        try:
            return int(float(text))
        except ValueError:
            return None

    return GpuInfo(row[0], num(row[1]), num(row[2]), row[3])


def driver_status(gpu: GpuInfo) -> tuple[bool, str]:
    """Whether an NVIDIA GPU is present with a driver new enough for CUDA 12 (Whisper's CTranslate2
    and the sound helper's PyTorch both need it)."""
    if not gpu.name:
        return False, "no NVIDIA GPU or driver found (nvidia-smi is missing or failed)"
    try:
        major = int((gpu.driver or "0").split(".")[0])
    except ValueError:
        major = 0
    if major < MIN_CUDA_DRIVER:
        return False, f"NVIDIA driver {gpu.driver or 'unknown'} is too old for CUDA 12 (needs {MIN_CUDA_DRIVER} or newer)"
    return True, f"driver {gpu.driver}"


def cuda_status(gpu: GpuInfo, platform: str | None = None, deep: bool | None = None) -> tuple[bool, str]:
    """Whether Whisper can use CUDA here and why not. Read-only and without downloads: nvidia-smi
    and the driver version, plus CTranslate2's own device check when its package is installed."""
    if platform_key(platform) == "darwin":
        return False, "macOS has no CUDA (CPU mode only; Metal is not supported yet)"
    if not gpu.name:
        return False, "no NVIDIA GPU or driver found (nvidia-smi is missing or failed)"
    driver_ok, why = driver_status(gpu)
    if not driver_ok:
        return False, why
    deep = heavy_env_present() if deep is None else deep
    if not deep:
        return True, f"ok (driver {gpu.driver}; CTranslate2 is checked on the first run)"
    if not nvidia_lib_dirs():
        return False, "the CUDA libraries (nvidia-cublas-cu12, nvidia-cudnn-cu12) are not installed in this environment"
    try:
        setup_cuda_libs()
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001 - any failure here means no usable CUDA
        return False, f"CTranslate2 could not use CUDA ({type(exc).__name__}: {exc})"
    if count < 1:
        return False, "CTranslate2 sees no CUDA device"
    return True, f"ok (driver {gpu.driver}, CTranslate2 {getattr(ctranslate2, '__version__', '?')})"


def recommend(gpu: GpuInfo, cuda_ok: bool) -> tuple[str, str, bool, str]:
    """(model, device, sound map yes/no, reason for the sound map) for this machine. Only a
    recommendation: any model can be chosen (with a WARN when it may not fit)."""
    total = gpu.total_mb or 0
    if not cuda_ok:
        return "small", "cpu", False, "it needs an NVIDIA GPU (CUDA)"
    model = "large-v3" if total >= RECOMMEND_LARGE_MB else "large-v3-turbo" if total >= RECOMMEND_TURBO_MB else "small"
    if total >= RECOMMEND_SOUNDS_MB:
        why = (
            f"NVIDIA GPU with {total / 1024:.0f} GB; it needs about 4.5 GB of free VRAM while it runs, "
            f"a one-time download of about {sound_setup_size_gb():.0f} GB and about +1 min per hour of video"
        )
        return model, "cuda", True, why
    return model, "cuda", False, f"it needs about 4.5 GB of free VRAM; this GPU has {total / 1024:.1f} GB in total"


def model_fit_warning(model: str, total_mb: int | None) -> str | None:
    """A WARN when the chosen model may not fit the GPU (never a refusal)."""
    if not total_mb:
        return None
    smallest = model_vram_mb(model, BATCH_CHOICES[-1]) + VRAM_MARGIN_MB
    if smallest > total_mb:
        return (
            f"{model} needs about {smallest / 1024:.1f} GB of VRAM even at batch 1; this GPU has "
            f"{total_mb / 1024:.1f} GB, so it will likely run out of memory (large-v3-turbo or small fit better)"
        )
    if model_vram_mb(model, BATCH_SIZE) + VRAM_MARGIN_MB > total_mb:
        return (
            f"memory may be tight for {model} on {total_mb / 1024:.1f} GB of VRAM: it runs with a smaller batch "
            f"(slower) and fails while other apps hold VRAM"
        )
    return None


def tool_version(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False, encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (proc.stdout or proc.stderr or "").strip().splitlines()
    return lines[0].strip() if proc.returncode == 0 and lines else None


def disk_free_gb(path: Path) -> float | None:
    """Free space on the drive of `path` (its nearest existing parent)."""
    for candidate in [path, *path.parents]:
        try:
            if candidate.exists():
                return round(shutil.disk_usage(candidate).free / 1e9, 1)
        except OSError:
            return None
    return None


def cmd_doctor(args: argparse.Namespace) -> int:
    """Read-only environment report as `KEY: value` lines. Never downloads and never writes."""
    import platform as platform_mod

    warns: list[str] = []
    out = [
        f"OS: {platform_mod.system()} {platform_mod.release()} ({platform_mod.machine()})",
        f"PYTHON: {sys.version.split()[0]}",
    ]
    uv = find_uv()
    uv_version = tool_version([uv, "--version"]) if uv else None
    out.append(f"UV: {uv_version or uv or 'missing'}")
    if not uv:
        warns.append("uv was not found on PATH; install it from https://docs.astral.sh/uv/ (the tool runs with uv run)")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    ff_version = tool_version([ffmpeg, "-version"]) if ffmpeg else None
    blocked = {name: blocked_executable(name) for name, found in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if not found}
    blocked = {name: b for name, b in blocked.items() if b}

    def tool_line(name: str, found: str | None, shown: str | None) -> str:
        if found:
            return shown or found
        return f"blocked: {blocked[name][0]} ({blocked[name][1]})" if name in blocked else "missing"

    out.append(f"FFMPEG: {tool_line('ffmpeg', ffmpeg, ff_version.split(' Copyright')[0] if ff_version else None)}")
    out.append(f"FFPROBE: {tool_line('ffprobe', ffprobe, None)}")
    if blocked:
        warns.append(f"ffmpeg was found but cannot be run by this process; {SANDBOX_HINT}")
    elif not ffmpeg or not ffprobe:
        warns.append(
            f"ffmpeg/ffprobe missing: frames, local files and the sound map need them (install: {FFMPEG_INSTALL}); "
            "transcripts of URLs work without them"
        )
    gpu = gpu_info()
    out.append(f"GPU: {f'{gpu.name} (driver {gpu.driver})' if gpu.name else 'none'}")
    out.append(f"VRAM_TOTAL_MB: {gpu.total_mb if gpu.total_mb is not None else 'unknown'}")
    out.append(f"VRAM_FREE_MB: {gpu.free_mb if gpu.free_mb is not None else 'unknown'}")
    cuda_ok, cuda_reason = cuda_status(gpu)
    out.append(f"CUDA: {cuda_reason if cuda_ok else 'unavailable: ' + cuda_reason}")

    path = config_path()
    try:
        config = load_config(path)
    except ConfigError as exc:
        warns.append(f"{exc}; defaults are shown ({CONFIG_HINT})")
        config = {}
    values: dict[str, Any] = {}
    for key in CONFIG_KEYS:
        try:
            values[key] = resolve_setting(key, None, config)[0]
        except ConfigError as exc:
            warns.append(f"{exc}; the default is shown")
            values[key] = default_setting(key)
    root = Path(values["model_dir"])
    sound_cache = Path(values["sound_cache"]) if values["sound_cache"] else hf_cache_dir()
    model_ok = model_cached(values["model"], root)
    sound_ok = sound_model_cached_in(sound_cache)
    helper_ok = sound_helper_ready()
    free_gb = disk_free_gb(root)
    out += [
        f"DISK_FREE_GB: {free_gb if free_gb is not None else 'unknown'}",
        f"CONFIG: {path}{'' if path.is_file() else ' (not created yet)'}",
        f"DEVICE: {values['device']}",
        f"MODEL: {values['model']}",
        f"MODEL_DIR: {root}",
        f"MODEL_CACHED: {'yes' if model_ok else 'no'}",
        f"SOUNDS: {'on' if values['sounds'] else 'off'}",
        f"SOUND_MODEL_CACHED: {'yes' if sound_ok else 'no'}",
        f"SOUND_HELPER_READY: {'yes' if helper_ok else 'no'}",
        f"WORK_DIR: {values['work_dir']}",
    ]
    rec_model, rec_device, rec_sounds, why = recommend(gpu, cuda_ok)
    out += [
        f"RECOMMEND_MODEL: {rec_model}",
        f"RECOMMEND_DEVICE: {rec_device}",
        f"RECOMMEND_SOUNDS: {'yes' if rec_sounds else 'no'} ({why})",
    ]
    device = values["device"]
    device_ok = device == "cpu" or cuda_ok
    if not device_ok:
        warns.append(
            f"device is {device} but CUDA is unavailable ({cuda_reason}); fetch would stop with an error. "
            "CPU mode is slow but works: setup --device cpu --model small"
        )
    if device == "cpu":
        warns.append("device is cpu: transcription is several times slower than on an NVIDIA GPU")
    elif cuda_ok:
        fit = model_fit_warning(values["model"], gpu.total_mb)
        if fit:
            warns.append(fit)
        if gpu.free_mb is not None and gpu.free_mb < model_vram_mb(values["model"], 1) + VRAM_MARGIN_MB:
            warns.append(f"only {gpu.free_mb} MB of VRAM are free right now; close other GPU apps before a fetch")
    if not model_ok:
        size = WHISPER_MODELS[values["model"]]["size_mb"] / 1000
        warns.append(f"the Whisper model {values['model']} is not downloaded yet (about {size:.1f} GB; setup downloads it)")
    if values["sounds"]:
        if not gpu.name:
            warns.append("the sound map is on but needs an NVIDIA GPU; it will be skipped")
        elif not (sound_ok and helper_ok):
            warns.append(f"the sound map is on but not set up (setup --sounds on, about {sound_setup_size_gb():.0f} GB)")
    if free_gb is not None and free_gb < 10:
        warns.append(f"only {free_gb} GB free on the model drive")
    work_problem = work_root_problem(Path(values["work_dir"]))
    if work_problem:
        warns.append(f"{work_problem}; fetch, frames and sounds would stop with an error ({WORK_DIR_HINT})")
    sounds_ok = sound_ok and helper_ok and bool(gpu.name)
    ready = device_ok and model_ok and bool(ffmpeg and ffprobe) and (not values["sounds"] or sounds_ok) and not work_problem
    out += [f"WARN: {w}" for w in warns]
    out.append(f"READY: {'yes' if ready else 'no'}")
    print("\n".join(out), flush=True)
    return EXIT_OK


def sound_model_cached_in(cache: Path) -> bool:
    snapshots = cache / ("models--" + SOUND_MODEL.replace("/", "--")) / "snapshots"
    try:
        return any(any(d.glob("*.safetensors")) or any(d.glob("*.bin")) for d in snapshots.iterdir())
    except OSError:
        return False


def download_whisper_model(name: str, root: Path) -> None:
    """Download a Whisper model into the model dir (progress on stderr)."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from faster_whisper.utils import download_model

    root.mkdir(parents=True, exist_ok=True)
    download_model(name, cache_dir=str(root))


def prepare_sound_helper() -> None:
    """Install the sound helper's environment (PyTorch with CUDA 12.8, transformers) and download
    the sound model, without the GPU. Raises SoundUnavailable."""
    uv = find_uv()
    if not uv:
        raise SoundUnavailable("uv was not found on PATH")
    cmd = [uv, "run", "--quiet", "--script", str(SOUND_HELPER), "--prepare", SOUND_MODEL]
    try:
        code = run_helper(cmd, helper_env(), SOUND_PREPARE_TIMEOUT_S)
    except OSError as exc:
        raise SoundUnavailable(f"uv could not start the sound helper ({exc})") from None
    if code != 0:
        raise SoundUnavailable(f"preparing the sound helper failed (exit {code}; see stderr)")
    if not sound_model_cached():
        raise SoundUnavailable("the sound model is still missing after the download")
    mark_sound_helper_ready()


def cmd_setup(args: argparse.Namespace) -> int:
    """Apply the user's choices non-interactively: write the config and pre-download what they
    need. Prints the downloads with their sizes first; downloads need --yes (the user agreed)."""
    path = config_path()
    try:
        config = load_config(path)
    except ConfigError as exc:
        raise FetchError(str(exc), EXIT_ERROR, "fix or delete the config file, then rerun setup") from None
    try:
        device = parse_config_value("device", args.device) if args.device else resolve_setting("device", None, config)[0]
        model = parse_config_value("model", args.model) if args.model else resolve_setting("model", None, config)[0]
        sounds = parse_bool(args.sounds) if args.sounds is not None else resolve_setting("sounds", None, config)[0]
        root = Path(parse_config_value("model_dir", args.model_dir)) if args.model_dir else Path(resolve_setting("model_dir", None, config)[0])
    except (ValueError, ConfigError) as exc:
        raise FetchError(f"invalid setup request: {exc}", EXIT_ERROR, "see: setup --help") from None
    gpu = gpu_info()
    cuda_ok, cuda_reason = cuda_status(gpu)
    if device != "cpu" and not cuda_ok:
        raise FetchError(
            f"CUDA is unavailable for device {device}: {cuda_reason}",
            EXIT_ERROR,
            "CPU mode is slow but works without an NVIDIA GPU: setup --device cpu --model small --sounds off",
        )
    if sounds:
        if not gpu.name:
            raise FetchError(
                "the sound map needs an NVIDIA GPU, and none was found", EXIT_ERROR, "rerun setup with --sounds off"
            )
        driver_ok, why = driver_status(gpu)
        if not driver_ok:
            raise FetchError(
                f"the sound map needs an NVIDIA GPU with CUDA 12: {why}",
                EXIT_ERROR,
                f"update the NVIDIA driver to {MIN_CUDA_DRIVER} or newer, or rerun setup with --sounds off",
            )

    sound_cache = sound_cache_dir()
    downloads: list[tuple[str, int]] = []
    if not heavy_env_present():
        size = WHISPER_ENV_SIZE_MB[platform_key()]
        downloads.append((f"python packages of video_watch.py (yt-dlp, faster-whisper{', CUDA libraries' if platform_key() != 'darwin' else ''}) into the uv cache (less if uv already has them)", size))
    whisper_needed = not model_cached(model, root)
    if whisper_needed:
        downloads.append((f"whisper model {model} into {root}", WHISPER_MODELS[model]["size_mb"]))
    env_needed = bool(sounds) and not sound_helper_ready()
    model_needed = bool(sounds) and not sound_model_cached()
    if env_needed:
        downloads.append((
            "sound map environment (PyTorch with CUDA 12.8, transformers) into the uv cache "
            "(less if uv already has it)", SOUND_ENV_SIZE_MB,
        ))
    if model_needed:
        downloads.append((f"sound model {SOUND_MODEL} into {sound_cache}", SOUND_MODEL_SIZE_MB))
    total_gb = sum(mb for _, mb in downloads) / 1000
    updates = {"device": device, "model": model, "sounds": bool(sounds)}
    if args.model_dir:
        updates["model_dir"] = str(root)
    out = [
        f"DEVICE: {device}",
        f"MODEL: {model}",
        f"SOUNDS: {'on' if sounds else 'off'}",
        f"MODEL_DIR: {root}",
        *[f"DOWNLOAD: {what}, about {mb / 1000:.1f} GB" for what, mb in downloads],
        f"DOWNLOAD_TOTAL_GB: {total_gb:.1f}",
    ]
    warn = model_fit_warning(model, gpu.total_mb) if device != "cpu" else None
    out += [f"WARN: {warn}"] if warn else []
    if device == "cpu":
        out.append("WARN: CPU mode is several times slower than an NVIDIA GPU")
    free_gb = disk_free_gb(root)
    if free_gb is not None and free_gb < total_gb + 1:
        out.append(f"WARN: only {free_gb} GB free on the model drive")
    delegate = not args.dry_run and not heavy_env_present() and (args.yes or not downloads)
    if not delegate:  # a delegated run prints the plan itself
        print("\n".join(out), flush=True)
    if args.dry_run:
        print(f"CONFIG: {path} (would write {json.dumps(updates)})")
        print("SETUP: dry-run")
        return EXIT_OK
    if downloads and not args.yes:
        raise FetchError(
            f"setup needs to download about {total_gb:.1f} GB",
            EXIT_ERROR,
            "tell the user the sizes above and rerun with --yes once they agree (--dry-run only shows the plan)",
        )
    if not heavy_env_present():  # the downloads need faster-whisper: rerun inside its environment
        uv = find_uv()
        if not uv:
            raise FetchError("uv was not found on PATH", EXIT_ERROR, "install uv: https://docs.astral.sh/uv/")
        log("installing the python packages of video_watch.py (uv run) and continuing setup there")
        cmd = [uv, "run", "--script", str(Path(__file__).resolve()), *getattr(args, "argv", sys.argv[1:])]
        return subprocess.run(cmd, check=False).returncode
    if whisper_needed:
        log(f"downloading whisper {model} (about {WHISPER_MODELS[model]['size_mb'] / 1000:.1f} GB) into {root}")
        try:
            download_whisper_model(model, root)
        except Exception as exc:  # noqa: BLE001 - network, disk or hub errors
            raise FetchError(f"the Whisper model download failed ({type(exc).__name__}: {exc})", EXIT_ERROR, "check the network and disk space, then rerun setup") from None
        if not model_cached(model, root):
            raise FetchError(f"the Whisper model {model} is still missing after the download", EXIT_ERROR, "rerun setup")
    print(f"MODEL_CACHED: {'yes' if model_cached(model, root) else 'no'}", flush=True)
    if sounds and (env_needed or model_needed):
        log("preparing the sound map: PyTorch environment and model (first time about 5 GB)")
        try:
            prepare_sound_helper()
        except SoundUnavailable as exc:
            raise FetchError(f"sound map setup failed: {exc}", EXIT_ERROR, "rerun setup, or use --sounds off") from None
    if sounds:
        print("SOUND_MODEL_CACHED: yes", flush=True)
    elif sound_model_cached():
        print(
            f"NOTE: the sound map is off; its model stays in the Hugging Face cache {sound_cache} (shared with other "
            "tools) and is used again if the sound map is turned back on"
        )
    save_config(updates, path=path)
    print(f"CONFIG: {path}")
    print("SETUP: done", flush=True)
    return EXIT_OK


def format_setting(key: str, value: Any) -> str:
    if key == "sounds":
        return "true" if value else "false"
    if key == "sound_cache" and not value:
        return str(hf_cache_dir())
    return str(value)


def cmd_config(args: argparse.Namespace) -> int:
    """`config show | path | get KEY | set KEY VALUE | unset KEY`."""
    path = config_path()
    if args.action == "path":
        print(f"CONFIG: {path}")
        return EXIT_OK
    if args.action in ("get", "set", "unset"):
        if not args.key:
            raise FetchError(f"invalid config request: config {args.action} needs a KEY", EXIT_ERROR, f"keys: {', '.join(CONFIG_KEYS)}")
        if args.key not in CONFIG_KEYS:
            raise FetchError(f"invalid config request: unknown key '{args.key}'", EXIT_ERROR, f"keys: {', '.join(CONFIG_KEYS)}")
    if args.action == "set":
        if args.value is None:
            raise FetchError("invalid config request: config set needs a VALUE", EXIT_ERROR, "e.g. config set sounds true")
        try:
            value = parse_config_value(args.key, args.value)
        except ValueError as exc:
            raise FetchError(f"invalid config request: {exc}", EXIT_ERROR, CONFIG_HINT) from None
        if args.key == "work_dir":
            work = Path(value)
            if work.exists() and not (work / WORK_ROOT_MARKER).is_file() and (not work.is_dir() or any(work.iterdir())):
                raise FetchError(f"invalid config request: {work} is not an empty folder", EXIT_ERROR, WORK_DIR_HINT)
        try:
            save_config({args.key: value}, path=path)
        except ConfigError as exc:
            raise FetchError(str(exc), EXIT_ERROR, CONFIG_HINT) from None
        print(f"SET: {args.key} = {format_setting(args.key, value)}")
        print(f"CONFIG: {path}")
        override = os.environ.get(CONFIG_ENV_VARS[args.key], "").strip()
        if override:
            print(f"WARN: {CONFIG_ENV_VARS[args.key]}={override} is set and overrides this value; unset it to use the config file")
        return EXIT_OK
    if args.action == "unset":
        try:
            save_config({}, remove=[args.key], path=path)
        except ConfigError as exc:
            raise FetchError(str(exc), EXIT_ERROR, CONFIG_HINT) from None
        print(f"UNSET: {args.key}")
        print(f"CONFIG: {path}")
        return EXIT_OK
    try:
        config = load_config(path)
    except ConfigError as exc:
        raise FetchError(str(exc), EXIT_ERROR, CONFIG_HINT) from None
    if args.action == "get":
        try:
            print(format_setting(args.key, resolve_setting(args.key, None, config)[0]))
        except ConfigError as exc:
            raise FetchError(str(exc), EXIT_ERROR, CONFIG_HINT) from None
        return EXIT_OK
    lines = [f"CONFIG: {path}{'' if path.is_file() else ' (not created yet)'}"]
    for key in CONFIG_KEYS:
        try:
            value, origin = resolve_setting(key, None, config)
        except ConfigError as exc:
            raise FetchError(str(exc), EXIT_ERROR, CONFIG_HINT) from None
        lines.append(f"{key.upper()}: {format_setting(key, value)} ({origin})")
    print("\n".join(lines), flush=True)
    return EXIT_OK


class _Parser(argparse.ArgumentParser):
    """argparse exits with 2 on usage errors, which would collide with NEEDS_COOKIES."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"ERROR: {message}")
        sys.exit(EXIT_ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="video_watch.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="fetch or generate a transcript")
    fetch.add_argument("url", help="video URL (YouTube or any yt-dlp site), YouTube video id, or local file path")
    fetch.add_argument("--whisper", action="store_true", help="always transcribe with Whisper")
    fetch.add_argument("--lang", help="override the original spoken language (e.g. tr, en)")
    fetch.add_argument("--no-vad", action="store_true", help="disable VAD (sequential, slower)")
    fetch.add_argument("--model", choices=MODEL_NAMES, help="Whisper model (default: config, else large-v3)")
    fetch.add_argument("--device", choices=DEVICES, help="auto or cuda (NVIDIA GPU) or cpu (slow); default: config, else auto")
    sound_flags = fetch.add_mutually_exclusive_group()
    sound_flags.add_argument("--sounds", action="store_const", const=True, default=None, help="build the sound map for this run")
    sound_flags.add_argument("--no-sounds", action="store_true", help="skip the sound map for this run")
    fetch.add_argument("--cookies-from-browser", metavar="BROWSER", help="use cookies from BROWSER[:PROFILE]")
    fetch.set_defaults(func=cmd_fetch)

    frames = sub.add_parser("frames", help="extract still frames: targeted moments or an overview")
    frames.add_argument("url", help="video URL (YouTube or any yt-dlp site), YouTube video id, or local file path")
    mode = frames.add_mutually_exclusive_group(required=True)
    mode.add_argument("--at", metavar="TIMES", help="comma-separated moments: ss, ss.s, m:ss or h:mm:ss")
    mode.add_argument("--overview", action="store_true", help="contact sheets from scene changes and interval fills")
    mode.add_argument("--range", metavar="START-END", help="dense frames over a window on contact sheets, e.g. 1:00-1:10")
    frames.add_argument("--interval", type=float, metavar="SECONDS", help="overview: longest stretch without a pick")
    frames.add_argument("--scene-threshold", type=float, metavar="X", help="overview: scene score threshold (default 0.3)")
    frames.add_argument("--fps", type=float, metavar="N", help=f"range: frames per second (default {RANGE_FPS_DEFAULT:g})")
    frames.add_argument("--max-frames", type=int, default=MAX_FRAMES_DEFAULT, help="safety ceiling per call (default 200)")
    frames.add_argument("--cookies-from-browser", metavar="BROWSER", help="use cookies from BROWSER[:PROFILE]")
    frames.set_defaults(func=cmd_frames)

    sounds = sub.add_parser("sounds", help="the sound map, or when given sounds happen (--query)")
    sounds.add_argument("url", help="video URL (YouTube or any yt-dlp site), YouTube video id, or local file path")
    sounds.add_argument("--query", metavar="TEXT", help='comma-separated sound descriptions, e.g. "car horn, glass breaking"')
    sounds.add_argument(
        "--threshold", type=float, metavar="LOGIT", help=f"query: model logit a sound must reach (default {SOUND_THRESHOLD:g})"
    )
    sounds.add_argument("--cookies-from-browser", metavar="BROWSER", help="use cookies from BROWSER[:PROFILE]")
    sounds.set_defaults(func=cmd_sounds)

    playlist = sub.add_parser("playlist", help="list a playlist's or channel's videos without downloading")
    playlist.add_argument("url", help="playlist, channel or album URL (YouTube watch?v=...&list=... lists the list)")
    playlist.add_argument(
        "--limit", type=int, default=PLAYLIST_LIMIT_DEFAULT, help=f"most entries to list (default {PLAYLIST_LIMIT_DEFAULT})"
    )
    playlist.add_argument("--cookies-from-browser", metavar="BROWSER", help="use cookies from BROWSER[:PROFILE]")
    playlist.set_defaults(func=cmd_playlist)

    clean = sub.add_parser("clean", help="delete cached work dirs")
    target = clean.add_mutually_exclusive_group(required=True)
    target.add_argument("video_id", nargs="?", help="video id, URL, local file path or work key to delete")
    target.add_argument("--all", action="store_true", help="delete all work dirs")
    clean.set_defaults(func=cmd_clean)

    doctor = sub.add_parser("doctor", help="read-only report: tools, GPU, CUDA, settings, recommendations")
    doctor.set_defaults(func=cmd_doctor)

    setup = sub.add_parser("setup", help="apply choices: write the config and pre-download the models")
    setup.add_argument("--device", choices=DEVICES, help="auto (GPU), cuda (GPU) or cpu (slow)")
    setup.add_argument("--model", choices=MODEL_NAMES, help="Whisper model")
    setup.add_argument("--sounds", choices=("on", "off"), help="the sound map (NVIDIA GPU, about 5 GB download)")
    setup.add_argument("--model-dir", metavar="PATH", help="where Whisper models are stored")
    setup.add_argument("--yes", action="store_true", help="the user agreed to the downloads listed")
    setup.add_argument("--dry-run", action="store_true", help="only print what would be downloaded, with sizes")
    setup.set_defaults(func=cmd_setup)

    config = sub.add_parser("config", help="show or change settings: show, path, get KEY, set KEY VALUE, unset KEY")
    config.add_argument("action", choices=("show", "path", "get", "set", "unset"))
    config.add_argument("key", nargs="?", help=f"one of: {', '.join(CONFIG_KEYS)}")
    config.add_argument("value", nargs="?", help="the new value (set)")
    config.set_defaults(func=cmd_config)
    return parser


# Commands that use the work root (purged on every run); doctor, setup and config never touch it.
WORK_COMMANDS = ("fetch", "frames", "sounds", "playlist", "clean")


def maybe_reexec_for_cuda(args: argparse.Namespace) -> None:
    """Linux: restart a GPU fetch once with the CUDA wheels' library dirs on LD_LIBRARY_PATH (the
    loader reads it only at start), same pid, same arguments. No-op elsewhere."""
    if args.command != "fetch" or platform_key() != "linux":
        return
    try:
        if resolve_device(args) == "cpu":
            return
    except ConfigError:
        return
    env = cuda_reexec_env()
    if env is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)


def use_work_root() -> None:
    """Point WORK_ROOT at the configured work_dir (when set) and prepare it."""
    global WORK_ROOT
    value, origin = resolve_setting("work_dir")
    if origin != "default":
        WORK_ROOT = Path(value)
    prepare_work_root(WORK_ROOT)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    if argv is None:
        maybe_reexec_for_cuda(args)
    try:
        if args.command in WORK_COMMANDS:
            for key in CONFIG_KEYS:  # an invalid setting stops here, before any work
                resolve_setting(key, args)
            use_work_root()
            purge()  # safety net on every run: stale work dirs and leftover audio
        return args.func(args)
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        print(f"HINT: {exc.hint or CONFIG_HINT}")
        return EXIT_ERROR
    except FetchError as exc:
        if exc.code == EXIT_NEEDS_COOKIES:
            print("NEEDS_COOKIES")
        print(f"ERROR: {exc}")
        if exc.hint:
            print(f"HINT: {exc.hint}")
        return exc.code
    except KeyboardInterrupt:
        print("ERROR: interrupted")
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print(f"HINT: if extraction failed, try: {UPGRADE_HINT}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())

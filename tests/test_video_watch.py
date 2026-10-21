"""Unit tests for video_watch.py (no network, no GPU; tests/conftest.py puts the script on sys.path)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path, PureWindowsPath

import pytest

import video_watch as yt  # noqa: E402
from video_watch import Segment  # noqa: E402


_REAL_BLOCKED = yt.blocked_executable
_REAL_APPLE_SILICON = yt.apple_silicon


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """No real user config, no VIDEO_WATCH_* variables, no real nvidia-smi. The sound map runs a
    GPU model in a subprocess; tests opt in with SOUND_MAP_DEFAULT."""
    for name in [*yt.CONFIG_ENV_VARS.values(), yt.REEXEC_ENV, "CUDA_VISIBLE_DEVICES"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(yt.CONFIG_ENV, str(tmp_path_factory.mktemp("cfg") / "config.json"))
    monkeypatch.setattr(yt, "SOUND_MAP_DEFAULT", False)
    monkeypatch.setattr(yt, "gpu_free_mb", lambda: None)  # never ask the real nvidia-smi
    monkeypatch.setattr(yt, "blocked_executable", lambda name: None)  # never scan the real PATH
    monkeypatch.setattr(yt, "apple_silicon", lambda platform=None, machine=None: False)  # tests choose their platform
    monkeypatch.setattr(yt, "sound_model_cached", lambda: True)  # the sound map counts as set up
    marker = tmp_path_factory.mktemp("helper") / "sound-helper.json"
    monkeypatch.setattr(yt, "sound_helper_marker", lambda: marker)  # never the real user cache
    yt.mark_sound_helper_ready()  # the sound helper counts as prepared


# --- URL parsing -------------------------------------------------------------

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&t=42s",
        f"https://m.youtube.com/watch?v={VID}",
        f"https://www.youtube.com/watch?v={VID}&list=PL1234567890&index=3",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?si=abcDEF&t=10",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://youtube.com/shorts/{VID}?feature=share",
        f"https://www.youtube.com/live/{VID}",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube-nocookie.com/embed/{VID}",
        f"https://music.youtube.com/watch?v={VID}",
        f"www.youtube.com/watch?v={VID}",
        f"youtu.be/{VID}",
        VID,
    ],
)
def test_parse_video_id_accepts_single_video_urls(url: str) -> None:
    assert yt.parse_video_id(url) == VID


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/playlist?list=PL1234567890",
        "https://www.youtube.com/watch?list=PL1234567890",
        "https://vimeo.com/123456",
        "https://www.youtube.com/@somechannel",
        "https://www.youtube.com/watch?v=short",
        "https://youtu.be/",
        "not a url",
    ],
)
def test_parse_video_id_rejects_non_video_urls(url: str) -> None:
    with pytest.raises(ValueError):
        yt.parse_video_id(url)


def test_parse_video_id_playlist_message() -> None:
    with pytest.raises(ValueError, match="Playlist"):
        yt.parse_video_id("https://www.youtube.com/playlist?list=PL1234567890")


# --- Timestamps --------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,long_form,expected",
    [
        (0, False, "00:00"),
        (5.9, False, "00:05"),
        (65, False, "01:05"),
        (3599, False, "59:59"),
        (0, True, "0:00:00"),
        (65, True, "0:01:05"),
        (3600, True, "1:00:00"),
        (3725.4, True, "1:02:05"),
        (36000, True, "10:00:00"),
        (-3, False, "00:00"),
    ],
)
def test_format_timestamp(seconds: float, long_form: bool, expected: str) -> None:
    assert yt.format_timestamp(seconds, long_form) == expected


def test_format_duration() -> None:
    assert yt.format_duration(754) == "12:34"
    assert yt.format_duration(3807) == "1:03:27"
    assert yt.format_duration(None) == "unknown"


# --- Subtitle parsing --------------------------------------------------------

MANUAL_VTT = """WEBVTT
Kind: captions
Language: tr

00:00:00.000 --> 00:00:02.500
Merhaba &amp; hoş geldiniz.

00:00:02.500 --> 00:00:05.000 align:start position:0%
<i>İkinci</i> satır
devamı burada

00:01:05.250 --> 00:01:07.000
Son cümle.
"""

SRT = """1
00:00:01,000 --> 00:00:02,000
First line

2
00:00:02,500 --> 00:00:04,000
Second
line

3
01:00:00,000 --> 01:00:01,500
Late line
"""

# Shape of YouTube auto-generated VTT: each cue repeats the previous line and adds the
# next one word by word, with 10 ms transition cues in between.
AUTO_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.160 --> 00:00:02.270 align:start position:0%
\x20
hello<00:00:00.480><c> everyone</c><00:00:00.800><c> and</c><00:00:01.120><c> welcome</c>

00:00:02.270 --> 00:00:02.280 align:start position:0%
hello everyone and welcome
\x20

00:00:02.280 --> 00:00:04.630 align:start position:0%
hello everyone and welcome
to<00:00:02.560><c> my</c><00:00:02.880><c> channel</c>

00:00:04.630 --> 00:00:04.640 align:start position:0%
to my channel
\x20

00:00:04.640 --> 00:00:07.000 align:start position:0%
to my channel
today<00:00:05.000><c> we</c><00:00:05.300><c> talk</c>
"""


def test_parse_vtt_manual() -> None:
    cues = yt.parse_subtitles(MANUAL_VTT)
    assert [c.start for c in cues] == [0.0, 2.5, 65.25]
    assert cues[0].text == "Merhaba & hoş geldiniz."
    assert cues[1].text == "İkinci satır\ndevamı burada"
    assert cues[2].end == 67.0
    flat = yt.flatten_cues(cues)
    assert flat[1].text == "İkinci satır devamı burada"


def test_parse_srt() -> None:
    cues = yt.parse_subtitles(SRT)
    assert [(c.start, c.end) for c in cues] == [(1.0, 2.0), (2.5, 4.0), (3600.0, 3601.5)]
    assert cues[1].text == "Second\nline"


def test_parse_vtt_crlf_and_short_timestamps() -> None:
    text = "WEBVTT\r\n\r\n00:01.000 --> 00:02.000\r\nshort form\r\n"
    cues = yt.parse_subtitles(text)
    assert len(cues) == 1 and cues[0].start == 1.0 and cues[0].text == "short form"


def test_clean_rolling_cues_removes_duplicates() -> None:
    segments = yt.clean_rolling_cues(yt.parse_subtitles(AUTO_VTT))
    assert [s.text for s in segments] == [
        "hello everyone and welcome",
        "to my channel",
        "today we talk",
    ]
    assert segments[0].start == pytest.approx(0.16)
    assert segments[1].start == pytest.approx(2.28)
    assert segments[2].start == pytest.approx(4.64)
    # Ends never overlap the next line's start.
    assert all(a.end <= b.start for a, b in zip(segments, segments[1:]))


def test_clean_rolling_cues_merges_growing_line() -> None:
    cues = [
        Segment(0.0, 1.0, "we are"),
        Segment(1.0, 2.0, "we are going"),
        Segment(2.0, 3.0, "we are going home"),
        Segment(3.0, 4.0, "next"),
    ]
    out = yt.clean_rolling_cues(cues)
    assert [(s.start, s.text) for s in out] == [(0.0, "we are going home"), (3.0, "next")]
    assert out[0].end == 3.0


# --- Paragraphs --------------------------------------------------------------


def _segs(spec: list[tuple[float, float, str]]) -> list[Segment]:
    return [Segment(a, b, t) for a, b, t in spec]


def test_group_paragraphs_splits_at_sentence_end_after_min_length() -> None:
    segments = _segs(
        [
            (0, 5, "One."),
            (5, 10, "Two."),
            (10, 16, "Three."),  # >= 15 s and a sentence end: close here
            (16, 20, "Four"),
            (20, 25, "five."),
        ]
    )
    paras = yt.group_paragraphs(segments)
    assert paras == [(0, "One. Two. Three."), (16, "Four five.")]


def test_group_paragraphs_splits_at_pause_without_punctuation() -> None:
    segments = _segs([(0, 8, "no punctuation"), (8, 16, "here either"), (18, 22, "after a pause")])
    assert yt.group_paragraphs(segments) == [(0, "no punctuation here either"), (18, "after a pause")]


def test_group_paragraphs_enforces_max_length() -> None:
    segments = _segs([(i * 4, i * 4 + 4, f"w{i}") for i in range(20)])  # 80 s, no breaks
    paras = yt.group_paragraphs(segments)
    starts = [p[0] for p in paras]
    ends = starts[1:] + [80]
    assert all(e - s <= 30 for s, e in zip(starts, ends))
    assert " ".join(p[1] for p in paras) == " ".join(f"w{i}" for i in range(20))


def test_group_paragraphs_empty() -> None:
    assert yt.group_paragraphs([]) == []


# --- Source selection --------------------------------------------------------

FMT = [{"ext": "vtt", "url": "https://example/vtt"}]


def test_manual_track_in_original_language() -> None:
    track, note = yt.choose_manual_track({"en": FMT, "tr": FMT}, "tr")
    assert track == "tr" and "original language" in note


def test_manual_track_matches_region_variant() -> None:
    assert yt.choose_manual_track({"en-GB": FMT, "de": FMT}, "en")[0] == "en-GB"
    assert yt.choose_manual_track({"en": FMT, "en-US": FMT}, "en-US")[0] == "en-US"
    assert yt.choose_manual_track({"en": FMT}, "en-US")[0] == "en"


def test_manual_track_never_uses_translations() -> None:
    track, note = yt.choose_manual_track({"es": FMT, "fr": FMT}, "en-US")
    assert track is None and "not in the original language" in note


def test_manual_track_unknown_language() -> None:
    # Unknown language means Whisper on every site, YouTube included (2026-10-20): a lone track
    # could be a translation.
    track, note = yt.choose_manual_track({"de": FMT}, None)
    assert track is None and note == "original language unknown; manual subtitles (de) not used, since they could be a translation"
    assert yt.choose_manual_track({"de": FMT, "en": FMT}, None)[0] is None
    assert yt.choose_manual_track({"de": FMT, "live_chat": FMT}, None)[0] is None


def test_manual_track_none_available() -> None:
    assert yt.choose_manual_track({}, "tr") == (None, "no manual subtitles")
    assert yt.choose_manual_track({"live_chat": FMT}, "tr")[0] is None


def test_auto_track_prefers_original_asr() -> None:
    auto = {"tr": FMT, "tr-orig": FMT, "en": FMT, "de": FMT}
    assert yt.choose_auto_track(auto, "tr") == "tr-orig"
    assert yt.choose_auto_track({"en-US-orig": FMT, "en": FMT}, "en") == "en-US-orig"


def test_auto_track_never_uses_translations() -> None:
    # Only translated tracks for the original language: nothing usable.
    assert yt.choose_auto_track({"tr": FMT, "de": FMT, "en-orig": FMT}, "tr") is None
    assert yt.choose_auto_track({"tr": FMT}, None) is None


def test_auto_track_unknown_language() -> None:
    assert yt.choose_auto_track({"tr-orig": FMT, "en": FMT}, None) == "tr-orig"
    assert yt.choose_auto_track({"tr-orig": FMT, "en-orig": FMT}, None) is None


# --- Cache rules -------------------------------------------------------------


def _args(**kw: object) -> argparse.Namespace:
    base = {"whisper": False, "model": None, "no_vad": False, "lang": None}
    return argparse.Namespace(**(base | kw))


WHISPER_META = {"source": "whisper", "language": "tr", "settings": {"model": "large-v3", "vad_filter": True}}
MANUAL_META = {"source": "manual-subtitles", "language": "tr", "settings": {"track": "tr"}}


def test_cache_rules() -> None:
    assert yt.cache_satisfies(MANUAL_META, _args())
    assert yt.cache_satisfies(WHISPER_META, _args())
    assert yt.cache_satisfies(WHISPER_META, _args(whisper=True))
    assert not yt.cache_satisfies(MANUAL_META, _args(whisper=True))
    assert not yt.cache_satisfies({"source": "auto-subtitles-fallback", "settings": {}}, _args())
    assert yt.cache_satisfies(WHISPER_META, _args(model="large-v3"))
    assert not yt.cache_satisfies(WHISPER_META, _args(model="large-v2"))
    assert not yt.cache_satisfies(WHISPER_META, _args(no_vad=True))
    assert yt.cache_satisfies(WHISPER_META, _args(lang="tr"))
    assert not yt.cache_satisfies(WHISPER_META, _args(lang="en"))


# --- Purge -------------------------------------------------------------------


def _make_dir(root: Path, name: str, age_s: float, now: float, files: tuple[str, ...] = ()) -> Path:
    _mark_root(root)
    d = root / name
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_text("x")
    os.utime(d, (now - age_s, now - age_s))
    return d


def test_purge_removes_old_dirs_and_leftover_audio(tmp_path: Path) -> None:
    now = time.time()
    old = _make_dir(tmp_path, "old_video_1", 25 * 3600, now, ("transcript.md",))
    fresh = _make_dir(tmp_path, "new_video_1", 3600, now, ("transcript.md", "audio.webm", "audio.webm.part"))
    removed = yt.purge(tmp_path, now=now)
    assert removed == [old]
    assert not old.exists()
    assert fresh.exists()
    assert sorted(p.name for p in fresh.iterdir()) == ["transcript.md"]


def test_purge_skips_dirs_locked_by_live_process(tmp_path: Path) -> None:
    now = time.time()
    busy = _make_dir(tmp_path, "busy_video_", 25 * 3600, now, ("audio.m4a",))
    (busy / ".lock").write_text(str(os.getpid()))
    os.utime(busy, (now - 25 * 3600, now - 25 * 3600))
    assert yt.purge(tmp_path, now=now) == []
    assert (busy / "audio.m4a").exists()


def test_purge_ignores_stale_lock(tmp_path: Path) -> None:
    now = time.time()
    stale = _make_dir(tmp_path, "stale_video", 3600, now, ("audio.m4a",))
    (stale / ".lock").write_text("999999999")
    yt.purge(tmp_path, now=now)
    assert not (stale / "audio.m4a").exists()
    assert not (stale / ".lock").exists()


def test_purge_missing_root(tmp_path: Path) -> None:
    assert yt.purge(tmp_path / "missing") == []


# --- Error classification ----------------------------------------------------


def test_classify_ytdlp_error() -> None:
    bot = yt.classify_ytdlp_error("ERROR: [youtube] x: Sign in to confirm you’re not a bot.", False)
    assert bot.code == yt.EXIT_NEEDS_COOKIES and "--cookies-from-browser" in (bot.hint or "")
    gone = yt.classify_ytdlp_error("\x1b[0;31mERROR:\x1b[0m [youtube] x: This video is unavailable", False)
    assert gone.code == yt.EXIT_NO_TRANSCRIPT
    assert str(gone) == "[youtube] x: This video is unavailable"
    other = yt.classify_ytdlp_error("ERROR: Unable to extract player response", False)
    assert other.code == yt.EXIT_ERROR and "--upgrade-package yt-dlp" in (other.hint or "")


def test_is_gpu_error() -> None:
    assert yt.is_gpu_error(RuntimeError("CUDA failed with error out of memory"))
    assert yt.is_gpu_error(RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))
    assert yt.is_gpu_error(RuntimeError("cuDNN failed with status CUDNN_STATUS_NOT_INITIALIZED"))
    assert not yt.is_gpu_error(RuntimeError("Invalid input features shape"))


# --- Frames: moments ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("83", 83.0),
        ("83.5", 83.5),
        ("1:23", 83.0),
        ("01:23.5", 83.5),
        ("75:00", 4500.0),
        ("1:02:03", 3723.0),
        (" [01:23] ", 83.0),
        ("[1:00:00]", 3600.0),
    ],
)
def test_parse_moment(text: str, expected: float) -> None:
    assert yt.parse_moment(text) == expected


@pytest.mark.parametrize("text", ["", "1:75", "1:60:00", "abc", "-5", "1::2", "1:2:3:4", "1,5"])
def test_parse_moment_rejects_invalid(text: str) -> None:
    with pytest.raises(ValueError):
        yt.parse_moment(text)


def test_parse_moments() -> None:
    assert yt.parse_moments("1:23, 4:56,10") == [83.0, 296.0, 10.0]
    with pytest.raises(ValueError):
        yt.parse_moments(" , ")


def test_normalize_moments_clamps_dedupes_and_sorts() -> None:
    moments, notes = yt.normalize_moments([90, 10, 10.04, 600, 120, 119.5], 120)
    assert moments == [10.0, 90.0, 119.5]
    assert len(notes) == 2 and "past the end (02:00)" in notes[0]
    assert notes[0] == "10:00 is at or past the end (02:00); used 01:59.5"


@pytest.mark.parametrize(
    "t,long_form,expected",
    [(5, False, "00:05"), (5.5, False, "00:05.5"), (5.04, False, "00:05"), (59.96, False, "01:00"),
     (3723.5, True, "1:02:03.5"), (0.3, False, "00:00.3")],
)
def test_format_moment_shows_tenths_only_when_present(t: float, long_form: bool, expected: str) -> None:
    assert yt.format_moment(t, long_form) == expected


def test_normalize_moments_unknown_duration() -> None:
    assert yt.normalize_moments([5000, 3], None) == ([3.0, 5000.0], [])


def test_frame_name_encodes_time() -> None:
    assert yt.frame_name(83.5) == "frame-0h01m23.5s.jpg"
    assert yt.frame_name(3723.04) == "frame-1h02m03.0s.jpg"
    assert yt.frame_name(0) == "frame-0h00m00.0s.jpg"
    assert yt.frame_name(59.96) == "frame-0h01m00.0s.jpg"


# --- Frames: format choice ---------------------------------------------------


def _fmt(fid: str, w: int, h: int, vcodec: str = "avc1.4d401e", protocol: str = "https", **kw: object) -> dict:
    base = {"format_id": fid, "width": w, "height": h, "vcodec": vcodec, "acodec": "none", "protocol": protocol}
    return base | {"url": f"https://example/{fid}"} | kw


def test_choose_video_format_prefers_largest_allowed_then_smallest_file() -> None:
    formats = [
        _fmt("137", 1920, 1080, filesize=17_000_000),
        _fmt("248", 1920, 1080, "vp9", filesize=6_500_000),
        _fmt("401", 3840, 2160, "av01.0.12M.08", filesize=50_000_000),
        _fmt("135", 854, 480, filesize=4_000_000),
        _fmt("hls", 1920, 1080, protocol="m3u8_native"),
        {"format_id": "sb0", "width": 160, "height": 90, "vcodec": "none", "url": "x", "protocol": "mhtml"},
    ]
    assert yt.choose_video_format(formats, 1080)["format_id"] == "248"
    assert yt.choose_video_format(formats, 480)["format_id"] == "135"


def test_choose_video_format_uses_short_side_for_vertical_video() -> None:
    formats = [_fmt("137", 1080, 1920), _fmt("135", 480, 854), _fmt("134", 360, 640)]
    assert yt.choose_video_format(formats, 480)["format_id"] == "135"


def test_choose_video_format_stream_and_fallback_rules() -> None:
    formats = [_fmt("hls", 1280, 720, protocol="m3u8_native"), _fmt("18", 640, 360, acodec="mp4a.40.2")]
    assert yt.choose_video_format(formats, 1080)["format_id"] == "18"  # muxed https beats HLS
    assert yt.choose_video_format(formats, 1080, stream=False)["format_id"] == "hls"
    assert yt.choose_video_format([_fmt("hls", 1280, 720, protocol="m3u8_native")], 1080) is None
    assert yt.choose_video_format([_fmt("x", 1280, 720, has_drm=True)], 1080) is None


def test_choose_video_format_codec_order_when_sizes_unknown() -> None:
    formats = [_fmt("av1", 1280, 720, "av01.0.05M.08"), _fmt("h264", 1280, 720), _fmt("vp9", 1280, 720, "vp9")]
    assert yt.choose_video_format(formats, 1080)["format_id"] == "h264"


# --- Frames: overview picks --------------------------------------------------


def test_default_interval_and_min_gap() -> None:
    assert yt.default_interval(60) == 10.0
    assert yt.default_interval(730) == 24.0
    assert yt.default_interval(7200) == 60.0
    assert yt.default_interval(None) == 10.0
    assert yt.min_gap_for(10) == 2.5
    assert yt.min_gap_for(4) == 2.0
    assert yt.min_gap_for(60) == 15.0


def test_pick_overview_times_keeps_start_and_spaced_scene_changes() -> None:
    scenes = [5.0, 5.2, 5.4, 9.0, 29.6, 30.1]
    times, n_scene, n_fill = yt.pick_overview_times(scenes, 40, interval=100, min_gap=2)
    assert times == [0.0, 5.0, 9.0, 30.0]  # whole seconds; 29.6 rounds up into its scene
    assert (n_scene, n_fill) == (3, 0)


def test_pick_overview_times_fills_long_stretches() -> None:
    times, n_scene, n_fill = yt.pick_overview_times([50.0], 100, interval=20, min_gap=5)
    assert (n_scene, n_fill) == (1, 4)
    assert times == [0.0, 17.0, 33.0, 50.0, 67.0, 83.0]
    assert max(b - a for a, b in zip(times, times[1:] + [100])) <= 20


def test_pick_overview_times_without_scenes_or_duration() -> None:
    times, _, n_fill = yt.pick_overview_times([], 30, interval=10, min_gap=2.5)
    assert times == [0.0, 10.0, 20.0] and n_fill == 2
    assert yt.pick_overview_times([], None, 10, 2.5)[0] == [0.0]


def test_pick_overview_times_ignores_scenes_at_the_end() -> None:
    times, n_scene, _ = yt.pick_overview_times([59.2], 60, interval=100, min_gap=2)
    assert times == [0.0] and n_scene == 0


def test_thin_evenly() -> None:
    times = [float(i) for i in range(10)]
    assert yt.thin_evenly(times, 20) == times
    assert yt.thin_evenly(times, 4) == [0.0, 3.0, 6.0, 9.0]
    assert yt.thin_evenly(times, 1) == [0.0]
    thinned = yt.thin_evenly([float(i) for i in range(1000)], 200)
    assert len(set(thinned)) == 200 and thinned[0] == 0.0 and thinned[-1] == 999.0


def test_pick_overview_times_skips_black_stretches() -> None:
    # Fade-in from black until 1.3 s, a fade to black at 20-22.4 s and a cut into it at 20.2 s.
    black = [(0.0, 1.3), (20.0, 22.4)]
    times, n_scene, _ = yt.pick_overview_times([10.2, 20.2], 40, interval=100, min_gap=2, black=black)
    assert times == [2.0, 11.0, 23.0] and n_scene == 2
    # Interval fills that land in black move past it too.
    times, _, _ = yt.pick_overview_times([], 30, interval=10, min_gap=2.5, black=[(9.5, 12.2)])
    assert times == [0.0, 13.0, 20.0]
    # An all-black video still gets its start.
    assert yt.pick_overview_times([], 10, interval=100, min_gap=2, black=[(0.0, 10.0)])[0] == [0.0]


def test_parse_blackdetect() -> None:
    text = (
        "[blackdetect @ 0000] black_start:0 black_end:1.268 black_duration:1.268\n"
        "[Parsed_showinfo_2 @ 0001] n:   0 pts:  4000 pts_time:4\n"
        "[blackdetect @ 0000] black_start:55.12 black_end:56.04 black_duration:0.92\n"
    )
    assert yt.parse_blackdetect(text) == [(0.0, 1.268), (55.12, 56.04)]
    assert yt.parse_blackdetect("") == []


# --- Frames: sheets and tokens -----------------------------------------------


def test_sheet_groups_and_layout() -> None:
    groups = yt.sheet_groups([float(i) for i in range(20)])
    assert [len(g) for g in groups] == [9, 9, 2]
    assert yt.sheet_layout(9) == (3, 3)
    assert yt.sheet_layout(5) == (3, 2)
    assert yt.sheet_layout(2) == (2, 1)
    assert yt.sheet_layout(1) == (1, 1)


def test_scaled_size_never_upscales() -> None:
    assert yt.scaled_size(1920, 1080, 1280) == (1280, 720)
    assert yt.scaled_size(1080, 1920, 1280) == (720, 1280)
    assert yt.scaled_size(854, 480, 1280) == (854, 480)
    assert yt.scaled_size(480, 854, 640) == (360, 640)
    assert yt.scaled_size(640, 360, 640) == (640, 360)


def test_image_tokens() -> None:
    assert yt.image_tokens(1280, 720) == 1229
    assert yt.image_tokens(1928, 1088) == pytest.approx(1851, abs=3)  # downscaled to 1568 px
    assert yt.image_tokens(854, 480) == 547


def test_jpeg_size() -> None:
    app0 = b"\xff\xe0\x00\x10" + b"JFIF\x00" + b"\x00" * 9
    sof0 = b"\xff\xc0\x00\x11\x08" + (720).to_bytes(2, "big") + (1280).to_bytes(2, "big") + b"\x00" * 10
    assert yt.jpeg_size(b"\xff\xd8" + app0 + sof0 + b"\xff\xd9") == (1280, 720)
    assert yt.jpeg_size(b"not a jpeg") is None
    assert yt.jpeg_size(b"\xff\xd8" + app0) is None


SHOWINFO = """[Parsed_showinfo_1 @ 000001] config in time_base: 1/1000, frame_rate: 30/1
[Parsed_showinfo_1 @ 000001] n:   0 pts:  47567 pts_time:47.567  duration: 33 fmt:yuv420p
[Parsed_showinfo_1 @ 000001] n:   1 pts: 193000 pts_time:193     duration: 33 fmt:yuv420p
[out#0/null @ 000002] video:1kB pts_time:999
"""


def test_parse_showinfo_times() -> None:
    assert yt.parse_showinfo_times(SHOWINFO) == [47.567, 193.0]
    assert yt.parse_showinfo_times("") == []


def test_drawtext_filter_escapes_colons() -> None:
    vf = yt.drawtext_filter("1:02:03", PureWindowsPath(r"C:\Windows\Fonts\arial.ttf"), 28)
    assert "text='1\\:02\\:03'" in vf
    assert "fontfile='C\\:/Windows/Fonts/arial.ttf'" in vf


# --- Frames: CLI and lifecycle -----------------------------------------------


@pytest.mark.parametrize("argv", [["frames", VID], ["frames", VID, "--at", "1:00", "--overview"]])
def test_frames_cli_needs_exactly_one_mode(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        yt.build_parser().parse_args(argv)
    assert exc.value.code == yt.EXIT_ERROR
    assert capsys.readouterr().out.startswith("ERROR:")


def test_frames_cli_defaults() -> None:
    args = yt.build_parser().parse_args(["frames", VID, "--overview"])
    assert args.max_frames == 200 and args.interval is None and args.scene_threshold is None


def _mark_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / yt.WORK_ROOT_MARKER).write_text("", encoding="utf-8")
    return root


def _work_dirs_in(root: Path) -> list[Path]:
    """What a run left in the work root, apart from the root's marker file."""
    return [p for p in root.iterdir() if p.name != yt.WORK_ROOT_MARKER] if root.is_dir() else []


def _run_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]) -> int:
    monkeypatch.setattr(yt, "purge", lambda *a, **k: [])
    monkeypatch.setattr(yt, "WORK_ROOT", _mark_root(tmp_path))
    return yt.main(argv)


def test_frames_without_ffmpeg_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(yt.shutil, "which", lambda name: None)
    code = _run_main(monkeypatch, tmp_path, ["frames", VID, "--at", "1:00"])
    out = capsys.readouterr().out.splitlines()
    assert code == yt.EXIT_ERROR
    assert out[0] == "ERROR: ffmpeg was not found on PATH"
    assert out[1].startswith("HINT:") and "fetch" in out[1]
    assert _work_dirs_in(tmp_path) == []  # stopped before any work dir or network access


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--at", "1:00", "--interval", "5"], "only apply to --overview"),
        (["--at", "1:99"], "invalid timestamp"),
        (["--at", "1,2,3", "--max-frames", "2"], "safety ceiling"),
        (["--overview", "--max-frames", "0"], "--max-frames"),
        (["--overview", "--scene-threshold", "1.5"], "--scene-threshold"),
    ],
)
def test_frames_argument_errors(
    extra: list[str], message: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run_main(monkeypatch, tmp_path, ["frames", VID, *extra])
    out = capsys.readouterr().out
    assert code == yt.EXIT_ERROR
    assert message in out and out.startswith("ERROR: invalid frames request:")
    assert "HINT:" in out and "yt-dlp may be outdated" not in out
    assert _work_dirs_in(tmp_path) == []


def test_frames_rejects_playlist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    url = "https://www.youtube.com/playlist?list=PL1234567890"
    assert _run_main(monkeypatch, tmp_path, ["frames", url, "--overview"]) == yt.EXIT_NO_TRANSCRIPT


def test_remove_downloads_covers_video_and_frame_intermediates(tmp_path: Path) -> None:
    (tmp_path / "frames").mkdir()
    for name in ["video.webm", "video.webm.part", "audio.m4a", "transcript.md"]:
        (tmp_path / name).write_text("x")
    for name in ["tile-0001.jpg", "overview-01.tmp.jpg", "overview-01.jpg", "frame-0h01m00.0s.jpg"]:
        (tmp_path / "frames" / name).write_text("x")
    yt.remove_downloads(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["frames", "transcript.md"]
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == ["frame-0h01m00.0s.jpg", "overview-01.jpg"]


def test_has_results(tmp_path: Path) -> None:
    assert not yt.has_results(tmp_path)
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "info.json").write_text("{}")
    assert not yt.has_results(tmp_path)
    (tmp_path / "frames" / "frame-0h00m01.0s.jpg").write_text("x")
    assert yt.has_results(tmp_path)


# --- Frames: format choice for overview and fallback -------------------------


def test_choose_overview_format_prefers_360_then_smallest_up_to_480() -> None:
    base = [_fmt("160", 256, 144, filesize=1), _fmt("133", 426, 240, filesize=2)]
    with_360 = base + [_fmt("134", 640, 360, filesize=9), _fmt("243", 640, 360, "vp9", filesize=7), _fmt("135", 854, 480)]
    assert yt.choose_overview_format(with_360)["format_id"] == "243"
    no_360 = base + [_fmt("135", 854, 480), _fmt("136", 1280, 720)]
    assert yt.choose_overview_format(no_360)["format_id"] == "135"  # not the 240p one
    vertical = [_fmt("133", 240, 426), _fmt("135", 480, 854), _fmt("137", 1080, 1920)]
    assert yt.choose_overview_format(vertical)["format_id"] == "135"
    assert yt.choose_overview_format([_fmt("133", 426, 240), _fmt("137", 1920, 1080)])["format_id"] == "133"  # closest
    assert yt.choose_overview_format([_fmt("hls", 640, 360, protocol="m3u8_native")])["format_id"] == "hls"
    assert yt.choose_overview_format([]) is None


def test_seek_formats_order_stream_then_hls_then_other() -> None:
    formats = [
        _fmt("248", 1920, 1080, "vp9", filesize=6),
        _fmt("137", 1920, 1080, filesize=17),
        _fmt("614", 1920, 1080, "vp09.00.40.08", protocol="m3u8_native"),
        _fmt("401", 3840, 2160, "av01.0.12M.08", filesize=50),
    ]
    assert [f["format_id"] for f in yt.seek_formats(formats)] == ["248", "614", "137"]
    other_codec_hls = formats[:2] + [_fmt("270", 1920, 1080, protocol="m3u8_native"), formats[2]]
    assert [f["format_id"] for f in yt.seek_formats(other_codec_hls)][:2] == ["248", "614"]  # same codec first
    no_same = [formats[0], _fmt("270", 1920, 1080, protocol="m3u8_native")]
    assert [f["format_id"] for f in yt.seek_formats(no_same)] == ["248", "270"]
    hls_only = [_fmt("614", 1920, 1080, protocol="m3u8_native"), _fmt("231", 854, 480, protocol="m3u8_native")]
    assert [f["format_id"] for f in yt.seek_formats(hls_only)] == ["614", "231"]
    assert yt.seek_formats([]) == []


def test_is_stream_error() -> None:
    assert yt.is_stream_error("https://x: Server returned 403 Forbidden (access denied)")
    assert yt.is_stream_error("ffmpeg timed out")
    assert not yt.is_stream_error("Decoder (codec av1) not found for input stream #0:0")


# --- Work-dir lock -----------------------------------------------------------


def test_acquire_lock_is_exclusive(tmp_path: Path) -> None:
    (tmp_path / ".lock").write_text(str(os.getpid()))  # held by a live process
    with pytest.raises(yt.FetchError, match="another run is using this video") as exc:
        yt.acquire_lock(tmp_path)
    assert exc.value.code == yt.EXIT_ERROR and "parallel" in (exc.value.hint or "")


def test_acquire_lock_takes_over_stale_lock_and_release_only_own(tmp_path: Path) -> None:
    (tmp_path / ".lock").write_text("999999999")
    yt.acquire_lock(tmp_path)
    assert (tmp_path / ".lock").read_text() == str(os.getpid())
    yt.release_lock(tmp_path)
    assert not (tmp_path / ".lock").exists()
    (tmp_path / ".lock").write_text("999999999")
    yt.release_lock(tmp_path)  # someone else's lock stays
    assert (tmp_path / ".lock").exists()


@pytest.mark.parametrize("argv", [["frames", VID, "--overview"], ["fetch", VID]])
def test_busy_video_is_left_untouched(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / VID
    (work / "frames").mkdir(parents=True)
    for name in ("video.webm", "audio.webm", ".lock"):
        (work / name).write_text(str(os.getpid()) if name == ".lock" else "x")
    (work / "frames" / "tile-0000.jpg").write_text("x")
    monkeypatch.setattr(yt.shutil, "which", lambda name: "ffmpeg")
    code = _run_main(monkeypatch, tmp_path, argv)
    out = capsys.readouterr().out
    assert code == yt.EXIT_ERROR
    assert out.startswith("ERROR: another run is using this video") and "HINT:" in out
    assert sorted(p.name for p in work.iterdir()) == [".lock", "audio.webm", "frames", "video.webm"]
    assert (work / "frames" / "tile-0000.jpg").exists()


def test_ffmpeg_error_has_its_own_hint() -> None:
    exc = yt.FfmpegError("ffmpeg could not build a contact sheet: boom")
    assert exc.code == yt.EXIT_ERROR and str(exc).startswith("ffmpeg")
    assert "not a yt-dlp problem" in (exc.hint or "") and "upgrade" not in (exc.hint or "")


# --- Description and chapters ------------------------------------------------


def test_video_context_normalizes_description_and_chapters() -> None:
    info = {
        "description": "line one\r\nline two\r\n\r\n#tag  \n",
        "chapters": [
            {"start_time": 65.0, "end_time": 120.0, "title": "  Second   part "},
            {"start_time": 0.0, "end_time": 65.0, "title": "Intro"},
            {"start_time": 90.0, "title": ""},
        ],
    }
    ctx = yt.video_context(info)
    assert ctx["description"] == "line one\nline two\n\n#tag"
    assert ctx["chapters"] == [{"start": 0.0, "title": "Intro"}, {"start": 65.0, "title": "Second part"}]
    assert yt.video_context({"description": None, "chapters": None}) == {"description": "", "chapters": []}


def test_chapter_lines_use_transcript_stamps() -> None:
    chapters = [{"start": 0.0, "title": "Intro"}, {"start": 3725.4, "title": "Late"}]
    assert yt.chapter_lines(chapters, False) == ["- [00:00] Intro", "- [62:05] Late"]
    assert yt.chapter_lines(chapters, True) == ["- [0:00:00] Intro", "- [1:02:05] Late"]


def _meta(**kw: object) -> dict:
    base = {
        "video_id": VID,
        "url": f"https://www.youtube.com/watch?v={VID}",
        "title": "Title",
        "channel": "Chan",
        "upload_date": "20261020",
        "duration": 120,
        "language": "tr",
        "source": "manual-subtitles",
        "settings": {"track": "tr"},
        "notes": [],
        "generated_at": "2026-10-20T12:00:00+03:00",
        "description": "A short placeholder description\nhttps://example.com",
        "chapters": [{"start": 0.0, "title": "Intro"}, {"start": 65.0, "title": "Main"}],
    }
    return base | kw


def test_header_lines_with_description_and_chapters() -> None:
    head = "\n".join(yt.header_lines(_meta()))
    assert "## Description\n\nA short placeholder description\nhttps://example.com\n\n" in head
    assert "## Chapters\n\n- [00:00] Intro\n- [01:05] Main\n\n---\n" in head
    assert head.index("- Generated:") < head.index("## Description") < head.index("## Chapters")


def test_header_lines_paragraphs_variant_keeps_chapters_only() -> None:
    head = "\n".join(yt.header_lines(_meta(), description=False))
    assert "## Description" not in head and "placeholder" not in head
    assert "- [01:05] Main" in head


def test_header_lines_omit_empty_sections() -> None:
    old_style = {k: v for k, v in _meta().items() if k not in ("description", "chapters")}
    for meta in (_meta(description="", chapters=[]), old_style):
        head = yt.header_lines(meta)
        assert "## Description" not in head and "## Chapters" not in head
        assert head[-2:] == ["---", ""] and head[-3] == ""


def test_header_lines_vad_source_and_note() -> None:
    whisper = {"model": "large-v3", "compute_type": "float16", "beam_size": 5}
    on = _meta(source="whisper", settings=whisper | {"vad_filter": True, "vad_min_silence_ms": 500, "batched": True, "batch_size": 8})
    retry = _meta(source="whisper", settings=whisper | {"vad_filter": False, "vad_auto_retry": True, "batched": False})
    manual = _meta(source="whisper", settings=whisper | {"vad_filter": False, "batched": False})
    on_head, retry_head, manual_head = ("\n".join(yt.header_lines(m)) for m in (on, retry, manual))
    assert "VAD on (min silence 500 ms)" in on_head and "- Note:" not in on_head
    assert "VAD off (auto-retry), beam 5, sequential)" in retry_head and "Amara.org" in retry_head
    assert "VAD off, beam 5" in manual_head and "- Note:" in manual_head
    assert "- Note:" not in "\n".join(yt.header_lines(_meta()))  # subtitles never get the note


def test_write_outputs_description_only_in_transcript(tmp_path: Path) -> None:
    segments = _segs([(0, 4, "Merhaba."), (3700, 3704, "Son.")])
    yt.write_outputs(tmp_path, _meta(duration=3710, chapters=[{"start": 3650.0, "title": "End"}]), segments)
    transcript = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    paragraphs = (tmp_path / "transcript-paragraphs.md").read_text(encoding="utf-8")
    assert "## Description" in transcript and "## Description" not in paragraphs
    assert "- [1:00:50] End" in transcript and "- [1:00:50] End" in paragraphs
    assert transcript.endswith("---\n\n[0:00:00] Merhaba.\n[1:01:40] Son.\n")
    meta = yt.json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["chapters"] == [{"start": 3650.0, "title": "End"}]
    assert [s.text for s in yt.load_segments(tmp_path)] == ["Merhaba.", "Son."]


def test_print_summary_reports_chapters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yt.write_outputs(tmp_path, _meta(), _segs([(0, 4, "Hi.")]))
    yt.print_summary(tmp_path, _meta(), cached=False, extra_notes=["extra"])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "SOURCE: manual-subtitles"
    assert "CHAPTERS: 2" in out and out[-1] == "NOTE: extra"
    yt.print_summary(tmp_path, _meta(chapters=[]), cached=True)
    assert "CHAPTERS: 0" in capsys.readouterr().out


# --- SPEECH hint: words per minute and the longest gap without speech ---


@pytest.mark.parametrize(
    ("spec", "audio_s", "expected"),
    [
        ([(10, 20, "a"), (30, 40, "b")], 60, (40.0, 60.0)),  # trailing gap is the longest
        ([(30, 40, "a"), (45, 60, "b")], 60, (0.0, 30.0)),  # leading gap (e.g. a music intro)
        ([(0, 10, "a"), (100, 110, "b"), (115, 120, "c")], 120, (10.0, 100.0)),  # a gap mid-video
        ([(0, 50, "a"), (10, 20, "b"), (55, 60, "c")], 60, (50.0, 55.0)),  # overlap keeps the outer end
        ([(5, 10, "a")], None, (0.0, 5.0)),  # unknown length: no trailing gap
        ([], 90, (0.0, 90.0)),
    ],
)
def test_longest_gap(spec: list[tuple[float, float, str]], audio_s: float | None, expected: tuple[float, float]) -> None:
    assert yt.longest_gap(_segs(spec), audio_s) == expected


def test_speech_line() -> None:
    segs = _segs([(0, 30, "bir iki üç dört beş altı yedi sekiz dokuz on"), (220, 240, "on bir on iki")])
    assert yt.speech_line(segs, 240, False) == "SPEECH: 4 words/min; longest gap 03:10 (00:30-03:40)"
    assert yt.speech_line(segs, 240, True).endswith("(0:00:30-0:03:40)")
    assert yt.speech_line([], None, False) == "SPEECH: 0 words/min; longest gap 00:00 (00:00-00:00)"


def test_print_summary_speech_only_for_whisper(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    segs = _segs([(0, 30, "Merhaba dünya."), (100, 120, "Son.")])
    yt.write_outputs(tmp_path, _meta(), segs)
    yt.print_summary(tmp_path, _meta(), cached=False)
    assert not any(line.startswith("SPEECH:") for line in capsys.readouterr().out.splitlines())
    yt.print_summary(tmp_path, _meta(source="whisper", settings={"vad_filter": True}), cached=True)
    out = capsys.readouterr().out.splitlines()
    i = out.index("SPEECH: 2 words/min; longest gap 01:10 (00:30-01:40)")
    assert out[i - 1].startswith("CHAPTERS:") and out[i + 1] == "CACHED: yes"


# --- Cache compatibility: transcripts cached before description and chapters ---


class _FakeYDL:
    def __init__(self, *a: object, **k: object) -> None:
        pass

    def __enter__(self) -> _FakeYDL:
        return self

    def __exit__(self, *a: object) -> None:
        pass


def _old_cache(root: Path) -> Path:
    work = root / VID
    work.mkdir()
    old = {k: v for k, v in _meta().items() if k not in ("description", "chapters")}
    yt.write_outputs(work, old, _segs([(0, 4, "Merhaba."), (70, 74, "Devam.")]))
    return work


def test_old_cache_gets_description_and_chapters_without_transcribing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = _old_cache(tmp_path)
    assert not yt.has_context(yt.json.loads((work / "meta.json").read_text(encoding="utf-8")))
    info = {"description": "A short placeholder description", "chapters": [{"start_time": 60, "title": "Part 2"}]}
    monkeypatch.setitem(sys.modules, "yt_dlp", argparse.Namespace(YoutubeDL=_FakeYDL))
    monkeypatch.setattr(yt, "extract_info", lambda *a: info)
    monkeypatch.setattr(yt, "run_whisper", lambda *a: pytest.fail("must not transcribe"))
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert "CACHED: yes" in out and "CHAPTERS: 1" in out and not any(o.startswith("NOTE:") for o in out)
    transcript = (work / "transcript.md").read_text(encoding="utf-8")
    assert "A short placeholder description" in transcript and "- [01:00] Part 2" in transcript
    assert "[01:10] Devam." in transcript
    meta = yt.json.loads((work / "meta.json").read_text(encoding="utf-8"))
    assert yt.has_context(meta) and meta["generated_at"] == _meta()["generated_at"]
    assert not (work / ".lock").exists()
    # The next fetch is a pure cache hit: no metadata request.
    monkeypatch.setattr(yt, "extract_info", lambda *a: pytest.fail("no refresh needed"))
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    assert "CHAPTERS: 1" in capsys.readouterr().out


def test_old_cache_refresh_failure_still_serves_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = _old_cache(tmp_path)

    def fail(*a: object) -> None:
        raise yt.FetchError("network down")

    monkeypatch.setitem(sys.modules, "yt_dlp", argparse.Namespace(YoutubeDL=_FakeYDL))
    monkeypatch.setattr(yt, "extract_info", fail)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "CACHED: yes" in out and "CHAPTERS: 0" in out
    assert "NOTE: description and chapters unavailable: network down" in out
    assert not yt.has_context(yt.json.loads((work / "meta.json").read_text(encoding="utf-8")))
    assert not (work / ".lock").exists()


def test_cache_accepts_auto_retry_result() -> None:
    settings = {"model": "large-v3", "vad_filter": False, "vad_auto_retry": True}
    retry_meta = {"source": "whisper", "language": "tr", "settings": settings}
    assert yt.cache_satisfies(retry_meta, _args())
    assert yt.cache_satisfies(retry_meta, _args(no_vad=True))
    assert yt.cache_satisfies(retry_meta, _args(whisper=True))


# --- No-VAD auto-retry -------------------------------------------------------


def test_count_words_prefers_word_timestamps() -> None:
    words = [{"start": 0.0, "end": 0.1, "word": "x", "p": 1.0}] * 7
    assert yt.count_words([Segment(0, 1, "あいうえおかき", words)]) == 7  # no spaces, 7 timed words
    assert yt.count_words(_segs([(0, 1, "three plain  words"), (1, 2, "two more")])) == 5
    assert yt.count_words([]) == 0


@pytest.mark.parametrize(
    "words,audio_s,expected",
    [
        (0, 72, True),  # empty: always retried
        (5, 72, True),  # 4.2 wpm
        (6, 72, False),  # exactly 5 wpm
        (20, 600, True),  # 2 wpm over 10 min: VAD missed nearly everything
        (60, 600, False),  # sparse commentary, a few sentences in 10 min
        (3, None, False),  # unknown length: only an empty run counts
        (3, 0, False),
    ],
)
def test_nearly_empty_rule(words: int, audio_s: float | None, expected: bool) -> None:
    segments = [Segment(0, 1, " ".join(["w"] * words))] if words else []
    assert yt.nearly_empty(segments, audio_s) is expected


def test_prefer_retry() -> None:
    few = _segs([(0, 1, "la la")])
    more = _segs([(0, 1, "lyrics here and more")])
    assert yt.prefer_retry([], more) and yt.prefer_retry(few, more)
    assert not yt.prefer_retry(few, [])  # an empty retry never replaces a non-empty VAD result
    assert yt.prefer_retry([], [])  # both empty: the caller keeps the original error


class _FakeWhisperYDL:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def process_ie_result(self, info: dict, download: bool) -> dict:
        audio = self.directory / "audio.webm"
        audio.write_text("x")
        return {"requested_downloads": [{"filepath": str(audio)}]}


def _whisper_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, by_vad: dict, no_vad: bool = False, duration: float = 72
) -> tuple[yt.Result, list[bool]]:
    calls: list[bool] = []

    def fake_transcribe(model: object, audio: Path, language: object, vad: bool, dur: object, batch: int = 8) -> tuple:
        calls.append(vad)
        outcome = by_vad[vad]
        if isinstance(outcome, Exception):
            raise outcome
        info = argparse.Namespace(language="tr", language_probability=1.0, duration=duration)
        return outcome, info, ({"batched": True, "batch_size": 8} if vad else {"batched": False})

    monkeypatch.setattr(yt, "load_whisper", lambda name, device="cuda", root=None: object())
    monkeypatch.setattr(yt, "whisper_language", lambda code: code)
    monkeypatch.setattr(yt, "transcribe", fake_transcribe)
    args = argparse.Namespace(model=None, no_vad=no_vad, lang=None, cookies_from_browser=None)
    result = yt.run_whisper(_FakeWhisperYDL(tmp_path), {"duration": duration}, tmp_path, args, "tr")
    assert not (tmp_path / "audio.webm").exists()  # audio deleted after both runs
    return result, calls


LYRICS = _segs([(0, 5, "the waves roll in and the night is long"), (5, 10, "bir iki üç dört beş altı")])


def test_auto_retry_replaces_empty_vad_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result, calls = _whisper_run(monkeypatch, tmp_path, {True: [], False: LYRICS})
    assert calls == [True, False]
    assert result.segments == LYRICS
    assert result.settings["vad_filter"] is False and result.settings["vad_auto_retry"] is True
    assert result.settings["batched"] is False and result.settings["vad_min_silence_ms"] is None
    assert result.notes == ["VAD found little speech (0 words in 01:12); retried with VAD off"]
    assert "retry_transcribe_s" in result.timings
    assert "ESTIMATE: ~0.1 min for the no-VAD retry" in capsys.readouterr().err


def test_no_auto_retry_for_normal_speech_or_explicit_no_vad(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, calls = _whisper_run(monkeypatch, tmp_path, {True: LYRICS}, duration=10)
    assert calls == [True] and result.settings["vad_auto_retry"] is False and result.notes == []
    result, calls = _whisper_run(monkeypatch, tmp_path, {False: []}, no_vad=True)
    assert calls == [False] and result.segments == [] and result.settings["vad_auto_retry"] is False


def test_auto_retry_keeps_vad_result_when_retry_finds_less(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    few = _segs([(30, 31, "hey")])
    result, calls = _whisper_run(monkeypatch, tmp_path, {True: few, False: []})
    assert calls == [True, False] and result.segments == few
    assert result.settings["vad_filter"] is True and result.settings["vad_auto_retry"] is True
    assert "kept the VAD result" in result.notes[0]


def test_auto_retry_gpu_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    oom = RuntimeError("CUDA failed with error out of memory")
    few = _segs([(30, 31, "hey")])
    result, _ = _whisper_run(monkeypatch, tmp_path, {True: few, False: oom})
    assert result.segments == few and "failed on the GPU" in result.notes[0]
    with pytest.raises(yt.GpuUnavailable):  # nothing to keep: the auto-captions fallback applies
        _whisper_run(monkeypatch, tmp_path, {True: [], False: oom})


# --- Frames paired with speech (SAID) ----------------------------------------

SPEECH = _segs([(0.0, 4.0, "first line"), (3.5, 6.0, "overlapping line"), (10.0, 12.0, "after a gap")])


@pytest.mark.parametrize(
    "t,expected",
    [
        (1.0, "first line"),  # containing segment
        (3.8, "overlapping line"),  # overlap: the latest-starting one
        (7.5, "overlapping line"),  # nearest within the window, 1.5 s after its end
        (8.5, "after a gap"),  # nearest within the window, 1.5 s before its start
        (8.0, "overlapping line"),  # equidistant: the earlier one
        (14.5, None),  # 2.5 s after the last segment
    ],
)
def test_said_segment(t: float, expected: str | None) -> None:
    seg = yt.said_segment(SPEECH, t)
    assert (seg.text if seg else None) == expected


def test_said_segment_empty() -> None:
    assert yt.said_segment([], 3.0) is None


def test_said_text_one_line_and_cap() -> None:
    assert yt.said_text("  two\nlines \t here ") == "two lines here"
    long = " ".join(f"word{i}" for i in range(200))
    capped = yt.said_text(long)
    assert len(capped) <= yt.SAID_MAX_CHARS and capped.endswith("…")
    assert long.startswith(capped[:-1]) and capped[:-1].split()[-1] in long.split()  # word boundary
    assert yt.said_text("x" * 400, 10) == "x" * 9 + "…"  # no space to cut at


def test_said_lines_use_frame_stamps() -> None:
    assert yt.said_lines(SPEECH, [1.0, 5.5, 20.0], False) == [
        "SAID: [00:01] first line",
        "SAID: [00:05.5] overlapping line",
    ]
    assert yt.said_lines(SPEECH, [3601.0], True) == []
    assert yt.said_lines(None, [1.0], False) == []


def _cached_frames(root: Path, with_transcript: bool) -> Path:
    work = root / VID
    (work / "frames").mkdir(parents=True)
    (work / "frames" / "info.json").write_text('{"title": "T", "duration": 60}')
    for t in (1.0, 9.0):
        (work / "frames" / yt.frame_name(t)).write_text("x")
    if with_transcript:
        yt.write_outputs(work, _meta(duration=60), SPEECH)
    return work


def test_frames_at_prints_said_for_cached_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _cached_frames(tmp_path, with_transcript=True)
    monkeypatch.setattr(yt.shutil, "which", lambda name: "ffmpeg")
    assert _run_main(monkeypatch, tmp_path, ["frames", VID, "--at", "0:09,1"]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    frame_lines = [o for o in out if o.startswith(("FRAME:", "SAID:"))]
    assert frame_lines[0].startswith("FRAME: [00:01] ") and frame_lines[0].endswith("(cached)")
    assert frame_lines[1] == "SAID: [00:01] first line"
    assert frame_lines[2].startswith("FRAME: [00:09] ")
    assert frame_lines[3] == "SAID: [00:09] after a gap"
    assert "COUNT: 2" in out  # frames only, not SAID lines
    assert not any(o.startswith("NOTE:") for o in out)


def test_frames_at_without_transcript_notes_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _cached_frames(tmp_path, with_transcript=False)
    monkeypatch.setattr(yt.shutil, "which", lambda name: "ffmpeg")
    assert _run_main(monkeypatch, tmp_path, ["frames", VID, "--at", "1,9"]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert not any(o.startswith("SAID:") for o in out)
    assert out.count("NOTE: " + yt.NO_TRANSCRIPT_NOTE) == 1


# --- Phase 4: source classification -----------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://www.youtube.com/watch?v={VID}&list=PL1234567890&index=3",
        f"https://youtu.be/{VID}?si=abcDEF&t=10",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/live/{VID}",
        f"https://www.youtube-nocookie.com/embed/{VID}",
        f"https://music.youtube.com/watch?v={VID}",
        f"www.youtube.com/watch?v={VID}",
        f"youtu.be/{VID}",
        f'"https://youtu.be/{VID}"',
        VID,
    ],
)
def test_resolve_source_youtube_unchanged(url: str) -> None:
    source = yt.resolve_source(url)
    assert (source.kind, source.key, source.url) == ("youtube", VID, f"https://www.youtube.com/watch?v={VID}")


def test_resolve_source_other_sites_use_offline_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(yt, "site_key", lambda url: seen.append(url) or "vimeo-76979871")
    source = yt.resolve_source("https://vimeo.com/76979871")
    assert (source.kind, source.key, source.url) == ("site", "vimeo-76979871", "https://vimeo.com/76979871")
    assert yt.resolve_source("vimeo.com/76979871").url == "https://vimeo.com/76979871"  # schemeless domain
    assert seen == ["https://vimeo.com/76979871"] * 2


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/playlist?list=PL1234567890",
        "https://www.youtube.com/watch?list=PL1234567890",
        "https://www.youtube.com/@somechannel",
    ],
)
def test_resolve_source_youtube_lists_point_to_playlist(url: str) -> None:
    with pytest.raises(yt.FetchError) as exc:
        yt.resolve_source(url)
    assert exc.value.code == yt.EXIT_NO_TRANSCRIPT and "playlist <url>" in (exc.value.hint or "")


@pytest.mark.parametrize("text", ["ftp://example.com/a.mp4", "not a url", "", "   "])
def test_resolve_source_rejects_unknown_inputs(text: str) -> None:
    with pytest.raises(yt.FetchError) as exc:
        yt.resolve_source(text)
    assert exc.value.code == yt.EXIT_NO_TRANSCRIPT


def test_resolve_source_local_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yt, "WORK_ROOT", tmp_path / "work")
    folder = tmp_path / "my videos"
    folder.mkdir()
    clip = folder / "summer trip.mp4"
    clip.write_bytes(b"data")
    absolute = yt.resolve_source(str(clip))
    assert absolute.kind == "file" and absolute.path == clip.resolve() and absolute.url is None
    assert absolute.key.startswith("file-") and len(absolute.key) == 5 + yt.SHORT_HASH_LEN
    assert yt.resolve_source(f'"{clip}"').key == absolute.key  # quoted, with spaces
    assert yt.resolve_source(clip.as_uri()).key == absolute.key  # file: URI
    monkeypatch.chdir(folder)
    assert yt.resolve_source("summer trip.mp4").key == absolute.key  # relative
    if os.name == "nt":  # a backslash separates folders only on Windows
        assert yt.resolve_source(".\\summer trip.mp4").key == absolute.key


@pytest.mark.parametrize("name", ["missing.mp4", "sub\\missing", "C:\\nowhere\\missing.mkv", "./missing"])
def test_resolve_source_missing_local_file(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(yt.FetchError, match="local file not found") as exc:
        yt.resolve_source(name)
    assert exc.value.code == yt.EXIT_NO_TRANSCRIPT


def test_resolve_source_rejects_dirs_and_work_root_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yt, "WORK_ROOT", tmp_path)
    with pytest.raises(yt.FetchError, match="not a file"):
        yt.local_source(tmp_path.parent)
    inside = tmp_path / "file-abc" / "audio.wav"
    inside.parent.mkdir()
    inside.write_bytes(b"x")
    with pytest.raises(yt.FetchError, match="work root"):
        yt.resolve_source(str(inside))


# --- Phase 4: work keys ------------------------------------------------------


def test_work_key_sanitizes_and_caps() -> None:
    assert yt.work_key("TwitchVod", "v123456", "u") == "twitchvod-v123456"
    assert yt.work_key("TikTok", "6718335390845095173", "u") == "tiktok-6718335390845095173"
    assert yt.work_key("Some:IE", 'a/b\\c:d*e?"<>|', "u") == "someie-a_b_c_d_e"
    assert yt.work_key("Generic", None, "https://example.com/x") == f"generic-{yt.short_hash('https://example.com/x')}"
    assert yt.work_key("Reddit", "...", "https://r/x") == f"reddit-{yt.short_hash('https://r/x')}"  # nothing usable
    long_a, long_b = yt.work_key("Site", "a" * 200, "u"), yt.work_key("Site", "a" * 199 + "b", "u")
    assert len(long_a) <= yt.WORK_KEY_MAX_LEN and long_a != long_b  # capped, still unique
    assert all(yt.WORK_KEY_RE.match(k) for k in (long_a, long_b))


def test_site_key_matches_like_ytdlp_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    class IE:
        def __init__(self, name: str, prefix: str | None, vid: str | None) -> None:
            self.name, self.prefix, self.vid = name, prefix, vid

        def suitable(self, url: str) -> bool:
            return self.prefix is None or url.startswith(self.prefix)

        def ie_key(self) -> str:
            return self.name

        def get_temp_id(self, url: str) -> str | None:
            return self.vid

    classes = [IE("Vimeo", "https://vimeo.com/", "76979871"), IE("Dailymotion", "https://dai", None), IE("Generic", None, "x")]
    monkeypatch.setitem(sys.modules, "yt_dlp.extractor", argparse.Namespace(gen_extractor_classes=lambda: classes))
    assert yt.site_key("https://vimeo.com/76979871") == "vimeo-76979871"
    assert yt.site_key("https://dai.ly/x") == f"dailymotion-{yt.short_hash('https://dai.ly/x')}"
    assert yt.site_key("https://example.com/v") == f"generic-{yt.short_hash('https://example.com/v')}"


def test_file_key_changes_with_size_and_mtime(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"one")
    first = yt.file_key(clip)
    assert first == yt.file_key(clip)
    os.utime(clip, (1_000_000, 1_000_000))
    touched = yt.file_key(clip)
    clip.write_bytes(b"three")
    os.utime(clip, (1_000_000, 1_000_000))
    resized = yt.file_key(clip)
    assert len({first, touched, resized}) == 3


# --- Phase 4: subtitle choice for sites and local files ------------------------


def test_site_tracks_need_a_stated_language() -> None:
    assert yt.choose_manual_track({"en": FMT, "de": FMT}, "en", strict=True)[0] == "en"
    assert yt.choose_manual_track({"en-US": FMT}, "eng", strict=True)[0] == "en-US"  # 639-2 metadata
    track, note = yt.choose_manual_track({"de": FMT}, None, strict=True)
    assert track is None and "could be a translation" in note
    assert yt.choose_manual_track({"fr": FMT, "es": FMT}, "en", strict=True)[0] is None  # translations only
    assert yt.choose_manual_track({"en-x-autogen": FMT}, "en")[0] == "en-x-autogen"  # name filter is site-only


def test_site_tracks_ignore_chat_and_machine_tracks() -> None:
    for kind in ("live_chat", "danmaku", "rechat", "comments"):
        assert yt.choose_manual_track({kind: FMT}, "en", strict=True) == (None, "no manual subtitles")
    assert yt.choose_manual_track({"en-x-autogen": FMT, "en-orig": FMT}, "en", strict=True)[0] is None
    assert yt.choose_manual_track({"danmaku": FMT, "en": FMT}, "en")[0] == "en"  # chat ignored


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("eng", "en", True),
        ("en-GB", "en-US", True),
        ("ger", "deu", True),
        ("tur", "tr", True),
        ("und", "und", False),
        (None, "en", False),
        ("fre", "en", False),
    ],
)
def test_lang_matches(a: str | None, b: str | None, expected: bool) -> None:
    assert yt.lang_matches(a, b) is expected


def _stream(index: int, kind: str, codec: str, lang: str | None = None, **disposition: int) -> dict:
    tags = {"language": lang} if lang else {}
    return {"index": index, "codec_type": kind, "codec_name": codec, "tags": tags, "disposition": disposition}


def test_embedded_subtitle_language_match() -> None:
    streams = [
        _stream(0, "video", "h264"),
        _stream(1, "audio", "aac", "eng"),
        _stream(2, "subtitle", "subrip", "tur"),
        _stream(3, "subtitle", "ass", "eng"),
    ]
    pick, note = yt.choose_embedded_subtitle(streams, "eng")
    assert pick["index"] == 3 and "matches the audio language" in note
    assert yt.choose_embedded_subtitle(streams, "en")[0]["index"] == 3  # --lang en
    pick, note = yt.choose_embedded_subtitle(streams, "jpn")
    assert pick is None and "do not match" in note
    pick, note = yt.choose_embedded_subtitle(streams, None)
    assert pick is None and "could be a translation" in note


def test_embedded_subtitle_ignores_bitmap_and_forced() -> None:
    bitmap = [
        _stream(1, "audio", "aac", "eng"),
        _stream(2, "subtitle", "hdmv_pgs_subtitle", "eng"),
        _stream(3, "subtitle", "dvd_subtitle", "eng"),
    ]
    pick, note = yt.choose_embedded_subtitle(bitmap, "eng")
    assert pick is None and "dvd_subtitle, hdmv_pgs_subtitle" in note
    forced = [
        _stream(2, "subtitle", "subrip", "eng", forced=1),
        _stream(4, "subtitle", "subrip", "eng"),
        _stream(3, "subtitle", "mov_text", "eng", default=1),
    ]
    assert yt.choose_embedded_subtitle(forced, "eng")[0]["index"] == 3  # default wins; forced skipped
    assert yt.choose_embedded_subtitle([_stream(1, "audio", "aac", "eng")], "eng") == (None, "no embedded subtitles")


def test_parse_subtitles_strips_ass_overrides() -> None:
    srt = "1\n00:00:01,000 --> 00:00:02,000\n{\\an8}<i>Top</i> line\n"
    assert yt.parse_subtitles(srt)[0].text == "Top line"


def test_file_info_from_ffprobe(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mkv"
    probe = {
        "format": {"duration": "65.5", "tags": {"TITLE": "Trip", "COMMENT": "a description"}},
        "streams": [
            _stream(0, "video", "mjpeg", attached_pic=1),
            {**_stream(1, "video", "h264"), "width": 1920, "height": 1080, "side_data_list": [{"rotation": -90}]},
            _stream(2, "audio", "aac", "tur"),
        ],
        "chapters": [{"start_time": "0.000000", "tags": {"title": "Intro"}}, {"start_time": "30.5", "tags": {}}],
    }
    info = yt.file_info(probe, clip)
    assert (info["title"], info["duration"], info["description"]) == ("Trip", 65.5, "a description")
    assert (info["width"], info["height"], info["vcodec"]) == (1080, 1920, "h264")  # rotated phone video
    assert (info["language"], info["audio_index"]) == ("tur", 2)
    assert yt.video_context(info)["chapters"] == [{"start": 0.0, "title": "Intro"}]
    bare = yt.file_info({"format": {}, "streams": [_stream(0, "audio", "mp3", "und")]}, clip)
    assert bare["title"] == "clip.mkv" and bare["language"] is None and bare["width"] is None
    with pytest.raises(yt.FetchError) as exc:
        yt.file_info({"streams": [_stream(0, "subtitle", "subrip")]}, clip)
    assert exc.value.code == yt.EXIT_NO_TRANSCRIPT


def test_header_shows_file_for_local_sources() -> None:
    head = "\n".join(yt.header_lines(_meta(source_path=r"C:\v\my clip.mp4", url="file:///C:/v/my%20clip.mp4")))
    assert "- File: C:\\v\\my clip.mp4" in head and "- URL:" not in head
    assert "- URL: https://www.youtube.com/watch?v=" in "\n".join(yt.header_lines(_meta()))


# --- Phase 4: errors, live and DRM ---------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "ERROR: [Instagram] C1a2b3: Requested content is not available, rate-limit reached or login required. "
        "Use --cookies, --cookies-from-browser, --username and --password",
        "ERROR: [twitter] 123: NSFW tweet requires authentication. Use --cookies-from-browser or --cookies for the authentication.",
        "ERROR: [facebook] 456: This video is only available for registered users. Use --cookies-from-browser or --cookies",
        "ERROR: [vimeo] 1: You need to log in to access this video",
    ],
)
def test_login_walls_map_to_needs_cookies(message: str) -> None:
    exc = yt.classify_ytdlp_error(message, False)
    hint = exc.hint or ""
    assert exc.code == yt.EXIT_NEEDS_COOKIES
    assert "--cookies-from-browser firefox" in hint and "ask the user" in hint and "YouTube" not in hint
    used = yt.classify_ytdlp_error(message, True)
    assert used.code == yt.EXIT_NEEDS_COOKIES and "still refused" in (used.hint or "")


def test_classify_drm_live_and_missing_video() -> None:
    drm = yt.classify_ytdlp_error("ERROR: [Foo] 1: This video is DRM protected", False)
    assert drm.code == yt.EXIT_NO_TRANSCRIPT and str(drm).startswith("DRM-protected content is not supported")
    upcoming = yt.classify_ytdlp_error("ERROR: [youtube] x: This live event will begin in 3 hours.", False)
    assert upcoming.code == yt.EXIT_NO_TRANSCRIPT and "live" in str(upcoming)
    page = yt.classify_ytdlp_error("ERROR: Unsupported URL: https://example.com/", False)
    assert page.code == yt.EXIT_NO_TRANSCRIPT and str(page).startswith("no video found")


@pytest.mark.parametrize("status", ["is_live", "is_upcoming", "post_live"])
def test_check_playable_rejects_live_states_on_every_site(status: str) -> None:
    with pytest.raises(yt.FetchError, match="finished, processed recordings work") as exc:
        yt.check_playable({"extractor_key": "TwitchVod", "live_status": status})
    assert exc.value.code == yt.EXIT_NO_TRANSCRIPT


def test_check_playable_accepts_vods_and_rejects_drm_only() -> None:
    yt.check_playable({"live_status": "was_live", "formats": [{"url": "x"}]})  # a finished, processed VOD
    yt.check_playable({"live_status": "not_live"})
    with pytest.raises(yt.FetchError, match="DRM") as exc:
        yt.check_playable({"_has_drm": True, "formats": [{"url": "x", "has_drm": True}]})
    assert exc.value.code == yt.EXIT_NO_TRANSCRIPT


class _PlaylistYDL:
    def __init__(self, results: dict) -> None:
        self.results = results

    def extract_info(self, url: str, download: bool = False, ie_key: str | None = None) -> dict:
        return self.results[url]


def test_extract_info_rejects_playlists_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", argparse.Namespace(DownloadError=RuntimeError))
    two = {"_type": "playlist", "entries": [{"url": "a"}, {"url": "b"}]}
    with pytest.raises(yt.FetchError) as exc:
        yt.extract_info(_PlaylistYDL({"p": two}), "p", False)
    assert exc.value.code == yt.EXIT_NO_TRANSCRIPT and "playlist <url>" in (exc.value.hint or "")
    one = {"_type": "playlist", "entries": [{"url": "v", "ie_key": "X"}]}
    video = {"id": "v", "title": "Only video"}
    assert yt.extract_info(_PlaylistYDL({"p": one, "v": video}), "p", False) == video


def test_fetch_playlist_url_exits_3_with_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run_main(monkeypatch, tmp_path, ["fetch", "https://www.youtube.com/playlist?list=PL1234567890"])
    out = capsys.readouterr().out.splitlines()
    assert code == yt.EXIT_NO_TRANSCRIPT and out[0].startswith("ERROR:") and "playlist <url>" in out[1]
    assert _work_dirs_in(tmp_path) == []


# --- Phase 4: playlist listing -------------------------------------------------


def test_playlist_url_uses_youtube_list() -> None:
    watch = f"https://www.youtube.com/watch?v={VID}&list=PLabc&index=2"
    assert yt.playlist_url(watch) == "https://www.youtube.com/playlist?list=PLabc"
    assert yt.playlist_url("vimeo.com/showcase/123") == "https://vimeo.com/showcase/123"
    assert yt.playlist_url("https://www.youtube.com/@3blue1brown") == "https://www.youtube.com/@3blue1brown/videos"
    assert yt.playlist_url("https://youtube.com/channel/UCx/") == "https://www.youtube.com/channel/UCx/videos"
    assert yt.playlist_url("https://www.youtube.com/@3blue1brown/shorts") == "https://www.youtube.com/@3blue1brown/shorts"
    with pytest.raises(yt.FetchError):
        yt.playlist_url("C:\\videos")


def test_entry_line_and_markers() -> None:
    first = {"url": "https://u/1", "title": "  First \n video ", "duration": 754}
    assert yt.entry_line(1, first) == "ENTRY: 1 [12:34] https://u/1 First video"
    private = {"url": "https://u/2", "title": "[Private video]"}
    assert yt.entry_line(2, private) == "ENTRY: 2 [?] https://u/2 (unavailable: private) [Private video]"
    members = {"url": "https://u/3", "title": "Members", "availability": "subscriber_only", "duration": 60}
    assert yt.entry_line(3, members).endswith("(unavailable: members only) Members")
    assert "(upcoming)" in yt.entry_line(4, {"url": "https://u/4", "title": "Soon", "live_status": "is_upcoming"})
    long = {"webpage_url": "https://u/5", "title": "Long", "duration": 3725}
    assert yt.entry_line(5, long) == "ENTRY: 5 [1:02:05] https://u/5 Long"


def test_playlist_lines_counts_estimate_and_truncation() -> None:
    entries = [{"url": f"https://u/{i}", "title": f"V{i}", "duration": 600, "playlist_index": i + 1} for i in range(3)]
    entries[1] = {"url": "https://u/1", "title": "[Deleted video]", "playlist_index": 2}
    info = {"title": "My list", "uploader": "Chan", "playlist_count": 3}
    out = yt.playlist_lines(info, entries, 200)
    assert out[:6] == [
        "PLAYLIST: My list",
        "CHANNEL: Chan",
        "COUNT: 3",
        "TOTAL: 3",
        "DURATION_TOTAL: 20:00",
        f"ESTIMATE: ~{2 * yt.whisper_estimate_min(600):.1f} min if every listed video needed Whisper",
    ]
    assert out[6].startswith("ENTRY: 1 [10:00]") and "(unavailable: deleted)" in out[7] and out[8].startswith("ENTRY: 3 ")
    assert out[-1] == "NOTE: 1 entries are unavailable, live or upcoming; ESTIMATE leaves them out"
    many = [{"url": f"https://u/{i}", "title": "x"} for i in range(6)]
    out = yt.playlist_lines({"title": "Big"}, many, 5)
    assert "COUNT: 5" in out and "TOTAL: unknown" in out and sum(o.startswith("ENTRY:") for o in out) == 5
    assert "NOTE: truncated to the first 5 entries (--limit 5); the playlist has more" in out
    assert any("no known duration" in o for o in out)
    assert "TOTAL: 4" in yt.playlist_lines({"title": "Small"}, many[:4], 5)


def test_playlist_limit_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert yt.build_parser().parse_args(["playlist", "https://x"]).limit == yt.PLAYLIST_LIMIT_DEFAULT == 200
    assert _run_main(monkeypatch, tmp_path, ["playlist", "https://x", "--limit", "0"]) == yt.EXIT_ERROR
    assert capsys.readouterr().out.startswith("ERROR: invalid playlist request")


# --- Phase 4: frames inputs ----------------------------------------------------


def test_scoped_ffmpeg_cookies_and_local_input_args() -> None:
    cookies = "tt_chain_token=abc; Domain=.tiktok.com; Path=/; Secure; Expires=1790000000; ttwid=1%7Cx; Domain=.tiktok.com"
    url = "https://v16-webapp.tiktok.com/video/1.mp4"
    assert yt.ffmpeg_cookies(cookies, url) == (
        "tt_chain_token=abc; path=/; domain=.tiktok.com;\r\nttwid=1%7Cx; path=/; domain=.tiktok.com;\r\n"
    )
    fmt = {"url": url, "http_headers": {"Referer": "https://www.tiktok.com/", "Cookie": "leak=1"}, "cookies": cookies}
    args = yt.stream_input_args(fmt)
    headers = args[args.index("-headers") + 1]
    assert "Referer:" in headers and "Cookie" not in headers and "leak" not in headers  # never a raw header
    assert args[args.index("-cookies") + 1] == yt.ffmpeg_cookies(cookies, url)
    assert yt.stream_input_args({"local": True}) == ()
    assert "-cookies" not in yt.stream_input_args({"url": url, "http_headers": {}})
    # Quoted values are decoded; a cookie without Domain is scoped to the format's host.
    assert yt.ffmpeg_cookies('t="a/b+c=="; x="semi\\073colon"; Domain=.tiktok.com', url) == (
        "t=a/b+c==; path=/; domain=v16-webapp.tiktok.com;\r\nx=semi;colon; path=/; domain=.tiktok.com;\r\n"
    )
    assert yt.ffmpeg_cookies(None, url) == ""


def test_formats_without_size_or_codec() -> None:
    formats = [
        {"format_id": "hls-720", "url": "u1", "vcodec": None, "height": 720, "protocol": "m3u8_native", "ext": "mp4"},
        {"format_id": "http-sd", "url": "u2", "vcodec": None, "protocol": "https", "ext": "mp4"},
        {"format_id": "audio", "url": "u3", "vcodec": None, "protocol": "https", "ext": "m4a"},
    ]
    assert yt.choose_video_format(formats, 1080)["format_id"] == "http-sd"  # the only plain-HTTPS video
    assert [f["format_id"] for f in yt.seek_formats(formats)] == ["http-sd", "hls-720"]
    assert yt.choose_overview_format(formats)["format_id"] == "hls-720"
    unsized = [{"format_id": "a", "url": "u", "vcodec": "h264", "tbr": 900}, {"format_id": "b", "url": "u", "vcodec": "h264", "tbr": 300}]
    assert yt.choose_overview_format(unsized)["format_id"] == "b"


# --- Phase 4: clean without network ---------------------------------------------


def _work(root: Path, key: str, **stored: str) -> Path:
    d = root / key
    d.mkdir(parents=True)
    (d / "meta.json").write_text(yt.json.dumps(stored), encoding="utf-8")
    return d


def test_clean_resolves_urls_paths_and_keys_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "work"
    monkeypatch.setattr(yt, "WORK_ROOT", root)
    # The offline key; no yt-dlp request. The TikTok page's own key differs from the dir's (a vm. short link).
    monkeypatch.setattr(yt, "site_key", lambda url: "vimeo-76979871" if "vimeo" in url else "tiktok-1")
    clip = tmp_path / "my clip.mp4"
    clip.write_bytes(b"video")
    site = _work(root, "vimeo-76979871", url="https://vimeo.com/76979871")
    redirected = _work(root, "tiktokvm-zm123", input_url="https://vm.tiktok.com/ZM123/", url="https://www.tiktok.com/@a/video/1")
    local = _work(root, yt.file_key(clip.resolve()), source_path=str(clip.resolve()))
    older = _work(root, "file-000000000000", source_path=str(clip.resolve()))  # before an edit
    other = _work(root, VID, url=f"https://www.youtube.com/watch?v={VID}")
    assert yt.matching_work_dirs("https://vimeo.com/76979871") == [site]
    assert yt.matching_work_dirs("https://www.tiktok.com/@a/video/1") == [redirected]  # by stored URL
    assert yt.matching_work_dirs(f'"{clip}"') == sorted([local, older])
    assert yt.matching_work_dirs("tiktokvm-zm123") == [redirected]  # a work key
    assert yt.matching_work_dirs(f"https://youtu.be/{VID}") == [other]
    clip.unlink()  # a deleted original is still found by its stored path
    assert yt.matching_work_dirs(str(clip)) == sorted([local, older])
    assert yt.matching_work_dirs("gibberish words") is None


def test_clean_local_path_keeps_the_original(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "work"
    clip = tmp_path / "clip with spaces.mkv"
    clip.write_bytes(b"original bytes")
    monkeypatch.setattr(yt, "WORK_ROOT", root)
    _work(root, yt.file_key(clip.resolve()), source_path=str(clip.resolve()))
    keep = _work(root, VID, url="x")
    assert _run_main(monkeypatch, root, ["clean", str(clip)]) == yt.EXIT_OK
    assert capsys.readouterr().out.strip() == "REMOVED: 1"
    assert clip.read_bytes() == b"original bytes" and keep.exists()
    assert _run_main(monkeypatch, root, ["clean", "gibberish words"]) == yt.EXIT_ERROR


# --- Phase 4: fetch for sites and local files (no network, no GPU) -------------

SITE_VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello from Vimeo.\n"


def _site_info(**kw: object) -> dict:
    base = {
        "id": "76979871",
        "extractor_key": "Vimeo",
        "webpage_url": "https://vimeo.com/76979871",
        "title": "Site video",
        "uploader": "Uploader",
        "duration": 10,
        "description": "desc",
        "language": "en",
        "subtitles": {"en": [{"ext": "vtt", "data": SITE_VTT}]},
    }
    return base | kw


def _fake_site(monkeypatch: pytest.MonkeyPatch, info: dict) -> None:
    monkeypatch.setitem(sys.modules, "yt_dlp", argparse.Namespace(YoutubeDL=_FakeYDL))
    monkeypatch.setattr(yt, "site_key", lambda url: "vimeo-76979871")
    monkeypatch.setattr(yt, "extract_info", lambda *a: info)


def test_site_fetch_uses_manual_subtitles_in_original_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_site(monkeypatch, _site_info())
    monkeypatch.setattr(yt, "run_whisper", lambda *a, **k: pytest.fail("must not transcribe"))
    assert _run_main(monkeypatch, tmp_path, ["fetch", "https://vimeo.com/76979871"]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "SOURCE: manual-subtitles" and "LANGUAGE: en" in out
    transcript = (tmp_path / "vimeo-76979871" / "transcript.md").read_text(encoding="utf-8")
    assert "- URL: https://vimeo.com/76979871" in transcript and "[00:01] Hello from Vimeo." in transcript
    meta = yt.json.loads((tmp_path / "vimeo-76979871" / "meta.json").read_text(encoding="utf-8"))
    assert (meta["source_kind"], meta["extractor"], meta["video_id"]) == ("site", "Vimeo", "vimeo-76979871")


@pytest.mark.parametrize("info_kw", [{"language": None}, {"extractor_key": "TikTok"}])
def test_site_fetch_whisper_when_language_unknown_or_machine_captions(
    info_kw: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_site(monkeypatch, _site_info(**info_kw))
    calls = []

    def fake_whisper(*a: object, **k: object) -> yt.Result:
        calls.append(a)
        return yt.Result("whisper", "en", _segs([(0, 4, "Hello there.")]), [], {"model": "large-v3", "vad_filter": True}, {})

    monkeypatch.setattr(yt, "run_whisper", fake_whisper)
    assert _run_main(monkeypatch, tmp_path, ["fetch", "https://vimeo.com/76979871"]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("SOURCE: whisper") and len(calls) == 1 and "using whisper" in out


def test_site_fetch_gpu_unavailable_has_no_auto_captions_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_site(monkeypatch, _site_info(language=None, automatic_captions={"en-orig": FMT}))

    def no_gpu(*a: object, **k: object) -> None:
        raise yt.GpuUnavailable("CUDA failed with error out of memory")

    monkeypatch.setattr(yt, "run_whisper", no_gpu)
    assert _run_main(monkeypatch, tmp_path, ["fetch", "https://vimeo.com/76979871"]) == yt.EXIT_NO_TRANSCRIPT
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("ERROR: Whisper could not run on the GPU") and "only for YouTube" in out[1]
    assert not (tmp_path / "vimeo-76979871").exists()


LOCAL_PROBE = {
    "format": {"duration": "12.0", "tags": {"title": "Local clip"}},
    "streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 640, "height": 360},
        {"index": 1, "codec_type": "audio", "codec_name": "aac", "tags": {"language": "eng"}},
        {"index": 2, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "eng"}},
    ],
}


def _fake_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, probe: dict) -> Path:
    clip = tmp_path / "videos" / "my clip.mkv"
    clip.parent.mkdir()
    clip.write_bytes(b"original")
    monkeypatch.setattr(yt.shutil, "which", lambda name: name)
    monkeypatch.setattr(yt, "probe_file", lambda ffprobe, path: probe)
    return clip


def test_local_fetch_uses_matching_embedded_subtitles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = _fake_local(monkeypatch, tmp_path, LOCAL_PROBE)
    monkeypatch.setattr(yt, "extract_embedded_subtitle", lambda f, p, i: yt.parse_subtitles(SRT))
    monkeypatch.setattr(yt, "run_whisper", lambda *a, **k: pytest.fail("must not transcribe"))
    work = tmp_path / "work"
    assert _run_main(monkeypatch, work, ["fetch", str(clip)]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "SOURCE: manual-subtitles" and "TITLE: Local clip" in out and "LANGUAGE: eng" in out
    transcript = (work / yt.file_key(clip.resolve()) / "transcript.md").read_text(encoding="utf-8")
    assert f"- File: {clip.resolve()}" in transcript and "track embedded #2 subrip 'eng'" in transcript
    assert clip.read_bytes() == b"original"


def test_local_fetch_whisper_on_language_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probe = yt.json.loads(yt.json.dumps(LOCAL_PROBE))
    probe["streams"][2]["tags"]["language"] = "tur"
    clip = _fake_local(monkeypatch, tmp_path, probe)
    seen: dict = {}

    def fake_whisper(ydl: object, info: dict, directory: Path, args: object, lang: str, get_audio: object = None) -> yt.Result:
        seen.update(ydl=ydl, lang=lang, get_audio=get_audio, audio_index=info["audio_index"])
        return yt.Result("whisper", "en", _segs([(0, 4, "Spoken words here.")]), [], {"model": "large-v3", "vad_filter": True}, {})

    monkeypatch.setattr(yt, "run_whisper", fake_whisper)
    assert _run_main(monkeypatch, tmp_path / "work", ["fetch", str(clip)]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("SOURCE: whisper") and "do not match the audio language 'eng'" in out
    assert seen["ydl"] is None and callable(seen["get_audio"]) and seen["lang"] == "eng" and seen["audio_index"] == 1
    assert clip.read_bytes() == b"original"


def test_local_fetch_without_ffmpeg_fails_before_work_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(yt.shutil, "which", lambda name: None)
    work = tmp_path / "work"
    work.mkdir()
    assert _run_main(monkeypatch, work, ["fetch", str(clip)]) == yt.EXIT_ERROR
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "ERROR: ffmpeg was not found on PATH" and "URLs" in out[1]
    assert _work_dirs_in(work) == []


def test_work_outputs_never_target_the_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yt, "WORK_ROOT", tmp_path / "work")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    expected = tmp_path / "work" / "file-1" / "audio.wav"
    assert yt._work_output(tmp_path / "work" / "file-1", "audio.wav", clip) == expected
    with pytest.raises(yt.FetchError, match="refusing"):
        yt._work_output(tmp_path, "clip.mp4", clip)  # outside the work root


def test_has_audio() -> None:
    assert yt.has_audio({"formats": [{"acodec": "none", "vcodec": "avc1"}, {"acodec": None}]})  # unknown counts
    assert not yt.has_audio({"formats": [{"acodec": "none", "vcodec": "avc1"}, {"acodec": "none", "vcodec": None}]})
    assert yt.has_audio({})  # no formats listed: let the download decide


# --- Fixes after independent verification (2026-10-20) ---------------------------


def test_extract_info_rejects_multi_video(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", argparse.Namespace(DownloadError=RuntimeError))
    parts = {"_type": "multi_video", "entries": [{"url": "p1"}, {"url": "p2"}]}
    with pytest.raises(yt.FetchError) as exc:
        yt.extract_info(_PlaylistYDL({"m": parts}), "m", False)
    assert exc.value.code == yt.EXIT_NO_TRANSCRIPT and "playlist <url>" in (exc.value.hint or "")


def test_fetch_multi_video_leaves_no_work_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "yt_dlp", argparse.Namespace(YoutubeDL=lambda *a, **k: _PlaylistYDLContext()))
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", argparse.Namespace(DownloadError=RuntimeError))
    monkeypatch.setattr(yt, "site_key", lambda url: "bilibili-bv1")
    assert _run_main(monkeypatch, tmp_path, ["fetch", "https://www.bilibili.com/video/BV1"]) == yt.EXIT_NO_TRANSCRIPT
    assert "HINT: list its videos" in capsys.readouterr().out
    assert _work_dirs_in(tmp_path) == []


class _PlaylistYDLContext(_FakeYDL):
    def extract_info(self, url: str, download: bool = False, ie_key: str | None = None) -> dict:
        return {"_type": "multi_video", "entries": [{"url": "p1"}, {"url": "p2"}]}


def _make_link(link: Path, target: Path) -> None:
    """A directory junction (Windows) or symlink; skips the test when neither is permitted."""
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        return
    except (ImportError, OSError, AttributeError):
        pass
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating links is not permitted here")


def test_purge_and_clean_all_skip_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, outside = tmp_path / "work", tmp_path / "outside"
    outside.mkdir()
    (outside / "audio.webm").write_text("user data")
    (outside / "video.mp4").write_text("user data")
    root.mkdir()
    _make_link(root / "linked", outside)
    assert yt.is_link(root / "linked") and not yt.is_link(outside)
    now = time.time()
    os.utime(outside, (now - 25 * 3600, now - 25 * 3600))  # old enough to be purged if followed
    assert yt.purge(root, now=now) == []
    assert sorted(p.name for p in outside.iterdir()) == ["audio.webm", "video.mp4"]
    monkeypatch.setattr(yt, "WORK_ROOT", root)
    assert _run_main(monkeypatch, root, ["clean", "--all"]) == yt.EXIT_OK
    assert capsys.readouterr().out.strip() == "REMOVED: 0"
    assert yt.matching_work_dirs("linked", root) is None or yt.matching_work_dirs("linked", root) == []
    assert sorted(p.name for p in outside.iterdir()) == ["audio.webm", "video.mp4"]


def test_cache_language_compares_normalized() -> None:
    meta = {"source": "whisper", "language": "eng", "settings": {"model": "large-v3", "vad_filter": True}}
    assert yt.cache_satisfies(meta, _args(lang="en"))
    assert yt.cache_satisfies(meta | {"language": "en"}, _args(lang="eng"))
    assert not yt.cache_satisfies(meta, _args(lang="tr"))


def test_existing_id_like_file_is_a_local_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yt, "WORK_ROOT", tmp_path / "work")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "myclip-2024").write_bytes(b"x")  # 11 characters, matches the YouTube id pattern
    assert yt.resolve_source("myclip-2024").kind == "file"
    (tmp_path / "otherclip01").mkdir()  # a directory of that name does not count
    assert yt.resolve_source("otherclip01").kind == "youtube"
    assert yt.resolve_source(VID).kind == "youtube"


@pytest.mark.skipif(os.name != "nt", reason="UNC paths are Windows paths")
def test_file_uri_unc_and_local() -> None:
    assert yt.file_uri_path("file://server/share/my%20clip.mp4") == Path(r"\\server\share\my clip.mp4")
    assert yt.file_uri_path("file:///C:/videos/a.mp4") == Path(r"C:\videos\a.mp4")
    assert yt.file_uri_path("file://localhost/C:/videos/a.mp4") == Path(r"C:\videos\a.mp4")
    assert yt.file_uri_path("file:C:/videos/a.mp4") == Path(r"C:\videos\a.mp4")


def test_clean_accepts_work_dir_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "work"
    monkeypatch.setattr(yt, "WORK_ROOT", root)
    work = _work(root, "vimeo-76979871", url="https://vimeo.com/76979871")
    (work / "transcript.md").write_text("x")
    keep = _work(root, VID, url="x")
    outside = tmp_path / "transcript.md"
    outside.write_text("user file")
    assert yt.matching_work_dirs(str(work / "transcript.md"), root) == [work]
    assert yt.matching_work_dirs(f'"{work}"', root) == [work]
    assert yt.matching_work_dirs(str(root / "gone-dir" / "transcript.md"), root) == []
    assert _run_main(monkeypatch, root, ["clean", str(work / "transcript.md")]) == yt.EXIT_OK
    assert capsys.readouterr().out.strip() == "REMOVED: 1"
    assert not work.exists() and keep.exists() and outside.read_text() == "user file"
    assert _run_main(monkeypatch, root, ["clean", str(root)]) == yt.EXIT_OK  # the root itself: nothing
    assert capsys.readouterr().out.strip() == "REMOVED: 0" and keep.exists()


# --- Phase 5: sound map (pure logic) -------------------------------------------

FR = 25  # model frames per second


def _track(seconds: float, spikes: list[tuple[float, float, float]] = (), base: float = -5.0) -> list[int]:
    """A logit track (hundredths) at `base` with (start_s, end_s, logit) plateaus."""
    out = [round(base * 100)] * round(seconds * FR)
    for a, b, v in spikes:
        for i in range(round(a * FR), min(len(out), round(b * FR))):
            out[i] = round(v * 100)
    return out


def test_chunk_plan_covers_audio_with_end_aligned_chunks() -> None:
    assert yt.chunk_plan(0, 750, 100) == []
    assert yt.chunk_plan(300, 750, 100) == [(0, 300)]  # shorter than one chunk
    assert yt.chunk_plan(750, 750, 100) == [(0, 750)]
    plan = yt.chunk_plan(2000, 750, 100)
    assert plan == [(0, 750), (650, 750), (1250, 750)]  # the last one ends exactly at the end
    assert all(n == 750 for _, n in plan) and plan[-1][0] + 750 == 2000


def test_stitch_tracks_takes_each_half_of_an_overlap_from_one_chunk() -> None:
    a = [1] * 10  # frames 0-9
    b = [2] * 10  # frames 6-15: overlap 6-9, midpoint 8
    assert list(yt.stitch_tracks([(6, b), (0, a)], 16)) == [1] * 8 + [2] * 8
    assert list(yt.stitch_tracks([(0, [5, 5])], 4, fill=-1)) == [5, 5, -1, -1]


def test_span_across_a_chunk_boundary_is_one_span() -> None:
    # A sound from 27 to 33 s; chunk 1 covers 0-30 s, chunk 2 26-56 s (the overlap midpoint is 28 s).
    first = _track(30, [(27, 30, 8.0)])
    second = _track(30, [(1, 7, 8.0)])  # its 1-7 s is 27-33 s of the audio
    track = yt.stitch_tracks([(0, first), (26 * FR, second)], 56 * FR)
    assert yt.detect_spans(track, 4.0, 5, 0) == [(27.0, 33.0, 8.0)]


def test_second_and_rolling_medians() -> None:
    assert yt.second_medians([1, 2, 3, 100, 5] + [7] * 25, 5) == [3, 7, 7, 7, 7, 7]
    assert yt.rolling_median([1, 9, 2, 8, 3], 1) == [5.0, 2, 8, 3, 5.5]


def test_detect_spans_threshold_and_hysteresis() -> None:
    # 1-2 s at 5 (reaches 4), dips to 3.5 (above 4 - 1) for 0.4 s, back to 5: one span 1-3 s.
    track = _track(5, [(1, 2, 5.0), (2, 2.4, 3.5), (2.4, 3, 5.0)])
    assert yt.detect_spans(track, 4.0, 5, 0) == [(1.0, 3.0, 5.0)]
    # A run that only reaches 3.5 never counts.
    assert yt.detect_spans(_track(5, [(1, 3, 3.5)]), 4.0, 5, 0) == []
    # Without hysteresis the dip splits it.
    assert len(yt.detect_spans(track, 4.0, 1, 0, hysteresis=0.0)) == 2


def test_detect_spans_merges_gaps_before_the_minimum_duration() -> None:
    claps = _track(5, [(1.0, 1.12, 6.0), (1.4, 1.52, 6.0), (1.8, 1.92, 6.0)])  # 0.12 s hits, 0.28 s apart
    assert yt.detect_spans(claps, 4.0, round(0.2 * FR), round(0.5 * FR)) == [(1.0, 1.92, 6.0)]
    assert yt.detect_spans(claps, 4.0, round(0.2 * FR), 0) == []  # unmerged, each hit is too short


def test_detect_spans_music_margin_only_where_music_is_heard() -> None:
    track = _track(10, [(0, 10, 4.5), (5, 6, 12.0)])  # high throughout, a real event at 5-6 s
    base = yt.rolling_median(yt.second_medians(track), 45)
    gated = yt.detect_spans(track, 4.0, 5, 12, baseline=base, margin=6.0, gate=[True] * 10)
    assert gated == [(5.0, 6.0, 12.0)]
    ungated = yt.detect_spans(track, 4.0, 5, 12, baseline=base, margin=6.0, gate=[False] * 10)
    assert ungated == [(0.0, 10.0, 12.0)]


def test_music_gate() -> None:
    music = _track(10, [(0, 5, 10.0)], base=-8.0)
    gate = yt.music_gate(music)
    assert gate[:3] == [True] * 3 and gate[-3:] == [False] * 3


def test_prune_colabels_keeps_the_clear_label() -> None:
    spans = [
        {"label": "waves", "start": 10.0, "end": 14.0, "score": 12.0},
        {"label": "water", "start": 10.5, "end": 13.0, "score": 7.0},  # 5 below waves: dropped
        {"label": "wind", "start": 11.0, "end": 12.0, "score": 9.0},  # within 4: kept
        {"label": "dog barking", "start": 30.0, "end": 31.0, "score": 5.0},  # alone: kept
    ]
    kept = [s["label"] for s in yt.prune_colabels(spans)]
    assert kept == ["waves", "wind", "dog barking"]
    many = [{"label": f"l{i}", "start": 0.0, "end": 1.0, "score": 10.0 - 0.1 * i} for i in range(6)]
    assert [s["label"] for s in yt.prune_colabels(many)] == ["l0", "l1", "l2", "l3"]


def test_sound_spans_coverage_and_events() -> None:
    tracks = {label: _track(60) for label, _ in yt.SOUND_PROMPTS}
    tracks["speech"] = _track(60, [(0, 20, 6.0), (20.8, 40, 6.0)])  # a 0.8 s pause is bridged
    tracks["laughter"] = _track(60, [(30, 31, 7.0), (50, 50.1, 9.0)])  # 0.1 s: too short
    spans = yt.sound_spans(tracks)
    assert [(s["label"], s["start"], s["end"]) for s in spans] == [("speech", 0.0, 40.0), ("laughter", 30.0, 31.0)]
    assert yt.coverage(spans, 60) == {"music": 0.0, "speech": 0.667, "singing": 0.0}
    assert yt.event_counts(spans) == {"laughter": 1}


def test_sounds_summary_line() -> None:
    cover = {"music": 0.39, "speech": 0.612, "singing": 0.0}
    assert yt.sounds_summary(cover, {}) == "music 39%, speech 61%, singing 0%; no other sounds detected"
    counts = {"laughter": 3, "applause": 2, "a": 1, "b": 1}
    assert yt.sounds_summary(cover, counts, limit=2) == "music 39%, speech 61%, singing 0%; laughter 3, applause 2 (+2 more labels)"


def _span(label: str, a: float, b: float, score: float = 6.0) -> dict:
    return {"label": label, "start": a, "end": b, "score": score}


def test_timeline_lines_both_stamp_forms_and_grouping() -> None:
    spans = [_span("music", 0, 28.4), _span("speech", 0, 60), _span("laughter", 65.2, 65.9), _span("applause", 65.5, 65.8)]
    lines, dropped = yt.timeline_lines(spans, False)
    assert lines == ["[00:00-00:28] music", "[01:05] laughter, applause"] and dropped == 0
    lines, _ = yt.timeline_lines([_span("explosion", 3725.0, 3727.5)], True)
    assert lines == ["[1:02:05-1:02:07] explosion"]


def test_timeline_noise_cap_keeps_the_highest_scores() -> None:
    spans = [_span("beeping", i, i + 0.5, score=float(i)) for i in range(10)]
    lines, dropped = yt.timeline_lines(spans, False, limit=3)
    assert dropped == 7 and lines == ["[00:07] beeping", "[00:08] beeping", "[00:09] beeping"]


EBUR_LOG = """[Parsed_ebur128_0 @ 0000] t: 0.0999792  TARGET:-23 LUFS    M:-120.7 S:-120.7     I: -70.0 LUFS       LRA:   0.0 LU
[Parsed_ebur128_0 @ 0000] t: 3.0999792  TARGET:-23 LUFS    M: -19.5 S: -17.1     I: -18.4 LUFS       LRA:   5.2 LU
[Parsed_ebur128_0 @ 0000] t: 3.1999792  TARGET:-23 LUFS    M: -16.7 S: -12.0     I: -18.4 LUFS       LRA:   5.2 LU
[Parsed_ebur128_0 @ 0000] t: 20.5  TARGET:-23 LUFS    M: -16.7 S: -14.0     I: -18.4 LUFS       LRA:   5.2 LU
[Parsed_ebur128_0 @ 0000] t: 22.0  TARGET:-23 LUFS    M: -16.7 S: -13.0     I: -18.4 LUFS       LRA:   5.2 LU
[Parsed_ebur128_0 @ 0000] Summary:

  Integrated loudness:
    I:         -17.4 LUFS
    Threshold: -27.5 LUFS

  Loudness range:
    LRA:         7.3 LU
"""
SILENCE_LOG = """[silencedetect @ 0000] silence_start: 10.5
[silencedetect @ 0000] silence_end: 14 | silence_duration: 3.5
[silencedetect @ 0000] silence_start: 55.25
"""


def test_parse_ebur128_and_peaks() -> None:
    samples, integrated, lra = yt.parse_ebur128(EBUR_LOG)
    assert integrated == -17.4 and lra == 7.3
    assert samples[0] == (3.0999792, -17.1) and len(samples) == 4  # the -120.7 silence is dropped
    # Peaks are window centres (t - 1.5 s), at least 5 s apart, in time order.
    assert yt.loudness_peaks(samples, 5, 5.0) == [(1.7, -12.0), (20.5, -13.0)]
    assert yt.peak_separation(30) == 5.0 and yt.peak_separation(300) == 30.0 and yt.peak_separation(7200) == 60.0


def test_parse_silencedetect_closes_an_open_stretch_at_the_end() -> None:
    assert yt.parse_silencedetect(SILENCE_LOG, 60.0) == [(10.5, 14.0), (55.25, 60.0)]
    assert yt.parse_silencedetect("", 60.0) == []


def test_loudness_lines() -> None:
    loud = {"integrated_lufs": -17.4, "range_lu": 7.3, "peaks": [{"time": 65.0, "lufs": -11.0}], "silences": [{"start": 10.5, "end": 14.0}]}
    lines = yt.loudness_lines(loud, False)
    assert lines[0] == "- Integrated loudness: -17.4 LUFS; loudness range 7.3 LU"
    assert lines[1] == "- Loudest moments (short-term loudness, 3 s): [01:05] -11 LUFS"
    assert lines[2] == "- Silences (below -50 dB for 3 s or more): [00:10-00:14]"
    assert yt.loudness_lines(None, False) == ["- Not measured."]


def test_heard_lookup() -> None:
    spans = [_span("music", 0, 30, 9.0), _span("speech", 0, 30, 9.0), _span("laughter", 10.0, 11.0, 7.0), _span("applause", 12.5, 13.0, 8.0)]
    assert yt.heard_labels(spans, 11.8) == ["applause", "laughter", "music"]  # events first; within 1 s of both
    assert yt.heard_labels(spans, 20.0) == ["music"]  # speech is left out
    assert yt.heard_lines(spans, [11.5, 40.0], False) == ["HEARD: [00:11.5] applause, laughter, music"]
    assert yt.heard_lines(None, [1.0], False) == []


def test_query_parsing_and_event_lines() -> None:
    assert yt.parse_query(" car horn,glass  breaking , car horn,") == ["car horn", "glass breaking"]
    with pytest.raises(ValueError):
        yt.parse_query(" , ")
    with pytest.raises(ValueError):
        yt.parse_query(",".join(f"s{i}" for i in range(yt.SOUND_QUERY_MAX + 1)))
    spans = [_span("glass breaking", 65.24, 66.0, 7.25), _span("car horn", 3.0, 4.5, 5.0)]
    assert yt.event_lines(spans, False) == ["EVENT: [00:03-00:04.5] car horn 5.0", "EVENT: [01:05.2-01:06] glass breaking 7.2"]
    assert yt.event_lines([_span("x", 3600.0, 3601.0, 4.0)], True) == ["EVENT: [1:00:00-1:00:01] x 4.0"]


def test_query_miss_note_names_the_strongest_moment() -> None:
    note = yt.query_miss_note("a punch", _track(80, [(71.2, 71.3, 3.6)]), 4.0, False)
    assert note.startswith("'a punch' not found") and "strongest at [01:11.2] with score 3.6" in note


def test_parse_range_and_times() -> None:
    assert yt.parse_range("1:00-1:10") == (60.0, 70.0)
    assert yt.parse_range("[01:00] \u2013 [01:02.5]") == (60.0, 62.5)
    for bad in ("1:10-1:00", "1:00", "-1:00", "a-b"):
        with pytest.raises(ValueError):
            yt.parse_range(bad)
    assert yt.range_times(60, 62, 2) == [60.0, 60.5, 61.0, 61.5, 62.0]  # the end is included
    assert len(yt.range_times(0, 10, yt.RANGE_FPS_DEFAULT)) == 21
    assert yt.range_times(5, 5.3, 2) == [5.0]


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--range", "1:00-1:10", "--fps", "5", "--max-frames", "20"], "51 frames"),
        (["--range", "1:10-1:00"], "the end must be after the start"),
        (["--range", "1:00-1:10", "--fps", "0"], "--fps"),
        (["--range", "1:00-1:10", "--fps", "30"], "--fps"),
        (["--at", "1:00", "--fps", "2"], "--fps only applies to --range"),
        (["--range", "1:00-1:10", "--interval", "5"], "only apply to --overview"),
    ],
)
def test_frames_range_argument_errors(
    extra: list[str], message: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run_main(monkeypatch, tmp_path, ["frames", VID, *extra])
    out = capsys.readouterr().out
    assert code == yt.EXIT_ERROR and out.startswith("ERROR: invalid frames request:") and message in out
    assert _work_dirs_in(tmp_path) == []  # before any work dir or network access


def test_sounds_markdown_sections() -> None:
    data = {
        "model": yt.SOUND_MODEL, "threshold": 4.0, "duration": 72.0, "generated_at": "2026-10-20T12:00:00+03:00",
        "coverage": {"music": 1.0, "speech": 0.0, "singing": 0.6}, "counts": {"explosion": 1},
        "spans": [_span("music", 0, 72), _span("explosion", 71.0, 71.5)],
        "loudness": {"integrated_lufs": -17.4, "range_lu": 7.3, "peaks": [], "silences": []},
    }
    md = yt.sounds_markdown(data, "Meme")
    assert md.startswith("# Sound map: Meme\n") and "not emotions" in md
    assert "- Coverage: music 100%, speech 0%, singing 60%" in md and "- Events: explosion 1" in md
    assert "## Timeline\n\n[00:00-01:12] music\n[01:11] explosion\n" in md
    assert "## Loudness\n\n- Integrated loudness: -17.4 LUFS" in md


def test_decode_logits_round_trip() -> None:
    import base64 as b64
    from array import array

    values = array("h", [-32700, -5, 0, 412])
    if sys.byteorder == "big":
        values.byteswap()
    assert list(yt.decode_logits(b64.b64encode(values.tobytes()).decode())) == [-32700, -5, 0, 412]


# --- Phase 5: sound backend and fetch integration (model mocked) ----------------


def _pcm(path: Path, seconds: float = 4.0) -> Path:
    """Raw 48 kHz mono s16le silence, as the sound model reads it."""
    path.write_bytes(b"\0\0" * int(seconds * yt.SOUND_SAMPLE_RATE))
    return path


def _b64(values: list[int]) -> str:
    import base64 as b64
    from array import array

    a = array("h", values)
    if sys.byteorder == "big":
        a.byteswap()
    return b64.b64encode(a.tobytes()).decode()


def _helper_writes(response: dict | None, seen: dict | None = None):
    """A fake run_helper that answers the request like sound_events.py would."""

    def fake(cmd: list[str], env: dict, timeout: float) -> int:
        req = yt.json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        if seen is not None:
            seen.update(cmd=cmd, env=env, req=req, timeout=timeout)
        if response is not None:
            body = response(req) if callable(response) else response | {"request_id": req["request_id"]}
            Path(req["out"]).write_text(yt.json.dumps(body), encoding="utf-8")
        return 0 if response else 1

    return fake


def test_run_sound_backend_writes_request_and_stitches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pcm = _pcm(tmp_path / "audio.48k.pcm", 2.0)  # 50 frames: one chunk
    seen: dict = {}

    def answer(req: dict) -> dict:
        logits = [_b64([100 * (i + 1)] * 50) for i in range(len(req["prompts"]))]
        chunk = {"start": 0, "frames": 50, "logits": logits, "level": _b64([-2000] * 50)}
        return {"ok": True, "request_id": req["request_id"], "prompts": req["prompts"], "chunks": [chunk], "timings": {}}

    monkeypatch.setattr(yt, "find_uv", lambda: "uv")
    monkeypatch.setattr(yt, "run_helper", _helper_writes(answer, seen))
    tracks, levels, _ = yt.run_sound_backend(pcm, ["music", "laughter"], tmp_path)
    assert seen["cmd"][:4] == ["uv", "run", "--quiet", "--script"] and seen["cmd"][4].endswith("sound_events.py")
    req = seen["req"]
    assert req["chunks"] == [[0, 50]] and req["model"] == yt.SOUND_MODEL and req["format"] == "s16le"
    assert req["parent_pid"] == os.getpid() and req["deadline"] > time.time() and len(req["request_id"]) == 16
    assert {k: list(v) for k, v in tracks.items()} == {"music": [100] * 50, "laughter": [200] * 50}
    assert list(levels) == [-2000] * 50


def test_helper_env_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    venv_bin = str(Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin"))
    tools = str(Path(sys.prefix).parent / "tools")
    monkeypatch.setattr(yt, "PATH_AT_START", os.pathsep.join([venv_bin, tools]))
    monkeypatch.setenv("PATH", os.pathsep.join([str(Path(sys.prefix) / "nvidia" / "cublas" / "bin"), tools]))  # Whisper's
    monkeypatch.setenv("VIRTUAL_ENV", str(Path(sys.prefix).parent / "parent-env"))
    env = yt.helper_env()
    assert env["PATH"] == tools and "VIRTUAL_ENV" not in env and yt.SOUND_CACHE_ENV not in env


def test_helper_env_restores_ld_library_path_and_passes_the_sound_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(yt.REEXEC_ENV, "/usr/local/lib")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/env/nvidia/cublas/lib:/usr/local/lib")
    monkeypatch.setenv("VIDEO_WATCH_SOUND_CACHE", str(tmp_path / "hf"))
    env = yt.helper_env()
    assert env["LD_LIBRARY_PATH"] == "/usr/local/lib" and yt.REEXEC_ENV not in env
    assert env[yt.SOUND_CACHE_ENV] == str((tmp_path / "hf").absolute())
    monkeypatch.setenv(yt.REEXEC_ENV, "")
    assert "LD_LIBRARY_PATH" not in yt.helper_env()


@pytest.mark.parametrize(
    "response,message,gpu",
    [
        ({"ok": False, "kind": "gpu", "message": "CUDA out of memory"}, "the GPU is busy or unavailable (CUDA out of memory)", True),
        ({"ok": False, "kind": "model", "message": "404"}, "could not be loaded", False),
        ({"ok": True, "prompts": ["music"], "chunks": [{"start": 0}]}, "malformed", False),
        (None, "gave no result", False),
    ],
)
def test_run_sound_backend_failures(
    response: dict | None, message: str, gpu: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pcm = _pcm(tmp_path / "audio.48k.pcm", 1.0)
    monkeypatch.setattr(yt, "find_uv", lambda: "uv")
    monkeypatch.setattr(yt, "run_helper", _helper_writes(response))
    with pytest.raises(yt.SoundUnavailable) as exc:
        yt.run_sound_backend(pcm, ["music"], tmp_path)
    assert message in str(exc.value) and exc.value.gpu is gpu


def test_stale_response_of_another_request_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pcm = _pcm(tmp_path / "audio.48k.pcm", 1.0)
    late = {"ok": True, "request_id": "0123456789abcdef", "prompts": ["music"], "chunks": []}
    monkeypatch.setattr(yt, "find_uv", lambda: "uv")
    monkeypatch.setattr(yt, "run_helper", _helper_writes(lambda req: late))
    with pytest.raises(yt.SoundUnavailable, match="another request"):
        yt.run_sound_backend(pcm, ["music"], tmp_path)


def test_empty_audio_skips_the_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(yt, "find_uv", lambda: "uv")
    monkeypatch.setattr(yt, "run_helper", lambda *a: pytest.fail("no model run for empty audio"))
    with pytest.raises(yt.SoundUnavailable) as exc:
        yt.run_sound_backend(_pcm(tmp_path / "audio.48k.pcm", 0.0), ["music"], tmp_path)
    assert exc.value.permanent and "empty" in str(exc.value)


class _FakeProc:
    def __init__(self, *a: object, **k: object) -> None:
        self.pid = 4242
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        if not self.killed:
            raise yt.subprocess.TimeoutExpired("uv", timeout)
        return 1

    def kill(self) -> None:
        self.killed = True


def test_run_helper_timeout_kills_the_whole_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    procs: list[_FakeProc] = []
    killed: list[int] = []

    def popen(*a: object, **k: object) -> _FakeProc:
        procs.append(_FakeProc())
        return procs[-1]

    def kill_tree(proc: _FakeProc, job: object = None) -> None:
        killed.append(proc.pid)
        proc.killed = True

    monkeypatch.setattr(yt.subprocess, "Popen", popen)
    monkeypatch.setattr(yt, "kill_tree", kill_tree)
    monkeypatch.setattr(yt._KillOnCloseJob, "add", lambda self, proc: True)
    with pytest.raises(yt.SoundUnavailable, match="timed out .* and was stopped"):
        yt.run_helper(["uv", "run", "x"], {}, 0.01)
    assert killed == [4242]


def _laughing_tracks(seconds: float = 4.0) -> dict[str, list[int]]:
    tracks = {prompt: _track(seconds) for _, prompt in yt.SOUND_PROMPTS}
    tracks["speech"] = _track(seconds, [(0, seconds, 6.0)])
    tracks["laughter"] = _track(seconds, [(1.0, 2.0, 8.0)])
    return tracks


class _SoundFakes:
    """Mocks for the sound map: ffmpeg, uv, the 48 kHz conversion, loudness and the model."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, fail: Exception | None = None) -> None:
        self.backend_calls: list[list[str]] = []
        self.sources: list[tuple[Path, int | None]] = []
        self.fail = fail
        monkeypatch.setattr(yt, "SOUND_MAP_DEFAULT", True)
        monkeypatch.setattr(yt.shutil, "which", lambda name: name)
        monkeypatch.setattr(yt, "find_uv", lambda: "uv")
        monkeypatch.setattr(yt, "to_sound_pcm", self.to_pcm)
        monkeypatch.setattr(yt, "analyze_loudness", lambda f, w, d: ({"integrated_lufs": -20.0, "range_lu": 4.0, "peaks": [], "silences": []}, []))
        monkeypatch.setattr(yt, "run_sound_backend", self.backend)
        monkeypatch.setattr(yt, "silent_stretches", lambda f, p, d: [])

    def to_pcm(self, ffmpeg: str, src: Path, directory: Path, stream: int | None = None) -> Path:
        self.sources.append((src, stream))
        return _pcm(directory / "audio.48k.pcm")

    def backend(self, pcm: Path, prompts: list[str], directory: Path, silences: object = ()) -> tuple[dict, None, dict]:
        self.backend_calls.append(prompts)
        if self.fail:
            raise self.fail
        tracks = _laughing_tracks()
        return {p: tracks[p] for p in prompts}, None, {"timings": {"infer_s": 0.1}, "peak_vram_mb": 4200}


def _fake_youtube(monkeypatch: pytest.MonkeyPatch, info: dict) -> list[Path]:
    downloads: list[Path] = []

    def download(ydl: object, info: dict, directory: Path, args: object) -> Path:
        audio = directory / "audio.webm"
        audio.write_bytes(b"opus")
        downloads.append(audio)
        return audio

    monkeypatch.setitem(sys.modules, "yt_dlp", argparse.Namespace(YoutubeDL=_FakeYDL))
    monkeypatch.setattr(yt, "extract_info", lambda *a: info)
    monkeypatch.setattr(yt, "download_audio", download)
    monkeypatch.setattr(yt, "download_subtitle", lambda ydl, fmts: yt.parse_subtitles(SRT))
    return downloads


YT_INFO = {"id": VID, "title": "Talk", "duration": 4, "language": "en", "subtitles": {"en": FMT}}


def test_fetch_builds_the_sound_map_after_the_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = _SoundFakes(monkeypatch)
    downloads = _fake_youtube(monkeypatch, YT_INFO)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "SOURCE: manual-subtitles"
    i = next(k for k, line in enumerate(out) if line.startswith("SOUNDS: "))
    assert out[i] == "SOUNDS: music 0%, speech 100%, singing 0%; laughter 1"
    assert out[i + 1] == f"SOUNDS_FILE: {(tmp_path / VID / 'sounds.md').resolve()}" and out[i + 2] == "CACHED: no"
    assert out[i - 1].startswith("CHAPTERS:")
    work = tmp_path / VID
    data = yt.json.loads((work / "sounds.json").read_text(encoding="utf-8"))
    assert data["version"] == yt.sound_version() and data["spans"][1] == {"label": "laughter", "start": 1.0, "end": 2.0, "score": 8.0}
    meta = yt.json.loads((work / "meta.json").read_text(encoding="utf-8"))
    assert meta["sounds"]["model"] == yt.SOUND_MODEL and meta["sounds"]["threshold"] == yt.SOUND_THRESHOLD
    assert "[00:01-00:02] laughter" in (work / "sounds.md").read_text(encoding="utf-8")
    # The audio was downloaded for the sound map only (manual subtitles) and is gone again.
    assert len(downloads) == 1 and fakes.sources == [(downloads[0], None)]
    assert not list(work.glob("audio.*")) and not list(work.glob("sound-*.json"))


def test_fetch_sound_map_failure_is_a_note_and_keeps_exit_0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _SoundFakes(monkeypatch, fail=yt.SoundUnavailable("the GPU is busy or unavailable (CUDA out of memory)", gpu=True))
    _fake_youtube(monkeypatch, YT_INFO)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("SOURCE: manual-subtitles") and "SOUNDS" not in out
    assert "NOTE: sound map skipped: the GPU is busy or unavailable (CUDA out of memory)" in out
    work = tmp_path / VID
    assert (work / "transcript.md").is_file() and not (work / "sounds.json").exists()
    meta = yt.json.loads((work / "meta.json").read_text(encoding="utf-8"))
    assert "gpu_busy_at" in meta["sounds"]  # a cached fetch retries after the back-off
    assert not list(work.glob("audio.*"))


def test_fetch_no_sounds_skips_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = _SoundFakes(monkeypatch)
    downloads = _fake_youtube(monkeypatch, YT_INFO)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID, "--no-sounds"]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "SOUNDS" not in out and "sound map" not in out
    assert fakes.backend_calls == [] and downloads == []


def test_fetch_whisper_keeps_its_audio_for_the_sound_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = _SoundFakes(monkeypatch)
    downloads = _fake_youtube(monkeypatch, YT_INFO | {"subtitles": {}})
    seen: dict = {}

    def fake_whisper(ydl: object, info: dict, directory: Path, args: object, lang: str, keep_audio: bool = False) -> yt.Result:
        seen["keep_audio"] = keep_audio
        (directory / "audio.webm").write_bytes(b"opus")  # the download Whisper used
        return yt.Result("whisper", "en", _segs([(0, 3, "Hello there friends.")]), [], {"model": "large-v3", "vad_filter": True}, {})

    monkeypatch.setattr(yt, "run_whisper", fake_whisper)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert seen["keep_audio"] is True and downloads == []  # no second download
    assert fakes.sources[0][0].name == "audio.webm" and "SOUNDS: " in out
    assert out.index("SPEECH:") < out.index("SOUNDS:") < out.index("CACHED:")
    assert not list((tmp_path / VID).glob("audio.*"))


def test_fetch_auto_captions_fallback_skips_the_sound_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = _SoundFakes(monkeypatch)
    _fake_youtube(monkeypatch, YT_INFO | {"subtitles": {}, "automatic_captions": {"en-orig": FMT}})

    def no_gpu(*a: object, **k: object) -> None:
        raise yt.GpuUnavailable("CUDA failed with error out of memory")

    monkeypatch.setattr(yt, "run_whisper", no_gpu)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("SOURCE: auto-subtitles-fallback") and "NOTE: sound map skipped: the GPU is unavailable" in out
    assert fakes.backend_calls == []


def test_silent_video_skip_is_remembered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = _SoundFakes(monkeypatch)
    _fake_youtube(monkeypatch, YT_INFO | {"formats": [{"format_id": "v", "acodec": "none", "url": "u"}]})
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    assert "NOTE: sound map skipped: the video has no audio track" in capsys.readouterr().out
    meta = yt.json.loads((tmp_path / VID / "meta.json").read_text(encoding="utf-8"))
    assert meta["sounds"] == {"skipped": "the video has no audio track"}
    monkeypatch.setattr(yt, "extract_info", lambda *a: pytest.fail("no back-fill for a silent video"))
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    assert "CACHED: yes" in capsys.readouterr().out and fakes.backend_calls == []


def test_cached_fetch_backfills_the_sound_map_without_whisper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / VID
    work.mkdir()
    yt.write_outputs(work, _meta(duration=4), _segs([(0, 3, "Merhaba.")]))  # a cache from before phase 5
    fakes = _SoundFakes(monkeypatch)
    downloads = _fake_youtube(monkeypatch, YT_INFO)
    monkeypatch.setattr(yt, "run_whisper", lambda *a, **k: pytest.fail("must not transcribe"))
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID, "--no-sounds"]) == yt.EXIT_OK
    assert "SOUNDS" not in capsys.readouterr().out and downloads == []  # --no-sounds: no back-fill
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "CACHED: yes" in out and "SOUNDS: music 0%, speech 100%, singing 0%; laughter 1" in out
    assert len(downloads) == 1 and not list(work.glob("audio.*")) and not (work / ".lock").exists()
    # Current map: the next fetch reuses it without any audio.
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    assert "SOUNDS: " in capsys.readouterr().out and len(fakes.backend_calls) == 1 and len(downloads) == 1


def test_outdated_sound_map_is_rebuilt(tmp_path: Path) -> None:
    (tmp_path / "sounds.json").write_text(yt.json.dumps({"version": "old", "spans": []}), encoding="utf-8")
    assert yt.needs_sound_map(tmp_path, {})
    (tmp_path / "sounds.json").write_text(yt.json.dumps({"version": yt.sound_version(), "spans": []}), encoding="utf-8")
    assert not yt.needs_sound_map(tmp_path, {})
    assert not yt.needs_sound_map(tmp_path / "missing", {"sounds": {"skipped": "no audio"}})


def test_local_fetch_sound_map_reads_the_original_in_place(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = _fake_local(monkeypatch, tmp_path, LOCAL_PROBE)
    fakes = _SoundFakes(monkeypatch)
    monkeypatch.setattr(yt, "extract_embedded_subtitle", lambda f, p, i: yt.parse_subtitles(SRT))
    assert _run_main(monkeypatch, tmp_path / "work", ["fetch", str(clip)]) == yt.EXIT_OK
    assert "SOUNDS: " in capsys.readouterr().out
    assert fakes.sources == [(clip.resolve(), 1)] and clip.read_bytes() == b"original"


def test_has_results_counts_a_sound_map(tmp_path: Path) -> None:
    assert not yt.has_results(tmp_path)
    (tmp_path / "sounds.json").write_text("{}")
    assert yt.has_results(tmp_path)


def test_frames_print_heard_from_the_cached_sound_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = _cached_frames(tmp_path, with_transcript=True)
    spans = [_span("music", 0, 60, 9.0), _span("laughter", 8.5, 9.5, 7.0)]
    (work / "sounds.json").write_text(yt.json.dumps({"version": yt.sound_version(), "spans": spans}), encoding="utf-8")
    monkeypatch.setattr(yt.shutil, "which", lambda name: "ffmpeg")
    assert _run_main(monkeypatch, tmp_path, ["frames", VID, "--at", "0:09,1"]) == yt.EXIT_OK
    out = [o for o in capsys.readouterr().out.splitlines() if o.startswith(("FRAME:", "SAID:", "HEARD:"))]
    assert out[1:3] == ["SAID: [00:01] first line", "HEARD: [00:01] music"]
    assert out[3].startswith("FRAME: [00:09]") and out[4:6] == ["SAID: [00:09] after a gap", "HEARD: [00:09] laughter, music"]


def test_frames_without_sound_map_print_no_heard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _cached_frames(tmp_path, with_transcript=True)
    monkeypatch.setattr(yt.shutil, "which", lambda name: "ffmpeg")
    assert _run_main(monkeypatch, tmp_path, ["frames", VID, "--at", "1"]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "HEARD" not in out and "NOTE" not in out


def test_sounds_query_prints_events_and_misses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _SoundFakes(monkeypatch)
    _fake_youtube(monkeypatch, YT_INFO)

    def backend(pcm: Path, prompts: list[str], directory: Path, silences: object = ()) -> tuple[dict, None, dict]:
        assert prompts == ["music", "glass breaking", "car horn"]
        return {"music": _track(4), "glass breaking": _track(4, [(2.4, 3.0, 9.0)]), "car horn": _track(4, [(1, 1.1, 3.0)])}, None, {}

    monkeypatch.setattr(yt, "run_sound_backend", backend)
    code = _run_main(monkeypatch, tmp_path, ["sounds", VID, "--query", "glass breaking, car horn"])
    out = capsys.readouterr().out.splitlines()
    assert code == yt.EXIT_OK and out[0] == f"VIDEO_ID: {VID}" and out[3] == "QUERY: glass breaking, car horn"
    assert out[4] == "EVENT: [00:02.4-00:03] glass breaking 9.0"
    assert out[5].startswith("NOTE: 'car horn' not found") and "strongest at [00:01] with score 3.0" in out[5]
    work = tmp_path / VID
    assert not work.exists() or not list(work.glob("audio.*"))  # the audio is deleted; nothing else kept


def test_sounds_query_threshold_and_argument_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_main(monkeypatch, tmp_path, ["sounds", VID, "--threshold", "3"]) == yt.EXIT_ERROR
    assert "ERROR: invalid sounds request: --threshold only applies to --query" in capsys.readouterr().out
    assert _run_main(monkeypatch, tmp_path, ["sounds", VID, "--query", " , "]) == yt.EXIT_ERROR
    assert _work_dirs_in(tmp_path) == []


def test_sounds_command_builds_then_reuses_the_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = _SoundFakes(monkeypatch)
    _fake_youtube(monkeypatch, YT_INFO)
    assert _run_main(monkeypatch, tmp_path, ["sounds", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == [f"VIDEO_ID: {VID}", "TITLE: Talk", "DURATION: 00:04"]
    assert out[3].startswith("SOUNDS: ") and out[4].startswith("SOUNDS_FILE: ") and out[5] == "CACHED: no"
    data = yt.json.loads((tmp_path / VID / "sounds.json").read_text(encoding="utf-8"))
    assert data["url"] == f"https://www.youtube.com/watch?v={VID}"  # clean finds it without network
    assert _run_main(monkeypatch, tmp_path, ["sounds", VID]) == yt.EXIT_OK
    assert "CACHED: yes" in capsys.readouterr().out and len(fakes.backend_calls) == 1
    assert yt.matching_work_dirs(f"https://youtu.be/{VID}", tmp_path) == [tmp_path / VID]


def test_sounds_command_gpu_failure_exits_3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _SoundFakes(monkeypatch, fail=yt.SoundUnavailable("the GPU is busy or unavailable (OOM)", gpu=True))
    _fake_youtube(monkeypatch, YT_INFO)
    assert _run_main(monkeypatch, tmp_path, ["sounds", VID]) == yt.EXIT_NO_TRANSCRIPT
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "ERROR: sound map unavailable: the GPU is busy or unavailable (OOM)" and out[1] == f"HINT: {yt.SOUNDS_HINT}"
    assert not (tmp_path / VID).exists()  # a failed first run leaves nothing behind


# --- Phase 5 fixes after independent verification -------------------------------


def test_silence_is_muted_before_spans() -> None:
    seconds = 70
    tracks = {label: _track(seconds) for label, _ in yt.SOUND_PROMPTS}
    tracks["music"] = _track(seconds, [(0, seconds, 7.0)])  # the model's verdict on digital zeros
    tracks["motorcycle"] = _track(seconds, [(30, 50, 9.0)])
    tracks["speech"] = _track(seconds, [(0, 20, 6.0), (65, 70, 6.0)])
    levels = [-2000] * (20 * FR) + [-20000] * (45 * FR) + [-2000] * (5 * FR)  # speech, 45 s of zeros, speech
    spans = yt.sound_spans(tracks, levels=levels, duration=seconds)
    labels = {(s["label"], s["start"], s["end"]) for s in spans}
    assert ("speech", 0.0, 20.0) in labels and ("speech", 65.0, 70.0) in labels
    assert not any(s["start"] < 65 and s["end"] > 20 for s in spans)  # nothing heard in the silence
    assert yt.silent_runs([-2000, -8000, -8000, -2000, -7100]) == [(1, 3), (4, 5)]
    assert yt.silence_gate(tracks, None) is tracks  # another backend without levels: unchanged


def test_span_ends_are_clamped_to_the_audio() -> None:
    tracks = {label: _track(4) for label, _ in yt.SOUND_PROMPTS}
    tracks["laughter"] = _track(4, [(3.0, 4.0, 8.0)])
    spans = yt.sound_spans(tracks, duration=3.71)
    assert [(s["label"], s["end"]) for s in spans] == [("laughter", 3.71)]


def test_prune_colabels_sweep_matches_the_pairwise_rule() -> None:
    import random

    def naive(spans: list[dict]) -> list[dict]:
        kept = []
        for s in spans:
            best: dict = {}
            for o in spans:
                if o["start"] < s["end"] and o["end"] > s["start"]:
                    best[o["label"]] = max(best.get(o["label"], -1e9), o["score"])
            ranked = sorted(best, key=lambda label: (-best[label], label))
            if s["score"] >= best[ranked[0]] - yt.SOUND_COLABEL_MARGIN and s["label"] in ranked[: yt.SOUND_COLABEL_TOP]:
                kept.append(s)
        return kept

    rng = random.Random(5)
    for _ in range(30):
        spans = []
        for _ in range(rng.randint(0, 60)):
            a = round(rng.uniform(0, 100), 2)
            spans.append(_span(f"l{rng.randint(0, 7)}", a, round(a + rng.choice([0.2, 1.0, 3.0, 40.0]), 2), round(rng.uniform(4, 14), 2)))
        assert yt.prune_colabels(spans) == naive(spans)


def test_sound_version_covers_the_coverage_labels_and_silence_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    before = yt.sound_version()
    monkeypatch.setattr(yt, "SOUND_COVERAGE_LABELS", ("music", "speech"))
    assert yt.sound_version() != before
    monkeypatch.undo()
    monkeypatch.setattr(yt, "SOUND_SILENCE_DBFS", -60.0)
    assert yt.sound_version() != before


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(13, "The process cannot access the file", "audio.webm.part"),
        KeyError("chunks"),
        EOFError(),
    ],
)
def test_any_sound_map_error_keeps_the_fetch_output_and_exit_code(
    error: Exception, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _SoundFakes(monkeypatch, fail=error)
    _fake_youtube(monkeypatch, YT_INFO)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    captured = capsys.readouterr()
    out = captured.out.splitlines()
    assert out[0] == "SOURCE: manual-subtitles" and "CACHED: no" in out and not any(o.startswith("ERROR") for o in out)
    assert f"NOTE: sound map skipped: unexpected error ({type(error).__name__}" in captured.out
    assert "Traceback" in captured.err
    assert (tmp_path / VID / "transcript.md").is_file() and not list((tmp_path / VID).glob("audio.*"))


def test_backfill_errors_are_notes_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / VID
    work.mkdir()
    yt.write_outputs(work, _meta(duration=4), _segs([(0, 3, "Merhaba.")]))
    _SoundFakes(monkeypatch)
    _fake_youtube(monkeypatch, YT_INFO)

    def broken(*a: object) -> dict:
        raise OSError(22, "yt-dlp wrapped an OSError")

    monkeypatch.setattr(yt, "extract_info", broken)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "CACHED: yes" in out and "NOTE: sound map skipped: unexpected error (OSError" in out
    assert not (work / ".lock").exists()


def test_busy_gpu_is_detected_before_any_download_and_backs_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = _SoundFakes(monkeypatch)
    downloads = _fake_youtube(monkeypatch, YT_INFO)
    monkeypatch.setattr(yt, "gpu_free_mb", lambda: 1200)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "NOTE: sound map skipped: the GPU is busy (1200 MB of VRAM free" in out
    assert downloads == [] and fakes.backend_calls == []  # nothing fetched for a busy GPU
    meta = yt.json.loads((tmp_path / VID / "meta.json").read_text(encoding="utf-8"))
    assert meta["sounds"]["gpu_busy_at"] <= time.time()
    # A cached fetch inside the back-off does not even check again.
    monkeypatch.setattr(yt, "gpu_free_mb", lambda: pytest.fail("no check during the back-off"))
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "CACHED: yes" in out and "NOTE: sound map skipped: the GPU was busy 0 min ago" in out
    # After the back-off, with a free GPU, the back-fill runs.
    meta["sounds"]["gpu_busy_at"] = time.time() - yt.SOUND_GPU_RETRY_S - 1
    (tmp_path / VID / "meta.json").write_text(yt.json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(yt, "gpu_free_mb", lambda: 12000)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    assert "SOUNDS: " in capsys.readouterr().out and len(downloads) == 1


def test_gpu_free_mb_reads_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # the autouse fixture stubs gpu_free_mb
    monkeypatch.setattr(yt, "SOUND_MAP_DEFAULT", False)
    monkeypatch.setattr(yt.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    seen: list = []

    def run(cmd: list[str], **k: object) -> argparse.Namespace:
        seen.append(cmd)
        return argparse.Namespace(returncode=0, stdout="9876\n")

    monkeypatch.setattr(yt.subprocess, "run", run)
    assert yt.gpu_free_mb() == 9876 and seen[0][-2:] == ["-i", "1"]
    monkeypatch.setattr(yt.shutil, "which", lambda name: None)
    assert yt.gpu_free_mb() is None


def test_fetch_reuses_a_current_sound_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = _SoundFakes(monkeypatch)
    downloads = _fake_youtube(monkeypatch, YT_INFO)
    assert _run_main(monkeypatch, tmp_path, ["sounds", VID]) == yt.EXIT_OK  # the map first, no transcript
    capsys.readouterr()
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "SOUNDS: " in out and len(fakes.backend_calls) == 1 and len(downloads) == 1  # no second download
    meta = yt.json.loads((tmp_path / VID / "meta.json").read_text(encoding="utf-8"))
    assert meta["sounds"]["version"] == yt.sound_version()


def test_sounds_prints_the_video_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _SoundFakes(monkeypatch)
    _fake_youtube(monkeypatch, YT_INFO | {"duration": 5})  # metadata says 0:05; the audio is 4 s
    assert _run_main(monkeypatch, tmp_path, ["sounds", VID]) == yt.EXIT_OK
    assert "DURATION: 00:05" in capsys.readouterr().out
    data = yt.json.loads((tmp_path / VID / "sounds.json").read_text(encoding="utf-8"))
    assert data["duration"] == 4.0 and data["media_duration"] == 5
    assert _run_main(monkeypatch, tmp_path, ["sounds", VID]) == yt.EXIT_OK  # cached: the same duration
    assert "DURATION: 00:05" in capsys.readouterr().out


def test_query_for_music_is_not_measured_against_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _SoundFakes(monkeypatch)
    _fake_youtube(monkeypatch, YT_INFO)
    music = _track(4, [(0, 4, 12.0)])
    monkeypatch.setattr(yt, "run_sound_backend", lambda pcm, prompts, d, sil=(): ({"music": music, "singing": music}, None, {}))
    assert _run_main(monkeypatch, tmp_path, ["sounds", VID, "--query", "music, singing"]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "EVENT: [00:00-00:04] music 12.0" in out and "EVENT: [00:00-00:04] singing 12.0" in out


@pytest.mark.parametrize("value", ["inf", "nan", "100", "-11"])
def test_sounds_threshold_must_be_a_sensible_logit(
    value: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_main(monkeypatch, tmp_path, ["sounds", VID, "--query", "x", "--threshold", value]) == yt.EXIT_ERROR
    assert "ERROR: invalid sounds request: --threshold must be a logit" in capsys.readouterr().out
    assert _work_dirs_in(tmp_path) == []


def test_remove_downloads_retries_a_file_held_briefly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    held = tmp_path / "audio.48k.pcm"
    held.write_bytes(b"x")
    real_unlink = Path.unlink
    calls = {"n": 0}

    def flaky(self: Path, missing_ok: bool = False) -> None:
        if self == held and calls["n"] < 2:
            calls["n"] += 1
            raise PermissionError(32, "being used by another process")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky)
    monkeypatch.setattr(yt.time, "sleep", lambda s: None)
    yt.remove_downloads(tmp_path)
    assert not held.exists() and calls["n"] == 2


def test_segment_plan_keeps_long_silences_out_of_the_model_input() -> None:
    plan = yt.segment_plan(65 * FR, [(20.0, 65.0)], 750, 100)  # speech, then 45 s of silence
    assert plan == [(0, 500)]
    plan = yt.segment_plan(85 * FR, [(20.0, 65.0)], 750, 100)  # speech, silence, speech
    assert plan == [(0, 500), (65 * FR, 500)]
    assert yt.segment_plan(85 * FR, [(20.0, 20.5)], 750, 100) == yt.chunk_plan(85 * FR, 750, 100)  # short: kept
    long_tail = yt.segment_plan(100 * FR, [(0.0, 10.0)], 750, 100)
    assert long_tail[0][0] == 10 * FR and long_tail[-1][0] + long_tail[-1][1] == 100 * FR
    assert yt.segment_plan(40 * FR, [(0.0, 40.0)], 750, 100) == []


def test_all_silent_audio_gives_an_empty_map_without_the_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(yt, "find_uv", lambda: "uv")
    monkeypatch.setattr(yt, "run_helper", lambda *a: pytest.fail("no model run for silence"))
    tracks, levels, resp = yt.run_sound_backend(_pcm(tmp_path / "audio.48k.pcm", 4.0), ["music"], tmp_path, [(0.0, 4.0)])
    assert levels is None and resp["silent"] and set(tracks["music"]) == {yt.SOUND_FILL}
    assert yt.sound_spans({"music": tracks["music"]}) == []


def test_backend_request_skips_long_silences(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict = {}

    def answer(req: dict) -> dict:
        chunks = [{"start": s, "frames": n, "logits": [_b64([500] * n)], "level": _b64([-2000] * n)} for s, n in req["chunks"]]
        return {"ok": True, "request_id": req["request_id"], "prompts": req["prompts"], "chunks": chunks}

    monkeypatch.setattr(yt, "find_uv", lambda: "uv")
    monkeypatch.setattr(yt, "run_helper", _helper_writes(answer, seen))
    tracks, _, _ = yt.run_sound_backend(_pcm(tmp_path / "audio.48k.pcm", 10.0), ["music"], tmp_path, [(2.0, 8.0)])
    assert seen["req"]["chunks"] == [[0, 50], [200, 50]]
    music = list(tracks["music"])
    assert music[:50] == [500] * 50 and set(music[50:200]) == {yt.SOUND_FILL} and music[200:] == [500] * 50


# --- Phase 6: platform dirs, configuration, devices, models, doctor and setup -----


def test_platform_dirs_per_os(tmp_path: Path) -> None:
    home = tmp_path / "home"
    win = {"APPDATA": str(tmp_path / "Roaming"), "LOCALAPPDATA": str(tmp_path / "Local")}
    assert yt.user_config_dir("win32", win, home) == tmp_path / "Roaming" / "video-watch"
    assert yt.user_cache_dir("win32", win, home) == tmp_path / "Local" / "video-watch"
    assert yt.user_config_dir("win32", {}, home) == home / "AppData" / "Roaming" / "video-watch"
    assert yt.user_config_dir("darwin", {}, home) == home / "Library" / "Application Support" / "video-watch"
    assert yt.user_cache_dir("darwin", {}, home) == home / "Library" / "Caches" / "video-watch"
    assert yt.user_config_dir("linux", {}, home) == home / ".config" / "video-watch"
    assert yt.user_cache_dir("linux", {}, home) == home / ".cache" / "video-watch"
    xdg = {"XDG_CONFIG_HOME": str(tmp_path / "cfg"), "XDG_CACHE_HOME": str(tmp_path / "cache")}
    assert yt.user_config_dir("linux", xdg, home) == tmp_path / "cfg" / "video-watch"
    assert yt.user_cache_dir("freebsd", xdg, home) == tmp_path / "cache" / "video-watch"  # other POSIX: like Linux
    assert yt.user_config_dir("linux", {"XDG_CONFIG_HOME": "relative"}, home) == home / ".config" / "video-watch"


def test_default_model_dir_is_the_user_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    assert yt.model_dir() == yt.user_cache_dir() / "whisper"
    assert yt.default_setting("work_dir") == str(Path(yt.tempfile.gettempdir()) / "video-watch")


def test_hf_cache_dir_follows_huggingface_rules(tmp_path: Path) -> None:
    assert yt.hf_cache_dir({"HF_HUB_CACHE": str(tmp_path / "a")}) == tmp_path / "a"
    assert yt.hf_cache_dir({"HF_HOME": str(tmp_path / "h")}) == tmp_path / "h" / "hub"
    assert yt.hf_cache_dir({"XDG_CACHE_HOME": str(tmp_path)}) == tmp_path / "huggingface" / "hub"


# A drive path (`X:\\folder`, also with a doubled backslash as in Python source) or a home folder.
_MACHINE_PATH_RE = yt.re.compile(
    r"(?<![A-Za-z\\])[A-Za-z]:(?:\\{1,2}|/)[A-Za-z][\w .-]{2,}|/home/[a-z]|/Users/[A-Za-z]|\\Users\\"
)


def test_no_hard_coded_machine_paths_in_the_scripts() -> None:
    """The shipped scripts never name a machine's own folders; paths come from config, env vars
    and the platform's user dirs."""
    scripts = Path(yt.__file__).parent
    for name in ("video_watch.py", "sound_events.py"):
        for i, line in enumerate((scripts / name).read_text(encoding="utf-8").splitlines(), 1):
            assert not _MACHINE_PATH_RE.search(line), f"{name}:{i}: {line.strip()}"


def test_machine_path_check_catches_drive_and_home_paths() -> None:
    for text in (r"ROOT = r'D:\models'", r"ROOT = 'G:\\data'", "ROOT = 'E:\\\\cache'", "x = 'F:/data'", "'/home/someone/x'", "'/Users/someone'"):
        assert _MACHINE_PATH_RE.search(text), text
    for text in ("url = 'https://example.com/a:b'", r"re.compile(r'\bM:\s*(-?[\d.]+)\s+S:\s*')"):
        assert not _MACHINE_PATH_RE.search(text), text


def test_no_private_names_in_the_repo() -> None:
    """Maintainers can list private words (old names, folders) in VIDEO_WATCH_PRIVATE_NEEDLES,
    comma-separated; no text file of the repository may contain any of them as a whole word
    (case-sensitive, so a longer name that merely contains one does not count)."""
    needles = [n.strip() for n in os.environ.get("VIDEO_WATCH_PRIVATE_NEEDLES", "").split(",") if n.strip()]
    if not needles:
        pytest.skip("set VIDEO_WATCH_PRIVATE_NEEDLES to check for private words")
    repo = Path(__file__).resolve().parents[1]
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".venv"}
    hits = []
    for path in repo.rglob("*"):
        if path.is_dir() or skip_dirs & set(path.relative_to(repo).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        hits += [f"{path.relative_to(repo)}: {n}" for n in needles if yt.re.search(rf"(?<!\w){yt.re.escape(n)}(?!\w)", text)]
    assert hits == []


def _config_file() -> Path:
    return Path(os.environ[yt.CONFIG_ENV])


def test_config_precedence_flag_env_config_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert yt.resolve_setting("model") == ("large-v3", "default")
    assert yt.resolve_setting("sounds") == (False, "default")
    yt.save_config({"model": "small", "sounds": True})
    assert yt.resolve_setting("model") == ("small", "config")
    assert yt.resolve_setting("sounds") == (True, "config")
    monkeypatch.setenv("VIDEO_WATCH_MODEL", "medium")
    monkeypatch.setenv("VIDEO_WATCH_SOUNDS", "off")
    assert yt.resolve_setting("model") == ("medium", "env")
    assert yt.resolve_setting("sounds") == (False, "env")
    args = argparse.Namespace(model="tiny", sounds=True, no_sounds=False, device=None)
    assert yt.resolve_setting("model", args) == ("tiny", "flag")
    assert yt.resolve_setting("sounds", args) == (True, "flag")
    assert yt.resolve_setting("sounds", argparse.Namespace(sounds=None, no_sounds=True)) == (False, "flag")
    assert yt.resolve_setting("device", args) == ("auto", "default")


def test_config_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="model must be one of"):
        yt.parse_config_value("model", "huge")
    with pytest.raises(ValueError, match="device must be one of"):
        yt.parse_config_value("device", "gpu")
    with pytest.raises(ValueError, match="not on/off"):
        yt.parse_config_value("sounds", "maybe")
    assert yt.parse_config_value("device", "CPU") == "cpu"
    assert yt.parse_config_value("sounds", "yes") is True and yt.parse_config_value("sounds", False) is False
    monkeypatch.setenv("VIDEO_WATCH_DEVICE", "vulkan")  # not an engine (yet)
    with pytest.raises(yt.ConfigError, match="VIDEO_WATCH_DEVICE"):
        yt.resolve_setting("device")
    _config_file().write_text("{not json", encoding="utf-8")
    with pytest.raises(yt.ConfigError, match="not valid JSON"):
        yt.load_config()
    _config_file().write_text('{"model": "large-v9", "future_key": 1}', encoding="utf-8")
    with pytest.raises(yt.ConfigError, match="model must be one of"):
        yt.load_config()


def test_save_config_keeps_unknown_keys() -> None:
    _config_file().write_text('{"future_key": 1}', encoding="utf-8")
    yt.save_config({"model": "small"})
    yt.save_config({}, remove=["model"])
    assert yt.json.loads(_config_file().read_text(encoding="utf-8")) == {"future_key": 1}


def test_invalid_setting_stops_a_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("VIDEO_WATCH_MODEL", "huge")
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_ERROR
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("ERROR: VIDEO_WATCH_MODEL: model must be one of") and out[1].startswith("HINT: ")


def test_config_command(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert yt.main(["config", "path"]) == yt.EXIT_OK
    assert capsys.readouterr().out.strip() == f"CONFIG: {_config_file()}"
    assert yt.main(["config", "set", "sounds", "true"]) == yt.EXIT_OK
    assert capsys.readouterr().out.splitlines()[0] == "SET: sounds = true"
    assert yt.main(["config", "get", "sounds"]) == yt.EXIT_OK
    assert capsys.readouterr().out.strip() == "true"
    assert yt.main(["config", "set", "model", "large-v3-turbo"]) == yt.EXIT_OK
    capsys.readouterr()
    assert yt.main(["config", "show"]) == yt.EXIT_OK
    shown = capsys.readouterr().out.splitlines()
    assert shown[0] == f"CONFIG: {_config_file()}"
    assert "MODEL: large-v3-turbo (config)" in shown and "SOUNDS: true (config)" in shown and "DEVICE: auto (default)" in shown
    assert yt.main(["config", "unset", "model"]) == yt.EXIT_OK
    capsys.readouterr()
    assert yt.main(["config", "get", "model"]) == yt.EXIT_OK and capsys.readouterr().out.strip() == "large-v3"
    assert yt.main(["config", "set", "model", "huge"]) == yt.EXIT_ERROR
    assert capsys.readouterr().out.startswith("ERROR: invalid config request: model must be one of")
    assert yt.main(["config", "get", "nope"]) == yt.EXIT_ERROR
    capsys.readouterr()
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "notes.txt").write_text("user file")
    assert yt.main(["config", "set", "work_dir", str(busy)]) == yt.EXIT_ERROR
    assert "is not an empty folder" in capsys.readouterr().out


def test_configured_work_dir_must_be_new_empty_or_marked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "projects"
    folder.mkdir()
    (folder / "old").mkdir()
    os.utime(folder / "old", (0, 0))  # older than a day: the purge would delete it
    monkeypatch.setenv("VIDEO_WATCH_WORK_DIR", str(folder))
    monkeypatch.setattr(yt, "WORK_ROOT", yt.DEFAULT_WORK_ROOT)
    assert yt.main(["clean", "--all"]) == yt.EXIT_ERROR
    out = capsys.readouterr().out
    assert "already holds other files and is not a video-watch work dir" in out and "HINT: set work_dir" in out
    assert (folder / "old").is_dir()  # nothing was purged or cleaned
    fresh = tmp_path / "fresh"
    monkeypatch.setenv("VIDEO_WATCH_WORK_DIR", str(fresh))
    assert yt.main(["clean", "--all"]) == yt.EXIT_OK and yt.WORK_ROOT == fresh.absolute()
    assert (fresh / yt.WORK_ROOT_MARKER).is_file() and capsys.readouterr().out.strip() == "REMOVED: 0"


def test_purge_ignores_an_unmarked_root(tmp_path: Path) -> None:
    (tmp_path / "old").mkdir()
    os.utime(tmp_path / "old", (0, 0))
    assert yt.purge(tmp_path) == [] and (tmp_path / "old").is_dir()


def test_parser_accepts_all_models_devices_and_sound_flags() -> None:
    parser = yt.build_parser()
    for name in yt.MODEL_NAMES:
        assert parser.parse_args(["fetch", VID, "--model", name]).model == name
    args = parser.parse_args(["fetch", VID, "--device", "cpu", "--sounds"])
    assert args.device == "cpu" and args.sounds is True and args.no_sounds is False
    assert parser.parse_args(["fetch", VID]).sounds is None
    assert yt.MODEL_NAMES == ("large-v3", "large-v3-turbo", "large-v2", "medium", "small", "base", "tiny")


def test_parser_rejects_unknown_model_and_both_sound_flags(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (["fetch", VID, "--model", "huge"], ["fetch", VID, "--sounds", "--no-sounds"], ["fetch", VID, "--device", "gpu"]):
        with pytest.raises(SystemExit) as exc:
            yt.build_parser().parse_args(argv)
        assert exc.value.code == yt.EXIT_ERROR
    assert "ERROR:" in capsys.readouterr().out


def test_sounds_are_off_by_default_and_forced_per_run() -> None:
    plain = argparse.Namespace(no_sounds=False, sounds=None)
    assert not yt.sounds_wanted(plain)
    assert yt.sounds_wanted(argparse.Namespace(no_sounds=False, sounds=True))
    yt.save_config({"sounds": True})
    assert yt.sounds_wanted(plain)
    assert not yt.sounds_wanted(argparse.Namespace(no_sounds=True, sounds=None))


def test_fetch_with_sounds_off_prints_no_sound_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = _SoundFakes(monkeypatch)
    monkeypatch.setattr(yt, "SOUND_MAP_DEFAULT", False)
    downloads = _fake_youtube(monkeypatch, YT_INFO)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "SOUNDS" not in out and "sound map" not in out and fakes.backend_calls == [] and downloads == []
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID, "--sounds"]) == yt.EXIT_OK  # forced once: back-fill
    assert "SOUNDS: music 0%, speech 100%" in capsys.readouterr().out


def test_sound_map_not_set_up_is_a_note_for_fetch_and_exit_3_for_sounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = _SoundFakes(monkeypatch)
    monkeypatch.setattr(yt, "sound_model_cached", lambda: False)
    downloads = _fake_youtube(monkeypatch, YT_INFO)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "NOTE: sound map skipped: the sound map is not set up yet (setup --sounds on downloads about 5 GB once)" in out
    assert downloads == [] and fakes.backend_calls == []  # nothing downloaded for a map that is not set up
    assert _run_main(monkeypatch, tmp_path, ["sounds", VID]) == yt.EXIT_NO_TRANSCRIPT
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("ERROR: sound map unavailable: the sound map is not set up yet")
    assert out[1] == f"HINT: {yt.SOUNDS_SETUP_HINT}" and "setup --sounds on" in out[1]


def test_sound_map_without_nvidia_smi_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yt.shutil, "which", lambda name: None if name == "nvidia-smi" else name)
    with pytest.raises(yt.SoundUnavailable, match="needs an NVIDIA GPU"):
        yt.check_gpu_free()


def test_device_resolution_never_falls_back_silently(monkeypatch: pytest.MonkeyPatch) -> None:
    assert yt.resolve_device(argparse.Namespace(device=None)) == "cuda"  # auto means the GPU
    assert yt.resolve_device(argparse.Namespace(device="cuda")) == "cuda"
    assert yt.resolve_device(argparse.Namespace(device="cpu")) == "cpu"
    monkeypatch.setenv("VIDEO_WATCH_DEVICE", "cpu")
    assert yt.resolve_device(None) == "cpu"


def test_gpu_errors_split_no_cuda_from_busy() -> None:
    assert yt.gpu_unavailable(RuntimeError("CUDA failed with error out of memory")).no_cuda is False
    assert yt.gpu_unavailable(RuntimeError("CUDA driver version is insufficient for CUDA runtime version")).no_cuda
    assert yt.gpu_unavailable(OSError("Could not load library cudnn_ops64_9.dll")).no_cuda
    assert yt.gpu_unavailable(OSError("libcublas.so.12: cannot open shared object file")).no_cuda
    assert yt.GpuUnavailable("x", no_cuda=True).hint == yt.NO_CUDA_HINT
    assert yt.GpuUnavailable("x").hint == yt.GPU_RETRY_HINT and "only for YouTube" in yt.GPU_RETRY_HINT


def test_no_cuda_is_an_error_with_the_cpu_hint_even_on_youtube(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_youtube(monkeypatch, YT_INFO | {"subtitles": {}, "automatic_captions": {"en-orig": FMT}})

    def no_cuda(*a: object, **k: object) -> None:
        raise yt.GpuUnavailable("no CUDA device visible to CTranslate2", no_cuda=True)

    monkeypatch.setattr(yt, "run_whisper", no_cuda)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_NO_TRANSCRIPT
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "ERROR: no usable CUDA GPU for Whisper (no CUDA device visible to CTranslate2)"
    assert out[1] == f"HINT: {yt.NO_CUDA_HINT}" and "--device cpu" in out[1]


def test_load_whisper_without_cuda_raises_no_cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "ctranslate2", argparse.Namespace(get_cuda_device_count=lambda: 0))
    monkeypatch.setitem(sys.modules, "faster_whisper", argparse.Namespace(WhisperModel=lambda *a, **k: pytest.fail("no load")))
    with pytest.raises(yt.GpuUnavailable) as exc:
        yt.load_whisper("small", "cuda", tmp_path)
    assert exc.value.no_cuda


def test_load_whisper_on_cpu_uses_int8(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict = {}

    def model(name: str, **kw: object) -> str:
        seen.update(kw, name=name)
        return "model"

    monkeypatch.setitem(sys.modules, "faster_whisper", argparse.Namespace(WhisperModel=model))
    monkeypatch.setitem(sys.modules, "ctranslate2", argparse.Namespace(get_cuda_device_count=lambda: pytest.fail("no CUDA probe")))
    assert yt.load_whisper("small", "cpu", tmp_path) == "model"
    assert seen["device"] == "cpu" and seen["compute_type"] == "int8" and seen["cpu_threads"] >= 1
    assert seen["download_root"] == str(tmp_path) and seen["local_files_only"] is False


def test_cpu_run_is_labelled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict = {}

    def fake_transcribe(model: object, audio: Path, language: object, vad: bool, dur: object, batch: int = 8) -> tuple:
        seen["batch"] = batch
        info = argparse.Namespace(language="en", language_probability=1.0, duration=10)
        return _segs([(0, 5, "one two three four five six seven")]), info, {"batched": True, "batch_size": batch}

    def fake_load(name: str, device: str = "cuda", root: object = None) -> object:
        seen["device"] = device
        return object()

    monkeypatch.setattr(yt, "load_whisper", fake_load)
    monkeypatch.setattr(yt, "whisper_language", lambda code: code)
    monkeypatch.setattr(yt, "transcribe", fake_transcribe)
    args = argparse.Namespace(model="small", device="cpu", no_vad=False, lang=None, cookies_from_browser=None)
    result = yt.run_whisper(_FakeWhisperYDL(tmp_path), {"duration": 10}, tmp_path, args, "en")
    assert seen == {"device": "cpu", "batch": yt.CPU_BATCH_SIZE}
    assert result.settings["device"] == "cpu" and result.settings["compute_type"] == "int8"
    assert any("transcribed on the CPU" in n for n in result.notes)
    head = "\n".join(yt.header_lines(_meta(source="whisper", settings=result.settings | {"vad_min_silence_ms": 500})))
    assert "whisper (small, int8, cpu, VAD on" in head


def test_model_cache_layout(tmp_path: Path) -> None:
    assert not yt.model_cached("large-v3-turbo", tmp_path)
    snap = yt.model_cache_path("large-v3-turbo", tmp_path) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    assert snap.parent.parent.name == "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
    assert not yt.model_cached("large-v3-turbo", tmp_path)  # an interrupted download
    (snap / "model.bin").write_bytes(b"x")
    assert yt.model_cached("large-v3-turbo", tmp_path)


@pytest.mark.parametrize(
    "model,free,expected",
    [
        ("large-v3", None, 8),  # no nvidia-smi: the default
        ("large-v3", 14000, 8),
        ("large-v3", 7000, 4),  # an 8 GB card with the desktop running
        ("large-v3", 5300, 2),  # a 6 GB card
        ("large-v3", 4700, 1),
        ("large-v3", 3000, 1),  # will likely fail; the GPU error path reports it
        ("large-v3-turbo", 4000, 8),
        ("small", 2000, 8),
    ],
)
def test_batch_size_follows_free_vram(model: str, free: int | None, expected: int) -> None:
    assert yt.choose_batch_size(model, free) == expected


def test_model_vram_interpolates() -> None:
    assert yt.model_vram_mb("large-v3", 8) == 7156 and yt.model_vram_mb("large-v3", 1) == 4221
    assert yt.model_vram_mb("large-v3-turbo", 4) == round(2525 + (3325 - 2525) * 2 / 6)
    assert yt.model_vram_mb("unknown", 8) == 7156


def test_estimates_depend_on_model_device_and_batch() -> None:
    base = yt.whisper_estimate_min(600)
    assert base == pytest.approx(yt.ESTIMATE_OVERHEAD_MIN + yt.ESTIMATE_FACTOR * 10)
    assert yt.whisper_estimate_min(600, batch=2) > base > yt.whisper_estimate_min(600, model="large-v3-turbo")
    assert yt.whisper_estimate_min(600, model="small", device="cpu") == pytest.approx(yt.ESTIMATE_OVERHEAD_MIN + 0.75)
    assert yt.whisper_estimate_min(600, model="small", batch=1) == yt.whisper_estimate_min(600, model="small")


def _gpu(total: int | None, free: int | None = None, driver: str = "580.10") -> yt.GpuInfo:
    if not total:
        return yt.GpuInfo()
    return yt.GpuInfo("NVIDIA Test", total, free if free is not None else total, driver)


@pytest.mark.parametrize(
    "total,cuda_ok,expected",
    [
        (16311, True, ("large-v3", "cuda", True)),
        (8188, True, ("large-v3", "cuda", True)),
        (6144, True, ("large-v3-turbo", "cuda", True)),
        (4096, True, ("large-v3-turbo", "cuda", False)),
        (2048, True, ("small", "cuda", False)),
        (None, False, ("small", "cpu", False)),
    ],
)
def test_recommendations_by_vram(total: int | None, cuda_ok: bool, expected: tuple) -> None:
    assert yt.recommend(_gpu(total), cuda_ok)[:3] == expected


def test_model_fit_warnings() -> None:
    assert yt.model_fit_warning("large-v3", 16311) is None
    assert "memory may be tight" in yt.model_fit_warning("large-v3", 6144)
    assert "likely run out of memory" in yt.model_fit_warning("large-v3", 4096)
    assert yt.model_fit_warning("large-v3-turbo", 6144) is None
    assert yt.model_fit_warning("large-v3", None) is None


def test_cuda_status_without_heavy_packages() -> None:
    assert yt.cuda_status(_gpu(None), "linux")[0] is False
    assert "macOS has no CUDA" in yt.cuda_status(_gpu(16311), "darwin")[1]
    assert "too old" in yt.cuda_status(_gpu(16311, driver="470.1"), "win32", deep=False)[1]
    ok, why = yt.cuda_status(_gpu(16311), "linux", deep=False)
    assert ok and why.startswith("ok (driver 580.10")


def _doctor_env(monkeypatch: pytest.MonkeyPatch, gpu: yt.GpuInfo, cuda: tuple[bool, str], tools: bool = True) -> None:
    monkeypatch.setattr(yt, "gpu_info", lambda: gpu)
    monkeypatch.setattr(yt, "cuda_status", lambda g: cuda)
    monkeypatch.setattr(yt.shutil, "which", lambda name: f"/bin/{name}" if tools else None)
    monkeypatch.setattr(yt, "find_uv", lambda: "/bin/uv" if tools else None)
    monkeypatch.setattr(yt, "tool_version", lambda cmd: "uv 0.11.3" if "uv" in cmd[0] else "ffmpeg version 9.0 Copyright x")
    monkeypatch.setattr(yt, "disk_free_gb", lambda path: 120.5)


def _doctor_lines(capsys: pytest.CaptureFixture[str]) -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}
    for line in capsys.readouterr().out.splitlines():
        key, _, value = line.partition(": ")
        keys.setdefault(key, []).append(value)
    return keys


def test_doctor_reports_every_key_and_recommends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _doctor_env(monkeypatch, _gpu(16311, free=13000), (True, "ok (driver 580.10)"))
    monkeypatch.setenv("VIDEO_WATCH_MODEL_DIR", str(tmp_path / "models"))
    assert yt.main(["doctor"]) == yt.EXIT_OK
    keys = _doctor_lines(capsys)
    for key in (
        "OS", "PYTHON", "UV", "FFMPEG", "FFPROBE", "GPU", "VRAM_TOTAL_MB", "VRAM_FREE_MB", "CUDA", "DISK_FREE_GB", "CONFIG",
        "DEVICE", "MODEL", "MODEL_CACHED", "SOUNDS", "SOUND_MODEL_CACHED", "RECOMMEND_MODEL", "RECOMMEND_DEVICE",
        "RECOMMEND_SOUNDS", "READY",
    ):
        assert key in keys, key
    assert keys["GPU"] == ["NVIDIA Test (driver 580.10)"] and keys["VRAM_TOTAL_MB"] == ["16311"]
    assert keys["CUDA"] == ["ok (driver 580.10)"] and keys["MODEL_CACHED"] == ["no"] and keys["SOUNDS"] == ["off"]
    assert keys["FFMPEG"] == ["ffmpeg version 9.0"] and keys["UV"] == ["uv 0.11.3"]
    assert keys["RECOMMEND_MODEL"] == ["large-v3"] and keys["RECOMMEND_SOUNDS"][0].startswith("yes (")
    assert keys["READY"] == ["no"] and any("not downloaded yet" in w for w in keys["WARN"])
    assert not (tmp_path / "models").exists() and not _config_file().exists()  # read-only


def test_doctor_without_gpu_recommends_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _doctor_env(monkeypatch, _gpu(None), (False, "no NVIDIA GPU or driver found (nvidia-smi is missing or failed)"), tools=False)
    snap = yt.model_cache_path("large-v3", tmp_path) / "snapshots" / "x"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")
    monkeypatch.setenv("VIDEO_WATCH_MODEL_DIR", str(tmp_path))
    assert yt.main(["doctor"]) == yt.EXIT_OK
    keys = _doctor_lines(capsys)
    assert keys["GPU"] == ["none"] and keys["CUDA"][0].startswith("unavailable: no NVIDIA GPU")
    assert keys["RECOMMEND_DEVICE"] == ["cpu"] and keys["RECOMMEND_MODEL"] == ["small"]
    assert keys["RECOMMEND_SOUNDS"][0].startswith("no (it needs an NVIDIA GPU")
    assert keys["FFMPEG"] == ["missing"] and keys["UV"] == ["missing"] and keys["READY"] == ["no"]
    assert any("CPU mode is slow but works" in w for w in keys["WARN"])


def test_doctor_ready_on_cpu_and_survives_a_broken_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _doctor_env(monkeypatch, _gpu(None), (False, "macOS has no CUDA"))
    snap = yt.model_cache_path("small", tmp_path) / "snapshots" / "x"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")
    yt.save_config({"device": "cpu", "model": "small", "model_dir": str(tmp_path)})
    assert yt.main(["doctor"]) == yt.EXIT_OK
    keys = _doctor_lines(capsys)
    assert keys["DEVICE"] == ["cpu"] and keys["MODEL_CACHED"] == ["yes"] and keys["READY"] == ["yes"]
    _config_file().write_text("{broken", encoding="utf-8")
    assert yt.main(["doctor"]) == yt.EXIT_OK
    keys = _doctor_lines(capsys)
    assert keys["DEVICE"] == ["auto"] and any("not valid JSON" in w for w in keys["WARN"])


def _setup_env(monkeypatch: pytest.MonkeyPatch, gpu: yt.GpuInfo, cuda_ok: bool, heavy: bool = True) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(yt, "gpu_info", lambda: gpu)
    monkeypatch.setattr(yt, "cuda_status", lambda g: (cuda_ok, "ok" if cuda_ok else "no NVIDIA GPU"))
    monkeypatch.setattr(yt, "heavy_env_present", lambda: heavy)
    monkeypatch.setattr(yt, "disk_free_gb", lambda path: 500.0)
    monkeypatch.setattr(yt, "sound_model_cached", lambda: "prepare" in calls)
    monkeypatch.setattr(yt, "sound_helper_ready", lambda: "prepare" in calls)

    def download(name: str, root: Path) -> None:
        calls.append(f"download {name}")
        snap = yt.model_cache_path(name, root) / "snapshots" / "x"
        snap.mkdir(parents=True)
        (snap / "model.bin").write_bytes(b"x")

    monkeypatch.setattr(yt, "download_whisper_model", download)
    monkeypatch.setattr(yt, "prepare_sound_helper", lambda: calls.append("prepare"))
    return calls


def test_setup_dry_run_prints_sizes_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _setup_env(monkeypatch, _gpu(16311), True, heavy=False)
    monkeypatch.setattr(yt, "platform_key", lambda platform=None: "win32")
    argv = ["setup", "--device", "auto", "--model", "large-v3", "--sounds", "on", "--model-dir", str(tmp_path / "m"), "--dry-run"]
    assert yt.main(argv) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    downloads = [line for line in out if line.startswith("DOWNLOAD: ")]
    assert len(downloads) == 4 and "whisper model large-v3" in downloads[1] and downloads[1].endswith("about 3.1 GB")
    assert "DOWNLOAD_TOTAL_GB: 9.8" in out and out[-1] == "SETUP: dry-run"
    assert calls == [] and not _config_file().exists() and not (tmp_path / "m").exists()


def test_setup_needs_yes_for_downloads_then_writes_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _setup_env(monkeypatch, _gpu(16311), True)
    argv = ["setup", "--device", "cuda", "--model", "large-v3-turbo", "--sounds", "on", "--model-dir", str(tmp_path)]
    assert yt.main(argv) == yt.EXIT_ERROR
    out = capsys.readouterr().out
    assert "ERROR: setup needs to download about 6.9 GB" in out and "--yes" in out and calls == []
    assert not _config_file().exists()
    assert yt.main([*argv, "--yes"]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert calls == ["download large-v3-turbo", "prepare"] and out[-1] == "SETUP: done"
    assert "MODEL_CACHED: yes" in out and "SOUND_MODEL_CACHED: yes" in out
    saved = yt.json.loads(_config_file().read_text(encoding="utf-8"))
    assert saved == {"device": "cuda", "model": "large-v3-turbo", "sounds": True, "model_dir": str(tmp_path.absolute())}
    assert yt.main(["setup", "--sounds", "off"]) == yt.EXIT_OK  # nothing to download: no --yes needed
    assert capsys.readouterr().out.splitlines()[-1] == "SETUP: done"
    assert yt.load_config()["sounds"] is False and yt.load_config()["model"] == "large-v3-turbo"


def test_setup_refuses_sounds_with_an_old_driver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _setup_env(monkeypatch, _gpu(8188, driver="470.82"), False)
    argv = ["setup", "--device", "cpu", "--model", "small", "--sounds", "on", "--model-dir", str(tmp_path), "--yes"]
    assert yt.main(argv) == yt.EXIT_ERROR
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "ERROR: the sound map needs an NVIDIA GPU with CUDA 12: NVIDIA driver 470.82 is too old for CUDA 12 (needs 525 or newer)"
    assert out[1].startswith("HINT: update the NVIDIA driver to 525 or newer") and calls == []
    assert not _config_file().exists()
    assert yt.driver_status(_gpu(16311))[0] and not yt.driver_status(_gpu(None))[0]


def test_hints_are_copy_paste_ready() -> None:
    assert "setup --device cpu --model small --yes" in yt.NO_CUDA_HINT
    assert f'uv run "{Path(yt.__file__).resolve()}" setup --sounds on --yes' in yt.SOUNDS_SETUP_HINT


def test_setup_refuses_gpu_choices_without_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup_env(monkeypatch, _gpu(None), False)
    assert yt.main(["setup", "--device", "auto", "--model", "small", "--sounds", "off", "--dry-run"]) == yt.EXIT_ERROR
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("ERROR: CUDA is unavailable for device auto") and "setup --device cpu" in out[1]
    assert yt.main(["setup", "--device", "cpu", "--model", "small", "--sounds", "on", "--dry-run"]) == yt.EXIT_ERROR
    assert "the sound map needs an NVIDIA GPU" in capsys.readouterr().out
    argv = ["setup", "--device", "cpu", "--model", "small", "--sounds", "off", "--model-dir", str(tmp_path), "--dry-run"]
    assert yt.main(argv) == yt.EXIT_OK
    out = capsys.readouterr().out
    assert "WARN: CPU mode is several times slower" in out and "about 0.5 GB" in out


def test_cuda_reexec_env_on_linux_only() -> None:
    dirs = ["/env/nvidia/cublas/lib", "/env/nvidia/cudnn/lib"]
    env = yt.cuda_reexec_env({"LD_LIBRARY_PATH": "/usr/local/lib"}, dirs, "linux")
    assert env["LD_LIBRARY_PATH"] == "/env/nvidia/cublas/lib:/env/nvidia/cudnn/lib:/usr/local/lib"
    assert env[yt.REEXEC_ENV] == "/usr/local/lib"
    assert yt.cuda_reexec_env(env, dirs, "linux") is None  # already re-executed: never loops
    assert yt.cuda_reexec_env({"LD_LIBRARY_PATH": ":".join(dirs)}, dirs, "linux") is None
    assert yt.cuda_reexec_env({}, dirs, "win32") is None and yt.cuda_reexec_env({}, dirs, "darwin") is None
    assert yt.cuda_reexec_env({}, [], "linux") is None
    assert yt.cuda_reexec_env({}, dirs, "linux")[yt.REEXEC_ENV] == ""


def test_reexec_only_for_gpu_fetch_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(yt, "platform_key", lambda platform=None: "linux")
    monkeypatch.setattr(yt, "cuda_reexec_env", lambda: {"LD_LIBRARY_PATH": "/x"})
    monkeypatch.setattr(yt.os, "execve", lambda *a: calls.append(a))
    yt.maybe_reexec_for_cuda(argparse.Namespace(command="frames"))
    yt.maybe_reexec_for_cuda(argparse.Namespace(command="fetch", device="cpu"))
    assert calls == []
    yt.maybe_reexec_for_cuda(argparse.Namespace(command="fetch", device=None))
    assert len(calls) == 1 and calls[0][0] == sys.executable and calls[0][2] == {"LD_LIBRARY_PATH": "/x"}


def test_posix_helper_runs_in_its_own_session_and_the_group_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert yt.helper_popen_options("posix") == {"start_new_session": True}
    assert yt.helper_popen_options("nt") == {}
    killed: list = []
    monkeypatch.setattr(yt.os, "name", "posix")
    monkeypatch.setattr(yt.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)), raising=False)
    monkeypatch.setattr(yt.subprocess, "run", lambda *a, **k: pytest.fail("no taskkill on POSIX"))
    proc = _FakeProc()
    yt.kill_tree(proc)
    assert killed == [(4242, getattr(yt.signal, "SIGKILL", 9))] and proc.killed


def test_windows_kill_uses_taskkill(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list = []
    monkeypatch.setattr(yt.os, "name", "nt")
    monkeypatch.setattr(yt.subprocess, "run", lambda cmd, **k: runs.append(cmd))
    proc = _FakeProc()
    yt.kill_tree(proc, None)
    assert runs == [["taskkill", "/T", "/F", "/PID", "4242"]] and proc.killed


def test_font_discovery_per_os(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fonts = tmp_path / "Fonts"
    fonts.mkdir()
    (fonts / "segoeui.ttf").write_bytes(b"x")
    assert yt.find_label_font("win32", {"WINDIR": str(tmp_path)}) == fonts / "segoeui.ttf"
    assert yt.find_label_font("win32", {}) is None  # no WINDIR: ffmpeg's default font
    found = tmp_path / "DejaVuSans.ttf"
    found.write_bytes(b"x")
    assert yt.find_label_font("linux", {}, fc_match=lambda: str(found)) == found
    monkeypatch.setattr(yt, "LABEL_FONT_CANDIDATES", {"darwin": (str(tmp_path / "missing.ttf"),), "linux": ()})
    assert yt.find_label_font("darwin", {}) is None
    assert yt.find_label_font("linux", {}, fc_match=lambda: None) is None


def test_drawtext_without_a_font_uses_the_ffmpeg_default() -> None:
    vf = yt.drawtext_filter("00:05", None, 20)
    assert vf.startswith("drawtext=text='00\\:05'") and "fontfile" not in vf


def test_ffmpeg_install_hint_per_os() -> None:
    assert "winget" in yt.ffmpeg_install_hint("win32")
    assert "brew" in yt.ffmpeg_install_hint("darwin")
    assert "apt" in yt.ffmpeg_install_hint("linux")


# --- Phase 6 review fixes ---------------------------------------------------------------


def test_setup_counts_the_helper_environment_until_a_prepare_succeeded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _setup_env(monkeypatch, _gpu(16311), True)
    monkeypatch.setattr(yt, "sound_model_cached", lambda: True)  # the model is in a shared HF cache already
    snap = yt.model_cache_path("small", tmp_path) / "snapshots" / "x"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")
    argv = ["setup", "--device", "auto", "--model", "small", "--sounds", "on", "--model-dir", str(tmp_path)]
    assert yt.main([*argv, "--dry-run"]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert [line for line in out if line.startswith("DOWNLOAD: ")] == [
        "DOWNLOAD: sound map environment (PyTorch with CUDA 12.8, transformers) into the uv cache "
        "(less if uv already has it), about 3.0 GB"
    ]
    assert yt.main(argv) == yt.EXIT_ERROR and "setup needs to download about 3.0 GB" in capsys.readouterr().out
    assert calls == []
    assert yt.main([*argv, "--yes"]) == yt.EXIT_OK and calls == ["prepare"]
    capsys.readouterr()
    assert yt.main(argv) == yt.EXIT_OK and calls == ["prepare"]  # ready now: nothing to download or prepare
    assert "DOWNLOAD_TOTAL_GB: 0.0" in capsys.readouterr().out


def test_sound_helper_marker_tracks_the_helper_header(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(yt, "sound_helper_marker", lambda: tmp_path / "sound-helper.json")
    assert not yt.sound_helper_ready()
    yt.mark_sound_helper_ready()
    assert yt.sound_helper_ready()
    monkeypatch.setattr(yt, "sound_helper_fingerprint", lambda: "another header")
    assert not yt.sound_helper_ready()


def test_fetch_needs_the_prepared_helper_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yt.shutil, "which", lambda name: name)
    monkeypatch.setattr(yt, "sound_helper_ready", lambda: False)
    with pytest.raises(yt.SoundUnavailable, match="not set up yet") as exc:
        yt.check_gpu_free()
    assert exc.value.setup


def test_light_setup_delegates_without_printing_the_plan_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup_env(monkeypatch, _gpu(16311), True, heavy=False)
    runs: list = []
    monkeypatch.setattr(yt, "find_uv", lambda: "uv")
    monkeypatch.setattr(yt.subprocess, "run", lambda cmd, **k: runs.append(cmd) or argparse.Namespace(returncode=0))
    argv = ["setup", "--device", "auto", "--model", "small", "--sounds", "off", "--model-dir", str(tmp_path), "--yes"]
    assert yt.main(argv) == yt.EXIT_OK
    assert capsys.readouterr().out == ""  # the uv run --script child prints the plan
    assert runs == [["uv", "run", "--script", str(yt.SCRIPT_PATH), *argv]]


def test_sounds_off_note_does_not_suggest_deleting_a_shared_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup_env(monkeypatch, _gpu(16311), True)
    monkeypatch.setattr(yt, "sound_model_cached", lambda: True)
    snap = yt.model_cache_path("small", tmp_path) / "snapshots" / "x"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")
    assert yt.main(["setup", "--model", "small", "--sounds", "off", "--model-dir", str(tmp_path)]) == yt.EXIT_OK
    note = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("NOTE: "))
    assert "shared with other tools" in note and "delete" not in note


def test_hints_use_the_light_form_and_the_reference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    light = f'uv run --no-project --python ">=3.11" python "{yt.SCRIPT_PATH}"'
    assert yt.CONFIG_HINT == f"fix the value with: {light} config set KEY VALUE (or edit or delete the config file)"
    assert f"{light} config set work_dir PATH" in yt.WORK_DIR_HINT and f"{light} config unset work_dir" in yt.WORK_DIR_HINT
    for hint in (yt.ARGS_HINT, yt.SOUNDS_ARGS_HINT):
        assert yt.REFERENCE_URL in hint and "README" not in hint
    _config_file().write_text("{broken", encoding="utf-8")
    assert _run_main(monkeypatch, tmp_path, ["clean", "--all"]) == yt.EXIT_ERROR
    assert capsys.readouterr().out.splitlines()[1] == f"HINT: {yt.CONFIG_HINT}"


def test_an_invalid_env_var_hint_names_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VIDEO_WATCH_DEVICE", "vulkan")  # not an engine (yet)
    assert _run_main(monkeypatch, tmp_path, ["clean", "--all"]) == yt.EXIT_ERROR
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "HINT: fix or unset the environment variable VIDEO_WATCH_DEVICE; it overrides the config file"


def test_config_set_warns_when_an_env_var_overrides(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("VIDEO_WATCH_MODEL", "tiny")
    assert yt.main(["config", "set", "model", "small"]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "WARN: VIDEO_WATCH_MODEL=tiny is set and overrides this value; unset it to use the config file"
    monkeypatch.delenv("VIDEO_WATCH_MODEL")
    assert yt.main(["config", "set", "model", "small"]) == yt.EXIT_OK
    assert "WARN" not in capsys.readouterr().out


def test_doctor_warns_about_a_refused_work_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _doctor_env(monkeypatch, _gpu(16311), (True, "ok"))
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "notes.txt").write_text("user file")
    monkeypatch.setenv("VIDEO_WATCH_WORK_DIR", str(busy))
    assert yt.main(["doctor"]) == yt.EXIT_OK
    keys = _doctor_lines(capsys)
    assert any("already holds other files" in w for w in keys["WARN"]) and keys["READY"] == ["no"]
    assert keys["SOUND_HELPER_READY"] == ["yes"]
    assert yt.work_root_problem(tmp_path / "new") is None and not (tmp_path / "new").exists()


@pytest.mark.parametrize(
    "platform,machine,logical,expected",
    [("darwin", "arm64", 10, 10), ("darwin", "x86_64", 8, 4), ("linux", "x86_64", 12, 6), ("win32", "AMD64", 64, 16), ("linux", "aarch64", 1, 1)],
)
def test_cpu_threads(platform: str, machine: str, logical: int, expected: int) -> None:
    assert yt.cpu_threads(platform, machine, logical) == expected


@pytest.mark.parametrize(
    "value,hidden", [(None, False), ("0", False), ("1,0", False), ("GPU-1234abcd", False), ("", True), ("-1", True), ("none", True)]
)
def test_cuda_visible_devices_that_hide_the_gpu(value: str | None, hidden: bool) -> None:
    if value is None:
        assert yt.cuda_visible_devices_hides_gpu("") is not None  # set but empty hides it
        return
    assert bool(yt.cuda_visible_devices_hides_gpu(value)) is hidden


def test_hidden_gpu_is_no_usable_gpu_for_the_sound_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(yt.shutil, "which", lambda name: name)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(yt.SoundUnavailable, match="no usable GPU: CUDA_VISIBLE_DEVICES='-1' hides the GPU") as exc:
        yt.check_gpu_free()
    assert exc.value.no_gpu and not exc.value.gpu
    assert yt._sound_failure(exc.value).hint == yt.SOUNDS_NO_GPU_HINT
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(yt, "find_uv", lambda: "uv")
    response = {"ok": False, "kind": "error", "message": "RuntimeError: CUDA error: invalid device ordinal / Invalid device id"}
    monkeypatch.setattr(yt, "run_helper", _helper_writes(response))
    with pytest.raises(yt.SoundUnavailable, match="^no usable GPU \\(") as exc:
        yt.run_sound_backend(_pcm(tmp_path / "audio.48k.pcm"), ["music"], tmp_path)
    assert exc.value.no_gpu


# --- ffmpeg blocked by an agent sandbox ---------------------------------------------------


def _tool_file(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + (".exe" if os.name == "nt" else ""))
    path.write_bytes(b"x")
    return path


def test_blocked_executable_finds_a_tool_it_may_not_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tool = _tool_file(tmp_path / "bin", "ffmpeg")
    env = {"PATH": os.pathsep.join([str(tmp_path / "empty"), str(tmp_path / "bin")]), "PATHEXT": ".COM;.EXE"}
    monkeypatch.setattr(yt.os, "access", lambda path, mode: False)
    assert _REAL_BLOCKED("ffmpeg", env) == (str(tool), "not executable by this user")
    monkeypatch.setattr(yt.os, "access", lambda path, mode: True)
    assert _REAL_BLOCKED("ffmpeg", env) is None  # runnable: shutil.which would have found it
    assert _REAL_BLOCKED("ffprobe", env) is None  # absent


def test_blocked_executable_in_an_unreadable_folder(tmp_path: Path) -> None:
    def denied(path: str) -> os.stat_result:
        raise PermissionError(13, "Access is denied", path)

    named = str(tmp_path / "ffmpeg-9.0" / "bin")
    other = str(tmp_path / "private")
    found = _REAL_BLOCKED("ffmpeg", {"PATH": os.pathsep.join([other, named]), "PATHEXT": ".EXE"}, denied)
    assert found is not None and found[0].startswith(named) and found[1] == "Access is denied"
    assert _REAL_BLOCKED("ffmpeg", {"PATH": other, "PATHEXT": ".EXE"}, denied) is None  # may hold anything


def test_blocked_ffmpeg_is_reported_as_such(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(yt.shutil, "which", lambda name: None)
    blocked = str(tmp_path / "ffmpeg" / "bin" / "ffmpeg.exe")
    monkeypatch.setattr(yt, "blocked_executable", lambda name: (blocked, "Access is denied") if name == "ffmpeg" else None)
    assert _run_main(monkeypatch, tmp_path / "work", ["frames", VID, "--at", "0:01"]) == yt.EXIT_ERROR
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"ERROR: ffmpeg was found but cannot be run by this process ({blocked}: Access is denied)"
    assert out[1] == f"HINT: {yt.SANDBOX_HINT}" and out[0].startswith("ERROR: ffmpeg")
    assert "codex exec -s danger-full-access" in yt.SANDBOX_HINT
    monkeypatch.setattr(yt, "blocked_executable", lambda name: None)
    with pytest.raises(yt.FetchError, match="^ffmpeg was not found on PATH$"):
        yt.find_ffmpeg()
    monkeypatch.setattr(yt, "blocked_executable", lambda name: (blocked, "denied") if name == "ffprobe" else None)
    with pytest.raises(yt.FetchError, match="^ffmpeg was found but cannot be run by this process"):
        yt.find_ffprobe(str(tmp_path / "nowhere" / "ffmpeg"))


def test_doctor_shows_a_blocked_ffmpeg(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _doctor_env(monkeypatch, _gpu(16311), (True, "ok"), tools=False)
    monkeypatch.setattr(yt, "blocked_executable", lambda name: (f"/opt/ffmpeg/{name}", "Permission denied"))
    assert yt.main(["doctor"]) == yt.EXIT_OK
    keys = _doctor_lines(capsys)
    assert keys["FFMPEG"] == ["blocked: /opt/ffmpeg/ffmpeg (Permission denied)"]
    assert keys["FFPROBE"] == ["blocked: /opt/ffmpeg/ffprobe (Permission denied)"]
    assert any(w.startswith("ffmpeg was found but cannot be run by this process; the agent's sandbox") for w in keys["WARN"])


# --- Apple GPU engine (mlx-whisper) --------------------------------------------------------------
# No Mac, no MLX and no numpy needed: the engine's modules are faked.


def _apple(monkeypatch: pytest.MonkeyPatch, on: bool = True) -> None:
    monkeypatch.setattr(yt, "apple_silicon", lambda platform=None, machine=None: on)


def _mlx_snapshot(root: Path, name: str, weights: str = "weights.safetensors") -> Path:
    snap = yt.model_cache_path(name, root, "metal") / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / weights).write_bytes(b"x")
    return snap


class _FakeHolder:
    loaded: list[str] = []
    model: object | None = None
    model_path: str | None = None

    @classmethod
    def get_model(cls, path: str, dtype: object) -> object:
        if cls.model is None or cls.model_path != path:
            cls.loaded.append(path)
            cls.model, cls.model_path = object(), path
        return cls.model


def _fake_mlx(monkeypatch: pytest.MonkeyPatch, metal: bool = True, transcribe: object = None) -> argparse.Namespace:
    """Fake `mlx.core`, `mlx_whisper` and `mlx_whisper.transcribe` (with a tqdm hook) in sys.modules."""
    import types

    state = argparse.Namespace(holder=_FakeHolder, cleared=[], calls=[], real_bar=lambda *a, **k: None)
    _FakeHolder.loaded, _FakeHolder.model, _FakeHolder.model_path = [], None, None
    core = types.ModuleType("mlx.core")
    core.float16 = "float16"
    core.metal = argparse.Namespace(is_available=lambda: metal)
    core.clear_cache = lambda: state.cleared.append(1)
    mlx = types.ModuleType("mlx")
    mlx.core = core
    module = types.ModuleType("mlx_whisper.transcribe")
    module.ModelHolder = _FakeHolder
    module.tqdm = argparse.Namespace(tqdm=state.real_bar)  # mlx-whisper 0.4 calls `tqdm.tqdm(...)`
    package = types.ModuleType("mlx_whisper")

    def run(samples: object, **kw: object) -> dict:
        state.calls.append((len(samples), kw, getattr(getattr(module, "tqdm", None), "tqdm", None)))
        return transcribe(samples, **kw) if transcribe else {"segments": [], "text": "", "language": kw["language"]}

    package.transcribe = run
    for name, mod in (("mlx", mlx), ("mlx.core", core), ("mlx_whisper", package), ("mlx_whisper.transcribe", module)):
        monkeypatch.setitem(sys.modules, name, mod)
    return state


def test_device_auto_means_the_apple_gpu_only_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    assert yt.effective_device("auto") == "cuda" and yt.effective_device("cuda") == "cuda"
    _apple(monkeypatch)
    assert yt.effective_device("auto") == "metal" and yt.effective_device("metal") == "metal"
    assert yt.effective_device("cpu") == "cpu" and yt.effective_device("cuda") == "cuda"  # explicit choices are kept
    assert yt.resolve_device(argparse.Namespace(device=None)) == "metal"
    assert yt.resolve_device(argparse.Namespace(device="cpu")) == "cpu"  # CPU mode stays the user's choice
    assert yt.parse_config_value("device", "Metal") == "metal"


def test_apple_silicon_detection_by_platform_and_machine() -> None:
    assert _REAL_APPLE_SILICON("darwin", "arm64") and _REAL_APPLE_SILICON("darwin", "ARM64")
    assert not _REAL_APPLE_SILICON("darwin", "x86_64")  # an Intel Mac, or Python under Rosetta
    assert not _REAL_APPLE_SILICON("linux", "aarch64") and not _REAL_APPLE_SILICON("win32", "arm64")


def test_mlx_model_cache_layout_and_sizes(tmp_path: Path) -> None:
    assert not yt.model_cached("small", tmp_path, "metal")
    assert yt.model_cache_path("small", tmp_path, "metal").name == "models--mlx-community--whisper-small-mlx"
    assert yt.model_cache_path("small", tmp_path).name == "models--Systran--faster-whisper-small"
    snap = yt.model_cache_path("small", tmp_path, "metal") / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "weights.npz").write_bytes(b"x")
    assert not yt.model_cached("small", tmp_path, "metal")  # no config.json yet
    (snap / "config.json").write_text("{}")
    assert yt.model_cached("small", tmp_path, "metal") and yt.mlx_snapshot("small", tmp_path) == snap
    assert not yt.model_cached("small", tmp_path)  # the CTranslate2 conversion is separate
    assert yt.model_size_mb("small", "metal") == 481 and yt.model_size_mb("small", "cuda") == 485
    _mlx_snapshot(tmp_path, "large-v3-turbo")
    assert yt.model_cached("large-v3-turbo", tmp_path, "metal")
    assert all(m["mlx_repo"].startswith("mlx-community/whisper-") for m in yt.WHISPER_MODELS.values())


def test_download_mlx_model_fetches_config_and_weights_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict = {}

    def snapshot_download(repo: str, **kw: object) -> str:
        seen.update(kw, repo=repo)
        return str(tmp_path / "snap")

    monkeypatch.setitem(sys.modules, "huggingface_hub", argparse.Namespace(snapshot_download=snapshot_download))
    assert yt.download_mlx_model("small", tmp_path / "m") == tmp_path / "snap"
    assert seen["repo"] == "mlx-community/whisper-small-mlx" and seen["cache_dir"] == str(tmp_path / "m")
    assert seen["allow_patterns"] == ["config.json", "weights.safetensors", "weights.npz"]


def test_metal_status_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    ok, why = yt.metal_status()
    assert not ok and "macOS on Apple Silicon" in why
    _apple(monkeypatch)
    ok, why = yt.metal_status(deep=False)
    assert ok and "mlx-whisper is checked on the first run" in why
    monkeypatch.setattr(yt.importlib.metadata, "version", lambda name: "0.4.3")
    _fake_mlx(monkeypatch, metal=False)
    assert yt.metal_status(deep=True) == (False, "MLX sees no Metal GPU")
    _fake_mlx(monkeypatch)
    assert yt.metal_status(deep=True) == (True, "ok (mlx-whisper 0.4.3, Metal)")
    monkeypatch.setitem(sys.modules, "mlx.core", None)  # import fails
    ok, why = yt.metal_status(deep=True)
    assert not ok and why.startswith("MLX could not start")


def test_load_metal_loads_the_cached_snapshot_without_downloading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _apple(monkeypatch)
    state = _fake_mlx(monkeypatch)
    snap = _mlx_snapshot(tmp_path, "small")
    monkeypatch.setattr(yt, "download_mlx_model", lambda *a: pytest.fail("cached: no download"))
    model = yt.load_whisper("small", "metal", tmp_path)
    assert model == yt.MetalModel("small", str(snap)) and state.holder.loaded == [str(snap)]


def test_load_metal_downloads_a_missing_model_into_the_model_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _apple(monkeypatch)
    state = _fake_mlx(monkeypatch)
    snap = tmp_path / "fresh"
    monkeypatch.setattr(yt, "download_mlx_model", lambda name, root: snap)
    assert yt.load_whisper("small", "metal", tmp_path).path == str(snap)
    assert state.holder.loaded == [str(snap)]


def test_load_metal_failures_are_gpu_unavailable_never_a_cpu_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(yt.GpuUnavailable, match="macOS on Apple Silicon") as exc:  # not a Mac with an Apple chip
        yt.load_whisper("small", "metal", tmp_path)
    assert exc.value.permanent and exc.value.hint == yt.METAL_NOT_MAC_HINT and "--device auto" in yt.METAL_NOT_MAC_HINT
    _apple(monkeypatch)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    with pytest.raises(yt.GpuUnavailable, match="mlx-whisper is not installed") as exc:
        yt.load_whisper("small", "metal", tmp_path)
    assert exc.value.permanent and exc.value.hint == yt.METAL_HINT and "--device cpu" in yt.METAL_HINT
    _fake_mlx(monkeypatch, metal=False)
    with pytest.raises(yt.GpuUnavailable, match="no Metal GPU") as exc:
        yt.load_whisper("small", "metal", tmp_path)
    assert exc.value.permanent
    state = _fake_mlx(monkeypatch)
    _mlx_snapshot(tmp_path, "small")

    def boom(path: str, dtype: object) -> None:
        raise RuntimeError("[metal::malloc] Attempting to allocate too much memory")

    monkeypatch.setattr(state.holder, "get_model", boom)
    with pytest.raises(yt.GpuUnavailable, match="too much memory") as exc:  # a busy or full GPU: not permanent
        yt.load_whisper("small", "metal", tmp_path)
    assert not exc.value.permanent and exc.value.hint == yt.GPU_RETRY_HINT

    def corrupt(path: str, dtype: object) -> None:
        raise RuntimeError("[load_safetensors] Invalid json header length")

    monkeypatch.setattr(state.holder, "get_model", corrupt)
    with pytest.raises(RuntimeError, match="json header"):  # not a GPU problem: passes through, as on CUDA
        yt.load_whisper("small", "metal", tmp_path)


def test_load_metal_download_failure_is_a_fetch_error_not_a_gpu_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _apple(monkeypatch)
    _fake_mlx(monkeypatch)

    def offline(name: str, root: Path) -> Path:
        raise OSError("Cannot reach https://huggingface.co")

    monkeypatch.setattr(yt, "download_mlx_model", offline)
    with pytest.raises(yt.FetchError, match="the Whisper model download failed") as exc:
        yt.load_whisper("small", "metal", tmp_path)
    assert exc.value.code == yt.EXIT_ERROR and exc.value.hint == yt.MODEL_DOWNLOAD_HINT


def test_a_failed_mlx_download_never_falls_back_to_auto_captions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    downloads = _fake_youtube(monkeypatch, YT_INFO | {"subtitles": {}, "automatic_captions": {"en-orig": FMT}})
    _apple(monkeypatch)
    _fake_mlx(monkeypatch)
    monkeypatch.setenv("VIDEO_WATCH_MODEL_DIR", str(tmp_path / "models"))

    def offline(name: str, root: Path) -> Path:
        raise OSError("Cannot reach https://huggingface.co")

    monkeypatch.setattr(yt, "download_mlx_model", offline)
    assert _run_main(monkeypatch, tmp_path / "work", ["fetch", VID]) == yt.EXIT_ERROR
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("ERROR: the Whisper model download failed (OSError: Cannot reach")
    assert out[1] == f"HINT: {yt.MODEL_DOWNLOAD_HINT}" and not any(l.startswith("SOURCE:") for l in out)
    assert downloads == []  # stopped before the audio download


def test_a_metal_run_on_another_machine_stops_before_the_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(yt, "load_whisper", lambda *a, **k: pytest.fail("never loads a model"))
    args = argparse.Namespace(model="small", device="metal", no_vad=False, lang=None, cookies_from_browser=None)
    with pytest.raises(yt.GpuUnavailable) as exc:
        yt.run_whisper(_FakeWhisperYDL(tmp_path), {"duration": 10}, tmp_path, args, None)
    assert exc.value.permanent and exc.value.hint == yt.METAL_NOT_MAC_HINT
    assert "ESTIMATE" not in capsys.readouterr().err


def test_release_whisper_frees_the_metal_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = _fake_mlx(monkeypatch)
    model = yt.MetalModel("small", str(tmp_path))
    state.holder.get_model(model.path, None)
    yt.release_whisper(model)
    assert state.holder.model is None and state.cleared == [1]
    yt.release_whisper(object())  # the other engines are released by dropping the reference
    assert state.cleared == [1]


def test_speech_timeline_maps_joined_speech_back_to_the_original_audio() -> None:
    timeline = yt.SpeechTimeline([(1.0, 3.0), (6.0, 8.0), (20.0, 21.0)])  # joined: 0-2, 2-4, 4-5
    assert timeline.total == 5.0
    assert timeline.original(0.0) == 1.0 and timeline.original(1.5) == 2.5
    assert timeline.original(2.0) == 6.0  # a start on the boundary belongs to the later chunk
    assert timeline.original(2.0, is_end=True) == 3.0  # an end on the boundary to the earlier one
    assert timeline.original(3.5) == 7.5 and timeline.original(5.0, is_end=True) == 21.0
    assert timeline.original(9.0, is_end=True) == 21.0 and yt.SpeechTimeline([]).original(1.5) == 1.5
    assert timeline.original(9.0) == 21.0  # a start past the speech is clamped like an end, never after it
    assert timeline.original(1.99) == pytest.approx(2.99) and timeline.original(-0.5) == 1.0


def test_segments_from_mlx_keep_word_timestamps_and_restore_times() -> None:
    raw = [
        {"start": 0.5, "end": 1.5, "text": " Merhaba dünya", "words": [
            {"word": " Merhaba", "start": 0.5, "end": 1.0, "probability": 0.91234},
            {"word": " dünya", "start": 1.0, "end": 1.5, "probability": 0.8},
        ]},
        {"start": 1.5, "end": 1.9, "text": "  ", "words": []},  # blank: dropped
        {"start": 2.5, "end": 3.5, "text": "bir şey"},
    ]
    plain = yt.segments_from_mlx(raw)
    assert [(s.start, s.end, s.text) for s in plain] == [(0.5, 1.5, "Merhaba dünya"), (2.5, 3.5, "bir şey")]
    assert plain[0].words == [
        {"start": 0.5, "end": 1.0, "word": " Merhaba", "p": 0.912}, {"start": 1.0, "end": 1.5, "word": " dünya", "p": 0.8}
    ] and plain[1].words is None
    mapped = yt.segments_from_mlx(raw, yt.SpeechTimeline([(10.0, 12.0), (30.0, 32.0)]))
    assert (mapped[0].start, mapped[0].end) == (10.5, 11.5) and mapped[0].words[1]["end"] == 11.5
    assert (mapped[1].start, mapped[1].end) == (30.5, 31.5)  # 2.5 s of speech is 0.5 s into the second chunk


def _fake_audio_stack(monkeypatch: pytest.MonkeyPatch, seconds: int, spans: list[tuple[int, int]]) -> argparse.Namespace:
    """faster_whisper.audio / .vad and numpy faked with plain lists (16 kHz samples)."""
    import types

    seen = argparse.Namespace(vad_calls=0)
    audio = types.ModuleType("faster_whisper.audio")
    audio.decode_audio = lambda path, sampling_rate=16000: [0.0] * (seconds * sampling_rate)
    vad = types.ModuleType("faster_whisper.vad")
    vad.VadOptions = lambda **kw: argparse.Namespace(**kw)

    def speech(samples: object, options: object) -> list[dict]:
        seen.vad_calls += 1
        seen.options = options
        return [{"start": a * 16000, "end": b * 16000} for a, b in spans]

    vad.get_speech_timestamps = speech
    for name, mod in (("faster_whisper", types.ModuleType("faster_whisper")), ("faster_whisper.audio", audio), ("faster_whisper.vad", vad)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, "numpy", argparse.Namespace(concatenate=lambda parts: [x for part in parts for x in part]))
    return seen


def test_transcribe_metal_with_vad_decodes_only_the_speech_and_maps_the_times_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _fake_mlx(monkeypatch, transcribe=lambda samples, **kw: {"language": kw["language"], "segments": [
        {"start": 0.5, "end": 1.5, "text": "bir", "words": [{"word": "bir", "start": 0.5, "end": 1.5, "probability": 0.9}]},
        {"start": 2.5, "end": 3.5, "text": "iki", "words": [{"word": "iki", "start": 2.5, "end": 3.5, "probability": 0.9}]},
    ]})
    seen = _fake_audio_stack(monkeypatch, 10, [(1, 3), (6, 8)])
    monkeypatch.setattr(yt, "detect_language_mlx", lambda model, samples: ("tr", 0.97) if len(samples) == 64000 else (None, None))
    model = yt.MetalModel("small", str(tmp_path))
    segments, info, mode = yt.transcribe(model, tmp_path / "audio.webm", None, True, 10)
    assert seen.options.min_silence_duration_ms == yt.VAD_MIN_SILENCE_MS
    length, kw, bar = state.calls[0]
    assert length == 64000  # 4 s of speech out of 10 s
    assert kw["language"] == "tr" and kw["word_timestamps"] is True and kw["condition_on_previous_text"] is False
    assert kw["verbose"] is False  # the bar, which the hook turns into stderr lines
    assert bar is yt._ProgressLog and sys.modules["mlx_whisper.transcribe"].tqdm.tqdm is state.real_bar  # hooked only during the call
    assert [(s.start, s.end, s.text) for s in segments] == [(1.5, 2.5, "bir"), (6.5, 7.5, "iki")]
    assert segments[1].words[0]["start"] == 6.5
    assert (info.language, info.language_probability, info.duration) == ("tr", 0.97, 10.0) and mode == {"batched": False}


def test_transcribe_metal_without_speech_returns_nothing_so_the_no_vad_retry_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _fake_mlx(monkeypatch)
    _fake_audio_stack(monkeypatch, 10, [])
    segments, info, _ = yt.transcribe(yt.MetalModel("small", str(tmp_path)), tmp_path / "a.webm", "en", True, 10)
    assert segments == [] and state.calls == [] and info.language == "en"  # no mlx call at all on an empty VAD result


def test_transcribe_metal_without_vad_decodes_the_whole_audio_and_keeps_a_given_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _fake_mlx(monkeypatch, transcribe=lambda samples, **kw: {"language": "en", "segments": [
        {"start": 1.0, "end": 2.0, "text": "la la", "words": [{"word": "la", "start": 1.0, "end": 1.5, "probability": 0.5}]}]})
    seen = _fake_audio_stack(monkeypatch, 6, [(1, 3)])
    monkeypatch.setattr(yt, "detect_language_mlx", lambda *a: pytest.fail("the language is given"))
    segments, info, _ = yt.transcribe(yt.MetalModel("small", str(tmp_path)), tmp_path / "a.webm", "en", False, 6)
    assert seen.vad_calls == 0 and state.calls[0][0] == 6 * 16000
    assert segments[0].words[0]["word"] == "la" and (info.language, info.language_probability) == ("en", None)


def test_transcribe_metal_without_the_tqdm_hook_runs_without_a_bar_and_keeps_stdout_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def chatty(samples: object, **kw: object) -> dict:
        print("Detected language: English")  # mlx-whisper prints this when it detects the language itself
        return {"language": "en", "segments": []}

    state = _fake_mlx(monkeypatch, transcribe=chatty)
    monkeypatch.delattr(sys.modules["mlx_whisper.transcribe"], "tqdm")  # a later mlx-whisper without `tqdm.tqdm`
    _fake_audio_stack(monkeypatch, 4, [])
    monkeypatch.setattr(yt, "detect_language_mlx", lambda *a: (None, None))
    segments, info, _ = yt.transcribe(yt.MetalModel("small", str(tmp_path)), tmp_path / "a.webm", None, False, 4)
    assert segments == [] and info.language == "en"
    assert state.calls[0][1]["verbose"] is None and state.calls[0][2] is None  # no bar instead of a crash
    captured = capsys.readouterr()
    assert captured.out == "" and "Detected language: English" in captured.err


def test_detect_language_mlx_returns_the_probability_and_survives_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    audio = types.ModuleType("mlx_whisper.audio")
    audio.N_SAMPLES, audio.N_FRAMES = 480000, 3000
    audio.log_mel_spectrogram = lambda samples, n_mels, padding: argparse.Namespace(astype=lambda t: "mel")
    audio.pad_or_trim = lambda mel, n, axis: argparse.Namespace(astype=lambda dtype: "mel")
    _fake_mlx(monkeypatch)
    monkeypatch.setitem(sys.modules, "mlx_whisper.audio", audio)
    model = argparse.Namespace(dims=argparse.Namespace(n_mels=128), detect_language=lambda mel: (None, {"en": 0.1, "tr": 0.9}))
    assert yt.detect_language_mlx(model, [0.0]) == ("tr", 0.9)
    assert yt.detect_language_mlx(argparse.Namespace(), [0.0]) == (None, None)  # mlx-whisper then detects it itself


def test_progress_log_reports_every_ten_percent(capsys: pytest.CaptureFixture[str]) -> None:
    with yt._ProgressLog(total=6000, unit="frames") as bar:  # 60 s of audio in 10 ms mel frames
        for _ in range(60):
            bar.update(100)
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "transcribe: 10% (00:06 / 01:00)" and lines[-1] == "transcribe: 100% (01:00 / 01:00)" and len(lines) == 10


def test_metal_errors_become_gpu_unavailable_but_other_errors_pass_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(message: object):
        def run(*a: object, **k: object) -> None:
            raise message if isinstance(message, BaseException) else RuntimeError(message)
        return run

    monkeypatch.setattr(yt, "transcribe", fail("Command buffer execution failed: Insufficient Memory"))
    with pytest.raises(yt.GpuUnavailable, match="Insufficient Memory") as exc:
        yt._transcribe_on_device(object(), tmp_path, None, True, 10, 1, "metal")
    assert not exc.value.no_cuda and not exc.value.permanent
    monkeypatch.setattr(yt, "transcribe", fail(MemoryError()))
    with pytest.raises(yt.GpuUnavailable):
        yt._transcribe_on_device(object(), tmp_path, None, True, 10, 1, "metal")
    monkeypatch.setattr(yt, "transcribe", fail("tensor shape mismatch"))
    with pytest.raises(RuntimeError, match="shape mismatch"):  # a bug, not a GPU problem
        yt._transcribe_on_device(object(), tmp_path, None, True, 10, 1, "metal")


def test_metal_run_is_labelled_and_estimated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict = {}
    model = yt.MetalModel("small", str(tmp_path))

    def fake_transcribe(m: object, audio: Path, language: object, vad: bool, dur: object, batch: int = 8) -> tuple:
        seen["batch"], seen["model"] = batch, m
        info = yt.TranscribeInfo("tr", None, 10)  # no language probability from the engine
        return _segs([(0, 5, "bir iki üç dört beş altı yedi")]), info, {"batched": False}

    _apple(monkeypatch)
    monkeypatch.setattr(yt, "metal_status", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(yt, "load_whisper", lambda name, device="cuda", root=None: seen.update(device=device) or model)
    monkeypatch.setattr(yt, "whisper_language", lambda code: code)
    monkeypatch.setattr(yt, "transcribe", fake_transcribe)
    released: list[object] = []
    monkeypatch.setattr(yt, "release_whisper", released.append)
    args = argparse.Namespace(model="small", device="metal", no_vad=False, lang=None, cookies_from_browser=None)
    result = yt.run_whisper(_FakeWhisperYDL(tmp_path), {"duration": 10}, tmp_path, args, None)
    assert seen["device"] == "metal" and seen["batch"] == 1 and released == [model]
    assert result.settings["device"] == "metal" and result.settings["beam_size"] == 1 and result.settings["compute_type"] == "float16"
    assert "language auto-detected as 'tr'" in result.notes[0] and "p=" not in result.notes[0]
    assert any("Apple GPU (mlx-whisper, float16, greedy" in n for n in result.notes)
    assert "on the Apple GPU" in capsys.readouterr().err
    head = "\n".join(yt.header_lines(_meta(source="whisper", settings=result.settings | {"vad_min_silence_ms": 500})))
    assert "whisper (small, float16, metal, VAD on (min silence 500 ms), beam 1, sequential)" in head


def test_metal_estimate_uses_the_measured_speed() -> None:
    base = yt.whisper_estimate_min(600, model="large-v3-turbo", device="metal")
    assert base == pytest.approx(yt.ESTIMATE_OVERHEAD_MIN + 0.31)
    assert yt.whisper_estimate_min(600, model="large-v3-turbo", device="metal", no_vad=True) == base  # same decoder either way
    assert yt.whisper_estimate_min(600, model="large-v3", device="metal") > base > yt.whisper_estimate_min(600, model="small", device="metal")
    assert base < yt.whisper_estimate_min(600, model="large-v3-turbo", device="cpu")


@pytest.mark.parametrize("memory_mb,model", [(36864, "large-v3"), (16384, "large-v3"), (8192, "large-v3-turbo"), (4096, "small")])
def test_recommend_on_apple_silicon_by_unified_memory(memory_mb: int, model: str) -> None:
    assert yt.recommend(_gpu(None), False, apple_mb=memory_mb)[:3] == (model, "metal", False)
    assert yt.recommend(_gpu(None), False)[:3] == ("small", "cpu", False)  # no Apple GPU: unchanged
    assert yt.recommend(_gpu(16311), True, apple_mb=8192)[:3] == ("large-v3", "cuda", True)  # CUDA wins


def test_metal_fit_warning() -> None:
    assert yt.metal_fit_warning("large-v3", 24576) is None and yt.metal_fit_warning("large-v3", None) is None
    assert "may run out of memory" in yt.metal_fit_warning("large-v3", 6144)
    assert yt.metal_fit_warning("large-v3-turbo", 8192) is None


def test_cuda_status_on_a_mac_points_to_the_apple_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    assert yt.cuda_status(_gpu(None), "darwin") == (False, "macOS has no CUDA (CPU mode only)")
    _apple(monkeypatch)
    assert yt.cuda_status(_gpu(None), "darwin") == (False, "macOS has no CUDA (Whisper runs on the Apple GPU instead)")


def _doctor_apple(monkeypatch: pytest.MonkeyPatch, memory_mb: int = 24576, metal: tuple = (True, "ok (mlx-whisper 0.4.3, Metal)")) -> None:
    _doctor_env(monkeypatch, _gpu(None), (False, "macOS has no CUDA (Whisper runs on the Apple GPU instead)"))
    _apple(monkeypatch)
    monkeypatch.setattr(yt, "apple_gpu_info", lambda: yt.GpuInfo("Apple M9 Max", memory_mb, None, None))
    monkeypatch.setattr(yt, "metal_status", lambda *a, **k: metal)


def test_doctor_on_apple_silicon_reports_the_engine_and_is_ready_after_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _doctor_apple(monkeypatch)
    monkeypatch.setenv("VIDEO_WATCH_MODEL_DIR", str(tmp_path))
    assert yt.main(["doctor"]) == yt.EXIT_OK
    keys = _doctor_lines(capsys)
    assert keys["GPU"] == ["Apple M9 Max (Apple GPU, unified memory)"] and keys["VRAM_TOTAL_MB"] == ["24576"]
    assert keys["CUDA"][0].startswith("unavailable: macOS has no CUDA") and keys["METAL"] == ["ok (mlx-whisper 0.4.3, Metal)"]
    assert keys["DEVICE"] == ["auto"] and keys["ENGINE"] == ["mlx-whisper (Apple GPU, float16)"]
    assert keys["RECOMMEND_MODEL"] == ["large-v3"] and keys["RECOMMEND_DEVICE"] == ["metal"]
    assert keys["MODEL_CACHED"] == ["no"] and keys["READY"] == ["no"]
    assert any("not downloaded yet (about 3.1 GB" in w for w in keys["WARN"])
    _mlx_snapshot(tmp_path, "large-v3", "weights.npz")
    assert yt.main(["doctor"]) == yt.EXIT_OK
    keys = _doctor_lines(capsys)
    assert keys["MODEL_CACHED"] == ["yes"] and keys["READY"] == ["yes"]
    assert yt.main(["doctor"]) == yt.EXIT_OK and "WARN" not in _doctor_lines(capsys)


def test_doctor_cpu_mode_on_apple_silicon_needs_the_ct2_model_and_suggests_the_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _doctor_apple(monkeypatch)
    monkeypatch.setenv("VIDEO_WATCH_MODEL_DIR", str(tmp_path))
    _mlx_snapshot(tmp_path, "small")  # the MLX weights do not count for CPU mode
    yt.save_config({"device": "cpu", "model": "small"})
    assert yt.main(["doctor"]) == yt.EXIT_OK
    keys = _doctor_lines(capsys)
    assert keys["ENGINE"] == ["faster-whisper (CPU, int8)"] and keys["MODEL_CACHED"] == ["no"] and keys["READY"] == ["no"]
    assert any("several times slower than on the Apple GPU (setup --device metal)" in w for w in keys["WARN"])


def test_doctor_warns_when_the_apple_gpu_engine_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _doctor_apple(monkeypatch, metal=(False, "MLX sees no Metal GPU"))
    monkeypatch.setenv("VIDEO_WATCH_MODEL_DIR", str(tmp_path))
    _mlx_snapshot(tmp_path, "large-v3")
    assert yt.main(["doctor"]) == yt.EXIT_OK
    keys = _doctor_lines(capsys)
    assert keys["METAL"] == ["unavailable: MLX sees no Metal GPU"] and keys["READY"] == ["no"]
    assert keys["RECOMMEND_DEVICE"] == ["cpu"]
    assert any("the Apple GPU engine is unavailable (MLX sees no Metal GPU)" in w for w in keys["WARN"])


def test_doctor_on_other_machines_says_metal_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _doctor_env(monkeypatch, _gpu(16311), (True, "ok (driver 580.10)"))
    assert yt.main(["doctor"]) == yt.EXIT_OK
    keys = _doctor_lines(capsys)
    assert keys["METAL"][0].startswith("unavailable: needs macOS on Apple Silicon")
    assert keys["ENGINE"] == ["faster-whisper (CUDA, float16)"] and keys["RECOMMEND_DEVICE"] == ["cuda"]


def test_setup_for_the_apple_gpu_downloads_the_mlx_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _setup_env(monkeypatch, _gpu(None), False)
    _apple(monkeypatch)
    monkeypatch.setattr(yt, "metal_status", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(yt, "apple_gpu_info", lambda: yt.GpuInfo("Apple M9 Max", 24576, None, None))
    monkeypatch.setattr(yt, "download_whisper_model", lambda *a: pytest.fail("the CTranslate2 model is for CUDA and CPU mode"))
    monkeypatch.setattr(yt, "download_mlx_model", lambda name, root: calls.append(f"mlx {name}") or _mlx_snapshot(root, name))
    argv = ["setup", "--device", "auto", "--model", "large-v3-turbo", "--sounds", "off", "--model-dir", str(tmp_path)]
    assert yt.main([*argv, "--dry-run"]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert any(line.startswith("DOWNLOAD: whisper model large-v3-turbo (MLX weights for the Apple GPU)") and line.endswith("about 1.6 GB") for line in out)
    assert "DOWNLOAD_TOTAL_GB: 1.6" in out and calls == []
    assert yt.main(argv) == yt.EXIT_ERROR and "setup needs to download about 1.6 GB" in capsys.readouterr().out
    assert yt.main([*argv, "--yes"]) == yt.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert calls == ["mlx large-v3-turbo"] and "MODEL_CACHED: yes" in out and out[-1] == "SETUP: done"
    assert yt.load_config()["device"] == "auto" and yt.model_cached("large-v3-turbo", tmp_path, "metal")
    assert yt.main([*argv, "--yes"]) == yt.EXIT_OK  # cached now: nothing to download
    assert calls == ["mlx large-v3-turbo"] and not any(l.startswith("DOWNLOAD: ") for l in capsys.readouterr().out.splitlines())


def test_setup_stops_when_the_apple_gpu_engine_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup_env(monkeypatch, _gpu(None), False)
    monkeypatch.setattr(yt, "metal_status", lambda *a, **k: (False, "needs macOS on Apple Silicon (an arm64 Mac)"))
    assert yt.main(["setup", "--device", "metal", "--model", "small", "--model-dir", str(tmp_path), "--yes"]) == yt.EXIT_ERROR
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "ERROR: the Apple GPU engine is unavailable for device metal: needs macOS on Apple Silicon (an arm64 Mac)"
    assert "setup --device cpu --model small" in out[1] and not _config_file().exists()


def test_setup_cuda_on_apple_silicon_points_to_the_apple_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup_env(monkeypatch, _gpu(None), False)
    _apple(monkeypatch)
    assert yt.main(["setup", "--device", "cuda", "--model", "small", "--model-dir", str(tmp_path), "--yes"]) == yt.EXIT_ERROR
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("ERROR: CUDA is unavailable for device cuda") and "--device metal" in out[1]


def test_heavy_env_on_apple_silicon_includes_mlx_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    present = {"faster_whisper", "ctranslate2", "yt_dlp"}
    monkeypatch.setattr(yt.importlib.util, "find_spec", lambda name: object() if name in present else None)
    assert yt.heavy_env_present()
    _apple(monkeypatch)
    assert not yt.heavy_env_present()
    present.add("mlx_whisper")
    assert yt.heavy_env_present()


def test_a_permanent_apple_gpu_failure_never_falls_back_to_auto_captions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_youtube(monkeypatch, YT_INFO | {"subtitles": {}, "automatic_captions": {"en-orig": FMT}})

    def no_metal(*a: object, **k: object) -> None:
        raise yt.GpuUnavailable("the Apple GPU engine needs macOS on Apple Silicon (an arm64 Mac)", permanent=True)

    monkeypatch.setattr(yt, "run_whisper", no_metal)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_NO_TRANSCRIPT
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "ERROR: Whisper could not run on the GPU (the Apple GPU engine needs macOS on Apple Silicon (an arm64 Mac))"
    assert out[1] == f"HINT: {yt.METAL_HINT}" and "--device cpu" in out[1]


def test_a_busy_apple_gpu_still_uses_the_youtube_auto_captions_like_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_youtube(monkeypatch, YT_INFO | {"subtitles": {}, "automatic_captions": {"en-orig": FMT}})

    def full(*a: object, **k: object) -> None:
        raise yt.GpuUnavailable("[metal::malloc] out of memory")

    monkeypatch.setattr(yt, "run_whisper", full)
    assert _run_main(monkeypatch, tmp_path, ["fetch", VID]) == yt.EXIT_OK
    assert capsys.readouterr().out.splitlines()[0] == "SOURCE: auto-subtitles-fallback"

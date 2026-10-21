# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # PyTorch with CUDA 12.8 kernels (sm_120 / RTX 50-series needs torch >= 2.7 from the cu128 index).
#     "torch==2.11.0",
#     # PeAudioFrameLevelModel (PE-A-Frame) is in transformers 5.x.
#     "transformers==5.20.2",
#     "numpy>=1.26,<3",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""Sound-event backend for video_watch.py: frame-level scores of text prompts over audio.

video_watch.py runs this script as a subprocess (`uv run sound_events.py <request.json>`), so
torch and transformers never enter the Whisper environment, `--no-sounds` never pays their import
cost, and all GPU memory is released when the process exits. The contract is two JSON files, so
another model only needs another script that honours it:

Request (written by video_watch.py):
    {"request_id": "...", "audio": "<raw PCM path>", "format": "s16le", "sample_rate": 48000,
     "prompts": ["music", ...], "model": "<HF model id>", "chunks": [[start_frame, n_frames], ...],
     "batch_size": 1, "out": "<response path>", "parent_pid": 1234, "deadline": <epoch seconds>}

Response:
    {"ok": true, "request_id": "...", "model": ..., "frame_rate": 25.0, "prompts": [...],
     "chunks": [{"start": start_frame, "frames": n, "logits": ["<base64 int16 per prompt>", ...],
                 "level": "<base64 int16>"}],
     "device": ..., "peak_vram_mb": ..., "timings": {...}}
    or {"ok": false, "request_id": "...", "kind": "gpu" | "model" | "audio" | "error", "message": "..."}

The audio is headerless 48 kHz mono 16-bit little-endian PCM (no 4 GiB WAV limit). A logit is the
model's per-frame score scale * <audio_frame, text> + bias (its probability is the sigmoid),
stored as little-endian int16 of round(logit * 100). A level is the frame's RMS in dBFS, stored
the same way (video_watch.py mutes silent frames with it). Frames are 40 ms (1920 samples).
The model never falls back to the CPU: without a usable CUDA device the response is kind "gpu".

The script exits on its own when the process that asked for it (`parent_pid`) is gone or the
`deadline` has passed, so a stopped fetch never leaves the model holding the GPU.

The model is downloaded once into the Hugging Face cache (HF_HOME / HF_HUB_CACHE), or into
VIDEO_WATCH_SOUND_CACHE when that variable is set. `uv run sound_events.py --prepare` (run by
`video_watch.py setup --sounds on`) installs this script's environment and downloads the model
without using the GPU, then exits.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

SAMPLE_RATE = 48_000
HOP = 1920  # samples per frame: 25 frames per second
CACHE_ENV = "VIDEO_WATCH_SOUND_CACHE"
DEFAULT_MODEL = "facebook/pe-a-frame-base"
SCALE = 100  # logits and levels travel as int16 hundredths
WATCH_INTERVAL_S = 2.0


def log(message: str) -> None:
    print(f"sounds: {message}", file=sys.stderr, flush=True)


def write_response(out: Path, data: dict) -> None:
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)  # sound-response.tmp.json: cleaned like the rest
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, out)


def fail(out: Path, request_id: str | None, kind: str, message: str) -> int:
    write_response(out, {"ok": False, "request_id": request_id, "kind": kind, "message": message})
    log(f"{kind} error: {message}")
    return 1


def is_gpu_error(exc: BaseException) -> bool:
    low = str(exc).lower()
    return any(p in low for p in ("cuda", "cudnn", "cublas", "out of memory", "no kernel image", "device-side"))


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True  # unknown parent: rely on the deadline
    if os.name == "nt":
        import ctypes

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


def start_watchdog(parent_pid: int, deadline: float | None) -> None:
    """Exit hard when the requesting process is gone or the deadline has passed."""

    def watch() -> None:
        while True:
            time.sleep(WATCH_INTERVAL_S)
            if not pid_alive(parent_pid):
                log("the requesting process is gone; stopping")
                os._exit(3)
            if deadline and time.time() > deadline:
                log("deadline passed; stopping")
                os._exit(3)

    threading.Thread(target=watch, daemon=True).start()


def load_model(model_id: str):
    import torch
    from transformers import AutoTokenizer, PeAudioFrameLevelModel
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    cache = os.environ.get(CACHE_ENV) or None
    kw = {"cache_dir": cache}
    try:  # no network round trip once the model is cached
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, **kw)
        model = PeAudioFrameLevelModel.from_pretrained(model_id, dtype=torch.float16, local_files_only=True, **kw)
    except OSError:
        log(f"downloading {model_id} (about 2.3 GB, once)")
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kw)
        model = PeAudioFrameLevelModel.from_pretrained(model_id, dtype=torch.float16, **kw)
    return tokenizer, model


def read_chunk(audio, start_frame: int, n_frames: int):
    """Samples of [start_frame, start_frame + n_frames) of raw s16le PCM as float32, zero-padded."""
    import numpy as np

    audio.seek(start_frame * HOP * 2)
    raw = audio.read(n_frames * HOP * 2)
    x = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    return np.pad(x, (0, n_frames * HOP - len(x)))


def levels_db(x, n_frames: int):
    """RMS level of each 40 ms frame in dBFS (digital silence about -200)."""
    import numpy as np

    frames = x[: n_frames * HOP].reshape(n_frames, HOP)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-10)


def b64_int16(values) -> str:
    import numpy as np

    q = np.clip(np.round(np.asarray(values, dtype=np.float64) * SCALE), -32700, 32700).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def run(request: dict) -> dict:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available to PyTorch (the sound map never runs on the CPU)")
    if request.get("format", "s16le") != "s16le" or int(request.get("sample_rate", SAMPLE_RATE)) != SAMPLE_RATE:
        raise ValueError("audio must be raw 48 kHz mono s16le PCM")
    t0 = time.monotonic()
    tokenizer, model = load_model(request["model"])
    model = model.to("cuda").eval()
    t_load = time.monotonic() - t0
    torch.cuda.reset_peak_memory_stats()

    prompts = list(request["prompts"])
    chunks = [tuple(c) for c in request["chunks"]]
    batch = max(1, int(request.get("batch_size") or 1))
    out_chunks = []
    t1 = time.monotonic()
    with torch.inference_mode(), open(request["audio"], "rb") as audio:
        tok = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        # transformers' get_text_audio_embeds() does not request hidden states; do what forward() does.
        hidden = model.text_model(**tok, output_hidden_states=True).hidden_states[-1][:, 0]
        text = model.text_audio_head(hidden)  # (prompts, dim)
        scale, bias = model.text_audio_logit_scale, model.text_audio_logit_bias
        # Chunks of equal length are batched; a shorter one (audio shorter than a chunk) runs alone.
        groups: dict[int, list[tuple[int, int]]] = {}
        for c in chunks:
            groups.setdefault(c[1], []).append(c)
        done = 0
        for members in groups.values():
            for i in range(0, len(members), batch):
                part = members[i : i + batch]
                x = np.stack([read_chunk(audio, s, n) for s, n in part])
                values = torch.from_numpy(x).to("cuda", torch.float16)[:, None, :]
                emb = model.get_audio_embeds(values)  # (batch, frames, dim)
                logits = ((emb @ text.T) * scale + bias).float().cpu().numpy()  # (batch, frames, prompts)
                for (s, n), arr, samples in zip(part, logits, x):
                    arr = arr[:n]
                    out_chunks.append({
                        "start": s,
                        "frames": n,
                        "logits": [b64_int16(arr[:, k]) for k in range(len(prompts))],
                        "level": b64_int16(levels_db(samples, n)),
                    })
                done += len(part)
                if len(chunks) > 8 and (done == len(chunks) or done % max(1, len(chunks) // 5) < len(part)):
                    log(f"{done}/{len(chunks)} chunks")
        torch.cuda.synchronize()
    t_infer = time.monotonic() - t1
    out_chunks.sort(key=lambda c: c["start"])
    return {
        "ok": True,
        "request_id": request.get("request_id"),
        "model": request["model"],
        "frame_rate": SAMPLE_RATE / HOP,
        "logit_scale": SCALE,
        "prompts": prompts,
        "chunks": out_chunks,
        "device": torch.cuda.get_device_name(0),
        "peak_vram_mb": round(torch.cuda.max_memory_reserved() / 2**20),
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
        "timings": {"load_s": round(t_load, 2), "infer_s": round(t_infer, 2)},
    }


def prepare(model_id: str) -> int:
    """Install-time step: import PyTorch and transformers (uv has installed them by now) and
    download the model into its cache. No GPU is used; prints the torch build to stderr."""
    import torch

    load_model(model_id)
    cuda = "cuda " + torch.version.cuda if torch.version.cuda else "no CUDA build"
    log(f"ready: torch {torch.__version__} ({cuda}), model {model_id}")
    print("PREPARED: yes", flush=True)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3) or (len(argv) == 3 and argv[1] != "--prepare"):
        print("usage: uv run sound_events.py <request.json> | --prepare [MODEL]", file=sys.stderr)
        return 2
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    # torch.distributed logs "NOTE: Redirects are currently not supported in Windows or MacOs." at
    # import; it must not look like one of the tool's NOTE: lines.
    import logging

    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    if argv[1] == "--prepare":
        return prepare(argv[2] if len(argv) == 3 else DEFAULT_MODEL)
    request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    out = Path(request["out"])
    request_id = request.get("request_id")
    start_watchdog(int(request.get("parent_pid") or 0), request.get("deadline"))
    try:
        result = run(request)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a typed response
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        if type(exc).__name__ == "OutOfMemoryError" or is_gpu_error(exc):
            return fail(out, request_id, "gpu", message)
        if (isinstance(exc, ValueError) and "pcm" in message.lower()) or getattr(exc, "filename", None) == request.get("audio"):
            return fail(out, request_id, "audio", message)
        if isinstance(exc, OSError):
            return fail(out, request_id, "model", f"{type(exc).__name__}: {message}")
        return fail(out, request_id, "error", f"{type(exc).__name__}: {message}")
    write_response(out, result)
    log(f"{len(result['chunks'])} chunks in {result['timings']['infer_s']} s, peak VRAM {result['peak_vram_mb']} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

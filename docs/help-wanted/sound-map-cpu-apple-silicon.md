# Sound map on the CPU and Apple Silicon

**Labels:** `help wanted`, `enhancement`, `good first issue` (for the measurement part)

## Context

The optional sound map (music, laughter, applause, explosions, ... with times) runs Meta's
PE-A-Frame base model in `skills/video-watch/scripts/sound_events.py`, a separate PyTorch helper.
In this release it requires an NVIDIA GPU: the helper installs PyTorch from the CUDA 12.8 index and
runs in fp16 on CUDA (about 4.4 GB of VRAM, about 34 s of model time per hour of audio on an RTX
5060 Ti). Without NVIDIA, the sound map is not offered.

PyTorch also runs on the CPU and on Apple GPUs (MPS), so a CPU or MPS path may be possible. Whether
it is fast enough, and whether it gives the same results, is unknown.

## Suggested approach

1. **Measure first (good first issue).** Run the helper's model on the CPU (fp32) and, on a Mac, on
   MPS, for a 5-minute and a 1-hour public video. Report wall time, peak memory and whether the
   per-frame logits match the CUDA output closely (the CUDA fp16 vs fp32 check found a largest
   probability difference of 0.005).
2. If the speed is acceptable, propose how to expose it: for example a device for the helper, a CPU
   or MPS PyTorch build chosen by `setup`, and a clear "very slow" label in `doctor` and the install
   guide.
3. Keep the helper's JSON request/response contract unchanged, so `video_watch.py` needs only the
   device choice and the time estimate.

Notes:

- The helper runs after Whisper has finished, in its own process; keep it that way.
- The calibration (thresholds, music margin) was done on CUDA output; a new device needs a check
  that the same rules give the same spans on a few videos (see
  [DESIGN.md](../DESIGN.md#calibration)).

## Acceptance criteria

- [ ] Measured timings and memory for CPU and/or MPS posted in this issue.
- [ ] If implemented: `setup --sounds on` works on the new device with sizes shown before
      downloading; `doctor` reports `RECOMMEND_SOUNDS` with an honest reason (for example "CPU:
      about N minutes per hour of audio").
- [ ] Spans on at least three public videos match the CUDA spans closely (differences explained).
- [ ] A failure still becomes `NOTE: sound map skipped: …` and never changes `fetch`'s exit code.
- [ ] Unit tests without a GPU; README, `docs/REFERENCE.md` and `CHANGELOG.md` updated.

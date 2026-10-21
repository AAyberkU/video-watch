# Hearing beyond speech: model research for the sound map

> Research from 2026-10-20, before the sound map was built; model versions, benchmarks, licences and
> prices were current then. What was finally chosen and measured is in
> [DESIGN.md](../DESIGN.md#sound-map).

Research date: 2026-10-20. Method: web research only, using primary sources wherever possible (papers,
official repos, model cards, PyPI, vendor changelogs). Nothing was installed or run.

Conventions used in this report:

- **Fact**: stated by a primary source (cited in the Sources section).
- **(est.)**: my estimate, usually derived from parameter count or architecture. It is not measured.
- **(unverified)**: I could not confirm it from a primary source, or the sources conflict.
- mAP numbers are AudioSet-2M (full AS) single-model eval mAP unless stated. Ensembles are marked.

Test environment: Windows 11, RTX 5060 Ti 16 GB (Blackwell, sm_120), `uv run` + PEP 723, one model loaded
at a time, Whisper large-v3 (faster-whisper fp16, about 7 GB) already in the pipeline.

---

## 0. Executive summary

| Category | Recommendation | Runner-up | Driving facts |
| --- | --- | --- | --- |
| 1. Sound events (adopt now) | **Meta PE-A-Frame (base, 0.6B)**: frame-level (40 ms), open-vocabulary SED in HF `transformers`, Apache-2.0, not gated. Confirm it with a one-day bake-off before locking it in. | **PretrainedSED `BEATs_strong`** (closed set, 447 AudioSet-Strong classes, 40 ms, best AudioSet-Strong PSDS). Safe, simple fallback: **CED-base / CED-mini** windowed tagging (527 classes, 50.0 / 49.0 mAP, Apache-2.0 on HF, ONNX). | PE-A-Frame-L: AudioSet-Strong PSDS1A 0.43 / PSDS1T 0.61, best on DESED. PretrainedSED: PSDS1A 0.47 / PSDS1T 0.52 on AudioSet-Strong. CED-base: 50.0 mAP at 86M params. |
| 2. Diarization (later) | **pyannote `speaker-diarization-community-1`** (pyannote.audio 4.0.x). It works in any language, has an "exclusive" mode built for STT alignment, and its weights are CC-BY-4.0. | **NeMo Streaming Sortformer v2.1** (very fast, commercial-OK licence, but at most 4 speakers, English-trained, NeMo on Windows is unsupported). DiariZen has the best open DER but NC weights. | community-1 DER (no collar): VoxConverse 11.2, AMI-SDM 19.9, DIHARD3 20.2. The download is gated (HF token plus accepting the terms). Paid precision-2 scores DIHARD3 14.7. |
| 3. Emotion | **Skip** as a standalone feature. Surface paralinguistic *events* (laughter, crying, shouting, screaming, sighing) from the SED model instead. | Experimental only: emotion2vec_plus_base / large (9 classes) behind a flag, labelled "low confidence". | Natural speech: the best MSP-Podcast 8-class macro-F1 is 0.45 (challenge leaderboard, in-domain). Zero-shot speech-LLMs on Turkish TurEV-DB (4 classes) score about 22–24 F1, roughly chance. |

Is there one model that does it all? No. SenseVoice-Small has only 8 event tags, gives no event
timestamps, and supports 5 languages (no Turkish). Audio-LLMs describe sound well in prose but tag
poorly: MiDashengLM-7B scores 8.86 AudioSet mAP and Qwen2.5-Omni-7B scores 6.48, against about 50 for
specialist taggers. They also need 30-s windows and roughly 10–36 GB of VRAM. Keep them as an
optional on-demand "describe this segment" tool, not as the timeline engine.

---

## 1. Sound event detection / audio tagging

### 1.1 What "good" means for us

- We need a **timeline over long audio** (up to hours), not one label per clip. That leaves two families:
  1. **Clip taggers** (AudioSet weak labels, 527 classes, trained on 10-s clips) run in sliding windows.
     Mature and accurate, but onsets and offsets are only as precise as the window hop (about 1–5 s).
  2. **Frame-level SED** (strong labels, or frame-wise audio-text alignment) gives 40 ms resolution and
     real onsets and offsets. This family became practical only in 2024–2025 (AudioSet-Strong
     pre-training, PE-A-Frame, FlexSED).
- AudioSet mAP measures clip-level tagging quality. PSDS on AudioSet-Strong / DESED measures
  timeline quality. They are not comparable.

### 1.2 Clip-level taggers (AudioSet-2M, 527 classes unless noted)

| Model | AS-2M mAP (single) | Params | Size on disk | Temporal output | Code license | Weights license / hosting | Install path | Maintenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PANNs CNN14 | 43.1 (CNN14 @32 kHz); 43.8 (16 kHz); DecisionLevelMax 38.5 | ~81M | 327 MB (Zenodo) | CNN14: clip. DecisionLevelMax/Att: framewise output (10 ms hop at 32 kHz (est. from code config)) | MIT (`panns_inference`) | CC BY 4.0 (Zenodo) | `pip install panns-inference` (easy) | Dormant since ~2020 |
| YAMNet | 30.6 (balanced mAP, 521 classes) | 3.7M | ~15 MB (est.) | 0.96-s windows, 0.48-s hop | Apache-2.0 (tensorflow/models) | Apache-2.0 | TF/Keras 2 only ("incompatible with Keras 3"). Native-Windows TF has had no GPU support since TF 2.10. ONNX ports exist (unverified quality) | Legacy |
| AST (MIT/ast-finetuned-audioset-10-10-0.4593) | 45.9 (ensemble 48.5) | 87M | ~350 MB (est.) | clip (10.24 s) | BSD-3-Clause | BSD-3-Clause, HF, not gated | `transformers` native (easiest) | Stable, old |
| HTS-AT | 47.1 | ~30M | ~120 MB (est.) | clip, plus framewise localisation head (DESED recipe) | MIT | Google Drive checkpoint | Clone the repo (moderate) | Dormant (ICASSP 2022) |
| PaSST-S (`passt_s_swa_p16_128_ap476`) | 47.6 (ensemble 49.6) | ~86M | ~350 MB (est.) | clip; `hear21passt` gives timestamp embeddings | Apache-2.0 | GitHub releases | `pip install hear21passt` (easy, 32 kHz) | Low activity |
| EfficientAT mn10_as / dymn10_as / mn40_as_ext / dymn20_as | 47.1 / 47.7 / 48.7 / 49.1 | 4.9M / 10.6M / 68M / 40M | ~20 / 42 / 270 / 160 MB (est.) | clip. Repo ships `windowed_inference.py` (CLI default 10-s window, 2.5-s hop, top-10 tags per window) | MIT | GitHub releases (auto-download); no separate weight licence stated | Not pip-packaged (on the roadmap): vendor about 3 files | Moderate |
| BEATs iter3+ AS2M (iter1/2/3: 47.9/48.1/48.0) | 48.6 (ensembles: 5 models 50.4, 10 models 50.6) | ~90M | ~360 MB (est.) | clip | MIT (microsoft/unilm) | OneDrive links; HF community mirrors | Copy `BEATs.py`; pure torch (moderate) | Frozen since 2022–23 |
| M2D-AS / M2D-CLAP_2025 | 48.5 / 49.0 | ~86M | ~350 MB (est.) | clip; 40-ms-frame variant exists | **LICENSE.pdf (NTT), terms not readable online (unverified; possibly research-only)** | GitHub releases | Clone (moderate) | Active (2025 release) |
| EAT-base (30 ep) / EAT-large | 48.9 / 49.5 (base 10 ep: 48.6) | 88M / 309M | ~350 MB / 1.2 GB (est.) | clip, variable-length supported | MIT | HF + GDrive; HF `AutoModel` integration since 2025-05-03 (no fairseq needed for inference) | `transformers` + `trust_remote_code` (easy) | Active 2025 |
| CED-Tiny / Mini / Small / Base | 48.1 / 49.0 / 49.6 / 50.0 | 5.5M / 9.6M / 22M / 86M | Base ~340 MB fp32 (est.); Mini ~40 MB (est.) | clip, 16 kHz, 64 mel. Designed to run fast on CPU. | **GitHub code GPL-3.0** | **HF `mispeech/ced-*` tagged Apache-2.0** | `transformers` (`trust_remote_code`) or `optimum.onnxruntime` ONNX; sherpa-onnx builds (easy, cross-platform) | ICASSP 2024; light maintenance |
| Dasheng-Base AudioSet fine-tune | 49.7 | 86M (encoders up to 1.2B) | ~350 MB (est.) | clip (mean-pooled head) | Apache-2.0 | Zenodo checkpoint | `pip install dasheng` (easy) | Active (Xiaomi) |
| SSLAM (AS2M fine-tuned) | **50.2** (best single model found) | 90.4M | ~360 MB (est.) | clip | MIT | HF `ta012/SSLAM_AS2M_Finetuned` (MIT tag) | HF `trust_remote_code`; full repo is fairseq-based (fragile) | ICLR 2025 |

Notes:

- The single-model ceiling sits around 49–50 mAP and has been flat since about 2023. Differences of about
  1 mAP are within noise for our use. Pick on licence, install, speed and temporal output instead.
- Third-party "AudioSet leaderboards" (e.g. codesota, March 2026) mix single models and ensembles
  (they list AST at 0.485, which is the AST *ensemble*). Treat them as unverified. Neither the
  leaderboards nor arXiv showed a 2026 single model above 50.2.
- AudioSet-R (Aug 2025) relabels AudioSet with LLMs. It is a data effort, not a released tagger.

### 1.3 Frame-level SED (timestamps)

| Model | What it outputs | Resolution | Benchmarks | Params | License | Install | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **PretrainedSED** (Schmid et al., ICASSP 2025), checkpoints `BEATs_strong_1`, `ATST-F_strong_1`, `fpasst_strong_1`, `ASIT_strong_1`, `M2D_strong_1`, plus low-complexity `frame_mn10` (3.83M) and `frame_mn06` (1.62M) | 447 AudioSet-Strong classes, per-frame probabilities; `inference.py` handles "arbitrary length" audio | 40 ms (training targets at 25 Hz; inference hop not stated (unverified)) | AudioSet-Strong PSDS1: BEATs 46.5, M2D 46.3, ASiT 46.2, ATST-F 45.8, fPaSST 45.4 (teacher ensemble 47.1). In the PE-AV paper: PSDS1A 0.47 / PSDS1T 0.52 / AUROC 0.98. frame_mn scores not published (unverified) | ~90M for transformers | Repo MIT; weights in the repo's v0.0.1 GitHub release, no separate weight licence stated. The M2D_strong derivative may inherit NTT terms (unverified) | Not packaged. README targets Python 3.9 conda, torch for your CUDA, `minimp3py` from git. Vendor the model code and decode with ffmpeg | Best closed-set timeline quality on the AudioSet ontology. Research-grade repo |
| **PE-A-Frame** small / base / large (Meta, 2025-12) | Frame embeddings scored against **any text prompt**. Returns `(start, end)` spans per prompt via `threshold` | 40 ms (25 Hz) | AudioSet-Strong (L): PSDS1A 0.43, PSDS1T 0.61, AUROC 0.96. DESED: L 0.34 / 0.58 / 0.97, **B 0.39 / 0.56 / 0.98**, S 0.32 / 0.54 / 0.96 (best on DESED of all tested). Open-vocab internal AUROC: S 0.91, B 0.92, L 0.91 | 0.4B / 0.6B / 1–1.4B (card says 1.4B, safetensors 1B) | **Apache-2.0**, not gated | HF `transformers` (`PeAudioFrameLevelModel`, `PeAudioProcessor`; transformers v5 era, docs added 2025-12-16) or Meta's `perception_models` | Open vocabulary lets Claude ask "when is the explosion?". Weaknesses: more false positives than PretrainedSED (lower PSDS1A), one global similarity threshold across prompts (calibration risk), sample rate and max length not stated (unverified), the model-card snippet has naming inconsistencies |
| FlexSED (WASPAA 2025, Dasheng backbone + CLAP text) | Open-vocab frame-level | 25 Hz | AudioSet-Strong 0.45 / 0.58 / 0.96; DESED 0.16 / 0.27 | ~0.1–0.6B (unverified) | Licence not found (unverified) | HF Space; training still WIP | Superseded by PE-A-Frame on most metrics |
| OpenFLAM (Adobe, ICML 2025) | Open-vocab frame-wise | 3.2 Hz | AudioSet-Strong PSDS1A 0.35 | — | **Non-commercial Adobe Research License** | `pip install openflam` | Licence trap for a public repo |
| HTS-AT / PANNs DecisionLevel heads | Weak-label framewise outputs | 10 ms–~0.3 s | Not competitive with strong-label models | — | MIT | — | Legacy |
| FireRedVAD AED (2026-03) | speech / singing / music only (3 classes) | frame | VAD F1 97.6 on FLEURS-VAD-102 | small (DFSMN) | Licence not checked (unverified) | pip / Python API | Too few classes. Could be a useful music/singing detector later |

### 1.4 Zero-shot CLAP family (text-prompted, mostly clip-level)

| Model | Zero-shot evidence | License | Status |
| --- | --- | --- | --- |
| LAION-CLAP (`laion-clap`, HTSAT backbones) | ESC-50 about 90–93 % (repo); no AudioSet mAP published | CC0-1.0 | Repo says "work in progress"; 2023 checkpoints |
| MS-CLAP 2023 (`msclap`) | ESC-50 93.9 % | MIT | **Repository archived 2026-09-15** (read-only) |
| M2D-CLAP_2025 | Claims SOTA zero-shot plus 49.0 mAP fine-tuned | NTT LICENSE.pdf (unverified) | Active |
| PE-AV (Meta, 2025-12; 0.8–2B incl. video) | Claims SOTA zero-shot sound/music/speech classification (tables do not report AudioSet mAP) | Apache-2.0 (unverified per checkpoint; frame models confirmed) | New |

CLAP models are useful for ad-hoc text queries. For a timeline, PE-A-Frame already covers this
with frame resolution, so a separate CLAP is not needed.

### 1.5 SenseVoice as an event detector

- Event tags: BGM, Speech, Applause, Laughter, Cry, Sneeze, Breath, Cough (8). Emotion: HAPPY, SAD, ANGRY,
  NEUTRAL, FEARFUL, DISGUSTED, SURPRISED.
- Per the FunAudioLLM paper, **SenseVoice-Small "can only report what happened, with at most one event per
  utterance, and cannot give timing"**. Only SenseVoice-Large predicts event start and end, and only Small
  is open-sourced (Large not found as an open release (unverified)).
- The authors say AED "has some gaps compared to specialized AED models" (BEATs/PANNs score higher F1 on ESC-50).
- SenseVoice-Small ASR languages: zh, yue, en, ja, ko. **No Turkish.**
- Speed is excellent: 70 ms for 10 s of audio (vendor claim). Code is MIT. Weights use the FunASR Model
  License v1.1 (attribution, keep model names, a "no denigration" clause; commercial use is not
  explicitly addressed, though a maintainer said in an issue that it is allowed).
- Verdict: not a fit for our timeline.

### 1.6 Audio-language models (describe sound in natural language)

| Model | Size | Tagging evidence | VRAM reality | License | Notes |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-Omni-3B / 7B (2025-03/04) | 3B / 7B | MMAU sound 70.3 / 67.9. **AudioSet mAP 6.48 (7B, per MiDashengLM paper)**. MELD emotion 0.558 / 0.570 | Official (video input, 15 s): 3B BF16 18.4 GB; 7B BF16 31.1 GB; 7B GPTQ-Int4 11.6 GB / AWQ 11.8 GB; "real usage typically ≥1.2x". Audio-only is lower (est.) | Apache-2.0 repo; 3B has a separate licence (unverified, possibly Qwen Research) | The only realistic 16 GB option is 7B-Int4 or 3B |
| Qwen3-Omni-30B-A3B Instruct / Thinking / Captioner (2025-09) | 30B MoE (3B active) | MMAU 77.5. Captioner gives "low-hallucination" captions, recommended for ≤30-s clips. Turkish is among its 19 speech-input languages | 68–79 GB BF16 for 15 s (official). **Does not fit 16 GB** | Apache-2.0 | Captioner quality is the best open option, but only with server-class GPUs |
| MiDashengLM-7B (Xiaomi, 2025-08) | 8B | **AudioSet mAP 8.86**, FSD50K 37.6, VGGSound 52.1, AudioCaps FENSE 62.2 | BF16 about 16–17 GB (est.): needs quantisation to fit | Apache-2.0 | Fastest 7B ALM (3.2x throughput vs Qwen2.5-Omni) |
| MOSS-Audio 4B / 8B (OpenMOSS, 2026-04) | 4.6B / 8.6B | MMAU 77.6 (4B-Thinking). "Detects key acoustic events". Time-aware QA | Not stated. 4B BF16 about 9–10 GB (est.) | Apache-2.0 | Newest small ALM. Repo install (conda, Python 3.12), not `transformers` |
| Kimi-Audio-7B-Instruct | ~7B (+ detokenizer) | Claims sound-event / scene classification. AudioSet mAP 3.47 (per MiDashengLM paper) | ~10 GB weights bf16 (third-party); 23 GB target mentioned (unverified) | MIT + Apache (Qwen-derived) (unverified) | Docker-oriented install |
| Audio Flamingo 3 (NVIDIA, 2025-07) | ~8B | Strong on benchmarks; 30-s windows, 10-min cap | Loaded at about 36 GB (third-party) | **NVIDIA OneWay Non-Commercial** | Licence trap |
| Nemotron 3 Nano Omni (NVIDIA, 2026-04) | 30B-A3B; BF16 / FP8 / NVFP4 | Speech-focused; sound tagging unconfirmed | NVFP4 about 18 GB-equivalent (third-party est.). Does not fit 16 GB | NVIDIA Open Model License (commercial OK) | Needs vLLM ≥0.20 (Linux-centric) |
| Inkling (Thinking Machines, 2026-07) | 975B MoE | MMAU 77.2 | Datacenter only | Apache-2.0 | Irrelevant locally |
| SALMONN, Qwen2-Audio-7B | 7–13B | Older; superseded by the above | 16–30 GB BF16 (est.) | Qwen2-Audio Apache-2.0; SALMONN depends on Vicuna/LLaMA terms (unverified) | Not recommended |

Takeaways:

- ALMs are **10–15x worse at multi-label tagging** than an 86M tagger (AudioSet mAP about 3.5–8.9 vs
  about 50). They are good at "what's happening and what's the mood" captions for a short segment.
- Practical limits: 30-s windows, 10–36 GB of VRAM at BF16, generation speed of seconds per window (est.), and
  hallucination risk. Running Whisper and an ALM back to back on 16 GB works only with a ≤4B or
  4-bit model.
- Future option: on-demand "describe 01:05–01:35" using Qwen2.5-Omni-7B-AWQ or MOSS-Audio-4B,
  triggered by Claude for specific segments. Never use one for the full-length timeline.

### 1.7 Speed and VRAM (all estimates unless noted)

| Model | VRAM (fp16, batch of 10-s windows) | Throughput on RTX 5060 Ti |
| --- | --- | --- |
| EfficientAT mn10 / CED-mini / frame_mn10 | < 0.5 GB | > 1000x real time; CPU also fine (CED-Tiny "faster than MobileNets on one x86 core", per the author) |
| CED-base / BEATs / AST / PaSST / BEATs_strong | 0.5–1.5 GB | ~300–1000x real time (1 h of audio in seconds to tens of seconds) |
| PE-A-Frame-base (0.6B) | 2–4 GB | ~100–400x real time; text-prompt embeddings are computed once (est., unverified) |
| PE-A-Frame-large (~1–1.4B) | 4–6 GB | ~50–200x real time (est.) |
| 7B ALM (4-bit) | 10–14 GB | about 1–10x real time on 30-s windows (est.) |

No primary source publishes RTFs for these taggers on consumer GPUs. A bake-off should measure them.

### 1.8 Recommendation: sound events

**Adopt now: PE-A-Frame-base (`facebook/pe-a-frame-base`)**, conditional on a short bake-off.

- Gives a true timeline with 40 ms frames and spans. Best DESED scores. Close to the closed-set
  specialist on AudioSet-Strong.
- Open vocabulary. We can ship a curated label set (about 60–150 video-relevant prompts: music,
  singing, laughter, applause, cheering, crowd, shouting, screaming, crying, gunshot, explosion, engine,
  siren, dog, footsteps, door, typing, game UI beeps, and so on). Claude can also run ad-hoc queries
  later, which is a strong fit for an agent tool.
- Apache-2.0, not gated, in `transformers`, maintained by Meta and HF, and cross-platform. A pure
  PyTorch stack that fits PEP 723.
- Risks: newer and less battle-tested; one global similarity threshold (PSDS hides threshold
  calibration); possible prompt sensitivity; transformers v5 API churn; sample rate and max length not
  documented (chunk it ourselves).

**Runner-up: PretrainedSED `BEATs_strong_1`** (or `frame_mn10` for CPU users).

- Calibrated per-class probabilities over a fixed 447-class AudioSet-Strong ontology, which is
  simpler to threshold. Best closed-set PSDS1A (0.47).
- Costs: vendoring research code (MIT), Python 3.9-era instructions, and an unstated weight licence.

**Lowest-risk fallback: CED-base (GPU) / CED-mini (CPU)** run as a sliding-window tagger (e.g.
10-s window with 1–2.5-s hop, plus hysteresis merging).

- 50.0 / 49.0 mAP, Apache-2.0 on HF, ONNX path, tiny.
- Onsets are fuzzy (± window/2), so "[01:05] laughter" becomes "[~01:03–01:08]".
- Mind the licence split: CED's GitHub code is GPL-3.0. Only use the HF `transformers` / ONNX artefacts
  (Apache-2.0 tag).

Suggested bake-off (1 day): 5–6 public test videos (a non-English vlog with music, gameplay, podcast,
music video, live event with applause, shorts), about 30 min total.

- Hand-mark about 50 events.
- Compare PE-A-Frame-base, BEATs_strong and CED-base on hit rate, false positives per minute,
  onset error, wall time and VRAM.
- Pick PE-A-Frame only if its false-positive rate at a single tuned threshold is acceptable.

---

## 2. Speaker diarization (optional, future)

### 2.1 Comparison

DER in %, lower is better. "No collar, overlap included" unless stated. Conditions differ across
vendors, so compare within a row only where noted.

| System | AMI-SDM | DIHARD3 full | VoxConverse | AISHELL-4 | AliMeeting | CALLHOME p2 | Max speakers | Speed | License / gating | Windows install |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pyannote 3.1 (legacy) | 22.7 | 21.4 | 11.2 | 12.2 | 24.5 | 28.5 | unbounded (clustering) | ~45x RT (ETH benchmark, 2025-09) | MIT code; weights MIT, **gated** | pyannote.audio 3.x; older torch |
| **pyannote community-1** (2025-09, pyannote.audio 4.0.x; latest 4.0.7, 2026-06-30) | 19.9 | 20.2 | 11.2 | 11.7 | 20.3 | 26.7 | unbounded | 31 s per 1 h of audio on H100 (~116x RT); ~40–80x RT on 5060 Ti (est.) | MIT code; **weights CC-BY-4.0, gated (HF token plus accept terms)**; offline use after the first download is documented. **Telemetry**: disable with `PYANNOTE_METRICS_ENABLED=0` | `torch>=2.8`, `torchaudio>=2.8`, `torchcodec>=0.7`. **torchcodec on Windows is CPU-only via pip and needs FFmpeg shared DLLs** (workaround: pass an in-memory `{"waveform", "sample_rate"}`) |
| pyannoteAI precision-2 (paid) | 15.6 | 14.7 | 8.5 | 11.4 | 15.2 | 16.6 | — | 14 s per 1 h on H100 | API, from €0.096/h. **precision-3** (2026-09-17): average DER 16.02 → 14.35, same price | API |
| DiariZen Large-s80-v2 (BUT, 2025-12) | 13.9 | 14.5 | 9.1 | 10.1 | 10.8 (far) | — | unbounded | ~20x RT (ETH benchmark) | MIT code; **weights CC BY-NC 4.0 (research only)** | Conda, **pins torch 2.1.1+cu121 (no sm_120 kernels)**, bundled pyannote fork. Fragile on Blackwell/Windows |
| NeMo Streaming Sortformer 4spk v2.1 | 17.8 (30.4-s latency) | 19.5 | — | — | 13.6 (far) | 10.1 (0.25-s collar) | **4** (degrades at ≥5; DIHARD ≥5spk 38.9) | RTF 0.002 (~500x) at 30-s latency on RTX 6000 Ada | NVIDIA Open Model License (commercial OK); not gated (unverified) | `nemo_toolkit[asr]`; NVIDIA only lists Linux, WSL recommended. Native Windows works for some users (unverified) |
| NeMo MSDD (clustering + MSDD) | — | — | — | — | — | — | unbounded | — | CC-BY-4.0 (unverified) | Heavy NeMo stack; superseded by Sortformer for most uses |
| 3D-Speaker pipeline (Alibaba) | 21.8 | — | 11.75 | 10.3 | 19.7 | — | unbounded | — | Apache-2.0 code; ModelScope weights (licence unstated) | Repo badges say Linux; conda Python 3.8 |
| Reverb diarization v1 / v2 (Rev, 2024) | WDER only: Earnings21 0.047 / 0.046 vs pyannote 3.0 0.051 | — | — | — | — | — | unbounded | — | Weights licence "other", **gated** (reported non-commercial (unverified)) | pyannote-based |
| SpeechBrain ECAPA + spectral clustering | No comparable DER (headline numbers use oracle VAD) | — | — | — | — | — | unbounded | fast | Apache-2.0 | pip; needs our own VAD and overlap handling (none) |
| sherpa-onnx diarization (pyannote-seg-3.0 or reverb-v1 segmentation + 3D-Speaker/NeMo embeddings, ONNX) | Not published | — | — | — | — | — | user-set or threshold | fast, CPU | Apache-2.0 runtime; model licences follow their sources | pip wheels, cross-platform. Easiest Windows path |
| WhisperX diarization | Uses pyannote **community-1** (pyproject pins `pyannote-audio>=4.0.0`, `torch~=2.8.0`) | | | | | | | | needs HF token | Its wav2vec2 aligner has Turkish: `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` |

Independent benchmark (ETH Zürich, arXiv 2509.26177, 2025-09): overall DER was PyannoteAI 11.2 % and
DiariZen 13.3 %. Speed: Sortformer v2 about 214x RT, pyannote 3.1 about 45x, DiariZen about 20x. Missed speech
(onset/offset timing) was the dominant error. Spanish was hardest; English and Mandarin were best.
Sortformer degraded with more than 4 speakers.

Turkish: none of these systems publishes Turkish DER. Clustering pipelines (pyannote, DiariZen,
3D-Speaker) rely mostly on language-independent acoustics; pyannote states community-1 "works in any
language" (vendor blog). Sortformer's card warns of degradation on non-English speech.

Combining with Whisper word timestamps: community-1 has `exclusive_speaker_diarization` (one
speaker at a time, backported from precision-2) for "simpler reconciliation with transcription
timestamps". Assign each faster-whisper word to the speaker with maximum overlap. WhisperX's
`assign_word_speakers` does the same.

### 2.2 Recommendation: diarization (when we add it)

- **Choose pyannote community-1.**
  - Best open, commercially usable, language-agnostic option.
  - Unbounded speaker count and overlap-aware, with the exclusive mode for word assignment.
  - Low VRAM (about 1–2 GB, est.) and pip-installable.
- Costs:
  - Gated download: each user needs an HF token and must accept the terms. Document it and keep
    the feature opt-in.
  - torchcodec/FFmpeg friction on Windows: feed decoded waveforms ourselves.
  - Telemetry: set the env var off.
- Upgrade paths:
  - Precision-2/3 API (paid; DIHARD3 14.7 vs 20.2).
  - DiariZen for personal, non-commercial use only.
- Sortformer v2.1 only if we need very fast processing of recordings with at most 4 speakers.

---

## 3. Speech emotion recognition / paralinguistics

### 3.1 Evidence

| Model / system | Classes | Acted-data results | Natural-data results | Cross-lingual / Turkish | Size | License |
| --- | --- | --- | --- | --- | --- | --- |
| emotion2vec (base, frozen + linear) | 4 (IEMOCAP) | IEMOCAP WA 71.8 / UA 72.7; RAVDESS 82.4; EmoDB 84.3; SUBESCO 90.9 | **MELD WA 51.9 / UA 28.0**; M3ED UA 27.0 | Pre-trained on 262 h of **English-only** emotion data; linear probes on 9 other languages beat SSL baselines; **no Turkish result** | ~94M | MIT code; weights FunASR model licence |
| emotion2vec_plus seed / base / large (2024-05) | 9 (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown) | Numbers only in figures (no tables) | Same | Pseudo-labelled 40k h (mostly Chinese/English, inferred) | seed ? / ~90M / ~300M | "model-license" (FunASR Model License v1.1: attribution, no denigration clause; commercial use not explicit) |
| SenseVoice-Small (zero-shot SER) | 7 | WA: CREMA-D 74.0, ESD 85.5, IEMOCAP 65.7, CASIA 70.0 | **MELD 57.8, MSP-Podcast 64.1 (WA; class-imbalanced, so UA would be much lower)** | ASR in 5 languages, **no Turkish** | small (unverified ~234M) | FunASR licence |
| Whisper-large-v3 features + head (EmoBox, intra-corpus, best of 10 models) | dataset-specific | IEMOCAP UA 73.5; **TurEV-DB (Turkish, acted, 4 classes) UA 81.3** | **MELD UA 31.5; MSP-Podcast UA 22.2** | Trained per corpus | 1.5B | — |
| Speech-LLMs zero-shot (VoxEmo, 2026-03): Qwen2-Audio-7B, Audio Flamingo 3 | prompt-based | IEMOCAP macro-F1 about 64 (AF3) | MELD about 30; MSP-Podcast about 29 | **TurEV-DB macro-F1 about 22–24 zero-shot (4 classes: chance is 25)**; even after SFT only 42–51 vs 81 supervised | 7–8B | AF3 non-commercial |
| Qwen2.5-Omni 3B / 7B | MELD | — | MELD 0.558 / 0.570 | — | 3–7B | Apache-2.0 |
| MSP-Podcast challenge, best systems (2025 leaderboard) | 8 categorical | — | **Best macro-F1 0.453** (Konkuk-AICV, in-domain, ensembles); attributes CCC 0.61 (valence 0.68, arousal 0.64) | English podcasts | ensembles | — |
| audeering wav2vec2-large-robust-12-ft-emotion-msp-dim | arousal / dominance / valence | — | Trained on MSP-Podcast v1.7 | English | ~165M (est.) | **CC BY-NC-SA 4.0** (licence trap) |

### 3.2 Verdict: skip as a feature (confirmed)

- On natural speech, the field's best in-domain systems reach about 0.45 macro-F1 on 8 classes. Off-the-shelf
  models land at 22–32 % UA on MELD and MSP-Podcast.
- High headline accuracies (70–90 %) come from acted corpora with exaggerated emotions and per-corpus training.
- Turkish evidence is the worst case:
  - The only Turkish SER corpus in the big benchmarks (TurEV-DB) is acted and tiny (0.5 h, 6 speakers).
  - Zero-shot speech-LLMs score at chance on it.
  - No Turkish natural-speech results exist for emotion2vec or SenseVoice.
- Emotion is also highly context- and culture-dependent, and annotator agreement on natural data is
  low. A wrong "angry" label next to a transcript actively misleads Claude, which already infers
  tone from the words.
- Licences add friction: audeering is NC; FunASR licence wording is non-standard.
- **What to do instead (cheap, robust)**:
  - Emit paralinguistic *events* from the SED model: laughter, giggle, crying/sobbing,
    screaming, shouting/yelling, sigh, whispering, cheering, booing, gasp. These are acoustically
    distinct, detected reliably, and language-independent.
  - Optionally add simple prosody stats per speech segment (loudness and pitch range/speed via
    ffmpeg/librosa) as "energetic / calm" hints.
- If we ever experiment: emotion2vec_plus_base behind a flag, output only when probability is above a
  high threshold, labelled "acoustic emotion guess (low reliability)".

---

## 4. Can one model cover several categories?

| Candidate | Covers | Trade-off |
| --- | --- | --- |
| SenseVoice-Small | ASR + 8 events + 7 emotions + LID | Utterance-level tags only, no event timestamps, no Turkish ASR, weaker AED than specialists. Not useful next to Whisper. |
| Audio-LLMs (Qwen2.5-Omni, MOSS-Audio, MiDashengLM, Qwen3-Omni) | Speech, sounds, music, emotion and reasoning in prose | AudioSet mAP about 3–9 vs about 50, so they miss most events in a dense multi-label timeline. 30-s windows. 10–80 GB VRAM. Slow. Hallucinate. Emotion on Turkish at chance. Good only as an on-demand segment describer. |
| PE-A-Frame | Events + paralinguistic events (laughter, crying, screaming) via prompts | No identity, no categorical emotion. This is the closest thing to "two categories in one" that is still reliable. |
| pyannote / Sortformer | Diarization + VAD / overlap | Already well scoped. |

Conclusion: specialised models win on accuracy, speed, VRAM and licence clarity. The best combined
design is Whisper (speech) + one SED model (events, including paralinguistic ones), plus later diarization,
plus an optional on-demand ALM describer.

---

## 5. Risks

### 5.1 Blackwell (sm_120) / CUDA

- PyTorch: stable sm_120 support starts at **torch 2.7 (cu128 wheels)**. Pin `torch>=2.7` and use the
  cu128 index in PEP 723 metadata. Older pins (DiariZen torch 2.1.1+cu121, PretrainedSED Python-3.9
  instructions, old NeMo images) fail with "no kernel image is available".
- onnxruntime-gpu: **1.23 shipped a Blackwell regression** (built for `90a-virtual`, so the Slice
  kernel was missing on SM 12.0). It was fixed by reverting to `90-virtual` (PR #26230). 1.22 worked. Pin
  around the broken release, or use CPU ORT for small taggers.
- Triton on Windows: `ptxas ... sm_120 not defined` reports. Avoid `torch.compile`.
- torchcodec (pyannote 4 dependency): Windows pip wheels are CPU-only and need FFmpeg 4–7 *shared* DLLs
  on the DLL search path (FFmpeg 8 is not supported on Windows). Pass in-memory waveforms instead.
- TensorFlow (YAMNet): no native-Windows GPU since TF 2.10.
- NeMo: Windows is not an officially supported platform (WSL recommended).
- funasr (SenseVoice / emotion2vec): `editdistance` lacks wheels for some Python versions on
  Windows (e.g. 3.13), so pin Python 3.12.

### 5.2 Licence traps for a public repo

| Item | Issue |
| --- | --- |
| DiariZen weights | CC BY-NC 4.0, research only |
| audeering dim-emotion | CC BY-NC-SA 4.0 |
| Audio Flamingo 2/3 | NVIDIA OneWay Non-Commercial |
| OpenFLAM | Adobe Research non-commercial |
| M2D / M2D-CLAP (and `M2D_strong` in PretrainedSED) | NTT LICENSE.pdf, terms not verified, possibly research-only. Avoid until read. |
| Reverb diarization | Custom licence, gated (reported non-commercial, unverified) |
| CED | GitHub code GPL-3.0 vs HF weights Apache-2.0. Use only the HF / ONNX artefacts, don't copy GitHub code |
| FunASR models (SenseVoice, emotion2vec) | Non-standard licence: attribution, keep names, "no denigration" termination clause |
| PANNs weights | CC BY 4.0: attribution required |
| pyannote community-1 | CC-BY-4.0 (fine), but **gated**: users need an HF account, token and acceptance. We cannot bundle the download in a frictionless installer. |
| Qwen2.5-Omni-3B | May carry a different (research) licence than 7B (unverified) |
| PretrainedSED / EfficientAT weights | Repo is MIT but no explicit weight licence. Low risk, but cite and link. |
| AudioSet-trained models generally | Trained on YouTube audio; labels are CC BY 4.0. Industry-standard practice, no known restriction on weights. Note it in the README. |

Clean picks: PE-A-Frame (Apache-2.0), CED via HF (Apache-2.0), EAT / SSLAM (MIT), BEATs (MIT),
PaSST (Apache-2.0), AST (BSD-3), pyannote community-1 (CC-BY-4.0, gated), Sortformer (NVIDIA Open Model
License, commercial OK), Qwen2.5-Omni-7B / Qwen3-Omni / MiDashengLM / MOSS-Audio (Apache-2.0).

### 5.3 Maintenance

- MS-CLAP was archived (2026-09-15).
- BEATs, HTS-AT, PANNs and AST are frozen research releases.
- Active as of 2025–26: PE-A-Frame (Meta + transformers), pyannote (4.0.7, 2026-06), DiariZen
  (2026-01), EAT (HF integration 2025-05), M2D, Dasheng, NeMo / Sortformer.

---

## 6. Recommended stack

1. **SED now**: `facebook/pe-a-frame-base` via `transformers` (pin an exact v5.x), fp16 on GPU, CPU
   fallback with `pe-a-frame-small`.
   - Curated prompt list plus on-demand prompts.
   - Chunk audio (e.g. 30–60 s with overlap) and merge spans with hysteresis and minimum-duration rules.
   - Summarise as "[00:00–00:28] music; [01:05] laughter, applause".
   - Run sequentially after Whisper unloads.
   - **Gate the choice on the bake-off.** If threshold calibration disappoints, switch to the runner-up
     **PretrainedSED BEATs_strong** (vendored, MIT) or the fallback **CED-base windowed** (Apache-2.0 HF / ONNX).
2. **Diarization later**: pyannote community-1 (pyannote.audio 4.0.x).
   - Opt-in, behind an HF token, with exclusive mode, word-level max-overlap assignment and telemetry off.
   - Precision-2/3 can be documented as the paid upgrade.
3. **Emotion**: skip. Expose paralinguistic events from SED instead; reconsider only if a natural-speech,
   multilingual model shows above 0.5 macro-F1 without in-domain training.
4. **Optional future**: an on-demand ALM segment describer (Qwen2.5-Omni-7B-AWQ or MOSS-Audio-4B), only for
   specific ≤30-s segments Claude asks about.

---

## 7. Unverified items and open questions

- PE-A-Frame: input sample rate, max chunk length, real VRAM and RTF, the exact `transformers` version
  that introduced `PeAudioFrameLevelModel`, and the model-card code inconsistencies
  (`processor` vs `transform`, `text_embeds` vs `text_audio_embeds`).
- PretrainedSED: inference frame hop, `frame_mn06` / `frame_mn10` accuracy, weight licence.
- M2D LICENSE.pdf terms. Reverb licence terms. Qwen2.5-Omni-3B licence. Kimi-Audio licence and VRAM.
  MOSS-Audio VRAM. SenseVoice-Small parameter count. Sortformer gating.
- Native-Windows success of NeMo Sortformer.
- Third-party leaderboards (codesota, llm-stats, benchmarklist) disagree with each other and with papers.
  Only paper and model-card numbers are used for decisions.
- All throughput and VRAM numbers marked (est.) need measurement on the 5060 Ti.

---

## 8. Sources (accessed 2026-10-20; publication dates where known)

Sound events / tagging

- EfficientAT README (MIT; model table): https://github.com/fschmid56/EfficientAT
- EfficientAT windowed inference: https://raw.githubusercontent.com/fschmid56/EfficientAT/main/windowed_inference.py
- Schmid et al., Efficient Large-scale Audio Tagging via Transformer-to-CNN KD (2022): https://arxiv.org/abs/2211.04772
- PretrainedSED repo (ICASSP 2025): https://github.com/fschmid56/PretrainedSED
- Schmid et al., Effective Pre-Training of Audio Transformers for SED (arXiv 2409.09546, 2024-09): https://arxiv.org/abs/2409.09546
- CED repo (GPL-3.0 code): https://github.com/RicherMans/CED ; LICENSE: https://raw.githubusercontent.com/RicherMans/CED/main/LICENSE
- CED paper (ICASSP 2024): https://arxiv.org/abs/2308.11957
- CED-base HF card (Apache-2.0): https://huggingface.co/mispeech/ced-base
- Dasheng repo (Apache-2.0; AS-FT 49.7): https://github.com/RicherMans/Dasheng ; paper (Interspeech 2024): https://arxiv.org/abs/2406.06992
- BEATs (MIT, unilm): https://github.com/microsoft/unilm/tree/master/beats ; paper tables: https://arxiv.org/html/2212.09058v1
- EAT repo (MIT; HF integration 2025-05-03): https://github.com/cwx-worst-one/EAT ; paper (IJCAI 2024): https://arxiv.org/abs/2401.03497
- SSLAM repo (ICLR 2025): https://github.com/ta012/SSLAM ; HF: https://huggingface.co/ta012/SSLAM_AS2M_Finetuned ; paper: https://arxiv.org/abs/2506.12222
- PaSST (Apache-2.0): https://github.com/kkoutini/PaSST ; HEAR wrapper: https://github.com/kkoutini/passt_hear21
- M2D: https://github.com/nttcslab/m2d
- HTS-AT: https://github.com/RetroCirce/HTS-Audio-Transformer
- PANNs inference (MIT): https://github.com/qiuqiangkong/panns_inference ; weights (CC BY 4.0, 2019/2020): https://zenodo.org/records/3987831
- YAMNet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- AST HF card (BSD-3): https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
- OpenBEATs (2025-07): https://arxiv.org/abs/2507.14129
- AudioSet-R (2025-08): https://arxiv.org/abs/2508.15429
- PE-AV / PE-A-Frame paper (2025-12-22), SED table 13: https://arxiv.org/html/2512.19687v1
- PE-A-Frame cards: https://huggingface.co/facebook/pe-a-frame-base , https://huggingface.co/facebook/pe-a-frame-large ; collection: https://huggingface.co/collections/facebook/perception-encoder-audio-visual
- Meta SAM Audio / PE-AV blog (2025-12-16): https://ai.meta.com/blog/sam-audio/
- FlexSED (2025-09): https://arxiv.org/abs/2509.18606
- OpenFLAM (Adobe, non-commercial): https://github.com/adobe-research/openflam
- FireRedVAD (2026-03): https://github.com/FireRedTeam/FireRedVAD
- LAION-CLAP (CC0): https://github.com/LAION-AI/CLAP
- MS-CLAP (MIT, archived 2026-09-15): https://github.com/microsoft/CLAP
- SenseVoice repo: https://github.com/FunAudioLLM/SenseVoice ; HF: https://huggingface.co/FunAudioLLM/SenseVoiceSmall ; FunAudioLLM paper (2024-07): https://arxiv.org/html/2407.04051
- FunASR model licence v1.1: https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
- Third-party leaderboard (unverified, 2026-03): https://www.codesota.com/audio/classification

Audio-language models

- Qwen2.5-Omni (VRAM table, benchmarks, 2025-03..06): https://github.com/QwenLM/Qwen2.5-Omni
- Qwen3-Omni (2025-09-22; VRAM table): https://github.com/QwenLM/Qwen3-Omni ; Captioner: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner
- MiDashengLM-7B (Apache-2.0, 2025-08): https://huggingface.co/mispeech/midashenglm-7b-0804-fp32 ; paper: https://arxiv.org/abs/2508.03983
- MOSS-Audio (Apache-2.0, 2026-04-13): https://github.com/OpenMOSS/MOSS-Audio
- Audio Flamingo 3 (non-commercial): https://huggingface.co/nvidia/audio-flamingo-3-hf
- Nemotron 3 Nano Omni report (2026-04-27): https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Omni-report.pdf
- Inkling (2026-07-15): https://www.hpcwire.com/aiwire/2026/07/16/thinking-machines-launches-open-weight-inkling-foundation-model-for-fine-tuning/
- MMAU leaderboard aggregator (2026-10-20): https://llm-stats.com/benchmarks/mmau

Diarization

- pyannote community-1 card (benchmark 2025-09; CC-BY-4.0, gated): https://huggingface.co/pyannote/speaker-diarization-community-1
- pyannote.audio repo / speed table: https://github.com/pyannote/pyannote-audio ; pyproject: https://raw.githubusercontent.com/pyannote/pyannote-audio/main/pyproject.toml
- pyannote.audio PyPI (4.0.7, 2026-06-30): https://pypi.org/project/pyannote.audio/ ; telemetry (4.0.2 page): https://pypi.org/project/pyannote-audio/4.0.2/
- pyannoteAI changelog (precision-3 2026-09-17): https://www.pyannote.ai/changelog
- pyannoteAI "Top 7 diarization 2026" (vendor, 2026-10-20): https://www.pyannote.ai/blog/top-7-speaker-diarization-apis-and-models-in-2026
- DiariZen (2025-12 v2 benchmarks; CC BY-NC weights): https://github.com/BUTSpeechFIT/DiariZen
- Streaming Sortformer v2.1 card: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- Benchmarking Diarization Models (ETH, 2025-09-30): https://arxiv.org/abs/2509.26177
- 3D-Speaker: https://github.com/modelscope/3D-Speaker
- Reverb: https://github.com/revdotcom/reverb ; https://huggingface.co/Revai/reverb-diarization-v2
- SpeechBrain ECAPA: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- sherpa-onnx diarization: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html
- WhisperX: https://github.com/m-bain/whisperX ; pyproject: https://raw.githubusercontent.com/m-bain/whisperX/main/pyproject.toml ; aligner list: https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/alignment.py

Emotion

- emotion2vec repo: https://github.com/ddlBoJack/emotion2vec ; paper (2023-12): https://arxiv.org/html/2312.15185 ; plus-large card: https://huggingface.co/emotion2vec/emotion2vec_plus_large
- EmoBox (Interspeech 2024; includes TurEV-DB): https://arxiv.org/html/2406.07162v1
- VoxEmo (2026-03-09): https://arxiv.org/html/2603.08936
- MSP-Podcast challenge leaderboard (2025): https://lab-msp.com/MSP-Podcast_Competition/leaderboard.php ; Odyssey 2024 challenge paper: https://www.isca-archive.org/odyssey_2024/goncalves24_odyssey.html
- audeering dim model (CC BY-NC-SA): https://github.com/audeering/w2v2-how-to

Platform / compatibility

- PyTorch sm_120 support thread: https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099
- onnxruntime Blackwell fix commit (#26230): https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/11b23ad2dfd6cadcf7827acf4ced66cbd5da2e9d
- torchcodec PyPI: https://pypi.org/project/torchcodec/ ; Windows torchcodec issue thread: https://discuss.huggingface.co/t/cannot-load-torchcodec/169260
- FunASR Windows setup notes (third-party): https://git.nite07.com/nite/video-edit-planner-skill/raw/branch/main/references/windows-funasr-uv-setup.md

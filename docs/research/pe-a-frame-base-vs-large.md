# PE-A-Frame base vs large (research 2026-10-20)

> Research from 2026-10-20, before the sound map was built; published figures were current then.
> The measured speed, VRAM and calibration are in [DESIGN.md](../DESIGN.md#sound-map).

## Verified facts
### Official numbers - arXiv 2512.19687 (Dec 22 2025), Table 13 (read via arxiv.org/html/2512.19687; extracted by a summarizer, not eyeballed from PDF)
Columns: Internal AUROC | ASFX-SED AUROC | AudioSet-Strong PSDS1A/PSDS1T/AUROC | DESED PSDS1A/PSDS1T/AUROC | UrbanSED PSDS1A/PSDS1T/AUROC
- L: 0.91 | 0.83 | 0.43/0.61/0.96 | 0.34/0.58/0.97 | 0.12/0.22/0.89
- B: 0.92 | 0.83 | 0.42/0.60/0.96 | 0.39/0.56/0.98 | 0.12/0.21/0.89
- S: 0.91 | 0.83 | 0.39/0.59/0.96 | 0.32/0.54/0.96 | 0.09/0.19/0.88
- B from scratch: 0.89 | 0.76 | 0.22/0.55/0.89 | 0.10/0.52/0.89
Base's AudioSet-Strong IS published (paper Table 13). Base-large gap on AS-Strong: 0.01 PSDS1A, 0.01 PSDS1T, 0 AUROC. Base beats large on DESED PSDS1A and Internal AUROC.
- Frame rate 25 Hz (one embedding per 40 ms). Training data (Table 2): 1.2k h real + 12.8k h synthetic; 1.3M real / 4.6M synthetic events.
- Open-vocab baselines in paper: PretrainedSED, FLAM (3.2 Hz), FlexSED. No VGGSound/AVE/Epic-Sounds eval for PE-A-Frame found. Not found: separate video-domain or real-world-video eval, speed/VRAM figures in paper.
- Audio encoder params in paper (PE-AV, Table 3/11): S 0.09B, B 0.21B, L 1.11B (these are PE-AV audio towers; PE-A-Frame param counts are not given in the paper).

### Model facts (Hugging Face)
- Cards (huggingface.co/facebook/pe-a-frame-base): small 450M, base 560M, large 1.4B params. Apache 2.0. Collection "perception-encoder-audio-visual" 9 items; pe-a-frame-* dated Dec 23 2025.
- Files: base model.safetensors 2.24 GB, F32; large model.safetensors 5.83 GB (F32 implied, ~1.46B). Small not checked.
- preprocessor_config.json: sampling_rate 48000, hop_length 1920 (=25 Hz), mono. 
- Usage: PEAudioFrame (core.audio_visual_encoder) or transformers PeAudioFrameLevelModel + PeAudioProcessor; return_spans=True with `threshold` (card example 0.2; lower = more recall). Model card has no benchmarks, no speed/VRAM, no max length; the base card's examples load pe-a-frame-large (card inconsistency). Paper: 30 s clip = 750 frames; no explicit max length found.

### Third-party evidence
- Not found: no GitHub issues (perception_models issues search showed only unrelated/"About PE-AV" #113), HF discussions (only the welcome post), Reddit/X, or blogs benchmarking base vs large, or reporting Windows/Blackwell/transformers problems. Citing 2026 papers seen (SpotSound 2604.13023, 2609.15215, JASTIN 2605.04505) were not verified to benchmark sizes. MarkTechPost (2025-12-22) covers the release only.

### Newer releases
- Not found: no PE-A-Frame v2 or new sizes. Related: pe-av-{small,base,large} and *-16-frame updated Jan 29 (2026) in the same collection (HF collection page).

## Inferences
- Base is within noise of large on real-audio AudioSet-Strong (AudioSet = YouTube clips, the closest proxy to the target domain). Large is 2.6x bigger on disk and ~2.5x params; expect roughly proportionally slower/more VRAM (inference, unmeasured).
- Table values are rounded to 2 decimals, so differences of 0.01 are not meaningful; single run, no variance given.
- Threshold calibration will matter more than model size; no third-party calibration data exists.

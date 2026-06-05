# Qwen3-TTS Golden ASR A/B

## Changelog

- 2026-06-04 23:54:00 PDT, @codex-369-rv -- Documented the PR #393 unified-ORT fix: the all-in-one `golden-ab` executable now runs Sherpa, Moonshine, and Whisper together without the `free(): invalid pointer` abort by linking one workspace `ort-sys` static ONNX Runtime.
- 2026-06-04 22:22:00 PDT, @codex-369-rv -- Clarified that Moonshine is not worse than kroko on PM/orchestration WER after #393; the non-default live decision is due to CPU real-time factor/headroom and final-flush latency.
- 2026-06-04 22:15:00 PDT, @codex-369-rv -- Updated the matrix with PR #393 validation after the Moonshine rechunk fix plus #376 backend hardening: Moonshine now produces real transcripts, all 8 backend/codec cells populate with no skips, and PM kroko variance is documented as Qwen3-TTS WAV provenance drift.
- 2026-06-03 20:48:10 PDT, @codex-369-rv -- Replaced the contaminated pre-pad kroko matrix with the validated post-#378 padded call-center results and recorded the balanced default policy: `kroko-2025` for mixed call-center plus PM/technical use, `sherpa-2023` for call-center-only deployments.
- 2026-06-02 14:24:49 PDT, @codex-191-impl -- Added the offline Qwen3-TTS call-center golden corpus workflow and recorded the first full DGX run for #371/#191.

## Purpose

This harness produces reproducible offline ASR selection data without a live Telnyx call. It synthesizes short call-center utterances with Qwen3-TTS, round-trips each WAV through Telnyx-compatible audio specs, then runs every selected ASR backend through the gateway `StreamingTranscriber`/session contract.

Ground truth is the corpus `text` field in `bins/telnyx-gateway/corpus/qwen3-call-center-golden.json`. Digit strings are written as spoken words to make WER compare the intended speech instead of punctuation or numeric formatting.

## Commands

Generate the 16 kHz mono source WAVs:

```bash
cargo run -p motlie-telnyx-gateway --features golden-ab --   golden-tts bins/telnyx-gateway/corpus/qwen3-call-center-golden.json   --output-dir /tmp/motlie-qwen3-call-center-golden --force
```

Run the full matrix and write JSON:

```bash
cargo run -p motlie-telnyx-gateway --features golden-ab --   asr-golden-ab bins/telnyx-gateway/corpus/qwen3-call-center-golden.json   --audio-dir /tmp/motlie-qwen3-call-center-golden   --output-json /tmp/motlie-asr-golden-ab-full.json
```

The harness materializes codec-specific audit WAVs under `/tmp/motlie-qwen3-call-center-golden/asr-inputs/`.

## PR #393 Validation Run

Run on amd1 on 2026-06-04 PDT after retargeting the Moonshine rechunk fix and #376 backend hardening to `feature/telnyx-voice`, then applying the unified-ORT native-link fix for PR #393 / #396.

The validation generated local Qwen3-TTS WAVs for both 72-sample corpora:

- `/tmp/motlie-qwen3-call-center-golden`
- `/tmp/motlie-pm-golden`

The matrix used `chunk_ms=20`, `trailing_silence_pad_ms=800`, and both `L16-16k` and `PCMU-8k` codec round-trips. The final validation ran the all-in-one `golden-ab` executable with Sherpa, Moonshine, Whisper, and Qwen linked together. It no longer requires per-backend feature binaries: PR #393 patches `sherpa-onnx-sys` so upstream Sherpa keeps `OnlineRecognizer` but does not link its bundled `libonnxruntime.a`; the workspace `ort-sys` archive supplies the single static ONNX Runtime for Sherpa and Moonshine. Each corpus produced 576 entries, all 8 backend/codec cells populated, and no skipped cells or `free(): invalid pointer` abort.

### Call-Center Corpus

Total reference words: 626.

| Backend | Codec | Agg WER | Errors / Words | Avg wall latency | Selection signal |
|---|---:|---:|---:|---:|---|
| sherpa-zipformer-en-2023-06-26 | L16-16k | 11.2% | 70 / 626 | ~1163 ms | Best call-center-only backend |
| sherpa-zipformer-en-2023-06-26 | PCMU-8k | 11.3% | 71 / 626 | ~1162 ms | Best call-center-only backend |
| sherpa-zipformer-en-kroko-2025-08-06 | L16-16k | 15.2% | 95 / 626 | ~1034 ms | Balanced live default candidate |
| sherpa-zipformer-en-kroko-2025-08-06 | PCMU-8k | 15.2% | 95 / 626 | ~1034 ms | Balanced live default candidate |
| moonshine-streaming-en | L16-16k | 31.2% | 195 / 626 | ~5018 ms | Valid after #393; CPU headroom is the blocker |
| moonshine-streaming-en | PCMU-8k | 30.2% | 189 / 626 | ~4955 ms | Valid after #393; CPU headroom is the blocker |
| whisper-base-en | L16-16k | 29.7% | 186 / 626 | ~2321 ms | Batch/final-pass only; weak on digit-heavy categories |
| whisper-base-en | PCMU-8k | 30.2% | 189 / 626 | ~2322 ms | Batch/final-pass only; weak on digit-heavy categories |

### PM / Orchestration Corpus

Fresh PR #393 Qwen3-TTS WAVs, total reference words: 556.

| Backend | Codec | Agg WER | Errors / Words | Avg wall latency | Selection signal |
|---|---:|---:|---:|---:|---|
| sherpa-zipformer-en-2023-06-26 | L16-16k | 18.9% | 105 / 556 | ~1075 ms | Call-center-only profile |
| sherpa-zipformer-en-2023-06-26 | PCMU-8k | 19.4% | 108 / 556 | ~1076 ms | Call-center-only profile |
| sherpa-zipformer-en-kroko-2025-08-06 | L16-16k | 15.8% | 88 / 556 | ~956 ms | Balanced live default candidate |
| sherpa-zipformer-en-kroko-2025-08-06 | PCMU-8k | 15.6% | 87 / 556 | ~954 ms | Balanced live default candidate |
| moonshine-streaming-en | L16-16k | 15.8% | 88 / 556 | ~4032 ms | PM WER is competitive with kroko; CPU headroom is the blocker |
| moonshine-streaming-en | PCMU-8k | 14.2% | 79 / 556 | ~4014 ms | PM WER is competitive with kroko; CPU headroom is the blocker |
| whisper-base-en | L16-16k | 16.5% | 92 / 556 | ~2319 ms | Batch/final-pass only |
| whisper-base-en | PCMU-8k | 16.5% | 92 / 556 | ~2322 ms | Batch/final-pass only |

PM/orchestration `kroko-2025` is sensitive to regenerated Qwen3-TTS WAV provenance. On the fresh PR #393 PM WAVs it measures `15.8% / 15.8%`; on the earlier local PM WAV directory from the previous validation, the same PR #393 head measures `14.0% / 14.0%` with the same manifest, chunking, and trailing-silence pad. Treat this as audio-provenance variance, not a code-path regression.

Moonshine is no longer the old canned-output failure after PR #393. It now produces real transcripts through the same replay harness, and on PM/orchestration it is not worse than kroko on WER (`15.8% / 14.2%` versus kroko's `15.8% / 15.6%` on fresh PR #393 WAVs). The live-default concern is CPU streaming headroom: kroko processes average PM samples in about `0.32x` real time, while Moonshine runs around `1.34x` real time plus a `~257 ms` final flush; on call-center audio Moonshine is `1.49-1.51x` real time while kroko is about `0.31x`. Moonshine therefore remains an offline/final-pass or future-optimization candidate rather than the live gateway default.

## Previous Valid Padded DGX Run

Run on dgx1 on 2026-06-02 PDT after PR #378 added an 800 ms trailing-silence pad before `finish()`. The earlier first matrix was contaminated by too-short golden WAV tails; it starved streaming zipformer final-token flush and inflated kroko-2025 WER from about `14%` to about `38%`. The padded run is the valid selection data.

Full matrix size: 72 samples x 2 codecs x 4 ASR backends = 576 ASR runs. Total reference words: 626.

| Backend | Codec | Agg WER | Errors / Words | Median wall latency | Selection signal |
|---|---:|---:|---:|---:|---|
| sherpa-zipformer-en-2023-06-26 | L16-16k | 10.5% | 66 / 626 | ~680 ms | Best call-center-only backend |
| sherpa-zipformer-en-2023-06-26 | PCMU-8k | 12.0% | 75 / 626 | ~680 ms | Best call-center-only backend |
| sherpa-zipformer-en-kroko-2025-08-06 | L16-16k | 14.1% | 88 / 626 | ~583 ms | Balanced live default candidate |
| sherpa-zipformer-en-kroko-2025-08-06 | PCMU-8k | 13.9% | 87 / 626 | ~583 ms | Balanced live default candidate |
| whisper-base-en | L16-16k | 29.7% | 186 / 626 | ~2188 ms | Batch/final-pass only; weak on digit-heavy categories |
| whisper-base-en | PCMU-8k | 29.9% | 187 / 626 | ~2188 ms | Batch/final-pass only; weak on digit-heavy categories |
| moonshine-streaming-en | L16-16k | ~99.8% | 625 / 626 | ~5500 ms | Disqualified for this harness |
| moonshine-streaming-en | PCMU-8k | ~100.0% | 626 / 626 | ~5500 ms | Disqualified for this harness |

Observed call-center signal: `sherpa-2023` is the best call-center recognizer (`10.5% / 12.0%` on `L16-16k / PCMU-8k`), but `kroko-2025` is still below target (`14.1% / 13.9%`) and stronger on digit categories. The PM/orchestration corpus reorders the live default decision: `kroko-2025` measures `14.0% / 14.0%`, while `sherpa-2023` measures `19.8% / 22.1%`. The Telnyx gateway therefore defaults to `kroko-2025` as the balanced cross-domain profile, and operators can select `sherpa-2023` with `asr use sherpa-2023` for call-center-only deployments.

The padded call-center JSON report for this run was written to `/tmp/motlie-asr-golden-ab-padded.json`; the PM/orchestration report was written to `/tmp/motlie-pm-ab.json`.

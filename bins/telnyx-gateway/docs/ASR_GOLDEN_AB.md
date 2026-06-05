# Qwen3-TTS Golden ASR A/B

## Changelog

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

Run on amd1 on 2026-06-04 PDT at PR #393 head `ae61c9dbe90e223d2d559aa9af65977ff749fc32` after retargeting the Moonshine rechunk fix and #376 backend hardening to `feature/telnyx-voice`.

The validation generated local Qwen3-TTS WAVs for both 72-sample corpora:

- `/tmp/motlie-qwen3-call-center-golden`
- `/tmp/motlie-pm-golden`

The matrix used `chunk_ms=20`, `trailing_silence_pad_ms=800`, and both `L16-16k` and `PCMU-8k` codec round-trips. Because the all-in-one `golden-ab` executable aborts with `free(): invalid pointer` when Sherpa and Moonshine are linked into the same process on this host, the final validation ran per-backend feature binaries (`sherpa`, `moonshine`, `whisper`). The resulting matrix has 576 entries and all 8 backend/codec cells populated for each corpus; no cells are skipped.

### Call-Center Corpus

Total reference words: 626.

| Backend | Codec | Agg WER | Errors / Words | Avg wall latency | Selection signal |
|---|---:|---:|---:|---:|---|
| sherpa-zipformer-en-2023-06-26 | L16-16k | 11.0% | 69 / 626 | ~1208 ms | Best call-center-only backend |
| sherpa-zipformer-en-2023-06-26 | PCMU-8k | 11.2% | 70 / 626 | ~1206 ms | Best call-center-only backend |
| sherpa-zipformer-en-kroko-2025-08-06 | L16-16k | 15.3% | 96 / 626 | ~1042 ms | Balanced live default candidate |
| sherpa-zipformer-en-kroko-2025-08-06 | PCMU-8k | 15.0% | 94 / 626 | ~1041 ms | Balanced live default candidate |
| moonshine-streaming-en | L16-16k | 31.2% | 195 / 626 | ~5155 ms | Valid after #393, but too slow for live telephony |
| moonshine-streaming-en | PCMU-8k | 30.2% | 189 / 626 | ~5088 ms | Valid after #393, but too slow for live telephony |
| whisper-base-en | L16-16k | 29.7% | 186 / 626 | ~2374 ms | Batch/final-pass only; weak on digit-heavy categories |
| whisper-base-en | PCMU-8k | 30.2% | 189 / 626 | ~2375 ms | Batch/final-pass only; weak on digit-heavy categories |

### PM / Orchestration Corpus

Fresh PR #393 Qwen3-TTS WAVs, total reference words: 556.

| Backend | Codec | Agg WER | Errors / Words | Avg wall latency | Selection signal |
|---|---:|---:|---:|---:|---|
| sherpa-zipformer-en-2023-06-26 | L16-16k | 19.1% | 106 / 556 | ~1106 ms | Call-center-only profile |
| sherpa-zipformer-en-2023-06-26 | PCMU-8k | 19.4% | 108 / 556 | ~1106 ms | Call-center-only profile |
| sherpa-zipformer-en-kroko-2025-08-06 | L16-16k | 15.8% | 88 / 556 | ~956 ms | Balanced live default candidate |
| sherpa-zipformer-en-kroko-2025-08-06 | PCMU-8k | 15.8% | 88 / 556 | ~956 ms | Balanced live default candidate |
| moonshine-streaming-en | L16-16k | 15.8% | 88 / 556 | ~4120 ms | Valid after #393, but too slow for live telephony |
| moonshine-streaming-en | PCMU-8k | 14.2% | 79 / 556 | ~4102 ms | Valid after #393, but too slow for live telephony |
| whisper-base-en | L16-16k | 16.5% | 92 / 556 | ~2358 ms | Batch/final-pass only |
| whisper-base-en | PCMU-8k | 16.5% | 92 / 556 | ~2358 ms | Batch/final-pass only |

PM/orchestration `kroko-2025` is sensitive to regenerated Qwen3-TTS WAV provenance. On the fresh PR #393 PM WAVs it measures `15.8% / 15.8%`; on the earlier local PM WAV directory from the previous validation, the same PR #393 head measures `14.0% / 14.0%` with the same manifest, chunking, and trailing-silence pad. Treat this as audio-provenance variance, not a code-path regression.

Moonshine is no longer the old canned-output failure after PR #393. It now produces real transcripts through the same replay harness, but its wall latency remains several seconds per sample, so it remains an offline/final-pass or future-optimization candidate rather than the live gateway default.

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

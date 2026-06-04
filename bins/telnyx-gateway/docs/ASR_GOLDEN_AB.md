# Qwen3-TTS Golden ASR A/B

## Changelog

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

## Valid Padded DGX Run

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

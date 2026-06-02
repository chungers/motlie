# Qwen3-TTS Golden ASR A/B

## Changelog

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

## First DGX Run

Run by @codex-191-impl on 2026-06-02 PDT. Qwen3-TTS used CPU backend on this run; it loaded the Q8_0 model and tokenizer from `.agents/skills/voice/artifacts/hf-cache`, used about 1.9 GB RSS, and generated 72 source WAVs.

Full matrix size: 72 samples x 2 codecs x 4 ASR backends = 576 ASR runs. Total reference words: 626.

| Backend | Codec | WER | Errors / Words | Ingest Avg ms | Finish Avg ms | Wall Avg ms |
|---|---:|---:|---:|---:|---:|---:|
| sherpa-zipformer-en-2023-06-26 | L16-16k | 24.3% | 152 / 626 | 2.9 | 4.1 | 559.8 |
| sherpa-zipformer-en-2023-06-26 | PCMU-8k | 25.9% | 162 / 626 | 2.9 | 4.0 | 559.8 |
| whisper-base-en | L16-16k | 29.7% | 186 / 626 | 0.0 | 2187.2 | 2187.3 |
| whisper-base-en | PCMU-8k | 29.9% | 187 / 626 | 0.0 | 2186.5 | 2186.6 |
| sherpa-zipformer-en-kroko-2025-08-06 | L16-16k | 38.2% | 239 / 626 | 2.4 | 2.1 | 468.3 |
| sherpa-zipformer-en-kroko-2025-08-06 | PCMU-8k | 38.0% | 238 / 626 | 2.4 | 2.1 | 468.1 |
| moonshine-streaming-en | L16-16k | 99.7% | 624 / 626 | 28.0 | 36.7 | 4618.8 |
| moonshine-streaming-en | PCMU-8k | 99.7% | 624 / 626 | 27.2 | 36.2 | 4484.3 |

Observed selection signal: Sherpa 2023 is the best overall live-style recognizer in this corpus; Whisper base.en is competitive on greetings, yes/no, short commands, short questions, and spelled names, but weak on digit-heavy categories. PCMU 8 kHz changed overall WER only slightly in this synthetic set, but category breakdown remains the right place to evaluate narrowband robustness.

The full JSON report for this run was written to `/tmp/motlie-asr-golden-ab-full.json`.

# Eval Matrix Run

- snapshot: `curated-v2-smoke`
- profile: `local-cpu-x86_64`
- launched cells: `4`
- pre-run records: `29`

## Run context (@486-rv-speech 2026-06-12 00:43 PDT)

- **AUTHORITATIVE warmed audio run** (issue #486 / #490 methodology, post-#491): build SHA (pin) `874c9f6956d127a0e5fdeb72fd08b99ab9a18a88` = identity.git_sha in every record; schema v5.
- host: amd2 (`amd-ryzen7-2`), AMD Ryzen 7 7730U, x86_64, profile `local-cpu-x86_64`; HF_TOKEN_PRESENT=false; release matrix children; warmup_iterations=1, iterations=3 (curated scenario config).
- outcome mix: 4 passed / 29 blocked (19 `hf_token_missing`, 10 `artifact_missing` — scope blocks as in the baseline run) / 0 failed.
- Warmed audio metrics (mean over 3 measured iterations after 1 discarded warm-up; raw samples in records):
  - `whisper_base_en` (batch): mean_transcription 1288.3ms, warmup_ms 1299; `ttfp` honest null + batch gap entry
  - `sherpa_onnx_streaming_zipformer_en`: mean_ttfp **109.7ms** (samples [108,110,111]), mean_transcription 515.0ms, warmup_ms 510
  - `moonshine_streaming_en`: mean_ttfp **58.0ms** (samples [56,59,59]), mean_transcription 3094.3ms, warmup_ms 3237
  - `piper_en_us_ljspeech_medium`: mean_ttfa **70.7ms** (samples [72,75,65]), mean_synthesis 70.7ms, warmup_ms 131
- **Cold(v4 baseline, PR #489) vs warm(v5, this run) on identical hardware+pin-lineage (both release children):**
  | cell | cold single-shot | warm mean | delta |
  |---|---:|---:|---|
  | sherpa ttfp | 110 | 109.7 | ~0 — no first-call premium on CPU ASR first-partial |
  | moonshine ttfp | 61 | 58.0 | -5% — marginal |
  | piper ttfa | 183 | 70.7 | **-61%** — ORT first-call premium was dominating TTS first-chunk |
  | whisper latency | 1291 | 1288.3 | ~0 — ggml has no lazy-init premium |
  The #490 thesis is confirmed where it was predicted (ORT TTS first-call) and bounded where it does not apply (CPU ASR streaming): both numbers are real, and the v4 baseline remains the cold-start record.

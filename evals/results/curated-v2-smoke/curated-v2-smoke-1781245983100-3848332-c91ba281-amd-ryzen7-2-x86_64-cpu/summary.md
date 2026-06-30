# Eval Matrix Run

- snapshot: `curated-v2-smoke`
- profile: `local-cpu-x86_64`
- launched cells: `4`
- pre-run records: `29`

## Run context (@486-rv-speech 2026-06-11 23:36 PDT)

- build SHA (pin): `c91ba28108fe589ad75482d66e1ce906c52e8063` (= identity.git_sha in every record)
- host: amd2 (`amd-ryzen7-2`), AMD Ryzen 7 7730U, x86_64, profile `local-cpu-x86_64`; HF_TOKEN_PRESENT=false
- outcome mix: 4 passed / 29 blocked (19 `hf_token_missing` gated text models, 10 `artifact_missing` uncached non-audio + qwen3-tts-cpp) / 0 failed
- directed scope (issue #486 run phase): audio first-latency cells. All four audio cells PASSED with the new schema-v4 fields:
  - `whisper_base_en` asr: `ttfp_first_partial_ms=null` + gap `metric_not_applicable_for_batch_engine` (honest batch null), latency 1291ms
  - `sherpa_onnx_streaming_zipformer_en` asr: `ttfp_first_partial_ms=110`, latency 507ms
  - `moonshine_streaming_en` asr: `ttfp_first_partial_ms=61`, latency 3087ms
  - `piper_en_us_ljspeech_medium` tts: `ttfa_first_chunk_ms=183`, synthesis 418ms
- blocked cells are scope blocks (no token / artifacts deliberately not prefetched on this host), not model failures; no #435 sub-issues filed.

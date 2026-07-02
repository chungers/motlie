# Curated Eval Coverage Report

- input files: `3`
- records: `37`
- overall: `blocked`

## Outcome Summary

| outcome | count |
|---|---:|
| `blocked` | 10 |
| `passed` | 27 |

## Platform Notes

- No platform-specific caveats detected in the input records.

## LLM Accelerator Comparison

Decode throughput (tok/s) and TTFT (ms) per LLM bundle and quantization scheme, by target accelerator. Values are from passing `perf` cells; `—` = no passing perf metric for that pairing.

| bundle | quant | family | cpu tok/s | cpu ttft ms | cuda tok/s | cuda ttft ms | metal tok/s | metal ttft ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `gemma4_e2b` | `isq_q4` | mistralrs/HF | 10.4 | 16507 | — | — | — | — |
| `gemma4_e2b` | `isq_q8` | mistralrs/HF | 9.6 | 20755 | — | — | — | — |
| `gemma4_e4b` | `isq_q4` | mistralrs/HF | 7.3 | 26774 | — | — | — | — |
| `gemma4_e4b` | `isq_q8` | mistralrs/HF | 6.3 | 41900 | — | — | — | — |
| `ornith_1_0_35b_gguf` | `gguf_q4_k_m` | llama.cpp/GGUF | 19.7 | 12748 | 66.6 | 389 | — | — |
| `ornith_1_0_35b_gguf` | `gguf_q8_0` | llama.cpp/GGUF | 16.7 | 12315 | 46.4 | 436 | — | — |
| `qwen3_4b` | `isq_q4` | mistralrs/HF | 14.4 | 21227 | — | — | — | — |
| `qwen3_4b` | `isq_q8` | mistralrs/HF | 11.2 | 34934 | — | — | — | — |
| `qwen3_4b_gguf` | `gguf_q4_k_m` | llama.cpp/GGUF | 17.5 | 16836 | — | — | — | — |

### Backend-family viability

`on_target` = passed cells that actually resolved to the requested accelerator (a passed cell that silently fell back to CPU is counted in `passed` but not `on_target`). `mean decode tok/s` is averaged over on-target passing `perf` cells.

| family | accelerator | cells | passed | on_target | blocked | failed | mean decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama.cpp/GGUF | `cpu` | 10 | 10 | 10 | 0 | 0 | 18.0 |
| llama.cpp/GGUF | `cuda` | 6 | 6 | 6 | 0 | 0 | 56.5 |
| mistralrs/HF | `cpu` | 15 | 6 | 6 | 9 | 0 | 9.9 |

### Build provenance

Distinct build SHAs (`identity.git_sha`) of the on-target passing records backing the numbers above, per accelerator. An accelerator whose SHA set differs from the others is **prior-pin** data (pin mismatch — confirmatory only, not a fresh re-run).

| accelerator | build SHAs |
|---|---|
| `cpu` | `5a990b58`, `8079ae7b` |
| `cuda` | `8079ae7b` |
| `metal` | — |

## Coverage Accounting Matrix

4-state coverage per `(bundle × capability)` × Profile, reconciled from the records against the `BackendKind::accel_support` declaration. ✅ Validated · ⛔ NotApplicable(reason) · 🔧 BuildGap · ⏳ Gap · — none.

| bundle | quant | capability | local-cpu-x86_64 | local-cpu-aarch64 | apple-metal | dgx-spark | cuda-workstation |
|---|---|---|---|---|---|---|---|
| `embeddinggemma_300m` | `fp32` | `embeddings` | — | ✅ | — | — | — |
| `gemma4_e2b` | `isq_q4` | `chat` | — | ⏳ | — | — | — |
| `gemma4_e2b` | `bf16` | `tool_use` | — | ⏳ | — | — | — |
| `gemma4_e4b` | `isq_q4` | `chat` | — | ⏳ | — | — | — |
| `gemma4_e4b` | `bf16` | `tool_use` | — | ⏳ | — | — | — |
| `ornith_1_0_35b_gguf` | `gguf_q4_k_m` | `chat` | — | ✅ | — | ✅ | — |
| `ornith_1_0_35b_gguf` | `gguf_q4_k_m` | `completion` | — | ✅ | — | ✅ | — |
| `ornith_1_0_35b_gguf` | `gguf_q4_k_m` | `tool_use` | — | ✅ | — | ✅ | — |
| `qwen3_4b` | `isq_q4` | `chat` | — | ⏳ | — | — | — |
| `qwen3_4b` | `bf16` | `tool_use` | — | ⏳ | — | — | — |
| `qwen3_4b_gguf` | `gguf_q4_k_m` | `chat` | — | ✅ | — | — | — |
| `qwen3_4b_gguf` | `gguf_q4_k_m` | `tool_use` | — | ✅ | — | — | — |
| `qwen3_6_27b_gguf` | `gguf_q4_k_m` | `chat` | — | ✅ | — | — | — |
| `qwen3_embedding_06b` | `bf16` | `embeddings` | — | ⏳ | — | — | — |

## Per-Cell Coverage

| cell | host | arch | run | bundle | capability | depth | profile | requested | resolved | outcome | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `embeddinggemma_300m__embeddings_similarity__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `embeddinggemma_300m` | `embeddings` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_embedding_06b__embeddings_similarity__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `qwen3_embedding_06b` | `embeddings` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `child_run_failed` |
| `qwen3_4b__chat_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `qwen3_4b` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `child_run_failed` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `qwen3_4b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `child_run_failed` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `qwen3_4b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `qwen3_4b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `qwen3_4b` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `child_run_failed` |
| `gemma4_e2b__chat_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `gemma4_e2b` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `child_run_failed` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `gemma4_e2b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `child_run_failed` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `gemma4_e2b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `gemma4_e2b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `gemma4_e2b` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `child_run_failed` |
| `gemma4_e4b__chat_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `gemma4_e4b` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `child_run_failed` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `gemma4_e4b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `child_run_failed` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `gemma4_e4b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `gemma4_e4b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `gemma4_e4b` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `child_run_failed` |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `qwen3_4b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `qwen3_4b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `qwen3_4b_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` | `qwen3_6_27b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782970098195-439707-8079ae7b-spark-2f6e-aarch64-cpu` | `ornith_1_0_35b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782970098195-439707-8079ae7b-spark-2f6e-aarch64-cpu` | `ornith_1_0_35b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `ornith_1_0_35b_gguf__chat_completion_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782970098195-439707-8079ae7b-spark-2f6e-aarch64-cpu` | `ornith_1_0_35b_gguf` | `completion` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `ornith_1_0_35b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782970098195-439707-8079ae7b-spark-2f6e-aarch64-cpu` | `ornith_1_0_35b_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782970098195-439707-8079ae7b-spark-2f6e-aarch64-cpu` | `ornith_1_0_35b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782970098195-439707-8079ae7b-spark-2f6e-aarch64-cpu` | `ornith_1_0_35b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `ornith_1_0_35b_gguf__chat_completion_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782970098195-439707-8079ae7b-spark-2f6e-aarch64-cpu` | `ornith_1_0_35b_gguf` | `completion` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `ornith_1_0_35b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782970098195-439707-8079ae7b-spark-2f6e-aarch64-cpu` | `ornith_1_0_35b_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782971578478-486133-8079ae7b-spark-2f6e-aarch64-cuda` | `ornith_1_0_35b_gguf` | `chat` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782971578478-486133-8079ae7b-spark-2f6e-aarch64-cuda` | `ornith_1_0_35b_gguf` | `perf` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `ornith_1_0_35b_gguf__chat_completion_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782971578478-486133-8079ae7b-spark-2f6e-aarch64-cuda` | `ornith_1_0_35b_gguf` | `completion` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `ornith_1_0_35b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782971578478-486133-8079ae7b-spark-2f6e-aarch64-cuda` | `ornith_1_0_35b_gguf` | `tool_use` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782971578478-486133-8079ae7b-spark-2f6e-aarch64-cuda` | `ornith_1_0_35b_gguf` | `chat` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782971578478-486133-8079ae7b-spark-2f6e-aarch64-cuda` | `ornith_1_0_35b_gguf` | `perf` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `ornith_1_0_35b_gguf__chat_completion_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782971578478-486133-8079ae7b-spark-2f6e-aarch64-cuda` | `ornith_1_0_35b_gguf` | `completion` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `ornith_1_0_35b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1782971578478-486133-8079ae7b-spark-2f6e-aarch64-cuda` | `ornith_1_0_35b_gguf` | `tool_use` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |

## Latency Metrics

Streaming TTS note: `first_pcm_before_synth_complete` is meaningful as a streaming-benefit signal only for longer inputs. Short and medium Kokoro inputs around the current ~166-374 character segment-emission threshold may synthesize as a single segment, so proof=true there is bookkeeping; use `max_inter_chunk_gap_ms` and `underrun_count` as the breakdown signals.

| cell | host | capability | ttft_first_token_ms | ttft_first_answer_token_ms | thinking_tokens_to_answer | completion_tokens | mean_ttft_first_token_ms | p95_ttft_first_token_ms | mean_ttft_first_answer_token_ms | p95_ttft_first_answer_token_ms | mean_transcription_latency_ms | p95_transcription_latency_ms | mean_ttfp_first_partial_ms | p95_ttfp_first_partial_ms | mean_synthesis_latency_ms | p95_synthesis_latency_ms | mean_ttfa_first_chunk_ms | p95_ttfa_first_chunk_ms | mean_synth_complete_ms | p95_synth_complete_ms | mean_inter_chunk_gap_ms | p95_inter_chunk_gap_ms | max_inter_chunk_gap_ms | underrun_count | streaming_frame_ms | packetized_frame_count | ttfp_first_partial_ms | ttfa_first_chunk_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `spark-2f6e` | `perf` | null | null | null | null | 21227.00 | 21318.00 | 21227.00 | 21318.00 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `spark-2f6e` | `perf` | null | null | null | null | 34934.33 | 35375.00 | 34934.33 | 35375.00 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `spark-2f6e` | `perf` | null | null | null | null | 16507.33 | 16528.00 | 16507.33 | 16528.00 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `spark-2f6e` | `perf` | null | null | null | null | 20754.67 | 20795.00 | 20754.67 | 20795.00 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `spark-2f6e` | `perf` | null | null | null | null | 26774.00 | 26954.00 | 26774.00 | 26954.00 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `spark-2f6e` | `perf` | null | null | null | null | 41900.33 | 42001.00 | 41900.33 | 42001.00 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `chat` | 1291 | 14636 | 273 | 299 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `perf` | null | null | null | null | 16836.00 | 16851.00 | 35487.00 | 36303.00 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `chat` | 10100 | 638814 | 1954 | 1985 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `chat` | 1080 | 66983 | 1270 | 1299 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `perf` | null | null | null | null | 12747.67 | 12788.00 | 88608.33 | 91332.00 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `chat` | 1102 | 70530 | 1171 | 1213 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q8_0` | `spark-2f6e` | `perf` | null | null | null | null | 12314.67 | 12372.00 | 79608.00 | 87329.00 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `chat` | 335 | 30801 | 1987 | 2049 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `perf` | null | null | null | null | 388.67 | 393.00 | 12573.67 | 12603.00 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `chat` | 397 | 22813 | 1029 | 1069 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q8_0` | `spark-2f6e` | `perf` | null | null | null | null | 436.00 | 442.00 | 36737.67 | 36964.00 | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null | null |

## Model x Capability

| model_family | capability | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `gemma` | `chat` | 0 | 0 | 2 | 0 |
| `gemma` | `embeddings` | 1 | 0 | 0 | 0 |
| `gemma` | `perf` | 4 | 0 | 2 | 0 |
| `gemma` | `tool_use` | 0 | 0 | 2 | 0 |
| `ornith` | `chat` | 4 | 0 | 0 | 0 |
| `ornith` | `completion` | 4 | 0 | 0 | 0 |
| `ornith` | `perf` | 4 | 0 | 0 | 0 |
| `ornith` | `tool_use` | 4 | 0 | 0 | 0 |
| `qwen3` | `chat` | 2 | 0 | 1 | 0 |
| `qwen3` | `embeddings` | 0 | 0 | 1 | 0 |
| `qwen3` | `perf` | 3 | 0 | 1 | 0 |
| `qwen3` | `tool_use` | 1 | 0 | 1 | 0 |

## Capability x Profile

| capability | profile | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `chat` | `dgx-spark` | 2 | 0 | 0 | 0 |
| `chat` | `local-cpu-aarch64` | 4 | 0 | 3 | 0 |
| `completion` | `dgx-spark` | 2 | 0 | 0 | 0 |
| `completion` | `local-cpu-aarch64` | 2 | 0 | 0 | 0 |
| `embeddings` | `local-cpu-aarch64` | 1 | 0 | 1 | 0 |
| `perf` | `dgx-spark` | 2 | 0 | 0 | 0 |
| `perf` | `local-cpu-aarch64` | 9 | 0 | 3 | 0 |
| `tool_use` | `dgx-spark` | 2 | 0 | 0 | 0 |
| `tool_use` | `local-cpu-aarch64` | 3 | 0 | 3 | 0 |

## Capability x Depth

| capability | depth | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `chat` | `smoke` | 6 | 0 | 3 | 0 |
| `completion` | `smoke` | 4 | 0 | 0 | 0 |
| `embeddings` | `smoke` | 1 | 0 | 1 | 0 |
| `perf` | `smoke` | 11 | 0 | 3 | 0 |
| `tool_use` | `smoke` | 5 | 0 | 3 | 0 |

## Backend x Profile

| backend | profile | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `llama_cpp` | `dgx-spark` | 8 | 0 | 0 | 0 |
| `llama_cpp` | `local-cpu-aarch64` | 12 | 0 | 0 | 0 |
| `mistralrs` | `local-cpu-aarch64` | 7 | 0 | 10 | 0 |

## Model x Quantization x Backend/Profile/Depth

| model | quantization | backend | profile | depth | passed | failed | blocked | skipped |
|---|---|---|---|---|---:|---:|---:|---:|
| `gemma` | `bf16` | `mistralrs` | `local-cpu-aarch64` | `smoke` | 0 | 0 | 6 | 0 |
| `gemma` | `fp32` | `mistralrs` | `local-cpu-aarch64` | `smoke` | 1 | 0 | 0 | 0 |
| `gemma` | `isq_q4` | `mistralrs` | `local-cpu-aarch64` | `smoke` | 2 | 0 | 0 | 0 |
| `gemma` | `isq_q8` | `mistralrs` | `local-cpu-aarch64` | `smoke` | 2 | 0 | 0 | 0 |
| `ornith` | `gguf_q4_k_m` | `llama_cpp` | `dgx-spark` | `smoke` | 4 | 0 | 0 | 0 |
| `ornith` | `gguf_q4_k_m` | `llama_cpp` | `local-cpu-aarch64` | `smoke` | 4 | 0 | 0 | 0 |
| `ornith` | `gguf_q8_0` | `llama_cpp` | `dgx-spark` | `smoke` | 4 | 0 | 0 | 0 |
| `ornith` | `gguf_q8_0` | `llama_cpp` | `local-cpu-aarch64` | `smoke` | 4 | 0 | 0 | 0 |
| `qwen3` | `bf16` | `mistralrs` | `local-cpu-aarch64` | `smoke` | 0 | 0 | 4 | 0 |
| `qwen3` | `gguf_q4_k_m` | `llama_cpp` | `local-cpu-aarch64` | `smoke` | 4 | 0 | 0 | 0 |
| `qwen3` | `isq_q4` | `mistralrs` | `local-cpu-aarch64` | `smoke` | 1 | 0 | 0 | 0 |
| `qwen3` | `isq_q8` | `mistralrs` | `local-cpu-aarch64` | `smoke` | 1 | 0 | 0 | 0 |

## Requested x Resolved Accelerator

| requested | resolved | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `cpu` | `cpu` | 19 | 0 | 10 | 0 |
| `cuda` | `cuda` | 8 | 0 | 0 | 0 |

## Blocker Rollups

| reason | profile | count |
|---|---|---:|
| `child_run_failed` | `local-cpu-aarch64` | 10 |

## Missing Coverage

| cell | bundle | capability | depth | profile | reason |
|---|---|---|---|---|---|
| `embeddinggemma_300m__embeddings_similarity__smoke__hf_default` | `embeddinggemma_300m` | `embeddings` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `embeddinggemma_300m__embeddings_similarity__smoke__hf_default` | `embeddinggemma_300m` | `embeddings` | `smoke` | `apple-metal` | `no_record` |
| `embeddinggemma_300m__embeddings_similarity__smoke__hf_default` | `embeddinggemma_300m` | `embeddings` | `smoke` | `dgx-spark` | `no_record` |
| `embeddinggemma_300m__embeddings_similarity__smoke__hf_default` | `embeddinggemma_300m` | `embeddings` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_embedding_06b__embeddings_similarity__smoke__hf_default` | `qwen3_embedding_06b` | `embeddings` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_embedding_06b__embeddings_similarity__smoke__hf_default` | `qwen3_embedding_06b` | `embeddings` | `smoke` | `apple-metal` | `no_record` |
| `qwen3_embedding_06b__embeddings_similarity__smoke__hf_default` | `qwen3_embedding_06b` | `embeddings` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_embedding_06b__embeddings_similarity__smoke__hf_default` | `qwen3_embedding_06b` | `embeddings` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b__chat_smoke__smoke__hf_safetensors_default` | `qwen3_4b` | `chat` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_4b__chat_smoke__smoke__hf_safetensors_default` | `qwen3_4b` | `chat` | `smoke` | `apple-metal` | `no_record` |
| `qwen3_4b__chat_smoke__smoke__hf_safetensors_default` | `qwen3_4b` | `chat` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_4b__chat_smoke__smoke__hf_safetensors_default` | `qwen3_4b` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default` | `qwen3_4b` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default` | `qwen3_4b` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default` | `qwen3_4b` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default` | `qwen3_4b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `qwen3_4b` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `qwen3_4b` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `qwen3_4b` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `qwen3_4b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `qwen3_4b` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `qwen3_4b` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `qwen3_4b` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `qwen3_4b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `qwen3_4b` | `tool_use` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `qwen3_4b` | `tool_use` | `smoke` | `apple-metal` | `no_record` |
| `qwen3_4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `qwen3_4b` | `tool_use` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `qwen3_4b` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b__chat_smoke__smoke__hf_safetensors_default` | `gemma4_e2b` | `chat` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e2b__chat_smoke__smoke__hf_safetensors_default` | `gemma4_e2b` | `chat` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e2b__chat_smoke__smoke__hf_safetensors_default` | `gemma4_e2b` | `chat` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e2b__chat_smoke__smoke__hf_safetensors_default` | `gemma4_e2b` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default` | `gemma4_e2b` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default` | `gemma4_e2b` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default` | `gemma4_e2b` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default` | `gemma4_e2b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `gemma4_e2b` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `gemma4_e2b` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `gemma4_e2b` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `gemma4_e2b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `gemma4_e2b` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `gemma4_e2b` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `gemma4_e2b` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `gemma4_e2b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `gemma4_e2b` | `tool_use` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e2b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `gemma4_e2b` | `tool_use` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e2b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `gemma4_e2b` | `tool_use` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e2b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `gemma4_e2b` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b__chat_smoke__smoke__hf_safetensors_default` | `gemma4_e4b` | `chat` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e4b__chat_smoke__smoke__hf_safetensors_default` | `gemma4_e4b` | `chat` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e4b__chat_smoke__smoke__hf_safetensors_default` | `gemma4_e4b` | `chat` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e4b__chat_smoke__smoke__hf_safetensors_default` | `gemma4_e4b` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default` | `gemma4_e4b` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default` | `gemma4_e4b` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default` | `gemma4_e4b` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default` | `gemma4_e4b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `gemma4_e4b` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `gemma4_e4b` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `gemma4_e4b` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q4` | `gemma4_e4b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `gemma4_e4b` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `gemma4_e4b` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `gemma4_e4b` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_isq_q8` | `gemma4_e4b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `gemma4_e4b` | `tool_use` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `gemma4_e4b` | `tool_use` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `gemma4_e4b` | `tool_use` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `gemma4_e4b` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `chat` | `smoke` | `apple-metal` | `no_record` |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `chat` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `tool_use` | `smoke` | `apple-metal` | `no_record` |
| `qwen3_4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `tool_use` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `qwen3_6_27b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `qwen3_6_27b_gguf` | `chat` | `smoke` | `apple-metal` | `no_record` |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `qwen3_6_27b_gguf` | `chat` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `qwen3_6_27b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `qwen3_6_27b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `qwen3_6_27b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `qwen3_6_27b_gguf` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `qwen3_6_27b_gguf` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `qwen3_6_27b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `chat` | `smoke` | `apple-metal` | `no_record` |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `ornith_1_0_35b_gguf__chat_completion_smoke__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `completion` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `ornith_1_0_35b_gguf__chat_completion_smoke__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `completion` | `smoke` | `apple-metal` | `no_record` |
| `ornith_1_0_35b_gguf__chat_completion_smoke__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `completion` | `smoke` | `cuda-workstation` | `no_record` |
| `ornith_1_0_35b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `ornith_1_0_35b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `tool_use` | `smoke` | `apple-metal` | `no_record` |
| `ornith_1_0_35b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `ornith_1_0_35b_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `chat` | `smoke` | `apple-metal` | `no_record` |
| `ornith_1_0_35b_gguf__chat_smoke__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `ornith_1_0_35b_gguf__bench_chat_startup__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `ornith_1_0_35b_gguf__chat_completion_smoke__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `completion` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `ornith_1_0_35b_gguf__chat_completion_smoke__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `completion` | `smoke` | `apple-metal` | `no_record` |
| `ornith_1_0_35b_gguf__chat_completion_smoke__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `completion` | `smoke` | `cuda-workstation` | `no_record` |
| `ornith_1_0_35b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `ornith_1_0_35b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `tool_use` | `smoke` | `apple-metal` | `no_record` |
| `ornith_1_0_35b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q8_0` | `ornith_1_0_35b_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `chat` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `chat` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e2b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_e2b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e2b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e2b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `tool_use` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `tool_use` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `chat` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `chat` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_e4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `tool_use` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `tool_use` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `chat` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `chat` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_12b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_12b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_12b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_12b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `tool_use` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `tool_use` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_qat_gguf__chat_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_12b_qat_gguf__chat_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_12b_qat_gguf__chat_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `chat` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_12b_qat_gguf__chat_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `chat` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_12b_qat_gguf__chat_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_qat_gguf__bench_chat_startup__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_12b_qat_gguf__bench_chat_startup__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_12b_qat_gguf__bench_chat_startup__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `perf` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_12b_qat_gguf__bench_chat_startup__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `perf` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_12b_qat_gguf__bench_chat_startup__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_qat_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `gemma4_12b_qat_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `gemma4_12b_qat_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `tool_use` | `smoke` | `apple-metal` | `no_record` |
| `gemma4_12b_qat_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `tool_use` | `smoke` | `dgx-spark` | `no_record` |
| `gemma4_12b_qat_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `whisper_base_en` | `asr` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `whisper_base_en` | `asr` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `whisper_base_en` | `asr` | `smoke` | `apple-metal` | `no_record` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `whisper_base_en` | `asr` | `smoke` | `dgx-spark` | `no_record` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `whisper_base_en` | `asr` | `smoke` | `cuda-workstation` | `no_record` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `moonshine_streaming_en` | `asr` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `moonshine_streaming_en` | `asr` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `moonshine_streaming_en` | `asr` | `smoke` | `apple-metal` | `no_record` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `moonshine_streaming_en` | `asr` | `smoke` | `dgx-spark` | `no_record` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `moonshine_streaming_en` | `asr` | `smoke` | `cuda-workstation` | `no_record` |
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `sherpa_onnx_streaming_zipformer_en` | `asr` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `sherpa_onnx_streaming_zipformer_en` | `asr` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `sherpa_onnx_streaming_zipformer_en` | `asr` | `smoke` | `apple-metal` | `no_record` |
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `sherpa_onnx_streaming_zipformer_en` | `asr` | `smoke` | `dgx-spark` | `no_record` |
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `sherpa_onnx_streaming_zipformer_en` | `asr` | `smoke` | `cuda-workstation` | `no_record` |
| `sherpa_onnx_streaming_zipformer_en_kroko_2025__asr_short_transcription__smoke__hf_default` | `sherpa_onnx_streaming_zipformer_en_kroko_2025` | `asr` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `sherpa_onnx_streaming_zipformer_en_kroko_2025__asr_short_transcription__smoke__hf_default` | `sherpa_onnx_streaming_zipformer_en_kroko_2025` | `asr` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `sherpa_onnx_streaming_zipformer_en_kroko_2025__asr_short_transcription__smoke__hf_default` | `sherpa_onnx_streaming_zipformer_en_kroko_2025` | `asr` | `smoke` | `apple-metal` | `no_record` |
| `sherpa_onnx_streaming_zipformer_en_kroko_2025__asr_short_transcription__smoke__hf_default` | `sherpa_onnx_streaming_zipformer_en_kroko_2025` | `asr` | `smoke` | `dgx-spark` | `no_record` |
| `sherpa_onnx_streaming_zipformer_en_kroko_2025__asr_short_transcription__smoke__hf_default` | `sherpa_onnx_streaming_zipformer_en_kroko_2025` | `asr` | `smoke` | `cuda-workstation` | `no_record` |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `apple-metal` | `no_record` |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `dgx-spark` | `no_record` |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `cuda-workstation` | `no_record` |
| `kokoro_82m__tts_synthesis_smoke__smoke__onnx_default` | `kokoro_82m` | `tts` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `kokoro_82m__tts_synthesis_smoke__smoke__onnx_default` | `kokoro_82m` | `tts` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `kokoro_82m__tts_synthesis_smoke__smoke__onnx_default` | `kokoro_82m` | `tts` | `smoke` | `apple-metal` | `no_record` |
| `kokoro_82m__tts_synthesis_smoke__smoke__onnx_default` | `kokoro_82m` | `tts` | `smoke` | `dgx-spark` | `no_record` |
| `kokoro_82m__tts_synthesis_smoke__smoke__onnx_default` | `kokoro_82m` | `tts` | `smoke` | `cuda-workstation` | `no_record` |
| `kokoro_82m__tts_streaming_synthesis__smoke__onnx_default` | `kokoro_82m` | `tts` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `kokoro_82m__tts_streaming_synthesis__smoke__onnx_default` | `kokoro_82m` | `tts` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `kokoro_82m__tts_streaming_synthesis__smoke__onnx_default` | `kokoro_82m` | `tts` | `smoke` | `apple-metal` | `no_record` |
| `kokoro_82m__tts_streaming_synthesis__smoke__onnx_default` | `kokoro_82m` | `tts` | `smoke` | `dgx-spark` | `no_record` |
| `kokoro_82m__tts_streaming_synthesis__smoke__onnx_default` | `kokoro_82m` | `tts` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `local-cpu-x86_64` | `no_record` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `local-cpu-aarch64` | `no_record` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `dgx-spark` | `no_record` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `cuda-workstation` | `no_record` |

## Metric Gaps

| metric | reason | source | count |
|---|---|---|---:|
| `gpu_memory_peak_bytes` | `metric_not_instrumented` | `accelerator_sampler` | 27 |
| `warmup_ms` | `metric_not_instrumented` | `chat_runner` | 10 |

## Inputs

- `evals/results/ornith-2026-07-01/curated-v2-smoke/curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu/results.jsonl`
- `evals/results/ornith-2026-07-01/curated-v2-smoke/curated-v2-smoke-1782970098195-439707-8079ae7b-spark-2f6e-aarch64-cpu/results.jsonl`
- `evals/results/ornith-2026-07-01/curated-v2-smoke/curated-v2-smoke-1782971578478-486133-8079ae7b-spark-2f6e-aarch64-cuda/results.jsonl`

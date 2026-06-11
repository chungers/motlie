# Curated Eval Coverage Report

- input files: `8`
- records: `143`
- overall: `fail`

## Outcome Summary

| outcome | count |
|---|---:|
| `blocked` | 43 |
| `failed` | 4 |
| `passed` | 95 |
| `skipped` | 1 |

## Platform Notes

- `apple-metal` mistralrs rows with `accelerator_mismatch` are expected at this head: the curated `metal` profile feature does not currently enable the mistralrs Metal backend, and forced candle-metal probing is blocked by upstream M4 threadgroup-memory limits. These rows are honest CPU-fallback/blocked coverage, not an eval-framework failure.

## Per-Cell Coverage

| cell | host | arch | run | bundle | capability | depth | profile | requested | resolved | outcome | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `embeddinggemma_300m__embeddings_similarity__smoke__hf_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `embeddinggemma_300m` | `embeddings` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `qwen3_embedding_06b__embeddings_similarity__smoke__hf_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `qwen3_embedding_06b` | `embeddings` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `qwen3_4b__chat_smoke__smoke__hf_safetensors_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `qwen3_4b` | `chat` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `qwen3_4b` | `perf` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `qwen3_4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `qwen3_4b` | `tool_use` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `gemma4_e2b__chat_smoke__smoke__hf_safetensors_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e2b` | `chat` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e2b` | `perf` | `smoke` | `apple-metal` | `metal` | `metal` | `blocked` | `runtime_budget_exceeded` |
| `gemma4_e2b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e2b` | `tool_use` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `gemma4_e4b__chat_smoke__smoke__hf_safetensors_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e4b` | `chat` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e4b` | `perf` | `smoke` | `apple-metal` | `metal` | `metal` | `blocked` | `runtime_budget_exceeded` |
| `gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e4b` | `tool_use` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `qwen3_4b_gguf` | `chat` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `qwen3_4b_gguf` | `perf` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `qwen3_4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `qwen3_4b_gguf` | `tool_use` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `qwen3_6_27b_gguf` | `chat` | `smoke` | `apple-metal` | `metal` | `metal` | `failed` | `behavior_assertion_failed` |
| `qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `qwen3_6_27b_gguf` | `perf` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e2b_gguf` | `chat` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_e2b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e2b_gguf` | `perf` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e2b_gguf` | `tool_use` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e4b_gguf` | `chat` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_e4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e4b_gguf` | `perf` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_e4b_gguf` | `tool_use` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_12b_gguf` | `chat` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_12b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_12b_gguf` | `perf` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_12b_gguf` | `tool_use` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_12b_qat_q4_0_gguf` | `chat` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__bench_chat_startup__smoke__gguf_q4_0` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_12b_qat_q4_0_gguf` | `perf` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `gemma4_12b_qat_q4_0_gguf` | `tool_use` | `smoke` | `apple-metal` | `metal` | `metal` | `passed` | `none` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `whisper_base_en` | `asr` | `smoke` | `apple-metal` | `metal` | `metal` | `blocked` | `artifact_missing` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `moonshine_streaming_en` | `asr` | `smoke` | `apple-metal` | `metal` | `metal` | `blocked` | `artifact_missing` |
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `sherpa_onnx_streaming_zipformer_en` | `asr` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `apple-metal` | `metal` | `metal` | `skipped` | `profile_not_applicable` |
| `embeddinggemma_300m__embeddings_similarity__smoke__hf_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `embeddinggemma_300m` | `embeddings` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_embedding_06b__embeddings_similarity__smoke__hf_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `qwen3_embedding_06b` | `embeddings` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b__chat_smoke__smoke__hf_safetensors_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `qwen3_4b` | `chat` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `qwen3_4b` | `perf` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `qwen3_4b` | `tool_use` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b__chat_smoke__smoke__hf_safetensors_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e2b` | `chat` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e2b` | `perf` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e2b` | `tool_use` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b__chat_smoke__smoke__hf_safetensors_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e4b` | `chat` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e4b` | `perf` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `blocked` | `runtime_budget_exceeded` |
| `gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e4b` | `tool_use` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `qwen3_4b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `qwen3_4b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `qwen3_4b_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `qwen3_6_27b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `failed` | `behavior_assertion_failed` |
| `qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `qwen3_6_27b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `blocked` | `runtime_budget_exceeded` |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e2b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e2b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e2b_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e4b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e4b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_e4b_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_12b_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_12b_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_12b_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_12b_qat_q4_0_gguf` | `chat` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__bench_chat_startup__smoke__gguf_q4_0` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_12b_qat_q4_0_gguf` | `perf` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `gemma4_12b_qat_q4_0_gguf` | `tool_use` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `whisper_base_en` | `asr` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `moonshine_streaming_en` | `asr` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `sherpa_onnx_streaming_zipformer_en` | `asr` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `blocked` | `artifact_missing` |
| `embeddinggemma_300m__embeddings_similarity__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `embeddinggemma_300m` | `embeddings` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `native_link_failed` |
| `qwen3_embedding_06b__embeddings_similarity__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `qwen3_embedding_06b` | `embeddings` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `native_link_failed` |
| `qwen3_4b__chat_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `qwen3_4b` | `chat` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `native_link_failed` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `qwen3_4b` | `perf` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `native_link_failed` |
| `qwen3_4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `qwen3_4b` | `tool_use` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `native_link_failed` |
| `gemma4_e2b__chat_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e2b` | `chat` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `native_link_failed` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e2b` | `perf` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `native_link_failed` |
| `gemma4_e2b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e2b` | `tool_use` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `native_link_failed` |
| `gemma4_e4b__chat_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e4b` | `chat` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `native_link_failed` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e4b` | `perf` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `native_link_failed` |
| `gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e4b` | `tool_use` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `native_link_failed` |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `qwen3_4b_gguf` | `chat` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `qwen3_4b_gguf` | `perf` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `qwen3_4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `qwen3_4b_gguf` | `tool_use` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `qwen3_6_27b_gguf` | `chat` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `failed` | `behavior_assertion_failed` |
| `qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `qwen3_6_27b_gguf` | `perf` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e2b_gguf` | `chat` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_e2b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e2b_gguf` | `perf` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e2b_gguf` | `tool_use` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e4b_gguf` | `chat` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_e4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e4b_gguf` | `perf` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_e4b_gguf` | `tool_use` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_12b_gguf` | `chat` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_12b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_12b_gguf` | `perf` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_12b_gguf` | `tool_use` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_12b_qat_q4_0_gguf` | `chat` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__bench_chat_startup__smoke__gguf_q4_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_12b_qat_q4_0_gguf` | `perf` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `gemma4_12b_qat_q4_0_gguf` | `tool_use` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `whisper_base_en` | `asr` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `artifact_missing` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `moonshine_streaming_en` | `asr` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `artifact_missing` |
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `sherpa_onnx_streaming_zipformer_en` | `asr` | `smoke` | `dgx-spark` | `cuda` | `cpu` | `blocked` | `accelerator_mismatch` |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `dgx-spark` | `cuda` | `cpu` | `blocked` | `accelerator_mismatch` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `artifact_missing` |
| `embeddinggemma_300m__embeddings_similarity__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `embeddinggemma_300m` | `embeddings` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_embedding_06b__embeddings_similarity__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `qwen3_embedding_06b` | `embeddings` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b__chat_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `qwen3_4b` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `qwen3_4b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `qwen3_4b` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b__chat_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e2b` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e2b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e2b` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b__chat_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e4b` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e4b` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e4b` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `qwen3_4b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `qwen3_4b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `qwen3_4b_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `qwen3_6_27b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `failed` | `behavior_assertion_failed` |
| `qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `qwen3_6_27b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e2b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e2b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e2b_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e4b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e4b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_e4b_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_12b_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_12b_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_12b_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_12b_qat_q4_0_gguf` | `chat` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__bench_chat_startup__smoke__gguf_q4_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_12b_qat_q4_0_gguf` | `perf` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `gemma4_12b_qat_q4_0_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `gemma4_12b_qat_q4_0_gguf` | `tool_use` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `whisper_base_en` | `asr` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `artifact_missing` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `moonshine_streaming_en` | `asr` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `artifact_missing` |
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `sherpa_onnx_streaming_zipformer_en` | `asr` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `artifact_missing` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781168668566-1963-e14eaa7e-mac-mini-m4pro-local-aarch64-metal` | `whisper_base_en` | `asr` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `mac-mini-m4pro-local` | `aarch64` | `curated-v2-smoke-1781168668566-1963-e14eaa7e-mac-mini-m4pro-local-aarch64-metal` | `moonshine_streaming_en` | `asr` | `smoke` | `apple-metal` | `metal` | `cpu` | `blocked` | `accelerator_mismatch` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781168781390-337905-e14eaa7e-spark-2f6e-aarch64-cuda` | `whisper_base_en` | `asr` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781168781390-337905-e14eaa7e-spark-2f6e-aarch64-cuda` | `moonshine_streaming_en` | `asr` | `smoke` | `dgx-spark` | `cuda` | `cpu` | `blocked` | `accelerator_mismatch` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781168781390-337905-e14eaa7e-spark-2f6e-aarch64-cuda` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `blocked` | `artifact_missing` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781169018588-4036013-e14eaa7e-amd-ryzen7-1-x86_64-cpu` | `whisper_base_en` | `asr` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781169018588-4036013-e14eaa7e-amd-ryzen7-1-x86_64-cpu` | `moonshine_streaming_en` | `asr` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781169018588-4036013-e14eaa7e-amd-ryzen7-1-x86_64-cpu` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `blocked` | `artifact_missing` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781169935139-380628-e14eaa7e-spark-2f6e-aarch64-cpu` | `whisper_base_en` | `asr` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781169935139-380628-e14eaa7e-spark-2f6e-aarch64-cpu` | `moonshine_streaming_en` | `asr` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `spark-2f6e` | `aarch64` | `curated-v2-smoke-1781169935139-380628-e14eaa7e-spark-2f6e-aarch64-cpu` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `local-cpu-aarch64` | `cpu` | `cpu` | `blocked` | `artifact_missing` |

## Model x Capability

| model_family | capability | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `gemma` | `chat` | 20 | 0 | 4 | 0 |
| `gemma` | `embeddings` | 2 | 0 | 2 | 0 |
| `gemma` | `perf` | 19 | 0 | 5 | 0 |
| `gemma` | `tool_use` | 20 | 0 | 4 | 0 |
| `moonshine` | `asr` | 3 | 0 | 5 | 0 |
| `piper` | `tts` | 2 | 0 | 2 | 0 |
| `qwen3` | `chat` | 6 | 4 | 2 | 0 |
| `qwen3` | `embeddings` | 2 | 0 | 2 | 0 |
| `qwen3` | `perf` | 9 | 0 | 3 | 0 |
| `qwen3` | `tool_use` | 6 | 0 | 2 | 0 |
| `qwen3` | `tts` | 0 | 0 | 6 | 1 |
| `sherpa_onnx` | `asr` | 2 | 0 | 2 | 0 |
| `whisper` | `asr` | 4 | 0 | 4 | 0 |

## Capability x Profile

| capability | profile | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `asr` | `apple-metal` | 0 | 0 | 5 | 0 |
| `asr` | `dgx-spark` | 1 | 0 | 4 | 0 |
| `asr` | `local-cpu-aarch64` | 3 | 0 | 2 | 0 |
| `asr` | `local-cpu-x86_64` | 5 | 0 | 0 | 0 |
| `chat` | `apple-metal` | 5 | 1 | 3 | 0 |
| `chat` | `dgx-spark` | 5 | 1 | 3 | 0 |
| `chat` | `local-cpu-aarch64` | 8 | 1 | 0 | 0 |
| `chat` | `local-cpu-x86_64` | 8 | 1 | 0 | 0 |
| `embeddings` | `apple-metal` | 0 | 0 | 2 | 0 |
| `embeddings` | `dgx-spark` | 0 | 0 | 2 | 0 |
| `embeddings` | `local-cpu-aarch64` | 2 | 0 | 0 | 0 |
| `embeddings` | `local-cpu-x86_64` | 2 | 0 | 0 | 0 |
| `perf` | `apple-metal` | 6 | 0 | 3 | 0 |
| `perf` | `dgx-spark` | 6 | 0 | 3 | 0 |
| `perf` | `local-cpu-aarch64` | 9 | 0 | 0 | 0 |
| `perf` | `local-cpu-x86_64` | 7 | 0 | 2 | 0 |
| `tool_use` | `apple-metal` | 5 | 0 | 3 | 0 |
| `tool_use` | `dgx-spark` | 5 | 0 | 3 | 0 |
| `tool_use` | `local-cpu-aarch64` | 8 | 0 | 0 | 0 |
| `tool_use` | `local-cpu-x86_64` | 8 | 0 | 0 | 0 |
| `tts` | `apple-metal` | 0 | 0 | 1 | 1 |
| `tts` | `dgx-spark` | 0 | 0 | 3 | 0 |
| `tts` | `local-cpu-aarch64` | 1 | 0 | 2 | 0 |
| `tts` | `local-cpu-x86_64` | 1 | 0 | 2 | 0 |

## Capability x Depth

| capability | depth | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `asr` | `smoke` | 9 | 0 | 11 | 0 |
| `chat` | `smoke` | 26 | 4 | 6 | 0 |
| `embeddings` | `smoke` | 4 | 0 | 4 | 0 |
| `perf` | `smoke` | 28 | 0 | 8 | 0 |
| `tool_use` | `smoke` | 26 | 0 | 6 | 0 |
| `tts` | `smoke` | 2 | 0 | 8 | 1 |

## Backend x Profile

| backend | profile | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `llama_cpp` | `apple-metal` | 16 | 1 | 0 | 0 |
| `llama_cpp` | `dgx-spark` | 16 | 1 | 0 | 0 |
| `llama_cpp` | `local-cpu-aarch64` | 16 | 1 | 0 | 0 |
| `llama_cpp` | `local-cpu-x86_64` | 15 | 1 | 1 | 0 |
| `mistralrs` | `apple-metal` | 0 | 0 | 11 | 0 |
| `mistralrs` | `dgx-spark` | 0 | 0 | 11 | 0 |
| `mistralrs` | `local-cpu-aarch64` | 11 | 0 | 0 | 0 |
| `mistralrs` | `local-cpu-x86_64` | 10 | 0 | 1 | 0 |
| `ort` | `apple-metal` | 0 | 0 | 3 | 0 |
| `ort` | `dgx-spark` | 0 | 0 | 3 | 0 |
| `ort` | `local-cpu-aarch64` | 2 | 0 | 1 | 0 |
| `ort` | `local-cpu-x86_64` | 3 | 0 | 0 | 0 |
| `qwen3_tts_cpp` | `apple-metal` | 0 | 0 | 0 | 1 |
| `qwen3_tts_cpp` | `dgx-spark` | 0 | 0 | 2 | 0 |
| `qwen3_tts_cpp` | `local-cpu-aarch64` | 0 | 0 | 2 | 0 |
| `qwen3_tts_cpp` | `local-cpu-x86_64` | 0 | 0 | 2 | 0 |
| `sherpa_onnx` | `apple-metal` | 0 | 0 | 1 | 0 |
| `sherpa_onnx` | `dgx-spark` | 0 | 0 | 1 | 0 |
| `sherpa_onnx` | `local-cpu-aarch64` | 1 | 0 | 0 | 0 |
| `sherpa_onnx` | `local-cpu-x86_64` | 1 | 0 | 0 | 0 |
| `whisper_cpp` | `apple-metal` | 0 | 0 | 2 | 0 |
| `whisper_cpp` | `dgx-spark` | 1 | 0 | 1 | 0 |
| `whisper_cpp` | `local-cpu-aarch64` | 1 | 0 | 1 | 0 |
| `whisper_cpp` | `local-cpu-x86_64` | 2 | 0 | 0 | 0 |

## Model x Quantization x Backend/Profile/Depth

| model | quantization | backend | profile | depth | passed | failed | blocked | skipped |
|---|---|---|---|---|---:|---:|---:|---:|
| `gemma` | `default` | `mistralrs` | `apple-metal` | `smoke` | 0 | 0 | 7 | 0 |
| `gemma` | `default` | `mistralrs` | `dgx-spark` | `smoke` | 0 | 0 | 7 | 0 |
| `gemma` | `default` | `mistralrs` | `local-cpu-aarch64` | `smoke` | 7 | 0 | 0 | 0 |
| `gemma` | `default` | `mistralrs` | `local-cpu-x86_64` | `smoke` | 6 | 0 | 1 | 0 |
| `gemma` | `q4_0` | `llama_cpp` | `apple-metal` | `smoke` | 3 | 0 | 0 | 0 |
| `gemma` | `q4_0` | `llama_cpp` | `dgx-spark` | `smoke` | 3 | 0 | 0 | 0 |
| `gemma` | `q4_0` | `llama_cpp` | `local-cpu-aarch64` | `smoke` | 3 | 0 | 0 | 0 |
| `gemma` | `q4_0` | `llama_cpp` | `local-cpu-x86_64` | `smoke` | 3 | 0 | 0 | 0 |
| `gemma` | `q4_k_m` | `llama_cpp` | `apple-metal` | `smoke` | 9 | 0 | 0 | 0 |
| `gemma` | `q4_k_m` | `llama_cpp` | `dgx-spark` | `smoke` | 9 | 0 | 0 | 0 |
| `gemma` | `q4_k_m` | `llama_cpp` | `local-cpu-aarch64` | `smoke` | 9 | 0 | 0 | 0 |
| `gemma` | `q4_k_m` | `llama_cpp` | `local-cpu-x86_64` | `smoke` | 9 | 0 | 0 | 0 |
| `moonshine` | `default` | `ort` | `apple-metal` | `smoke` | 0 | 0 | 2 | 0 |
| `moonshine` | `default` | `ort` | `dgx-spark` | `smoke` | 0 | 0 | 2 | 0 |
| `moonshine` | `default` | `ort` | `local-cpu-aarch64` | `smoke` | 1 | 0 | 1 | 0 |
| `moonshine` | `default` | `ort` | `local-cpu-x86_64` | `smoke` | 2 | 0 | 0 | 0 |
| `piper` | `default` | `ort` | `apple-metal` | `smoke` | 0 | 0 | 1 | 0 |
| `piper` | `default` | `ort` | `dgx-spark` | `smoke` | 0 | 0 | 1 | 0 |
| `piper` | `default` | `ort` | `local-cpu-aarch64` | `smoke` | 1 | 0 | 0 | 0 |
| `piper` | `default` | `ort` | `local-cpu-x86_64` | `smoke` | 1 | 0 | 0 | 0 |
| `qwen3` | `default` | `mistralrs` | `apple-metal` | `smoke` | 0 | 0 | 4 | 0 |
| `qwen3` | `default` | `mistralrs` | `dgx-spark` | `smoke` | 0 | 0 | 4 | 0 |
| `qwen3` | `default` | `mistralrs` | `local-cpu-aarch64` | `smoke` | 4 | 0 | 0 | 0 |
| `qwen3` | `default` | `mistralrs` | `local-cpu-x86_64` | `smoke` | 4 | 0 | 0 | 0 |
| `qwen3` | `q4_k_m` | `llama_cpp` | `apple-metal` | `smoke` | 4 | 1 | 0 | 0 |
| `qwen3` | `q4_k_m` | `llama_cpp` | `dgx-spark` | `smoke` | 4 | 1 | 0 | 0 |
| `qwen3` | `q4_k_m` | `llama_cpp` | `local-cpu-aarch64` | `smoke` | 4 | 1 | 0 | 0 |
| `qwen3` | `q4_k_m` | `llama_cpp` | `local-cpu-x86_64` | `smoke` | 3 | 1 | 1 | 0 |
| `qwen3` | `q8_0` | `qwen3_tts_cpp` | `apple-metal` | `smoke` | 0 | 0 | 0 | 1 |
| `qwen3` | `q8_0` | `qwen3_tts_cpp` | `dgx-spark` | `smoke` | 0 | 0 | 2 | 0 |
| `qwen3` | `q8_0` | `qwen3_tts_cpp` | `local-cpu-aarch64` | `smoke` | 0 | 0 | 2 | 0 |
| `qwen3` | `q8_0` | `qwen3_tts_cpp` | `local-cpu-x86_64` | `smoke` | 0 | 0 | 2 | 0 |
| `sherpa_onnx` | `default` | `sherpa_onnx` | `apple-metal` | `smoke` | 0 | 0 | 1 | 0 |
| `sherpa_onnx` | `default` | `sherpa_onnx` | `dgx-spark` | `smoke` | 0 | 0 | 1 | 0 |
| `sherpa_onnx` | `default` | `sherpa_onnx` | `local-cpu-aarch64` | `smoke` | 1 | 0 | 0 | 0 |
| `sherpa_onnx` | `default` | `sherpa_onnx` | `local-cpu-x86_64` | `smoke` | 1 | 0 | 0 | 0 |
| `whisper` | `default` | `whisper_cpp` | `apple-metal` | `smoke` | 0 | 0 | 2 | 0 |
| `whisper` | `default` | `whisper_cpp` | `dgx-spark` | `smoke` | 1 | 0 | 1 | 0 |
| `whisper` | `default` | `whisper_cpp` | `local-cpu-aarch64` | `smoke` | 1 | 0 | 1 | 0 |
| `whisper` | `default` | `whisper_cpp` | `local-cpu-x86_64` | `smoke` | 2 | 0 | 0 | 0 |

## Requested x Resolved Accelerator

| requested | resolved | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `cpu` | `cpu` | 62 | 2 | 8 | 0 |
| `cuda` | `cpu` | 0 | 0 | 3 | 0 |
| `cuda` | `cuda` | 17 | 1 | 15 | 0 |
| `metal` | `cpu` | 0 | 0 | 13 | 0 |
| `metal` | `metal` | 16 | 1 | 4 | 1 |

## Blocker Rollups

| reason | profile | count |
|---|---|---:|
| `accelerator_mismatch` | `apple-metal` | 13 |
| `accelerator_mismatch` | `dgx-spark` | 3 |
| `artifact_missing` | `apple-metal` | 2 |
| `artifact_missing` | `dgx-spark` | 4 |
| `artifact_missing` | `local-cpu-aarch64` | 4 |
| `artifact_missing` | `local-cpu-x86_64` | 2 |
| `behavior_assertion_failed` | `apple-metal` | 1 |
| `behavior_assertion_failed` | `dgx-spark` | 1 |
| `behavior_assertion_failed` | `local-cpu-aarch64` | 1 |
| `behavior_assertion_failed` | `local-cpu-x86_64` | 1 |
| `native_link_failed` | `dgx-spark` | 11 |
| `runtime_budget_exceeded` | `apple-metal` | 2 |
| `runtime_budget_exceeded` | `local-cpu-x86_64` | 2 |

## Missing Coverage

| cell | bundle | capability | depth | profile | reason |
|---|---|---|---|---|---|
| `embeddinggemma_300m__embeddings_similarity__smoke__hf_default` | `embeddinggemma_300m` | `embeddings` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_embedding_06b__embeddings_similarity__smoke__hf_default` | `qwen3_embedding_06b` | `embeddings` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b__chat_smoke__smoke__hf_safetensors_default` | `qwen3_4b` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default` | `qwen3_4b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `qwen3_4b` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b__chat_smoke__smoke__hf_safetensors_default` | `gemma4_e2b` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default` | `gemma4_e2b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `gemma4_e2b` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b__chat_smoke__smoke__hf_safetensors_default` | `gemma4_e4b` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default` | `gemma4_e4b` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `gemma4_e4b` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `qwen3_4b_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `qwen3_6_27b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `qwen3_6_27b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e2b_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_e4b_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `gemma4_12b_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_q4_0_gguf` | `chat` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_qat_q4_0_gguf__bench_chat_startup__smoke__gguf_q4_0` | `gemma4_12b_qat_q4_0_gguf` | `perf` | `smoke` | `cuda-workstation` | `no_record` |
| `gemma4_12b_qat_q4_0_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0` | `gemma4_12b_qat_q4_0_gguf` | `tool_use` | `smoke` | `cuda-workstation` | `no_record` |
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `whisper_base_en` | `asr` | `smoke` | `cuda-workstation` | `no_record` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `moonshine_streaming_en` | `asr` | `smoke` | `cuda-workstation` | `no_record` |
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `sherpa_onnx_streaming_zipformer_en` | `asr` | `smoke` | `cuda-workstation` | `no_record` |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `cuda-workstation` | `no_record` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `cuda-workstation` | `no_record` |

## Metric Gaps

| metric | reason | source | count |
|---|---|---|---:|
| `gpu_memory_peak_bytes` | `metric_not_instrumented` | `accelerator_sampler` | 115 |
| `mean_ttft_first_answer_token_ms` | `metric_not_reported_by_backend` | `chat_response.timing.first_answer_token_at` | 3 |
| `process_swap_delta_peak_bytes` | `metric_unavailable_on_platform` | `mach_task_info_unimplemented` | 30 |
| `ttft_first_answer_token_ms` | `metric_not_reported_by_backend` | `chat_response.timing.first_answer_token_at` | 4 |
| `warmup_ms` | `metric_not_instrumented` | `chat_runner` | 33 |

## Inputs

- `/tmp/final-agg-runs/curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal/results.jsonl`
- `/tmp/final-agg-runs/curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu/results.jsonl`
- `/tmp/final-agg-runs/curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda/results.jsonl`
- `/tmp/final-agg-runs/curated-v2-smoke-1781162017635-162832-99ac891d-spark-2f6e-aarch64-cpu/results.jsonl`
- `/tmp/final-agg-runs/curated-v2-smoke-1781168668566-1963-e14eaa7e-mac-mini-m4pro-local-aarch64-metal/results.jsonl`
- `/tmp/final-agg-runs/curated-v2-smoke-1781168781390-337905-e14eaa7e-spark-2f6e-aarch64-cuda/results.jsonl`
- `/tmp/final-agg-runs/curated-v2-smoke-1781169018588-4036013-e14eaa7e-amd-ryzen7-1-x86_64-cpu/results.jsonl`
- `/tmp/final-agg-runs/curated-v2-smoke-1781169935139-380628-e14eaa7e-spark-2f6e-aarch64-cpu/results.jsonl`

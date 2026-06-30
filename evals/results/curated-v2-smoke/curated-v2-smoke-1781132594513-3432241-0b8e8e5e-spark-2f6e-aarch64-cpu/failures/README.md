# Failure Excerpts

- run_id: `curated-v2-smoke-1781132594513-3432241-0b8e8e5e-spark-2f6e-aarch64-cpu`
- git_sha: `0b8e8e5ecb53c5256037b8970041446a1637515b`
- profile: `local-cpu-aarch64`
- build_profile: `debug`
- non_pass_records: `15`

- [qwen3_4b__chat_smoke__smoke__hf_safetensors_default](01_qwen3_4b__chat_smoke__smoke__hf_safetensors_default.md) — `blocked` / `child_run_failed`
- [qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default](02_qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default.md) — `blocked` / `runtime_budget_exceeded`
- [gemma4_e2b__chat_smoke__smoke__hf_safetensors_default](03_gemma4_e2b__chat_smoke__smoke__hf_safetensors_default.md) — `blocked` / `child_run_failed`
- [gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default](04_gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default.md) — `blocked` / `runtime_budget_exceeded`
- [gemma4_e4b__chat_smoke__smoke__hf_safetensors_default](05_gemma4_e4b__chat_smoke__smoke__hf_safetensors_default.md) — `blocked` / `child_run_failed`
- [gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default](06_gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default.md) — `blocked` / `runtime_budget_exceeded`
- [qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m](07_qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m.md) — `failed` / `behavior_assertion_failed`
- [qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m](08_qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m.md) — `blocked` / `runtime_budget_exceeded`
- [gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m](09_gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m.md) — `failed` / `behavior_assertion_failed`
- [gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m](10_gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m.md) — `failed` / `behavior_assertion_failed`
- [gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0](11_gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0.md) — `failed` / `behavior_assertion_failed`
- [moonshine_streaming_en__asr_short_transcription__smoke__hf_default](12_moonshine_streaming_en__asr_short_transcription__smoke__hf_default.md) — `blocked` / `child_run_failed`
- [sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default](13_sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default.md) — `blocked` / `artifact_unauthorized`
- [piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default](14_piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default.md) — `blocked` / `artifact_unauthorized`
- [qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0](15_qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0.md) — `blocked` / `artifact_missing`

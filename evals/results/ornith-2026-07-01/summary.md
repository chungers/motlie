# Ornith 2026-07-01 Results Summary

## Changelog

- 2026-07-01 PDT, @codex-593-ornith -- Documented Ornith GGUF MoE metadata, CPU offload proof, CUDA run proof, and completion-path assertion findings.

## Canonical Runs

- CPU sanity rerun: `curated-v2-smoke-1782960036816-254165-e350a09b-spark-2f6e-aarch64-cpu`
- CUDA run: `curated-v2-smoke-1782959238266-236531-e350a09b-spark-2f6e-aarch64-cuda`
- Both runs use commit `e350a09b925ef0527bc2eed69120ee72f6cd3c65`, snapshot `curated-v2-smoke`, host `spark-2f6e`, arch `aarch64`.

## Sanity Check

Ornith-1.0-35B-GGUF is a Mixture-of-Experts GGUF, not a dense 35B decode workload. The child logs report `general.architecture=qwen35moe`, `qwen35moe.expert_count=256`, `qwen35moe.expert_used_count=8`, `model type=35B.A3B`, and `model params=34.66 B`. The CPU decode speed is therefore plausible and should be called out as active-parameter MoE throughput, not dense 35B throughput.

The loaded artifacts were the expected quantized GGUFs from Hugging Face snapshot `c2e1703039380de4ce6820e97afd185682d3c16c`:

- Q4_K_M: `ornith-1.0-35b-Q4_K_M.gguf`, blob `ff25291b2599fb927a835e624d2b3540106af61761c3fa57ac4264046dbec002`, `21166757760` bytes, log file type `Q4_K - Medium`.
- Q8_0: `ornith-1.0-35b-Q8_0.gguf`, blob `cbc992bca07901c1a51f33e65e6fc5d687de179c852a772dfd15e4c3261dbf5c`, `36903138880` bytes, log file type `Q8_0`.

The canonical CPU rerun used `MOTLIE_MODEL_FORCE_CPU=1`. Every CPU Ornith record reports `backend_mode=llama_cpp:cpu`, `offload=gpu_layers=0`, `runtime.gpu_layers=0`, `MOTLIE_MODEL_FORCE_CPU=1`, and `use_proof_source=env:motlie_model_force_cpu`. During the active CPU child, `nvidia-smi` reported no running GPU processes.

The CUDA matrix built and ran llama.cpp with CUDA. Every CUDA Ornith record reports `backend_mode=llama_cpp:cuda` and `offload=gpu_layers=9999;selected_device=0`; no structured BuildGap was emitted.

For LLM chat records, `tokens_per_second` is total generated tokens over request duration and `decode_tokens_per_second` is generated tokens over first-token-to-last-token decode duration. Report decode throughput from `decode_tokens_per_second`.

## Matrix Outcomes

| Profile | Quant | Scenario | Outcome | Decode tok/s | Notes |
|---|---|---:|---|---:|---|
| local-cpu-aarch64 | Q4_K_M | chat_smoke | passed | 17.83 | forced CPU, `gpu_layers=0` |
| local-cpu-aarch64 | Q4_K_M | chat_completion_smoke | failed | 18.57 | completion leg passed with `completion_chars=220`; primary chat leg failed with `response_chars=0` |
| local-cpu-aarch64 | Q4_K_M | tool_use_weather_cel_smoke | passed | n/a | round-trip latency `32239` ms, final response chars `67` |
| local-cpu-aarch64 | Q8_0 | chat_smoke | passed | 17.12 | forced CPU, `gpu_layers=0` |
| local-cpu-aarch64 | Q8_0 | chat_completion_smoke | failed | 12.63 | completion leg passed with `completion_chars=220`; primary chat leg failed with `response_chars=0` |
| local-cpu-aarch64 | Q8_0 | tool_use_weather_cel_smoke | passed | n/a | round-trip latency `30511` ms, final response chars `67` |
| dgx-spark | Q4_K_M | chat_smoke | passed | 65.82 | CUDA selected device `0` |
| dgx-spark | Q4_K_M | chat_completion_smoke | failed | 61.40 | completion leg passed with `completion_chars=213`; primary chat leg failed with `response_chars=0` |
| dgx-spark | Q4_K_M | tool_use_weather_cel_smoke | passed | n/a | round-trip latency `3710` ms, final response chars `76` |
| dgx-spark | Q8_0 | chat_smoke | passed | 46.15 | CUDA selected device `0` |
| dgx-spark | Q8_0 | chat_completion_smoke | failed | 46.34 | completion leg passed with `completion_chars=218`; primary chat leg failed with `response_chars=0` |
| dgx-spark | Q8_0 | tool_use_weather_cel_smoke | passed | n/a | round-trip latency `5093` ms, final response chars `67` |

The apparent Q4 chat inconsistency is the `chat_completion_smoke` cell. That scenario is implemented as a chat scenario that also invokes the completion capability. In all four CPU/CUDA x Q4/Q8 completion-path records, the completion assertion passed, but the primary chat assertion failed because the primary chat response was empty.

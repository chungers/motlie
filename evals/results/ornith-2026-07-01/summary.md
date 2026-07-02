# Ornith 2026-07-01 Results Summary

## Changelog

- 2026-07-02 UTC, @codex-593-ornith -- Addressed PR #595 review: relabeled completion coverage, removed the 64-token completion cap, bounded chat smoke generation, reran CPU/CUDA Ornith matrices, and documented MoE/offload/provenance caveats.
- 2026-07-01 PDT, @codex-593-ornith -- Documented Ornith GGUF MoE metadata, CPU offload proof, CUDA run proof, and completion-path assertion findings.

## Canonical Runs

Final Ornith raw runs are the two `8079ae7b` run directories below. Older Ornith six-cell and intermediate rerun directories were removed from the aggregate inputs because they either lacked perf cells, mislabeled completion cells as chat, or were produced before the source fix landed in `identity.git_sha`.

- CPU matrix: `curated-v2-smoke-1782970098195-439707-8079ae7b-spark-2f6e-aarch64-cpu`
- CUDA matrix: `curated-v2-smoke-1782971578478-486133-8079ae7b-spark-2f6e-aarch64-cuda`
- Source commit: `8079ae7b6da852db26e97e0771fa75d740e52f54`
- Snapshot: `curated-v2-smoke`
- Host/arch: `spark-2f6e` / `aarch64`

The aggregate report still includes the earlier broad CPU comparator run `curated-v2-smoke-1782949898890-80870-5a990b58-spark-2f6e-aarch64-cpu` for non-Ornith baseline rows. Ornith conclusions below are based only on the two final `8079ae7b` runs.

## Sanity Check

Ornith-1.0-35B-GGUF is a Mixture-of-Experts GGUF, not a dense 35B decode workload. Direct GGUF metadata for both Q4_K_M and Q8_0 reports:

- `general.architecture = qwen35moe`
- `qwen35moe.expert_count = 256`
- `qwen35moe.expert_used_count = 8`
- `qwen35moe.block_count = 40`
- `qwen35moe.embedding_length = 2048`
- `qwen35moe.expert_feed_forward_length = 512`
- `qwen35moe.expert_shared_feed_forward_length = 512`

The CPU decode speed is therefore plausible active-expert MoE throughput, not dense 35B throughput.

The loaded artifacts were the expected GGUFs from Hugging Face snapshot `c2e1703039380de4ce6820e97afd185682d3c16c`:

- Q4_K_M: `ornith-1.0-35b-Q4_K_M.gguf`, blob `ff25291b2599fb927a835e624d2b3540106af61761c3fa57ac4264046dbec002`, `21166757760` bytes, log file type `Q4_K - Medium`.
- Q8_0: `ornith-1.0-35b-Q8_0.gguf`, blob `cbc992bca07901c1a51f33e65e6fc5d687de179c852a772dfd15e4c3261dbf5c`, `36903138880` bytes, log file type `Q8_0`.

The CPU matrix used `MOTLIE_MODEL_FORCE_CPU=1`. Every CPU Ornith record reports `backend_mode=llama_cpp:cpu`, `offload=gpu_layers=0`, `runtime.gpu_layers=0`, `MOTLIE_MODEL_FORCE_CPU=1`, and `use_proof_source=env:motlie_model_force_cpu`; the CPU run was not silently offloaded to GPU.

The CUDA matrix built and ran llama.cpp with CUDA. Every CUDA Ornith record reports `backend_mode=llama_cpp:cuda` and `offload=gpu_layers=9999;selected_device=0`; no structured BuildGap was emitted.

For LLM chat records, `tokens_per_second` is total generated tokens over request duration and `decode_tokens_per_second` is generated tokens over first-token-to-last-token decode duration. The throughput figures below use decode throughput.

## Matrix Outcomes

| Profile | Quant | Scenario | Outcome | Decode tok/s | TTFT answer ms | Notes |
|---|---|---|---|---:|---:|---|
| local-cpu-aarch64 | Q4_K_M | chat_smoke | passed | 19.28 | 66983 | forced CPU, `gpu_layers=0`, 1299 generated tokens |
| local-cpu-aarch64 | Q4_K_M | bench_chat_startup | passed | 19.69 | 88608 mean | 3-sample perf cell |
| local-cpu-aarch64 | Q4_K_M | chat_completion_smoke | passed | 21.04 | 8900 | completion capability; `completion_chars=6870` |
| local-cpu-aarch64 | Q4_K_M | tool_use_weather_cel_smoke | passed | n/a | n/a | forced CPU, tool round trip passed |
| local-cpu-aarch64 | Q8_0 | chat_smoke | passed | 16.92 | 70530 | forced CPU, `gpu_layers=0`, 1213 generated tokens |
| local-cpu-aarch64 | Q8_0 | bench_chat_startup | passed | 16.71 | 79608 mean | 3-sample perf cell |
| local-cpu-aarch64 | Q8_0 | chat_completion_smoke | passed | 18.85 | 9358 | completion capability; `completion_chars=11889` |
| local-cpu-aarch64 | Q8_0 | tool_use_weather_cel_smoke | passed | n/a | n/a | forced CPU, tool round trip passed |
| dgx-spark | Q4_K_M | chat_smoke | passed | 65.22 | 30801 | CUDA selected device `0`, 2049 generated tokens |
| dgx-spark | Q4_K_M | bench_chat_startup | passed | 66.65 | 12574 mean | 3-sample perf cell |
| dgx-spark | Q4_K_M | chat_completion_smoke | passed | 66.11 | 2591 | completion capability; `completion_chars=6426` |
| dgx-spark | Q4_K_M | tool_use_weather_cel_smoke | passed | n/a | n/a | CUDA selected device `0`, tool round trip passed |
| dgx-spark | Q8_0 | chat_smoke | passed | 45.95 | 22813 | CUDA selected device `0`, 1069 generated tokens |
| dgx-spark | Q8_0 | bench_chat_startup | passed | 46.40 | 36738 mean | 3-sample perf cell |
| dgx-spark | Q8_0 | chat_completion_smoke | passed | 46.06 | 11074 | completion capability; `completion_chars=8928` |
| dgx-spark | Q8_0 | tool_use_weather_cel_smoke | passed | n/a | n/a | CUDA selected device `0`, tool round trip passed |

## Review Findings

The prior failed Q4 "chat" cell was not a second chat sample. It was `chat_completion_smoke`, which used a chat runner to exercise the completion path and was incorrectly reported as `capability=chat`. The snapshot and matrix runner now carry an explicit coverage capability, so `chat_completion_smoke` reports `capability=completion` in raw records, aggregate coverage, and `coverage/INDEX.json`.

The failed 64-token completion rows were noisy harness artifacts, not valid throughput samples. The completion prompt asks the model to produce a long fixed completion, while the primary chat leg only emits a short readiness response. With `max_tokens=64`, the thinking budget could consume the entire cap before the answer. `chat_completion_smoke` no longer sets a 64-token cap, and the final CPU/CUDA completion cells all pass.

The chat smoke scenario is now capped at `max_tokens=4096`. That cap is high enough for thinking-model coverage but prevents no-EOS samples from consuming the matrix runtime budget or truncating the follow-up answer.

TTFT remains an important caveat. CPU chat first-answer latency is long even when throughput is plausible: final single chat cells are 66.98s Q4 and 70.53s Q8, and perf-cell means are 88.61s Q4 and 79.61s Q8. CUDA improves first-token latency and throughput, but thinking latency can still be material: final CUDA chat cells are 30.80s Q4 and 22.81s Q8, with perf-cell first-answer means of 12.57s Q4 and 36.74s Q8.

Remaining interpretation caveats: chat and completion smoke cells are n=1; perf cells are n=3 and should be used for throughput comparison. The aggregate has no non-Ornith CUDA baseline in this PR, so CUDA comparisons are CPU-vs-CUDA for Ornith only.

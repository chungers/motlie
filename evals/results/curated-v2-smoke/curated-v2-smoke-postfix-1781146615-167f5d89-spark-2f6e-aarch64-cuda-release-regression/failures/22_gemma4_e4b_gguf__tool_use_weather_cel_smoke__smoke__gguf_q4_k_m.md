# gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m

- outcome: `blocked`
- reason: `resource_gate_failed`
- acceptance: `blocked`
- failure_reason: `resources section not accepted: resource metric gpu_memory_peak_bytes blocked: CUDA peak VRAM sampler not instrumented`
- bundle: `gemma4_e4b_gguf`
- capability: `tool_use`
- scenario: `tool_use_weather_cel_smoke`
- profile: `dgx-spark`
- host_id: `spark-2f6e`
- platform: `linux/aarch64`
- requested_accelerator: `cuda`
- resolved_accelerator: `cuda`
- backend: `llama_cpp`
- checkpoint_format: `gguf`
- quantization: `q4_k_m`
- git_sha: `167f5d898bab823ae39ca3acf0c7f798328c3ee9`
- build_profile: `release`
- cargo_features: `model-gemma4-12b-gguf, model-gemma4-12b-qat-q4-0-gguf, model-qwen3-4b-gguf, model-gemma4-e2b-gguf, model-gemma4-e4b-gguf, model-qwen3-6-27b-gguf, llama-cpp-cuda, cuda`
- accelerator_backend_mode: `llama_cpp:cuda`
- accelerator_offload: `gpu_layers=9999;selected_device=0`
- child_build_status: `None`
- child_build_duration_ms: `None`

## Runtime Environment

- CUDA_VISIBLE_DEVICES: `None`
- MOTLIE_GGUF_BINDGEN_INCLUDE_WIRED: `true`
- MOTLIE_MODEL_FORCE_CPU: `None`
- MOTLIE_MODEL_GPU_LAYERS: `None`
- MOTLIE_PAGED_ATTN_CONTEXT: `None`

## Repro Command

```sh
/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/release/evals run --bundle gemma4_e4b_gguf --scenario tool_use_weather_cel_smoke --profile dgx-spark --root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/artifacts/models/hf-cache --jsonl /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-postfix-167f5d89/curated-v2-smoke/curated-v2-smoke-postfix-1781146615-167f5d89-spark-2f6e-aarch64-cuda-release-regression/results.jsonl --run-id curated-v2-smoke-postfix-1781146615-167f5d89-spark-2f6e-aarch64-cuda-release-regression --snapshot-id curated-v2-smoke --cell-id gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m --depth smoke --checkpoint-format gguf --artifact-quantization q4_k_m --model-family gemma --backend llama_cpp --requested-accelerator cuda --quiet-backend-logs --precision q4
```

## Child Log Tail

```text
[tail excerpt: last 180 of 1373 lines]
llama_kv_cache: layer  26: does not have KV cache
llama_kv_cache: layer  27: does not have KV cache
llama_kv_cache: layer  28: does not have KV cache
llama_kv_cache: layer  29: does not have KV cache
llama_kv_cache: layer  30: does not have KV cache
llama_kv_cache: layer  31: does not have KV cache
llama_kv_cache: layer  32: does not have KV cache
llama_kv_cache: layer  33: does not have KV cache
llama_kv_cache: layer  34: does not have KV cache
llama_kv_cache: layer  35: does not have KV cache
llama_kv_cache: layer  36: does not have KV cache
llama_kv_cache: layer  37: does not have KV cache
llama_kv_cache: layer  38: does not have KV cache
llama_kv_cache: layer  39: does not have KV cache
llama_kv_cache: layer  40: does not have KV cache
llama_kv_cache: layer  41: does not have KV cache
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: no reuse
llama_kv_cache: - layer  16: no reuse
llama_kv_cache: - layer  17: no reuse
llama_kv_cache: - layer  18: no reuse
llama_kv_cache: - layer  19: no reuse
llama_kv_cache: - layer  20: no reuse
llama_kv_cache: - layer  21: no reuse
llama_kv_cache: - layer  22: no reuse
llama_kv_cache: - layer  23: no reuse
llama_kv_cache: - layer  24: filtered
llama_kv_cache: - layer  25: filtered
llama_kv_cache: - layer  26: filtered
llama_kv_cache: - layer  27: filtered
llama_kv_cache: - layer  28: filtered
llama_kv_cache: - layer  29: reuse layer 23, is_swa = 0
llama_kv_cache: - layer  30: filtered
llama_kv_cache: - layer  31: filtered
llama_kv_cache: - layer  32: filtered
llama_kv_cache: - layer  33: filtered
llama_kv_cache: - layer  34: filtered
llama_kv_cache: - layer  35: reuse layer 23, is_swa = 0
llama_kv_cache: - layer  36: filtered
llama_kv_cache: - layer  37: filtered
llama_kv_cache: - layer  38: filtered
llama_kv_cache: - layer  39: filtered
llama_kv_cache: - layer  40: filtered
llama_kv_cache: - layer  41: reuse layer 23, is_swa = 0
llama_kv_cache:      CUDA0 KV buffer size =   512.00 MiB
llama_kv_cache: size =  512.00 MiB ( 32768 cells,   4 layers,  1/1 seqs), K (f16):  256.00 MiB, V (f16):  256.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 512
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 512
llama_kv_cache_iswa: creating     SWA KV cache, size = 32768 cells
llama_kv_cache: layer   0: dev = CUDA0
llama_kv_cache: layer   1: dev = CUDA0
llama_kv_cache: layer   2: dev = CUDA0
llama_kv_cache: layer   3: dev = CUDA0
llama_kv_cache: layer   4: dev = CUDA0
llama_kv_cache: layer   5: filtered
llama_kv_cache: layer   6: dev = CUDA0
llama_kv_cache: layer   7: dev = CUDA0
llama_kv_cache: layer   8: dev = CUDA0
llama_kv_cache: layer   9: dev = CUDA0
llama_kv_cache: layer  10: dev = CUDA0
llama_kv_cache: layer  11: filtered
llama_kv_cache: layer  12: dev = CUDA0
llama_kv_cache: layer  13: dev = CUDA0
llama_kv_cache: layer  14: dev = CUDA0
llama_kv_cache: layer  15: dev = CUDA0
llama_kv_cache: layer  16: dev = CUDA0
llama_kv_cache: layer  17: filtered
llama_kv_cache: layer  18: dev = CUDA0
llama_kv_cache: layer  19: dev = CUDA0
llama_kv_cache: layer  20: dev = CUDA0
llama_kv_cache: layer  21: dev = CUDA0
llama_kv_cache: layer  22: dev = CUDA0
llama_kv_cache: layer  23: filtered
llama_kv_cache: layer  24: does not have KV cache
llama_kv_cache: layer  25: does not have KV cache
llama_kv_cache: layer  26: does not have KV cache
llama_kv_cache: layer  27: does not have KV cache
llama_kv_cache: layer  28: does not have KV cache
llama_kv_cache: layer  29: does not have KV cache
llama_kv_cache: layer  30: does not have KV cache
llama_kv_cache: layer  31: does not have KV cache
llama_kv_cache: layer  32: does not have KV cache
llama_kv_cache: layer  33: does not have KV cache
llama_kv_cache: layer  34: does not have KV cache
llama_kv_cache: layer  35: does not have KV cache
llama_kv_cache: layer  36: does not have KV cache
llama_kv_cache: layer  37: does not have KV cache
llama_kv_cache: layer  38: does not have KV cache
llama_kv_cache: layer  39: does not have KV cache
llama_kv_cache: layer  40: does not have KV cache
llama_kv_cache: layer  41: does not have KV cache
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: no reuse
llama_kv_cache: - layer  16: no reuse
llama_kv_cache: - layer  17: no reuse
llama_kv_cache: - layer  18: no reuse
llama_kv_cache: - layer  19: no reuse
llama_kv_cache: - layer  20: no reuse
llama_kv_cache: - layer  21: no reuse
llama_kv_cache: - layer  22: no reuse
llama_kv_cache: - layer  23: no reuse
llama_kv_cache: - layer  24: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  25: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  26: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  27: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  28: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  29: filtered
llama_kv_cache: - layer  30: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  31: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  32: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  33: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  34: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  35: filtered
llama_kv_cache: - layer  36: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  37: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  38: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  39: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  40: reuse layer 22, is_swa = 1
llama_kv_cache: - layer  41: filtered
llama_kv_cache:      CUDA0 KV buffer size =  1280.00 MiB
llama_kv_cache: size = 1280.00 MiB ( 32768 cells,  20 layers,  1/1 seqs), K (f16):  640.00 MiB, V (f16):  640.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 2
sched_reserve: reserving ...
sched_reserve: max_nodes = 5768
sched_reserve: reserving full memory module
sched_reserve: worst-case: n_tokens = 512, n_seqs = 1, n_outputs = 1
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: fused Gated Delta Net (autoregressive) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =   16, n_seqs =  1, n_outputs =   16
sched_reserve: fused Gated Delta Net (chunked) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
sched_reserve:      CUDA0 compute buffer size =   569.01 MiB
sched_reserve:  CUDA_Host compute buffer size =   159.02 MiB
sched_reserve: graph nodes  = 1867
sched_reserve: graph splits = 2
sched_reserve: reserve took 117.15 ms, sched copies = 1
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
ggml_backend_cuda_graph_compute: CUDA graph warmup reset
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
~llama_context:      CUDA0 compute buffer size is 569.0098 MiB, matches expectation of 569.0098 MiB
~llama_context:  CUDA_Host compute buffer size is 159.0234 MiB, matches expectation of 159.0234 MiB
REPRO: /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/release/evals run --bundle gemma4_e4b_gguf --scenario tool_use_weather_cel_smoke --profile dgx-spark --root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/artifacts/models/hf-cache --jsonl /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-postfix-167f5d89/curated-v2-smoke/curated-v2-smoke-postfix-1781146615-167f5d89-spark-2f6e-aarch64-cuda-release-regression/results.jsonl --run-id curated-v2-smoke-postfix-1781146615-167f5d89-spark-2f6e-aarch64-cuda-release-regression --snapshot-id curated-v2-smoke --cell-id gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m --depth smoke --checkpoint-format gguf --artifact-quantization q4_k_m --model-family gemma --backend llama_cpp --requested-accelerator cuda --quiet-backend-logs --precision q4
```

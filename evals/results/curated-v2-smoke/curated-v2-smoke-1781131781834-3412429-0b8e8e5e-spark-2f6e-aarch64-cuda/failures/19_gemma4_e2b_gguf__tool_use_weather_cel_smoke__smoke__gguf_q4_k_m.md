# gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m

- outcome: `blocked`
- reason: `resource_gate_failed`
- acceptance: `blocked`
- failure_reason: `resources section not accepted: resource metric gpu_memory_peak_bytes blocked: CUDA peak VRAM sampler not instrumented`
- bundle: `gemma4_e2b_gguf`
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
- git_sha: `0b8e8e5ecb53c5256037b8970041446a1637515b`
- build_profile: `debug`
- cargo_features: `model-gemma4-e2b-gguf, llama-cpp-cuda, cuda`
- accelerator_backend_mode: `llama_cpp:cuda`
- accelerator_offload: `gpu_layers=9999;selected_device=0`
- child_build_status: `0`
- child_build_duration_ms: `287`

## Runtime Environment

- CUDA_VISIBLE_DEVICES: `None`
- MOTLIE_GGUF_BINDGEN_INCLUDE_WIRED: `true`
- MOTLIE_MODEL_FORCE_CPU: `None`
- MOTLIE_MODEL_GPU_LAYERS: `None`
- MOTLIE_PAGED_ATTN_CONTEXT: `None`

## Repro Command

```sh
/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/evals run --bundle gemma4_e2b_gguf --scenario tool_use_weather_cel_smoke --profile dgx-spark --root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models/../../artifacts/models/hf-cache --jsonl /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e/curated-v2-smoke/curated-v2-smoke-1781131781834-3412429-0b8e8e5e-spark-2f6e-aarch64-cuda/results.jsonl --run-id curated-v2-smoke-1781131781834-3412429-0b8e8e5e-spark-2f6e-aarch64-cuda --snapshot-id curated-v2-smoke --cell-id gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m --depth smoke --checkpoint-format gguf --artifact-quantization q4_k_m --model-family gemma --backend llama_cpp --requested-accelerator cuda --child-build-log /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e/curated-v2-smoke/curated-v2-smoke-1781131781834-3412429-0b8e8e5e-spark-2f6e-aarch64-cuda/logs/gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m.log --child-build-status 0 --child-build-duration-ms 287 --quiet-backend-logs
```

## Child Log Tail

```text
[tail excerpt: last 180 of 1191 lines]
llama_context:  CUDA_Host  output buffer size =     1.00 MiB
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache_iswa: creating non-SWA KV cache, size = 4096 cells
llama_kv_cache: layer   0: filtered
llama_kv_cache: layer   1: filtered
llama_kv_cache: layer   2: filtered
llama_kv_cache: layer   3: filtered
llama_kv_cache: layer   4: dev = CUDA0
llama_kv_cache: layer   5: filtered
llama_kv_cache: layer   6: filtered
llama_kv_cache: layer   7: filtered
llama_kv_cache: layer   8: filtered
llama_kv_cache: layer   9: dev = CUDA0
llama_kv_cache: layer  10: filtered
llama_kv_cache: layer  11: filtered
llama_kv_cache: layer  12: filtered
llama_kv_cache: layer  13: filtered
llama_kv_cache: layer  14: dev = CUDA0
llama_kv_cache: layer  15: does not have KV cache
llama_kv_cache: layer  16: does not have KV cache
llama_kv_cache: layer  17: does not have KV cache
llama_kv_cache: layer  18: does not have KV cache
llama_kv_cache: layer  19: does not have KV cache
llama_kv_cache: layer  20: does not have KV cache
llama_kv_cache: layer  21: does not have KV cache
llama_kv_cache: layer  22: does not have KV cache
llama_kv_cache: layer  23: does not have KV cache
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
llama_kv_cache: - layer  15: filtered
llama_kv_cache: - layer  16: filtered
llama_kv_cache: - layer  17: filtered
llama_kv_cache: - layer  18: filtered
llama_kv_cache: - layer  19: reuse layer 14, is_swa = 0
llama_kv_cache: - layer  20: filtered
llama_kv_cache: - layer  21: filtered
llama_kv_cache: - layer  22: filtered
llama_kv_cache: - layer  23: filtered
llama_kv_cache: - layer  24: reuse layer 14, is_swa = 0
llama_kv_cache: - layer  25: filtered
llama_kv_cache: - layer  26: filtered
llama_kv_cache: - layer  27: filtered
llama_kv_cache: - layer  28: filtered
llama_kv_cache: - layer  29: reuse layer 14, is_swa = 0
llama_kv_cache: - layer  30: filtered
llama_kv_cache: - layer  31: filtered
llama_kv_cache: - layer  32: filtered
llama_kv_cache: - layer  33: filtered
llama_kv_cache: - layer  34: reuse layer 14, is_swa = 0
llama_kv_cache:      CUDA0 KV buffer size =    24.00 MiB
llama_kv_cache: size =   24.00 MiB (  4096 cells,   3 layers,  1/1 seqs), K (f16):   12.00 MiB, V (f16):   12.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 512
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 512
llama_kv_cache_iswa: creating     SWA KV cache, size = 4096 cells
llama_kv_cache: layer   0: dev = CUDA0
llama_kv_cache: layer   1: dev = CUDA0
llama_kv_cache: layer   2: dev = CUDA0
llama_kv_cache: layer   3: dev = CUDA0
llama_kv_cache: layer   4: filtered
llama_kv_cache: layer   5: dev = CUDA0
llama_kv_cache: layer   6: dev = CUDA0
llama_kv_cache: layer   7: dev = CUDA0
llama_kv_cache: layer   8: dev = CUDA0
llama_kv_cache: layer   9: filtered
llama_kv_cache: layer  10: dev = CUDA0
llama_kv_cache: layer  11: dev = CUDA0
llama_kv_cache: layer  12: dev = CUDA0
llama_kv_cache: layer  13: dev = CUDA0
llama_kv_cache: layer  14: filtered
llama_kv_cache: layer  15: does not have KV cache
llama_kv_cache: layer  16: does not have KV cache
llama_kv_cache: layer  17: does not have KV cache
llama_kv_cache: layer  18: does not have KV cache
llama_kv_cache: layer  19: does not have KV cache
llama_kv_cache: layer  20: does not have KV cache
llama_kv_cache: layer  21: does not have KV cache
llama_kv_cache: layer  22: does not have KV cache
llama_kv_cache: layer  23: does not have KV cache
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
llama_kv_cache: - layer  15: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  16: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  17: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  18: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  19: filtered
llama_kv_cache: - layer  20: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  21: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  22: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  23: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  24: filtered
llama_kv_cache: - layer  25: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  26: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  27: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  28: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  29: filtered
llama_kv_cache: - layer  30: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  31: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  32: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  33: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  34: filtered
llama_kv_cache:      CUDA0 KV buffer size =    48.00 MiB
llama_kv_cache: size =   48.00 MiB (  4096 cells,  12 layers,  1/1 seqs), K (f16):   24.00 MiB, V (f16):   24.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 2
sched_reserve: reserving ...
sched_reserve: max_nodes = 4816
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
sched_reserve:      CUDA0 compute buffer size =   515.00 MiB
sched_reserve:  CUDA_Host compute buffer size =    39.52 MiB
sched_reserve: graph nodes  = 1500
sched_reserve: graph splits = 2
sched_reserve: reserve took 42.99 ms, sched copies = 1
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
ggml_backend_cuda_graph_compute: CUDA graph warmup reset
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
~llama_context:      CUDA0 compute buffer size is 515.0000 MiB, matches expectation of 515.0000 MiB
~llama_context:  CUDA_Host compute buffer size is  39.5234 MiB, matches expectation of  39.5234 MiB
```

# gemma4_12b_qat_q4_0_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0

- outcome: `blocked`
- reason: `resource_gate_failed`
- acceptance: `blocked`
- failure_reason: `resources section not accepted: resource metric gpu_memory_peak_bytes blocked: CUDA peak VRAM sampler not instrumented`
- bundle: `gemma4_12b_qat_q4_0_gguf`
- capability: `tool_use`
- scenario: `tool_use_weather_cel_smoke`
- profile: `dgx-spark`
- host_id: `spark-2f6e`
- platform: `linux/aarch64`
- requested_accelerator: `cuda`
- resolved_accelerator: `cuda`
- backend: `llama_cpp`
- checkpoint_format: `gguf`
- quantization: `q4_0`
- git_sha: `0b8e8e5ecb53c5256037b8970041446a1637515b`
- build_profile: `debug`
- cargo_features: `model-gemma4-12b-qat-q4-0-gguf, llama-cpp-cuda, cuda`
- accelerator_backend_mode: `llama_cpp:cuda`
- accelerator_offload: `gpu_layers=9999;selected_device=0`
- child_build_status: `0`
- child_build_duration_ms: `303`

## Runtime Environment

- CUDA_VISIBLE_DEVICES: `None`
- MOTLIE_GGUF_BINDGEN_INCLUDE_WIRED: `true`
- MOTLIE_MODEL_FORCE_CPU: `None`
- MOTLIE_MODEL_GPU_LAYERS: `None`
- MOTLIE_PAGED_ATTN_CONTEXT: `None`

## Repro Command

```sh
/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/evals run --bundle gemma4_12b_qat_q4_0_gguf --scenario tool_use_weather_cel_smoke --profile dgx-spark --root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models/../../artifacts/models/hf-cache --jsonl /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e/curated-v2-smoke/curated-v2-smoke-1781131781834-3412429-0b8e8e5e-spark-2f6e-aarch64-cuda/results.jsonl --run-id curated-v2-smoke-1781131781834-3412429-0b8e8e5e-spark-2f6e-aarch64-cuda --snapshot-id curated-v2-smoke --cell-id gemma4_12b_qat_q4_0_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0 --depth smoke --checkpoint-format gguf --artifact-quantization q4_0 --model-family gemma --backend llama_cpp --requested-accelerator cuda --child-build-log /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e/curated-v2-smoke/curated-v2-smoke-1781131781834-3412429-0b8e8e5e-spark-2f6e-aarch64-cuda/logs/gemma4_12b_qat_q4_0_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0.log --child-build-status 0 --child-build-duration-ms 303 --quiet-backend-logs
```

## Child Log Tail

```text
[tail excerpt: last 180 of 1361 lines]
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
llama_kv_cache: - layer  24: no reuse
llama_kv_cache: - layer  25: no reuse
llama_kv_cache: - layer  26: no reuse
llama_kv_cache: - layer  27: no reuse
llama_kv_cache: - layer  28: no reuse
llama_kv_cache: - layer  29: no reuse
llama_kv_cache: - layer  30: no reuse
llama_kv_cache: - layer  31: no reuse
llama_kv_cache: - layer  32: no reuse
llama_kv_cache: - layer  33: no reuse
llama_kv_cache: - layer  34: no reuse
llama_kv_cache: - layer  35: no reuse
llama_kv_cache: - layer  36: no reuse
llama_kv_cache: - layer  37: no reuse
llama_kv_cache: - layer  38: no reuse
llama_kv_cache: - layer  39: no reuse
llama_kv_cache: - layer  40: no reuse
llama_kv_cache: - layer  41: no reuse
llama_kv_cache: - layer  42: no reuse
llama_kv_cache: - layer  43: no reuse
llama_kv_cache: - layer  44: no reuse
llama_kv_cache: - layer  45: no reuse
llama_kv_cache: - layer  46: no reuse
llama_kv_cache: - layer  47: no reuse
llama_kv_cache:      CUDA0 KV buffer size =   512.00 MiB
llama_kv_cache: size =  512.00 MiB ( 32768 cells,   8 layers,  1/1 seqs), K (f16):  256.00 MiB, V (f16):  256.00 MiB
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
llama_kv_cache: layer  24: dev = CUDA0
llama_kv_cache: layer  25: dev = CUDA0
llama_kv_cache: layer  26: dev = CUDA0
llama_kv_cache: layer  27: dev = CUDA0
llama_kv_cache: layer  28: dev = CUDA0
llama_kv_cache: layer  29: filtered
llama_kv_cache: layer  30: dev = CUDA0
llama_kv_cache: layer  31: dev = CUDA0
llama_kv_cache: layer  32: dev = CUDA0
llama_kv_cache: layer  33: dev = CUDA0
llama_kv_cache: layer  34: dev = CUDA0
llama_kv_cache: layer  35: filtered
llama_kv_cache: layer  36: dev = CUDA0
llama_kv_cache: layer  37: dev = CUDA0
llama_kv_cache: layer  38: dev = CUDA0
llama_kv_cache: layer  39: dev = CUDA0
llama_kv_cache: layer  40: dev = CUDA0
llama_kv_cache: layer  41: filtered
llama_kv_cache: layer  42: dev = CUDA0
llama_kv_cache: layer  43: dev = CUDA0
llama_kv_cache: layer  44: dev = CUDA0
llama_kv_cache: layer  45: dev = CUDA0
llama_kv_cache: layer  46: dev = CUDA0
llama_kv_cache: layer  47: filtered
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
llama_kv_cache: - layer  24: no reuse
llama_kv_cache: - layer  25: no reuse
llama_kv_cache: - layer  26: no reuse
llama_kv_cache: - layer  27: no reuse
llama_kv_cache: - layer  28: no reuse
llama_kv_cache: - layer  29: no reuse
llama_kv_cache: - layer  30: no reuse
llama_kv_cache: - layer  31: no reuse
llama_kv_cache: - layer  32: no reuse
llama_kv_cache: - layer  33: no reuse
llama_kv_cache: - layer  34: no reuse
llama_kv_cache: - layer  35: no reuse
llama_kv_cache: - layer  36: no reuse
llama_kv_cache: - layer  37: no reuse
llama_kv_cache: - layer  38: no reuse
llama_kv_cache: - layer  39: no reuse
llama_kv_cache: - layer  40: no reuse
llama_kv_cache: - layer  41: no reuse
llama_kv_cache: - layer  42: no reuse
llama_kv_cache: - layer  43: no reuse
llama_kv_cache: - layer  44: no reuse
llama_kv_cache: - layer  45: no reuse
llama_kv_cache: - layer  46: no reuse
llama_kv_cache: - layer  47: no reuse
llama_kv_cache:      CUDA0 KV buffer size = 10240.00 MiB
llama_kv_cache: size = 10240.00 MiB ( 32768 cells,  40 layers,  1/1 seqs), K (f16): 5120.00 MiB, V (f16): 5120.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 2
sched_reserve: reserving ...
sched_reserve: max_nodes = 5344
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
sched_reserve:      CUDA0 compute buffer size =   519.50 MiB
sched_reserve:  CUDA_Host compute buffer size =   143.02 MiB
sched_reserve: graph nodes  = 1972
sched_reserve: graph splits = 2
sched_reserve: reserve took 98.26 ms, sched copies = 1
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
ggml_backend_cuda_graph_compute: CUDA graph warmup reset
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
~llama_context:      CUDA0 compute buffer size is 519.5000 MiB, matches expectation of 519.5000 MiB
~llama_context:  CUDA_Host compute buffer size is 143.0215 MiB, matches expectation of 143.0215 MiB
```

# qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m

- outcome: `blocked`
- reason: `resource_gate_failed`
- acceptance: `blocked`
- failure_reason: `resources section not accepted: resource metric gpu_memory_peak_bytes blocked: CUDA peak VRAM sampler not instrumented`
- bundle: `qwen3_4b_gguf`
- capability: `perf`
- scenario: `bench_chat_startup`
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
/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/release/evals run --bundle qwen3_4b_gguf --scenario bench_chat_startup --profile dgx-spark --root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/artifacts/models/hf-cache --jsonl /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-postfix-167f5d89/curated-v2-smoke/curated-v2-smoke-postfix-1781146615-167f5d89-spark-2f6e-aarch64-cuda-release-regression/results.jsonl --run-id curated-v2-smoke-postfix-1781146615-167f5d89-spark-2f6e-aarch64-cuda-release-regression --snapshot-id curated-v2-smoke --cell-id qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m --depth smoke --checkpoint-format gguf --artifact-quantization q4_k_m --model-family qwen3 --backend llama_cpp --requested-accelerator cuda --quiet-backend-logs --precision q4
```

## Child Log Tail

```text
[tail excerpt: last 180 of 896 lines]
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: fused Gated Delta Net (autoregressive) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =   16, n_seqs =  1, n_outputs =   16
sched_reserve: fused Gated Delta Net (chunked) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
sched_reserve:      CUDA0 compute buffer size =   301.75 MiB
sched_reserve:  CUDA_Host compute buffer size =    18.01 MiB
sched_reserve: graph nodes  = 1267
sched_reserve: graph splits = 2
sched_reserve: reserve took 24.32 ms, sched copies = 1
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
ggml_backend_cuda_graph_compute: CUDA graph warmup reset
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
~llama_context:      CUDA0 compute buffer size is 301.7500 MiB, matches expectation of 301.7500 MiB
~llama_context:  CUDA_Host compute buffer size is  18.0137 MiB, matches expectation of  18.0137 MiB
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 4096
llama_context: n_ctx_seq     = 4096
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_seq (4096) < n_ctx_train (40960) -- the full capacity of the model will not be utilized
set_abort_callback: call
llama_context:  CUDA_Host  output buffer size =     0.58 MiB
llama_kv_cache: layer   0: dev = CUDA0
llama_kv_cache: layer   1: dev = CUDA0
llama_kv_cache: layer   2: dev = CUDA0
llama_kv_cache: layer   3: dev = CUDA0
llama_kv_cache: layer   4: dev = CUDA0
llama_kv_cache: layer   5: dev = CUDA0
llama_kv_cache: layer   6: dev = CUDA0
llama_kv_cache: layer   7: dev = CUDA0
llama_kv_cache: layer   8: dev = CUDA0
llama_kv_cache: layer   9: dev = CUDA0
llama_kv_cache: layer  10: dev = CUDA0
llama_kv_cache: layer  11: dev = CUDA0
llama_kv_cache: layer  12: dev = CUDA0
llama_kv_cache: layer  13: dev = CUDA0
llama_kv_cache: layer  14: dev = CUDA0
llama_kv_cache: layer  15: dev = CUDA0
llama_kv_cache: layer  16: dev = CUDA0
llama_kv_cache: layer  17: dev = CUDA0
llama_kv_cache: layer  18: dev = CUDA0
llama_kv_cache: layer  19: dev = CUDA0
llama_kv_cache: layer  20: dev = CUDA0
llama_kv_cache: layer  21: dev = CUDA0
llama_kv_cache: layer  22: dev = CUDA0
llama_kv_cache: layer  23: dev = CUDA0
llama_kv_cache: layer  24: dev = CUDA0
llama_kv_cache: layer  25: dev = CUDA0
llama_kv_cache: layer  26: dev = CUDA0
llama_kv_cache: layer  27: dev = CUDA0
llama_kv_cache: layer  28: dev = CUDA0
llama_kv_cache: layer  29: dev = CUDA0
llama_kv_cache: layer  30: dev = CUDA0
llama_kv_cache: layer  31: dev = CUDA0
llama_kv_cache: layer  32: dev = CUDA0
llama_kv_cache: layer  33: dev = CUDA0
llama_kv_cache: layer  34: dev = CUDA0
llama_kv_cache: layer  35: dev = CUDA0
llama_kv_cache:      CUDA0 KV buffer size =   576.00 MiB
llama_kv_cache: size =  576.00 MiB (  4096 cells,  36 layers,  1/1 seqs), K (f16):  288.00 MiB, V (f16):  288.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 128
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 128
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 2
sched_reserve: reserving ...
sched_reserve: max_nodes = 3192
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
sched_reserve:      CUDA0 compute buffer size =   301.75 MiB
sched_reserve:  CUDA_Host compute buffer size =    18.01 MiB
sched_reserve: graph nodes  = 1267
sched_reserve: graph splits = 2
sched_reserve: reserve took 23.97 ms, sched copies = 1
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
ggml_backend_cuda_graph_compute: CUDA graph warmup reset
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
~llama_context:      CUDA0 compute buffer size is 301.7500 MiB, matches expectation of 301.7500 MiB
~llama_context:  CUDA_Host compute buffer size is  18.0137 MiB, matches expectation of  18.0137 MiB
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 4096
llama_context: n_ctx_seq     = 4096
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_seq (4096) < n_ctx_train (40960) -- the full capacity of the model will not be utilized
set_abort_callback: call
llama_context:  CUDA_Host  output buffer size =     0.58 MiB
llama_kv_cache: layer   0: dev = CUDA0
llama_kv_cache: layer   1: dev = CUDA0
llama_kv_cache: layer   2: dev = CUDA0
llama_kv_cache: layer   3: dev = CUDA0
llama_kv_cache: layer   4: dev = CUDA0
llama_kv_cache: layer   5: dev = CUDA0
llama_kv_cache: layer   6: dev = CUDA0
llama_kv_cache: layer   7: dev = CUDA0
llama_kv_cache: layer   8: dev = CUDA0
llama_kv_cache: layer   9: dev = CUDA0
llama_kv_cache: layer  10: dev = CUDA0
llama_kv_cache: layer  11: dev = CUDA0
llama_kv_cache: layer  12: dev = CUDA0
llama_kv_cache: layer  13: dev = CUDA0
llama_kv_cache: layer  14: dev = CUDA0
llama_kv_cache: layer  15: dev = CUDA0
llama_kv_cache: layer  16: dev = CUDA0
llama_kv_cache: layer  17: dev = CUDA0
llama_kv_cache: layer  18: dev = CUDA0
llama_kv_cache: layer  19: dev = CUDA0
llama_kv_cache: layer  20: dev = CUDA0
llama_kv_cache: layer  21: dev = CUDA0
llama_kv_cache: layer  22: dev = CUDA0
llama_kv_cache: layer  23: dev = CUDA0
llama_kv_cache: layer  24: dev = CUDA0
llama_kv_cache: layer  25: dev = CUDA0
llama_kv_cache: layer  26: dev = CUDA0
llama_kv_cache: layer  27: dev = CUDA0
llama_kv_cache: layer  28: dev = CUDA0
llama_kv_cache: layer  29: dev = CUDA0
llama_kv_cache: layer  30: dev = CUDA0
llama_kv_cache: layer  31: dev = CUDA0
llama_kv_cache: layer  32: dev = CUDA0
llama_kv_cache: layer  33: dev = CUDA0
llama_kv_cache: layer  34: dev = CUDA0
llama_kv_cache: layer  35: dev = CUDA0
llama_kv_cache:      CUDA0 KV buffer size =   576.00 MiB
llama_kv_cache: size =  576.00 MiB (  4096 cells,  36 layers,  1/1 seqs), K (f16):  288.00 MiB, V (f16):  288.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 128
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 128
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 2
sched_reserve: reserving ...
sched_reserve: max_nodes = 3192
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
sched_reserve:      CUDA0 compute buffer size =   301.75 MiB
sched_reserve:  CUDA_Host compute buffer size =    18.01 MiB
sched_reserve: graph nodes  = 1267
sched_reserve: graph splits = 2
sched_reserve: reserve took 24.18 ms, sched copies = 1
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
ggml_backend_cuda_graph_compute: CUDA graph warmup reset
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
~llama_context:      CUDA0 compute buffer size is 301.7500 MiB, matches expectation of 301.7500 MiB
~llama_context:  CUDA_Host compute buffer size is  18.0137 MiB, matches expectation of  18.0137 MiB
REPRO: /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/release/evals run --bundle qwen3_4b_gguf --scenario bench_chat_startup --profile dgx-spark --root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/artifacts/models/hf-cache --jsonl /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-postfix-167f5d89/curated-v2-smoke/curated-v2-smoke-postfix-1781146615-167f5d89-spark-2f6e-aarch64-cuda-release-regression/results.jsonl --run-id curated-v2-smoke-postfix-1781146615-167f5d89-spark-2f6e-aarch64-cuda-release-regression --snapshot-id curated-v2-smoke --cell-id qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m --depth smoke --checkpoint-format gguf --artifact-quantization q4_k_m --model-family qwen3 --backend llama_cpp --requested-accelerator cuda --quiet-backend-logs --precision q4
```

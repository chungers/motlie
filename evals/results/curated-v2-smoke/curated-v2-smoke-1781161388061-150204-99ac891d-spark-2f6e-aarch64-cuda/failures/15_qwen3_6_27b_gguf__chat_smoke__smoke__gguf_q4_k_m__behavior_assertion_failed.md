# qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m

## Record

- schema_version: 3
- git_sha: 99ac891d8a2adabe823cce61b2a9fec0aa5dbde3
- run_id: curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda
- profile: dgx-spark
- host_id: spark-2f6e
- arch: aarch64
- capability: chat
- bundle_id: qwen3_6_27b_gguf
- backend: llama_cpp
- checkpoint_format: gguf
- quantization: q4_k_m
- requested_accelerator: cuda
- resolved_accelerator: cuda
- accelerator_backend_mode: llama_cpp:cuda
- accelerator_use_proof_source: backend_observation
- outcome: failed
- reason: behavior_assertion_failed
- overall_status: fail
- failure_reason: behavior assertion `min_response_chars` failed: response_chars=0 min=16
- child_build_profile: release
- child_build_status: 0

## Repro

```sh
/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/release/evals run --bundle qwen3_6_27b_gguf --scenario chat_smoke --profile dgx-spark --root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models/../../artifacts/models/hf-cache --jsonl /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-final-99ac891d/curated-v2-smoke/curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda/results.jsonl --run-id curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda --snapshot-id curated-v2-smoke --cell-id qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m --depth smoke --checkpoint-format gguf --artifact-quantization q4_k_m --precision q4 --model-family qwen3 --backend llama_cpp --requested-accelerator cuda --child-build-log /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-final-99ac891d/curated-v2-smoke/curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda/logs/qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m.log --child-build-status 0 --child-build-duration-ms 264 --quiet-backend-logs
```

Child build command:

```sh
n/a
```

## Child Log Tail

```text
set_abort_callback: call
llama_context:  CUDA_Host  output buffer size =     0.95 MiB
llama_kv_cache: layer   0: filtered
llama_kv_cache: layer   1: filtered
llama_kv_cache: layer   2: filtered
llama_kv_cache: layer   3: dev = CUDA0
llama_kv_cache: layer   4: filtered
llama_kv_cache: layer   5: filtered
llama_kv_cache: layer   6: filtered
llama_kv_cache: layer   7: dev = CUDA0
llama_kv_cache: layer   8: filtered
llama_kv_cache: layer   9: filtered
llama_kv_cache: layer  10: filtered
llama_kv_cache: layer  11: dev = CUDA0
llama_kv_cache: layer  12: filtered
llama_kv_cache: layer  13: filtered
llama_kv_cache: layer  14: filtered
llama_kv_cache: layer  15: dev = CUDA0
llama_kv_cache: layer  16: filtered
llama_kv_cache: layer  17: filtered
llama_kv_cache: layer  18: filtered
llama_kv_cache: layer  19: dev = CUDA0
llama_kv_cache: layer  20: filtered
llama_kv_cache: layer  21: filtered
llama_kv_cache: layer  22: filtered
llama_kv_cache: layer  23: dev = CUDA0
llama_kv_cache: layer  24: filtered
llama_kv_cache: layer  25: filtered
llama_kv_cache: layer  26: filtered
llama_kv_cache: layer  27: dev = CUDA0
llama_kv_cache: layer  28: filtered
llama_kv_cache: layer  29: filtered
llama_kv_cache: layer  30: filtered
llama_kv_cache: layer  31: dev = CUDA0
llama_kv_cache: layer  32: filtered
llama_kv_cache: layer  33: filtered
llama_kv_cache: layer  34: filtered
llama_kv_cache: layer  35: dev = CUDA0
llama_kv_cache: layer  36: filtered
llama_kv_cache: layer  37: filtered
llama_kv_cache: layer  38: filtered
llama_kv_cache: layer  39: dev = CUDA0
llama_kv_cache: layer  40: filtered
llama_kv_cache: layer  41: filtered
llama_kv_cache: layer  42: filtered
llama_kv_cache: layer  43: dev = CUDA0
llama_kv_cache: layer  44: filtered
llama_kv_cache: layer  45: filtered
llama_kv_cache: layer  46: filtered
llama_kv_cache: layer  47: dev = CUDA0
llama_kv_cache: layer  48: filtered
llama_kv_cache: layer  49: filtered
llama_kv_cache: layer  50: filtered
llama_kv_cache: layer  51: dev = CUDA0
llama_kv_cache: layer  52: filtered
llama_kv_cache: layer  53: filtered
llama_kv_cache: layer  54: filtered
llama_kv_cache: layer  55: dev = CUDA0
llama_kv_cache: layer  56: filtered
llama_kv_cache: layer  57: filtered
llama_kv_cache: layer  58: filtered
llama_kv_cache: layer  59: dev = CUDA0
llama_kv_cache: layer  60: filtered
llama_kv_cache: layer  61: filtered
llama_kv_cache: layer  62: filtered
llama_kv_cache: layer  63: dev = CUDA0
llama_kv_cache:      CUDA0 KV buffer size =  2048.00 MiB
llama_kv_cache: size = 2048.00 MiB ( 32768 cells,  16 layers,  1/1 seqs), K (f16): 1024.00 MiB, V (f16): 1024.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_memory_recurrent, layer   0: dev = CUDA0
llama_memory_recurrent, layer   1: dev = CUDA0
llama_memory_recurrent, layer   2: dev = CUDA0
llama_memory_recurrent: layer   3: skipped
llama_memory_recurrent, layer   4: dev = CUDA0
llama_memory_recurrent, layer   5: dev = CUDA0
llama_memory_recurrent, layer   6: dev = CUDA0
llama_memory_recurrent: layer   7: skipped
llama_memory_recurrent, layer   8: dev = CUDA0
llama_memory_recurrent, layer   9: dev = CUDA0
llama_memory_recurrent, layer  10: dev = CUDA0
llama_memory_recurrent: layer  11: skipped
llama_memory_recurrent, layer  12: dev = CUDA0
llama_memory_recurrent, layer  13: dev = CUDA0
llama_memory_recurrent, layer  14: dev = CUDA0
llama_memory_recurrent: layer  15: skipped
llama_memory_recurrent, layer  16: dev = CUDA0
llama_memory_recurrent, layer  17: dev = CUDA0
llama_memory_recurrent, layer  18: dev = CUDA0
llama_memory_recurrent: layer  19: skipped
llama_memory_recurrent, layer  20: dev = CUDA0
llama_memory_recurrent, layer  21: dev = CUDA0
llama_memory_recurrent, layer  22: dev = CUDA0
llama_memory_recurrent: layer  23: skipped
llama_memory_recurrent, layer  24: dev = CUDA0
llama_memory_recurrent, layer  25: dev = CUDA0
llama_memory_recurrent, layer  26: dev = CUDA0
llama_memory_recurrent: layer  27: skipped
llama_memory_recurrent, layer  28: dev = CUDA0
llama_memory_recurrent, layer  29: dev = CUDA0
llama_memory_recurrent, layer  30: dev = CUDA0
llama_memory_recurrent: layer  31: skipped
llama_memory_recurrent, layer  32: dev = CUDA0
llama_memory_recurrent, layer  33: dev = CUDA0
llama_memory_recurrent, layer  34: dev = CUDA0
llama_memory_recurrent: layer  35: skipped
llama_memory_recurrent, layer  36: dev = CUDA0
llama_memory_recurrent, layer  37: dev = CUDA0
llama_memory_recurrent, layer  38: dev = CUDA0
llama_memory_recurrent: layer  39: skipped
llama_memory_recurrent, layer  40: dev = CUDA0
llama_memory_recurrent, layer  41: dev = CUDA0
llama_memory_recurrent, layer  42: dev = CUDA0
llama_memory_recurrent: layer  43: skipped
llama_memory_recurrent, layer  44: dev = CUDA0
llama_memory_recurrent, layer  45: dev = CUDA0
llama_memory_recurrent, layer  46: dev = CUDA0
llama_memory_recurrent: layer  47: skipped
llama_memory_recurrent, layer  48: dev = CUDA0
llama_memory_recurrent, layer  49: dev = CUDA0
llama_memory_recurrent, layer  50: dev = CUDA0
llama_memory_recurrent: layer  51: skipped
llama_memory_recurrent, layer  52: dev = CUDA0
llama_memory_recurrent, layer  53: dev = CUDA0
llama_memory_recurrent, layer  54: dev = CUDA0
llama_memory_recurrent: layer  55: skipped
llama_memory_recurrent, layer  56: dev = CUDA0
llama_memory_recurrent, layer  57: dev = CUDA0
llama_memory_recurrent, layer  58: dev = CUDA0
llama_memory_recurrent: layer  59: skipped
llama_memory_recurrent, layer  60: dev = CUDA0
llama_memory_recurrent, layer  61: dev = CUDA0
llama_memory_recurrent, layer  62: dev = CUDA0
llama_memory_recurrent: layer  63: skipped
llama_memory_recurrent:      CUDA0 RS buffer size =   149.62 MiB
llama_memory_recurrent: size =  149.62 MiB (     1 cells,  64 layers,  1 seqs), R (f32):    5.62 MiB, S (f32):  144.00 MiB
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 2
sched_reserve: reserving ...
sched_reserve: max_nodes = 27232
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
sched_reserve:      CUDA0 compute buffer size =   495.00 MiB
sched_reserve:  CUDA_Host compute buffer size =    84.02 MiB
sched_reserve: graph nodes  = 3657
sched_reserve: graph splits = 2
sched_reserve: reserve took 232.87 ms, sched copies = 1
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
~llama_context:      CUDA0 compute buffer size is 495.0000 MiB, matches expectation of 495.0000 MiB
~llama_context:  CUDA_Host compute buffer size is  84.0196 MiB, matches expectation of  84.0196 MiB

```

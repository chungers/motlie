# Failure: gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m

- bundle: `gemma4_e2b_gguf`
- scenario: `chat_smoke`
- capability: `chat`
- model_family: `gemma`
- backend: `llama_cpp`
- checkpoint_format: `gguf`
- quantization: `q4_k_m`
- runtime_precision: `q4`
- platform: `amd1 x86 CPU`
- profile: `local-cpu-x86_64`
- host: `amd-ryzen7-1`
- git_sha: `167f5d898bab823ae39ca3acf0c7f798328c3ee9`
- build_profile: `release`
- terminal_outcome: `failed`
- reason: `resource_gate_failed`
- behavior_status: `pass`
- resource_status: `fail`
- overall_status: `fail`
- failure_reason: `resources section not accepted: resource gate max_process_swap_delta_bytes=0 exceeded: process_swap_delta_peak=93.50MiB (98037760 bytes)`

## Repro Command

```sh
/tmp/motlie-regression-target-167f5d89-release/release/evals run --bundle gemma4_e2b_gguf --scenario chat_smoke --profile local-cpu-x86_64 --root /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-regression-167f5d89-20260610-1924/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-pr452-validation-20260610-1400/artifacts/models/hf-cache --jsonl /tmp/motlie-regression-results-167f5d89/results.jsonl --run-id regression-amd-x86-release-167f5d89 --snapshot-id curated-v2-smoke --depth smoke --requested-accelerator cpu --download-artifacts --quiet-backend-logs --cell-id gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m --checkpoint-format gguf --artifact-quantization q4_k_m --model-family gemma --backend llama_cpp --precision q4
```

## Log Tail

```text
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
llama_kv_cache:        CPU KV buffer size =    48.00 MiB
llama_kv_cache: size =   48.00 MiB (  4096 cells,  12 layers,  1/1 seqs), K (f16):   24.00 MiB, V (f16):   24.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 1
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
sched_reserve:        CPU compute buffer size =   518.00 MiB
sched_reserve: graph nodes  = 1500
sched_reserve: graph splits = 1
sched_reserve: reserve took 5.49 ms, sched copies = 1
~llama_context:        CPU compute buffer size is 518.0020 MiB, matches expectation of 518.0020 MiB
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
llama_context: n_ctx_seq (4096) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
set_abort_callback: call
llama_context:        CPU  output buffer size =     1.00 MiB
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache_iswa: creating non-SWA KV cache, size = 4096 cells
llama_kv_cache: layer   0: filtered
llama_kv_cache: layer   1: filtered
llama_kv_cache: layer   2: filtered
llama_kv_cache: layer   3: filtered
llama_kv_cache: layer   4: dev = CPU
llama_kv_cache: layer   5: filtered
llama_kv_cache: layer   6: filtered
llama_kv_cache: layer   7: filtered
llama_kv_cache: layer   8: filtered
llama_kv_cache: layer   9: dev = CPU
llama_kv_cache: layer  10: filtered
llama_kv_cache: layer  11: filtered
llama_kv_cache: layer  12: filtered
llama_kv_cache: layer  13: filtered
llama_kv_cache: layer  14: dev = CPU
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
llama_kv_cache:        CPU KV buffer size =    24.00 MiB
llama_kv_cache: size =   24.00 MiB (  4096 cells,   3 layers,  1/1 seqs), K (f16):   12.00 MiB, V (f16):   12.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 512
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 512
llama_kv_cache_iswa: creating     SWA KV cache, size = 4096 cells
llama_kv_cache: layer   0: dev = CPU
llama_kv_cache: layer   1: dev = CPU
llama_kv_cache: layer   2: dev = CPU
llama_kv_cache: layer   3: dev = CPU
llama_kv_cache: layer   4: filtered
llama_kv_cache: layer   5: dev = CPU
llama_kv_cache: layer   6: dev = CPU
llama_kv_cache: layer   7: dev = CPU
llama_kv_cache: layer   8: dev = CPU
llama_kv_cache: layer   9: filtered
llama_kv_cache: layer  10: dev = CPU
llama_kv_cache: layer  11: dev = CPU
llama_kv_cache: layer  12: dev = CPU
llama_kv_cache: layer  13: dev = CPU
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
llama_kv_cache:        CPU KV buffer size =    48.00 MiB
llama_kv_cache: size =   48.00 MiB (  4096 cells,  12 layers,  1/1 seqs), K (f16):   24.00 MiB, V (f16):   24.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 1
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
sched_reserve:        CPU compute buffer size =   518.00 MiB
sched_reserve: graph nodes  = 1500
sched_reserve: graph splits = 1
sched_reserve: reserve took 4.00 ms, sched copies = 1
~llama_context:        CPU compute buffer size is 518.0020 MiB, matches expectation of 518.0020 MiB
```

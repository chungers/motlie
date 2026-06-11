# Failure: gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m

- bundle: `gemma4_12b_gguf`
- scenario: `tool_use_weather_cel_smoke`
- capability: `tool_use`
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
- failure_reason: `resources section not accepted: resource gate max_process_swap_delta_bytes=0 exceeded: process_swap_delta_peak=12.85MiB (13475840 bytes)`

## Repro Command

```sh
/tmp/motlie-regression-target-167f5d89-release/release/evals run --bundle gemma4_12b_gguf --scenario tool_use_weather_cel_smoke --profile local-cpu-x86_64 --root /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-regression-167f5d89-20260610-1924/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-pr452-validation-20260610-1400/artifacts/models/hf-cache --jsonl /tmp/motlie-regression-results-167f5d89/results.jsonl --run-id regression-amd-x86-release-167f5d89 --snapshot-id curated-v2-smoke --depth smoke --requested-accelerator cpu --download-artifacts --quiet-backend-logs --cell-id gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m --checkpoint-format gguf --artifact-quantization q4_k_m --model-family gemma --backend llama_cpp --precision q4
```

## Log Tail

```text
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 32768
llama_context: n_ctx_seq     = 32768
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_seq (32768) < n_ctx_train (262144) -- the full capacity of the model will not be utilized
set_abort_callback: call
llama_context:        CPU  output buffer size =     1.00 MiB
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache_iswa: creating non-SWA KV cache, size = 32768 cells
llama_kv_cache: layer   0: filtered
llama_kv_cache: layer   1: filtered
llama_kv_cache: layer   2: filtered
llama_kv_cache: layer   3: filtered
llama_kv_cache: layer   4: filtered
llama_kv_cache: layer   5: dev = CPU
llama_kv_cache: layer   6: filtered
llama_kv_cache: layer   7: filtered
llama_kv_cache: layer   8: filtered
llama_kv_cache: layer   9: filtered
llama_kv_cache: layer  10: filtered
llama_kv_cache: layer  11: dev = CPU
llama_kv_cache: layer  12: filtered
llama_kv_cache: layer  13: filtered
llama_kv_cache: layer  14: filtered
llama_kv_cache: layer  15: filtered
llama_kv_cache: layer  16: filtered
llama_kv_cache: layer  17: dev = CPU
llama_kv_cache: layer  18: filtered
llama_kv_cache: layer  19: filtered
llama_kv_cache: layer  20: filtered
llama_kv_cache: layer  21: filtered
llama_kv_cache: layer  22: filtered
llama_kv_cache: layer  23: dev = CPU
llama_kv_cache: layer  24: filtered
llama_kv_cache: layer  25: filtered
llama_kv_cache: layer  26: filtered
llama_kv_cache: layer  27: filtered
llama_kv_cache: layer  28: filtered
llama_kv_cache: layer  29: dev = CPU
llama_kv_cache: layer  30: filtered
llama_kv_cache: layer  31: filtered
llama_kv_cache: layer  32: filtered
llama_kv_cache: layer  33: filtered
llama_kv_cache: layer  34: filtered
llama_kv_cache: layer  35: dev = CPU
llama_kv_cache: layer  36: filtered
llama_kv_cache: layer  37: filtered
llama_kv_cache: layer  38: filtered
llama_kv_cache: layer  39: filtered
llama_kv_cache: layer  40: filtered
llama_kv_cache: layer  41: dev = CPU
llama_kv_cache: layer  42: filtered
llama_kv_cache: layer  43: filtered
llama_kv_cache: layer  44: filtered
llama_kv_cache: layer  45: filtered
llama_kv_cache: layer  46: filtered
llama_kv_cache: layer  47: dev = CPU
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
llama_kv_cache:        CPU KV buffer size =   512.00 MiB
llama_kv_cache: size =  512.00 MiB ( 32768 cells,   8 layers,  1/1 seqs), K (f16):  256.00 MiB, V (f16):  256.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 512
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 512
llama_kv_cache_iswa: creating     SWA KV cache, size = 32768 cells
llama_kv_cache: layer   0: dev = CPU
llama_kv_cache: layer   1: dev = CPU
llama_kv_cache: layer   2: dev = CPU
llama_kv_cache: layer   3: dev = CPU
llama_kv_cache: layer   4: dev = CPU
llama_kv_cache: layer   5: filtered
llama_kv_cache: layer   6: dev = CPU
llama_kv_cache: layer   7: dev = CPU
llama_kv_cache: layer   8: dev = CPU
llama_kv_cache: layer   9: dev = CPU
llama_kv_cache: layer  10: dev = CPU
llama_kv_cache: layer  11: filtered
llama_kv_cache: layer  12: dev = CPU
llama_kv_cache: layer  13: dev = CPU
llama_kv_cache: layer  14: dev = CPU
llama_kv_cache: layer  15: dev = CPU
llama_kv_cache: layer  16: dev = CPU
llama_kv_cache: layer  17: filtered
llama_kv_cache: layer  18: dev = CPU
llama_kv_cache: layer  19: dev = CPU
llama_kv_cache: layer  20: dev = CPU
llama_kv_cache: layer  21: dev = CPU
llama_kv_cache: layer  22: dev = CPU
llama_kv_cache: layer  23: filtered
llama_kv_cache: layer  24: dev = CPU
llama_kv_cache: layer  25: dev = CPU
llama_kv_cache: layer  26: dev = CPU
llama_kv_cache: layer  27: dev = CPU
llama_kv_cache: layer  28: dev = CPU
llama_kv_cache: layer  29: filtered
llama_kv_cache: layer  30: dev = CPU
llama_kv_cache: layer  31: dev = CPU
llama_kv_cache: layer  32: dev = CPU
llama_kv_cache: layer  33: dev = CPU
llama_kv_cache: layer  34: dev = CPU
llama_kv_cache: layer  35: filtered
llama_kv_cache: layer  36: dev = CPU
llama_kv_cache: layer  37: dev = CPU
llama_kv_cache: layer  38: dev = CPU
llama_kv_cache: layer  39: dev = CPU
llama_kv_cache: layer  40: dev = CPU
llama_kv_cache: layer  41: filtered
llama_kv_cache: layer  42: dev = CPU
llama_kv_cache: layer  43: dev = CPU
llama_kv_cache: layer  44: dev = CPU
llama_kv_cache: layer  45: dev = CPU
llama_kv_cache: layer  46: dev = CPU
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
llama_kv_cache:        CPU KV buffer size = 10240.00 MiB
llama_kv_cache: size = 10240.00 MiB ( 32768 cells,  40 layers,  1/1 seqs), K (f16): 5120.00 MiB, V (f16): 5120.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 1
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
sched_reserve:        CPU compute buffer size =   534.50 MiB
sched_reserve: graph nodes  = 1972
sched_reserve: graph splits = 1
sched_reserve: reserve took 4.41 ms, sched copies = 1
~llama_context:        CPU compute buffer size is 534.5020 MiB, matches expectation of 534.5020 MiB
```

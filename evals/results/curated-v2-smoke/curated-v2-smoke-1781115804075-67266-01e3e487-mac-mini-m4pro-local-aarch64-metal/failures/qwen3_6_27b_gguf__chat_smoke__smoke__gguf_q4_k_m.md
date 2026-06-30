# Failure: qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m

- **bundle**: qwen3_6_27b_gguf | **capability**: chat | **depth**: smoke
- **model_family**: qwen3 | **checkpoint**: gguf | **quant**: q4_k_m | **backend**: llama_cpp
- **profile**: apple-metal | **host**: mac-mini-m4pro-local | **arch**: aarch64 | **platform**: macOS 15.5 / Apple M4 Pro (Metal)
- **outcome**: blocked | **reason**: child_run_failed
- **failure_reason**: child eval invocation failed; see /tmp/eval-run-results/curated-v2-smoke/curated-v2-smoke-1781115804075-67266-01e3e487-mac-mini-m4pro-local-aarch64-metal/logs/qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m.log
- **accelerator**: requested=metal resolved=metal backend_mode=backend_offload_unverified use_proof=backend:unreported
- **budgets**: {'max_wall_time_secs': '1200'}
- **child build profile**: debug (driver child `cargo build` default)
- **repro**: `/Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/motlie/target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile apple-metal --results-root /tmp/eval-run-results`
- **log tail**: [qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m.log](./qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m.log) (last 200 lines)

@claude-fable5-399-rv 2026-06-10 PDT -- packaged per #435 failure-tracking requirement.

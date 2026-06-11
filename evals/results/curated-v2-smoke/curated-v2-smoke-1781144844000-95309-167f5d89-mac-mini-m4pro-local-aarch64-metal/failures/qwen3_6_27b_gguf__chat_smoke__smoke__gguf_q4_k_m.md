# Non-passed cell: qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m

- **bundle**: qwen3_6_27b_gguf | **capability**: chat | **quant**: q4_k_m | **backend**: llama_cpp
- **profile**: apple-metal | **host**: mac-mini-m4pro-local / aarch64 / Apple M4 Pro / macOS 15.5
- **outcome**: failed | **reason**: behavior_assertion_failed
- **failure_reason**: behavior assertion `min_response_chars` failed: response_chars=0 min=16
- **accelerator**: requested=metal resolved=metal backend_mode=llama_cpp:metal use_proof=backend_observation
- **build**: RELEASE @ 167f5d89 (direct `evals run`, no driver budget)
- **repro**: `/tmp/regression-167f-target/release/evals run --bundle qwen3_6_27b_gguf --scenario chat_smoke --profile apple-metal --root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/regression-rerun/evals --artifact-root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/artifacts/models/hf-cache --requested-accelerator metal --snapshot-id curated-v2-smoke --cell-id qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m --depth smoke --checkpoint-format gguf --artifact-quantization q4_k_m --model-family qwen3 --backend llama_cpp --run-id curated-v2-smoke-1781144844000-95309-167f5d89-mac-mini-m4pro-local-aarch64-metal --precision q4 --quiet-backend-logs --jsonl /tmp/reg167-results.jsonl`

@claude-fable5-399-rv 2026-06-10 PDT -- regression-rerun packaging per #435. Note: accelerator_mismatch rows are HONEST completed CPU runs (documented platform gap), not failures.

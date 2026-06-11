# Non-passed cell: qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m

- **bundle**: qwen3_6_27b_gguf | **capability**: chat | **quant**: q4_k_m | **backend**: llama_cpp
- **profile**: apple-metal | **host**: mac-mini-m4pro-local / aarch64 / Apple M4 Pro / macOS 15.5
- **outcome**: failed | **reason**: behavior_assertion_failed
- **failure_reason**: behavior assertion `min_response_chars` failed: response_chars=0 min=16
- **accelerator**: requested=metal resolved=metal backend_mode=llama_cpp:metal use_proof=backend_observation
- **build**: RELEASE child @ 99ac891d (native matrix driver)
- **repro**: `/tmp/final-run-target/release/evals run --bundle qwen3_6_27b_gguf --scenario chat_smoke --profile apple-metal --root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/final-run/evals --artifact-root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/artifacts/models/hf-cache --jsonl /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/final-run/evals/results/curated-v2-smoke/curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal/results.jsonl --run-id curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal --snapshot-id curated-v2-smoke --cell-id qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m --depth smoke --checkpoint-format gguf --artifact-quantization q4_k_m --precision q4 --model-family qwen3 --backend llama_cpp --requested-accelerator metal --child-build-log /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/final-run/evals/results/curated-v2-smoke/curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal/logs/qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m.log --child-build-status 0 --child-build-duration-ms 5054 --quiet-backend-logs`

@claude-fable5-399-rv 2026-06-11 PDT -- final-run packaging per #435. accelerator_mismatch rows are honest completed CPU runs (documented platform note).

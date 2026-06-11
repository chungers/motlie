# Failure: gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0

- **bundle**: gemma4_12b_qat_q4_0_gguf | **capability**: chat | **depth**: smoke
- **model_family**: gemma | **checkpoint**: gguf | **quant**: q4_0 | **backend**: llama_cpp
- **profile**: apple-metal | **host**: mac-mini-m4pro-local | **arch**: aarch64 | **platform**: macOS 15.5 / Apple M4 Pro (Metal)
- **outcome**: failed | **reason**: behavior_assertion_failed
- **failure_reason**: behavior assertion `min_response_chars` failed: response_chars=0 min=16
- **accelerator**: requested=metal resolved=metal backend_mode=llama_cpp:metal use_proof=backend_observation
- **budgets**: {}
- **child build profile**: debug (driver child `cargo build` default)
- **repro**: `/Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/motlie/target/debug/evals run --bundle gemma4_12b_qat_q4_0_gguf --scenario chat_smoke --profile apple-metal --root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/evals --artifact-root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/libs/models/../../artifacts/models/hf-cache --jsonl /tmp/eval-run-results/curated-v2-smoke/curated-v2-smoke-1781115804075-67266-01e3e487-mac-mini-m4pro-local-aarch64-metal/results.jsonl --run-id curated-v2-smoke-1781115804075-67266-01e3e487-mac-mini-m4pro-local-aarch64-metal --snapshot-id curated-v2-smoke --cell-id gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0 --depth smoke --checkpoint-format gguf --artifact-quantization q4_0 --model-family gemma --backend llama_cpp --requested-accelerator metal --child-build-log /tmp/eval-run-results/curated-v2-smoke/curated-v2-smoke-1781115804075-67266-01e3e487-mac-mini-m4pro-local-aarch64-metal/logs/gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0.log --child-build-status 0 --child-build-duration-ms 3045 --quiet-backend-logs`
- **log tail**: [gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0.log](./gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0.log) (last 200 lines)

@claude-fable5-399-rv 2026-06-10 PDT -- packaged per #435 failure-tracking requirement.

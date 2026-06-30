# Non-passed cell: gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default

- **bundle**: gemma4_e2b | **capability**: perf | **quant**: default | **backend**: mistralrs
- **profile**: apple-metal | **host**: mac-mini-m4pro-local / aarch64 / Apple M4 Pro / macOS 15.5
- **outcome**: blocked | **reason**: runtime_budget_exceeded
- **failure_reason**: child eval invocation failed; see /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/final-run/evals/results/curated-v2-smoke/curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal/logs/gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default.log
- **accelerator**: requested=metal resolved=metal backend_mode=backend_offload_unverified use_proof=backend:unreported
- **build**: RELEASE child @ 99ac891d (native matrix driver)
- **repro**: `/tmp/final-run-target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile apple-metal --artifact-root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/artifacts/models/hf-cache`

@claude-fable5-399-rv 2026-06-11 PDT -- final-run packaging per #435. accelerator_mismatch rows are honest completed CPU runs (documented platform note).

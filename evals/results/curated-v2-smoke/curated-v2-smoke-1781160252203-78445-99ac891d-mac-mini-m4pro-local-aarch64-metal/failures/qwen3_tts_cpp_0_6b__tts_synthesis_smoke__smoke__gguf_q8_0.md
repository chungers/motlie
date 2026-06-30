# Non-passed cell: qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0

- **bundle**: qwen3_tts_cpp_0_6b | **capability**: tts | **quant**: q8_0 | **backend**: qwen3_tts_cpp
- **profile**: apple-metal | **host**: mac-mini-m4pro-local / aarch64 / Apple M4 Pro / macOS 15.5
- **outcome**: skipped | **reason**: profile_not_applicable
- **failure_reason**: profile not applicable to snapshot cell
- **accelerator**: requested=metal resolved=metal backend_mode=backend_offload_unverified use_proof=backend:unreported
- **build**: RELEASE child @ 99ac891d (native matrix driver)
- **repro**: `/tmp/final-run-target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile apple-metal --artifact-root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/artifacts/models/hf-cache`

@claude-fable5-399-rv 2026-06-11 PDT -- final-run packaging per #435. accelerator_mismatch rows are honest completed CPU runs (documented platform note).

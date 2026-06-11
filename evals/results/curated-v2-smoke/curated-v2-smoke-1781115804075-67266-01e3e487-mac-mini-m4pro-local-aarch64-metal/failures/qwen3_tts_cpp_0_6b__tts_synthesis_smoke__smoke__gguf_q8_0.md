# Failure: qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0

- **bundle**: qwen3_tts_cpp_0_6b | **capability**: tts | **depth**: smoke
- **model_family**: qwen3 | **checkpoint**: gguf | **quant**: q8_0 | **backend**: qwen3_tts_cpp
- **profile**: apple-metal | **host**: mac-mini-m4pro-local | **arch**: aarch64 | **platform**: macOS 15.5 / Apple M4 Pro (Metal)
- **outcome**: skipped | **reason**: profile_not_applicable
- **failure_reason**: profile not applicable to snapshot cell
- **accelerator**: requested=metal resolved=metal backend_mode=backend_offload_unverified use_proof=backend:unreported
- **budgets**: {'max_wall_time_secs': '900'}
- **child build profile**: debug (driver child `cargo build` default)
- **repro**: `/Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/motlie/target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile apple-metal --results-root /tmp/eval-run-results`
- **log tail**: [qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0.log](./qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0.log) (last 200 lines)

@claude-fable5-399-rv 2026-06-10 PDT -- packaged per #435 failure-tracking requirement.

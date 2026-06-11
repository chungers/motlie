# Failure: `qwen3_6_27b_gguf__bench_chat_startup__smoke__gguf_q4_k_m`

- bundle: `qwen3_6_27b_gguf`
- capability: `perf`
- profile: `local-cpu-x86_64`
- arch: `x86_64`
- backend: `llama_cpp`
- checkpoint_format: `gguf`
- quantization: `q4_k_m`
- outcome: `blocked`
- reason: `child_run_failed`
- child_build_status: `0`
- child_build_profile: `debug`
- HF_TOKEN_PRESENT: `true`

## Child Build Command

```sh
cargo build -p evals --no-default-features --features model-qwen3-6-27b-gguf
```

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-x86_64 --results-root /tmp/motlie-codex-399-amd-rv-real-eval-01e3e487
```

## Child Log Tail

```text
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.21s
Error: failed to start bundle `qwen3_6_27b_gguf`

Caused by:
    invalid model configuration: artifact policy `LocalOnly` requires cached GGUF artifacts for `unsloth/Qwen3.6-27B-GGUF` under `/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/models/../../artifacts/models/hf-cache`; no refs/main found — run the download step first

```

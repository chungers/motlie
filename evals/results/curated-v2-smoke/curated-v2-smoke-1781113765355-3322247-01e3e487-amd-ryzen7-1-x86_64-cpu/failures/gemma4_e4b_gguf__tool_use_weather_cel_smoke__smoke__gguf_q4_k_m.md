# Failure: `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m`

- bundle: `gemma4_e4b_gguf`
- capability: `tool_use`
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
cargo build -p evals --no-default-features --features model-gemma4-e4b-gguf
```

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-x86_64 --results-root /tmp/motlie-codex-399-amd-rv-real-eval-01e3e487
```

## Child Log Tail

```text
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.22s
Error: failed to start bundle `gemma4_e4b_gguf`

Caused by:
    invalid model configuration: artifact policy `LocalOnly` requires cached GGUF artifacts for `unsloth/gemma-4-E4B-it-GGUF` under `/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/models/../../artifacts/models/hf-cache`; no refs/main found — run the download step first

```

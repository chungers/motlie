# Failure: `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default`

- bundle: `gemma4_e2b`
- capability: `perf`
- profile: `local-cpu-x86_64`
- arch: `x86_64`
- backend: `mistralrs`
- checkpoint_format: `hf_safetensors`
- quantization: `default`
- outcome: `blocked`
- reason: `runtime_budget_exceeded`
- child_build_status: `0`
- child_build_profile: `debug`
- HF_TOKEN_PRESENT: `true`

## Child Build Command

```sh
cargo build -p evals --no-default-features --features model-gemma4-e2b
```

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-x86_64 --results-root /tmp/motlie-codex-399-amd-rv-real-eval-01e3e487
```

## Child Log Tail

```text
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.42s

```

# Failure: `qwen3_4b__chat_smoke__smoke__hf_safetensors_default`

- bundle: `qwen3_4b`
- capability: `chat`
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
cargo build -p evals --no-default-features --features model-qwen3-4b
```

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-x86_64 --results-root /tmp/motlie-codex-399-amd-rv-real-eval-01e3e487
```

## Child Log Tail

```text
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/models)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/bins/evals)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 8.34s

```

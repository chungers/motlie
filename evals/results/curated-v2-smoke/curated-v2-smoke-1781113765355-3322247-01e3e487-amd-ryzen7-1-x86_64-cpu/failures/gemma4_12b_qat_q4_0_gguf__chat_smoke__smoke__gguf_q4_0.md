# Failure: `gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0`

- bundle: `gemma4_12b_qat_q4_0_gguf`
- capability: `chat`
- profile: `local-cpu-x86_64`
- arch: `x86_64`
- backend: `llama_cpp`
- checkpoint_format: `gguf`
- quantization: `q4_0`
- outcome: `blocked`
- reason: `child_run_failed`
- child_build_status: `0`
- child_build_profile: `debug`
- HF_TOKEN_PRESENT: `true`

## Child Build Command

```sh
cargo build -p evals --no-default-features --features model-gemma4-12b-qat-q4-0-gguf
```

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-x86_64 --results-root /tmp/motlie-codex-399-amd-rv-real-eval-01e3e487
```

## Child Log Tail

```text
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/models)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/bins/evals)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.69s
Error: failed to start bundle `gemma4_12b_qat_q4_0_gguf`

Caused by:
    invalid model configuration: artifact policy `LocalOnly` requires cached GGUF artifacts for `google/gemma-4-12B-it-qat-q4_0-gguf` under `/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/models/../../artifacts/models/hf-cache`; no refs/main found — run the download step first

```

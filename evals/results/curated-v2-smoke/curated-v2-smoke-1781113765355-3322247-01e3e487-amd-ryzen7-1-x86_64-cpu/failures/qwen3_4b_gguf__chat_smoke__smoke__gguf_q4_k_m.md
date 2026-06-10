# Failure: `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m`

- bundle: `qwen3_4b_gguf`
- capability: `chat`
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
cargo build -p evals --no-default-features --features model-qwen3-4b-gguf
```

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-x86_64 --results-root /tmp/motlie-codex-399-amd-rv-real-eval-01e3e487
```

## Child Log Tail

```text
   Compiling bindgen v0.72.1
   Compiling ring v0.17.14
   Compiling rustls v0.23.38
   Compiling rustls-webpki v0.103.11
   Compiling llama-cpp-sys-2 v0.1.146
   Compiling ureq v3.3.0
   Compiling hf-hub v0.5.0
   Compiling llama-cpp-2 v0.1.146
   Compiling motlie-model-llama-cpp v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/model/backends/llama_cpp)
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/models)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/bins/evals)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 12.97s
Error: failed to start bundle `qwen3_4b_gguf`

Caused by:
    invalid model configuration: artifact policy `LocalOnly` requires cached GGUF artifacts for `Qwen/Qwen3-4B-GGUF` under `/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/models/../../artifacts/models/hf-cache`; no refs/main found — run the download step first

```

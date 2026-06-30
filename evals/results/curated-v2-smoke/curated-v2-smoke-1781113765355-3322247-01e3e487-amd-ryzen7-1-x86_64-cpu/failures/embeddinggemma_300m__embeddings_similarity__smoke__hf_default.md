# Failure: `embeddinggemma_300m__embeddings_similarity__smoke__hf_default`

- bundle: `embeddinggemma_300m`
- capability: `embeddings`
- profile: `local-cpu-x86_64`
- arch: `x86_64`
- backend: `mistralrs`
- checkpoint_format: `hf_safetensors`
- quantization: `default`
- outcome: `blocked`
- reason: `child_run_failed`
- child_build_status: `0`
- child_build_profile: `debug`
- HF_TOKEN_PRESENT: `true`

## Child Build Command

```sh
cargo build -p evals --no-default-features --features model-google-gemma-300m
```

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-x86_64 --results-root /tmp/motlie-codex-399-amd-rv-real-eval-01e3e487
```

## Child Log Tail

```text
   Compiling ring v0.17.14
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/models)
   Compiling rustls v0.23.38
   Compiling rustls-webpki v0.103.11
   Compiling tokio-rustls v0.26.4
   Compiling rustls-platform-verifier v0.6.2
   Compiling ureq v2.12.1
   Compiling ureq v3.3.0
   Compiling hyper-rustls v0.27.8
   Compiling reqwest v0.12.28
   Compiling reqwest v0.13.2
   Compiling mistralrs-mcp v0.8.1
   Compiling hf-hub v0.4.3
   Compiling hf-hub v0.5.0
   Compiling mistralrs-quant v0.8.1
   Compiling openai-harmony v0.0.8
   Compiling mistralrs-core v0.8.1
   Compiling mistralrs v0.8.1
   Compiling motlie-model-mistral v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/model/backends/mistral)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/bins/evals)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1m 06s
Error: failed to start bundle `embeddinggemma_300m`

Caused by:
    invalid model configuration: artifact policy `LocalOnly` requires cached `config.json` for `google/embeddinggemma-300m` under `/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/models/../../artifacts/models/hf-cache`

```

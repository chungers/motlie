# moonshine_streaming_en__asr_short_transcription__smoke__hf_default

- outcome: `blocked`
- reason: `child_run_failed`
- acceptance: `blocked`
- failure_reason: `child eval invocation failed; see /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e/curated-v2-smoke/curated-v2-smoke-1781132594513-3432241-0b8e8e5e-spark-2f6e-aarch64-cpu/logs/moonshine_streaming_en__asr_short_transcription__smoke__hf_default.log`
- bundle: `moonshine_streaming_en`
- capability: `asr`
- scenario: `asr_short_transcription`
- profile: `local-cpu-aarch64`
- host_id: `spark-2f6e`
- platform: `linux/aarch64`
- requested_accelerator: `cpu`
- resolved_accelerator: `cpu`
- backend: `ort`
- checkpoint_format: `onnx`
- quantization: `default`
- git_sha: `0b8e8e5ecb53c5256037b8970041446a1637515b`
- build_profile: `None`
- cargo_features: `model-moonshine-streaming`
- accelerator_backend_mode: `cpu`
- accelerator_offload: `None`
- child_build_status: `0`
- child_build_duration_ms: `8162`

## Runtime Environment

- HF_TOKEN_PRESENT: `[REDACTED_PRESENT]`

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-aarch64 --results-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e
```

## Child Log Tail

```text
   Compiling rustls v0.23.38
   Compiling ureq v3.3.0
   Compiling hf-hub v0.5.0
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/bins/evals)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 8.10s
Error: failed to start ASR bundle

Caused by:
    invalid model configuration: artifact policy `LocalOnly` requires cached ONNX artifacts for `UsefulSensors/moonshine-streaming` under `/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models/../../artifacts/models/hf-cache`; no refs/main found
```

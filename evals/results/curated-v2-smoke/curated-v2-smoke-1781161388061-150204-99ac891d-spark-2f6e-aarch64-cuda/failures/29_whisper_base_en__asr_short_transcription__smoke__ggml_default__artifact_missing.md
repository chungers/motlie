# whisper_base_en__asr_short_transcription__smoke__ggml_default

## Record

- schema_version: 3
- git_sha: 99ac891d8a2adabe823cce61b2a9fec0aa5dbde3
- run_id: curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda
- profile: dgx-spark
- host_id: spark-2f6e
- arch: aarch64
- capability: asr
- bundle_id: whisper_base_en
- backend: whisper_cpp
- checkpoint_format: ggml
- quantization: default
- requested_accelerator: cuda
- resolved_accelerator: cuda
- accelerator_backend_mode: backend_offload_unverified
- accelerator_use_proof_source: backend:unreported
- outcome: blocked
- reason: artifact_missing
- overall_status: blocked
- failure_reason: artifact cache preflight missing required artifacts for `whisper_base_en` under `/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models/../../artifacts/models/hf-cache`; run `evals provision` or provide --artifact-root before matrix execution
- child_build_profile: n/a
- child_build_status: n/a

## Repro

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile dgx-spark --results-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-final-99ac891d
```

Child build command:

```sh
n/a
```

## Child Log Tail

```text
No child log was produced; this record was blocked before child launch.

```

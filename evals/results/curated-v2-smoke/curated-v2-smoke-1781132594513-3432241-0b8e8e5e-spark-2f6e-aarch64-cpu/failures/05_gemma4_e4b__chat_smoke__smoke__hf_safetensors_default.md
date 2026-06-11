# gemma4_e4b__chat_smoke__smoke__hf_safetensors_default

- outcome: `blocked`
- reason: `child_run_failed`
- acceptance: `blocked`
- failure_reason: `child eval invocation failed; see /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e/curated-v2-smoke/curated-v2-smoke-1781132594513-3432241-0b8e8e5e-spark-2f6e-aarch64-cpu/logs/gemma4_e4b__chat_smoke__smoke__hf_safetensors_default.log`
- bundle: `gemma4_e4b`
- capability: `chat`
- scenario: `chat_smoke`
- profile: `local-cpu-aarch64`
- host_id: `spark-2f6e`
- platform: `linux/aarch64`
- requested_accelerator: `cpu`
- resolved_accelerator: `cpu`
- backend: `mistralrs`
- checkpoint_format: `hf_safetensors`
- quantization: `default`
- git_sha: `0b8e8e5ecb53c5256037b8970041446a1637515b`
- build_profile: `None`
- cargo_features: `model-gemma4-e4b`
- accelerator_backend_mode: `cpu`
- accelerator_offload: `None`
- child_build_status: `0`
- child_build_duration_ms: `17133`

## Runtime Environment

- HF_TOKEN_PRESENT: `[REDACTED_PRESENT]`

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-aarch64 --results-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e
```

## Child Log Tail

```text
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/bins/evals)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 17.05s
Error: primary chat generation failed

Caused by:
    `mistralrs` backend failed during `send_chat_request`: response contained neither text content nor tool calls
```

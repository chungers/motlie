# Failure: gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default

- bundle: `gemma4_e4b`
- scenario: `tool_use_weather_cel_smoke`
- capability: `tool_use`
- model_family: `gemma`
- backend: `mistralrs`
- checkpoint_format: `hf_safetensors`
- quantization: `default`
- runtime_precision: ``
- platform: `amd1 x86 CPU`
- profile: `local-cpu-x86_64`
- host: `amd-ryzen7-1`
- git_sha: `167f5d898bab823ae39ca3acf0c7f798328c3ee9`
- build_profile: `release`
- terminal_outcome: `failed`
- reason: `resource_gate_failed`
- behavior_status: `pass`
- resource_status: `fail`
- overall_status: `fail`
- failure_reason: `resources section not accepted: resource gate max_process_swap_delta_bytes=0 exceeded: process_swap_delta_peak=3.45GiB (3701534720 bytes)`

## Repro Command

```sh
/tmp/motlie-regression-target-167f5d89-release/release/evals run --bundle gemma4_e4b --scenario tool_use_weather_cel_smoke --profile local-cpu-x86_64 --root /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-regression-167f5d89-20260610-1924/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-pr452-validation-20260610-1400/artifacts/models/hf-cache --jsonl /tmp/motlie-regression-results-167f5d89/results.jsonl --run-id regression-amd-x86-release-167f5d89 --snapshot-id curated-v2-smoke --depth smoke --requested-accelerator cpu --download-artifacts --quiet-backend-logs --cell-id gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default --checkpoint-format hf_safetensors --artifact-quantization default --model-family gemma --backend mistralrs
```

## Log Tail

```text

```

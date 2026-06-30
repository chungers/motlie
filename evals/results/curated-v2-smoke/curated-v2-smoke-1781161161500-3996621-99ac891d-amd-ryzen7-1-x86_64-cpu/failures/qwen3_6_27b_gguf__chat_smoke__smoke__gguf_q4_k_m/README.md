# qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m

- bundle: `qwen3_6_27b_gguf`
- scenario: `chat_smoke`
- capability: `chat`
- profile: `local-cpu-x86_64`
- outcome: `failed`
- reason: `behavior_assertion_failed`
- git_sha: `99ac891d8a2adabe823cce61b2a9fec0aa5dbde3`
- child_build_profile: `release`

## Repro command

```sh
'/tmp/motlie-final-99ac891d-target/release/evals' 'run' '--bundle' 'qwen3_6_27b_gguf' '--scenario' 'chat_smoke' '--profile' 'local-cpu-x86_64' '--root' '/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-final-99ac891d-20260610-2343/evals' '--artifact-root' '/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-final-99ac891d-20260610-2343/artifacts/models/hf-cache' '--jsonl' 'evals/results/curated-v2-smoke/curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu/results.jsonl' '--run-id' 'curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu' '--snapshot-id' 'curated-v2-smoke' '--cell-id' 'qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m' '--depth' 'smoke' '--checkpoint-format' 'gguf' '--artifact-quantization' 'q4_k_m' '--precision' 'q4' '--model-family' 'qwen3' '--backend' 'llama_cpp' '--requested-accelerator' 'cpu' '--child-build-log' 'evals/results/curated-v2-smoke/curated-v2-smoke-1781161161500-3996621-99ac891d-amd-ryzen7-1-x86_64-cpu/logs/qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m.log' '--child-build-status' '0' '--child-build-duration-ms' '8306' '--quiet-backend-logs'
```

## Files

- `record.json`: exact result record for this cell.
- `child-log-tail.txt`: final 240 lines of the child log, or the complete log if shorter.

# qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0

- bundle: `qwen3_tts_cpp_0_6b`
- scenario: `tts_synthesis_smoke`
- capability: `tts`
- profile: `local-cpu-x86_64`
- outcome: `blocked`
- reason: `artifact_missing`
- git_sha: `99ac891d8a2adabe823cce61b2a9fec0aa5dbde3`
- child_build_profile: `release`

## Repro command

```sh
'/tmp/motlie-final-99ac891d-target/release/evals' 'matrix' '--snapshot' 'evals/snapshots/curated-v2-smoke.toml' '--profile' 'local-cpu-x86_64' '--artifact-root' '/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-final-99ac891d-20260610-2343/artifacts/models/hf-cache' '--results-root' 'evals/results'
```

## Files

- `record.json`: exact result record for this cell.
- `child-log-tail.txt`: final 240 lines of the child log, or the complete log if shorter.

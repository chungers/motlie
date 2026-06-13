# Eval Matrix Run

- snapshot: `curated-v2-smoke`
- profile: `local-cpu-x86_64`
- launched cells: `4`
- pre-run records: `29`

## Run context (@486-rv-speech 2026-06-12 08:58 PDT) — PHASE: COLD

- **Same-pin two-phase protocol (RUNBOOK Cold/Warm)**: this is the **COLD** phase. Build SHA (pin) `d9b4edf3212cd37ba69a1dd14592e20891a29818` = identity.git_sha in every record; schema v5.
- host: amd2 (`amd-ryzen7-2`), AMD Ryzen 7 7730U, x86_64, `local-cpu-x86_64`; HF_TOKEN_PRESENT=false; release children.
- Invocation: \`matrix --cold --results-root evals/results/cold\` → audio warmup_iterations=0, iterations=1 (single process-cold first call per cell). Harness process exited after this phase; warm phase started as a fresh process.
- outcome mix: 4 passed / 29 blocked (scope: 19 hf_token_missing, 10 artifact_missing) / 0 failed.

### Same-pin cold-vs-warm delta (amd2, pin d9b4edf3, release children)
| cell | COLD single first-call | WARM mean (n=3, 1 discarded warmup) | cold premium |
|---|---:|---:|---|
| sherpa ttfp | 107 | 104.0 [103,105,104] | +3% (~noise) |
| moonshine ttfp | 59 | 57.3 [58,57,57] | +3% (~noise) |
| piper ttfa | 89 | 69.3 [69,73,66] | **+28% (+20ms): ORT first-call premium, same-pin measured** |
| whisper transcription | 1299 | 1289.7 [1291,1281,1297] | ~0 (ggml, no lazy init) |

Note: this SAME-PIN pair supersedes the earlier cross-pin cold story (v4 piper ttfa=183
at c91ba281): the same-day, same-pin cold draw is 89ms, indicating the 183ms draw also
carried colder host state (page cache / first-ever ORT load) beyond the process-cold
premium this protocol isolates. Per RUNBOOK: cold is process-cold, not disk-cold, n=1.

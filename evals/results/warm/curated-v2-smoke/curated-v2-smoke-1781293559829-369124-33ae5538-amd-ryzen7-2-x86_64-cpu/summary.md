# Eval Matrix Run

- snapshot: `curated-v2-smoke`
- profile: `local-cpu-x86_64`
- launched cells: `5`
- pre-run records: `29`

## Run context (@486-rv-speech 2026-06-12 12:47 PDT) — PHASE: WARM

- **Same-pin two-phase protocol (RUNBOOK Cold/Warm), kokoro included (post-#503)**: this is the **WARM** phase. Build SHA (pin) `33ae5538a36c8dceb5745c1df6e403ee80495e87` = identity.git_sha in every record; schema v5.
- host: amd2 (`amd-ryzen7-2`), AMD Ryzen 7 7730U, x86_64, `local-cpu-x86_64`; HF_TOKEN_PRESENT=false; release children.
- Invocation: \`matrix --results-root evals/results/warm\` from a fresh process (scenario defaults warmup 1, iterations 3).
- outcome mix: **5 passed** (now incl. kokoro_82m) / 29 blocked (scope) / 0 failed.

### Same-pin cold-vs-warm delta incl. kokoro (amd2, pin 33ae5538, release children)
| cell | COLD single first-call | WARM mean (n=3) | cold premium |
|---|---:|---:|---|
| sherpa ttfp | 114 | 103.0 [104,103,102] | +11% (+11ms) |
| moonshine ttfp | 58 | 62.0 [63,65,58] | -6% (~noise) |
| piper ttfa | 102 | 71.7 [66,83,66] | +42% (+30ms, ORT first-call) |
| kokoro ttfa | 2196 | 2099.7 [2124,2053,2122] | +4.6% (+96ms, ORT first-call) |
| whisper transcription | 1294 | 1285.0 [1285,1288,1282] | ~0 (ggml) |

### Piper vs Kokoro (the requested comparison, amd2 x86 CPU)
| metric | piper (ljspeech-medium) | kokoro (82M int8) | ratio |
|---|---:|---:|---|
| warm mean ttfa | 71.7ms | 2099.7ms | ~29x |
| warm RTF | 0.067 | 1.09 (slower than realtime) | ~16x |
| cold first-call premium (absolute) | +30ms | +96ms | — |
| audio duration (same text) | 1068ms | 1925ms | kokoro speaks ~1.8x slower |

Reading: on x86 CPU, kokoro's first audio is ~2.1s after request — not interactive;
piper stays ~70ms. Kokoro's relative cold premium is small (+4.6%) only because its
steady-state compute dominates; the absolute ORT first-call cost (+96ms) is ~3x piper's.
Both engines emit all chunks post-compute (ttfa == synthesis), so ttfa == whole-utterance
latency for these ORT TTS cells. RTF>1 means kokoro cannot sustain realtime synthesis on
this CPU; it needs GPU (kokoro-cuda) for conversational use.

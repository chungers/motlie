# Curated Eval Coverage Report

- input files: `1`
- records: `3`
- overall: `blocked`

## Outcome Summary

| outcome | count |
|---|---:|
| `blocked` | 1 |
| `passed` | 2 |

## Platform Notes

- No platform-specific caveats detected in the input records.

## Per-Cell Coverage

| cell | host | arch | run | bundle | capability | depth | profile | requested | resolved | outcome | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `whisper_base_en__asr_short_transcription__smoke__ggml_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781169018588-4036013-e14eaa7e-amd-ryzen7-1-x86_64-cpu` | `whisper_base_en` | `asr` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781169018588-4036013-e14eaa7e-amd-ryzen7-1-x86_64-cpu` | `moonshine_streaming_en` | `asr` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `passed` | `none` |
| `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0` | `amd-ryzen7-1` | `x86_64` | `curated-v2-smoke-1781169018588-4036013-e14eaa7e-amd-ryzen7-1-x86_64-cpu` | `qwen3_tts_cpp_0_6b` | `tts` | `smoke` | `local-cpu-x86_64` | `cpu` | `cpu` | `blocked` | `artifact_missing` |

## Model x Capability

| model_family | capability | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `moonshine` | `asr` | 1 | 0 | 0 | 0 |
| `qwen3` | `tts` | 0 | 0 | 1 | 0 |
| `whisper` | `asr` | 1 | 0 | 0 | 0 |

## Capability x Profile

| capability | profile | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `asr` | `local-cpu-x86_64` | 2 | 0 | 0 | 0 |
| `tts` | `local-cpu-x86_64` | 0 | 0 | 1 | 0 |

## Capability x Depth

| capability | depth | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `asr` | `smoke` | 2 | 0 | 0 | 0 |
| `tts` | `smoke` | 0 | 0 | 1 | 0 |

## Backend x Profile

| backend | profile | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `ort` | `local-cpu-x86_64` | 1 | 0 | 0 | 0 |
| `qwen3_tts_cpp` | `local-cpu-x86_64` | 0 | 0 | 1 | 0 |
| `whisper_cpp` | `local-cpu-x86_64` | 1 | 0 | 0 | 0 |

## Model x Quantization x Backend/Profile/Depth

| model | quantization | backend | profile | depth | passed | failed | blocked | skipped |
|---|---|---|---|---|---:|---:|---:|---:|
| `moonshine` | `default` | `ort` | `local-cpu-x86_64` | `smoke` | 1 | 0 | 0 | 0 |
| `qwen3` | `q8_0` | `qwen3_tts_cpp` | `local-cpu-x86_64` | `smoke` | 0 | 0 | 1 | 0 |
| `whisper` | `default` | `whisper_cpp` | `local-cpu-x86_64` | `smoke` | 1 | 0 | 0 | 0 |

## Requested x Resolved Accelerator

| requested | resolved | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `cpu` | `cpu` | 2 | 0 | 1 | 0 |

## Blocker Rollups

| reason | profile | count |
|---|---|---:|
| `artifact_missing` | `local-cpu-x86_64` | 1 |

## Missing Coverage

Snapshot manifest not supplied; pass `--snapshot <path>` to list missing cells.

## Metric Gaps

| metric | reason | source | count |
|---|---|---|---:|
| `gpu_memory_peak_bytes` | `metric_not_instrumented` | `accelerator_sampler` | 2 |

## Inputs

- `evals/results/curated-v2-smoke/curated-v2-smoke-1781169018588-4036013-e14eaa7e-amd-ryzen7-1-x86_64-cpu/results.jsonl`

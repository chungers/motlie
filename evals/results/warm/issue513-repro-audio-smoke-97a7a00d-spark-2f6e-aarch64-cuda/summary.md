# Curated Eval Coverage Report

- input files: `1`
- records: `2`
- overall: `pass`

## Outcome Summary

| outcome | count |
|---|---:|
| `passed` | 2 |

## Platform Notes

- No platform-specific caveats detected in the input records.

## Per-Cell Coverage

| cell | host | arch | run | bundle | capability | depth | profile | requested | resolved | outcome | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `spark-2f6e` | `aarch64` | `issue513-repro-warm-cuda-97a7a00d` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `kokoro_82m__tts_synthesis_smoke__smoke__onnx_default` | `spark-2f6e` | `aarch64` | `issue513-repro-warm-cuda-97a7a00d` | `kokoro_82m` | `tts` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |

## Latency Metrics

| cell | host | capability | ttft_first_token_ms | ttft_first_answer_token_ms | thinking_tokens_to_answer | completion_tokens | mean_ttft_first_token_ms | p95_ttft_first_token_ms | mean_ttft_first_answer_token_ms | p95_ttft_first_answer_token_ms | mean_transcription_latency_ms | p95_transcription_latency_ms | mean_ttfp_first_partial_ms | p95_ttfp_first_partial_ms | mean_synthesis_latency_ms | p95_synthesis_latency_ms | mean_ttfa_first_chunk_ms | p95_ttfa_first_chunk_ms | ttfp_first_partial_ms | ttfa_first_chunk_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `spark-2f6e` | `tts` | null | null | null | null | null | null | null | null | null | null | null | null | 47.67 | 54.00 | 47.67 | 54.00 | null | null |
| `kokoro_82m__tts_synthesis_smoke__smoke__onnx_default` | `spark-2f6e` | `tts` | null | null | null | null | null | null | null | null | null | null | null | null | 909.00 | 916.00 | 909.00 | 916.00 | null | null |

## Model x Capability

| model_family | capability | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `kokoro` | `tts` | 1 | 0 | 0 | 0 |
| `piper` | `tts` | 1 | 0 | 0 | 0 |

## Capability x Profile

| capability | profile | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `tts` | `dgx-spark` | 2 | 0 | 0 | 0 |

## Capability x Depth

| capability | depth | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `tts` | `smoke` | 2 | 0 | 0 | 0 |

## Backend x Profile

| backend | profile | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `ort` | `dgx-spark` | 2 | 0 | 0 | 0 |

## Model x Quantization x Backend/Profile/Depth

| model | quantization | backend | profile | depth | passed | failed | blocked | skipped |
|---|---|---|---|---|---:|---:|---:|---:|
| `kokoro` | `default` | `ort` | `dgx-spark` | `smoke` | 1 | 0 | 0 | 0 |
| `piper` | `default` | `ort` | `dgx-spark` | `smoke` | 1 | 0 | 0 | 0 |

## Requested x Resolved Accelerator

| requested | resolved | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `cuda` | `cuda` | 2 | 0 | 0 | 0 |

## Blocker Rollups

| reason | profile | count |
|---|---|---:|

## Missing Coverage

Snapshot manifest not supplied; pass `--snapshot <path>` to list missing cells.

## Metric Gaps

| metric | reason | source | count |
|---|---|---|---:|
| `gpu_memory_peak_bytes` | `metric_not_instrumented` | `accelerator_sampler` | 2 |

## Inputs

- `evals/results/warm/issue513-repro-audio-smoke-97a7a00d-spark-2f6e-aarch64-cuda/results.jsonl`

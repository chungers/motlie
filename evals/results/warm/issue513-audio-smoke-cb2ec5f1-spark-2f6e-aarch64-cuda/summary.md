# Curated Eval Coverage Report

- input files: `1`
- records: `4`
- overall: `pass`

## Outcome Summary

| outcome | count |
|---|---:|
| `passed` | 4 |

## Platform Notes

- No platform-specific caveats detected in the input records.

## Per-Cell Coverage

| cell | host | arch | run | bundle | capability | depth | profile | requested | resolved | outcome | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `spark-2f6e` | `aarch64` | `issue513-warm-cuda-cb2ec5f1` | `sherpa_onnx_streaming_zipformer_en` | `asr` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `spark-2f6e` | `aarch64` | `issue513-warm-cuda-cb2ec5f1` | `moonshine_streaming_en` | `asr` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `spark-2f6e` | `aarch64` | `issue513-warm-cuda-cb2ec5f1` | `piper_en_us_ljspeech_medium` | `tts` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |
| `kokoro_82m__tts_synthesis_smoke__smoke__onnx_default` | `spark-2f6e` | `aarch64` | `issue513-warm-cuda-cb2ec5f1` | `kokoro_82m` | `tts` | `smoke` | `dgx-spark` | `cuda` | `cuda` | `passed` | `none` |

## Latency Metrics

| cell | host | capability | ttft_first_token_ms | ttft_first_answer_token_ms | thinking_tokens_to_answer | completion_tokens | mean_ttft_first_token_ms | p95_ttft_first_token_ms | mean_ttft_first_answer_token_ms | p95_ttft_first_answer_token_ms | mean_transcription_latency_ms | p95_transcription_latency_ms | mean_ttfp_first_partial_ms | p95_ttfp_first_partial_ms | mean_synthesis_latency_ms | p95_synthesis_latency_ms | mean_ttfa_first_chunk_ms | p95_ttfa_first_chunk_ms | ttfp_first_partial_ms | ttfa_first_chunk_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default` | `spark-2f6e` | `asr` | null | null | null | null | null | null | null | null | 617.67 | 622.00 | 132.33 | 135.00 | null | null | null | null | null | null |
| `moonshine_streaming_en__asr_short_transcription__smoke__hf_default` | `spark-2f6e` | `asr` | null | null | null | null | null | null | null | null | 3215.00 | 3602.00 | 64.67 | 67.00 | null | null | null | null | null | null |
| `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default` | `spark-2f6e` | `tts` | null | null | null | null | null | null | null | null | null | null | null | null | 97.33 | 122.00 | 97.33 | 122.00 | null | null |
| `kokoro_82m__tts_synthesis_smoke__smoke__onnx_default` | `spark-2f6e` | `tts` | null | null | null | null | null | null | null | null | null | null | null | null | 1115.00 | 1152.00 | 1115.00 | 1152.00 | null | null |

## Model x Capability

| model_family | capability | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `kokoro` | `tts` | 1 | 0 | 0 | 0 |
| `moonshine` | `asr` | 1 | 0 | 0 | 0 |
| `piper` | `tts` | 1 | 0 | 0 | 0 |
| `sherpa_onnx` | `asr` | 1 | 0 | 0 | 0 |

## Capability x Profile

| capability | profile | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `asr` | `dgx-spark` | 2 | 0 | 0 | 0 |
| `tts` | `dgx-spark` | 2 | 0 | 0 | 0 |

## Capability x Depth

| capability | depth | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `asr` | `smoke` | 2 | 0 | 0 | 0 |
| `tts` | `smoke` | 2 | 0 | 0 | 0 |

## Backend x Profile

| backend | profile | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `ort` | `dgx-spark` | 3 | 0 | 0 | 0 |
| `sherpa_onnx` | `dgx-spark` | 1 | 0 | 0 | 0 |

## Model x Quantization x Backend/Profile/Depth

| model | quantization | backend | profile | depth | passed | failed | blocked | skipped |
|---|---|---|---|---|---:|---:|---:|---:|
| `kokoro` | `default` | `ort` | `dgx-spark` | `smoke` | 1 | 0 | 0 | 0 |
| `moonshine` | `default` | `ort` | `dgx-spark` | `smoke` | 1 | 0 | 0 | 0 |
| `piper` | `default` | `ort` | `dgx-spark` | `smoke` | 1 | 0 | 0 | 0 |
| `sherpa_onnx` | `default` | `sherpa_onnx` | `dgx-spark` | `smoke` | 1 | 0 | 0 | 0 |

## Requested x Resolved Accelerator

| requested | resolved | passed | failed | blocked | skipped |
|---|---|---:|---:|---:|---:|
| `cuda` | `cuda` | 4 | 0 | 0 | 0 |

## Blocker Rollups

| reason | profile | count |
|---|---|---:|

## Missing Coverage

Snapshot manifest not supplied; pass `--snapshot <path>` to list missing cells.

## Metric Gaps

| metric | reason | source | count |
|---|---|---|---:|
| `gpu_memory_peak_bytes` | `metric_not_instrumented` | `accelerator_sampler` | 4 |

## Inputs

- `evals/results/warm/issue513-audio-smoke-cb2ec5f1-spark-2f6e-aarch64-cuda/results.jsonl`

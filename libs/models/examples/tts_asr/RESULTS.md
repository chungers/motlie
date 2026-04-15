# TTS↔ASR DGX Spark Results

Measured on `2026-04-15` by `@codex-dgx-e2e`.

## Environment

- Host: `spark-2f6e` (`aarch64` Linux, NVIDIA GB10)
- Driver / CUDA: `580.126.09` / `13.0`
- ORT lib root: `~/.cache/ort.pyke.io/dfbin/aarch64-unknown-linux-gnu/<hash>`
- eSpeak data: `/usr/lib/aarch64-linux-gnu/espeak-ng-data`
- Dataset: `libs/models/examples/tts_asr/dataset/samples.json` (`100` samples)
- Branch base: `cld-review-models/tts-asr-e2e`

## Notes

- The whisper backend on this branch was tuned to set decode threads from
  `std::thread::available_parallelism()`. Without that change, the CPU
  `piper_whisper` pass was impractically slow on this host.
- All pipelines were executed in both CPU and CUDA configurations.
- Only `piper_whisper` completed sample execution. The remaining six lanes failed
  deterministically during bundle startup; those failures are documented below.

## Matrix Status

| Pipeline | Mode | Status | Samples | Result |
|----------|------|--------|---------|--------|
| Piper → whisper.cpp | CPU | Completed | 100/100 | Benchmark data captured |
| Piper → whisper.cpp | CUDA | Completed | 100/100 | Benchmark data captured |
| Piper → sherpa-onnx | CPU | Blocked | 0/100 | sherpa backend init failed on ONNX metadata parsing |
| Piper → sherpa-onnx | CUDA | Blocked | 0/100 | sherpa backend init failed on ONNX metadata parsing |
| Qwen3-TTS → whisper.cpp | CPU | Blocked | 0/100 | `encoder.onnx` missing from cached Qwen3-TTS snapshot |
| Qwen3-TTS → whisper.cpp | CUDA | Blocked | 0/100 | `encoder.onnx` missing from cached Qwen3-TTS snapshot |
| Qwen3-TTS → sherpa-onnx | CPU | Blocked | 0/100 | `encoder.onnx` missing from cached Qwen3-TTS snapshot |
| Qwen3-TTS → sherpa-onnx | CUDA | Blocked | 0/100 | `encoder.onnx` missing from cached Qwen3-TTS snapshot |

## Aggregate Results

### Overall

| Pipeline | Mode | Samples | Mean WER | Median WER | P95 WER | Max WER | Mean Total Latency | Median Total Latency | Mean TTS | Mean ASR |
|----------|------|---------|----------|------------|---------|---------|--------------------|----------------------|----------|----------|
| Piper → whisper.cpp | CPU | 100 | 0.459 | 0.566 | 0.741 | 1.000 | 58,090 ms | 46,490 ms | 1,366 ms | 56,723 ms |
| Piper → whisper.cpp | CUDA | 100 | 0.458 | 0.559 | 0.730 | 0.953 | 7,921 ms | 6,889 ms | 1,379 ms | 6,541 ms |

### By Category

| Category | CPU Mean WER | CUDA Mean WER | CPU Mean Latency | CUDA Mean Latency | Speedup |
|----------|--------------|---------------|------------------|-------------------|---------|
| Short | 0.045 | 0.073 | 1,955 ms | 284 ms | 6.89x |
| Medium | 0.552 | 0.521 | 18,405 ms | 2,672 ms | 6.89x |
| Long | 0.610 | 0.608 | 64,753 ms | 8,583 ms | 7.54x |
| Paragraph | 0.629 | 0.630 | 147,248 ms | 20,144 ms | 7.31x |

### CPU vs CUDA Summary

| Pipeline | CPU Mean WER | CUDA Mean WER | CPU Mean Latency | CUDA Mean Latency | Speedup |
|----------|--------------|---------------|------------------|-------------------|---------|
| Piper → whisper.cpp | 0.459 | 0.458 | 58,090 ms | 7,921 ms | 7.33x |

## Failure Details

### Piper → sherpa-onnx

CPU and CUDA both failed before the first sample:

```text
failed to initialize `sherpa-onnx` backend: invalid ONNX metadata list ``: cannot parse integer from empty string
```

The cached checkpoint is a zipformer2-style export whose metadata includes
`query_head_dims`, `value_head_dims`, and `num_heads`, while the current backend
still expects the older `attention_dims` metadata layout.

### Qwen3-TTS pipelines

Both Qwen3-TTS lanes failed before ASR startup:

```text
invalid model configuration: artifact policy `LocalOnly` requires `encoder.onnx`
in cached snapshot for `Qwen/Qwen3-TTS-12Hz-0.6B-Base`; ensure ONNX export has been performed
```

The cached Qwen3-TTS snapshot only provided `config.json` and `vocab.json`.
Compatible `encoder.onnx`, `decoder.onnx`, and `vocoder.onnx` artifacts were not
available locally.

## Raw Result Files

- `results_piper_whisper_cpu.jsonl`
- `results_piper_whisper_cuda.jsonl`
- `results_piper_sherpa_cpu.jsonl`
- `results_piper_sherpa_cuda.jsonl`
- `results_qwen3_whisper_cpu.jsonl`
- `results_qwen3_whisper_cuda.jsonl`
- `results_qwen3_sherpa_cpu.jsonl`
- `results_qwen3_sherpa_cuda.jsonl`

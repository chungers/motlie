# TTS↔ASR DGX Spark Results

Measured on `2026-04-15` and refreshed on `2026-04-16` by `@codex-dgx-e2e`.

## Environment

- Host: `spark-2f6e` (`aarch64` Linux, NVIDIA GB10)
- Driver / CUDA: `580.126.09` / `13.0`
- ORT lib root: `~/.cache/ort.pyke.io/dfbin/aarch64-unknown-linux-gnu/<hash>`
- eSpeak data: `/usr/lib/aarch64-linux-gnu`
- Dataset: `libs/models/examples/tts_asr/dataset/samples.json` (`100` samples)
- Branch base: `feature/models` merged into `codex-dgx-e2e/tts-asr-matrix`

## Notes

- The whisper backend on this branch was tuned to set decode threads from
  `std::thread::available_parallelism()`. Without that change, the CPU
  `piper_whisper` pass was impractically slow on this host.
- `Piper → sherpa-onnx` was rerun after the merged PR `#183` fix on
  `feature/models`, and both CPU/CUDA lanes completed successfully.
- The `2026-04-16` refresh reran all committed Piper benchmark files after
  adding per-sample RSS capture (`resident_memory_bytes`,
  `peak_resident_memory_bytes`) and correcting the aggregator's even-sample
  median computation.
- Qwen3-TTS pipelines were intentionally skipped in this follow-up because the
  current exported decoder/vocoder produce noisy output and ~100% WER, making
  those lanes uninformative for benchmarking.

## Matrix Status

| Pipeline | Mode | Status | Samples | Result |
|----------|------|--------|---------|--------|
| Piper → whisper.cpp | CPU | Completed | 100/100 | Benchmark data captured |
| Piper → whisper.cpp | CUDA | Completed | 100/100 | Benchmark data captured |
| Piper → sherpa-onnx | CPU | Completed | 100/100 | Benchmark data captured after PR `#183` |
| Piper → sherpa-onnx | CUDA | Completed | 100/100 | Benchmark data captured after PR `#183` |
| Qwen3-TTS → whisper.cpp | CPU | Skipped | 0/100 | Current ONNX export path produces noisy output and ~100% WER |
| Qwen3-TTS → whisper.cpp | CUDA | Skipped | 0/100 | Current ONNX export path produces noisy output and ~100% WER |
| Qwen3-TTS → sherpa-onnx | CPU | Skipped | 0/100 | Current ONNX export path produces noisy output and ~100% WER |
| Qwen3-TTS → sherpa-onnx | CUDA | Skipped | 0/100 | Current ONNX export path produces noisy output and ~100% WER |

## Aggregate Results

### Overall

| Pipeline | Mode | Samples | Mean WER | Median WER | P95 WER | Max WER | Mean Total Latency | Median Total Latency | Mean TTS | Mean ASR |
|----------|------|---------|----------|------------|---------|---------|--------------------|----------------------|----------|----------|
| Piper → sherpa-onnx | CPU | 100 | 0.296 | 0.251 | 0.714 | 1.000 | 11,195 ms | 7,127 ms | 6,544 ms | 4,651 ms |
| Piper → sherpa-onnx | CUDA | 100 | 0.303 | 0.251 | 0.800 | 1.000 | 10,870 ms | 7,526 ms | 6,341 ms | 4,529 ms |
| Piper → whisper.cpp | CPU | 100 | 0.464 | 0.552 | 0.776 | 0.935 | 60,223 ms | 40,198 ms | 6,104 ms | 54,119 ms |
| Piper → whisper.cpp | CUDA | 100 | 0.441 | 0.553 | 0.730 | 0.844 | 13,754 ms | 9,069 ms | 6,422 ms | 7,331 ms |

### By Category

| Category | Pipeline | CPU Mean WER | CUDA Mean WER | CPU Mean Latency | CUDA Mean Latency | Speedup |
|----------|----------|--------------|---------------|------------------|-------------------|---------|
| Short | Piper → sherpa-onnx | 0.466 | 0.505 | 522 ms | 520 ms | 1.00x |
| Short | Piper → whisper.cpp | 0.075 | 0.053 | 2,312 ms | 570 ms | 4.06x |
| Medium | Piper → sherpa-onnx | 0.239 | 0.223 | 3,809 ms | 3,875 ms | 0.98x |
| Medium | Piper → whisper.cpp | 0.528 | 0.477 | 21,533 ms | 4,829 ms | 4.46x |
| Long | Piper → sherpa-onnx | 0.242 | 0.243 | 12,269 ms | 12,017 ms | 1.02x |
| Long | Piper → whisper.cpp | 0.613 | 0.585 | 65,600 ms | 15,084 ms | 4.35x |
| Paragraph | Piper → sherpa-onnx | 0.239 | 0.241 | 28,181 ms | 27,069 ms | 1.04x |
| Paragraph | Piper → whisper.cpp | 0.640 | 0.651 | 151,447 ms | 34,531 ms | 4.39x |

### CPU vs CUDA Summary

| Pipeline | CPU Mean WER | CUDA Mean WER | CPU Mean Latency | CUDA Mean Latency | Speedup |
|----------|--------------|---------------|------------------|-------------------|---------|
| Piper → sherpa-onnx | 0.296 | 0.303 | 11,195 ms | 10,870 ms | 1.03x |
| Piper → whisper.cpp | 0.464 | 0.441 | 60,223 ms | 13,754 ms | 4.38x |

## Findings

- Best aggregate WER: `Piper → sherpa-onnx` CPU (`0.296`)
- Best aggregate latency: `Piper → sherpa-onnx` CUDA (`10,870 ms`)
- Largest CUDA gain: `Piper → whisper.cpp` (`4.38x` mean latency speedup)
- `Piper → sherpa-onnx` sees essentially no CUDA benefit on this host. Mean
  latency was only marginally better on CUDA (`10,870 ms`) than CPU (`11,195 ms`), with
  the win coming from ASR model choice rather than GPU acceleration.
- The refreshed RSS-enabled harness substantially increased measured TTS
  overhead versus the earlier latency-only numbers, so the current JSONLs and
  tables should be treated as the source of truth for PR `#182`.

## Known Exclusions

### Qwen3-TTS pipelines

```text
decoder/vocoder exports currently produce noisy output, yielding ~100% WER
```

These lanes were intentionally skipped in the Sherpa follow-up run because their
current export state does not produce useful benchmark comparisons.

## Raw Result Files

- `results_piper_whisper_cpu.jsonl`
- `results_piper_whisper_cuda.jsonl`
- `results_piper_sherpa_cpu.jsonl`
- `results_piper_sherpa_cuda.jsonl`
- `results_qwen3_whisper_cpu.jsonl`
- `results_qwen3_whisper_cuda.jsonl`
- `results_qwen3_sherpa_cpu.jsonl`
- `results_qwen3_sherpa_cuda.jsonl`

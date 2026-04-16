# TTS↔ASR DGX Spark Results

Measured on `2026-04-15` by `@codex-dgx-e2e`.

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
| Piper → sherpa-onnx | CPU | 100 | 0.308 | 0.250 | 0.800 | 1.000 | 5,303 ms | 4,395 ms | 1,362 ms | 3,940 ms |
| Piper → sherpa-onnx | CUDA | 100 | 0.298 | 0.253 | 0.625 | 0.800 | 5,347 ms | 4,517 ms | 1,360 ms | 3,986 ms |
| Piper → whisper.cpp | CPU | 100 | 0.459 | 0.566 | 0.741 | 1.000 | 58,090 ms | 46,490 ms | 1,366 ms | 56,723 ms |
| Piper → whisper.cpp | CUDA | 100 | 0.458 | 0.559 | 0.730 | 0.953 | 7,921 ms | 6,889 ms | 1,379 ms | 6,541 ms |

### By Category

| Category | Pipeline | CPU Mean WER | CUDA Mean WER | CPU Mean Latency | CUDA Mean Latency | Speedup |
|----------|----------|--------------|---------------|------------------|-------------------|---------|
| Short | Piper → sherpa-onnx | 0.528 | 0.466 | 238 ms | 248 ms | 0.96x |
| Short | Piper → whisper.cpp | 0.045 | 0.073 | 1,955 ms | 284 ms | 6.89x |
| Medium | Piper → sherpa-onnx | 0.223 | 0.241 | 1,825 ms | 1,834 ms | 1.00x |
| Medium | Piper → whisper.cpp | 0.552 | 0.521 | 18,405 ms | 2,672 ms | 6.89x |
| Long | Piper → sherpa-onnx | 0.243 | 0.242 | 5,785 ms | 5,773 ms | 1.00x |
| Long | Piper → whisper.cpp | 0.610 | 0.608 | 64,753 ms | 8,583 ms | 7.54x |
| Paragraph | Piper → sherpa-onnx | 0.237 | 0.241 | 13,363 ms | 13,532 ms | 0.99x |
| Paragraph | Piper → whisper.cpp | 0.629 | 0.630 | 147,248 ms | 20,144 ms | 7.31x |

### CPU vs CUDA Summary

| Pipeline | CPU Mean WER | CUDA Mean WER | CPU Mean Latency | CUDA Mean Latency | Speedup |
|----------|--------------|---------------|------------------|-------------------|---------|
| Piper → sherpa-onnx | 0.308 | 0.298 | 5,303 ms | 5,347 ms | 0.99x |
| Piper → whisper.cpp | 0.459 | 0.458 | 58,090 ms | 7,921 ms | 7.33x |

## Findings

- Best aggregate WER: `Piper → sherpa-onnx` CUDA (`0.298`)
- Best aggregate latency: `Piper → sherpa-onnx` CPU (`5,303 ms`)
- Largest CUDA gain: `Piper → whisper.cpp` (`7.33x` mean latency speedup)
- `Piper → sherpa-onnx` sees essentially no CUDA benefit on this host. Mean
  latency was slightly worse on CUDA (`5,347 ms`) than CPU (`5,303 ms`), with
  the win coming from ASR model choice rather than GPU acceleration.

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

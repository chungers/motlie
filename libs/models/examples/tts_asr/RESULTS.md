# TTS↔ASR DGX Spark Results

Measured on `2026-04-15` and refreshed on `2026-04-16` by `@codex-dgx-e2e`.

## Environment

- Host: `spark-2f6e` (`aarch64` Linux, NVIDIA GB10)
- Driver / CUDA: `580.126.09` / `13.0`
- ORT lib root: `~/.cache/ort.pyke.io/dfbin/aarch64-unknown-linux-gnu/<hash>`
- eSpeak data: `/usr/lib/aarch64-linux-gnu`
- Dataset: `libs/models/examples/tts_asr/dataset/samples.json` (`100` samples)
- Branch base: `feature/models` merged into `cld-review-models/tts-asr-e2e`

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
- The Qwen3-TTS ONNX pipeline path is now treated as dead/deprecated for this
  suite and the example binaries were removed from PR `#182`.
- Merged backend PR `#193` added Moonshine ASR and merged backend PR `#194`
  added qwen3-tts.cpp TTS. PR `#182` now includes four new TTS↔ASR binaries for
  those backends; their DGX benchmark execution is pending.

## Matrix Status

| Pipeline | Mode | Status | Samples | Result |
|----------|------|--------|---------|--------|
| Piper → whisper.cpp | CPU | Completed | 100/100 | Benchmark data captured |
| Piper → whisper.cpp | CUDA | Completed | 100/100 | Benchmark data captured |
| Piper → sherpa-onnx | CPU | Completed | 100/100 | Benchmark data captured after PR `#183` |
| Piper → sherpa-onnx | CUDA | Completed | 100/100 | Benchmark data captured after PR `#183` |
| Piper → Moonshine | CPU | Pending | 0/100 | Binary added; DGX run not started yet |
| Piper → Moonshine | CUDA | Pending | 0/100 | Hybrid CUDA build will accelerate Piper only; Moonshine ASR remains CPU-only |
| qwen3-tts.cpp → whisper.cpp | CPU | Pending | 0/100 | Binary added; DGX run not started yet |
| qwen3-tts.cpp → whisper.cpp | CUDA | Pending | 0/100 | Full CUDA-capable lane after merged PR `#194` |
| qwen3-tts.cpp → sherpa-onnx | CPU | Pending | 0/100 | Binary added; DGX run not started yet |
| qwen3-tts.cpp → sherpa-onnx | CUDA | Pending | 0/100 | Full CUDA-capable lane after merged PRs `#193` and `#194` |
| qwen3-tts.cpp → Moonshine | CPU | Pending | 0/100 | Binary added; DGX run not started yet |
| qwen3-tts.cpp → Moonshine | CUDA | Pending | 0/100 | Hybrid CUDA build will accelerate qwen3-tts.cpp only; Moonshine ASR remains CPU-only |

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
  latency was only marginally better on CUDA (`10,870 ms`) than CPU (`11,195 ms`),
  with the win coming from ASR model choice rather than GPU acceleration.
- The refreshed RSS-enabled harness substantially increased measured TTS
  overhead versus the earlier latency-only numbers, so the current JSONLs and
  tables should be treated as the source of truth for PR `#182`.
- The active matrix is now larger than the currently measured subset: four new
  binaries landed with the Moonshine and qwen3-tts.cpp backend merges, but only
  the original Piper→whisper and Piper→sherpa lanes have DGX numbers so far.

## Known Exclusions

### Qwen3-TTS ONNX pipelines

```text
decoder/vocoder exports currently produce noisy output, yielding ~100% WER
```

These lanes have now been removed from PR `#182` entirely because their ONNX
export state does not produce useful benchmark comparisons.

### Newly added, not yet benchmarked

- `piper_moonshine`
- `qwen3cpp_whisper`
- `qwen3cpp_sherpa`
- `qwen3cpp_moonshine`

These binaries now build on the merged `feature/models` base and are queued for
the next DGX validation pass.

## Raw Result Files

- `results_piper_whisper_cpu.jsonl`
- `results_piper_whisper_cuda.jsonl`
- `results_piper_sherpa_cpu.jsonl`
- `results_piper_sherpa_cuda.jsonl`

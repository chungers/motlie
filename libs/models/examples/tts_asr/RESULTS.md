# TTS↔ASR DGX Spark Results

Measured on `2026-04-16` by `@codex-dgx-e2e`.

## Environment

- Host: `spark-2f6e` (`aarch64` Linux, NVIDIA GB10)
- Driver / CUDA: `580.126.09` / `13.0`
- ORT lib root: `~/.cache/ort.pyke.io/dfbin/aarch64-unknown-linux-gnu/<hash>`
- eSpeak data: `/usr/lib/aarch64-linux-gnu`
- Dataset: `libs/models/examples/tts_asr/dataset/samples.json` (`100` samples)
- Branch base: `feature/models` merged into `cld-review-models/tts-asr-e2e`

## Notes

- All successful runs used release builds.
- `piper_whisper` and `piper_sherpa` completed on both CPU and CUDA with full
  `100/100` sample coverage.
- `piper_moonshine` CPU completed a full `100`-sample pass but Moonshine ASR
  rejected every sample. No transcript rows were emitted.
- `piper_moonshine` CUDA reproduced the same deterministic Moonshine failure in
  a `60s` confirmation run. I did not spend another multi-hour full pass on a
  lane that never emitted a single transcript and whose ASR half is CPU-only.
- Every `qwen3cpp_*` lane aborted during TTS initialization before the first
  sample with the same `ggml` assertion while loading the curated
  `qwen3-tts-0.6b-q8_0.gguf` snapshot.

## Full Matrix Status

| Pipeline | Mode | Status | Samples | Mean WER | Mean Total Latency | Wall Clock | Notes |
|----------|------|--------|---------|----------|--------------------|------------|-------|
| Piper → whisper.cpp | CPU | Completed | 100/100 | 0.463 | 61,960 ms | 6,198.48 s | Full release run |
| Piper → whisper.cpp | CUDA | Completed | 100/100 | 0.459 | 13,261 ms | 1,328.87 s | Full release run |
| Piper → sherpa-onnx | CPU | Completed | 100/100 | 0.285 | 10,988 ms | 1,101.66 s | Full release run |
| Piper → sherpa-onnx | CUDA | Completed | 100/100 | 0.300 | 10,826 ms | 1,085.57 s | Full release run |
| Piper → Moonshine | CPU | Blocked | 0/100 successful | N/A | N/A | 9,313.88 s | Full 100-sample pass, every sample failed in Moonshine ASR |
| Piper → Moonshine | CUDA | Blocked | 0/100 successful | N/A | N/A | 60.17 s | `60s` confirmation run only; same Moonshine failure as CPU |
| qwen3-tts.cpp → whisper.cpp | CPU | Blocked | 0/100 | N/A | N/A | 1.24 s | TTS init aborts in `ggml` before first sample |
| qwen3-tts.cpp → whisper.cpp | CUDA | Blocked | 0/100 | N/A | N/A | 3.32 s | TTS init aborts in `ggml` before first sample |
| qwen3-tts.cpp → sherpa-onnx | CPU | Blocked | 0/100 | N/A | N/A | 1.24 s | TTS init aborts in `ggml` before first sample |
| qwen3-tts.cpp → sherpa-onnx | CUDA | Blocked | 0/100 | N/A | N/A | 2.28 s | TTS init aborts in `ggml` before first sample |
| qwen3-tts.cpp → Moonshine | CPU | Blocked | 0/100 | N/A | N/A | 2.29 s | TTS init aborts in `ggml` before first sample |
| qwen3-tts.cpp → Moonshine | CUDA | Blocked | 0/100 | N/A | N/A | 1.31 s | TTS init aborts in `ggml` before first sample |

## Aggregate Results

Only the four successful Piper result files are numerically comparable.

### Overall

| Pipeline | Mode | Samples | Mean WER | Median WER | P95 WER | Max WER | Mean Total Latency | Median Total Latency | Mean TTS | Mean ASR |
|----------|------|---------|----------|------------|---------|---------|--------------------|----------------------|----------|----------|
| Piper → sherpa-onnx | CPU | 100 | 0.285 | 0.243 | 0.600 | 1.000 | 10,988 ms | 7,158 ms | 6,375 ms | 4,612 ms |
| Piper → sherpa-onnx | CUDA | 100 | 0.300 | 0.245 | 0.750 | 1.000 | 10,826 ms | 7,520 ms | 6,266 ms | 4,560 ms |
| Piper → whisper.cpp | CPU | 100 | 0.463 | 0.558 | 0.746 | 0.950 | 61,960 ms | 42,736 ms | 6,140 ms | 55,819 ms |
| Piper → whisper.cpp | CUDA | 100 | 0.459 | 0.556 | 0.764 | 1.200 | 13,261 ms | 8,620 ms | 6,134 ms | 7,126 ms |

### By Category

| Category | Pipeline | CPU Mean WER | CUDA Mean WER | CPU Mean Latency | CUDA Mean Latency | Speedup |
|----------|----------|--------------|---------------|------------------|-------------------|---------|
| Short | Piper → sherpa-onnx | 0.431 | 0.485 | 522 ms | 477 ms | 1.09x |
| Short | Piper → whisper.cpp | 0.095 | 0.073 | 2,286 ms | 536 ms | 4.27x |
| Medium | Piper → sherpa-onnx | 0.233 | 0.233 | 3,988 ms | 3,836 ms | 1.04x |
| Medium | Piper → whisper.cpp | 0.513 | 0.522 | 22,296 ms | 4,550 ms | 4.90x |
| Long | Piper → sherpa-onnx | 0.237 | 0.239 | 12,032 ms | 12,232 ms | 0.98x |
| Long | Piper → whisper.cpp | 0.627 | 0.607 | 69,229 ms | 14,357 ms | 4.82x |
| Paragraph | Piper → sherpa-onnx | 0.238 | 0.242 | 27,410 ms | 26,759 ms | 1.02x |
| Paragraph | Piper → whisper.cpp | 0.620 | 0.635 | 154,028 ms | 33,601 ms | 4.58x |

### CPU vs CUDA Summary

| Pipeline | CPU Mean WER | CUDA Mean WER | CPU Mean Latency | CUDA Mean Latency | Latency Speedup | CPU Wall Clock | CUDA Wall Clock | Wall-Clock Speedup |
|----------|--------------|---------------|------------------|-------------------|-----------------|----------------|-----------------|--------------------|
| Piper → sherpa-onnx | 0.285 | 0.300 | 10,988 ms | 10,826 ms | 1.01x | 1,101.66 s | 1,085.57 s | 1.01x |
| Piper → whisper.cpp | 0.463 | 0.459 | 61,960 ms | 13,261 ms | 4.67x | 6,198.48 s | 1,328.87 s | 4.67x |

## Findings

- Best aggregate WER: `Piper → sherpa-onnx` CPU (`0.285`)
- Best aggregate latency: `Piper → sherpa-onnx` CUDA (`10,826 ms`)
- Largest CUDA gain: `Piper → whisper.cpp` (`4.67x` by both mean latency and
  full-run wall clock)
- `Piper → sherpa-onnx` still gets essentially no useful GPU acceleration on
  this host.
- `Piper → Moonshine` is not benchmarkable in the current harness/backend
  combination: the CPU full pass produced `0/100` transcripts and the CUDA
  confirmation reproduced the same deterministic ASR shape failures.
- Every `qwen3cpp_*` lane is blocked earlier, in shared TTS initialization,
  before the first sample can run. The failure is independent of ASR backend and
  reproduces in both CPU and CUDA builds.

## Failure Details

### qwen3-tts.cpp TTS init

All `qwen3cpp_*` runs fail with the same startup abort:

```text
GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx)) failed
```

This triggers while loading the curated `qwen3-tts-0.6b-q8_0.gguf` snapshot, so
the qwen half of the matrix is blocked before ASR startup.

### Moonshine ASR pipeline

`piper_moonshine` fails deterministically for every sample:

- short/medium/long samples fail in `process_audio_chunk` with
  `Conv ... Invalid input shape: {4}`
- paragraph samples later fail in `encode_streaming` with an ONNX broadcast
  mismatch (`8 by 18`)

Because no sample ever produced a transcript, I did not treat Moonshine as a
comparable benchmark lane.

## Raw Result Files

- `results_piper_whisper_cpu.jsonl`
- `results_piper_whisper_cuda.jsonl`
- `results_piper_sherpa_cpu.jsonl`
- `results_piper_sherpa_cuda.jsonl`
- `results_piper_moonshine_cpu.jsonl`
- `results_piper_moonshine_cuda.jsonl`
- `results_qwen3cpp_whisper_cpu.jsonl`
- `results_qwen3cpp_whisper_cuda.jsonl`
- `results_qwen3cpp_sherpa_cpu.jsonl`
- `results_qwen3cpp_sherpa_cuda.jsonl`
- `results_qwen3cpp_moonshine_cpu.jsonl`
- `results_qwen3cpp_moonshine_cuda.jsonl`

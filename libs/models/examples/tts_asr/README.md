# TTS↔ASR End-to-End Validation Suite

Tests all 4 combinations of TTS and ASR backends by synthesizing text,
feeding the PCM directly into transcription, and comparing the output
against the original text via word error rate (WER).

## Pipeline Matrix

| Pipeline | TTS | ASR | Features |
|----------|-----|-----|----------|
| `piper_whisper` | Piper en_US ljspeech medium | whisper.cpp base.en | `model-piper-en-us-ljspeech-medium`, `model-whisper-base-en` |
| `piper_sherpa` | Piper en_US ljspeech medium | sherpa-onnx streaming | `model-piper-en-us-ljspeech-medium`, `model-sherpa-onnx-streaming` |
| `qwen3_whisper` | Qwen3-TTS 12Hz 0.6B | whisper.cpp base.en | `model-qwen3-tts-0_6b`, `model-whisper-base-en` |
| `qwen3_sherpa` | Qwen3-TTS 12Hz 0.6B | sherpa-onnx streaming | `model-qwen3-tts-0_6b`, `model-sherpa-onnx-streaming` |

## Test Dataset

`dataset/samples.json` contains 100 English text samples:
- 25 short (5-15 words)
- 25 medium (30-60 words)
- 25 long (100-200 words)
- 25 paragraphs (300-1000 words)

## Prerequisites

1. Download model artifacts for each backend you want to test.

2. Environment requirements:
   - **Piper:** ONNX Runtime (`ORT_LIB_PATH`) + eSpeak-ng data (`PIPER_ESPEAKNG_DATA_DIRECTORY`)
   - **whisper.cpp:** Pre-downloaded `ggml-base.en.bin` under the artifact root
   - **sherpa-onnx:** ONNX Runtime + pre-downloaded sherpa zipformer model files
   - **Qwen3-TTS:** ONNX Runtime + pre-exported ONNX model components (see DESIGN_TTS.md Phase 2)

## Build

Each pipeline binary requires its specific feature flags:

```bash
# Piper + whisper.cpp
cargo build -p motlie-models --example tts_asr_piper_whisper \
  --no-default-features --features model-piper-en-us-ljspeech-medium,model-whisper-base-en

# Piper + sherpa-onnx
cargo build -p motlie-models --example tts_asr_piper_sherpa \
  --no-default-features --features model-piper-en-us-ljspeech-medium,model-sherpa-onnx-streaming

# Qwen3-TTS + whisper.cpp
cargo build -p motlie-models --example tts_asr_qwen3_whisper \
  --no-default-features --features model-qwen3-tts-0_6b,model-whisper-base-en

# Qwen3-TTS + sherpa-onnx
cargo build -p motlie-models --example tts_asr_qwen3_sherpa \
  --no-default-features --features model-qwen3-tts-0_6b,model-sherpa-onnx-streaming

# Aggregator (no model features needed)
cargo build -p motlie-models --example tts_asr_aggregate
```

## Run the Full Matrix

```bash
DATA_DIR=libs/models/examples/tts_asr/dataset

# Run each pipeline, capturing JSONL output
cargo run -p motlie-models --example tts_asr_piper_whisper \
  --no-default-features --features model-piper-en-us-ljspeech-medium,model-whisper-base-en \
  -- --data-dir $DATA_DIR > results_piper_whisper.jsonl 2>piper_whisper.log

cargo run -p motlie-models --example tts_asr_piper_sherpa \
  --no-default-features --features model-piper-en-us-ljspeech-medium,model-sherpa-onnx-streaming \
  -- --data-dir $DATA_DIR > results_piper_sherpa.jsonl 2>piper_sherpa.log

cargo run -p motlie-models --example tts_asr_qwen3_whisper \
  --no-default-features --features model-qwen3-tts-0_6b,model-whisper-base-en \
  -- --data-dir $DATA_DIR > results_qwen3_whisper.jsonl 2>qwen3_whisper.log

cargo run -p motlie-models --example tts_asr_qwen3_sherpa \
  --no-default-features --features model-qwen3-tts-0_6b,model-sherpa-onnx-streaming \
  -- --data-dir $DATA_DIR > results_qwen3_sherpa.jsonl 2>qwen3_sherpa.log
```

For CUDA-enabled runs, add `--cuda` flag:
```bash
cargo run ... -- --data-dir $DATA_DIR --cuda > results.jsonl
```

## Aggregate Results

```bash
cargo run -p motlie-models --example tts_asr_aggregate -- \
  --input results_piper_whisper.jsonl \
  --input results_piper_sherpa.jsonl \
  --input results_qwen3_whisper.jsonl \
  --input results_qwen3_sherpa.jsonl
```

Or via stdin:
```bash
cat results_*.jsonl | cargo run -p motlie-models --example tts_asr_aggregate
```

## Output Format

### Pipeline Binaries

Each binary outputs one JSON line per sample to stdout:

```json
{
  "pipeline": "piper_whisper",
  "sample_id": "short_001",
  "category": "short",
  "word_count": 5,
  "original_text": "Hello world, how are you?",
  "transcribed_text": "hello world how are you",
  "wer": 0.0,
  "tts_latency_ms": 120,
  "asr_latency_ms": 450,
  "total_latency_ms": 570,
  "pcm_bytes": 44100,
  "pcm_duration_ms": 1000
}
```

Status messages go to stderr so they don't interfere with JSONL output.

### Aggregator

The aggregator produces a human-readable comparison table:

```
=== TTS↔ASR Pipeline Comparison Report ===

Pipeline             Samples   Mean WER    Med WER    P95 WER    Max WER  Mean Lat(ms)  Med Lat(ms)
------------------------------------------------------------------------------------------------------
piper_sherpa             100      0.150      0.120      0.400      0.800          1200         1100
piper_whisper            100      0.080      0.050      0.250      0.600          2500         2200
qwen3_sherpa             100      0.200      0.180      0.500      0.900          3500         3200
qwen3_whisper            100      0.120      0.090      0.350      0.700          4800         4500

=== Rankings ===

Best WER (accuracy):  piper_whisper (mean WER = 0.080)
Best latency (speed): piper_sherpa (mean = 1200ms)
```

## Predicted Pipeline Rankings

Based on model architecture characteristics (not yet validated with real data):

### CPU (no CUDA)

| Rank | Pipeline | Rationale |
|------|----------|-----------|
| 1 | Piper → whisper.cpp | Both CPU-optimized. Piper ~60 MiB ONNX + whisper-base.en ~142 MiB ggml. Fastest startup, lowest memory. |
| 2 | Piper → sherpa-onnx | True streaming ASR but heavier ORT runtime overhead |
| 3 | Qwen3-TTS → whisper.cpp | Qwen3 ~1.2 GiB 3-model pipeline too heavy for CPU |
| 4 | Qwen3-TTS → sherpa-onnx | Both heavy, slowest CPU path |

### CUDA (DGX Spark GB10)

| Rank | Pipeline | Rationale |
|------|----------|-----------|
| 1 (quality) | Qwen3-TTS → whisper.cpp | Best voice quality on GPU + proven ASR |
| 1 (streaming) | Qwen3-TTS → sherpa-onnx | Both true streaming on ORT/CUDA |
| 3 | Piper → sherpa-onnx | Piper already fast on CPU, CUDA adds less |
| 4 | Piper → whisper.cpp | Both already CPU-fast, least CUDA benefit |

### By text length

| Length | Best CPU | Best CUDA |
|--------|----------|-----------|
| Short (5-15 words) | Piper → whisper.cpp (least overhead) | Piper → whisper.cpp (startup dominates) |
| Medium (30-60 words) | Piper → whisper.cpp | Qwen3-TTS → whisper.cpp (quality wins) |
| Long (100-200 words) | Piper → sherpa-onnx (streaming) | Qwen3-TTS → sherpa-onnx (quality + streaming) |
| Paragraphs (300-1000) | Piper → sherpa-onnx (memory bounded) | Qwen3-TTS → sherpa-onnx (best quality at scale) |

These are **PREDICTIONS** based on design characteristics. Run the validation matrix to produce real data.

## Interpreting Results

- **WER** (Word Error Rate): lower is better. 0.0 = perfect transcription.
  WER = (substitutions + insertions + deletions) / reference_word_count
- **Latency**: TTS + ASR combined. Lower is better.
- **CPU-only recommendation**: Piper pipelines (lightweight, no GPU required)
- **CUDA recommendation**: Qwen3-TTS pipelines (higher quality potential with GPU acceleration)

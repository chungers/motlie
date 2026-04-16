# ASR Phase 3 — `transcribe-rs` Streaming Evaluation

## Status: Draft

## Change Log

| Date | Change |
|------|--------|
| 2026-04-15 | @cdx-models: Initial Phase 3 research note covering `transcribe-rs`, its streaming-vs-batch model support, and a ranking against Motlie's existing `whisper.cpp` and `sherpa-onnx` ASR backends under the requirement that Phase 3 must support real-time telephony streaming. |

## Overview

Motlie already has two ASR backends:

- Phase 1: `whisper.cpp`
- Phase 2: `sherpa-onnx`

Phase 3 is not a general ASR bake-off. The requirement is narrower:

- streaming input is mandatory
- chunk-by-chunk processing is mandatory
- the target workload is real-time telephony via Telnyx

That immediately changes the ranking. A backend that only exposes whole-buffer transcription is not viable for Phase 3 even if its offline WER is strong.

This document evaluates `transcribe-rs` with that constraint in mind.

## Executive Summary

`transcribe-rs` is a real Rust ASR library with a useful multi-engine API and solid ONNX Runtime feature gating, but it is not a drop-in Phase 3 backend substrate for Motlie.

Key conclusions:

- `transcribe-rs` itself is mature enough to study, but still young.
- The crate currently documents support for `Moonshine`, `Parakeet`, `Canary`, `Cohere`, `SenseVoice`, `GigaAM`, `Whisper`, `Whisperfile`, and OpenAI. It does not currently implement `Nemotron`.
- For Motlie Phase 3, only `Moonshine Streaming` is a credible `transcribe-rs` candidate.
- `Parakeet` in `transcribe-rs` is explicitly non-streaming today.
- `Nemotron` is relevant to DGX Spark, but it is not a `transcribe-rs` backend today.
- `transcribe-rs` uses a synchronous whole-buffer `SpeechModel` API plus optional chunking helpers. That does not match Motlie's first-class `TranscriptionModel` / `TranscriptionStream` contract.
- If Motlie wants to evaluate `transcribe-rs` seriously, the right scope is a narrow spike around `Moonshine Streaming`, not a generic `transcribe-rs` integration layer.

## What `transcribe-rs` Is

Repository and package:

- Repo: <https://github.com/cjpais/transcribe-rs>
- Crate docs: <https://docs.rs/transcribe-rs/latest/transcribe_rs/>
- License: MIT

Current maturity snapshot as of 2026-04-15:

- GitHub stars: `170`
- Forks: `56`
- Open issues: `17`
- Repo created: `2025-09-08`
- Latest default-branch commit: `d97ae65` on `2026-04-08`
- Latest docs.rs release: `0.3.11`

Assessment:

- This is a real, maintained Rust library, not abandonware.
- It is still early-stage infrastructure, not a deeply entrenched speech runtime.
- The rapid release cadence is good for momentum but means API stability is still settling.

## Supported Models Relevant To This Evaluation

### `Moonshine`

There are two distinct Moonshine paths:

- `MoonshineModel`: non-streaming sequence-to-sequence model
- `onnx::moonshine::StreamingModel`: streaming Moonshine variant

What it is:

- Useful Sensors ASR family optimized for low-latency and resource-constrained environments
- base Moonshine is small and edge-oriented
- Moonshine Streaming is the streaming-specific family with tiny/small/medium variants

Relevant facts:

- Official Moonshine base sizes: `27M` params (`tiny`) and `61M` params (`base`)
- Official Moonshine Streaming sizes: `34M` (`tiny`), `123M` (`small`), `245M` (`medium`)
- Official Moonshine Streaming average OpenASR WER:
  - tiny: `12.01`
  - small: `7.84`
  - medium: `6.65`

Motlie relevance:

- This is the only `transcribe-rs` path that is streaming-oriented enough to remain in scope for Phase 3.

### `Parakeet`

What `transcribe-rs` supports today:

- an ONNX `ParakeetModel`
- English-only
- timestamps supported
- `supports_streaming: false`

Important distinction:

- NVIDIA's broader Parakeet family now includes newer unified and streaming-capable models.
- `transcribe-rs` does not currently expose those newer streaming Parakeet variants as a true streaming API.
- The implemented `ParakeetModel` in `transcribe-rs` is still a whole-buffer transcription model.

Relevant facts:

- `transcribe-rs` README benchmark for its INT8 ONNX Parakeet path:
  - `~30x` real-time on M4 Max
  - `~20x` real-time on Zen 3 5700X
  - `~5x` real-time on i5-6500
- Official current NVIDIA `parakeet-unified-en-0.6b` results:
  - average WER `6.29` at `1.12s` latency
  - average WER `8.44` at `0.16s` latency

Motlie relevance:

- Strong model family.
- Wrong crate integration path for Phase 3, because the current `transcribe-rs` Parakeet integration is not chunk-native streaming.

### `Nemotron`

What it is:

- NVIDIA cache-aware streaming ASR family
- current public English checkpoint: `nemotron-speech-streaming-en-0.6b`

Relevant facts:

- official model size: `600M` params
- official checkpoint artifact: `.nemo` around `2.47 GB`
- official average WER:
  - `6.92` at `1.12s`
  - `7.07` at `0.56s`
  - `7.84` at `0.16s`
  - `8.70` at `0.08s`
- official deployment support is NVIDIA/Linux-centric

Critical point:

- `transcribe-rs` does not currently document or implement `Nemotron`.
- There are no `Nemotron` source hits in the crate.

Motlie relevance:

- DGX-relevant, yes.
- `transcribe-rs` candidate, no.
- If Motlie wants Nemotron, it is probably a separate backend exploration, not a `transcribe-rs` integration.

## API Comparison Against Motlie

### Motlie Contract

Motlie's current ASR contract is stream-native:

- `TranscriptionModel: Send + Sync`
- `open_stream(AudioSpec, TranscriptionParams) -> Box<dyn TranscriptionStream>`
- `TranscriptionStream::push_chunk(PcmChunk) -> Result<Option<TranscriptionUpdate>, ModelError>`
- `finish() -> TranscriptionUpdate`

Properties:

- explicit session lifecycle
- explicit chunk sequencing
- explicit end-of-stream handling
- partial vs final segment semantics
- transport-neutral PCM input contract
- natural fit for websocket and telephony frames

### `transcribe-rs` Contract

`transcribe-rs` centers on:

- `SpeechModel: Send`
- synchronous `transcribe_raw(&mut self, samples: &[f32], options: &TranscribeOptions)`
- convenience `transcribe()` / `transcribe_file()`
- optional `transcriber` helpers that do VAD chunking or energy-adaptive chunking around a borrowed `&mut dyn SpeechModel`

Properties:

- model object is mutable and session-like
- primary API is still whole-buffer
- chunking is helper-layer orchestration, not the core model contract
- no Motlie-equivalent `open_stream()` object boundary
- no Motlie-equivalent `PcmChunk { sequence, end_of_stream }`
- no Motlie-equivalent partial/final incremental event protocol

### Practical Fit

`transcribe-rs` can inform Motlie's backend work, but it does not map 1:1 onto our contract.

The biggest gap is that Motlie treats streaming as a first-class capability surface, while `transcribe-rs` mostly treats chunking as a wrapper strategy around synchronous models.

## Streaming Reality Check

This is the gating section for Phase 3.

### `whisper.cpp` in Motlie

Current state:

- working
- useful baseline
- not a viable Phase 3 target under the new requirement

Reason:

- our `whisper.cpp` backend simulates streaming with rolling-window repeated decode
- that is acceptable for earlier ASR slices
- it is not the right architecture for always-on telephony streaming

Conclusion:

- keep as baseline
- do not select for Phase 3

### `sherpa-onnx` in Motlie

Current state:

- working
- true streaming shape
- already aligned with the Motlie PCM stream contract

Reason:

- persistent decoder/runtime state
- incremental chunk feeding
- already proven in our backend architecture

Conclusion:

- this remains the current Phase 3 bar to beat
- any new Phase 3 candidate must be competitive with `sherpa-onnx`, not just with `whisper.cpp`

### `transcribe-rs` Moonshine Streaming

This is the interesting case.

Good:

- the model family is explicitly designed for streaming
- the implementation keeps internal streaming state while processing chunks
- `ModelCapabilities.supports_streaming` is `true`
- ONNX Runtime feature flags already cover CPU, CUDA, ROCm, CoreML, DirectML, WebGPU, XNNPACK

Bad:

- the public `SpeechModel` implementation still exposes whole-buffer `transcribe_raw()`
- `StreamingModel::transcribe_raw()` creates a fresh state and processes the entire sample buffer in one call
- no public chunk-by-chunk session API is exposed by the crate
- no timestamps
- no partial/final update surface

Conclusion:

- the underlying model is streaming-capable
- the current crate API is not telephony-stream-ready in the same way Motlie's `TranscriptionStream` is
- integrating this cleanly would mean writing a Motlie-native backend layer around Moonshine streaming internals, not simply wrapping `SpeechModel`

### `transcribe-rs` Parakeet

Good:

- strong speed story
- timestamps supported
- ONNX Runtime hardware matrix is broad

Bad:

- `supports_streaming` is explicitly `false`
- public API is whole-buffer
- current implementation is not a persistent incremental stream runtime

Conclusion:

- not viable for Phase 3 through `transcribe-rs`

### `transcribe-rs` Nemotron

Good:

- NVIDIA's official model family is highly relevant for DGX Spark
- true streaming architecture
- very strong latency/accuracy story for voice-agent workloads

Bad:

- not implemented in `transcribe-rs`
- official runtime story is NeMo/NVIDIA, not Rust ONNX in this crate
- Linux/NVIDIA bias is strong
- artifact format is `.nemo`, not a simple ONNX snapshot

Conclusion:

- strategically interesting
- not a `transcribe-rs` Phase 3 option today

## Platform and Runtime Comparison

### `transcribe-rs`

What it does well:

- broad accelerator matrix for ONNX Runtime:
  - `ort-cuda`
  - `ort-tensorrt`
  - `ort-rocm`
  - `ort-directml`
  - `ort-coreml`
  - `ort-webgpu`
  - `ort-xnnpack`
- whisper-specific GPU feature flags:
  - `whisper-metal`
  - `whisper-vulkan`
  - `whisper-cuda`
- docs.rs builds include:
  - `aarch64-apple-darwin`
  - `aarch64-unknown-linux-gnu`

What it does not solve for Motlie:

- curated bundle lifecycle
- stable artifact resolution
- Motlie-style `ModelIdentity` / checkpoint registration
- async stream contract

### Motlie Today

- `whisper.cpp`: working, CPU-first, optional CUDA, but not Phase 3 material
- `sherpa-onnx`: working, streaming-first, optional CUDA, already integrated with Motlie contracts and curated bundles

## Model Artifacts and Download Shape

### Moonshine in `transcribe-rs`

Expected directories:

- base Moonshine: `encoder_model.onnx`, `decoder_model_merged.onnx`, `tokenizer.json`
- streaming Moonshine: `frontend`, `encoder`, `adapter`, `cross_kv`, `decoder_kv`, tokenizer/config sidecars

Practical note:

- the official Hugging Face Moonshine repositories are not the same as the prepacked `transcribe-rs` download tarballs
- Phase 3 would need an explicit curated artifact story, not ad hoc operator instructions

### Parakeet in `transcribe-rs`

Expected directory:

- `encoder-model.int8.onnx`
- `decoder_joint-model.int8.onnx`
- `nemo128.onnx`
- `vocab.txt`

This is manageable for curated bundles, but again it is the wrong runtime shape for telephony streaming.

### Nemotron

Official artifact story today is heavyweight:

- `.nemo` checkpoint around `2.47 GB`
- NeMo/NVIDIA runtime assumptions

That is not aligned with the current `transcribe-rs` packaging or Motlie's ONNX-curated bundle path.

## Ranking

This ranking is strictly for **Phase 3 streaming telephony**, not generic offline ASR.

| Rank | Option | Streaming Fit | Motlie Integration Fit | Recommendation |
|------|--------|---------------|------------------------|----------------|
| 1 | Motlie `sherpa-onnx` | Strong | Strong | Keep as primary streaming backend baseline. |
| 2 | `transcribe-rs` `Moonshine Streaming` | Medium | Medium-to-weak | Worth a narrow spike only if we want a second streaming family beyond Zipformer. |
| 3 | NVIDIA Nemotron | Strong in principle | Weak today | Interesting separate research track, but not through `transcribe-rs` as it exists now. |
| 4 | `transcribe-rs` Parakeet | Weak for Phase 3 | Medium | Do not pursue for Phase 3 unless `transcribe-rs` gains a real streaming Parakeet path. |
| 5 | Motlie `whisper.cpp` | Weak for Phase 3 | Already done | Keep as legacy baseline only; not viable for telephony streaming. |

## Recommendation

### Primary Recommendation

Do not make "integrate `transcribe-rs` generically" the Phase 3 plan.

That is too broad and solves the wrong abstraction problem.

### Narrow Recommendation

If Motlie wants to evaluate a second streaming ASR family after `sherpa-onnx`, run a targeted spike on:

- `Moonshine Streaming`
- using a dedicated Motlie backend crate
- with Motlie-native `TranscriptionStream` semantics
- without depending on `transcribe-rs`'s top-level `SpeechModel` API as the public contract

### What Not To Do

- do not choose `Parakeet` via current `transcribe-rs` for Phase 3
- do not treat `supports_streaming = true` in a crate capability struct as sufficient proof of telephony-grade stream semantics
- do not rank `whisper.cpp` as viable for Phase 3 merely because it can be wrapped in rolling-window decode

## Proposed Next Step

Open a single follow-up research issue with this scope:

- confirm whether `Moonshine Streaming` can be adapted into Motlie's `TranscriptionStream` without forking the upstream runtime
- determine whether timestamps and partial/final segment semantics can be surfaced cleanly
- decide whether a direct `Nemotron` track belongs in a separate Phase 4 GPU-first investigation

## References

- `transcribe-rs` repository: <https://github.com/cjpais/transcribe-rs>
- `transcribe-rs` docs.rs: <https://docs.rs/transcribe-rs/latest/transcribe_rs/>
- `Moonshine` model card: <https://huggingface.co/UsefulSensors/moonshine>
- `Moonshine Streaming` model card: <https://huggingface.co/UsefulSensors/moonshine-streaming-medium>
- `Moonshine` Transformers docs: <https://huggingface.co/docs/transformers/en/model_doc/moonshine>
- NVIDIA `nemotron-speech-streaming-en-0.6b`: <https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b>
- NVIDIA `parakeet-unified-en-0.6b`: <https://huggingface.co/nvidia/parakeet-unified-en-0.6b>

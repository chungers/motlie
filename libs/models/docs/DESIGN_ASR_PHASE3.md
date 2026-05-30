# ASR Phase 3 Design

## Changelog
| Date | Who | Summary |
| --- | --- | --- |
| 2026-04-17 | @codex-asr | Renamed the shipped Moonshine example path from `v0.7` to `asr_moonshine`. |
| 2026-04-16 | @codex-asr | Added the Phase 3 decision record: sherpa-onnx remains the primary telephony-grade streaming backend, while Moonshine is the secondary batch/offline backend. Documented the measured latency/accuracy tradeoff and the implementation constraint that Moonshine currently runs CPU-only in Motlie because incremental CUDA chunking is unstable. |
| 2026-04-16 | @codex-asr | Corrected the implementation note after the Moonshine backend switched from finish-only buffering to true chunk-driven inference. Moonshine still remains the secondary backend because the measured chunk latency is too high for telephony, not because the integration is batch-only. |

## Decision

- `sherpa-onnx` remains the primary ASR backend for telephony and real-time streaming.
- `Moonshine` is added as the secondary ASR backend for chunked streaming and offline transcription.

## Data

| Backend | Mode | Latency | WER | Streaming viability | Notes |
| --- | --- | ---: | ---: | --- | --- |
| sherpa-onnx | CPU chunked streaming | 6.6 ms/chunk | 0.296 | Yes | Telephony-grade incremental decode |
| Moonshine | CPU chunked streaming | 450 ms/chunk | 0.000-0.063 | No for telephony | Strong accuracy, but chunk latency is too high |
| Moonshine | CUDA incremental chunks | Crash | n/a | No | Whole-file path works; chunked incremental path is unstable |
| whisper.cpp | CUDA rolling-window batch | 13.7 s/file | 0.441 | No | Batch-oriented fallback, not true streaming |

## Rationale

- `sherpa-onnx` is about 68x faster per chunk than Moonshine on CPU.
- Moonshine accuracy is attractive, but that does not offset the chunk-latency gap for telephony workloads.
- Moonshine therefore fits best as the secondary chunk-capable backend that still shares the same PCM chunk contract as the primary backend, but is used for non-telephony workloads because its chunk latency is much higher.
- Current Moonshine CUDA incremental behavior is unstable, so the Motlie integration keeps the backend CPU-only for now.

## Integration Shape

- Backend crate: `libs/model/backends/moonshine/`
- Runtime substrate: ONNX Runtime via `transcribe-rs` Moonshine streaming runtime
- Contract: existing `TranscriptionModel` / `TranscriptionStream`
- Curated bundle: `libs/models/src/asr/moonshine_streaming_en.rs`
- Example: `libs/models/examples/asr_moonshine/main.rs`

## Contract Semantics

- The Moonshine backend uses the shared PCM chunk API so `.wav` files and websocket-style streams stay on one contract surface.
- The current integration advances the Moonshine inference state on every `push_chunk()`.
- When `emit_partials = true`, `push_chunk()` may emit a single interim transcript segment representing the current best full-text hypothesis.
- `finish()` only flushes deferred normalization state and performs the final decode pass, returning a committed transcript segment.
- Moonshine is still documented as the non-telephony backend because the measured per-chunk latency remains far above sherpa-onnx even though the integration itself is truly chunk-driven.

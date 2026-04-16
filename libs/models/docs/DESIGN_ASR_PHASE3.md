# ASR Phase 3 Design

## Changelog
| Date | Who | Summary |
| --- | --- | --- |
| 2026-04-16 | @codex-asr | Added the Phase 3 decision record: sherpa-onnx remains the primary telephony-grade streaming backend, while Moonshine is the secondary batch/offline backend. Documented the measured latency/accuracy tradeoff and the implementation constraint that Moonshine currently runs CPU-only in Motlie because incremental CUDA chunking is unstable. |

## Decision

- `sherpa-onnx` remains the primary ASR backend for telephony and real-time streaming.
- `Moonshine` is added as the secondary ASR backend for batch/offline transcription.

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
- Moonshine therefore fits best as the secondary batch/offline option that still shares the same PCM chunk contract as the primary backend.
- Current Moonshine CUDA incremental behavior is unstable, so the Motlie integration keeps the backend CPU-only for now.

## Integration Shape

- Backend crate: `libs/model/backends/moonshine/`
- Runtime substrate: ONNX Runtime via `transcribe-rs` Moonshine streaming runtime
- Contract: existing `TranscriptionModel` / `TranscriptionStream`
- Curated bundle: `libs/models/src/asr/moonshine_streaming_en.rs`
- Example: `libs/models/examples/v0.7/main.rs`

## Contract Semantics

- The Moonshine backend uses the shared PCM chunk API so `.wav` files and websocket-style streams stay on one contract surface.
- Unlike sherpa-onnx, the current Moonshine integration is intentionally batch/offline:
  - chunks are accepted incrementally
  - the stream buffers audio
  - the committed transcript is emitted on final flush
- This keeps the integration correct against the current runtime constraints without misrepresenting Moonshine as the telephony backend.

## Changelog

| Date | Who | Summary |
|---|---|---|
| 2026-06-14 PDT | @codex-m6-ds-rv | Initial #524 design for a true incremental-audio TTS contract, capability shape, gateway integration path, and success gates after Telnyx live-call TTFA profiling. |
| 2026-06-14 PDT | @codex-m6-ds-rv | Revised capability shape after David review: keep `CapabilityKind::Speech`, replace one-of `speech_generation` with a set of `SpeechGeneration` values, and avoid adding `CapabilityKind::IncrementalSpeech`. |
| 2026-06-14 PDT | @codex-m6-ds-rv | Addressed streaming-architect review: success gates now require an independent synthesis-complete proof, cancellation is out-of-band and non-consuming, packetization is stateful, and backpressure is normative. |

# Design: Incremental TTS Contract

## Status

Draft for #524. This is brownfield model-contract work: existing buffered TTS backends and callers stay valid, but the model layer must expose a separate, honest contract for backends that can produce playable audio before full utterance synthesis completes.

Related work:

- #523 / PR #525: gateway-side logging, prebuffer default, and real warmup fixes.
- #524: model contract/backend issue for true incremental TTS.
- `bins/telnyx-gateway/docs/PROFILING.md`: gateway span and report contract.
- `libs/model/docs/DESIGN_VOICE_CONTRACTS.md`: earlier brownfield split between buffered speech and chunk-consumable stream adapters.

## Problem

The current `SpeechSynthesizer -> SpeechStream` contract does not prove low-latency audio generation. A backend can synthesize full PCM first and then return a stream that only slices that buffer. That is exactly what current Piper and Kokoro do in Motlie.

For Telnyx, that distinction is critical. The gateway can remove its own prebuffer wait, but `tts.request_to_first_audio` is still bounded by the time the backend takes to return the first complete synthesized text chunk. To reduce first-audio latency further, the backend must yield playable PCM while synthesis is still in progress.

## Success Gate

#524 is not complete unless at least one backend path proves true incremental audio:

- the gateway receives a first playable PCM chunk before full utterance synthesis completes;
- a test proves first outbound media can be queued before an independent synthesis-complete signal, not merely before the stream reaches its terminal chunk;
- the fake/backend harness exposes synthesis-complete as a signal separate from `None`, `is_final`, or stream drop, so a buffered backend cannot pass by synthesizing full PCM first and dripping chunks later;
- a blocked-media-queue test proves the backend does not generate more audio than the configured buffering budget while the gateway is not accepting frames;
- a buffered full-audio result wrapped in a chunk iterator, sentence callback, or gateway-side slicer does not satisfy this gate.

The first production backend may be new. Existing Piper/Kokoro must remain advertised as buffered until they meet this gate.

## Current State

Current relevant contracts:

```rust
pub trait BufferedSpeechSynthesizer: Send + Sync {
    type Request;
    type Output;

    fn synthesize_buffered(
        &self,
        request: Self::Request,
    ) -> impl Future<Output = Result<Self::Output, ModelError>> + Send;
}

pub trait SpeechSynthesizer: Send + Sync {
    type Request;
    type Output;
    type Stream: SpeechStream<Chunk = Self::Output>;

    fn synthesize(
        &self,
        request: Self::Request,
    ) -> impl Future<Output = Result<Self::Stream, ModelError>> + Send;
}
```

`SpeechSynthesizer` only means "chunk-consumable audio." It does not distinguish:

- full PCM returned as one buffer;
- full PCM sliced into transport chunks;
- model-generated PCM yielded incrementally during inference.

Piper and Kokoro currently call `Session::run(...)`, extract the complete output tensor, convert to PCM, then wrap the completed buffer in `BufferedSpeechChunkStream`.

## Backend Finding

Current Motlie Piper/Kokoro backends are direct ONNX/runtime wrappers, not wrappers around a lower-level incremental audio generator.

- Piper: current Motlie code runs the ONNX session, extracts a full output tensor, appends all samples into `Vec<i16>`, and only then returns audio. Upstream Piper has a callback-oriented `textToAudio` surface, but the callback is after sentence/phrase synthesis and buffer append; it is useful for sentence-level flushing, not proof of intra-inference audio streaming.
- Kokoro: current Motlie code phonemizes/tokenizes the full request, runs one ONNX session, extracts a full output tensor, converts all samples, and only then returns audio. No local native callback or pull-stream contract exists in the backend.

Therefore #524 should not start by adding another wrapper around current `SpeechStream`. It needs a contract that can reject these buffered implementations until a backend can prove the success gate.

## Requirements

1. A caller can query true incremental TTS support without backend-name checks.
2. The type contract makes incremental audio distinct from buffered and buffered-chunked audio.
3. A backend must not advertise incremental support unless it can yield first playable PCM before terminal synthesis.
4. The gateway can packetize/enqueue each audio chunk as it arrives.
5. Cancellation can stop synthesis promptly when a call ends, a newer playback replaces it, or barge-in cancels it.
6. Backpressure must be explicit: gateway media queue pressure should not cause unbounded audio buffering.
7. Profiling must separate model first-audio, model full synthesis, packetization, queueing, and playback cadence.
8. Existing buffered backends keep their current contracts and behavior.

## Non-Goals

- Do not make Piper/Kokoro look incremental by wrapping buffered output differently.
- Do not add filler utterances or conversational masking.
- Do not change Telnyx media transport semantics.
- Do not require every TTS backend to implement incremental audio.
- Do not remove `BufferedSpeechSynthesizer` or `BufferedSpeechChunkStream`.

## Capability Design

Keep `CapabilityKind::Speech`; do not add a new capability kind for streaming TTS.

The current representation is too narrow because `CapabilityDescriptor` has a single `speech_generation: Option<SpeechGeneration>` field. That makes buffered-vs-streaming a one-of value and prevents a backend from honestly advertising both full-buffer synthesis and true incremental audio.

Preferred design: change the speech generation metadata from an `Option` to a set.

```rust
pub struct CapabilityDescriptor {
    pub kind: CapabilityKind,
    pub summary: &'static str,
    pub inputs: Vec<ContentKind>,
    pub outputs: Vec<ContentKind>,
    pub interaction: InteractionStyle,
    pub transcription_delivery: Option<TranscriptDelivery>,
    pub speech_generations: BTreeSet<SpeechGeneration>,
}
```

Semantics:

| Field | Meaning |
|---|---|
| `CapabilityKind::Speech` | Text-to-speech synthesis. This remains the only top-level TTS/speech capability kind. |
| `SpeechGeneration::Buffered` | Full-output buffered synthesis. Current Piper/Kokoro/Qwen3-TTS advertise this only. |
| `SpeechGeneration::Streaming` | True incremental audio: playable PCM can be yielded before full utterance synthesis completes. This is success-gated and cannot be satisfied by slicing a completed buffer. |

Callers should use explicit generation support checks:

```rust
capabilities.supports(CapabilityKind::Speech);
capabilities.supports_speech_generation(SpeechGeneration::Buffered);
capabilities.supports_speech_generation(SpeechGeneration::Streaming);
```

A backend that supports both should advertise both values in the set. There is no implicit rule that streaming is greater than or equal to buffered. If a streaming backend wants to support buffered callers through a collect-the-stream adapter, that adapter should be explicit and the backend can advertise both modes.

## Typed Contract

Add an incremental trait beside the existing buffered traits.

```rust
pub trait IncrementalSpeechSynthesizer: Send + Sync {
    type Request;
    type Stream: IncrementalSpeechStream;

    fn synthesize_incremental(
        &self,
        request: Self::Request,
        controls: IncrementalSpeechControls,
    ) -> impl Future<Output = Result<Self::Stream, ModelError>> + Send;
}

pub trait IncrementalSpeechStream: Send {
    fn next_audio_chunk(
        &mut self,
    ) -> impl Future<Output = Result<Option<IncrementalSpeechChunk>, ModelError>> + Send;

    fn finish(self) -> impl Future<Output = Result<IncrementalSpeechSummary, ModelError>> + Send;
}

pub struct IncrementalSpeechControls {
    pub cancel: CancellationToken,
    pub cancel_key: IncrementalSpeechCancelKey,
    pub max_buffered_audio_ms: u32,
}

pub struct IncrementalSpeechCancelKey {
    pub provisional_turn_id: ProvisionalTurnId,
    pub generation: u64,
}

pub struct IncrementalSpeechChunk {
    pub samples_i16: Vec<i16>,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub chunk_index: u64,
    pub is_final: bool,
}

pub struct IncrementalSpeechSummary {
    pub chunks: u64,
    pub audio_ms: u64,
    pub canceled: bool,
}
```

Notes:

- Runtime sample rate is deliberate. The gateway already resamples/packetizes from backend-native rates, and an incremental contract should not force a separate const-generic implementation per backend.
- Operational cancellation is out-of-band through `IncrementalSpeechControls::cancel`, not a consuming `cancel(self)` method. The gateway must be able to cancel while `synthesize_incremental()` or `next_audio_chunk(&mut self)` is pending.
- `cancel_key` uses the same `(provisional_turn_id, generation)` shape as the early-response pipeline in #527 so priority cancel/cancel-and-replace can suppress stale chunks from an older provisional response.
- `synthesize_incremental()` and every pending `next_audio_chunk()` must observe cancellation and resolve within the configured cancel-to-stop latency. After cancellation, the backend must not yield additional playable audio for the canceled generation.
- `finish(self)` is a post-stream summary/cleanup path after no `next_audio_chunk()` borrow is pending. It is not the operational cancel mechanism.
- Chunks must be non-empty PCM. `samples_i16.len()` must be divisible by `channels`, and sample rate/channel count must remain stable for the stream unless a future explicit format-change event is designed.
- Timing should be measured by callers at await boundaries. Backends may add metrics snapshots later, but the core chunk type should stay transport-neutral.

## Gateway Integration

### Selection

The gateway keeps the current buffered path as default fallback. A streaming path is eligible only when:

1. the configured backend advertises `CapabilityKind::Speech` and `supports_speech_generation(SpeechGeneration::Streaming)`;
2. the gateway binary was built with that backend feature;
3. the operator/runtime config allows incremental TTS for that playback.

If any check fails, route to the buffered path and log `incremental_tts_selected=false` with the reason.

### Data Flow

```text
agent.turn text
-> SpeechQueueRequest
-> select IncrementalSpeechSynthesizer
-> synthesize_incremental()
-> next_audio_chunk()
-> resample / packetize
-> bounded media queue
-> outbound Telnyx frames
```

The first outbound frame should be possible after the first chunk, not after full synthesis. Prebuffering remains explicit config. The default for an incremental backend should be no gateway-imposed extra wait beyond the first playable chunk.

### Stateful Packetization

The incremental gateway path must use a stateful packetizer, not the existing per-request buffered packetization shape per model chunk.

Requirements:

- carry resampler state across model chunks;
- carry sub-frame PCM remainders across chunks;
- emit explicit 20 ms Telnyx frame boundaries with monotonic media sequence state;
- pad only at final, cancel, or configured terminal flush, never after each model chunk;
- keep underrun metrics tied to outbound media cadence, not artificial per-chunk padding gaps;
- reject or fail the stream if a chunk is empty, has a sample count not divisible by channel count, or changes sample rate/channel count without a separately designed format-change event.

### Backpressure

The gateway media path is bounded. Backpressure is a contract requirement, not an implementation choice.

The external contract is pull-based: the gateway must not request the next model chunk until packetized frames from the previous chunk have been accepted by the bounded media queue or the playback has been canceled. That makes `next_audio_chunk()` the backpressure boundary.

Backends may use an internal producer task only if it is bounded by `max_buffered_audio_ms`. Tests must prove that when the gateway media queue is blocked, the backend blocks or pauses once that budget is full and does not continue generating model audio in the background.

### Cancellation

Cancellation must be tested in four windows:

- before first model chunk;
- after first chunk but before final chunk;
- while `next_audio_chunk()` is pending;
- while the gateway is blocked on media queue pressure.

The backend must stop producing new chunks after cancellation and return a terminal summary that lets the gateway emit accurate playback status. Cancel-and-replace uses the `(provisional_turn_id, generation)` key: a newer generation for the same provisional turn cancels older generation output, and stale chunks from older generations must be dropped even if a backend races to return them.

## Profiling

Gateway spans should keep the current names where possible and add incremental fields:

| Span/event | Additional fields |
|---|---|
| `tts.synthesis_first_chunk` | `incremental_tts=true`, `model_chunk_index=0`, `first_model_audio_ms` |
| `tts.synthesis_full` | `incremental_tts=true`, `model_chunks`, `audio_ms`, `canceled` |
| `tts.packetize_first_chunk` | `model_chunk_index`, `model_sample_rate_hz`, `frames` |
| `tts.request_to_first_audio` | `incremental_tts_selected`, `incremental_tts_backend`, `fallback_reason` |
| `tts.speak.chunk_queued` | `model_chunk_index`, `chunk_audio_ms`, `queued_frames` |
| `turn.finalize_to_first_audio` | unchanged envelope, but reports should attribute model TTFA separately from gateway packetization |

Reports must distinguish:

- model first-audio latency;
- full model synthesis latency;
- packetization/resampling time;
- queue wait/backpressure;
- outbound pacing underruns.

## Tests

Minimum contract tests:

- fake incremental backend yields first chunk, waits, then yields final chunk;
- fake/backend harness emits an independent synthesis-complete signal and the gateway queues first outbound media before that signal;
- gateway queues first outbound media frame before final model chunk is yielded, but this terminal-chunk check is only secondary evidence;
- backend advertising only `SpeechGeneration::Buffered` is never routed through incremental path;
- backend advertising `SpeechGeneration::Streaming` but returning no chunk before terminal synthesis fails the success-gate test;
- cancellation before/during/after first chunk, including while `next_audio_chunk()` is pending, stops synthesis and emits correct playback terminal status;
- blocked media queue prevents generation beyond `max_buffered_audio_ms`;
- stateful packetizer carries resampler state and sub-frame remainders across chunks and pads only at final/cancel;
- profiling emits `incremental_tts=true` and model chunk indexes for incremental path only.

## Alternatives Considered

### A. Add `CapabilityKind::IncrementalSpeech`

Rejected. It avoids the current one-of metadata limitation, but it splits one domain concept across two top-level capability kinds. That is extra taxonomy and creates a design smell when `CapabilityKind::Speech` plus speech-generation metadata already models the domain.

### B. Keep `speech_generation: Option<SpeechGeneration>` and define streaming as stronger-than-buffered

Rejected. It is compact, but it silently implies `Streaming >= Buffered`. That may be false for a backend API, and it hides whether buffered collection is native, adapted, or unsupported.

### C. Change speech generation metadata to a set

Preferred. It keeps `CapabilityKind::Speech`, removes the one-of limitation, and lets backends advertise exactly `Buffered`, `Streaming`, or both without inventing a new capability kind.

## Migration Plan

1. Change `CapabilityDescriptor::speech_generation: Option<SpeechGeneration>` to `speech_generations: BTreeSet<SpeechGeneration>`.
2. Add helpers such as `supports_speech_generation(...)` and descriptor constructors for buffered-only, streaming-only, and buffered-plus-streaming speech.
3. Add `IncrementalSpeechSynthesizer` and `IncrementalSpeechStream` traits for the `SpeechGeneration::Streaming` case.
4. Add a fake/test backend proving the contract and gateway integration before touching production backends.
5. Add gateway incremental path behind capability/config checks.
6. Evaluate candidate production backends. Piper/Kokoro stay buffered unless their underlying runtime exposes real incremental output.
7. Update docs and reports after live test data confirms lower TTFA without increased underruns/choppiness.

## Open Questions

- Should `IncrementalSpeechChunk` support `f32` samples, or should the first contract standardize on `i16` because the gateway packetizer already consumes that efficiently?
- Should `IncrementalSpeechControls` use a repo-local cancellation token type instead of `tokio_util::sync::CancellationToken` to keep `libs/model` dependency-light?
- Should a backend be allowed to yield sentence-level chunks if each sentence requires full inference? It can be useful, but it should not satisfy the `IncrementalSpeech` success gate unless the first sentence chunk arrives before full utterance synthesis completes and is advertised honestly as sentence-incremental.
- Which production backend should be the first real implementation if Piper/Kokoro remain full-output ONNX paths?

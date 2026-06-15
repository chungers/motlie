## Changelog

| Date | Who | Summary |
|---|---|---|
| 2026-04-29 | @codex-tts | Initial design for tightening speech model contracts and curated metadata so batch vs streaming semantics are explicit, compiler-visible, and easier to compose in higher-level apps such as the voice skills. |
| 2026-04-29 | @codex-tts | Implementation update: landed the brownfield compatibility pass in `libs/model` and `libs/models`, including buffered TTS traits, truthful curated metadata, and a shared buffered speech chunk stream. |
| 2026-04-30 | @codex-tts | Imported the voice-agent skill layer onto the branch and completed the capability-driven composition pass so the voice runtime now derives buffered/batch/streaming behavior from curated metadata after model selection. |
| 2026-05-02 | @codex-tts | Review follow-up: replaced descriptor-equality routing with typed `TranscriptDelivery` and `SpeechGeneration` metadata so execution semantics no longer depend on summary text, and simplified voice-agent adapter dispatch. |
| 2026-06-14 PDT | @codex-m6-ds-rv | Added #524 follow-on design pointer: true incremental-audio TTS stays under `CapabilityKind::Speech`, uses a set of `SpeechGeneration` values, and must prove first playable PCM before full utterance synthesis; buffered chunk streams remain non-incremental. |
| 2026-06-14 PDT | @codex-m6-ds-rv | Reconciled stale streaming-TTS wording with #524: true streaming now means `IncrementalSpeechSynthesizer` / `IncrementalSpeechStream` plus `SpeechGeneration::Streaming`, not the older ambiguous `SpeechStream` shim. |
| 2026-06-14 PDT | @codex-m6-ds-rv | Kept incremental-TTS controls app-neutral: examples use an opaque request label for diagnostics, not caller-specific stale-response keys. |

# Design: Voice Contracts

## Problem

The current speech contracts in `libs/model` and curated bundle metadata in `libs/models` do not make execution semantics explicit enough.

Today:
- ASR contracts already distinguish batch vs streaming in the type system.
- Curated ASR metadata can still advertise streaming even when the backend only implements batch transcription.
- TTS contracts only guarantee chunk-consumable output, not true incremental generation.
- Multiple TTS backends synthesize a full PCM buffer first, then wrap it in backend-local `SpeechStream` adapters, duplicating structure and making higher-level orchestration infer behavior from backend identity instead of capabilities.

This ambiguity leaks upward into voice skills and any future voice application that wants to compose TTS and ASR generically.

## Brownfield Scope

This is brownfield contract work.

In scope:
- tightening `libs/model` speech contracts
- tightening speech capability metadata
- defining a migration path for `libs/models`
- improving composition ergonomics for higher-level apps

Out of scope:
- selecting the next curated speech models
- removing current backends
- changing artifact formats or model download policy
- redesigning non-speech model contracts

## Requirements

### Functional

1. The contract layer must make ASR batch vs streaming behavior explicit and compiler-visible.
2. The contract layer must make TTS buffered vs true streaming behavior explicit and compiler-visible.
3. Curated metadata in `libs/models` must not overstate streaming support.
4. Higher-level callers must be able to choose orchestration behavior from typed capability and metadata instead of backend-name branching.
5. Voice cloning support must remain explicit and queryable.
6. Existing backends must have a migration path that does not require a flag day rewrite.

### Non-functional

1. The design must reduce duplicate wrapper structure in buffered TTS backends.
2. The design must favor static dispatch and type-level clarity over runtime trait-object routing.
3. The design must preserve current backend-specific optimization freedom.
4. The design must support incremental adoption in `libs/models` and higher-level apps.

## Current State

### ASR

The typed ASR layer already distinguishes:
- `BatchTranscriber`
- `StreamingTranscriber`
- `TranscriptionSession`

That is good.

The problem is the metadata layer:
- `whisper_cpp` is a batch backend
- `WhisperCppHandle` implements `BatchTranscriber`
- the curated `whisper_base_en` bundle still advertises `transcription_stream_only`

So type-level truth and catalog metadata diverge.

### TTS

The typed TTS layer currently distinguishes only:
- `SpeechSynthesizer -> SpeechStream`
- `VoiceCloneSynthesizer -> SpeechStream`

That means:
- callers can consume audio in chunks
- callers cannot tell whether chunks are produced incrementally during synthesis, or only after a full PCM buffer is ready

At least these backends currently behave as buffered synthesis:
- Piper
- qwen3-tts.cpp

The currently supported in-tree buffered TTS backends on `feature/models` are Piper and qwen3-tts.cpp. Historical Qwen3 ONNX notes are retained only as design context because that curated path has already been removed from this branch line.

Both supported backends:
- perform backend-specific synthesis to a full PCM buffer
- record metrics
- wrap the result in backend-local stream objects

That is duplicated wrapper structure caused partly by the contract not modeling buffered synthesis directly.

## Goals

The speech model layer should make it obvious:
- whether ASR is batch or streaming
- whether streaming ASR yields partial updates or only a final transcript
- whether TTS is buffered or truly streaming
- whether TTS supports voice cloning

Those facts should be:
- visible in typed contracts where possible
- reflected accurately in curated metadata
- cheap for higher-level apps to query and compose

## Implementation Status (2026-04-29)

This design started as the target end state. The implementation landed in this repo is the brownfield-compatible first slice:
- ASR typed contracts remain split between `BatchTranscriber` and `StreamingTranscriber`
- TTS gained explicit `BufferedSpeechSynthesizer` and `BufferedVoiceCloneSynthesizer` contracts
- existing `SpeechSynthesizer` and `VoiceCloneSynthesizer` remain in place as compatibility shims for chunk-consumable output
- metadata truth is expressed through richer `CapabilityDescriptor` / `Capabilities` constructors, including typed `TranscriptDelivery` and `SpeechGeneration` sub-discriminants instead of descriptor-equality checks
- a shared `BufferedSpeechChunkStream` now carries the common buffered chunking path used by Piper and both Qwen3 TTS integrations

That means the compiler can now distinguish buffered TTS backends from ASR streaming backends, while higher-level code can continue to consume chunked speech output during migration. On this branch the imported `voice-agent` runtime now uses curated capability metadata to decide whether a selected model is batch ASR, streaming ASR, or buffered TTS instead of inferring those semantics from backend names, and it resolves ASR delivery from typed `TranscriptDelivery` metadata rather than descriptor summary text.

Follow-on #524 now owns the true incremental-audio TTS split. See [DESIGN_INCREMENTAL_TTS.md](./DESIGN_INCREMENTAL_TTS.md). That design treats true incremental audio as a success gate while keeping `CapabilityKind::Speech`; it replaces the current one-of `speech_generation` metadata with a set of `SpeechGeneration` values so buffered and streaming support can be advertised exactly. True streaming support is represented by `SpeechGeneration::Streaming` plus `IncrementalSpeechSynthesizer` / `IncrementalSpeechStream`; the older `SpeechSynthesizer` / `SpeechStream` surface remains only a chunk-consumable compatibility shim and must not satisfy streaming support by itself.

## Proposed Design

### 1. Keep ASR type split, but tighten metadata

Retain:
- `BatchTranscriber`
- `StreamingTranscriber`
- `TranscriptionSession`

Add explicit metadata capabilities for execution semantics:
- `transcription_batch`
- `transcription_stream_partial`
- `transcription_stream_final_only`

Rules:
- a backend implementing `BatchTranscriber` must not advertise streaming
- a backend implementing `StreamingTranscriber` must declare whether it surfaces partial updates or only final updates during `finish()`

This preserves a good existing type shape while making metadata truthful.

### 2. Split TTS into buffered vs streaming contracts

The implemented brownfield step adds explicit buffered synthesis contracts beside the existing chunk-stream abstraction:

```rust
pub trait BufferedSpeechSynthesizer: Send + Sync {
    type Request;
    type Output;

    fn synthesize_buffered(
        &self,
        request: Self::Request,
    ) -> impl Future<Output = Result<Self::Output, ModelError>> + Send;
}

pub trait BufferedVoiceCloneSynthesizer<const RATE_HZ: u32, C: ChannelLayout>: Send + Sync {
    type Request;
    type Output;

    fn synthesize_with_reference_buffered(
        &self,
        request: Self::Request,
        reference: CloneReference<RATE_HZ, C>,
    ) -> impl Future<Output = Result<Self::Output, ModelError>> + Send;
}
```

`SpeechSynthesizer` and `VoiceCloneSynthesizer` are retained for compatibility, but they now clearly mean “returns chunk-consumable audio output”, not necessarily true incremental generation. Buffered backends implement the new buffered traits and then adapt into chunked streams only when callers still need the older stream-shaped surface.

The #524 follow-on narrows that future refinement to `IncrementalSpeechSynthesizer` / `IncrementalSpeechStream` for the `SpeechGeneration::Streaming` case. `SpeechSynthesizer` and `VoiceCloneSynthesizer` stay compatibility shims for chunk-consumable output and are not proof of true streaming.

### 3. Add explicit speech execution metadata

The implemented step encodes execution truth through richer `CapabilityDescriptor` constructors and `Capabilities` helpers:
- `transcription_batch`
- `transcription_stream_final_only`
- `transcription_stream_partial`
- `speech_buffered`
- `speech_stream`
- `voice_clone`

This keeps the metadata attached to the existing stable capability surface while making execution semantics obvious to callers and testable in curated bundle definitions. A future dedicated execution-mode enum could still be layered on later if higher-level composition needs a denser representation.

### 4. Centralize buffered TTS wrapper logic

This is now implemented as a shared `BufferedSpeechChunkStream` in `libs/model/src/typed.rs`. Piper and qwen3-tts.cpp now share the same buffer-to-chunk adapter instead of each owning a near-identical `speech.rs` stream wrapper.

That leaves backend-local code focused on:
- artifact loading
- runtime initialization
- actual synthesis
- backend metrics

### 5. Treat composition as a first-class design goal

Higher-level apps such as the voice skills should be able to compose against typed behavior instead of backend identity.

Examples:
- a low-latency turn-taking path should require `IncrementalSpeechSynthesizer`, `SpeechGeneration::Streaming`, and `StreamingTranscriber`
- a robust fallback path can use `BufferedSpeechSynthesizer + BatchTranscriber`
- a cloning workflow can require one of the explicit cloning traits

That gives composition a compiler-visible contract instead of hidden backend assumptions.

## Data Flow Impact

### Current buffered TTS path

1. Skill selects backend by name.
2. Backend synthesizes full PCM.
3. Backend wraps PCM in a backend-local speech stream type.
4. Higher-level app treats it as “streaming” even when first audio cannot arrive until full synthesis completes.

### Proposed buffered TTS path

1. Skill selects a model advertising `CapabilityDescriptor::speech_buffered()`.
2. Backend returns buffered output through `BufferedSpeechSynthesizer`.
3. Generic adapter optionally converts buffered output into chunked playback.
4. Higher-level app knows the latency model upfront.

### Proposed true streaming TTS path

1. Skill selects a model advertising `CapabilityKind::Speech` plus `SpeechGeneration::Streaming`.
2. Backend implements `IncrementalSpeechSynthesizer` and returns an `IncrementalSpeechStream`.
3. The stream yields first playable PCM before an independent synthesis-complete signal, not merely before a terminal stream event.
4. Higher-level app may compose it with streaming ASR or live playback confidently.

## API Ergonomics Examples

### Buffered TTS

```rust
let handle = piper_bundle.start_typed(opts).await?;
let audio = handle
    .synthesize_buffered(SynthesisRequest {
        text: "Hello from Motlie.".into(),
        params: Default::default(),
    })
    .await?;
```

### Streaming TTS

```rust
let handle = some_streaming_tts_bundle.start_typed(opts).await?;
let mut stream = handle
    .synthesize_incremental(
        SynthesisRequest {
            text: "Hello from Motlie.".into(),
            params: Default::default(),
        },
        IncrementalSpeechControls {
            cancel,
            request_label,
            max_buffered_audio_ms,
        },
    )
    .await?;

while let Some(chunk) = stream.next_audio_chunk().await? {
    play(chunk).await?;
}
stream.finish().await?;
```

### Batch ASR

```rust
let handle = whisper_bundle.start_typed(opts).await?;
let result = handle
    .transcribe(audio, Default::default())
    .await?;
```

### Streaming ASR

```rust
let handle = moonshine_bundle.start_typed(opts).await?;
let mut session = handle.open_session(Default::default()).await?;

while let Some(chunk) = next_audio_chunk().await {
    if let Some(update) = session.ingest(chunk).await? {
        render_partial(update);
    }
}

let final_update = session.finish().await?;
```

## Migration Strategy

This is brownfield, so migration must be explicit.

### Phase 1

Metadata-only correction:
- fix curated Whisper metadata first
- introduce explicit execution-mode descriptors without removing old helpers immediately

### Phase 2

Contract addition:
- add `BufferedSpeechSynthesizer`
- add buffered clone equivalent
- keep existing `SpeechSynthesizer` temporarily for migration

### Phase 3

Backend migration:
- migrate Piper
- migrate qwen3-tts.cpp
- document Qwen3 ONNX as an already-removed historical path on this branch line

### Phase 4

Higher-level app migration:
- update voice skills and examples to branch on typed behavior and/or execution metadata

### Phase 5

Cleanup:
- remove misleading capability helpers if superseded
- retire compatibility shims once downstream code no longer depends on them

## Alternatives Considered

### Alternative A: Keep current contracts, fix metadata only

Pros:
- minimal churn
- fixes Whisper mislabeling quickly

Cons:
- does not fix TTS ambiguity
- duplicated buffered wrapper structure remains
- higher-level apps still need backend-aware orchestration

Rejected because it corrects only the symptom, not the shape problem.

### Alternative B: Model everything as streams

Pros:
- one outward API shape
- chunked playback remains easy

Cons:
- conflates “stream-consumable” with “incrementally generated”
- keeps the current ambiguity intact
- makes latency semantics invisible to callers

Rejected because it preserves the main design defect.

### Alternative C: Separate buffered vs streaming contracts and explicit metadata

Pros:
- compiler-enforced execution semantics
- truthful metadata
- simpler higher-level composition
- lower duplication for buffered backends

Cons:
- additive contract churn
- temporary migration overhead

Accepted.

## Expected Impact on Voice Skills

Yes, this should simplify higher-level apps like the voice skills.

Specifically:
- voice skills can choose low-latency turn-taking only when both TTS and ASR actually support it
- Whisper can be handled as a robust batch fallback without metadata lying
- buffered TTS backends can be handled generically
- model selection logic can pivot from backend names to capability and execution metadata

That should reduce ad hoc branching and make future curated model additions easier to integrate correctly.

## Testing Scope for PLAN

PLAN should make concrete:
- unit tests for execution-mode metadata
- unit tests proving Whisper advertises batch semantics
- compile-time migration checks for buffered vs streaming TTS backends
- examples covering batch ASR, streaming ASR, buffered TTS, and streaming TTS
- voice-skill integration validation for capability-based selection

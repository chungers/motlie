# ASR Model Support Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-06-11 | @asr480-impl: Added the implemented `TranscriptSegment::confidence` carrier semantics for backend-native ASR confidence and documented that stability remains out of scope because no current engine emits it natively. | Core Contract Changes in `libs/model`, Streaming PCM API Contract, API Sketch, Testing Scope for PLAN |
| 2026-04-17 | @codex-asr: Renamed the ASR example paths to `asr_whisper`, `asr_sherpa_onnx`, and `asr_moonshine` and updated the design references accordingly. | API Sketch, Testing Scope for PLAN |
| 2026-04-14 | @codex-asr: Documented the implemented Phase 2 `sherpa-onnx` backend slice, including explicit ONNX Runtime provisioning, curated bundle wiring, and feature-flag status alongside the original `whisper.cpp` recommendation. | Overview, Research Summary, Recommended Vertical Slice, Generic Backend Design, Feature Flag Design, Alternatives Considered |
| 2026-04-13 | @codex-asr: Addressed R1 review feedback by stream-scoping `AudioSpec`, changing `push_chunk()` to return `Option`, documenting `Send`/not-`Sync` stream ownership, making quantization explicit, tightening edge-case semantics, and narrowing the first implementation slice to the `.wav` path. | Core Contract Changes in `libs/model`, Generic Backend Design, Curated Bundle Design in `libs/models`, Streaming PCM API Contract, Migration and Compatibility Strategy, API Sketch, Testing Scope for PLAN |
| 2026-04-12 | @codex-asr: Initial brownfield design for a voice-to-text (ASR) vertical slice in the Motlie model stack. Recommends a `whisper.cpp` backend with a curated `whisper-base.en` bundle, documents the streaming PCM contract, and evaluates `faster-whisper` and streaming ONNX alternatives. | All |

This document defines the design for adding voice-to-text transcription to the existing `libs/model` and `libs/models` architecture. The design is intentionally narrow: one end-to-end vertical slice from curated artifact download to transcription output, with the API shaped around streaming PCM chunks so both `.wav` files and websocket audio streams map to the same contract. The current brownfield plan now has two concrete backend slices on that shared contract: the original `whisper.cpp` rollout and the implemented Phase 2 `sherpa-onnx` streaming backend.

This is brownfield product work. Motlie already has a stable contract crate, backend crates, curated bundle registration, and feature-gated selector patterns. The ASR addition must extend those seams additively rather than creating a parallel subsystem.

## Table of Contents

- [Overview](#overview)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Research Summary](#research-summary)
- [Recommended Vertical Slice](#recommended-vertical-slice)
- [Phase 2 Extension: `sherpa-onnx`](#phase-2-extension-sherpa-onnx)
- [Architecture](#architecture)
- [Core Contract Changes in `libs/model`](#core-contract-changes-in-libsmodel)
- [Generic Backend Design](#generic-backend-design)
- [Curated Bundle Design in `libs/models`](#curated-bundle-design-in-libsmodels)
- [Streaming PCM API Contract](#streaming-pcm-api-contract)
- [Feature Flag Design](#feature-flag-design)
- [Migration and Compatibility Strategy](#migration-and-compatibility-strategy)
- [API Sketch](#api-sketch)
- [Alternatives Considered](#alternatives-considered)
- [Testing Scope for PLAN](#testing-scope-for-plan)
- [Open Concerns](#open-concerns)
- [References](#references)

---

## Overview

### Problem Statement

Motlie currently supports chat, completion, embeddings, and multimodal chat through a stable contract layer in `libs/model` and curated bundle composition in `libs/models`. It does not yet have a first-class voice-to-text capability.

The required ASR surface is:

- real-time transcription capability
- CPU-friendly execution by default
- optional CUDA acceleration
- a contract that accepts PCM chunks from a live stream
- support for both `.wav` files and websocket-delivered audio
- the same curated-bundle lifecycle used by the existing model stacks

The key design constraint is that `.wav` files and websocket streams are source formats, not separate model capabilities. The model contract should standardize on streaming PCM input and let higher layers adapt files or network frames into that shape.

### Solution

Add a new ASR capability to `libs/model`, implement generic ASR backend crates under `libs/model/backends/`, and expose curated ASR bundles from `libs/models`.

The rollout is:

1. `libs/model`
   Add transcription capability traits and streaming PCM request/response types.
2. `libs/model/backends/whisper_cpp`
   Provide the CPU-first Phase 1 backend adapter and bundle implementation over `whisper.cpp`.
3. `libs/model/backends/sherpa_onnx`
   Provide the true-streaming Phase 2 backend adapter and bundle implementation over ONNX Runtime.
4. `libs/models`
   Add an `asr/` namespace with curated bundle files, one direct enum family, one selector family, and bundle-local artifact resolution and feature gating.

This keeps the existing layering intact:

- `libs/model` owns the stable capability contract
- backend crates own generic runtime integration
- `libs/models` owns curated bundle identity, artifact download rules, selectors, and feature flags

## Goals and Non-Goals

### Goals

- Add a first-class ASR capability to the Motlie model contract
- Preserve the current `ModelBundle` -> `BundleHandle` -> capability-adapter pattern
- Standardize on a streaming PCM chunk API that works for files and live sources
- Support a first curated local bundle with explicit artifact download rules
- Prefer CPU-operable defaults while allowing optional CUDA acceleration
- Keep backend-specific artifact layout and model loading details out of application code
- Reuse the existing feature-gated `libs/models` selector and catalog conventions
- Land one vertical slice end to end before broadening model coverage

### Non-Goals

- Multi-model ASR breadth in the first cut
- Speaker diarization in the first cut
- Translation, language identification, or subtitle formatting in the first cut
- Browser or mobile runtime support in the first cut
- An HTTP or websocket server inside `libs/model` or `libs/models`
- Generic user-supplied ASR checkpoint loading in the first cut

## Research Summary

### Candidate Summary

| Candidate | Real-time Streaming Fit | CPU Fit | CUDA Fit | Rust Integration Fit | Bundle/Artifact Fit | Decision |
|-----------|-------------------------|---------|----------|----------------------|---------------------|----------|
| `whisper.cpp` + Whisper | Good. Streaming is implemented as repeated decode over rolling PCM windows, with realtime and VAD examples upstream. | Good for `tiny/base/small`; upstream is explicitly CPU-capable and memory requirements are clear. | Good. Upstream supports NVIDIA GPU via CUDA and cuBLAS. | Strong. `whisper.cpp` exposes a C API and upstream lists `whisper-rs` bindings. | Strong. Single-file `ggml` model artifacts can be curated with exact include rules. | Recommended for v1 |
| `faster-whisper` + Distil-Whisper / Whisper | Good for service-style streaming, but not as a native Rust backend. Real-time use is typically built around higher-level Python services. | Very good, especially with int8 CPU. | Very good. Benchmarks are strong on RTX-class GPUs. | Weak-to-medium. Best-supported path is Python/CTranslate2 rather than a Motlie-native Rust backend crate. | Medium. Converted CTranslate2 directories are manageable but less aligned with current backend patterns. | Keep as a performance reference, not v1 |
| Streaming ONNX models via `sherpa-onnx` / Moonshine / Zipformer | Excellent. Streaming, websocket, microphone, and Rust bindings already exist upstream. | Very good, especially for streaming-focused small models. | Good through ONNX Runtime providers. | Medium. Strong runtime fit, but model/runtime surface is more specialized and multi-file. | Medium. Fits `CheckpointFormat::Onnx`, but curated artifact sets are larger and runtime behavior is less Whisper-compatible. | Implemented in Phase 2 |

### Why Whisper Is Still the Best First Slice

`whisper.cpp` is not the most specialized streaming stack, and `faster-whisper` can be faster in some CPU/GPU configurations. It is still the best first Motlie slice because it balances the full set of requirements:

- mature ASR model family with broad ecosystem support
- clear CPU-only and CUDA paths
- simple, inspectable single-file checkpoint story
- a thin Rust integration path over a C API
- upstream examples for both `.wav` transcription and rolling realtime transcription

The recommended v1 model is `whisper-base.en`, not `small`, `medium`, or `large`, because the first slice is CPU-first. `base.en` gives better quality than `tiny.en` without pushing memory and latency into the range that would make a commodity-CPU vertical slice fragile.

### CPU vs CUDA Analysis

The current external evidence supports the following practical view:

- Whisper family sizing:
  OpenAI documents `tiny`, `base`, `small`, `medium`, `large`, and `turbo`, with `base` around 74M parameters and `small` around 244M. That already implies the first CPU-friendly slice should stay at `base.en` or `small.en`, not `medium+`.
- `whisper.cpp` operational footprint:
  upstream documents `base` at about 142 MiB on disk and about 388 MB in memory, and `small` at about 466 MiB on disk and about 852 MB in memory.
- `whisper.cpp` realtime support:
  upstream documents a realtime microphone example using a `500 ms` step and `5000 ms` rolling window, which is a direct fit for Motlie's PCM chunk API.
- CUDA bonus:
  upstream `whisper.cpp` supports NVIDIA GPU acceleration, and `faster-whisper` benchmark data shows that GPU throughput for Whisper-class models can be materially higher than CPU-only deployments.

Inference from these sources:

- `whisper-base.en` is the lowest-risk first curated bundle for CPU-first Motlie deployments.
- `whisper-small.en` is a plausible second curated bundle once the contract and backend are validated.
- CUDA support should remain backend-local and optional. The public ASR capability API should not fork into separate CPU and GPU contracts.

## Recommended Vertical Slice

### Curated Model Choice

Recommended first bundle:

- logical model: OpenAI Whisper
- backend/runtime: `whisper.cpp`
- curated artifact: `ggml-base.en.bin`
- capability surface: streaming transcription only
- primary deployment target: Linux and macOS CPU
- optional acceleration target: CUDA-enabled Linux builds

### Why `whisper-base.en`

- better quality than `tiny.en`
- materially lighter than `small.en`
- English-only model is acceptable for the first vertical slice
- keeps artifact size, memory use, and startup time within a manageable range
- lets the API shape stabilize before multilingual and larger-model pressure arrives

### What Is Intentionally Deferred

- multilingual bundle variants
- quantized Whisper bundle variants
- VAD sidecar model bundling
- ONNX-native streaming bundles
- diarization and speaker labels

### Phase 2 Extension: `sherpa-onnx`

The follow-on backend slice uses the same `TranscriptionModel` and `TranscriptionStream` contract but swaps the runtime strategy:

- logical model: streaming Zipformer via `sherpa-onnx`
- backend/runtime: `ort` with explicit operator-provided ONNX Runtime linkage and persistent stream state
- curated artifact set: encoder, decoder, joiner, and `tokens.txt`
- capability surface: streaming transcription only
- primary deployment target: Linux and macOS CPU
- optional acceleration target: CUDA-enabled builds via ONNX Runtime execution providers

This second slice exists because it is a better fit for true incremental decode than the rolling-window `whisper.cpp` implementation. It does not replace the Phase 1 recommendation; it broadens the backend menu while keeping the stable PCM contract unchanged.

## Architecture

### Crate Layout

```text
libs/model/
  src/
    lib.rs
    transcription.rs

libs/model/backends/whisper_cpp/
  Cargo.toml
  src/
    lib.rs
    common.rs
    transcription.rs

libs/model/backends/sherpa_onnx/
  Cargo.toml
  src/
    lib.rs
    common.rs
    transcription.rs

libs/models/
  src/
    asr/
      mod.rs
      whisper_base_en.rs
      sherpa_onnx_streaming_en.rs
```

### High-Level Data Flow

1. Caller chooses a curated selector such as `asr:openai/whisper_base_en` or `asr:sherpa-onnx/streaming_zipformer_en`, or the corresponding `AsrModels` enum.
2. `libs/models` resolves the curated descriptor and artifact rules.
3. On `start()`, the bundle resolves local artifacts or downloads them through the existing curated artifact path.
4. The curated bundle passes resolved local artifacts to the selected backend adapter.
5. The backend starts a loaded bundle handle that exposes the ASR capability.
6. Caller opens a transcription stream and pushes PCM chunks.
7. The backend emits partial/final transcript updates.
8. A `.wav` adapter or websocket server outside the model crates feeds the same PCM chunk type into that stream.

### Source Adaptation Boundary

The model-layer capability should not accept filesystem paths or websocket sockets directly.

Boundary rule:

- `libs/model` standardizes on PCM chunks and transcript events
- `libs/models` examples may add `.wav` convenience helpers
- websocket servers live in binaries or higher-level crates and translate frames into `PcmChunk`

This keeps the capability contract stable and transport-neutral.

## Core Contract Changes in `libs/model`

### Additive Capability Extension

Add:

- `CapabilityKind::Transcription`
- `CapabilityDescriptor::transcription_stream()`
- `BundleHandle::transcription() -> Result<&dyn TranscriptionModel, ModelError>`
- `TranscriptionModel`
- `TranscriptionStream`
- streaming PCM request/response types

Use the existing `ContentKind::Audio` and `InteractionStyle::Streaming` rather than inventing new capability metadata primitives.

### New Types

Recommended core shapes:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcmEncoding {
    S16Le,
    F32Le,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioSpec {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub encoding: PcmEncoding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PcmChunk {
    pub data: Vec<u8>,
    pub sequence: u64,
    pub end_of_stream: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TranscriptionParams {
    pub language: Option<String>,
    pub emit_partials: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    /// Optional backend-native confidence normalized to `0.0..=1.0` only by
    /// direct unit conversion. This is a carrier, not a derived score.
    pub confidence: Option<f32>,
    pub final_segment: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptionUpdate {
    pub segments: Vec<TranscriptSegment>,
}
```

Traits:

```rust
#[async_trait]
pub trait TranscriptionModel: Send + Sync {
    async fn open_stream(
        &self,
        spec: AudioSpec,
        params: TranscriptionParams,
    ) -> Result<Box<dyn TranscriptionStream>, ModelError>;
}

#[async_trait]
pub trait TranscriptionStream: Send {
    async fn push_chunk(
        &mut self,
        chunk: PcmChunk,
    ) -> Result<Option<TranscriptionUpdate>, ModelError>;

    async fn finish(self: Box<Self>) -> Result<TranscriptionUpdate, ModelError>;
}
```

`TranscriptionStream` is intentionally `Send` but not `Sync`.

Why this departs from the existing `ChatModel` / `CompletionModel` / `EmbeddingModel` pattern:

- a transcription stream owns mutable rolling-buffer state
- `push_chunk()` requires ordered, exclusive mutation
- callers should not share one live stream across tasks without their own synchronization layer

The sharable capability object remains `TranscriptionModel: Send + Sync`; the per-stream session is the stateful, exclusive object.

### Why the Contract Is Streaming-Only

The current capability-introspection model keeps one descriptor per `CapabilityKind`. If ASR were modeled as both request-response and streaming at the capability layer, the descriptor surface would need a broader redesign.

The simpler design is:

- ASR is a streaming capability
- `.wav` transcription is implemented by reading and chunking the file into the same stream contract

`push_chunk()` should return:

- `Ok(None)` when the new chunk does not cross a decode boundary and no new transcript output is available
- `Ok(Some(update))` when the backend emits new partial or final segments

This avoids the awkward empty-update pattern on every audio chunk while keeping the API pull-free.

This avoids unnecessary contract churn and is a cleaner fit to the user requirement that the API handle PCM chunks from a stream.

### Eval and Metadata Additions

Add:

- `EvalTrack::Transcription`
- `BundleFamily::Whisper`

For the first curated bundle and the first backend adapter, quantization should remain explicit and empty:

- `QuantizationSupport::none()`

`EvalTrack::primary_for_descriptor()` should map `CapabilityKind::Transcription` to `EvalTrack::Transcription` in the same way embeddings currently map directly to `EvalTrack::Embeddings`.

## Generic Backend Design

### New Backend Crates

Add ASR backend crates with the same structure as the existing backend crates:

- backend-specific spec type
- backend adapter implementing `BackendAdapter`
- generic `ModelBundle` implementation
- loaded handle implementing `BundleHandle`
- backend-local helper module for artifact policy, CPU/GPU selection, and streaming defaults

The first concrete crate is `libs/model/backends/whisper_cpp`. The second concrete crate is `libs/model/backends/sherpa_onnx`.

### Backend Kind and Checkpoint Format

Add:

- `BackendKind::WhisperCpp`
- `BackendKind::SherpaOnnx`
- `CheckpointFormat::Ggml`
- `CheckpointFormat::Onnx`

This is an additive extension that matches the current enum-based backend and checkpoint design. The ASR slices should not overload `Gguf` for a `ggml` Whisper artifact or overload `Ort` for the `sherpa-onnx` backend-specific execution path.

Current upstream verification:

- the `whisper.cpp` quick-start and model download scripts still describe the curated Whisper artifacts as `ggml` format
- the canonical first-slice filename remains `ggml-base.en.bin`

Design implication:

- add `CheckpointFormat::Ggml` now for the current artifact reality
- document that a future upstream migration to GGUF would justify either a second curated checkpoint variant or a later format migration plan

### Third-Party Dependency Choice

Recommended dependency strategy:

- backend crate depends on `whisper-rs`
- `whisper-rs` remains a thin wrapper over upstream `whisper.cpp`

Why this is the best initial dependency:

- keeps Rust code simpler than maintaining local bindgen output immediately
- preserves the generic-backend pattern already used by the other model backends
- keeps a clear escape hatch: if `whisper-rs` blocks an upstream feature, the backend crate can replace the wrapper internally without changing `libs/model` or `libs/models`

### Backend Runtime Behavior

The backend should:

- load one `ggml` model file from a resolved local path
- create one reusable model context per loaded bundle handle
- create one stream-specific runtime state per open transcription stream
- accumulate incoming PCM chunks into a rolling buffer
- decode at fixed intervals over a bounded window
- emit partial segments when text stabilizes
- emit final segments on `finish()`

Initial backend-local defaults:

- decode step: `500 ms`
- rolling window: `5000 ms`
- target model audio format: `16 kHz`, mono, PCM float internally

These defaults are taken from the upstream realtime example and are intentionally backend-local for v1. They do not need to become stable public config knobs yet.

### CPU and CUDA Handling

Follow the current backend-local pattern:

- CPU-only builds work without extra public API
- CUDA support is enabled by Cargo feature on the backend crate
- runtime still honors the existing `MOTLIE_MODEL_FORCE_CPU` convention where practical

For the `sherpa-onnx` backend specifically, ONNX Runtime provisioning should be explicit. Do not enable `ort` build-time binary download inside the library crate. Require operators to provide ONNX Runtime through `ORT_LIB_PATH`, `pkg-config`, or another explicit system installation path, then layer the backend-local CUDA execution-provider selection on top of that runtime.

This avoids expanding `StartOptions` in the first ASR slice. If multiple backends later need explicit accelerator selection, that can be added as a generic model-layer startup option.

## Curated Bundle Design in `libs/models`

### New Namespace

Add a new family module:

```text
libs/models/src/asr/
  mod.rs
  whisper_base_en.rs
```

This should mirror the current `chat/` and `embeddings/` organization.

### Direct Enum and Selector

Add:

- `AsrModels`
- `ModelSelector::Asr(AsrModels)`
- string selector prefix `asr:`

Recommended selector:

- `asr:openai/whisper_base_en`

This keeps the current family-prefixed selector pattern intact:

- `chat:...`
- `embedding:...`
- `asr:...`

### Curated Bundle Module

`whisper_base_en.rs` should expose:

- `SELECTOR`
- `identity()`
- `checkpoint()`
- `descriptor()`
- `bundle()`
- `register()`
- bundle-local artifact validation

Recommended descriptor contents:

- `id = BundleId::new("whisper_base_en")`
- `display_name = "Whisper Base.en"`
- `family = BundleFamily::Whisper`
- `backend = BackendKind::WhisperCpp`
- `capabilities = Capabilities::new(vec![CapabilityDescriptor::transcription_stream()])`
- `quantization = QuantizationSupport::none()`
- `eval_tracks = vec![EvalTrack::Transcription]`
- `requirements.platform = [Linux, Macos]`
- `requirements.build = [Feature("backend-whisper-cpp".into())]`

### Curated Artifact Rules

Recommended first artifact declaration:

```rust
ModelCheckpoint {
    format: CheckpointFormat::Ggml,
    source: ArtifactSource::HuggingFace {
        repo: "ggerganov/whisper.cpp",
    },
    include: vec![
        ArtifactRule::Exact("ggml-base.en.bin"),
    ],
    quantization: None,
}
```

This is deliberately simple:

- one curated model
- one exact file
- no quantized variants in v1
- no VAD sidecar model in v1

### Artifact Resolution

Bundle-local artifact resolution should follow the existing pattern:

- validate the expected file exists under the curated root
- resolve to the concrete local model path
- pass only the resolved path to the backend adapter

The backend crate should not know about Hugging Face cache layout.

## Streaming PCM API Contract

### Contract Rules

For v1:

- supported encodings: `S16Le`, `F32Le`
- required stream semantics: monotonically increasing `sequence`
- `open_stream()` establishes the immutable `AudioSpec` for the lifetime of the stream
- accepted source types:
  `.wav` decoded into PCM chunks by caller-side helper
  websocket binary frames mapped into PCM chunks by caller-side transport code
- finalization:
  callers must send `end_of_stream = true` on the last chunk or call `finish()`

Edge-case semantics for v1:

- `push_chunk()` after a chunk marked `end_of_stream = true`
  return `ModelError::InvalidConfiguration`
- non-monotonic `sequence`
  return `ModelError::InvalidConfiguration`
- empty `data` with `end_of_stream = false`
  return `Ok(None)` and do no work
- calling after `finish()`
  impossible by construction because `finish()` consumes `Box<Self>`

The first ASR slice reuses the existing `ModelError` surface. If request-validation failure modes grow materially, a future additive error variant can separate caller misuse from startup configuration issues.

### Normalization Rule

The stable public contract accepts PCM metadata and bytes. The backend normalizes to the runtime's preferred format.

For the first curated Whisper bundle:

- incoming audio may be `S16Le` or `F32Le`
- backend normalizes to mono `f32`
- backend resamples to `16 kHz` if needed

This normalization belongs in the backend, not in `libs/model`, because it is runtime-specific operational behavior.

### Partial vs Final Text

Streaming ASR is only useful if partials are represented explicitly. The `final_segment` flag is sufficient for v1:

- `final_segment = false`
  interim text that may still change
- `final_segment = true`
  committed text that should not be rewritten later

This is intentionally smaller than a full diff protocol. The caller can reconcile partials by segment timing.

Because `push_chunk()` returns `Option<TranscriptionUpdate>`, callers only handle transcript material when the backend actually emits it.

## Feature Flag Design

### `libs/model/backends/*`

Recommended features:

- default = `[]`
- `cuda`

Optional later:

- `openvino`
- `metal`

For the implemented backends, this maps to `whisper.cpp`'s `cuda` feature and `sherpa-onnx`'s `cuda` feature.

### `libs/models`

Recommended features:

```toml
[features]
model-whisper-base-en = ["dep:motlie-model-whisper-cpp"]
whisper-cpp-cuda = [
  "dep:motlie-model-whisper-cpp",
  "motlie-model-whisper-cpp/cuda",
]
model-sherpa-onnx-streaming = ["dep:motlie-model-sherpa-onnx"]
sherpa-onnx-cuda = [
  "dep:motlie-model-sherpa-onnx",
  "motlie-model-sherpa-onnx/cuda",
]
```

### Default Feature Policy

Do not add `model-whisper-base-en` to `libs/models` default features in the first ASR PR.

Reasoning:

- native backend dependency increases build complexity compared with the current default slice
- ASR is a new capability family and should stabilize behind an explicit opt-in first
- this matches the repo's existing pattern where not every backend or bundle variant is in the default set

Once the bundle is validated cross-platform, it can be promoted into the default curated slice if desired.

## Migration and Compatibility Strategy

This is brownfield contract work, so the migration path matters even though there are no existing ASR callers.

### Additive API Strategy

Additively extend `libs/model` and update all existing backend handles to return:

```rust
Err(ModelError::UnsupportedCapability(CapabilityKind::Transcription))
```

This is the same unsupported-capability behavior used today for chat, completion, embeddings, and vision mismatches.

### Source-Breaking Scope

The only intentional source break inside the workspace is the `BundleHandle` trait expansion. That break is acceptable because all implementers are in-tree and can be migrated in the same change series.

The exact in-tree handle implementations that must be updated are:

- `libs/model/backends/mistral/src/text.rs`
- `libs/model/backends/mistral/src/multimodal.rs`
- `libs/model/backends/mistral/src/embeddings.rs`
- `libs/model/backends/llama_cpp/src/text.rs`
- `libs/model/src/lib.rs` test `FakeHandle`

### No Existing Caller Migration

There is no public ASR surface to deprecate or preserve. Existing catalog selectors and bundle IDs remain unchanged. The ASR slice only adds:

- one new capability kind
- one new eval track
- one new bundle family
- one new selector family
- one new optional backend crate

## API Sketch

### Caller Experience

```rust
use motlie_model::{
    ArtifactPolicy, AudioSpec, PcmChunk, PcmEncoding, StartOptions, TranscriptionParams,
};
use motlie_models::asr::AsrModels;

let bundle = AsrModels::WhisperBaseEn.bundle();
let handle = bundle
    .start(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: motlie_models::default_artifact_root(),
        }),
        ..Default::default()
    })
    .await?;

let asr = handle.transcription()?;
let mut stream = asr
    .open_stream(
        AudioSpec {
            sample_rate_hz: 16_000,
            channels: 1,
            encoding: PcmEncoding::S16Le,
        },
        TranscriptionParams {
            language: Some("en".into()),
            emit_partials: true,
        },
    )
    .await?;

let update = stream
    .push_chunk(PcmChunk {
        data: pcm_bytes,
        sequence: 0,
        end_of_stream: false,
    })
    .await?;

if let Some(update) = update {
    for segment in update.segments {
        println!("{}", segment.text);
    }
}

let final_update = stream.finish().await?;
```

### `.wav` Source Adaptation

The `.wav` path is a caller-side adapter:

1. read `.wav`
2. decode to PCM
3. chunk into `PcmChunk`
4. feed the same ASR stream

This should be demonstrated in `libs/models/examples/asr_whisper`.

### Websocket Source Adaptation

A websocket server is also a caller-side adapter:

1. accept websocket frames
2. parse binary audio payload
3. map to `PcmChunk`
4. feed the same ASR stream
5. publish `TranscriptionUpdate` back to the client

The network transport belongs outside `libs/model` and `libs/models`.

## Alternatives Considered

### Alternative 1: `faster-whisper` / CTranslate2 as the First Backend

Pros:

- very strong CPU and CUDA throughput
- int8 CPU path is attractive
- robust Whisper-family ecosystem

Cons:

- Python-first operational model
- less natural fit for the existing Rust-native backend-crate pattern
- converted artifact directories are more complex than a single curated `ggml` file
- real-time mode is usually built at a service layer rather than expressed as a native Rust streaming contract

Decision:
Rejected for v1. Keep as the benchmark and service-backend reference when Motlie wants a higher-throughput server-oriented ASR backend.

### Alternative 2: Streaming ONNX Model as the First Bundle

Pros:

- best fit for true low-latency streaming
- good CPU story
- aligns with `CheckpointFormat::Onnx` and the dedicated `BackendKind::SherpaOnnx`
- upstream `sherpa-onnx` has Rust and websocket examples

Cons:

- more operationally specialized model/runtime surface
- multi-file artifact sets
- less compatible with Whisper-family expectations and evaluation datasets already common in the ecosystem
- higher complexity for the first bundle than a single Whisper artifact

Decision:
Rejected for v1 but implemented as the second ASR backend family once the generic transcription capability landed.

### Alternative 3: Add a File-Oriented `transcribe_file()` API Instead of a Streaming Contract

Pros:

- smaller immediate contract
- easiest way to demo `.wav` transcription

Cons:

- fails the live-stream requirement
- pushes websocket handling into a second, incompatible API path
- conflicts with the existing capability-introspection model if we later add streaming as a second interaction form

Decision:
Rejected. The first ASR capability should be streaming-first.

## Testing Scope for PLAN

The implementation plan should cover:

- contract-level unit tests for the new transcription types and capability discovery
- migration tests proving older backends reject `transcription()` cleanly
- backend tests for PCM normalization, rolling-window decode, and partial/final segment behavior
- backend runtime-metrics wiring using the existing latency/memory snapshot helpers
- curated bundle tests for descriptor reviewability and local artifact resolution
- selector and feature-gating tests in `libs/models`
- env-gated end-to-end `.wav` transcription test with pre-downloaded artifacts
- example validation for the `.wav` vertical slice, with websocket adaptation verified in a follow-up slice

## Open Concerns

- Whether the first ASR metric surface should stay within `RuntimeMetrics` only or introduce a dedicated `TranscriptionMetrics` struct immediately
- Whether `whisper-base.en` should remain off by default permanently or only until cross-platform build confidence improves
- Whether bundle-local VAD should stay deferred or be pulled into the first backend implementation if realtime partial quality is poor without it
- Whether `whisper-small.en` should follow immediately after `base.en` if CPU eval results are strong enough

## References

- OpenAI Whisper README: model sizes, memory guidance, and speed tradeoffs
  https://github.com/openai/whisper
- `whisper.cpp` upstream README: CPU support, CUDA support, realtime stream example, memory usage, `ggml` artifact format, and Rust binding pointer
  https://github.com/ggml-org/whisper.cpp
- `whisper.cpp` C API header: PCM-oriented C API and streaming-friendly callback/config surface
  https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/include/whisper.h
- `faster-whisper` README: CPU/GPU benchmarks, int8 support, and streaming-oriented ecosystem references
  https://github.com/SYSTRAN/faster-whisper
- Hugging Face Moonshine docs: realtime-oriented ASR model optimized for resource-constrained devices
  https://huggingface.co/docs/transformers/en/model_doc/moonshine
- Sherpa introduction and docs: Rust bindings, streaming ASR support, and websocket server examples
  https://k2-fsa.github.io/sherpa/
  https://k2-fsa.github.io/sherpa/onnx/python/streaming-websocket-server.html
- ONNX Runtime install docs: CPU/GPU execution-provider packaging expectations
  https://onnxruntime.ai/docs/install

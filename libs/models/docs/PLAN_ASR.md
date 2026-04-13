# ASR Model Support - Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-13 | @codex-asr | Addressed R1 review feedback by adding explicit tasks for the stream-scoped `AudioSpec`, `Option<TranscriptionUpdate>`, runtime metrics, exact brownfield file touch points, `QuantizationSupport::none()`, and the websocket deferral details for the first implementation slice. |
| 2026-04-12 | @codex-asr | Initial PLAN for the first ASR vertical slice. Covers the brownfield contract extension in `libs/model`, a new `whisper.cpp` backend crate, the curated `whisper_base_en` bundle in `libs/models`, and example validation for `.wav` and websocket-fed PCM. |

Derived from [DESIGN_ASR.md](./DESIGN_ASR.md). This PLAN is intentionally vertical-slice oriented: one curated model, one backend family, one streaming transcription contract, and one example path from artifact download to emitted transcript text.

---

## Phase 1: Extend the Core Model Contract

Add ASR to `libs/model` without disturbing the existing bundle lifecycle shape.

### 1.1 - Capability and metadata additions

- [ ] Add `CapabilityKind::Transcription` and `CapabilityDescriptor::transcription_stream()`.
  DESIGN reference: `Core Contract Changes in libs/model`
- [ ] Add `BundleFamily::Whisper`.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [ ] Add `EvalTrack::Transcription` and map it from the new capability descriptor.
  DESIGN reference: `Core Contract Changes in libs/model`
- [ ] Add unit tests for capability discovery and eval-track projection.
  DESIGN reference: `Testing Scope for PLAN`

### 1.2 - Streaming transcription types and traits

- [ ] Add `PcmEncoding`, `AudioSpec`, `PcmChunk`, `TranscriptionParams`, `TranscriptSegment`, and `TranscriptionUpdate`.
  DESIGN reference: `Core Contract Changes in libs/model`, `Streaming PCM API Contract`
- [ ] Add `TranscriptionModel` and `TranscriptionStream`.
  DESIGN reference: `Core Contract Changes in libs/model`
- [ ] Keep `AudioSpec` stream-scoped at `open_stream()` time; do not repeat it on each `PcmChunk`.
  DESIGN reference: `Core Contract Changes in libs/model`, `Streaming PCM API Contract`
- [ ] Make `push_chunk()` return `Option<TranscriptionUpdate>` so no-output decode steps do not force empty-update handling.
  DESIGN reference: `Core Contract Changes in libs/model`
- [ ] Document and test the deliberate ownership model: `TranscriptionModel` is shared, `TranscriptionStream` is `Send` and stateful, but not `Sync`.
  DESIGN reference: `Core Contract Changes in libs/model`
- [ ] Extend `BundleHandle` with `transcription() -> Result<&dyn TranscriptionModel, ModelError>`.
  DESIGN reference: `Core Contract Changes in libs/model`, `Migration and Compatibility Strategy`
- [ ] Extend the fake-handle tests in `libs/model` to cover unsupported transcription behavior.
  DESIGN reference: `Migration and Compatibility Strategy`
- [ ] Define and test the edge-case semantics for:
  post-EOS `push_chunk`,
  non-monotonic `sequence`,
  empty `data`,
  `finish()` ownership.
  DESIGN reference: `Streaming PCM API Contract`

### 1.3 - Brownfield backend migration

- [ ] Update existing backends in `mistral` and `llama_cpp` so `transcription()` returns `UnsupportedCapability(Transcription)`.
  DESIGN reference: `Migration and Compatibility Strategy`
- [ ] Touch the exact in-tree files required by the `BundleHandle` trait expansion:
  `libs/model/backends/mistral/src/text.rs`,
  `libs/model/backends/mistral/src/multimodal.rs`,
  `libs/model/backends/mistral/src/embeddings.rs`,
  `libs/model/backends/llama_cpp/src/text.rs`,
  `libs/model/src/lib.rs`.
  DESIGN reference: `Migration and Compatibility Strategy`
- [ ] Add regression tests proving the existing bundles still reject ASR cleanly.
  DESIGN reference: `Testing Scope for PLAN`

## Phase 2: Add a Generic `whisper.cpp` Backend Crate

Introduce a new backend crate that follows the same adapter-backed shape as the existing inference backends.

### 2.1 - Crate scaffolding and backend enums

- [ ] Add `libs/model/backends/whisper_cpp/Cargo.toml` and wire it into the workspace.
  DESIGN reference: `Generic Backend Design`
- [ ] Add `BackendKind::WhisperCpp` and `CheckpointFormat::Ggml`.
  DESIGN reference: `Generic Backend Design`
- [ ] Document in code comments and tests that the first curated artifact is currently true `ggml` (`ggml-base.en.bin`), not `gguf`.
  DESIGN reference: `Generic Backend Design`
- [ ] Add crate exports for `WhisperCppTranscriptionAdapter`, `WhisperCppTranscriptionBundle`, and `WhisperCppTranscriptionSpec`.
  DESIGN reference: `Generic Backend Design`

### 2.2 - Backend startup and artifact policy

- [ ] Add backend-local artifact-policy helpers that validate a resolved `.bin` model path.
  DESIGN reference: `Generic Backend Design`, `Curated Bundle Design in libs/models`
- [ ] Keep provider-specific cache layout logic out of the backend crate; only consume a resolved local model path from `libs/models`.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [ ] Add backend unit tests for missing-artifact and wrong-checkpoint-format failures.
  DESIGN reference: `Testing Scope for PLAN`

### 2.3 - Streaming runtime implementation

- [ ] Implement stream creation over the new `TranscriptionModel` trait.
  DESIGN reference: `Core Contract Changes in libs/model`, `Streaming PCM API Contract`
- [ ] Normalize incoming PCM to the runtime's internal mono `f32` / `16 kHz` representation.
  DESIGN reference: `Streaming PCM API Contract`
- [ ] Implement rolling-window decode with initial backend-local defaults:
  `step = 500 ms`, `window = 5000 ms`.
  DESIGN reference: `Generic Backend Design`
- [ ] Emit `final_segment = false` for interim results and `final_segment = true` on committed output.
  DESIGN reference: `Streaming PCM API Contract`
- [ ] Return `Ok(None)` from `push_chunk()` when the current chunk does not cross a decode boundary.
  DESIGN reference: `Core Contract Changes in libs/model`
- [ ] Add backend tests for:
  PCM normalization,
  monotonic chunk sequencing,
  partial-output emission,
  final flush on `finish()`.
  DESIGN reference: `Testing Scope for PLAN`

### 2.4 - CPU and CUDA feature support

- [ ] Add a backend-local `cuda` Cargo feature.
  DESIGN reference: `Feature Flag Design`
- [ ] Reuse the existing `MOTLIE_MODEL_FORCE_CPU` convention where practical.
  DESIGN reference: `Generic Backend Design`
- [ ] Wire the backend through the existing runtime metric helpers so ASR surfaces latency and memory snapshots like the current backends.
  DESIGN reference: `Testing Scope for PLAN`
- [ ] Add at least one non-CUDA and one CUDA-build compilation check to CI or local verification guidance.
  DESIGN reference: `Feature Flag Design`, `Testing Scope for PLAN`

## Phase 3: Curated Bundle and Catalog Integration in `libs/models`

Add the first curated ASR bundle using the same registration and selector patterns as the current model families.

### 3.1 - `asr/` namespace and direct enum

- [ ] Add `src/asr/mod.rs` with `AsrModels` and feature-gated `WhisperBaseEn`.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [ ] Add `ModelsError::UnknownAsrModel`.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [ ] Add `ModelSelector::Asr(AsrModels)` with `"asr:openai/whisper_base_en"` parsing.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [ ] Add selector round-trip and disabled-feature `ModelUnavailable` tests.
  DESIGN reference: `Testing Scope for PLAN`

### 3.2 - Curated bundle module

- [ ] Add `src/asr/whisper_base_en.rs` with `SELECTOR`, `identity()`, `checkpoint()`, `descriptor()`, `bundle()`, and `register()`.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [ ] Register the curated model variant through `adapter_backed_bundle()` and `register_model_variant()`.
  DESIGN reference: `Architecture`, `Curated Bundle Design in libs/models`
- [ ] Keep the curated artifact declaration to a single exact include rule for `ggml-base.en.bin`.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [ ] Set `QuantizationSupport::none()` explicitly for the first ASR bundle.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [ ] Add descriptor-reviewability tests and local-artifact resolution tests.
  DESIGN reference: `Testing Scope for PLAN`

### 3.3 - Feature flags

- [ ] Add `model-whisper-base-en = ["dep:motlie-model-whisper-cpp"]`.
  DESIGN reference: `Feature Flag Design`
- [ ] Add `whisper-cpp-cuda = ["dep:motlie-model-whisper-cpp", "motlie-model-whisper-cpp/cuda"]`.
  DESIGN reference: `Feature Flag Design`
- [ ] Keep the ASR bundle out of `default` features in the first implementation PR.
  DESIGN reference: `Feature Flag Design`
- [ ] Add `Catalog::with_defaults()` registration only when `model-whisper-base-en` is compiled in.
  DESIGN reference: `Curated Bundle Design in libs/models`

## Phase 4: Vertical Slice Examples and Verification Paths

Prove the first implementation slice through the simplest end-to-end caller path: `.wav` file transcription over the streaming PCM contract.

### 4.1 - `.wav` example

- [ ] Add `examples/v0.5/main.rs` as the first ASR example binary.
  DESIGN reference: `API Sketch`
- [ ] Support `--wav <path>` by decoding the file to PCM and feeding the stream contract.
  DESIGN reference: `Streaming PCM API Contract`, `API Sketch`
- [ ] Document expected preconditions and output in `examples/v0.5/README.md`.
  DESIGN reference: `API Sketch`

### 4.2 - Websocket-fed PCM example path

- [ ] Defer the websocket adapter example to a follow-up ASR PR after the `.wav` vertical slice is stable.
  DESIGN reference: `Architecture`, `API Sketch`
- [ ] In the follow-up PR, use `tokio-tungstenite` and document a fixed binary frame contract:
  `16 kHz`,
  mono,
  `S16Le`,
  one chunk per websocket message.
  DESIGN reference: `Streaming PCM API Contract`
- [ ] Keep the websocket transport outside the core model crates; only the follow-up example owns frame parsing and session lifecycle.
  DESIGN reference: `Architecture`, `API Sketch`

### 4.3 - Example feature wiring

- [ ] Add a `models_v0_5` example target with `required-features = ["model-whisper-base-en"]`.
  DESIGN reference: `Curated Bundle Design in libs/models`, `Feature Flag Design`
- [ ] Ensure the example builds with and without `whisper-cpp-cuda` as appropriate for the local environment.
  DESIGN reference: `Feature Flag Design`

## Phase 5: End-to-End Validation

Land the first curated ASR slice with concrete verification commands and env-gated runtime checks.

### 5.1 - Required build and test commands

- [ ] `cargo check -p motlie-model`
  DESIGN reference: `Testing Scope for PLAN`
- [ ] `cargo test -p motlie-model --lib`
  DESIGN reference: `Testing Scope for PLAN`
- [ ] `cargo check -p motlie-model-whisper-cpp`
  DESIGN reference: `Testing Scope for PLAN`
- [ ] `cargo test -p motlie-model-whisper-cpp --lib`
  DESIGN reference: `Testing Scope for PLAN`
- [ ] `cargo check -p motlie-models --no-default-features --features "model-whisper-base-en"`
  DESIGN reference: `Testing Scope for PLAN`
- [ ] `cargo test -p motlie-models --lib --no-default-features --features "model-whisper-base-en"`
  DESIGN reference: `Testing Scope for PLAN`
- [ ] `cargo build -p motlie-models --example models_v0_5 --no-default-features --features "model-whisper-base-en"`
  DESIGN reference: `Testing Scope for PLAN`

### 5.2 - Env-gated runtime checks

- [ ] Add an env-gated test that starts the curated bundle from pre-downloaded artifacts and transcribes a known short `.wav` clip.
  DESIGN reference: `Testing Scope for PLAN`
- [ ] Add an env-gated example run that verifies chunked `.wav` input produces partial and final transcript updates over the streaming contract.
  DESIGN reference: `Testing Scope for PLAN`, `Streaming PCM API Contract`
- [ ] Record the expected artifact env var name and directory layout in the example README.
  DESIGN reference: `Curated Bundle Design in libs/models`

## Phase 6: Brownfield Cleanup and Follow-Through

Keep the docs and existing contract artifacts consistent after the code lands.

### 6.1 - Doc synchronization

- [ ] Update `libs/model/docs/DESIGN.md` and `libs/model/docs/PLAN.md` if the implemented contract shape differs materially from the current transcription proposal.
  DESIGN reference: `Migration and Compatibility Strategy`
- [ ] Update `libs/models/docs/DESIGN.md`, `PLAN.md`, and `API.md` once the ASR slice is implemented.
  DESIGN reference: `Curated Bundle Design in libs/models`, `API Sketch`
- [ ] Add changelog entries with `@codex-asr` and the implementation date in each modified doc.
  DESIGN reference: `Migration and Compatibility Strategy`

### 6.2 - Commit and PR hygiene

- [ ] Keep commits scoped to the ASR contract, backend, curated bundle, examples, and related docs.
- [ ] Do not stage harness files such as `AGENTS.md`.
- [ ] Push the feature branch only after the relevant build/test checks for the current patch set pass.
- [ ] Summarize the design-vs-implementation deltas clearly in the PR body if any decisions change during execution.

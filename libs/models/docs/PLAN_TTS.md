# TTS Model Support - Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-16 | @codex-tts | Implemented the official backup TTS backend `motlie-model-qwen3-tts-cpp` with a vendored C API wrapper, curated GGUF bundle wiring, and the `tts_v0.4` example. Validated real CPU synthesis from local GGUF artifacts, validated the optional CUDA build/clippy path on GB10, and documented the current local CUDA runtime fallback to CPU. |
| 2026-04-15 | @cld-review-models | Implemented Phase 2 Qwen3-TTS slice: `motlie-model-qwen3-tts` backend with multi-model ONNX pipeline, vocabulary-based tokenizer, proper log-mel reference-audio conditioning, shape-preserving tensor pipeline, curated bundle `qwen3_tts_12hz_0_6b`, and `tts_v0.2` example with voice cloning. Runtime boundary is in-process ONNX via `motlie-model-ort` with offline safetensors-to-ONNX export prerequisite. |
| 2026-04-14 | @codex-tts | Addressed PR #179 review R1 by documenting the Phase 1 batch-then-chunk stream behavior, the Piper-only `SpeechParams.seed` rejection, and the current `ort` RC session-mutability constraint while keeping the validated artifact-policy fix in scope. |
| 2026-04-14 | @codex-tts | Implemented the Phase 1 Piper slice end to end: additive speech contracts in `libs/model`, shared ONNX helper extraction for Piper and sherpa, the `motlie-model-piper` backend, curated bundle/selector wiring in `libs/models`, the `models_tts_v0_1` example, and env-gated runtime verification. |
| 2026-04-14 | @codex-tts | Addressed R1 review by aligning the plan with the explicit Qwen3-TTS phase-2 slice, adding the shared ONNX Runtime refactor target with ASR sherpa-onnx, removing the duplicate `speaker_id` input path, turning stream edge cases into concrete validation work instead of open contract design, and renaming the planned TTS example path to avoid collision with the live ASR `v0.5` example. |
| 2026-04-13 | @codex-tts | Initial PLAN for the first TTS vertical slice. Centers on one Piper ONNX voice, streamed PCM output, and a `.wav` end-to-end example from artifact download through playback-ready output. |

Derived from [DESIGN_TTS.md](./DESIGN_TTS.md). This PLAN is intentionally vertical-slice oriented: one curated model, one runtime family, one streamed PCM output contract, and one deterministic example path from artifact download to emitted speech.

---

## Phase 1: Extend the Core Model Contract

Add TTS to `libs/model` without disturbing the existing bundle lifecycle shape.

### 1.1 - Capability and metadata additions

- [x] Add `CapabilityKind::Speech` and `CapabilityDescriptor::speech_stream()`.
  DESIGN reference: `Core Contract Changes in libs/model`
- [x] Add `BundleFamily::Piper`.
  DESIGN reference: `Core Contract Changes in libs/model`
- [x] Add `EvalTrack::Speech` and map it from the new capability descriptor.
  DESIGN reference: `Core Contract Changes in libs/model`
- [x] Add unit tests for capability discovery and eval-track projection.
  DESIGN reference: `Testing Scope for PLAN`

### 1.2 - Streaming speech traits and types

- [x] Add `PcmEncoding`, `AudioSpec`, `PcmChunk`, `SpeechParams`, `VoiceConditioning`, and `SpeechRequest`.
  DESIGN reference: `Streaming PCM API Contract`
- [x] Keep `VoiceConditioning` as the single source of truth for speaker selection and cloning; do not add a duplicate `speaker_id` field on `SpeechParams`.
  DESIGN reference: `Streaming PCM API Contract`
- [x] Add `SpeechModel` and `SpeechStream`.
  DESIGN reference: `Streaming PCM API Contract`, `Core Contract Changes in libs/model`
- [x] Extend `BundleHandle` with `speech() -> Result<&dyn SpeechModel, ModelError>`.
  DESIGN reference: `Core Contract Changes in libs/model`, `Migration and Compatibility Strategy`
- [x] Document and test the stream ownership model: `SpeechModel` is shareable, `SpeechStream` is `Send` and stateful, but not `Sync`.
  DESIGN reference: `Streaming PCM API Contract`
- [x] Implement and test the already-designed edge-case semantics for:
  empty text,
  repeated `next_chunk()` after exhaustion,
  `finish()` ownership,
  monotonic chunk sequencing,
  final `end_of_stream` chunk behavior.
  DESIGN reference: `Streaming PCM API Contract`

### 1.3 - Brownfield backend migration

- [x] Update existing backends in `mistral` and `llama_cpp` so `speech()` returns `UnsupportedCapability(Speech)`.
  DESIGN reference: `Migration and Compatibility Strategy`
- [x] Touch the exact in-tree files required by the `BundleHandle` trait expansion:
  `libs/model/src/lib.rs`,
  `libs/model/src/eval.rs`,
  `libs/model/backends/mistral/src/text.rs`,
  `libs/model/backends/mistral/src/multimodal.rs`,
  `libs/model/backends/mistral/src/embeddings.rs`,
  `libs/model/backends/llama_cpp/src/text.rs`.
  DESIGN reference: `Migration and Compatibility Strategy`
- [x] Add regression tests proving existing bundles still reject TTS cleanly.
  DESIGN reference: `Testing Scope for PLAN`

## Phase 2: Add a Generic Piper Backend Crate

Introduce a new backend crate that follows the same adapter-backed shape as the existing inference backends.

### 2.1 - Crate scaffolding and backend enums

- [x] Add `libs/model/backends/piper/Cargo.toml` and wire it into the workspace.
  DESIGN reference: `Generic Backend Design`
- [x] Reuse `BackendKind::Ort` and `CheckpointFormat::Onnx`; do not add a new runtime kind for the first slice.
  DESIGN reference: `Generic Backend Design`, `Distribution Considerations`
- [x] Add crate exports for `PiperSpeechAdapter`, `PiperSpeechBundle`, and `PiperSpeechSpec`.
  DESIGN reference: `Generic Backend Design`

### 2.2 - Backend startup and artifact policy

- [x] Validate that the resolved checkpoint path ends in `.onnx` and that the adjacent `.onnx.json` sidecar exists.
  DESIGN reference: `Generic Backend Design`
- [x] Parse the sidecar config into backend-local structs with explicit error context.
  DESIGN reference: `Generic Backend Design`
- [x] Keep provider-specific cache-layout logic out of this crate; consume only a resolved local checkpoint path from `libs/models`.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [x] Add unit tests for missing-artifact, missing-sidecar, and wrong-format failures.
  DESIGN reference: `Testing Scope for PLAN`

### 2.3 - Text normalization and phonemization

- [x] Implement backend-local text normalization suitable for the curated first voice.
  DESIGN reference: `Generic Backend Design`
- [x] Implement phonemization in-process; do not shell out to a `piper` binary for the production backend.
  DESIGN reference: `Generic Backend Design`
- [x] Add tests for common ASCII text, punctuation, and empty-input handling.
  DESIGN reference: `Testing Scope for PLAN`

### 2.4 - Streamed PCM generation

- [x] Implement `SpeechModel::open_stream()` over the Piper backend.
  DESIGN reference: `Streaming PCM API Contract`, `Generic Backend Design`
- [x] Expose backend audio-format metadata through `AudioSpec`.
  DESIGN reference: `Streaming PCM API Contract`
- [x] Convert backend inference output into monotonic `PcmChunk` emission.
  DESIGN reference: `Streaming PCM API Contract`
- [x] Emit the final chunk with `end_of_stream = true`, then `Ok(None)` on further reads.
  DESIGN reference: `Streaming PCM API Contract`
- [x] Document the current Piper implementation detail: `open_stream()` performs whole-utterance synthesis first, and `next_chunk()` then drains buffered PCM in order.
  DESIGN reference: `Recommended Vertical Slice`, `Streaming PCM API Contract`
- [x] Document the current Piper parameter surface: `speaking_rate` is supported; `seed` and reference-audio conditioning are rejected at runtime.
  DESIGN reference: `Recommended Vertical Slice`, `Streaming PCM API Contract`
- [x] Add backend tests for:
  monotonic chunk sequencing,
  non-empty output for normal text,
  final-chunk semantics,
  deterministic `.wav`-collectable output.
  DESIGN reference: `Testing Scope for PLAN`

### 2.5 - CPU and CUDA feature support

- [x] Add a backend-local `cuda` Cargo feature that enables the ORT CUDA provider.
  DESIGN reference: `Generic Backend Design`, `Feature Flag Design`
- [x] Keep CPU as the default startup mode.
  DESIGN reference: `Recommended Vertical Slice`
- [x] Add at least one non-CUDA and one CUDA-build compilation check to local verification guidance or CI.
  DESIGN reference: `Feature Flag Design`, `Testing Scope for PLAN`
- [x] Keep shared-session inference serialized until Motlie can upgrade off the current `ort` RC or introduce a validated session-pool design.
  DESIGN reference: `Cross-Capability ONNX Runtime Refactor Target`

### 2.6 - Shared ONNX Runtime refactor target

- [x] Factor ORT session/provider/config helpers so they can later be shared with the planned ASR `sherpa-onnx` backend rather than duplicated.
  DESIGN reference: `Generic Backend Design`, `Cross-Capability ONNX Runtime Refactor Target`
- [x] Keep capability-specific graph/session code in the Piper backend while isolating capability-neutral ORT bootstrap and provider selection.
  DESIGN reference: `Cross-Capability ONNX Runtime Refactor Target`

## Phase 3: Curated Bundle and Catalog Integration in `libs/models`

Add the first curated TTS bundle using the same registration and selector patterns as the current model families.

### 3.1 - `tts/` namespace and direct enum

- [x] Add `src/tts/mod.rs` with `TtsModels` and feature-gated `PiperEnUsLjspeechMedium`.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [x] Add `ModelsError::UnknownTtsModel`.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [x] Add `ModelSelector::Tts(TtsModels)` with `tts:piper/en_us_ljspeech_medium` parsing.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [x] Add selector round-trip and disabled-feature `ModelUnavailable` tests.
  DESIGN reference: `Testing Scope for PLAN`

### 3.2 - Curated bundle module

- [x] Add `src/tts/piper_en_us_ljspeech_medium.rs` with `SELECTOR`, `identity()`, `checkpoint()`, `descriptor()`, `bundle()`, and `register()`.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [x] Register the curated model variant through `adapter_backed_bundle()` and `register_model_variant()`.
  DESIGN reference: `Architecture`, `Curated Bundle Design in libs/models`
- [x] Use repo-relative exact artifact rules for:
  `en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx`,
  `en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json`.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [x] Add a local checkpoint resolver that returns the concrete `.onnx` file path under the downloaded HF snapshot.
  DESIGN reference: `Curated Bundle Design in libs/models`
- [x] Add descriptor-reviewability tests and local-artifact resolution tests.
  DESIGN reference: `Testing Scope for PLAN`

### 3.3 - Feature flags

- [x] Add `model-piper-en-us-ljspeech-medium = ["dep:motlie-model-piper"]`.
  DESIGN reference: `Feature Flag Design`
- [x] Add `piper-cuda = ["dep:motlie-model-piper", "motlie-model-piper/cuda"]`.
  DESIGN reference: `Feature Flag Design`
- [x] Keep the TTS bundle out of `default` features in the first implementation PR.
  DESIGN reference: `Feature Flag Design`
- [x] Add `Catalog::with_defaults()` registration only when `model-piper-en-us-ljspeech-medium` is compiled in.
  DESIGN reference: `Curated Bundle Design in libs/models`

## Phase 4: Vertical Slice Output Adapters and Example

Prove the first implementation slice through the simplest end-to-end caller path: synthesize text and write a `.wav` file while preserving the same streamed PCM contract for future sinks.

### 4.1 - `.wav` example

- [x] Add `examples/tts_v0.1/main.rs` as the first TTS example binary.
  DESIGN reference: `API Sketch`, `Output Adapter Boundary`
- [x] Support `--text <value>` and `--wav <path>` by opening a speech stream and collecting PCM into a `.wav` sink.
  DESIGN reference: `Streaming PCM API Contract`, `Output Adapter Boundary`
- [x] Document expected preconditions and output in `examples/tts_v0.1/README.md`.
  DESIGN reference: `API Sketch`

### 4.2 - Local playback sink

- [ ] Keep PulseAudio/ALSA playback out of the first implementation PR unless it is nearly free once the PCM stream exists.
  DESIGN reference: `Output Adapter Boundary`
- [ ] If included, implement it as an example-layer sink, not as part of the backend crate.
  DESIGN reference: `Output Adapter Boundary`

### 4.3 - Telephony/websocket adapters

- [ ] Defer Telnyx/Twilio websocket or RTP adapters to a follow-up PR after the `.wav` path is stable.
  DESIGN reference: `Output Adapter Boundary`
- [ ] In the follow-up PR, keep resampling, codec conversion, and transport framing in the adapter layer outside the core model crates.
  DESIGN reference: `Output Adapter Boundary`
- [ ] Document a simple internal contract for follow-up adapters:
  consume `AudioSpec` + `PcmChunk`,
  own resampling/codec/framing,
  own session lifecycle.
  DESIGN reference: `Output Adapter Boundary`

### 4.4 - Example feature wiring

- [x] Add a `models_tts_v0_1` example target with `required-features = ["model-piper-en-us-ljspeech-medium"]`.
  DESIGN reference: `Feature Flag Design`
- [x] Ensure the example builds both with default CPU features and, where available, with `piper-cuda`.
  DESIGN reference: `Feature Flag Design`

## Phase 5: End-to-End Validation

Land the first curated TTS slice with concrete verification commands and env-gated runtime checks.

### 5.1 - Required build and test commands

- [x] `cargo check -p motlie-model`
  DESIGN reference: `Testing Scope for PLAN`
- [x] `cargo test -p motlie-model --lib`
  DESIGN reference: `Testing Scope for PLAN`
- [x] `cargo check -p motlie-model-piper`
  DESIGN reference: `Testing Scope for PLAN`
- [x] `cargo test -p motlie-model-piper --lib`
  DESIGN reference: `Testing Scope for PLAN`
- [x] `cargo check -p motlie-models --no-default-features --features "model-piper-en-us-ljspeech-medium"`
  DESIGN reference: `Testing Scope for PLAN`
- [x] `cargo test -p motlie-models --lib --no-default-features --features "model-piper-en-us-ljspeech-medium"`
  DESIGN reference: `Testing Scope for PLAN`
- [x] `cargo build -p motlie-models --example models_tts_v0_1 --no-default-features --features "model-piper-en-us-ljspeech-medium"`
  DESIGN reference: `Testing Scope for PLAN`

### 5.2 - Env-gated runtime checks

- [x] Add an env-gated test that starts the curated bundle from pre-downloaded artifacts and synthesizes a known short sentence.
  DESIGN reference: `Testing Scope for PLAN`
- [x] Add an env-gated example run that verifies chunked output can be collected into a valid `.wav` file.
  DESIGN reference: `Testing Scope for PLAN`, `Output Adapter Boundary`
- [x] Record the expected artifact env var names and directory layout in the example README.
  DESIGN reference: `Curated Bundle Design in libs/models`

## Phase 6: Qwen3-TTS Second Vertical Slice

Only start this after the Piper slice is stable and the speech contract has held up under real usage.

### 6.1 - Backend/runtime boundary

- [x] Add `libs/model/backends/qwen3_tts/` as a distinct backend family rather than extending the Piper backend.
  DESIGN reference: `Phase-2 Qwen3-TTS Vertical Slice`
  @cld-review-models 2026-04-15 — implemented as `motlie-model-qwen3-tts` with multi-model ONNX pipeline.
- [x] Keep the public `SpeechRequest` / `SpeechStream` contract unchanged; all Qwen3-specific cloning and voice-design behavior must map onto the existing request surface.
  DESIGN reference: `Phase-2 Qwen3-TTS Vertical Slice`, `Streaming PCM API Contract`
  @cld-review-models 2026-04-15 — extended `VoiceConditioning::ReferenceAudio` with `reference_text: Option<String>` for prompted cloning.
- [x] Choose and document the runtime boundary explicitly: in-process ONNX Runtime via `motlie-model-ort` with offline safetensors-to-ONNX export prerequisite.
  DESIGN reference: `Phase-2 Qwen3-TTS Vertical Slice`
  @cld-review-models 2026-04-15 — documented in DESIGN_TTS.md Phase 2 section and export guide.

### 6.2 - Curated bundle shape and feature flags

- [x] Add `src/tts/qwen3_tts_12hz_0_6b.rs` with curated selector, descriptor, bundle, and registration.
  DESIGN reference: `Phase-2 Qwen3-TTS Vertical Slice`
  @cld-review-models 2026-04-15 — implemented with identity/checkpoint/descriptor/bundle/register pattern.
- [x] Add `model-qwen3-tts-0_6b = ["dep:motlie-model-qwen3-tts"]`.
  DESIGN reference: `Feature Flag Design`, `Phase-2 Qwen3-TTS Vertical Slice`
- [x] Add `qwen3-tts-cuda = ["dep:motlie-model-qwen3-tts", "motlie-model-qwen3-tts/cuda"]`.
  DESIGN reference: `Feature Flag Design`, `Phase-2 Qwen3-TTS Vertical Slice`
- [x] Keep the Qwen3-TTS bundle out of `default` features until artifact size and runtime deployment are proven manageable.
  DESIGN reference: `Phase-2 Qwen3-TTS Vertical Slice`

### 6.3 - End-to-end validation

- [x] Add one `.wav` parity example that uses the same sink contract as the Piper example.
  DESIGN reference: `Phase-2 Qwen3-TTS Vertical Slice`, `Output Adapter Boundary`
  @cld-review-models 2026-04-15 — `tts_v0.2` example with `--text` and `--wav` flags.
- [x] Add one reference-audio cloning example using `VoiceConditioning::ReferenceAudio`.
  DESIGN reference: `Phase-2 Qwen3-TTS Vertical Slice`, `Streaming PCM API Contract`
  @cld-review-models 2026-04-15 — `--reference-audio` and `--reference-text` flags in `tts_v0.2`.
- [ ] Add one CUDA-oriented smoke test on GB10-class hardware when such hardware is available in the validation environment.
  DESIGN reference: `Phase-2 Qwen3-TTS Vertical Slice`

## Phase 7: `qwen3-tts.cpp` Backup Backend

This phase is the official backup-TTS track attached to issue `#188`.

### 7.1 - Backend/runtime boundary

- [x] Add `libs/model/backends/qwen3_tts_cpp/` as a distinct backend family that wraps the vendored `qwen3tts_c_api.h` boundary rather than extending the ONNX-based Qwen backend.
  DESIGN reference: `Phase-3 qwen3-tts.cpp Backup Vertical Slice`
- [x] Reuse the public `SpeechRequest` / `SpeechStream` contract with buffered PCM chunk emission in front of whole-utterance native synthesis.
  DESIGN reference: `Phase-3 qwen3-tts.cpp Backup Vertical Slice`, `Streaming PCM API Contract`
- [x] Support `VoiceConditioning::ReferenceAudio` through the native voice-sample cloning entry point.
  DESIGN reference: `Phase-3 qwen3-tts.cpp Backup Vertical Slice`, `Streaming PCM API Contract`

### 7.2 - Curated bundle shape and feature flags

- [x] Add `src/tts/qwen3_tts_cpp.rs` with curated selector, descriptor, bundle, and registration.
  DESIGN reference: `Phase-3 qwen3-tts.cpp Backup Vertical Slice`
- [x] Add `model-qwen3-tts-cpp = ["dep:motlie-model-qwen3-tts-cpp"]`.
  DESIGN reference: `Feature Flag Design`, `Phase-3 qwen3-tts.cpp Backup Vertical Slice`
- [x] Add `qwen3-tts-cpp-cuda = ["dep:motlie-model-qwen3-tts-cpp", "motlie-model-qwen3-tts-cpp/cuda"]`.
  DESIGN reference: `Feature Flag Design`, `Phase-3 qwen3-tts.cpp Backup Vertical Slice`
- [x] Resolve GGUF artifacts from the curated `koboldcpp/tts` cache layout or a direct model directory.
  DESIGN reference: `Phase-3 qwen3-tts.cpp Backup Vertical Slice`, `Distribution Considerations`

### 7.3 - End-to-end validation

- [x] Add the `.wav` parity example `tts_v0.4`.
  DESIGN reference: `Phase-3 qwen3-tts.cpp Backup Vertical Slice`, `Output Adapter Boundary`
- [x] Validate real CPU synthesis from local GGUF artifacts and confirm that the emitted `.wav` is valid 24 kHz mono output.
  DESIGN reference: `Phase-3 qwen3-tts.cpp Backup Vertical Slice`
- [x] Validate CPU build/test/clippy for `motlie-model-qwen3-tts-cpp` and the curated `motlie-models` example target.
  DESIGN reference: `Testing Scope for PLAN`
- [x] Validate the optional `qwen3-tts-cpp-cuda` build/clippy path on GB10-class Linux and document the current local runtime fallback message (`ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected`).
  DESIGN reference: `Phase-3 qwen3-tts.cpp Backup Vertical Slice`

## Phase 8: Follow-On Breadth After the Vertical Slice

Only broaden after the Piper slice is stable and the speech contract has held up under real usage.

### 8.1 - Higher-value follow-ons

- [ ] Add a second curated Piper voice, ideally one that exercises speaker selection or a different sample rate.
  DESIGN reference: `Recommended Vertical Slice`, `Distribution Considerations`
- [ ] Evaluate Kokoro as the first higher-quality CPU-oriented alternate backend family if Piper maintenance risk becomes material.
  DESIGN reference: `Alternatives Considered`
- [ ] Evaluate XTTS v2 as a secondary cloning-focused backend family after Qwen3-TTS if the product later prioritizes reference-audio conditioning over CPU-first simplicity.
  DESIGN reference: `Alternatives Considered`

### 8.2 - Doc synchronization

- [ ] Update `libs/model/docs/DESIGN.md` and `libs/model/docs/PLAN.md` if the implemented speech contract differs materially from this proposal.
  DESIGN reference: `Migration and Compatibility Strategy`
- [ ] Update `libs/models/docs/DESIGN.md`, `PLAN.md`, and `API.md` once the TTS slice is implemented.
  DESIGN reference: `Curated Bundle Design in libs/models`, `API Sketch`
- [ ] Add changelog entries with `@codex-tts` and the implementation date in each modified doc.
  DESIGN reference: `Migration and Compatibility Strategy`

## Phase 9: Commit and PR Hygiene

- [ ] Keep commits scoped to the TTS contract, backend, curated bundle, examples, and related docs.
- [ ] Do not stage harness files such as `AGENTS.md`.
- [ ] Push the feature branch only after the relevant checks for the current patch set pass.
- [ ] Summarize any design-vs-implementation deltas clearly in the PR body.

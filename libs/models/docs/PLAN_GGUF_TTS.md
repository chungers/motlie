# GGUF-Based Qwen3-TTS Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-15 | @codex-asr | Initial phased plan for a `llama.cpp + ORT` Qwen3-TTS exploration that keeps Motlie at three engines total. Centers on proving the talker-stage boundary first, then only expanding to a hybrid backend if that proof succeeds. |

Derived from [DESIGN_GGUF_TTS.md](./DESIGN_GGUF_TTS.md).

Tracking issue: `#188` "Qwen3-TTS via GGUF — llama.cpp multi-stage pipeline and qwen3tts.cpp evaluation"

This plan is intentionally gated. The key architectural risk is not "can we write Rust around Qwen3-TTS?" The key risk is whether Motlie can make the large talker stage `llama.cpp`-compatible without dragging in a fourth runtime engine. The early phases exist to answer that question before implementation breadth expands.

---

## Phase 1: Prove the Stage Boundary

Goal: validate the exact boundary between the large causal talker stage and the smaller post-LM stages.

### 1.1 - Freeze the upstream contract

- [ ] Capture the official Qwen3-TTS 12 Hz config fields Motlie depends on:
  - text vocab size
  - codec vocab size
  - code predictor vocab size
  - number of code groups
  - special token IDs
  - language IDs
  DESIGN reference: `Qwen3-TTS Pipeline Decomposition`
- [ ] Check in a small inspection note or fixture under `docs/` or `libs/models/docs/` that records the expected prompt and output-token structure for the talker stage.
  DESIGN reference: `Qwen3-TTS Pipeline Decomposition`
- [ ] Confirm whether the code predictor can be treated as an ORT stage without changing the public `SpeechModel` contract.
  DESIGN reference: `Recommended Architecture for Motlie`

### 1.2 - Define success criteria for `llama.cpp` reuse

- [ ] Write explicit acceptance criteria for "Stage 1 is reusable through `llama.cpp`":
  - Motlie can load the talker weights through the existing `llama_cpp.rs` boundary or a minimal extension of it
  - Motlie can sample codebook-0 acoustic tokens with correct special-token handling
  - the remaining codebooks can be generated outside `llama.cpp` without changing the `SpeechRequest` contract
  DESIGN reference: `Key Finding: Can llama.cpp Run Qwen3-TTS GGUF Today?`
- [ ] Reject the phase if the only working path requires Motlie to embed a separate full custom runtime.
  DESIGN reference: `Recommendation`

## Phase 2: Build the Shared GGUF / GGML Substrate

Goal: factor artifact logic once so chat, ASR, and future TTS backends reuse the same GGUF / GGML resolution patterns.

### 2.1 - Consolidate artifact resolution helpers

- [ ] Extract common GGUF / GGML file-resolution logic from:
  - `libs/model/backends/llama_cpp/src/common.rs`
  - `libs/model/backends/whisper_cpp/src/common.rs`
  DESIGN reference: `Shared GGUF / GGML Substrate Design`
- [ ] Centralize:
  - exact filename resolution
  - suffix-based quantization selection
  - snapshot-root validation
  - ambiguous-artifact errors
  DESIGN reference: `Shared GGUF / GGML Substrate Design`
- [ ] Keep runtime execution concerns out of this layer.
  DESIGN reference: `Shared GGUF / GGML Substrate Design`

### 2.2 - Add metadata inspection hooks

- [ ] Add lightweight GGUF / GGML metadata probing sufficient to log or validate:
  - architecture
  - quantization label
  - selected artifact filename
  DESIGN reference: `Shared GGUF / GGML Substrate Design`
- [ ] Use the shared helper in at least one existing GGUF-backed path and one GGML-backed path before introducing TTS-specific code.
  DESIGN reference: `Shared GGUF / GGML Substrate Design`

## Phase 3: Prototype the Hybrid Runtime

Goal: prove the `llama.cpp + ORT` split end to end on a narrow non-production prototype.

### 3.1 - Talker proof of concept

- [ ] Implement a prototype loader for a talker-only checkpoint path.
  DESIGN reference: `Recommended Architecture for Motlie`
- [ ] Reproduce the TTS prefill layout with the correct text and codec special IDs.
  DESIGN reference: `Qwen3-TTS Pipeline Decomposition`
- [ ] Verify Motlie can sample codebook-0 tokens deterministically for a fixed prompt.
  DESIGN reference: `Tokens and Sampling`

### 3.2 - ORT post-LM proof of concept

- [ ] Run the smaller post-LM stages through ORT:
  - code predictor
  - speech-tokenizer decoder
  DESIGN reference: `Qwen3-TTS Pipeline Decomposition`
- [ ] Verify the codebook handoff shape exactly:
  - codebook-0 from `llama.cpp`
  - codebooks 1-15 from ORT code predictor
  - final decode to PCM from ORT tokenizer decoder
  DESIGN reference: `Recommended Architecture for Motlie`
- [ ] Keep voice-cloning support out of the first proof unless speaker encoder integration is nearly free.
  DESIGN reference: `Risks and Open Questions`

### 3.3 - Kill criteria

- [ ] Stop the implementation track if the prototype shows that Motlie must either:
  - add `qwen3-tts.cpp` as a fourth engine, or
  - carry a large permanent `llama.cpp` architecture fork
  DESIGN reference: `Recommendation`

## Phase 4: Integrate into `libs/model`

Goal: wrap the proven hybrid runtime behind the existing Motlie TTS contract.

### 4.1 - Backend shape

- [ ] Add a new backend crate only after the prototype passes:
  - recommended name: `libs/model/backends/qwen3_tts_hybrid/`
  DESIGN reference: `Curated Bundle Design Impact`
- [ ] Keep the public contract on the existing `SpeechModel` / `SpeechStream` traits.
  DESIGN reference: `Recommended Architecture for Motlie`
- [ ] Keep static dispatch and explicit typed errors at all backend boundaries.
  DESIGN reference: repository Rust guidelines

### 4.2 - Engine ownership inside the backend

- [ ] `llama.cpp` owns:
  - talker model startup
  - talker prompt assembly
  - codebook-0 generation
- [ ] ORT owns:
  - code predictor startup
  - tokenizer decoder startup
  - optional speaker encoder startup in later phases
  DESIGN reference: `Recommended Architecture for Motlie`

## Phase 5: Curated Bundle and Feature Flags

Goal: make the hybrid runtime a curated Motlie model, not a loose example.

### 5.1 - Bundle design

- [ ] Add one curated bundle in `libs/models/src/tts/` for the narrowest viable Qwen3-TTS slice.
  DESIGN reference: `Curated Bundle Design Impact`
- [ ] Keep the selector singular even if the backend consumes multiple checkpoints.
  DESIGN reference: `Curated Bundle Design Impact`
- [ ] Resolve all required assets from one curated snapshot root where possible.
  DESIGN reference: `Checkpoint and Engine Decoupling`

### 5.2 - Feature flags

- [ ] Add model and runtime feature flags that stay orthogonal:
  - model feature for the curated bundle
  - existing `llama.cpp` accelerator flags
  - existing ORT accelerator flags as needed
  DESIGN reference: `Overview`, `Recommended Architecture for Motlie`
- [ ] Do not add a `qwen3tts-cpp` feature flag because that would codify the wrong engine boundary.
  DESIGN reference: `Recommendation`

## Phase 6: Validation and UX

Goal: prove the vertical slice from artifact resolution to emitted PCM.

### 6.1 - Example program

- [ ] Add one example binary that:
  - starts the curated hybrid bundle
  - synthesizes a short utterance
  - writes a `.wav`
  DESIGN reference: `Curated Bundle Design Impact`
- [ ] Document exact runtime prerequisites:
  - cache layout
  - feature flags
  - any authenticated artifact-download prerequisite
  DESIGN reference: `Risks and Open Questions`

### 6.2 - Verification matrix

- [ ] `cargo check`
- [ ] `cargo test`
- [ ] `cargo clippy -- -D warnings`
- [ ] example build
- [ ] end-to-end `.wav` validation with a known prompt
- [ ] if CUDA is part of the slice, one CUDA-startup smoke check

### 6.3 - Behavioral checks

- [ ] verify correct language-ID handling for at least one multilingual prompt
- [ ] verify monotonic streamed PCM output through `SpeechStream`
- [ ] verify the hybrid runtime rejects unsupported artifact layouts with explicit error context
- [ ] verify no `unwrap()` or `expect()` exists in non-test code

## Phase 7: Deferred Follow-Ups

Only after the hybrid path is working and stable:

- [ ] voice cloning through ORT speaker-encoder integration
- [ ] broader quantization catalog
- [ ] larger `1.7B` curated variants
- [ ] upstream `llama.cpp` architecture contribution if Motlie decides the temporary export path should become first-class
- [ ] reevaluate direct `qwen3-tts.cpp` FFI only if:
  - its license becomes explicit and acceptable
  - the hybrid `llama.cpp + ORT` path proves infeasible

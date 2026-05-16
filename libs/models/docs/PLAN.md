# Curated Model Bundle Library — Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-24 | @codex-gpt55 | Started Phase 9 implementation: Qwen3.6 GGUF bundle/catalog wiring, Q4/Q5/Q8 quant support, and the dedicated example are in progress; FP8 remains blocked until a curated FP8 GGUF artifact exists. |
| 2026-04-24 | @codex-gpt55 | Clarified Phase 9 after reviewing `libs/model/src/chat.rs`: the core chat contract already accepts image content, so Qwen3.6 needs llama.cpp mmproj/image backend support before advertising vision, not a new core interface. |
| 2026-04-24 | @codex-gpt55 | Added Phase 9 for issue `#224`, covering Qwen3.6 27B GGUF via the existing llama.cpp backend, platform-aware GGUF quant defaults, optional multimodal support, feature-gated catalog wiring, and the dedicated example. |
| 2026-04-18 | @codex-asr | Added Phase 8 for the reusable strict audio-ingress follow-through required by issue `#198` and Telnyx PR `#186`. This phase keeps provider-agnostic decode/resample/chunking invariants out of the Telnyx adapter crate while making the first telephony vertical slice consume the typed model boundary explicitly. |
| 2026-04-17 | @codex-asr | Renamed the curated example targets from versioned names to capability/model names and updated the plan references accordingly (`embeddings`, `chat_mistral_qwen3`, `chat_multimodal_gemma4`, `chat_gguf_gwen3_gemma4`, `asr_whisper`, `asr_sherpa_onnx`, `asr_moonshine`, `tts_piper`, `tts_qwen3_onnx`, `tts_qwen3_tts_cpp`). |
| 2026-04-07 | @codex-researcher | Initial PLAN for `libs/models` vertical slice work. Covers the curated catalog, constructor registration, and the first `embeddinggemma_300m` bundle wired through the Mistral backend. |
| 2026-04-07 | @codex-researcher | Marked the completed catalog, descriptor, artifact-control, and verification work for the first embedding slice. | Phases 1-4 |
| 2026-04-07 | @codex-researcher | Updated the bundle plan after the NaN investigation. The `embeddinggemma_300m` descriptor now captures the full sentence-transformers module stack, and an env-gated catalog test verifies finite local-only embeddings end to end. |
| 2026-04-08 | @codex-researcher | Reconciled the PLAN with the current public API after review. `SupportTier` and `PackagingMode` were removed from the v1 surface earlier, so the PLAN now tracks the descriptor fields that actually exist in code and docs. | Phase 1, Phase 2 |
| 2026-04-08 | @claude | Added Phase 5 for the Qwen3-4B chat bundle (#141). Covers `ChatModels` enum, `ModelSelector::Chat`, curated artifact rules, HF cache resolution, and `v0.2` example. | Phase 5 |
| 2026-04-08 | @codex-researcher | Added Phase 6 for the Gemma 4 E2B-it multimodal chat bundle (#142). Covers the feature-gated chat module, multimodal artifact rules, local snapshot validation, selector/catalog wiring, and `v0.3`. | Phase 6 |
| 2026-04-09 | @codex-researcher | Tightened the versioned-example convention. `v0.2` and `v0.3` remain single-bundle builds; `v0.1` is now the explicit two-bundle embedding comparison example, so the binary requires `--embedding=...` and prints the compiled selector set. | Phases 4-7 |
| 2026-04-09 | @codex-researcher | Collapsed the duplicate Gemma 4 examples into a single `v0.3` flow. `v0.3` now owns both optional artifact download and local-only startup, preserving the one-model-per-example convention. | Phase 6 |
| 2026-04-09 | @codex-researcher | Added Phase 7 for the Qwen3-Embedding-0.6B curated bundle (#147). Covers the new Mistral embedding arch, curated bundle module, feature-gated selector/catalog wiring, and the generalized `v0.1` embedding comparison example. | Phase 7 |

Derived from [DESIGN.md](./DESIGN.md). This PLAN focuses on the first curated bundle slice rather than the full long-term catalog.

---

## Phase 1: Catalog and Descriptor Model

Make the curated catalog concrete enough to support both listing and instantiation.

### 1.1 — Descriptor model

- [x] Finalize `BundleDescriptor` to include:
  `id`, `display_name`, `family`, `capabilities`, `backend`, `requirements`, `eval_tracks`, and optional curated `artifacts`.
  DESIGN reference: `Bundle Catalog Model`
- [x] Finalize `BundleFamily`, `BackendKind`, `PlatformConstraint`, `BuildConstraint`, `BundleRequirements`, and curated artifact-control types.
  DESIGN reference: `Bundle Catalog Model`, `Packaging and Deployment Model`
- [x] Add unit tests for descriptor equality, evaluation-track filtering, and capability-descriptor projection.
  DESIGN reference: `Testing Scope for PLAN`

### 1.2 — Catalog behavior

- [x] Finalize `Catalog` as an in-memory registry of descriptors plus constructors.
  DESIGN reference: `Architecture`, `API Sketch`
- [x] Finalize `register`, `bundle`, `bundles`, `bundles_for_track`, and `instantiate`.
  DESIGN reference: `Architecture`
- [x] Add `Catalog::with_defaults()` for the first curated slice.
  DESIGN reference: `API Sketch`
- [x] Add unit tests for registration overwrite semantics and constructor-based instantiation.
  DESIGN reference: `Testing Scope for PLAN`

## Phase 2: First Curated Bundle

Define the first curated embedding stack explicitly in this crate.

### 2.1 — `embeddinggemma_300m` descriptor

- [x] Finalize `embeddinggemma_300m_descriptor()`.
  DESIGN reference: `Bundle Catalog Model`
- [x] Ensure the descriptor declares:
  `BundleFamily::Embeddings`,
  `BackendKind::MistralRs`,
  `EvalTrack::Embeddings`,
  and `CapabilityDescriptor::embeddings()`.
  DESIGN reference: `Capability Exposure`, `Evaluation and Curation Harnesses`
- [x] Add tests for descriptor contents so the first curated stack is reviewable as data.

### 2.2 — Curated constructor binding

- [x] Finalize `embeddinggemma_300m_bundle()` so it binds the descriptor to `libs/model/backends/mistral`.
  DESIGN reference: `Internal Backend Boundary`
- [x] Keep bundle-specific artifact/build constraints in `libs/models`, not in the backend crate.
  DESIGN reference: `Internal Backend Boundary`, `Bundle-Local Build Customization`
- [x] Add tests that `Catalog::with_defaults()` includes `embeddinggemma_300m` and can instantiate it.

## Phase 3: Packaging and Build Constraints for the First Slice

Keep this limited and explicit for the first embedder.

### 3.1 — First-slice constraints

- [x] Decide and document the first-slice artifact assumption:
  local sidecar assets only, or configurable artifact path.
  DESIGN reference: `Packaging and Deployment Model`
- [x] Keep the first build constraints minimal and inspectable.
  DESIGN reference: `Build Reasoning and Profile Clarity`
- [x] Ensure the first slice does not overcommit to DGX/Debian/macOS profile complexity before the utility work lands.
  DESIGN reference: `Build Reasoning and Profile Clarity`

## Phase 4: Vertical Slice Validation

### 4.1 — Cross-crate end-to-end check

- [x] Verify the following flow compiles and runs:
  `Catalog::with_defaults()` -> `instantiate("embeddinggemma_300m")` -> `start()` -> `embeddings()` -> `embed()`
  DESIGN reference: `API Sketch`
- [x] Add one small test in `libs/models` covering that path against the backend crate’s current implementation.

### 4.2 — Required verification commands

- [x] `cargo check -p motlie-models`
- [x] `cargo test -p motlie-models`
- [x] `cargo check -p motlie-model -p motlie-model-mistral -p motlie-models`

## Phase 5: Qwen3-4B Chat Bundle

Add the first curated chat bundle to validate the `ChatModel` + `CompletionModel` contract paths.

### 5.1 — Chat module and Qwen3-4B bundle

- [x] Add `src/chat/mod.rs` with `ChatModels` enum and feature-gated `Qwen3_4B` variant.
- [x] Add `src/chat/qwen3_4b.rs` with `descriptor()`, `bundle()`, and HF cache resolution.
- [x] Define curated artifact rules for standard transformer layout (config, tokenizer, safetensors).
- [x] Implement `ModelBundle` for `Qwen3_4B`, delegating to `MistralTextBundle`.

### 5.2 — Catalog and selector integration

- [x] Add `ModelSelector::Chat(ChatModels)` variant with `"chat:qwen/qwen3_4b"` parsing.
- [x] Add `ModelsError::UnknownChatModel` for unknown chat selectors.
- [x] Add `model-qwen3-4b` Cargo feature, included in defaults.
- [x] Register in `Catalog::with_defaults()`.

### 5.3 — Example and verification

- [x] Add `examples/chat_mistral_qwen3` demonstrating chat, multi-turn, completion, ISQ quantization.
- [x] `cargo check -p motlie-models`
- [x] `cargo test -p motlie-models --lib`
- [x] `cargo check -p motlie-models --no-default-features`
- [ ] Env-gated end-to-end test with pre-downloaded Qwen3-4B artifacts.

## Phase 6: Gemma 4 E2B-it Multimodal Chat Bundle

### 6.1 — Chat module and Gemma bundle

- [x] Add `src/chat/gemma4_e2b.rs` with `descriptor()`, `bundle()`, and curated local snapshot resolution.
- [x] Add `ChatModels::Gemma4E2B` behind `model-gemma4-e2b`.
- [x] Define curated artifact rules for Gemma 4 multimodal startup, including processor config files.
- [x] Implement `ModelBundle` for `Gemma4E2B`, delegating to `MistralMultimodalBundle`.

### 6.2 — Catalog and selector integration

- [x] Support `"chat:google/gemma4_e2b"` in `ModelSelector`.
- [x] Register the bundle in `Catalog::with_defaults()`.
- [x] Add selector/direct-enum tests for Gemma 4 matching the earlier bundle conventions.

### 6.3 — Example and verification

- [x] Add `examples/chat_multimodal_gemma4` demonstrating text-only and image+text chat through the Gemma 4 bundle, with optional `--download-artifacts` for the convenience path.
- [x] `cargo test -p motlie-models --lib`
- [x] `cargo build -p motlie-models --example chat_multimodal_gemma4`
- [ ] Env-gated end-to-end example run with pre-downloaded Gemma 4 E2B-it artifacts.

## Phase 7: Qwen3-Embedding-0.6B Curated Bundle

### 7.1 — Mistral embedding backend support

- [x] Add `MistralEmbeddingArch::Qwen3Embedding` to the generic `mistral` embedding backend.
- [x] Add `MistralEmbeddingSpec::qwen3_embedding_06b()` with curated identity and `Q8`-only quantization support, leaving `F32` as the unquantized default.
- [x] Add backend unit coverage for the new arch/spec and its quantization metadata.

### 7.2 — Curated bundle module and local-only artifact contract

- [x] Add `src/embeddings/qwen3_embedding_06b.rs` with `descriptor()`, `bundle()`, `embedding_spec()`, and local HF snapshot resolution.
- [x] Keep provider-specific cache-layout validation in `libs/models`, not the generic backend.
- [x] Add unit tests for descriptor reviewability, embedding semantics, and local snapshot acceptance/rejection.

### 7.3 — Selector, catalog, and build gating

- [x] Add `model-qwen3-embedding-06b` Cargo feature and include it in the default curated slice.
- [x] Extend `EmbeddingModels`, `ModelSelector`, and `Catalog::with_defaults()` with the new embedding bundle under the same per-bundle feature-gating convention as the earlier slices.
- [x] Add tests for selector round-trip, disabled-feature `ModelUnavailable`, and the single-embedding-build helper used by `embeddings`.

### 7.4 — Example and verification

- [x] Update `examples/embeddings` into the shared embedding comparison binary built with both embedding bundle features enabled, requiring `--embedding=...` to choose the model at download/run time.
- [x] Add optional `--precision=q4|q8|f32` handling to `embeddings`; bundle metadata enforces the supported subset at startup.
- [x] `cargo test -p motlie-model-mistral --lib`
- [x] `cargo test -p motlie-models --lib`
- [x] `cargo build -p motlie-models --no-default-features --features "model-google-gemma-300m model-qwen3-embedding-06b" --example embeddings`
- [ ] Env-gated end-to-end example run with pre-downloaded Qwen3-Embedding-0.6B artifacts.

## Phase 8: Strict Audio-Ingress Follow-Through for Live Voice

Keep the reusable audio invariants with the model/voice layers rather than in a
provider-specific Telnyx adapter. This phase is derived from `DESIGN.md`:
`Strict Typed Audio Boundary`, `Transport and Telephony Ingress Boundary`, and
`Testing Scope for PLAN`.

### 8.1 — Reusable typed ingress and transform surface

- [ ] Finalize the provider-agnostic typed ingress story so live voice adapters
  terminate in explicit `AudioBuf<...>` values rather than raw PCM bytes.
- [ ] Add the reusable transform types that the first telephony slice depends on
  where they belong in the shared stack:
  decode, sample-format conversion, resample, channel normalization, and
  explicit chunk/framing adapters.
- [ ] Keep those transforms closed over concrete input/output types so invalid
  pairings fail at compile time instead of being deferred to runtime.

### 8.2 — First telephony pairing contract

- [ ] Write down the only supported first live path explicitly:
  `G.711 mu-law 8 kHz mono -> decoded typed PCM -> resampled 16 kHz mono PCM -> sherpa-onnx`.
- [ ] Write down the outbound first live path explicitly:
  `Piper PCM -> explicit normalize/resample -> telephony encode`.
- [ ] Reject undeclared pairings in the first live voice slice rather than
  letting transport code broaden the backend matrix ad hoc.

### 8.3 — Verification

- [ ] Add compile-fail coverage for invalid typed audio compositions, for example
  feeding the wrong sample type, wrong sample rate, or wrong channel layout into
  a backend session.
- [ ] Keep every shipped ASR/TTS example on the typed API surface and reject
  example-only helper paths that reintroduce raw byte PCM or erased capability
  adapters.
- [ ] Add an integration test or example path that exercises the first telephony
  adaptation chain without any provider-specific network transport in the loop.
- [ ] Add the corresponding script/CI hook once that path exists so the
  telephony-ingress contract does not regress back to byte-oriented plumbing.

## Phase 9: Qwen3.6 27B GGUF via llama.cpp

Add the large Qwen3.6 27B GGUF curated bundle for issue `#224`. The detailed
design is in [`DESIGN_QWEN3_6_27B_GGUF.md`](./DESIGN_QWEN3_6_27B_GGUF.md).

### 9.1 — Design, artifact validation, and interface boundary

- [x] Add a dedicated design document under `libs/models/docs/` covering the
  model identity, backend reuse, feature gates, quant policy, interfaces, and
  planned files.
- [x] Validate the selected GGUF artifact source and record exact filenames for
  Q4-class, Q5-class, Q8, and optional mmproj artifacts.
- [x] Record that FP8 is not currently available as a curated GGUF artifact;
  the official FP8 release is safetensors/Transformers format.
- [ ] Confirm whether the current `llama-cpp-2` binding can support mmproj/image
  input for Qwen3.6 through the existing `ChatRequest` / `ContentPart` contract.
  If not, keep the first runtime slice text-only and do not advertise vision
  capability.
- [x] Confirm the first implementation can reuse the existing Qwen3 formatter
  for text-only chat while tracking Qwen3.6 behavior through `ThinkingMode::Auto`.

### 9.2 — llama.cpp backend support

- [x] Extend `motlie-model-llama-cpp` with a Qwen3.6 spec that reuses the
  existing GGUF load path and generation metrics.
- [x] Add native GGUF quant selection support for Q4-class, Q5-class, Q8, and
  reserved FP8 mapping. Do not overload `QuantizationBits::Eight` to mean both
  Q8 and FP8.
- [ ] Add platform-aware default quant resolution once FP8 GGUF exists:
  macOS defaults to Q5-class; CUDA-enabled hosts default to FP8 with GPU offload;
  CPU-only Linux fallback is explicit and documented.
- [x] Reuse the existing CUDA gate `llama-cpp-cuda` and GPU-layer policy,
  including `MOTLIE_MODEL_FORCE_CPU` and `MOTLIE_MODEL_GPU_LAYERS`.
- [ ] If llama.cpp multimodal support is feasible through the Rust binding, add
  backend runtime support that consumes the existing `ContentPart::Image` /
  `ContentPart::ImageUrl` inputs rather than extending the text-only prompt
  formatter with image stubs.

### 9.3 — Curated catalog wiring

- [x] Add `libs/models/src/chat/qwen3_6_27b_gguf.rs` with `descriptor()`,
  `variant_descriptor()`, `bundle()`, and local GGUF snapshot resolution.
- [x] Add selector `qwen/qwen3_6_27b_gguf` and bundle id
  `qwen3_6_27b_gguf`.
- [x] Wire `ChatModels`, `ModelSelector`, `CuratedBundle`, `Catalog::with_defaults`,
  `bundle_from_id`, and `bundle_from_resolved` using the existing feature-gated
  chat bundle pattern.
- [x] Add `model-qwen3-6-27b-gguf = ["dep:motlie-model-llama-cpp"]` to
  `libs/models/Cargo.toml`.
- [x] Keep the bundle opt-in only unless catalog policy later approves a 27B
  model in the default build.

### 9.4 — Example and docs

- [x] Add `libs/models/examples/chat_multimodal_qwen3_6_27b/main.rs`.
- [x] Add `libs/models/examples/chat_multimodal_qwen3_6_27b/README.md`.
- [x] Add the example target to `libs/models/Cargo.toml` with
  `required-features = ["model-qwen3-6-27b-gguf"]`.
- [x] Follow the shared example conventions: `--download-artifacts`, local-only
  startup, shared `support.rs`, descriptor/capability printing, model metrics,
  and single-bundle feature expectations.
- [x] Expose `--precision=q4|q5|q8|fp8`; `fp8` fails closed until a curated FP8
  GGUF artifact exists.
- [x] Demonstrate image+text chat only if the loaded bundle advertises real
  multimodal vision capability.
- [x] Update `libs/models/examples/README.md` and
  `libs/models/docs/BUILD_MODELS.md` with the new example and CUDA invocation.

### 9.5 — Verification

- [x] Add backend unit tests for Qwen3.6 spec metadata, quant label mapping, and
  text-only capability shape.
- [x] Add `motlie-models` catalog tests for descriptor contents, selector
  parsing, disabled-feature `ModelUnavailable`, and local artifact validation.
- [x] Run `cargo test -p motlie-model-llama-cpp --lib`.
- [x] Run `cargo test -p motlie-models --no-default-features --features model-qwen3-6-27b-gguf --lib`.
- [ ] Run full default `cargo test -p motlie-models --lib`; current host fails before
  Motlie tests in `gemm-f16` because the compiler rejects `fullfp16` inline
  assembly.
- [x] Run `cargo check -p motlie-models --no-default-features --features model-qwen3-6-27b-gguf --example chat_multimodal_qwen3_6_27b`.
- [x] Run `cargo check -p motlie-models --no-default-features --features model-qwen3-6-27b-gguf,llama-cpp-cuda --example chat_multimodal_qwen3_6_27b`.
- [ ] Add an env-gated smoke path for pre-downloaded Qwen3.6 GGUF artifacts once
  artifact names and hardware requirements are validated.

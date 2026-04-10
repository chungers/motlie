# Mistral Backend — Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-07 | @codex-researcher | Initial PLAN for the first `libs/model/backends/mistral` vertical slice. Focused on a generic embedding backend path for `google/embeddinggemma-300m` that satisfies `libs/model` contracts. |
| 2026-04-07 | @codex-researcher | Marked the completed backend scaffold, runtime integration, and unit-test work after the first `mistralrs`-backed embedding slice passed compile/test verification. | Phases 1-4 |
| 2026-04-07 | @codex-researcher | Updated the backend plan after investigating NaN embeddings. The slice now validates true `LocalOnly` startup from a local snapshot, requires the sentence-transformers module stack, and adds an env-gated finite-vector test. |
| 2026-04-08 | @codex-researcher | Clarified the remaining backend TODOs after PR review. Real multi-input inference is now validated by the env-gated test; the suggested `spec.rs` / `handle.rs` file split remains an explicit cleanup follow-up rather than an implied completed task. | Phases 1, 4 |
| 2026-04-08 | @codex-researcher | Moved the embedding loader choice into `MistralEmbeddingSpec` so the backend implementation is genuinely spec-driven rather than hardcoding `EmbeddingGemma` in the generic builder path. | Phase 2, Phase 3 |
| 2026-04-08 | @codex-researcher | Tightened the backend boundary after PR 139 review. Provider-specific local artifact validation moved back into the curated bundle layer, the backend crate dropped its direct `hf-hub` dependency, and `LocalOnly` startup now consumes a resolved local model path instead of reconstructing Hugging Face cache layout internally. | Phases 1, 3 |
| 2026-04-08 | @claude | Added text generation backend (`text.rs`) for #141. `MistralTextBundle` implements `ChatModel` + `CompletionModel` via `TextModelBuilder`, with ISQ quantization mapping and `MistralTextArch::Qwen3` as the first architecture. | Phase 5 |
| 2026-04-08 | @codex-researcher | Added the multimodal backend path for Gemma 4 E2B-it (#142). `MistralMultimodalBundle` uses `MultimodalModelBuilder`, shares the existing chat contract, and keeps vision support as a capability flag rather than a separate executable trait. | Phase 6 |
| 2026-04-09 | @codex-researcher | Added the Qwen3 embedding architecture for issue #147. `MistralEmbeddingArch` now supports both EmbeddingGemma and Qwen3Embedding, with bundle-level quantization metadata validating the first `Q8`-only embedding slice. | Phase 2, Phase 4 |

Derived from [../../docs/DESIGN.md](../../docs/DESIGN.md). This PLAN covers the generic `mistral` backend implementation work.

---

## Phase 1: Backend Scaffold

Make the backend crate structurally ready to host generic `mistral.rs` implementations.

### 1.1 — Crate shape and module structure

- [x] Keep the crate limited to generic backend logic, not curated bundle metadata.
  DESIGN reference: `libs/model/docs/DESIGN.md` / `Architecture`
- [ ] Add internal module structure for the first slice, for example:
  `embeddings.rs`, `spec.rs`, `handle.rs`.
  `@codex-researcher 2026-04-08 -- Deferred intentionally. The first slice still fits comfortably in one file, and splitting it before a second backend surface lands would be churn without a behavioral benefit.`
  DESIGN reference: `Architecture`
- [x] Keep the backend crate dependent only on `libs/model` plus `mistral.rs`-related runtime dependencies.
  DESIGN reference: `Architecture`

## Phase 2: Embedding Spec and Bundle Implementation

Implement the first generic embedding path.

### 2.1 — Static backend-facing spec

- [x] Finalize `MistralEmbeddingSpec` for:
  `id`, `display_name`, `model_id`, embedding architecture, and `Capabilities`.
  DESIGN reference: `libs/model/docs/DESIGN.md` / `Bundle Contract`
- [x] Provide a built-in constructor for `embeddinggemma_300m`.
  DESIGN reference: vertical slice target agreed during planning
- [x] Provide a built-in constructor for `qwen3_embedding_06b`.
  DESIGN reference: second embedding slice (#147)

### 2.2 — Generic embedding bundle implementation

- [x] Finalize `MistralEmbeddingBundle` as a generic `ModelBundle` implementation.
  DESIGN reference: `libs/model/docs/DESIGN.md` / `Bundle Contract`
- [x] Finalize `MistralEmbeddingHandle` as an embedding-only `BundleHandle`.
  DESIGN reference: `Loaded Handle`
- [x] Ensure unsupported capabilities return `UnsupportedCapability(CapabilityKind::...)`.
  DESIGN reference: `Lifecycle Rules`

## Phase 3: Replace Placeholder Embedder with Real `mistral.rs`

The current placeholder embedder is only acceptable for contract validation. The real slice must exercise `mistral.rs`.

### 3.1 — Runtime integration

- [x] Add the `mistral.rs` dependency set required for embeddings.
- [x] Identify the concrete embedding load path for `google/embeddinggemma-300m`.
- [x] Implement backend initialization from `StartOptions`.
- [x] Implement actual `EmbeddingModel::embed()` using the `mistral.rs` embedding API.
- [x] Normalize backend errors into `ModelError`.

### 3.2 — Operational behavior

- [x] Define how the backend consumes local model artifacts for the first slice.
- [x] Keep provider-specific cache-layout resolution and validation out of the generic backend crate.
- [x] Define what happens when required weights are unavailable.
- [x] Define whether the first slice assumes local sidecar assets only, or permits a preconfigured model path.

## Phase 4: Tests and Verification

### 4.1 — Unit tests

- [x] Add unit tests for spec construction and unsupported-capability behavior.
- [x] Add tests for deterministic handle metadata and capability exposure.

### 4.2 — Integration validation

- [x] Add at least one integration-style test or env-gated test for real `mistral.rs` embedding inference.
- [x] Validate multi-input embedding requests.
- [x] Validate that `embeddinggemma_300m` can be instantiated through `libs/models::Catalog`.

### 4.3 — Required verification commands

- [x] `cargo check -p motlie-model-mistral`
- [x] `cargo test -p motlie-model-mistral`
- [x] `cargo check -p motlie-model -p motlie-model-mistral -p motlie-models`

## Phase 5: Text Generation Backend

Add the generic text generation path alongside the embedding backend.

### 5.1 — Text architecture and spec

- [x] Add `MistralTextArch` enum with `Qwen3` as the first variant.
  Maps to `NormalLoaderType::Qwen3`.
- [x] Add `MistralTextSpec` with `id`, `display_name`, `model_id`, `arch`, `capabilities`.
- [x] Provide `MistralTextSpec::qwen3_4b()` built-in constructor.

### 5.2 — Text bundle and handle

- [x] Add `MistralTextBundle` implementing `ModelBundle`.
- [x] `start()` uses `TextModelBuilder` with arch-driven `NormalLoaderType`.
- [x] Map `QuantizationBits` → `mistralrs::IsqBits` via `with_auto_isq()`.
- [x] `MistralTextHandle` implements `BundleHandle` exposing `ChatModel` + `CompletionModel`.
- [x] `embeddings()` returns `UnsupportedCapability`.
- [x] `CompletionModel::complete()` delegates to single-turn chat.
- [x] `TextRuntime` internal trait enables stub testing.

### 5.3 — Tests

- [x] Spec identity, capability exposure, quantization mapping.
- [x] Artifact policy, unpack_root rejection.
- [ ] Env-gated integration test with pre-downloaded Qwen3-4B artifacts.

## Phase 6: Multimodal Chat Backend

### 6.1 — Multimodal architecture and spec

- [x] Add `MistralMultimodalArch` with `Gemma4` as the first variant.
- [x] Add `MistralMultimodalSpec` with `id`, `display_name`, `model_id`, `arch`, `capabilities`.
- [x] Provide `MistralMultimodalSpec::gemma4_e2b()` built-in constructor.

### 6.2 — Multimodal bundle and handle

- [x] Add `MistralMultimodalBundle` implementing `ModelBundle`.
- [x] Use `MultimodalModelBuilder` with `MultimodalLoaderType::Gemma4`.
- [x] Reuse the existing `ChatModel` trait for both text-only and image+text requests.
- [x] Keep `completion()` and `embeddings()` unsupported for the first multimodal slice.
- [x] Reject unsupported local-only inputs such as `ImageUrl` explicitly with typed `ModelError`.

### 6.3 — Tests

- [x] Spec identity and capability exposure for Gemma 4.
- [x] Handle capability exposure and unsupported-capability behavior.
- [x] Input validation test for unsupported image-url content parts.
- [ ] Env-gated integration test with pre-downloaded Gemma 4 E2B-it artifacts.

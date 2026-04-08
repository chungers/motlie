# Curated Model Bundle Library — Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-07 | @codex-researcher | Initial PLAN for `libs/models` vertical slice work. Covers the curated catalog, constructor registration, and the first `embeddinggemma_300m` bundle wired through the Mistral backend. |

Derived from [DESIGN.md](./DESIGN.md). This PLAN focuses on the first curated bundle slice rather than the full long-term catalog.

---

## Phase 1: Catalog and Descriptor Model

Make the curated catalog concrete enough to support both listing and instantiation.

### 1.1 — Descriptor model

- [ ] Finalize `BundleDescriptor` to include:
  `id`, `display_name`, `family`, `support_tier`, `capabilities`, `packaging`, `backend`, `requirements`, `eval_tracks`.
  DESIGN reference: `Bundle Catalog Model`
- [ ] Finalize `BundleFamily`, `SupportTier`, `PackagingMode`, `BackendKind`, `PlatformConstraint`, `BuildConstraint`, and `BundleRequirements`.
  DESIGN reference: `Bundle Catalog Model`, `Packaging and Deployment Model`
- [ ] Add unit tests for descriptor equality, evaluation-track filtering, and capability-descriptor projection.
  DESIGN reference: `Testing Scope for PLAN`

### 1.2 — Catalog behavior

- [ ] Finalize `Catalog` as an in-memory registry of descriptors plus constructors.
  DESIGN reference: `Architecture`, `API Sketch`
- [ ] Finalize `register`, `bundle`, `bundles`, `bundles_for_track`, and `instantiate`.
  DESIGN reference: `Architecture`
- [ ] Add `Catalog::with_defaults()` for the first curated slice.
  DESIGN reference: `API Sketch`
- [ ] Add unit tests for registration overwrite semantics and constructor-based instantiation.
  DESIGN reference: `Testing Scope for PLAN`

## Phase 2: First Curated Bundle

Define the first curated embedding stack explicitly in this crate.

### 2.1 — `embeddinggemma_300m` descriptor

- [ ] Finalize `embeddinggemma_300m_descriptor()`.
  DESIGN reference: `Bundle Catalog Model`
- [ ] Ensure the descriptor declares:
  `BundleFamily::Embeddings`,
  `BackendKind::MistralRs`,
  `PackagingMode::Sidecar`,
  `EvalTrack::Embeddings`,
  and `CapabilityDescriptor::embeddings()`.
  DESIGN reference: `Capability Exposure`, `Evaluation and Curation Harnesses`
- [ ] Add tests for descriptor contents so the first curated stack is reviewable as data.

### 2.2 — Curated constructor binding

- [ ] Finalize `embeddinggemma_300m_bundle()` so it binds the descriptor to `libs/model/backends/mistral`.
  DESIGN reference: `Internal Backend Boundary`
- [ ] Keep bundle-specific artifact/build constraints in `libs/models`, not in the backend crate.
  DESIGN reference: `Internal Backend Boundary`, `Bundle-Local Build Customization`
- [ ] Add tests that `Catalog::with_defaults()` includes `embeddinggemma_300m` and can instantiate it.

## Phase 3: Packaging and Build Constraints for the First Slice

Keep this limited and explicit for the first embedder.

### 3.1 — First-slice constraints

- [ ] Decide and document the first-slice artifact assumption:
  local sidecar assets only, or configurable artifact path.
  DESIGN reference: `Packaging and Deployment Model`
- [ ] Keep the first build constraints minimal and inspectable.
  DESIGN reference: `Build Reasoning and Profile Clarity`
- [ ] Ensure the first slice does not overcommit to DGX/Debian/macOS profile complexity before the utility work lands.
  DESIGN reference: `Build Reasoning and Profile Clarity`

## Phase 4: Vertical Slice Validation

### 4.1 — Cross-crate end-to-end check

- [ ] Verify the following flow compiles and runs:
  `Catalog::with_defaults()` -> `instantiate("embeddinggemma_300m")` -> `start()` -> `embeddings()` -> `embed()`
  DESIGN reference: `API Sketch`
- [ ] Add one small test in `libs/models` covering that path against the backend crate’s current implementation.

### 4.2 — Required verification commands

- [ ] `cargo check -p motlie-models`
- [ ] `cargo test -p motlie-models`
- [ ] `cargo check -p motlie-model -p motlie-model-mistral -p motlie-models`


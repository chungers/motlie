# Curated Model Bundle Library Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-07 | @codex-researcher: Initial greenfield design for `libs/models` as the curated bundle catalog and composition crate over `libs/model`. Migration and backward compatibility are explicitly out of scope for this first cut. | All |
| 2026-04-07 | @codex-researcher: Clarified that curated local model download is an explicit `libs/models` control, separate from backend cache-miss behavior. | Packaging and Deployment Model, Release Assembly Utility |
| 2026-04-08 | @codex-researcher: Clarified that `libs/models` public fallible APIs should use typed library errors rather than `anyhow`, and specified the first vertical-slice artifact contract for `embeddinggemma_300m`. | Solution, Bundle Catalog Model, Packaging and Deployment Model |
| 2026-04-08 | @codex-researcher: Locked down the crate hierarchy and direct bundle namespace so future work does not drift. Also removed over-modeled public metadata from the recommended surface and documented the preferred direct bundle API next to `Catalog`. | Architecture, Bundle Catalog Model, API Sketch |
| 2026-04-08 | @codex-researcher: Added the bundle-build convention: curated bundles are feature-gated individually, the direct enum and `ModelSelector` only expose compiled-in bundles, and known-but-disabled bundles should report `ModelUnavailable`. | Architecture, Bundle Catalog Model, Packaging and Deployment Model |
| 2026-04-08 | @codex-researcher: Clarified the reviewed backend/artifact boundary. Provider-specific cache layout resolution and local artifact validation belong in curated bundle modules under `libs/models`, while generic backends under `libs/model/backends/*` consume only resolved local model paths or fetch-enabled cache roots. | Architecture, Packaging and Deployment Model, Backend Composition |

This document defines the design for `libs/models`, the curated bundle library that exposes opinionated model stacks as deployable product modules. A bundle in this crate includes vetted weights when applicable, a chosen backend or transport, packaging policy when applicable, capability wiring, and consistent lifecycle behavior through the contracts defined in `libs/model`.

The design goal is not to provide generic access to arbitrary checkpoints and runtimes. The goal is to expose stable integration modules that behave like composable product building blocks.

The governing principle is curated sustainability: Motlie should be able to admit, evaluate, promote, and retire bundles as the model market changes while preserving a stable integration surface for the rest of the ecosystem.

## Table of Contents

- [Overview](#overview)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Architecture](#architecture)
- [Framework Principles](#framework-principles)
- [Bundle Catalog Model](#bundle-catalog-model)
- [Packaging and Deployment Model](#packaging-and-deployment-model)
- [Backend Composition](#backend-composition)
- [Evaluation and Curation Harnesses](#evaluation-and-curation-harnesses)
- [API Sketch](#api-sketch)
- [Alternatives Considered](#alternatives-considered)
- [Testing Scope for PLAN](#testing-scope-for-plan)
- [Open Concerns](#open-concerns)

---

## Overview

### Problem Statement

Different model families and checkpoints are not equally well supported across execution substrates. Some run best on `mistral.rs`, others on `llama.cpp`, others on `ONNX Runtime`, some are packaging- or platform-specific, and some may be served remotely over HTTP. Exposing this matrix directly to callers would make integration brittle and operationally expensive.

Motlie instead wants a curated system where each supported model bundle is a vetted unit:

- a specific checkpoint family or variant
- a specific backend selection
- a known artifact packaging or transport strategy
- a known capability surface
- a stable lifecycle and interaction contract

### Solution

`libs/models` owns:

1. the bundle catalog
2. bundle registration and discovery
3. curated binding from bundle to backend implementation
4. artifact packaging and/or transport policy
5. bundle-local build and distribution customization
6. concrete implementations of `libs/model` contracts
7. typed catalog and artifact errors for operator-facing bundle control

Callers integrate with curated bundle IDs and capability APIs. They do not choose runtimes directly.

Library-level fallible APIs in this crate should return typed errors defined with `thiserror`. CLI tools and examples layered on top of the crate may use `anyhow` for propagation and user-facing context.

## Goals and Non-Goals

### Goals

- Expose supported model bundles as stable product modules
- Preserve a common abstraction and contract across all bundles
- Hide runtime/backend selection behind bundle implementations
- Support capability-specific access without runtime-specific code in applications
- Keep bundles composable so they can be selected and substituted by catalog-driven code
- Provide a sustainable framework for admitting and evolving curated bundles as the market changes
- Support evaluation harnesses used to compare and validate bundles during curation
- Support deployment modes where weights are packaged with the binary or shipped as tightly controlled artifacts
- Allow per-bundle build customization for backend features, platform gates, CUDA enablement, and distribution constraints
- Make build feature flags and platform gating easy to reason about for distro-specific outputs such as DGX, Debian, or macOS bundles
- Allow gradual expansion of the curated bundle catalog
- Keep future room for protected artifact formats such as encrypted embedded sections

### Non-Goals

- Generic user-supplied checkpoint loading in the first cut
- Public backend/plugin APIs for third parties in the first cut
- Treating all backends as interchangeable at runtime
- Cross-bundle behavioral equivalence beyond the shared contract surface
- Migration from any prior Motlie model library, since this is greenfield product work

## Architecture

`libs/models` sits above `libs/model` and below application code.

Primary subsystems:

1. `lib.rs`
   Shared catalog, descriptor, artifact-control, and crate-level infrastructure only
2. capability-family namespaces such as `embeddings/`
   One directory per major capability family when more than one curated bundle is expected
3. bundle module files such as `embeddings/google_gemma_300m.rs`
   One curated bundle per file, with direct `descriptor()` and `bundle()` functions
4. `src/bin/`
   Small operational entrypoints such as artifact download helpers

Concrete hierarchy rule:

```text
libs/models/
  src/
    lib.rs
    embeddings/
      mod.rs
      google_gemma_300m.rs
    bin/
      download_artifacts.rs
```

Public namespace rule:

```rust
motlie_models::Catalog
motlie_models::embeddings::google_gemma_300m::descriptor()
motlie_models::embeddings::google_gemma_300m::bundle()
```

This split is intentional:

- `lib.rs` owns shared crate infrastructure
- capability-family directories keep related bundles grouped together
- each bundle file maps 1:1 to a curated bundle
- callers may use either direct bundle modules or the dynamic `Catalog`
- bundle modules may be compiled conditionally behind per-bundle features

### High-Level Data Flow

1. Caller selects `BundleId`
2. Catalog resolves a bundle descriptor and constructor
3. Loader resolves artifact or transport requirements
4. Bundle constructor binds the bundle to its selected backend implementation under `libs/model/backends/*`
5. The selected backend creates a loaded runtime handle
6. The loaded handle exposes capability adapters through `libs/model` traits

This keeps runtime-specific and packaging-specific logic behind the bundle constructor path while preserving a common contract for callers.

## Framework Principles

### Curated Sustainability

`libs/models` is the operational layer that lets Motlie track a changing model ecosystem without turning application code into a moving target.

Implications:

- the catalog should focus on specific Motlie use cases, not generic model availability
- admitting a new bundle should be a curation workflow, not merely a code import
- retiring a weak or stale bundle should be operationally straightforward
- backend churn should be isolated inside bundle implementations and release tooling

### Evaluation-Driven Curation

Bundle curation requires repeatable assessment of model quality. `libs/models` therefore must contribute to, and depend on, evaluation harnesses that can exercise curated bundles consistently.

Implications:

- bundle descriptors should expose enough metadata for harness selection
- bundle capability implementations should be runnable by shared harness drivers
- curation decisions should be informed by structured evaluation runs, not only ad hoc manual testing

## Bundle Catalog Model

The catalog is the product-facing index of supported bundles.

Recommended public concepts:

- `BundleId`
- `BundleDescriptor`
- `Catalog`
- `BundleFamily`
- direct bundle namespaces such as `motlie_models::embeddings::google_gemma_300m`

`BundleDescriptor` should describe:

- stable ID
- display name
- family
- supported capabilities
- selected backend
- artifact or transport requirements
- optional platform constraints
- optional build/distribution constraints
- optional evaluation tags or supported harness groups
- artifact requirements precise enough for deterministic local-only startup where applicable

`SupportTier` and `PackagingMode` are not required as first-class public catalog concepts in v1. If those ideas become operationally useful later, they can be added back with a clearer consumer and a narrower meaning.

### Bundle Build Convention

Curated bundles are not all expected to be linked into every build of `libs/models`.

The intended convention is:

- one Cargo feature per curated bundle
- direct bundle module, enum variant, parser support, and catalog registration are all gated together
- the default feature set may include a small recommended slice, but distro/profile builds should compose the exact curated set explicitly

For example:

```toml
[features]
default = ["model-google-gemma-300m"]

model-google-gemma-300m = []
model-qwen3-embed-600m = []
profile-macos = ["model-google-gemma-300m"]
profile-dgx = ["model-google-gemma-300m", "model-qwen3-embed-600m"]
```

This means:

- `EmbeddingModels::GoogleGemma300m` only exists when `model-google-gemma-300m` is enabled
- `Catalog::with_defaults()` only registers bundles that are actually compiled in
- string parsing through `ModelSelector` should distinguish:
  - unknown selector
  - known selector that is unavailable in the current build

Recommended typed error for the second case:

- `ModelUnavailable { selector: String }`

Example internal organization:

```text
libs/models/
  src/
    lib.rs
    embeddings/
      mod.rs
      google_gemma_300m.rs
    qwen/
      mod.rs
      qwen3_5_instruct.rs
    hermes/
      mod.rs
      hermes3_chat.rs
    bin/
      download_artifacts.rs
```

### Bundle Family versus Bundle Identity

The public selection unit is still `BundleId`, not `BundleFamily`.

Examples:

- `Qwen3_5Instruct`
- `Hermes3LlamaChat`
- `BgeSmallEn`

`BundleFamily` is metadata for organization, docs, and feature flags, not the primary runtime lookup key.

### Direct Bundle Access

`Catalog` remains useful for configuration-driven or dynamically selected bundle loading. It is not the only intended entrypoint.

For application code that wants a specific curated bundle directly, the preferred API is the bundle module path itself:

```rust
let descriptor = motlie_models::embeddings::google_gemma_300m::descriptor();
let bundle = motlie_models::embeddings::google_gemma_300m::bundle();
```

This avoids unnecessary catalog lookup in the common case while preserving catalog-driven selection for higher-level orchestration.

When the underlying bundle is an embedding bundle, the concrete bundle type should also implement the `motlie_model::Embedding` trait so callers can access embedding-specific metadata without loading the runtime handle first.

### First Vertical Slice Specificity

The first concrete local bundle, `embeddinggemma_300m`, is intentionally part of the framework specification because it validates the layering. Its curated artifact contract must include:

- root model config and tokenizer files
- `modules.json`
- sentence-transformers module configs for pooling and dense stages
- dense module safetensors files

That requirement is part of the bundle definition, not an implementation detail left to backend cache discovery.

### Capability Exposure

A bundle may support one or more capabilities:

- chat
- text completion
- embeddings

Capability checks happen at descriptor time and again at loaded-handle access time. This avoids late surprises and keeps unsupported-capability errors precise.

The common contract remains the primary integration surface even when individual bundles have very different build or runtime requirements.

Catalog metadata should also make it easy to identify which bundles participate in which evaluation tracks.

## Packaging and Deployment Model

The crate assumes a curated artifact supply chain. Bundle weights are sourced from vetted providers and packaged in controlled forms rather than being fetched ad hoc at runtime.

Supported design directions:

1. Sidecar artifacts packaged with the application bundle
2. Embedded artifacts linked into the binary for small models
3. Remote-served bundles over controlled HTTP integrations
4. Future protected artifact formats, including encrypted embedded data or custom bundle containers

The public crate does not need a first-class `PackagingMode` enum for v1. Artifact policy and startup behavior are the important public concerns; exact release packaging remains an internal concern of release tooling and curated bundle implementation.

For local curated bundles, model download should be an explicit control owned by `libs/models` rather than an implicit side effect of backend startup. A backend such as `mistral.rs` may still populate or consult its own cache layout, but curated provenance and download policy remain the responsibility of the curated bundle layer.

As part of the roadmap, Motlie should provide a utility that controls:

- which bundles are included
- which features are enabled
- which target operating system or distribution profile is selected
- whether release output is a large binary with embedded weights or a package bundle with deterministic extraction

The deployment goal is for either mode to feel operationally close to a single-binary deployment.

### Packaging Rules

- Artifact provenance must be explicit in bundle descriptors or manifests
- Curated local model download must be invokable explicitly, independent of backend cache-miss behavior
- Bundle loading must validate artifact integrity before runtime initialization when local artifacts are used
- Unpack-on-start must use deterministic cache paths and cleanup rules
- Backend-specific file layout transformations must remain internal to the bundle
- Remote-served bundles must validate endpoint configuration and credentials before capability use

### Bundle-Local Build Customization

Each bundle directory may carry build customization that reflects the reality of that bundle's execution substrate.

Examples:

- enable CUDA support only for selected bundles or targets
- gate a bundle by operating system, CPU architecture, or libc/distribution requirement
- opt into backend-specific compile flags or linked native libraries
- reject unsupported packaging modes at build time rather than at runtime
- expose explicit distro or build-profile selections such as DGX, Debian, or macOS

The important boundary is that build customization stays bundle-local or backend-local. It must not leak upward into the stable `libs/model` contract.

### Build Reasoning and Profile Clarity

Build gating must be easy for operators and release tooling to reason about. The system should favor explicit, inspectable configuration over scattered conditional logic.

Requirements:

- feature flags should map to understandable product or backend choices rather than opaque internal toggles
- platform and distribution gates should be declared close to the affected bundle
- profile selection should be explicit for targets such as DGX, Debian, and macOS
- unsupported combinations should fail clearly at build time
- documentation and descriptor metadata should make the active constraints visible

The design should support custom build profiles that combine platform, accelerator, and bundle-family choices into repeatable outputs. For example:

- a DGX-oriented profile may enable CUDA-capable bundles and exclude CPU-only fallback bundles
- a Debian distribution profile may select portable CPU-compatible bundles and approved sidecar packaging
- a macOS profile may exclude Linux-only backends and expose only supported local or remote-backed bundles

## Evaluation and Curation Harnesses

The roadmap should include shared harness runners that operate on curated bundles using canned inputs, expected outputs, and scoring rules.

Representative evaluation tracks include:

- embedding generation
- reasoning-oriented tasks
- summarization
- classification
- chat/instruction quality where relevant

The goal is not only test coverage. The goal is to make model curation sustainable by making bundle quality easier to compare across revisions and across candidate bundles.

### Harness Design Direction

The harness runner should:

- discover eligible bundles from catalog metadata and capabilities
- run canned evaluations against the common capability contracts
- emit structured results that support comparison, regression detection, and promotion decisions
- support bundle-specific setup where necessary without breaking the common runner model

This means `libs/models` should eventually provide enough descriptor metadata and helper surfaces for harness tooling to target bundles predictably.

## Release Assembly Utility

The roadmap should include a dedicated utility for producing curated release artifacts from the bundle catalog.

Responsibilities:

- select model bundles for inclusion
- select build features and backend options
- select target OS, distribution, and accelerator profile
- prefetch curated artifacts into deterministic artifact roots when local bundles are selected
- validate that the requested bundle set is compatible with the chosen target
- assemble either embedded-binary releases or package-bundle releases
- emit deterministic manifests so extraction and startup behavior are predictable and auditable

### Release Modes

1. Embedded image mode
   Large release binaries may include model weights directly in the produced image when the selected bundles and target constraints allow it.
2. Package bundle mode
   Release output may ship as an application binary plus a tightly controlled package bundle whose extraction path and unpack behavior are deterministic.

Both modes should preserve the product experience that deployment feels like "one thing to ship," even if the package-bundle mode expands into controlled local files on first start.

### Deterministic Extraction

When package bundles are used, extraction behavior should be deterministic across runs and environments.

Requirements:

- stable extraction paths derived from bundle identity, version, and integrity metadata
- explicit cleanup and replacement rules
- no ad hoc runtime downloads for curated local bundles
- predictable startup failure when required artifacts are missing or invalid

This keeps package-bundle deployment operationally close to a single-binary release while avoiding opaque filesystem side effects.

### Deployment Notes

The design supports small bundles embedded directly in the executable, including future custom-ELF strategies, but `libs/models` should not hard-code binary-format assumptions into the public API. The right abstraction is "embedded artifact" or "packaged artifact," not "ELF section."

Remote-served bundles should be described as transport-backed bundles rather than as a separate product category. For example, a `GPT4_5` bundle may satisfy the same contracts through HTTP while still appearing in the catalog as just another curated bundle.

## Backend Composition

Backend selection is opinionated per bundle. The bundle chooses the backend; the caller does not.

Examples:

- a Qwen bundle may use `mistral.rs`
- a Hermes bundle may use `llama.cpp`
- an embedding bundle may use `ONNX Runtime`
- a GPT-4.5 bundle may use HTTP

This selection is part of the bundle definition because backend maturity, correctness, performance, and packaging constraints differ across bundles.

Backend composition also includes build-time composition. A bundle may depend on different backend features, native toolchains, or platform rules than another bundle in the same catalog.

### Internal Backend Boundary

Generic backend implementations should live under `libs/model/backends/*`, not inside `libs/models`. This keeps reusable runtime machinery close to the contract layer while keeping `libs/models` focused on curated bundle stacks, artifacts, and bundle-specific adaptations.

`libs/models` may expose backend choice in descriptor metadata for observability, but applications should not instantiate backend adapters directly.

Recommended internal trait:

```rust
trait BackendFactory {
    async fn start_bundle(
        &self,
        descriptor: &BundleDescriptor,
        artifacts: &ResolvedArtifacts,
        options: &StartOptions,
    ) -> Result<Box<dyn BundleHandle>, ModelError>;
}
```

This trait is not required to be public in v1. The important boundary is structural:

- `libs/model/backends/mistral` owns generic `mistral.rs` contract implementations
- `libs/model/backends/ort` owns generic ORT contract implementations
- `libs/models` owns which curated bundle uses which backend implementation and with what artifacts or bundle-specific glue

## API Sketch

The application experience should feel like selecting and loading product modules.

```rust
use motlie_model::{BundleId, EmbeddingRequest};
use motlie_models::Catalog;

let catalog = Catalog::with_defaults();
let bundle = catalog
    .instantiate(&BundleId::new("embeddinggemma_300m"))?;
let handle = bundle.start(Default::default()).await?;
let embeddings = handle.embeddings()?;
let response = embeddings.embed(EmbeddingRequest {
    inputs: vec![
    "curated bundle catalog",
    "mistral embedding vertical slice",
    ],
}).await?;
```

### Feature-Gating Direction

The crate should support Cargo features for optional bundle families or heavyweight backend integrations, for example:

- `qwen`
- `hermes`
- `embeddings`
- `backend-mistral`
- `backend-llamacpp`
- `backend-ort`

Bundle-family features are the primary user-facing toggles. Backend features are mostly implementation plumbing and should be kept as implicit as practical.

Where a bundle has materially different build requirements, the bundle itself may also need feature flags or build-script enforcement. The catalog should make these constraints visible in descriptor metadata and documentation.

The same rule applies to build profiles: they should be comprehensible from the bundle and catalog metadata rather than inferred from hidden `cfg` interactions.

## Alternatives Considered

### Alternative A: Treat runtime/backend as the primary public abstraction

Pros:

- explicit about the true execution substrate
- may simplify low-level experimentation

Cons:

- wrong abstraction for application developers
- forces callers to understand backend compatibility
- does not match the curated bundle product model

Decision:

Rejected.

### Alternative B: Curated bundle catalog in `libs/models`, contracts in `libs/model`

Pros:

- matches the agreed architecture cleanly
- separates stable API from backend-heavy implementation
- supports feature-gated bundle families and deployment-specific packaging

Cons:

- requires one more crate and one more design boundary
- demands discipline around descriptor versus implementation concerns

Decision:

Accepted.

### Alternative C: One crate per concrete bundle from day one

Pros:

- maximum modularity
- very explicit ownership and dependency control

Cons:

- high workspace overhead too early
- repeated boilerplate for closely related bundles
- catalog ergonomics become fragmented before the abstraction is stable

Decision:

Rejected for the first cut. Keep concrete bundles inside `libs/models` until scale justifies further splitting.

### Alternative D: Separate local-bundle catalog and remote-provider catalog

Pros:

- clearer separation between packaged assets and hosted APIs
- different operational concerns can evolve independently

Cons:

- splits one user-facing concept into two parallel product systems
- weakens the "models are lego blocks" abstraction
- duplicates capability and lifecycle wiring

Decision:

Rejected. Remote-served models should still appear as curated bundles inside the same catalog.

### Alternative E: Centralize all build and distribution policy at the crate root

Pros:

- one place to inspect build logic
- fewer per-bundle build entry points

Cons:

- scales poorly as bundles diverge
- makes platform or CUDA exceptions harder to isolate
- makes distro-specific outputs harder to audit
- weakens the "bundle as product module" boundary

Decision:

Rejected. Bundle-local build customization is a feature, not a smell, as long as the top-level contract stays uniform.

### Alternative F: Require all deployments to use sidecar package bundles only

Pros:

- simpler build pipeline for very large models
- avoids oversized executable images

Cons:

- removes the option for true embedded single-artifact releases where they are practical
- weakens deployment ergonomics for smaller curated bundles
- does not fit the roadmap requirement for flexible release assembly

Decision:

Rejected. The system should support both embedded-image and deterministic package-bundle release modes.

## Testing Scope for PLAN

PLAN must specify concrete tests for:

- catalog registration and lookup
- descriptor correctness and capability advertisement
- bundle load and shutdown behavior
- backend binding per bundle
- artifact integrity and unpack flows
- embedded-artifact and sidecar-artifact startup paths
- remote transport startup and credential/config validation paths
- bundle-local build gating and platform restriction behavior
- backend feature selection such as CUDA-enabled versus CPU-only bundle builds
- distro/profile-specific build outputs for targets such as DGX, Debian, and macOS
- release-assembly utility behavior for embedded-image versus package-bundle outputs
- deterministic extraction paths, cleanup rules, and startup validation for package bundles
- shared harness-runner behavior across evaluation categories such as embeddings, reasoning, summarization, and classification
- catalog metadata used to select bundles for evaluation tracks
- unsupported-capability behavior for embedding-only or chat-only bundles
- feature-gated build surfaces

If protected or encrypted bundle packaging enters scope, PLAN must add explicit verification for key handling, integrity checks, and failure behavior.

## Open Concerns

- Whether bundle IDs should include provider or packaging suffixes when multiple vetted variants exist
- Whether `Catalog` should be fully static in v1 or support external registration hooks
- How much backend choice should be exposed in descriptor metadata for observability and support diagnostics
- Whether bundle manifests should live in Rust code, generated tables, or external metadata files

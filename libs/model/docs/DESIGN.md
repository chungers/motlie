# Model Contract Library Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-07 | @codex-researcher: Initial greenfield design for `libs/model` as the stable contract/lifecycle crate for packaged model bundles. Migration and backward compatibility are explicitly out of scope for this first cut. | All |
| 2026-04-07 | @codex-researcher: Clarified that curated artifact download is explicit in `libs/models`, while backends consume artifact roots through `StartOptions` and artifact contracts. | Overview, Architecture, Artifact and Packaging Contracts |
| 2026-04-07 | @codex-researcher: Added `ArtifactPolicy` to the startup contract so regulated local-only deployments can fail closed while permissive deployments may still allow runtime fetch. | Core Abstractions, Lifecycle, Artifact and Packaging Contracts |
| 2026-04-08 | @codex-researcher: Locked down the error-handling rule for the framework: library crates expose typed `thiserror` errors, while binaries/examples may use `anyhow` for propagation and CLI context. Also clarified that non-test runtime code should not panic. | Solution, Framework Principles, Core Abstractions |
| 2026-04-08 | @codex-researcher: Tightened the design to match the current implemented contract exactly: removed unstaged future type names from the core abstraction list, corrected trait signatures, and clarified that richer artifact manifests currently live in `libs/models` while `libs/model` only standardizes startup artifact policy. | Core Abstractions, Capability Surfaces, Artifact and Packaging Contracts |
| 2026-04-08 | @codex-researcher: Added the proposed bundle-level embedding metadata contract so curated embedding bundles can expose retrieval-relevant semantics such as preferred distance metric and normalization. This shape is intended to align cleanly with future `libs/db/vector::EmbeddingSpec` integration without creating a reverse dependency today. | Core Abstractions, Capability Surfaces, API Sketch |
| 2026-04-08 | @codex-researcher: Referenced the `libs/models` bundle-build convention so the contract layer is explicit that bundle availability is build-dependent and selector/catalog surfaces may vary by curated feature set. | Architecture, Framework Principles |

This document defines the design for `libs/model`, the contract crate for Motlie's packaged model system. The crate does not ship concrete model bundles or runtime implementations. Instead, it defines the stable public vocabulary, lifecycle, request/response types, capability adapters, composability boundaries, and artifact contracts that higher-level crates build on.

The target product shape is not "generic inference over arbitrary checkpoints." The target shape is a system of opinionated model bundles where a bundle includes vetted weights, a selected backend or transport, packaging metadata when applicable, and a stable capability surface. `libs/model` provides the uniform interface that those bundles implement.

The governing framework principle is curated sustainability: Motlie maintains opinionated bundles for concrete ecosystem use cases, and the contract exists to let those bundles evolve as the model market changes without forcing downstream code to absorb constant churn.

## Table of Contents

- [Overview](#overview)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Architecture](#architecture)
- [Framework Principles](#framework-principles)
- [Core Abstractions](#core-abstractions)
- [Lifecycle](#lifecycle)
- [Capability Surfaces](#capability-surfaces)
- [Evaluation Harness Support](#evaluation-harness-support)
- [API Sketch](#api-sketch)
- [Artifact and Packaging Contracts](#artifact-and-packaging-contracts)
- [Alternatives Considered](#alternatives-considered)
- [Testing Scope for PLAN](#testing-scope-for-plan)
- [Open Concerns](#open-concerns)

---

## Overview

### Problem Statement

Motlie wants to treat models as deployable product blocks rather than as loose combinations of checkpoint files and inference runtimes. Different model families have different runtime maturity, artifact formats, and operational constraints. The application integration surface should not expose those differences directly.

Some bundles may execute locally with packaged weights and a native backend, while others may be remote-backed bundles that satisfy the same contract over HTTP. The top-level lifecycle and capability API should remain stable across both cases.

For local bundles, `libs/model` does not own download policy. The contract only needs to carry enough artifact and startup information for higher-level curated tooling to stage artifacts explicitly and for backend implementations to consume the resulting artifact root predictably.

Without a dedicated contract crate, concrete bundle implementations would either:

1. expose backend-specific APIs and leak runtime details into callers, or
2. invent local abstractions independently, leading to incompatible lifecycle and request/response semantics across bundles.

The core product value is not only hiding backends. It is preserving a common abstraction so bundles compose cleanly inside higher-level Motlie systems regardless of whether they are local, embedded, or remote-backed.

### Solution

`libs/model` defines the shared contract for all packaged model bundles:

1. Stable bundle identity and metadata
2. Uniform load/start/stop lifecycle
3. Shared request/response types for chat, text completion, and embeddings
4. Capability discovery, introspection, and capability-specific adapters
5. Composable contracts so bundles can be selected, loaded, and invoked uniformly
6. Artifact descriptors for packaged weights and auxiliary assets
7. Common errors and unsupported-capability behavior
8. Startup artifact policy that backends must honor
9. Typed library errors suitable for downstream matching and policy handling

The crate is intentionally small, dependency-light, and backend-agnostic.

## Goals and Non-Goals

### Goals

- Define a stable contract for packaged model bundles
- Make common abstraction and contract consistency the primary integration surface
- Keep runtime/backend details out of the primary public API
- Support multiple capability classes without forcing every bundle to implement every API
- Preserve composability so application code can swap bundles without changing lifecycle logic
- Support sustainable curated-bundle evolution as models enter or leave the Motlie catalog
- Support evaluation harnesses used to assess model quality as part of the curation process
- Support simple introspection of capability input/output/interaction shape for catalogs and harnesses
- Support static or embedded deployment artifacts, including future encrypted bundle formats
- Allow higher-level crates to register and load curated bundles consistently
- Provide ergonomic Rust APIs suitable for application integration

### Non-Goals

- Shipping actual model bundles
- Selecting or orchestrating inference backends directly
- Supporting arbitrary user-supplied checkpoints in the first cut
- Modeling training, fine-tuning, or evaluation workflows
- Migration or backward-compatibility strategy for prior Motlie APIs

## Architecture

`libs/model` is the contract layer in a three-layer system:

1. `libs/model`
   Public contracts, lifecycle, metadata, capability types, lightweight `model::eval` abstractions, artifact descriptors, and errors
2. `libs/model/backends/*`
   Generic backend implementations such as `mistral`, `ort`, `llamacpp`, or later HTTP-backed providers that satisfy the `libs/model` contracts
3. `libs/model-eval`
   Substantial evaluation tooling such as harness runners, suite loading, scoring, reporting, and CLI-oriented support built on `model::eval`
4. `libs/models`
   Curated bundle catalog, bundle registration, bundle artifacts, packaging policy, and bundle-specific adaptations layered on top of `libs/model`

Dependency direction:

- applications depend on `libs/models` and may also use `libs/model` types directly
- `libs/model/backends/*` depend on `libs/model`
- `libs/model-eval` depends on `libs/model` and may optionally integrate with `libs/models` for catalog-driven runs
- `libs/models` depends on `libs/model` and the relevant `libs/model/backends/*` implementations
- `libs/model` does not depend on bundle catalogs, harness tooling, or backend implementations

Build note:

- `libs/model` defines stable contracts
- `libs/models` may expose only a subset of curated bundles in a given build, based on per-bundle feature flags and higher-level profile features
- callers should therefore treat curated selector/catalog surfaces as build-dependent, while still relying on the stable trait contracts defined here

### High-Level Data Flow

1. Caller selects a `BundleId` or receives one from configuration
2. A higher-level catalog loads a concrete bundle
3. The loaded bundle reports its metadata and capabilities
4. The caller requests a capability adapter such as chat or embeddings
5. The adapter executes requests through the bundle's selected backend implementation under `libs/model/backends/*`
6. Responses return through shared request/response types defined in `libs/model`

Curated artifact staging happens outside this flow in `libs/models`. Backend startup receives the resulting artifact/cache root through `StartOptions` or later artifact descriptors rather than initiating curated downloads itself.

## Framework Principles

### Curated Sustainability

The framework exists to support a curated catalog of models that work well for Motlie use cases, not to expose an open-ended inference substrate.

Implications:

- the public contract should remain stable even as curated bundles change over time
- model-market churn should be absorbed inside bundle curation and validation workflows
- bundle inclusion is a product decision based on quality, operability, and fit for Motlie scenarios
- the framework should make replacing or retiring bundles cheaper than changing downstream product code

### Evaluability as a First-Class Concern

Curation requires repeatable evaluation. The contract must therefore support building harnesses that assess bundles across different task categories without bespoke per-bundle glue.

Implications:

- capability request/response types must be regular enough for harness execution
- outputs should be structured enough to support scoring, comparison, and regression checks
- the contract should avoid bundle-specific response shapes as the primary integration path
- lifecycle and configuration should permit deterministic harness runs where practical
- bundles should be able to describe input/output content kinds and interaction style without backend-specific probing

### Layered Evaluation Boundary

Evaluation support is intentionally split across two layers:

- `libs/model::eval`
  Small, stable eval-facing types and contracts that belong next to the model API
- `libs/model-eval`
  Substantial tooling that operates on those abstractions, such as harness runners, suite loading, scoring, and reports

This split keeps the core contract crate small while still giving evaluation tooling a clear place to grow.

### Library Error Discipline

Library crates in this framework must expose typed errors defined with `thiserror`. Application-facing binaries, examples, and CLIs may use `anyhow` to propagate and annotate those library errors at the boundary.

Implications:

- `libs/model` and `libs/models` should return explicit error enums from public fallible APIs
- error variants should carry enough context for operators and callers to distinguish configuration problems from capability mismatches and internal failures
- non-test runtime code should not use `panic!`, `unwrap()`, or `expect()` as normal control flow
- examples and downloader binaries may use `anyhow` to attach CLI context without weakening the library API surface

## Core Abstractions

### Bundle Identity

The public identity unit is a packaged model bundle, not a raw model family or runtime.

Recommended types:

- `BundleId`
- `BundleMetadata`
- `CapabilityKind`
- `ContentKind`
- `InteractionStyle`
- `CapabilityDescriptor`
- `EmbeddingSpec`
- `EmbeddingDistance`
- `EmbeddingNormalization`

`BundleId` is a stable product-facing identifier such as:

- `Qwen3_5Instruct`
- `Hermes3Chat`
- `BgeSmallEn`

### Bundle Contract

Primary contract:

```rust
pub trait ModelBundle: Send + Sync {
    fn id(&self) -> &BundleId;
    fn metadata(&self) -> &BundleMetadata;
    fn capabilities(&self) -> &Capabilities;
    async fn start(&self, options: StartOptions) -> Result<Box<dyn BundleHandle>, ModelError>;
}
```

The bundle itself is metadata plus load entry point. Runtime state lives in a `BundleHandle`.

### Error Model

`libs/model` owns the stable cross-bundle error surface for lifecycle and capability use through `ModelError`. Higher-level crates such as `libs/models` may define their own typed errors for catalog lookup, artifact staging, or release assembly, but those errors should remain additive rather than replacing `ModelError` inside capability contracts.

### Loaded Handle

```rust
pub trait BundleHandle: Send + Sync {
    fn descriptor(&self) -> &LoadedBundleDescriptor;
    fn capabilities(&self) -> &Capabilities;
    fn supports(&self, capability: CapabilityKind) -> bool;

    fn chat(&self) -> Result<&dyn ChatModel, ModelError>;
    fn completion(&self) -> Result<&dyn CompletionModel, ModelError>;
    fn embeddings(&self) -> Result<&dyn EmbeddingModel, ModelError>;

    async fn shutdown(self: Box<Self>) -> Result<(), ModelError>;
}
```

This keeps lifecycle explicit:

- bundle definition is durable and lightweight
- bundle handle is loaded state
- capability access happens from the loaded handle

### Capability Model

Capabilities are first-class and discoverable. Not every bundle supports every capability.

```rust
pub enum CapabilityKind {
    Chat,
    Completion,
    Embeddings,
    Vision,
    Ocr,
}
```

```rust
pub struct CapabilityDescriptor {
    pub kind: CapabilityKind,
    pub summary: &'static str,
    pub inputs: Vec<ContentKind>,
    pub outputs: Vec<ContentKind>,
    pub interaction: InteractionStyle,
}
```

For embedding bundles, Motlie also needs a bundle-level metadata contract that describes how the vectors should be interpreted by downstream systems such as `libs/db/vector`.

Proposed stable shape:

```rust
pub enum EmbeddingDistance {
    Cosine,
    Dot,
    SquaredL2,
}

pub enum EmbeddingNormalization {
    None,
    L2,
}

pub struct EmbeddingSpec {
    pub dimensions: Option<usize>,
    pub distance: EmbeddingDistance,
    pub normalization: EmbeddingNormalization,
    pub input: ContentKind,
    pub output: ContentKind,
    pub summary: &'static str,
}

pub trait Embedding: ModelBundle {
    fn embedding_spec(&self) -> &EmbeddingSpec;
}
```

Design notes:

- this trait is about the curated bundle definition, not the loaded runtime handle
- `EmbeddingModel` remains the execution-time trait used to actually generate vectors
- `Embedding` is the bundle-level descriptive contract used for catalog display, evaluation selection, and future vector-store integration
- naming is intentionally close to a future `libs/db/vector::EmbeddingSpec`, but `libs/model` must not depend on `libs/db`
- conversion between the two specs should be straightforward later because both are intended to describe the same semantics rather than different layers of behavior

For `google_gemma_300m`, the intended first implementation would be roughly:

```rust
EmbeddingSpec {
    dimensions: Some(768),
    distance: EmbeddingDistance::Cosine,
    normalization: EmbeddingNormalization::L2,
    input: ContentKind::Text,
    output: ContentKind::EmbeddingVector,
    summary: "Normalized text embeddings for semantic similarity and retrieval.",
}
```

```rust
pub struct Capabilities {
    descriptors: Vec<CapabilityDescriptor>,
}
```

The curated model favors a closed enum for capability kinds rather than an open-ended registry. New capability kinds are expected to arrive through code review as part of the curation process. The descriptor layer makes capabilities introspectable for catalogs, harnesses, and operators without exposing backend-specific probing logic.

## Lifecycle

The lifecycle must be consistent across all bundles even when internal backends vary.

### Lifecycle States

1. `Bundle` defined but not loaded
2. `BundleHandle` loaded and ready
3. Zero or more capability requests executed
4. `shutdown()` releases runtime state, caches, device allocations, and temporary unpacked artifacts

### Lifecycle Rules

- `start()` is the only public entry point that may allocate heavyweight runtime state
- `shutdown()` is explicit; dropping without shutdown may be allowed but must not be the only cleanup strategy
- capability adapters must not be accessible before a bundle is loaded
- capability accessors return `UnsupportedCapability` when the bundle does not implement that surface

### Concurrency Expectations

The contract should allow a loaded bundle handle to serve multiple requests concurrently when the implementation supports it, but the trait contract must not promise unbounded concurrency. Concrete bundles may serialize internally.

`StartOptions` should leave room for deployment-specific tuning such as:

- device selection
- cache directories
- unpack roots
- max concurrency hints
- per-bundle runtime knobs

## Capability Surfaces

The capability APIs should expose a stable user-facing interaction surface while hiding backend-specific request plumbing.

They should also expose enough introspection for tooling to understand:

- what normalized content kinds a capability accepts
- what normalized content kinds it produces
- whether it is request/response, multi-turn, batch-oriented, or later streaming

This allows catalogs and harnesses to select the right interaction driver without having to infer semantics from bundle naming or backend choice.

### Chat

```rust
pub trait ChatModel: Send + Sync {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, ModelError>;
}
```

### Text Completion

```rust
pub trait CompletionModel: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError>;
}
```

### Embeddings

```rust
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, ModelError>;
}
```

### Shared Request/Response Types

Shared types should cover:

- input payloads
- generation parameters such as max tokens, temperature, top-p, stop sequences
- response content
- token usage and timing metadata when available
- normalized warnings for partial support

The contract should allow bundles to ignore unsupported tuning fields only when this behavior is explicit and inspectable. Silent best-effort drift across backends will make behavior impossible to reason about.

The contract should prefer shared normalized types even when concrete backends have richer or stranger option sets. Backend-specific extension points may exist, but they must not replace the common contract as the primary integration path.

## Evaluation Harness Support

The model contract should make it straightforward to build evaluation harnesses with canned inputs and expected outputs or scoring rules.

Target evaluation areas include:

- embedding generation
- reasoning-oriented tasks
- summarization
- classification
- chat or instruction following where relevant

The harness system itself may live outside `libs/model`, but the API in `libs/model` must support it cleanly.

Inside `libs/model`, evaluation support should remain lightweight and declarative. A small `model::eval` module is the right home for:

- eval task and track identifiers
- eval case and result envelope types
- capability-to-eval compatibility metadata
- minimal traits that are genuinely foundational to all harness tooling

Executable harness logic, reporting, and suite orchestration belong in `libs/model-eval`, not in the core contract crate.

Contract requirements that enable harnesses:

- stable request/response envelopes across bundles within a capability
- explicit capability discovery so harnesses can select only applicable tests
- introspective capability descriptors so harnesses can choose the right interaction driver
- inspectable execution metadata when available, such as token counts, timings, or warnings
- deterministic configuration surfaces where practical for reproducible comparisons
- normalized error handling so harnesses can distinguish unsupported tasks from execution failures

The design should assume that evaluation is part of the bundle curation lifecycle, not merely post hoc testing.

## API Sketch

Example ergonomic usage for downstream callers:

```rust
use motlie_model::BundleId;
use motlie_models::Catalog;

let catalog = Catalog::default();
let bundle = catalog.bundle(&BundleId::new("qwen3_5_instruct"))?;
let handle = bundle.start(Default::default()).await?;
let chat = handle.chat()?;

let response = chat.generate(ChatRequest::from_messages([
    ("system", "Be concise."),
    ("user", "Summarize the deployment model."),
])).await?;
```

Embedding flow:

```rust
let bundle = catalog.bundle(BundleId::BgeSmallEn)?;
let handle = bundle.start(Default::default()).await?;
let embeddings = handle.embeddings()?;

let response = embeddings.embed(EmbeddingRequest::from_inputs([
    "motlie bundles models with their runtime",
    "embeddings do not expose chat",
])).await?;
```

### API Design Notes

- `BundleId` is the stable selection mechanism
- callers ask for capabilities from a loaded handle rather than pattern-matching on bundle family
- runtime choice is not part of the public call path
- the concrete curated catalog belongs in `libs/models`, with the primary public path exposed as `motlie_models::Catalog`

## Artifact and Packaging Contracts

`libs/model` should define enough artifact vocabulary for `libs/models` to implement packaging and deployment without forcing the core contract crate to understand specific bundle layouts.

In the current implemented contract, that vocabulary is intentionally small:

- `ArtifactPolicy::LocalOnly { root }`
- `ArtifactPolicy::AllowFetch { root }`
- `StartOptions::artifact_policy`

Richer curated artifact manifests, inclusion rules, and provenance controls currently live in `libs/models`, not `libs/model`.

Potential future extensions, if they become necessary, include:

- `ArtifactKind`
- `ArtifactIntegrity`
- `ArtifactLocator`

Example concerns the contract should represent:

- embedded artifact versus sidecar file
- local packaged artifact versus remote-served bundle
- readonly packaged resource versus unpack-on-start cache
- integrity metadata such as digest or signed manifest reference
- optional encryption envelope metadata for future protected bundles

The contract should not encode `ELF`, `rodata`, backend-specific weight formats, or HTTP provider protocol details directly. Those are implementation details owned by bundle implementations and deployment tooling.

## Alternatives Considered

### Alternative A: Single `libs/models` crate with contracts and bundles together

Pros:

- fastest initial delivery
- fewer crates and fewer dependency edges
- simpler workspace wiring on day one

Cons:

- core API stability gets tangled with backend dependencies
- feature-gating and compile-time footprint grow quickly
- bundle catalog and lifecycle contracts become harder to evolve independently

Decision:

Rejected as the primary architecture. The product benefits from a small stable contract crate.

### Alternative B: `libs/model` as contract crate, `libs/models` as curated bundle crate

Pros:

- clear contract/composition split
- lower dependency surface for the stable API
- supports multiple bundle catalogs or backend integration crates later
- easier to feature-gate heavy bundles

Cons:

- adds one more workspace crate
- requires discipline to keep backend details out of `libs/model`

Decision:

Accepted.

### Alternative C: Backend-first API with `Backend`, `Checkpoint`, and `Session` as the public model

Pros:

- maps directly to inference infrastructure
- may help internal experimentation with arbitrary checkpoints

Cons:

- exposes the wrong abstraction to product consumers
- forces callers to reason about runtime compatibility
- conflicts with the curated "bundles as lego blocks" deployment model

Decision:

Rejected. Backend is an implementation concern, not the top-level product abstraction.

### Alternative D: Split local-bundle and remote-model APIs into separate contract systems

Pros:

- can optimize local packaging and remote HTTP semantics independently
- may reduce abstraction tension in the short term

Cons:

- duplicates lifecycle and capability interfaces
- prevents "bundle as lego block" from spanning both local and remote execution
- makes application integration less uniform

Decision:

Rejected. Remote-served models such as a GPT-4.5 HTTP bundle should still implement the same top-level contracts.

### Alternative E: Keep the lifecycle contract minimal and let bundles define richer local APIs

Pros:

- gives each bundle maximum freedom
- may expose backend strengths more directly

Cons:

- undermines composability across bundles
- pushes bundle-specific branching into application code
- weakens the main reason to have a contract crate at all

Decision:

Rejected. The shared contract and composable lifecycle are primary design goals, not secondary conveniences.

### Alternative F: Put all evaluation vocabulary and tooling directly into `libs/model`

Pros:

- fewer crates
- all model-related concepts live in one place

Cons:

- mixes stable API contracts with heavier harness machinery
- increases pressure to add nontrivial dependencies to the core crate
- makes the boundary between declarative eval contracts and executable tooling harder to maintain

Decision:

Rejected. `libs/model` should own only lightweight `model::eval` abstractions, while substantial harness tooling belongs in `libs/model-eval`.

## Testing Scope for PLAN

PLAN must specify concrete tests for:

- bundle lifecycle transitions
- unsupported-capability behavior
- request/response type normalization
- evaluation-harness-facing request/response stability
- artifact descriptor parsing and validation
- cleanup behavior during shutdown
- concurrency and reentrancy expectations for loaded handles

`DESIGN` intentionally leaves backend-specific and bundle-specific validation details to `libs/models` and later PLAN work.

## Open Concerns

- Whether any catalog trait should live in `libs/model` or remain only in `libs/models`
- Whether the first introspection vocabulary should stop at normalized content kinds or include richer transport/file-format annotations later
- How much generation-parameter normalization to promise across heterogeneous backends
- Whether streaming output should be part of the initial contract or deferred until the non-streaming API is stable

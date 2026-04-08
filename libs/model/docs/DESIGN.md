# Model Contract Library Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-07 | @codex-researcher: Initial greenfield design for `libs/model` as the stable contract/lifecycle crate for packaged model bundles. Migration and backward compatibility are explicitly out of scope for this first cut. | All |

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

Without a dedicated contract crate, concrete bundle implementations would either:

1. expose backend-specific APIs and leak runtime details into callers, or
2. invent local abstractions independently, leading to incompatible lifecycle and request/response semantics across bundles.

The core product value is not only hiding backends. It is preserving a common abstraction so bundles compose cleanly inside higher-level Motlie systems regardless of whether they are local, embedded, or remote-backed.

### Solution

`libs/model` defines the shared contract for all packaged model bundles:

1. Stable bundle identity and metadata
2. Uniform load/start/stop lifecycle
3. Shared request/response types for chat, text completion, and embeddings
4. Capability discovery and capability-specific adapters
5. Composable contracts so bundles can be selected, loaded, and invoked uniformly
6. Artifact descriptors for packaged weights and auxiliary assets
7. Common errors and unsupported-capability behavior

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
2. `libs/models`
   Curated bundle catalog, bundle registration, backend composition, packaging policy
3. `libs/model-eval`
   Substantial evaluation tooling such as harness runners, suite loading, scoring, reporting, and CLI-oriented support built on `model::eval`
4. Backend integration crates or private modules
   Execution adapters such as `mistral.rs`, `llama.cpp`, `ONNX Runtime`, or HTTP-backed providers

Dependency direction:

- applications depend on `libs/models` and may also use `libs/model` types directly
- `libs/models` depends on `libs/model`
- `libs/model-eval` depends on `libs/model` and may optionally integrate with `libs/models` for catalog-driven runs
- backend integrations depend on `libs/model` as needed for shared types
- `libs/model` does not depend on bundle catalogs, harness tooling, or backend crates

### High-Level Data Flow

1. Caller selects a `BundleId` or receives one from configuration
2. A higher-level catalog loads a concrete bundle
3. The loaded bundle reports its metadata and capabilities
4. The caller requests a capability adapter such as chat or embeddings
5. The adapter executes requests through the bundle's internal backend implementation
6. Responses return through shared request/response types defined in `libs/model`

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

### Layered Evaluation Boundary

Evaluation support is intentionally split across two layers:

- `libs/model::eval`
  Small, stable eval-facing types and contracts that belong next to the model API
- `libs/model-eval`
  Substantial tooling that operates on those abstractions, such as harness runners, suite loading, scoring, and reports

This split keeps the core contract crate small while still giving evaluation tooling a clear place to grow.

## Core Abstractions

### Bundle Identity

The public identity unit is a packaged model bundle, not a raw model family or runtime.

Recommended types:

- `BundleId`
- `BundleMetadata`
- `BundleVersion`
- `Capability`

`BundleId` is a stable product-facing identifier such as:

- `Qwen3_5Instruct`
- `Hermes3Chat`
- `BgeSmallEn`

### Bundle Contract

Primary contract:

```rust
pub trait ModelBundle: Send + Sync {
    fn id(&self) -> BundleId;
    fn metadata(&self) -> &BundleMetadata;
    fn capabilities(&self) -> Capabilities;
    async fn start(&self, options: StartOptions) -> Result<BundleHandle, ModelError>;
}
```

The bundle itself is metadata plus load entry point. Runtime state lives in a `BundleHandle`.

### Loaded Handle

```rust
pub trait BundleHandle: Send + Sync {
    fn descriptor(&self) -> &LoadedBundleDescriptor;
    fn capabilities(&self) -> Capabilities;

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
pub enum Capability {
    Chat,
    Completion,
    Embeddings,
}
```

```rust
pub struct Capabilities {
    pub chat: bool,
    pub completion: bool,
    pub embeddings: bool,
}
```

The first cut should prefer explicit booleans over a more abstract extensible registry. This keeps the contract obvious and makes unsupported-capability failures straightforward.

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
- inspectable execution metadata when available, such as token counts, timings, or warnings
- deterministic configuration surfaces where practical for reproducible comparisons
- normalized error handling so harnesses can distinguish unsupported tasks from execution failures

The design should assume that evaluation is part of the bundle curation lifecycle, not merely post hoc testing.

## API Sketch

Example ergonomic usage for downstream callers:

```rust
use motlie_model::{BundleId, ModelCatalog};

let bundle = catalog.bundle(BundleId::Qwen3_5Instruct)?;
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
- a catalog trait may live here if needed, but the concrete curated catalog belongs in `libs/models`

## Artifact and Packaging Contracts

`libs/model` should define enough artifact vocabulary for `libs/models` to implement packaging and deployment without forcing the core contract crate to understand specific bundle layouts.

Recommended types:

- `BundleArtifact`
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

- Whether `ModelCatalog` should live in `libs/model` as a trait or only in `libs/models`
- Whether `Capabilities` should remain fixed booleans in v1 or use an extensible capability registry
- How much generation-parameter normalization to promise across heterogeneous backends
- Whether streaming output should be part of the initial contract or deferred until the non-streaming API is stable

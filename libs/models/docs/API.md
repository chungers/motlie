# Curated Model Bundle API

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-07 | @codex-researcher: Initial API sketch for `libs/models` catalog and descriptor shapes. Reflects the current scaffold, not the final loaded-bundle runtime API. | All |
| 2026-04-07 | @codex-researcher: Added explicit curated artifact-control examples and updated the first embedding slice to use the real `mistralrs` builder path with separate pre-download support. | Overview, API Sketch, Notes |

This document sketches the concrete API shapes currently introduced in `libs/models`. The crate now owns both the descriptor catalog and the curated bundle constructors that bind those descriptors to a backend implementation.

## Overview

The first concrete `libs/models` API is an in-memory `Catalog` of curated bundle descriptors. This lets Motlie start expressing:

- stable bundle IDs
- family and support tier metadata
- packaging and backend selection
- introspective capability metadata
- build/platform constraints
- evaluation-track membership
- curated bundle instantiation
- explicit artifact download control separate from backend cache population

The goal is to make the product-facing bundle layer tangible before the runtime-facing bundle handle APIs are finalized.

## Core Types

Current public shapes:

- `BundleId`
- `BundleFamily`
- `SupportTier`
- `PackagingMode`
- `BackendKind`
- `PlatformConstraint`
- `BuildConstraint`
- `BundleRequirements`
- `BundleDescriptor`
- `Catalog`

`BundleId`, `Capabilities`, and capability introspection types come from `motlie_model`. Evaluation-track membership is expressed with `motlie_model::eval::EvalTrack`.

## API Sketch

### Defining a Bundle Descriptor

```rust
use motlie_model::eval::EvalTrack;
use motlie_models::{
    BackendKind, BuildConstraint, BundleDescriptor, BundleFamily, BundleId,
    BundleRequirements, PackagingMode, PlatformConstraint, SupportTier,
};
use motlie_model::{Capabilities, CapabilityDescriptor};

let qwen = BundleDescriptor {
    id: BundleId::new("qwen3_5_instruct"),
    display_name: "Qwen 3.5 Instruct".into(),
    family: BundleFamily::Qwen,
    support_tier: SupportTier::Supported,
    capabilities: Capabilities::new(vec![
        CapabilityDescriptor::chat(),
        CapabilityDescriptor::completion(),
    ]),
    packaging: PackagingMode::Sidecar,
    backend: BackendKind::MistralRs,
    requirements: BundleRequirements {
        platform: vec![PlatformConstraint::Linux],
        build: vec![BuildConstraint::Feature("mistral".into())],
    },
    eval_tracks: vec![
        EvalTrack::Chat,
        EvalTrack::Reasoning,
        EvalTrack::Summarization,
        EvalTrack::Classification,
    ],
};
```

### Registering Bundles in a Catalog

```rust
use motlie_model::eval::EvalTrack;
use motlie_models::{BundleId, Catalog};

let mut catalog = Catalog::new();
catalog.register(qwen.clone(), || {
    panic!("example constructor omitted for API illustration")
});

let bundle = catalog
    .bundle(&BundleId::new("qwen3_5_instruct"))
    .expect("bundle should exist");

assert!(bundle.capabilities.supports(motlie_model::CapabilityKind::Chat));
assert!(bundle.supports_track(EvalTrack::Reasoning));
assert_eq!(
    bundle.capability_descriptors()[0].summary,
    "Multi-turn text interaction with text output."
);
```

### Selecting Bundles for an Evaluation Track

```rust
use motlie_model::eval::EvalTrack;

let reasoning_bundles: Vec<_> = catalog
    .bundles_for_track(EvalTrack::Reasoning)
    .map(|bundle| bundle.id.as_str().to_owned())
    .collect();
```

### Instantiating the First Vertical Slice

```rust
use motlie_model::{BundleId, EmbeddingRequest, StartOptions};
use motlie_models::{default_artifact_root, Catalog};

let catalog = Catalog::with_defaults();
let bundle = catalog
    .instantiate(&BundleId::new("embeddinggemma_300m"))
    .expect("bundle should exist");

let handle = bundle
    .start(StartOptions {
        cache_root: Some(default_artifact_root()),
        ..Default::default()
    })
    .await?;
let embeddings = handle.embeddings()?;
let response = embeddings
    .embed(EmbeddingRequest {
        inputs: vec![
            "curated bundle catalog".into(),
            "mistral embedding vertical slice".into(),
        ],
    })
    .await?;
```

### Explicit Artifact Control

```rust
use motlie_model::BundleId;
use motlie_models::{default_artifact_root, download_bundle_artifacts, Catalog};

let catalog = Catalog::with_defaults();
let bundle_id = BundleId::new("embeddinggemma_300m");

let artifacts = catalog
    .artifacts(&bundle_id)
    .expect("bundle should expose artifact download control");

assert_eq!(artifacts.control_name, "embeddinggemma_300m");

let summary = download_bundle_artifacts(
    &catalog,
    &bundle_id,
    &default_artifact_root(),
)?;

assert!(!summary.downloaded.is_empty());
```

The same flow is available from the binary target:

```sh
cargo run -p motlie-models --bin motlie-models-download -- embeddinggemma_300m
```

## Notes

- `BundleId` is currently a string-backed newtype rather than an enum. This keeps the catalog extensible while still giving the crate a stable selection key.
- capability metadata comes from `motlie_model`, so the catalog can describe input/output shape and interaction style without inventing its own parallel schema
- `BackendKind` and `PackagingMode` are metadata for catalog reasoning and observability. They do not make runtime choice part of the application control path.
- `Catalog` now also owns curated bundle instantiation through registered constructors.
- Curated artifact download is explicit and independent of the backend library's own cache-miss behavior. Backends consume the curated artifact root through `StartOptions`, then populate or reuse their own internal cache layout from there as needed.

## Next Step

The next API layer should be added in `libs/model`:

- lifecycle types
- loaded-bundle handle contracts
- capability adapter traits
- lightweight eval-facing case/result types beyond `EvalTrack`

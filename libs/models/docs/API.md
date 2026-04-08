# Curated Model Bundle API

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-07 | @codex-researcher: Initial API sketch for `libs/models` catalog and descriptor shapes. Reflects the current scaffold, not the final loaded-bundle runtime API. | All |

This document sketches the concrete API shapes currently introduced in `libs/models`. It describes the bundle catalog and descriptor layer only. It does not yet describe the final loaded-bundle lifecycle or capability adapters; those still belong to the next round of `libs/model` contract work.

## Overview

The first concrete `libs/models` API is an in-memory `Catalog` of curated bundle descriptors. This lets Motlie start expressing:

- stable bundle IDs
- family and support tier metadata
- packaging and backend selection
- introspective capability metadata
- build/platform constraints
- evaluation-track membership

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
catalog.register(qwen);

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

## Notes

- `BundleId` is currently a string-backed newtype rather than an enum. This keeps the catalog extensible while still giving the crate a stable selection key.
- capability metadata comes from `motlie_model`, so the catalog can describe input/output shape and interaction style without inventing its own parallel schema
- `BackendKind` and `PackagingMode` are metadata for catalog reasoning and observability. They do not make runtime choice part of the application control path.
- `Catalog` is intentionally in-memory and lightweight in this first pass. Loading, bundle construction, and runtime handle APIs are still to come.

## Next Step

The next API layer should be added in `libs/model`:

- lifecycle types
- loaded-bundle handle contracts
- capability adapter traits
- lightweight eval-facing case/result types beyond `EvalTrack`

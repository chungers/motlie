# Model Evaluation Tooling API

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-08 | @codex-researcher: Added the first API document for `libs/model-eval` to make the current crate boundary explicit. The crate is intentionally scaffold-only today; executable harness shapes remain specified at the design level until implementation begins. | All |
| 2026-04-08 | @codex-researcher: Added the first implemented helper so the crate now proves cross-crate consumption of `motlie_model::eval` and `libs/models::Catalog` for the embedding vertical slice, while keeping the larger runner/report API deferred. | Current Public Surface, Notes |

This document records the current public surface of `libs/model-eval`.

The crate is intentionally minimal today. It exists to reserve the tooling boundary above `motlie_model::eval`, but it does not yet expose a stable runner, suite-loading, or scoring API. Implementers should therefore treat the DESIGN and PLAN docs as the source of truth for the first substantial API that will be added here.

## Current Public Surface

The current crate currently exports:

```rust
pub fn supports_track(capabilities: &Capabilities, track: EvalTrack) -> bool
pub struct ModelEvalScaffold;
```

`supports_track(...)` is intentionally small. It proves the dependency direction and contract split for the first embedding slice:

- the capability-to-track mapping lives in `motlie_model::eval`
- `libs/model-eval` consumes that mapping without knowing about specific bundle implementations
- `libs/models::Catalog` can then be queried and filtered using those same stable contracts

Illustrative usage:

```rust
use motlie_model::eval::EvalTrack;
use motlie_model::BundleId;
use motlie_model_eval::supports_track;
use motlie_models::Catalog;

let catalog = Catalog::with_defaults();
let descriptor = catalog
    .bundle(&BundleId::new("embeddinggemma_300m"))
    .expect("default catalog should contain the embedding bundle");

assert!(supports_track(&descriptor.capabilities, EvalTrack::Embeddings));
```

The rest of the crate remains intentionally non-committal so the larger runner/suite/report API can still be introduced cleanly in a later phase.

## Planned First Real API Layer

The first substantial public API for this crate is expected to include:

- `EvalSuite`
- `HarnessRunner`
- `EvalReport`
- `EvalSummary`
- task-specific runner modules layered on `motlie_model::eval::{EvalTrack, EvalCase, EvalResult}`

Those shapes are currently specified in:

- [DESIGN.md](/Users/dchung/projects/claude-mistral/motlie/libs/model-eval/docs/DESIGN.md)
- [PLAN.md](/Users/dchung/projects/claude-mistral/motlie/libs/model-eval/docs/PLAN.md)

## Notes

- `libs/model-eval` is not the home of lightweight eval vocabulary; that remains in `motlie_model::eval`.
- No typed error surface is specified here yet because the crate does not yet expose fallible runner APIs.
- When the real runner API lands, library-facing fallible APIs should follow the same framework rule as the other crates: typed `thiserror` errors in the library, `anyhow` only at binary or CLI boundaries.

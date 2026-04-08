# Model Evaluation Tooling API

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-08 | @codex-researcher: Added the first API document for `libs/model-eval` to make the current crate boundary explicit. The crate is intentionally scaffold-only today; executable harness shapes remain specified at the design level until implementation begins. | All |

This document records the current public surface of `libs/model-eval`.

The crate is intentionally minimal today. It exists to reserve the tooling boundary above `motlie_model::eval`, but it does not yet expose a stable runner, suite-loading, or scoring API. Implementers should therefore treat the DESIGN and PLAN docs as the source of truth for the first substantial API that will be added here.

## Current Public Surface

The current crate exports a single scaffold marker:

```rust
pub struct ModelEvalScaffold;
```

That surface is intentionally non-committal. It keeps the workspace wiring buildable while preserving room to introduce the real harness API in a later phase.

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

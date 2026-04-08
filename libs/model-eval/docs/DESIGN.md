# Model Evaluation Tooling Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-07 | @codex-researcher: Initial greenfield design for `libs/model-eval` as the substantial evaluation tooling crate layered on top of `libs/model::eval`. Migration and backward compatibility are explicitly out of scope for this first cut. | All |
| 2026-04-08 | @codex-researcher: Clarified the current crate boundary: `libs/model-eval` is scaffold-only in code today, with the first substantial runner API still specified in DESIGN/PLAN rather than implemented. | Overview, Solution, API Sketch |

This document defines the design for `libs/model-eval`, the evaluation tooling crate for Motlie's curated model framework. Its job is to provide the executable harness machinery that operates on the lightweight evaluation abstractions defined in `libs/model::eval`.

The governing principle is role clarity: `libs/model` owns small, stable contracts, including lightweight eval-facing vocabulary, while `libs/model-eval` owns the heavier machinery needed to run, score, compare, and report evaluations across curated bundles.

## Table of Contents

- [Overview](#overview)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Architecture](#architecture)
- [Framework Principles](#framework-principles)
- [Core Responsibilities](#core-responsibilities)
- [API Sketch](#api-sketch)
- [Alternatives Considered](#alternatives-considered)
- [Testing Scope for PLAN](#testing-scope-for-plan)
- [Open Concerns](#open-concerns)

---

## Overview

### Problem Statement

Motlie needs repeatable, structured ways to evaluate curated model bundles across multiple task categories such as embeddings, reasoning, summarization, classification, and chat quality. These evaluations are part of the curation workflow used to admit, compare, promote, and retire bundles.

The small contracts in `libs/model::eval` are necessary, but they are not sufficient. Real evaluation requires runners, suite loading, scoring logic, result aggregation, and reporting. Putting all of that directly into `libs/model` would blur the boundary between stable API contracts and heavier tooling.

### Solution

`libs/model-eval` provides the substantial evaluation machinery:

1. harness runners that operate on `libs/model` capability contracts
2. suite loading from structured evaluation definitions
3. task-specific scoring and normalization
4. result aggregation, comparison, and reporting
5. optional integration with `libs/models` for catalog-driven evaluation runs

## Goals and Non-Goals

### Goals

- Provide a clear home for substantial evaluation tooling
- Operate on the lightweight eval abstractions defined in `libs/model::eval`
- Support repeatable curated-bundle assessment across multiple task categories
- Keep evaluation logic reusable by both library code and future CLI tools
- Support candidate-versus-baseline comparisons as part of curation workflows

### Non-Goals

- Owning the core model lifecycle or capability contracts
- Replacing `libs/model::eval` as the home for small stable eval vocabulary
- Serving as a generic benchmark platform for arbitrary third-party models
- Defining product bundle policy on its own without the curated catalog

## Architecture

`libs/model-eval` is layered above `libs/model` and may optionally integrate with `libs/models`.

1. `libs/model`
   Owns model lifecycle contracts and lightweight `model::eval` abstractions
2. `libs/models`
   Owns curated bundle definitions and catalog metadata used for selection
3. `libs/model-eval`
   Owns executable harness logic, suite orchestration, scoring, comparison, and reporting

Dependency direction:

- `libs/model-eval` depends on `libs/model`
- `libs/model-eval` may depend on `libs/models` for catalog-driven runs
- `libs/model` does not depend on `libs/model-eval`

## Framework Principles

### Role Clarity

The split between `libs/model::eval` and `libs/model-eval` is intentional:

- `libs/model::eval` contains only lightweight, stable, dependency-light abstractions
- `libs/model-eval` contains executable, potentially heavier tooling

This avoids turning the core contract crate into a harness framework while still keeping eval vocabulary close to the model contracts.

### Evaluation Serves Curation

The purpose of this crate is to support Motlie's curated model lifecycle, not to become a generic public leaderboard system.

Implications:

- evaluation tracks should map to Motlie-relevant use cases
- result formats should support admission, regression review, and bundle promotion decisions
- harness logic should favor repeatability and explainability over novelty

## Core Responsibilities

Primary areas expected to live in this crate:

- harness runners for capability-specific evaluation flows
- suite loading and manifest parsing
- scoring implementations for evaluation tracks
- aggregation and reporting structures
- regression comparison helpers
- optional catalog-driven bundle selection helpers

Representative evaluation tracks:

- embeddings
- reasoning-oriented tasks
- summarization
- classification
- chat or instruction following

Examples of what belongs here rather than in `libs/model::eval`:

- loading a summarization suite from files
- running all embedding cases against a selected bundle
- computing score summaries and pass/fail thresholds
- comparing a candidate bundle against a promoted baseline

## API Sketch

Illustrative shapes:

```rust
pub struct EvalSuite {
    pub name: String,
    pub track: EvalTrack,
    pub cases: Vec<EvalCase>,
}

pub trait HarnessRunner {
    async fn run_suite(
        &self,
        bundle: &dyn BundleHandle,
        suite: &EvalSuite,
    ) -> Result<EvalReport, EvalError>;
}

pub struct EvalReport {
    pub suite_name: String,
    pub bundle_id: BundleId,
    pub summary: EvalSummary,
    pub results: Vec<EvalResult>,
}
```

The key point is that these APIs operate on `libs/model` contracts and `model::eval` vocabulary rather than inventing an unrelated runner interface.

## Alternatives Considered

### Alternative A: Put all evaluation contracts and tooling directly into `libs/model`

Pros:

- fewer crates
- one obvious place for all model-related code

Cons:

- burdens the core contract crate with heavier harness logic
- increases the chance of dependency creep
- makes role boundaries less clear over time

Decision:

Rejected. Keep lightweight eval contracts in `libs/model::eval`, and put substantial tooling here.

### Alternative B: Keep everything inside `libs/models`

Pros:

- evaluation stays close to curated bundle implementations
- fewer top-level crates

Cons:

- mixes bundle catalog concerns with harness tooling
- makes evaluation less reusable outside the curated catalog path
- risks turning `libs/models` into an operational catch-all

Decision:

Rejected. `libs/models` should contribute metadata and integrations, but `libs/model-eval` should own the harness machinery.

## Testing Scope for PLAN

PLAN must specify concrete tests for:

- suite loading and validation
- harness execution across capability types
- scoring behavior and threshold handling
- structured result aggregation
- catalog-driven bundle selection where integrated
- regression comparison behavior

## Open Concerns

- Whether evaluation suite assets should live under this crate or in a repo-level `evals/` directory
- How much scoring logic should be data-driven versus hardcoded Rust implementations
- Whether a dedicated CLI in `bins/` should be part of the first cut or follow shortly after

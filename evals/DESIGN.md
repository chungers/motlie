# Eval Coverage Ontology — DESIGN

## Changelog
- 2026-06-14 @ops48-orchestrator — initial cut. Captures the (bundle, capability, accelerator) coverage tuple + full-accounting rule. Outgrowth of #399/#486/#513/#514/#518: silent coverage gaps (kokoro, qwen3-tts, kroko-2025 missed) and the "ASR-on-GPU" surprise class.

## Problem
Curated-eval coverage is a flat, hand-curated cell list. "Does bundle X support capability Y on accelerator Z?" is answerable only by *absence* — a missing cell is indistinguishable from "not applicable" vs "we forgot." This causes (a) silent test gaps (models defined but never evaluated) and (b) deployment disappointments (e.g. picking an ASR model for GPU when it has no GPU backend). No single artifact accounts for the full space.

## Key insight: coverage is a product space
Coverage is the set of tuples **(CuratedBundle, Capability, Accelerator)**. Completeness = **every tuple in the space is explicitly classified** — none undeclared.

### Dimensions & cardinality
- **bundle**: `CuratedBundle` enum — **18** variants (the registry; the canonical `bundle_id` strings, #518).
- **capability**: `Capability` enum — `chat | asr | tts | embeddings | perf | tool_use` (perf/tool_use cross-cut LLM bundles).
- **accelerator**: the eval **profile** = (arch, accelerator), reusing the existing profile enum: `local-cpu-x86_64`, `local-cpu-aarch64`, `apple-metal` (arm64+metal), `dgx-spark`/`cuda-workstation` (aarch64/x86_64 + cuda). `aarch64`==`arm64`; arch and accelerator are one axis, not two.

The space is **sparse**: pruned by each bundle's *advertised* capabilities (a chat model has no `asr` row). Real cardinality ≈ `Σ_bundle (advertised-caps × applicable-profiles)`, far below `18 × |Capability| × |Profile|`.

### Cell state (each tuple is exactly one)
- **`Validated`** — produced a `passed` eval cell (carries the metric).
- **`NotApplicable(Reason)`** — declared can't/won't run, with a typed reason. *Full accounting, not a silent miss.*
- **`Gap`** — known-missing, must be filled. Transient, CI-visible.

### Reason taxonomy (typed, queryable — not freetext)
`NoBackendForAccelerator` (ORT audio on Metal), `UpstreamNoGpuForwarding` (mistralrs→CUDA/Metal), `CpuOnlyStaticArchive`, `NoGpuPath` (embeddings), `RequiresArtifactGate`, … (extensible enum).

## Enforcement
A CI test asserts: for every (bundle × advertised-capability × in-profile-accelerator), the declared state is `Validated`+passed **or** `NotApplicable`+reason. Any **undeclared** tuple, or a runtime `blocked` cell with no matching `NotApplicable` declaration, **fails CI**. Extends the #518 `CANONICAL_IDS` completeness test from the bundle axis to the full tuple. → No silent fails or disappointments.

## What changes (minimal)
1. **Schema (`libs/models/src/lib.rs`)** — each `CuratedBundle` descriptor advertises `capabilities` + an `accelerator_applicability: (Capability, Profile) -> Supported | NotApplicable(Reason)` table (model layer knows its backends). Add the `Reason` enum.
2. **Runner TOML (`evals/snapshots/*.toml`)** — cells *derived* from `advertised-caps × profiles × applicability` instead of hand-listed (closes silent-miss structurally). Minimal variant: add an `applicability` block; cleaner: generate the cell list.
3. **Result reconciliation (`bins/evals/src/result.rs`, `runner/support.rs`)** — a `blocked` outcome must map to a declared `NotApplicable` reason; unmatched block = flagged failure.
4. **Enforcement test (`bins/evals`)** — the tuple-completeness assertion above.
5. **Docs** — coverage-report **"Accounting Matrix"**: rows = bundle×capability, cols = profiles, cells = ✅ validated (+metric) / ⛔ N/A (reason) / ⏳ gap. The model-selection + no-surprise view.

## Alternatives considered
- **Flat hand-curated cell list (status quo)** — rejected: silent gaps, no accounting, the surprise class.
- **Auto-run every bundle on every profile** — rejected: wastes runs on impossible combos + still no *declared* rationale (a blocked cell stays ambiguous).

## In-flight gaps to fill (data-driven, not assumed)
- **kroko-2025 + qwen3-tts on CUDA** — currently `Gap`; run on the dgx (kroko via static-CUDA-ORT #513; qwen3-tts via ggml-CUDA) so the GPU-vs-CPU tradeoff is decided by data. (Standing finding so far: small ORT audio sees ~no CUDA speedup; LLMs do.)

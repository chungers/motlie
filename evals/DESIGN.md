# Eval Coverage Ontology — DESIGN

## Changelog
- 2026-06-14 @ops48-orchestrator — initial cut. Captures the (bundle, capability, accelerator) coverage tuple + full-accounting rule. Outgrowth of #399/#486/#513/#514/#518: silent coverage gaps (kokoro, qwen3-tts, kroko-2025 missed) and the "ASR-on-GPU" surprise class.
- 2026-06-14 @onto-impl — finalize for #521 (v1): accelerator→Profile rename + enum-grounded schema.
- 2026-06-14 @onto-impl — **v2, David-approved two-layer design.** Restructured to the compile-time/runtime split: compile-time accelerator capability is declared on **`BackendKind`** (the EP truth lives at the backend; bundles inherit), **not** a Profile enum in motlie-models. Profile stays the **runtime** eval instantiation in `bins/evals`; the existing `accelerator.rs` `Profile→AcceleratorClass` bridge is the join. Added the **4-state taxonomy** (Validated / NotApplicable / **BuildGap** / Gap) that resolves the compile-vs-runtime mismatch. Added §9 **Data Migration** layout (David: data migrated into the (bundle,capability,Profile) organization before merge — sign-off required before moving). Dropped v1's per-bundle applicability table and the motlie-models `Profile`/`Capability` enums. **No code/data edited — review gate.**

## Problem
Curated-eval coverage is a flat, hand-curated cell list. "Does bundle X support capability Y on profile Z?" is answerable only by *absence* — a missing cell is indistinguishable from "not applicable" vs "build didn't provision it" vs "we forgot." This causes (a) silent test gaps, (b) deployment disappointments (ASR-on-GPU surprise), and (c) **the compile-vs-runtime confusion**: a `blocked` aarch64-CUDA-ORT cell pre-#513 looked identical to a permanently-impossible ORT-on-Metal cell, though one is fixable (compile the path) and one never is.

## Key insight: coverage is a product space with a two-layer truth
Coverage is the set of tuples **(CuratedBundle, Capability, Profile)**. Completeness = every tuple in the (sparse) space is explicitly classified. But *what* classifies a tuple has **two layers** that must not be conflated:

- **Compile-time / model truth (motlie-models / motlie-model):** what a model's backend's execution provider can *ever* target. Permanent. ORT has no Metal EP — period. This is declared on **`BackendKind`** and inherited by every bundle using that backend.
- **Runtime / build truth (bins/evals):** what *this build* (its active cargo features + git SHA), running on *this Profile*, actually instantiated and resolved. A correct CUDA-capable backend still yields nothing if this build didn't compile the CUDA feature.

The 4-state taxonomy (below) is exactly the reconciliation of these two layers.

### Dimensions (keyed to existing enums — no freetext)
- **bundle**: `CuratedBundle` enum — 18 variants, canonical `bundle_id` (#518). `libs/models/src/lib.rs`.
- **capability**: the eval-matrix axis `chat | tool_use | asr | tts | embeddings | perf` (evals `CapabilityName`, `bins/evals/src/scenario.rs`). Each bundle's **advertised** set is **derived** from its existing `descriptor().capabilities` (`CapabilityKind`, `libs/model/src/lib.rs:156`): `asr←Transcription`, `tts←Speech`, `embeddings←Embeddings`, `chat←Chat`, `tool_use←ToolUse`, `perf←(any bundle advertising Chat)`. No new capability enum; no hand-listing.
- **Profile**: the **runtime** eval instantiation — `(arch, accelerator)`, one axis. 5 values: `local-cpu-x86_64`, `local-cpu-aarch64`, `apple-metal`, `dgx-spark` (aarch64+cuda GB10), `cuda-workstation` (x86_64+cuda). Stays string-keyed in `bins/evals` (`snapshot.rs`, `accelerator.rs`); **no Profile enum in motlie-models** — Profile is an eval concern, the model layer only knows `AcceleratorClass`.

The space is **sparse**: pruned by each bundle's advertised capabilities.

## §A Compile-time layer — accelerator capability on `BackendKind`
`BackendKind` (`libs/model/src/lib.rs:97`: `Http, LlamaCpp, MistralRs, Ort, Qwen3TtsCpp, SherpaOnnx, WhisperCpp`) is where the execution-provider truth lives. Add one declaration; bundles inherit it through `descriptor().backend`.

```rust
// motlie-model — minimal model-side accelerator enum (resolution states Any/Unavailable are runtime-only, excluded).
pub enum Accelerator { Cpu, Cuda, Metal }

pub enum AccelSupport {
    /// Backend's EP can target this accelerator. Compiled only when `feature` is
    /// active in the build (None = always present, e.g. Cpu). The feature gate is
    /// what distinguishes Validated from BuildGap at reconciliation time.
    Targetable { feature: Option<&'static str> },
    /// Backend fundamentally has no path here — permanent, independent of build.
    Unsupported(Reason),
}

impl BackendKind {
    pub fn accel_support(self, accel: Accelerator) -> AccelSupport;
}
```

Declared truth (feature-gated), e.g.:
| BackendKind | Cpu | Cuda | Metal |
|---|---|---|---|
| `Ort` / `SherpaOnnx` (audio EP) | `Targetable{None}` | `Targetable{feature:"…-cuda"}` *(static archive is CPU-only today, #495/#513 → BuildGap until shipped)* | `Unsupported(NoExecutionProviderForAccelerator)` **(permanent)** |
| `LlamaCpp` | `Targetable{None}` | `Targetable{feature:"llama-cpp-cuda"}` | `Targetable{feature:"metal"}` |
| `MistralRs` | `Targetable{None}` | `Targetable{feature:"cuda"}` | `Unsupported(UpstreamNoGpuForwarding)` |
| `WhisperCpp` / `Qwen3TtsCpp` (ggml) | `Targetable{None}` | `Targetable{feature:"…-cuda"}` | `Targetable{feature:"metal"}` |

`Reason` (new typed enum, `#[non_exhaustive]`, motlie-model): `NoExecutionProviderForAccelerator`, `UpstreamNoGpuForwarding`, `CpuOnlyStaticArchive`, `RequiresArtifactGate`, … . These are the *permanent* NotApplicable reasons; they are a function of `(BackendKind, Accelerator)`, not hand-authored per bundle. (Per-bundle runtime divergence — e.g. gemma-mistralrs failing on CUDA where qwen3-mistralrs passes — is a runtime outcome, reconciled in §C, not a compile-time NotApplicable.)

## §B Runtime layer — Profile is the eval instantiation
`bins/evals` keeps Profile (string) and the existing bridge `requested_for_profile(profile) -> AcceleratorClass` / `resolve_for_profile(profile, platform) -> AcceleratorSection` (`accelerator.rs`). This bridge is **the join**: it turns a runtime Profile into the `AcceleratorClass`/`Accelerator` that §A is keyed on. Records already carry the runtime evidence needed for reconciliation:
- `coverage.{bundle_id, capability, profile, resolved_accelerator}`, `coverage.terminal_outcome` + `OutcomeReason`
- `runtime.cargo_features` (the build's active feature set) and `identity.git_sha`.

## §C The 4-state cell taxonomy (the compile↔runtime reconciliation)
For each tuple `(bundle, capability, Profile)`, let `accel = bridge(Profile)`, `support = bundle.backend().accel_support(accel)`:

1. **`Validated`** — a `passed` record exists for the tuple with `resolved_accelerator == accel`. (Implies `support == Targetable`.) Carries the metric.
2. **`NotApplicable(Reason)`** — `support == Unsupported(Reason)`. The model fundamentally can't; **compile-permanent** (ORT-on-Metal). A runtime `blocked` here must reconcile to this Reason; anything else is a CI failure.
3. **`BuildGap`** — `support == Targetable{feature}` but **this build didn't provide the path**: either no record exists for the tuple, or the record is `blocked` and its `runtime.cargo_features` lacks `feature` (e.g. aarch64-CUDA-ORT pre-#513 — capable in principle, not compiled in that build/SHA). Distinct from NotApplicable: **fixable by building/provisioning the path**; reconciled via recorded feature flags + SHA.
4. **`Gap`** — `support == Targetable` and `feature` *was* compiled (or none required) but the cell was never scheduled / no passing record (e.g. kroko-2025 + qwen3-tts on CUDA). Should run; CI-visible, transient.

Reconciliation rule (results-side test, `bins/evals`):
- every `passed` record ⇒ tuple state `Validated` and `support == Targetable`.
- every `blocked`/`skipped` record ⇒ tuple is `NotApplicable(r)` with `OutcomeReason → r`, **or** `BuildGap` (Targetable but feature absent in the record's build). An undeclared tuple, or a `blocked` whose runtime reason contradicts §A, fails CI.
- This extends the #518 `CANONICAL_IDS` completeness test from the bundle axis to the full tuple.

## §D Coverage-report **Accounting Matrix**
New section in `evals report --aggregate` (`bins/evals/src/report.rs`): rows = `bundle × capability` (advertised only), cols = the 5 Profiles, cell = `✅ Validated (+metric)` / `⛔ NotApplicable (reason)` / `🔧 BuildGap (feature/SHA)` / `⏳ Gap`. Generated by joining §A's declaration (via the §B bridge) with the records. The model-selection + no-surprise + "what must we still build/run" view.

## §9 Data Migration — PROPOSAL (sign-off required before moving; David)
David: *"the data needs to be migrated so it's in this organization before merge to main."* So this is a real reorg of committed `evals/results/`, not just a derived index. The #518 discipline still holds: **propose + sign-off before moving**, and **preserve provenance** (no record content rewritten; identity fields stay intact inside each record).

**Proposed canonical layout — enum-keyed, one record per file:**
```
evals/results/coverage/<bundle_id>/<capability>/<profile>/<git_sha8>-<host_slug>-<run_ts>.json
                         │            │            │          └─ disambiguates multiple runs of the same tuple
                         │            │            └─ one of the 5 Profile strings (validated against the registry)
                         │            └─ one of {chat,tool_use,asr,tts,embeddings,perf} (CapabilityName)
                         └─ ∈ CuratedBundle::CANONICAL_IDS  (#518)
```
- Each leaf file is exactly one `ResultRecord` (the cell), **byte-identical** to its source row — migration only *relocates+splits*, never edits content. Provenance (`run_id`, `git_sha`, `command_line`, host) stays inside the record; the original run grouping is recoverable by `run_id`.
- **Path components are validated against the enums** at migration time and by the enforcement test — a path that doesn't parse to `(CuratedBundle, CapabilityName, Profile)` fails. This *is* the enforced results↔enums mapping.
- **Migration is mechanical + reproducible:** a `evals migrate-results` step reads each existing `results.jsonl`, emits leaf files by tuple. Re-runnable; deterministic given the same inputs.

**Open migration questions for review/sign-off:**
1. **Keep or retire the original run dirs?** Lean: **keep** `evals/results/curated-v2-smoke/<run-dir>/` as the immutable raw archive (provenance/audit) and add `coverage/` as the canonical enum-keyed view populated from them — belt-and-suspenders, no data loss. (Alt: move-and-delete to avoid duplication — riskier, loses run-grouping on disk.)
2. **One-record-per-file vs per-tuple `records.jsonl`** (all runs of a tuple appended in one file). Lean: per-tuple `records.jsonl` (fewer inodes; natural "history of this tuple"); the Accounting Matrix picks the current-SHA record.
3. Scope: migrate the full committed `evals/results/` history, or only the canonical/final-pin sets the coverage report consumes? Lean: the sets the report consumes (avoids dragging stale 33/33-blocked incident runs into the clean tree).

**Recommendation:** layout above, **keep raw run dirs + add enum-keyed `coverage/` tree**, per-tuple `records.jsonl`, migrate the report-consumed sets. Hold for David's sign-off before any move.

## Alternatives considered
- **Flat hand-curated cell list (status quo)** — rejected: silent gaps, no accounting, compile/runtime confusion.
- **Profile enum in motlie-models + per-bundle applicability table (v1)** — superseded: leaked the eval-runtime Profile into the model layer and hand-authored per-bundle what the backend already determines. The BackendKind declaration is the single source of EP truth; bundles inherit.
- **3-state taxonomy (Validated/NotApplicable/Gap)** — rejected: cannot distinguish "can't ever" from "this build didn't compile it." `BuildGap` is the state that makes the #513 ASR-on-aarch64-CUDA class legible and actionable.
- **Auto-run every bundle on every profile** — rejected: wastes runs on impossible combos; no declared rationale.

## §10 Open questions for @onto-rv
1. `Accelerator`/`AccelSupport`/`Reason` home: `motlie-model` (next to `BackendKind`) — agreed?
2. `accel_support` shape: single `(accel)->AccelSupport` fn (above) vs a const table per BackendKind. Lean: fn (feature-gated arms read naturally).
3. Feature-name strings in `AccelSupport::Targetable{feature}` are the one stringly bit — they must match the cargo feature recorded in `runtime.cargo_features`. Acceptable, or wrap in a typed feature id?
4. Data migration: §9 layout + the 3 sub-questions (keep run dirs / per-tuple file / scope).
5. Snapshot cells: validate the hand-listed TOML against the ontology this PR vs generate from `advertised-caps × profiles`. Lean: validate-now, generate-later.

## In-flight gaps to fill (data-driven; separate from this schema PR)
- **kroko-2025 + qwen3-tts on CUDA** — currently `Gap`; run on the dgx (kroko via static-CUDA-ORT #513; qwen3-tts via ggml-CUDA) so the GPU-vs-CPU tradeoff is decided by data. (Standing finding: small ORT audio sees ~no CUDA speedup; LLMs do.)

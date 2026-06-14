# Eval Coverage Ontology — DESIGN

## Changelog
- 2026-06-14 @ops48-orchestrator — initial cut. (bundle, capability, accelerator) tuple + full-accounting rule. Outgrowth of #399/#486/#513/#514/#518.
- 2026-06-14 @onto-impl — v1: accelerator→Profile rename + enum schema.
- 2026-06-14 @onto-impl — v2: David-approved two-layer (compile vs runtime) design; declaration on `BackendKind`; 4-state taxonomy.
- 2026-06-14 @onto-impl — **v3, locked decisions (David sign-off via @ops48).** Q1: reuse `motlie_model::CapabilityKind`; new shared `Accelerator`/`AccelSupport`/`Reason` in **motlie-model** (contracts crate) next to `BackendKind`+`CapabilityKind`; `CuratedBundle` stays in motlie-models. Q2: `AccelSupport` **declared** from backend intrinsic capability, runtime **reconciles** (declaration↔runtime disagreement = a flagged finding — the framework's value). Q3: backend defaults + per-bundle overrides, completeness test fail-closed. Q4: immutable raw run-dirs stay source of truth + a **deterministically-generated** enum-keyed view (byte-stable regen, no record edits). Q5: validate hand-listed TOML now, generate cells later. Refinement: **`BuildGap` keys on recorded native-provider evidence** (EP probe / static-archive / `resolved_accelerator`), not cargo-feature strings. **Open finding (§E): `CapabilityKind` has no `Perf` and no serde — perf is modeled as a scenario over Chat.**
- 2026-06-14 @onto-impl — **v4, scope additions (David, required before main merge).** (1) **Naming consistency** across all dimensions — one canonical closed label set per dimension, code==evals: `bundle_id` (#518), `capability` (`CapabilityKind`), and now a **closed `Profile` enum/registry in bins/evals** (canonical labels, no raw-string drift) + a closed canonical **`Quant`** label set. (2) **Quant is now a first-class dimension** — the coverage tuple is **(bundle_id, quant, capability, Profile)**, sourced from the existing record `quantization` field; `bundle_id ↔ quant` stays **1:1** for now (do not re-split #518 ids), so quant rides with bundle and does not enlarge the completeness space; decoupling for same-model-multi-quant is a **flagged future option**. (3) **Slice-and-dice** — the generated index must be first-class queryable by any dimension + cross-cuts (by model / quant / capability / Profile). Applicability + completeness are **unaffected by quant**; the view/matrix gain a quant axis/filter (§9, §F).

## Problem
Curated-eval coverage is a flat, hand-curated cell list. "Does bundle X support capability Y on profile Z?" is answerable only by *absence* — a missing cell conflates "not applicable" / "build didn't provision it" / "we forgot." This yields silent gaps, deployment surprises (ASR-on-GPU), and **compile-vs-runtime confusion**: a `blocked` aarch64-CUDA-ORT cell (pre-#513) looked identical to a permanently-impossible ORT-on-Metal cell, though one is buildable and one never is.

## Key insight: a product space with a two-layer truth
Coverage = tuples **(CuratedBundle, Quant, Capability, Profile)**; completeness = every (sparse) tuple explicitly classified. Because `bundle_id ↔ quant` is **1:1** today (the quant is baked into each #518 bundle variant, e.g. `gemma4_12b_qat_q4_0_gguf` = `q4_0`), `Quant` is a denormalized, sliceable label that **rides with the bundle** — it does not multiply the completeness space. *What* classifies a tuple has **two layers** that must not be conflated:
- **Compile-time / model truth (motlie-model / motlie-models):** what a model's backend execution provider can *ever* target. Permanent. Declared on **`BackendKind`**, inherited by bundles, with per-bundle overrides.
- **Runtime / build truth (bins/evals):** what *this build* (active cargo features, git SHA, recorded native-provider evidence), on *this Profile*, actually instantiated and resolved.

The reconciliation of the two layers — **declaration must agree with runtime, and any disagreement is a flagged finding** — is the framework's core value.

### Dimensions — four canonical closed label sets (no freetext; code==evals)
Every dimension is a single canonical closed set, identical in code and in eval data (David's naming-consistency rule). Parse strings to the typed value at the data boundary; stay typed internally.
- **bundle**: `CuratedBundle` (18, `CANONICAL_IDS`, #518) — motlie-models.
- **quant**: a closed canonical `Quant` label set in `bins/evals` — `{default, q4_k_m, q4_0, q8_0}` (extensible), sourced from the existing record `quantization` field. `CANONICAL_QUANTS` + round-trip test (mirrors #518). 1:1 with `bundle_id` today (a flagged future option decouples them for same-model-multi-quant).
- **capability**: **`motlie_model::CapabilityKind`** (`libs/model/src/lib.rs:156`). Eval-relevant subset `{Chat, ToolUse, Transcription(=asr), Speech(=tts), Embeddings}`; advertised set read from `descriptor().capabilities` — no hand-listing. (`perf` is a scenario over Chat, not a `CapabilityKind`; §E.)
- **Profile**: a **closed `Profile` enum/registry in `bins/evals`** (the runtime layer) — canonical labels `{local-cpu-x86_64, local-cpu-aarch64, apple-metal, dgx-spark, cuda-workstation}`, `CANONICAL_PROFILES` + `from_id`/`canonical_id`/`arch`/`accel` + round-trip test. Replaces the raw profile strings drifting through `snapshot.rs`/`accelerator.rs` (parse at the edges). **No Profile enum in motlie-models** — Profile is an eval concern; the model layer only knows `Accelerator`, reached via the `accelerator.rs` `Profile→AcceleratorClass` bridge (§B).

## §A Compile-time layer — accelerator capability on `BackendKind` (motlie-model)
```rust
// motlie-model, next to BackendKind + CapabilityKind.
pub enum Accelerator { Cpu, Cuda, Metal }       // resolution states (Any/Unavailable) are runtime-only

pub enum AccelSupport {
    /// Backend's EP can target this accelerator. Compiled only when `feature` is active
    /// (None = always, e.g. Cpu). The feature gate informs, but does NOT alone decide,
    /// BuildGap — that keys on recorded native-provider evidence (§C, Refinement).
    Targetable { feature: Option<&'static str> },
    /// Backend fundamentally has no path here — permanent, build-independent.
    Unsupported(Reason),
}

#[non_exhaustive]
pub enum Reason { NoExecutionProviderForAccelerator, UpstreamNoGpuForwarding, CpuOnlyStaticArchive, RequiresArtifactGate /* … */ }

impl BackendKind {
    /// Backend-intrinsic default (Q2: declared from intrinsic capability, NOT from runs).
    pub fn accel_support(self, accel: Accelerator) -> AccelSupport;
}
```
**Q3 — defaults + per-bundle overrides.** The backend gives the default; a bundle may override for genuine per-bundle intrinsics (e.g. an artifact gate). Resolution: `applicability(bundle, accel) = bundle_override(bundle, accel).unwrap_or(bundle.backend().accel_support(accel))`. The completeness test (fail-closed) asserts every (bundle × advertised-cap × Profile) tuple resolves to a declared `AccelSupport` — no silent gap. Declared truth (feature-gated):
| BackendKind | Cpu | Cuda | Metal |
|---|---|---|---|
| `Ort` / `SherpaOnnx` | `Targetable{None}` | `Targetable{"…-cuda"}` *(static archive CPU-only today → runtime BuildGap, #495/#513)* | **`Unsupported(NoExecutionProviderForAccelerator)`** |
| `LlamaCpp` | `Targetable{None}` | `Targetable{"llama-cpp-cuda"}` | `Targetable{"metal"}` |
| `MistralRs` | `Targetable{None}` | `Targetable{"cuda"}` | `Unsupported(UpstreamNoGpuForwarding)` |
| `WhisperCpp` / `Qwen3TtsCpp` (ggml) | `Targetable{None}` | `Targetable{"…-cuda"}` | `Targetable{"metal"}` |

## §B Runtime layer — Profile is the eval instantiation (bins/evals)
Profile stays a string; the existing `accelerator.rs` `requested_for_profile(profile) -> AcceleratorClass` / `resolve_for_profile(profile, platform)` bridge **is the join** — it turns a runtime Profile into the `Accelerator` §A is keyed on. The evals `AcceleratorClass` (Cpu/Cuda/Metal/Any/Unavailable) maps onto the model-side `Accelerator` (Cpu/Cuda/Metal); Any/Unavailable are runtime-only resolution states.

## §C 4-state taxonomy + reconciliation (compile ↔ runtime; Q2 + Refinement)
For tuple `(bundle, capability, Profile)`, let `accel = bridge(Profile)`, `support = applicability(bundle, accel)`:
1. **`Validated`** — a `passed` record exists with `resolved_accelerator == accel`. Implies `support == Targetable`. Carries the metric.
2. **`NotApplicable(Reason)`** — `support == Unsupported(Reason)`. Compile-permanent (ORT-on-Metal). A runtime record here **must never `passed`** (that's a contradiction finding); a `blocked` must reconcile to this Reason.
3. **`BuildGap`** — `support == Targetable` but **this build didn't provide the native path**, judged by **recorded native-provider evidence** (Refinement): `accelerator.use_proof_source` / `backend_mode` (e.g. `cuda_execution_provider=unavailable;ort_build=cpu_only`), `platform.accelerator_metadata` EP/driver probe, and `resolved_accelerator != accel` despite request — corroborated by `runtime.cargo_features` + `git_sha`. The buildable counterpart of NotApplicable (#513 aarch64-CUDA-ORT). A *finding*: "capable, not yet built here."
4. **`Gap`** — `support == Targetable`, native path present, but the cell was never scheduled / no passing record (kroko-2025 + qwen3-tts on CUDA). Transient, CI-visible.

**Reconciliation (the framework's value), enforced by a fail-closed results-side test:**
- `passed` ⇒ tuple `Validated` ∧ `support == Targetable`. A `passed` on an `Unsupported` tuple ⇒ **contradiction finding** (declaration wrong, or accelerator misreported).
- `blocked`/`skipped` ⇒ `NotApplicable(r)` (with `OutcomeReason → r`) **or** `BuildGap` (Targetable + native evidence shows the path absent). A `blocked` that is neither ⇒ **finding**.
- undeclared tuple, or runtime reason contradicting §A ⇒ **finding**. Extends the #518 `CANONICAL_IDS` completeness test to the full tuple. Reviewer (@onto-rv) mutation-tests: missing-tuple, bad-tuple, contradictory-blocked all must fail CI.

## §D Coverage-report **Accounting Matrix**
New `evals report --aggregate` section (`report.rs`): rows = `bundle × capability` (advertised), cols = the 5 Profiles, cell = `✅ Validated (+metric)` / `⛔ NotApplicable (reason)` / `🔧 BuildGap (evidence)` / `⏳ Gap`. Joins §A (via §B bridge) with records. The **quant** label annotates each bundle row (1:1 today) and is a filter/group key on the underlying index (§F) — so a quant-sliced matrix (e.g. only `q4_0`) is the same renderer over a filtered index, not a separate code path. Because applicability is quant-independent and `bundle_id ↔ quant` is 1:1, the cell **state** is unchanged by the quant axis; quant only labels/filters.

## §E Open finding — `perf` and `CapabilityKind` (Q1 consequence)
`motlie_model::CapabilityKind` = `{Chat, Completion, Embeddings, Ocr, Speech, ToolUse, Transcription, Vision, VoiceClone}` — it has **no `Perf`** (a model doesn't "do perf"; perf is a *measurement scenario* over the Chat capability — `bench_chat_startup` already sets `bundle_filter.capability="chat"`), and **no serde/FromStr** (internal-only). Evals' `CapabilityName` additionally uses `asr`/`tts` names. So "drop `CapabilityName`, use `CapabilityKind`" (Q1) is not a 1:1 swap. **Resolution (this PR):**
- The **ontology axis = `CapabilityKind`**. The 5 real eval capabilities map 1:1 (`asr→Transcription`, `tts→Speech`, others identity) via the existing `bundle_filter_capability_kind()`.
- **`perf` is modeled as a scenario over `CapabilityKind::Chat`**, not a capability row. Its perf metrics surface inside the Chat cell of the Accounting Matrix. The runner dispatch keeps routing perf cells to `PerfRunner` by **scenario**, not by a capability variant.
- To honor "no duplicate enum," `CapabilityKind` gains serde (snake_case) + a `from_eval_token`/`as_eval_token` bridge for the `asr`/`tts` spellings; evals' `CapabilityName` is removed in favor of it. The record's stored `capability` token is mapped to `CapabilityKind` (perf→Chat) when building the ontology view, so **immutable committed records are not rewritten**.
- ⚠️ Flagged for @onto-rv / @ops48: this perf-as-scenario reframing + the `asr/tts` spelling bridge are the substantive interpretation of Q1; confirm or redirect.

## §9 Data — generated enum-keyed view + slice-and-dice index (Q4; no record edits)
Immutable raw run-dirs (`evals/results/curated-v2-smoke/<run-dir>/`, chrono/ID-named, #518) **stay the source of truth**. We **generate** (not relocate) a deterministic, byte-stable, enum-keyed view over them:
```
evals/results/coverage/<bundle_id>/<quant>/<capability>/<profile>/records.jsonl   # per-tuple, append-ordered
evals/results/coverage/INDEX.json                                                 # see §F
```
- Path components are **validated against the four canonical sets** (`CuratedBundle::CANONICAL_IDS`, `CANONICAL_QUANTS`, `CapabilityKind`, `CANONICAL_PROFILES`) — an unparseable path fails. This *is* the enforced results↔enums mapping.
- **Deterministic regen** (David's constraint): a stable sort key `(bundle_id, quant, capability, profile, git_sha, host_slug, run_ts, cell_id)` over the raw records → byte-identical output every run; record content copied verbatim (never edited). A CI check asserts `regenerate == committed` (the view can't drift from the raw records).
- **Future-run reliability:** records already carry the machine-readable identity needed (`coverage.{bundle_id,quantization,capability,profile,resolved_accelerator}`, `accelerator.*`, `runtime.cargo_features`, `git_sha`); the enforcement test guarantees new runs stay parseable into the tuple.

## §F Slice-and-dice index (queryable by any dimension + cross-cuts)
`INDEX.json` is the first-class query surface (not ad-hoc grep). One flat array of typed cell entries, each:
```jsonc
{ "bundle_id": "...", "quant": "...", "capability": "...", "profile": "...",
  "state": "validated|not_applicable|build_gap|gap", "reason": "…?", "metric": {…}?,
  "git_sha": "...", "host_slug": "...", "run_ts": ..., "source_run_dir": "..." }
```
- **Sliceable by any single dimension** (all `profile=="dgx-spark"`, all `quant=="q4_0"`, all `capability=="asr"`, all `bundle_id=="qwen3_4b_gguf"`) **and any combination/cross-cut** (`capability=="chat" & profile.accel=="cuda"`), because every dimension is a typed field on every entry.
- The Accounting Matrix (§D) is one *view* over this index (pivot bundle×capability × Profile); other pivots (e.g. quant×Profile, capability×Profile) fall out of the same index. A small `evals coverage query --bundle … --quant … --capability … --profile …` selector reads `INDEX.json` and returns the matching cells — the canonical "slice-and-dice" entry point.
- Deterministic + byte-stable like §9 (stable field order, stable entry sort).

## §10 Status of decisions
Q1–Q5 **locked** (above). Live items for @onto-rv: (a) §E perf-as-scenario + asr/tts spelling bridge; (b) `accel_support` fn-vs-const-table; (c) the per-bundle-override set (which bundles genuinely deviate from their backend default — to be enumerated as I wire it). Mutation-test targets for the fail-closed test: missing-tuple, bad-tuple, contradictory-blocked, passed-on-Unsupported.

## In-flight gaps to fill (data-driven; after the schema lands)
- **kroko-2025 + qwen3-tts on CUDA** — currently `Gap`; run on the dgx (kroko via static-CUDA-ORT #513; qwen3-tts via ggml-CUDA) to decide GPU-vs-CPU by data. (Standing finding: small ORT audio sees ~no CUDA speedup; LLMs do.)

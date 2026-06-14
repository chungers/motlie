# Eval Coverage Ontology — DESIGN

## Changelog
- 2026-06-14 @ops48-orchestrator — initial cut. Captures the (bundle, capability, accelerator) coverage tuple + full-accounting rule. Outgrowth of #399/#486/#513/#514/#518: silent coverage gaps (kokoro, qwen3-tts, kroko-2025 missed) and the "ASR-on-GPU" surprise class.
- 2026-06-14 @onto-impl — finalize for #521. Renamed the third dimension **accelerator → Profile** throughout (Profile encodes arch+accelerator; one axis, not two). Grounded the schema in the real motlie-models type system (`CuratedBundle`, `Capabilities`/`CapabilityKind`, new `Profile`/`Capability`/`Reason` enums). Added the concrete applicability API, the runtime↔declared reconciliation rule, the enforcement-test plan, and a **Dataset Reorg proposal** (§9, derived index over immutable run dirs). Marked open questions for @onto-rv review. **No code/data edited yet — this doc is the review gate.**

## Problem
Curated-eval coverage is a flat, hand-curated cell list. "Does bundle X support capability Y on profile Z?" is answerable only by *absence* — a missing cell is indistinguishable from "not applicable" vs "we forgot." This causes (a) silent test gaps (models defined but never evaluated) and (b) deployment disappointments (e.g. picking an ASR model for GPU when it has no GPU backend). No single artifact accounts for the full space.

## Key insight: coverage is a product space
Coverage is the set of tuples **(CuratedBundle, Capability, Profile)**. Completeness = **every tuple in the (sparse) space is explicitly classified** — none undeclared.

### Dimensions & cardinality (all keyed to motlie-models enums — no freetext)
- **bundle**: `CuratedBundle` enum — **18** variants (the registry; canonical `bundle_id` strings, #518). `libs/models/src/lib.rs`.
- **capability**: `Capability` enum — the 6 eval-matrix categories `Chat | ToolUse | Asr | Tts | Embeddings | Perf`. *New canonical enum* (see §8.2). A bundle's **advertised** capability set is **derived** from its existing `descriptor().capabilities` (`CapabilityKind`), not hand-listed. `perf`/`tool_use` cross-cut text-generation bundles.
- **Profile**: `Profile` enum — the eval profile = (arch, accelerator), **one axis, not two**. *New enum* (see §8.1). 5 variants: `local-cpu-x86_64`, `local-cpu-aarch64`, `apple-metal` (arm64+metal), `dgx-spark` (aarch64+cuda, GB10), `cuda-workstation` (x86_64+cuda). `aarch64`==`arm64`. Today these are bare strings in `bins/evals` (`accelerator.rs`, `snapshot.rs`); the ontology promotes them to a typed enum.

The space is **sparse**: pruned by each bundle's *advertised* capabilities (a chat model has no `asr` row). Real cardinality ≈ `Σ_bundle (advertised-caps × 5 profiles)`, far below `18 × 6 × 5`.

### Cell state (each tuple is exactly one)
- **`Validated`** — produced a `passed` eval cell (carries the metric).
- **`NotApplicable(Reason)`** — declared can't/won't run, with a typed reason. *Full accounting, not a silent miss.*
- **`Gap`** — known-missing, must be filled. Transient, CI-visible.

### Reason taxonomy (typed, queryable — not freetext)
A new `Reason` enum in motlie-models. Initial variants (extensible `#[non_exhaustive]`):
`NoBackendForProfile` (no backend compiled/exists for this profile), `UpstreamNoGpuForwarding` (mistralrs→CUDA/Metal for the bundles that don't forward), `CpuOnlyStaticArchive` (ORT audio: static archive has no GPU EP, #495/#513), `NoGpuPath` (embeddings: no accelerated path advertised), `RequiresArtifactGate` (gated artifact / token), … . Distinct from the *runtime* `OutcomeReason` (§8.4).

## Enforcement
A CI test asserts: for every **(bundle × advertised-capability × Profile)** tuple, the declared applicability is `Supported` **or** `NotApplicable(Reason)` — the applicability function is **total** (no undeclared tuple). A second, results-side test asserts the *reconciliation* (§8.4): every `passed` record sits on a `Supported` tuple, and every `blocked` record's runtime `OutcomeReason` maps to the tuple's declared `NotApplicable(Reason)` — an unexplained `blocked`, or a `blocked` whose reason contradicts the declaration, **fails CI**. Extends the #518 `CANONICAL_IDS` completeness test from the bundle axis to the full tuple. → No silent fails or deployment surprises.

## High-level data flow
```
CuratedBundle.descriptor().capabilities   ──derive──>  advertised Capability set   ┐
Profile::ALL (5)                                                                    ├─> applicability(bundle, cap, profile) -> Supported | NotApplicable(Reason)
per-bundle applicability overrides (declared)                                       ┘            │
                                                                                                 ▼
results.jsonl records (coverage.{bundle_id,capability,profile}, terminal_outcome, reason) ──reconcile──> CoverageState{Validated|NotApplicable|Gap}
                                                                                                 │
                                                                                                 ▼
                                                                          evals report --aggregate  →  ## Accounting Matrix (§8.5)
```

## §8 Schema design (concrete, enum-grounded)

### 8.1 `Profile` enum — `libs/models/src/profile.rs` (new)
```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Profile { LocalCpuX86_64, LocalCpuAarch64, AppleMetal, DgxSpark, CudaWorkstation }

pub enum Arch { X86_64, Aarch64 }
pub enum Accel { Cpu, Cuda, Metal }

impl Profile {
    pub const ALL: [Profile; 5] = [ /* … */ ];
    pub const CANONICAL_IDS: [&'static str; 5] = ["local-cpu-x86_64", "local-cpu-aarch64", "apple-metal", "dgx-spark", "cuda-workstation"];
    pub fn canonical_id(self) -> &'static str;        // matches existing profile strings exactly
    pub fn from_id(id: &str) -> Option<Profile>;      // string -> enum at the snapshot/result boundary
    pub fn arch(self) -> Arch;
    pub fn accel(self) -> Accel;
}
```
*Home = motlie-models* so the applicability table (model layer) can key on it. Mirrors the `CuratedBundle::canonical_id`/`CANONICAL_IDS` pattern, incl. a completeness test. `bins/evals` (`accelerator.rs`, `snapshot.rs`, `driver.rs`) parse strings into `Profile` at the edges and stay enum-keyed internally; existing profile strings are unchanged on disk.

### 8.2 `Capability` enum (the matrix axis) — motlie-models (new)
The 6 eval-matrix categories: `Chat | ToolUse | Asr | Tts | Embeddings | Perf`. The **advertised** set per bundle is **derived** from existing metadata — no new hand-authored list:
| Capability | derived from `descriptor().capabilities.supports(..)` |
|---|---|
| Chat | `CapabilityKind::Chat` |
| ToolUse | `CapabilityKind::ToolUse` |
| Asr | `CapabilityKind::Transcription` |
| Tts | `CapabilityKind::Speech` |
| Embeddings | `CapabilityKind::Embeddings` |
| Perf | derived: any bundle advertising `Chat` (perf-bench cross-cuts text generation) |

`bins/evals` currently has its own `CapabilityName` (chat/tool_use/asr/tts/embeddings/perf) — **identical set**. Proposal: make `Capability` the single source of truth in motlie-models and have evals re-export it as `CapabilityName` (or replace). This is the main refactor surface — flagged for review (§10).

### 8.3 Applicability table — `libs/models/src/applicability.rs` (new)
```rust
pub enum Applicability { Supported, NotApplicable(Reason) }

/// Total over (advertised cap × Profile). Per-BUNDLE, not per-backend:
/// e.g. qwen3_4b (mistralrs) Chat@cuda = Supported (forwards, #515) while
/// gemma4_e2b (mistralrs) Chat@cuda = NotApplicable(UpstreamNoGpuForwarding).
pub fn applicability(bundle: CuratedBundle, cap: Capability, profile: Profile) -> Applicability;
```
Implemented as: **sensible defaults** (CPU profiles → `Supported` for every advertised cap; embeddings on any GPU → `NotApplicable(NoGpuPath)`; ORT-audio on metal/cuda → `NotApplicable(CpuOnlyStaticArchive)`) **+ per-bundle overrides** for the backend-forwarding nuances. Each override carries the `Reason`. The function is total → the enforcement test (§Enforcement) just asserts it returns for every tuple, which the type system already guarantees; the test's real job is asserting it agrees with the *records* (§8.4).

### 8.4 Runtime ↔ declared reconciliation
`bins/evals` records already carry a runtime `OutcomeReason` (28 variants, `result.rs`). We add a mapping `OutcomeReason -> Option<Reason>` (a record's runtime block reason → the declared family). Reconciliation rule, enforced by a results-side test in `bins/evals`:
- `passed` record  ⇒ its tuple's `applicability == Supported`.
- `blocked`/`skipped` record ⇒ tuple is `NotApplicable(r)` **and** the record's `OutcomeReason` maps to `r` (or to a compatible family).
- any record whose tuple is **undeclared**, or whose runtime reason contradicts the declaration ⇒ test failure.

### 8.5 Coverage-report **Accounting Matrix**
New section in `evals report --aggregate` (`bins/evals/src/report.rs`): rows = `bundle × capability` (advertised only), cols = the 5 `Profile`s, cell = `✅ validated (+metric)` / `⛔ N/A (reason)` / `⏳ gap`. The model-selection + no-surprise view; generated from the applicability table joined with records (not hand-authored).

## §9 Dataset Reorg — PROPOSAL (review before any data move; #518 discipline)
**Constraint (RUNBOOK + #518):** run-data dirs are chrono/ID-named *by design* — immutable records whose name **is** their identity (`ts-pid-SHA-host-arch-accel`). We must **not** silently move/rewrite committed records.

**Option A (RECOMMENDED) — derived enum-keyed index over immutable run dirs.**
Keep every `results.jsonl` exactly where it is. Add a generated, committed **coverage index** `evals/results/coverage-index.json` (and the Accounting Matrix view) keyed by `(CuratedBundle, Capability, Profile)` → `{ state, reason?, metric?, source_run_dir, git_sha }`. The index is *derived* (regenerated by `evals report`), so it is reproducible and never the source of truth; raw records stay immutable and provenance-preserving. "Reorg" = introduce the **enum-keyed selection/index layer**, not physically relocating data.
- Pros: zero rewrite of committed data; aligns with #518/RUNBOOK; the (bundle,cap,profile) lookup the issue asks for; trivially regenerated.
- Cons: the on-disk run dirs remain chrono-named (you still need the index to navigate by tuple) — acceptable, since the index *is* the navigation layer.

**Option B — physical tree `results/<bundle>/<capability>/<profile>/…`.**
Relocate/duplicate records into an enum-named tree.
- Pros: filesystem browsing by tuple.
- Cons: rewrites immutable identity-named records (breaks #518/RUNBOOK provenance); a tuple can have *many* runs over time (which one lives at the path?); merge/aggregation churn. **Rejected unless the reviewer wants it.**

**Recommendation:** Option A. Decision requested from @onto-rv / @ops48 before I touch any committed `results/`.

## Alternatives considered (ontology shape)
- **Flat hand-curated cell list (status quo)** — rejected: silent gaps, no accounting, the surprise class.
- **Auto-run every bundle on every profile** — rejected: wastes runs on impossible combos + still no *declared* rationale (a blocked cell stays ambiguous).
- **Profile as two axes (arch × accelerator)** — rejected per David: arch and accelerator are coupled (profile is the deployable unit); one `Profile` axis keeps the matrix legible and matches the existing profile registry.

## §10 Open questions for @onto-rv (settle before implementation)
1. **Capability home/refactor:** OK to promote a single `Capability` enum into motlie-models and have `bins/evals` re-export it (replacing the standalone `CapabilityName`)? Or keep `CapabilityName` in evals and add a `From<Capability>` bridge (less churn, two near-identical enums)?
2. **`Profile` home:** motlie-models (so applicability can key on it) — agreed? Or a smaller shared crate?
3. **Applicability authoring:** defaults + per-bundle overrides (§8.3) vs a fully explicit per-bundle table. Defaults reduce boilerplate but hide intent; explicit is verbose but auditable. Lean: defaults + overrides, with the enforcement test pinning every override against the records.
4. **Dataset reorg:** Option A (derived index) vs B (physical tree) — §9. Lean A.
5. **Snapshot generation:** derive the cell list from `advertised-caps × profiles × applicability` (closes silent-miss structurally) now, or keep the hand-listed TOML and only *validate* it against the ontology this PR (smaller blast radius)? Lean: validate-now, generate-later.

## In-flight gaps to fill (data-driven, not assumed; separate from this schema PR)
- **kroko-2025 + qwen3-tts on CUDA** — currently `Gap`; run on the dgx (kroko via static-CUDA-ORT #513; qwen3-tts via ggml-CUDA) so the GPU-vs-CPU tradeoff is decided by data. (Standing finding so far: small ORT audio sees ~no CUDA speedup; LLMs do.)

# Distributed Eval Runbook — evals/2026-06-infra

Live operational log of the first distributed eval exercise (issue #399 v2). Documents process steps **as executed**, gotchas, and open issues, so the next round is repeatable. Entries carry datetime + self-identifier. Final version ships in the coverage-report PR (David's direction, 2026-06-10).

## Decisions (David, 2026-06-10)
- Hosts: **amd1 (x86 CPU), dgx/local (CUDA GB10), mac1 (Apple Metal)** — amd2 excluded.
- Pin: **all runners use the latest of `evals/2026-06-infra`** at kickoff = `e722d23d` (post #424 merge).
- HF token: env-only at run time, SSH-provisioned by the orchestrator; never committed/logged; results record only `HF_TOKEN_PRESENT`.
- Coordination substrate: this branch. Per-host results PRs -> branch; orchestrator aggregates -> coverage-report PR; single final merge to main (David).

## Process steps (as executed)
1. `@ops48-orchestrator 2026-06-10 PDT` — Merged framework PR #424 (`43c6ce39`, all-3-approved x86/CUDA/Metal) into `evals/2026-06-infra` -> head `e722d23d`. Seeded this runbook.
2. _(pending)_ HF_TOKEN provisioning via SSH-written env file outside the repo checkout (`<work-root>/issue-399-eval-suite/.hf-env`, chmod 600); runners `source` it before `evals matrix`.
3. _(pending)_ Per-host runs: checkout `e722d23d`, `cargo run -p evals -- matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile <host profile>`, collision-free results dir, commit `results.jsonl` + `summary.md` + manifest, per-host results PR -> this branch.
4. _(pending)_ Orchestrator merges results PRs, runs `evals report --aggregate --snapshot ...`, opens coverage-report PR (includes final runbook).

## Gotchas
- (from pre-run reviews) `evals matrix` parent binary must be feature-light aware: preflight cache mapping is catalog-feature-gated; uncached gated bundles can mis-preflight if the parent lacks the bundle's features (fast-follow noted in #424 review).
- macOS: no global `[env] BINDGEN_EXTRA_CLANG_ARGS` (breaks llama-cpp-sys bindgen); Linux GGUF bindgen flows through driver child-build wiring + `tools/clang-compat/include` shims.
- mistralrs cells on `apple-metal` record honest `blocked: accelerator_mismatch` (libs/models `metal = []` non-forwarding + upstream candle-metal M4 threadgroup limit) — backend-support gap, NOT a framework failure (see Platform Notes in the aggregate).

## Failure & incompatibility tracking (David's directive, 2026-06-10)
Curated bundles are LOCAL models meant for CPU inference — failures and budget-exceeds are surprising and must never be lost. Capture structure:
- **GitHub issue #435** = umbrella tracker for this cycle; one **sub-issue per failing/budget-exceeded model** (model, environment, platform profile, capability, exact results.jsonl record, child log, repro command) — filed by the orchestrator from merged per-cell records.
- This RUNBOOK = process narrative + gotchas + links; the per-cell JSONL on this branch = raw evidence.
- v2 driver PRE-FLIGHT checklist lives on #435 (token presence, artifact-cache/provision, child build profile release-vs-debug, submodules) — so we stop discovering missing prerequisites mid-run.

## Gotchas (live, from the run)
- `@ops48-orchestrator 2026-06-10 PDT` — **"Run complete" != coverage.** dgx posted completion with token-clean PRs, but both result sets were 33/33 blocked, 0 passed (token env not sourced -> artifact_unauthorized; empty artifact cache -> child_run_failed). **Merge gate added: inspect the outcome mix (passed/skipped/blocked + reasons) BEFORE merging a results PR; a 100%-blocked set goes back for diagnosis + re-run.**
- `@ops48-orchestrator 2026-06-10 PDT` — **Debug child builds distort CPU wall-time.** Runners' matrix children built in debug; long chat/bench cells hit runtime_budget_exceeded on x86/Metal-CPU paths. Suspect debug-vs-release as primary factor; v2 preflight should pin release children for perf-bearing cells.
- `@ops48-orchestrator 2026-06-10 PDT` — Runner discipline: verify HF_TOKEN set in the launching shell (test -n, never print) BEFORE matrix start; codex runners need explicit periodic-progress instruction (~15min posts to the run Discussion); claude runners idle between phases and need keep-driving nudges.

## Open issues
- Umbrella tracker: #435 (sub-issues filed as runs land)

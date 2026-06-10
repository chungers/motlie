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

## Open issues
- (running list; file issues as encountered)

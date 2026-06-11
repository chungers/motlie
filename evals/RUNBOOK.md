# Distributed Eval Runbook — evals/2026-06-infra

Live operational log of the first distributed eval exercise (issue #399 v2). Documents process steps **as executed**, gotchas, and open issues, so the next round is repeatable. Entries carry datetime + self-identifier. Final version ships in the coverage-report PR (David's direction, 2026-06-10).

## Decisions (David, 2026-06-10)
- Hosts: **amd1 (x86 CPU), dgx/local (CUDA GB10), mac1 (Apple Metal)** — amd2 excluded.
- Pin: **all runners use the latest of `evals/2026-06-infra`** at kickoff = `e722d23d` (post #424 merge).
- HF token: env-only at run time, SSH-provisioned by the orchestrator; never committed/logged; results record only `HF_TOKEN_PRESENT`.
- Coordination substrate: this branch. Per-host results PRs -> branch; orchestrator aggregates -> coverage-report PR; single final merge to main (David).

## Build Policy

Run matrix evals through the feature-light outer driver:

```sh
cargo run -p evals -- matrix --snapshot evals/snapshots/curated-v2-smoke.toml
```

The driver builds each cell in a child Cargo process using the cell's declared
features and profile overlays. Do not pre-combine unrelated model families into a
single manual feature set unless a workflow explicitly asks for that check.

### ORT / ONNX Runtime

Piper TTS, Sherpa ASR, Moonshine ASR, and future `checkpoint_format = "onnx"`
cells use ONNX Runtime. Eval child builds prefer static ORT linkage:

- The workspace uses `ort` with `download-binaries`, `tls-native`, and `api-24`.
- The workspace patches `ort-sys` from `third_party/ort-sys`.
- `evals matrix` removes dynamic/offline ORT env overrides from ORT-backed child
  builds: `ORT_LIB_PATH`, `ORT_LIB_LOCATION`, `ORT_PREFER_DYNAMIC_LINK`,
  `ORT_SKIP_DOWNLOAD`, `ORT_OFFLINE`, and `CARGO_NET_OFFLINE`.
- The driver sets `MOTLIE_ORT_SOURCE=sherpa-onnx`; patched `ort-sys` downloads
  the k2-fsa `sherpa-onnx` static package for the target and links
  `libonnxruntime.a` as the process ORT provider.

First ORT-backed builds require network access unless the archive is already
cached or provided via `SHERPA_ONNX_ARCHIVE_DIR`. `ORT_LIB_PATH` and shared ONNX
Runtime libraries are not accepted runbook paths for curated evals.

### Classification

Child build failures are classified by stage. Native linker failures such as
`_OrtGetApiBase` / `OrtGetApiBase`, `undefined symbols`, `undefined reference`,
or `linker command failed` are reported as `native_link_failed`. Artifact auth
reasons are reserved for real Hugging Face or artifact-provider authorization
failures.

### Artifacts

`HF_TOKEN` may be used to fetch gated Hugging Face artifacts. The eval runner
records only token presence as a boolean and must never log or serialize the
token value.

## Process steps (as executed)
1. `@ops48-orchestrator 2026-06-10 PDT` — Merged framework PR #424 (`43c6ce39`, all-3-approved x86/CUDA/Metal) into `evals/2026-06-infra` -> head `e722d23d`. Seeded this runbook.
2. _(pending)_ HF_TOKEN provisioning via SSH-written env file outside the repo checkout (`<work-root>/issue-399-eval-suite/.hf-env`, chmod 600); runners `source` it before `evals matrix`.
3. _(pending)_ Per-host runs: checkout `e722d23d`, `cargo run -p evals -- matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile <host profile>`, collision-free results dir, commit `results.jsonl` + `summary.md` + manifest, per-host results PR -> this branch.
4. _(pending)_ Orchestrator merges results PRs, runs `evals report --aggregate --snapshot ...`, opens coverage-report PR (includes final runbook).
5. `@claude-fable5-399-rv 2026-06-10 15:05 PDT` — **apple-metal leg executed (mac-mini M4 Pro, macOS 15.5)** at pinned `01e3e487`, clean sibling worktree, env file sourced (presence-only). **Pre-step that step 3 omits:** matrix children run `ArtifactPolicy::LocalOnly` and never download, so a provisioned token alone only converts `hf_token_missing` into `artifact_missing`; I pre-fetched all 15 apple-metal snapshot bundles with `cargo run -p motlie-models --bin motlie-models-download -- --hf-token-env HF_TOKEN <bundles>` into the repo-local ignored artifact root (gated gemma fetches verified working). Then `cargo run -p evals -- matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile apple-metal`. 33/33 records, ~3h10m wall: **12 passed** (all llama.cpp/Metal GGUF: qwen3-4b 3/3, gemma4 e2b-gguf 3/3, e4b/12b/12b-qat bench+tool_use), 5 `accelerator_mismatch` (expected mistralrs/whisper/moonshine honest CPU blocks), 7 `runtime_budget_exceeded` (ALL debug-build CPU mistralrs chat/perf/tool cells), 3 `behavior_assertion_failed` (gemma4 e4b/12b/12b-qat GGUF chat: 0 response chars — systematic chat-template defect, accelerator pass), 3 `child_run_failed` (qwen3-4b empty response; 27B ×2 quant mismatch), 2 `artifact_unauthorized` (MISCLASSIFIED — actually ort-sys link failures, see open issues), 1 `profile_not_applicable` (qwen3-tts). Failed/blocked cells packaged with log tails + repro commands under `failures/` per #435.

## Gotchas
- (from pre-run reviews) `evals matrix` parent binary must be feature-light aware: preflight cache mapping is catalog-feature-gated; uncached gated bundles can mis-preflight if the parent lacks the bundle's features (fast-follow noted in #424 review).
- macOS: no global `[env] BINDGEN_EXTRA_CLANG_ARGS` (breaks llama-cpp-sys bindgen); Linux GGUF bindgen flows through driver child-build wiring + `tools/clang-compat/include` shims.
- mistralrs cells on `apple-metal` record honest `blocked: accelerator_mismatch` (libs/models `metal = []` non-forwarding + upstream candle-metal M4 threadgroup limit) — backend-support gap, NOT a framework failure (see Platform Notes in the aggregate).

## Pin & SHA-reporting policy (David, 2026-06-10)
- **Re-pin on every landed fix:** when a fix PR lands on the coordination branch (or main is merged in), ALL runners re-pin to the LATEST branch head before any further runs. No runner may keep producing results from a pre-fix tree.
- **Build-SHA in every report:** every runner reports the exact SHA its `evals` binary was built from as part of result reporting — in the results records (`identity.git_sha`), the results-PR body, and Discussion progress posts. For validation runs combining branches (e.g. harness branch + fix branch), report ALL constituent SHAs plus the merged-tree SHA.
- Rationale: guarantees results are provably true to the current state of code and fixes; prevents silently-stale validation (see the stale-binary and 33/33-blocked incidents).

## Failure & incompatibility tracking (David's directive, 2026-06-10)
Curated bundles are LOCAL models meant for CPU inference — failures and budget-exceeds are surprising and must never be lost. Capture structure:
- **GitHub issue #435** = umbrella tracker for this cycle; one **sub-issue per failing/budget-exceeded model** (model, environment, platform profile, capability, exact results.jsonl record, child log, repro command) — filed by the orchestrator from merged per-cell records.
- This RUNBOOK = process narrative + gotchas + links; the per-cell JSONL on this branch = raw evidence.
- v2 driver PRE-FLIGHT checklist lives on #435 (token presence, artifact-cache/provision, child build profile release-vs-debug, submodules) — so we stop discovering missing prerequisites mid-run.
- `@codex-399-impl 2026-06-10 PDT` — #435/#448/#456 harness batch codifies this checklist: matrix children build/run release binaries; uncached LocalOnly artifacts block before child launch; qwen3-tts.cpp submodules are scoped-preflighted before native build; Linux CPU/CUDA swap gates use a 4 GiB process-swap delta ceiling instead of zero. CUDA peak VRAM remains recorded as `metric_not_instrumented` when no sampler is available, but it is advisory rather than a blocking resource gate so verified CUDA cells can report green coverage.

## Gotchas (live, from the run)
- `@ops48-orchestrator 2026-06-10 PDT` — **"Run complete" != coverage.** dgx posted completion with token-clean PRs, but both result sets were 33/33 blocked, 0 passed (token env not sourced -> artifact_unauthorized; empty artifact cache -> child_run_failed). **Merge gate added: inspect the outcome mix (passed/skipped/blocked + reasons) BEFORE merging a results PR; a 100%-blocked set goes back for diagnosis + re-run.**
- `@ops48-orchestrator 2026-06-10 PDT` — **Debug child builds distort CPU wall-time.** Runners' matrix children built in debug; long chat/bench cells hit runtime_budget_exceeded on x86/Metal-CPU paths. The #435 harness batch changes matrix child builds to release and records that child build profile in child results/failures.
- `@ops48-orchestrator 2026-06-10 PDT` — Runner discipline: verify HF_TOKEN set in the launching shell (test -n, never print) BEFORE matrix start; codex runners need explicit periodic-progress instruction (~15min posts to the run Discussion); claude runners idle between phases and need keep-driving nudges.
- `@claude-fable5-399-rv 2026-06-10 PDT` — **disk near-miss during pre-fetch:** bundle `include` rules fetch ALL listed quants, not just the snapshot's (12B pulled Q4_K_M+Q8_0 = 18G; 27B lists Q4/Q5/Q8 ≈ 64G). Cache hit 105G with 12G free on mac1; I killed the fetch after the 27B Q4_K_M blob completed. Budget disk accordingly or add a quant filter to `motlie-models-download`.
- `@qwen447-impl 2026-06-10 18:03 PDT` — **#447 harness fix:** curated GGUF smoke cells now use quant-specific artifact patterns, `qwen3_6_27b_gguf` Q4 cells pin `Qwen3.6-27B-Q4_K_M.gguf`, the matrix child derives `--precision q4` from `quantization = "q4_k_m"`, and `motlie-models-download --precision q4|q5|q8|fp8` filters GGUF includes so labeled eval prefetch does not pull every listed quant.
- `@claude-fable5-399-rv 2026-06-10 PDT` — **debug child builds inflate wall time (mac1 data for the orchestrator gotcha above):** every mistralrs chat/perf/tool cell on this host burned the full 1200s budget (7 cells ≈ 2h20m for zero successful generations). The #435 harness batch switches matrix children to release before the next round.

## Open issues
- `@codex-399-impl 2026-06-10 PDT` -- **#444/#449/#451 ORT follow-up:** eval child builds for ONNX/ORT-backed Piper and Sherpa cells now use static ORT from the workspace `third_party/ort-sys` patch, scrub host dynamic ORT env, and classify `_OrtGetApiBase`/undefined-symbol linker failures as `native_link_failed`. Local aarch64 one-cell repros for Piper and Sherpa both passed with poisoned parent ORT env; see issue comments for raw commands and metrics.
- Umbrella tracker: #435 (sub-issues filed as runs land)
- `@claude-fable5-399-rv 2026-06-10 PDT` — **snapshot quant label vs runtime default mismatch:** `qwen3_6_27b_gguf` cells are labeled `q4_k_m` but the runtime default (no `--precision` from the driver) resolves the bundle's recommended quant **Q5_K_M** → `GGUF artifact Qwen3.6-27B-Q5_K_M.gguf not found`. Any green record under this condition would mislabel the quant axis. Fix: driver passes `--precision` derived from the cell's quantization label, or snapshot validation against `QuantizationSupport.recommended`.
- `@claude-fable5-399-rv 2026-06-10 PDT` — **failure-classification heuristics misfire (2 cases):** (a) the 27B `Q5_K_M.gguf not found` log classified `child_run_failed` instead of `artifact_missing` (phrasing dodges the heuristics); (b) sherpa/piper child **builds** failed with ort-sys link errors (`_OrtGetApiBase ... not found for architecture arm64` — ORT dylib not linkable on macOS without `ORT_LIB_PATH`) but classified `artifact_unauthorized`. Suggest matching on the specific loader error strings and checking child_build status before artifact heuristics.
- `@claude-fable5-399-rv 2026-06-10 PDT` — **qwen3_4b (mistralrs) chat_smoke empty response:** `response contained neither text content nor tool calls` — suspected Qwen3 think-token exhaustion of `max_tokens = 96`; cross-check on x86 leg (same CPU path).
- `@claude-fable5-399-rv 2026-06-10 PDT` — **gemma4 GGUF chat-template defect (systematic):** e4b/12b/12b-qat GGUF `chat_smoke` all produce 0 response chars on llama.cpp/Metal (accelerator pass, true `failed` records) while their bench/tool_use cells pass; qwen3-4b GGUF chat is fine. Likely gemma4 chat-template handling in the llama.cpp text path.
- `@claude-fable5-399-rv 2026-06-10 PDT` — **ort-backed cells (sherpa/piper) could not link on macOS in the matrix child build** (historical mac1 finding from the first run). Follow-up #444/#449/#451 now routes ORT-backed eval child builds through static ORT policy and classifies link failures as `native_link_failed`; rerun required on mac/x86 to replace the old failed records.

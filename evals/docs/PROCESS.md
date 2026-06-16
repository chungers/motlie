# Eval Process

This is the authoritative policy and process document for curated eval cycles. Keep operational command details in [../RUNBOOK.md](../RUNBOOK.md), deep ontology design in [DESIGN.md](DESIGN.md), generated artifact provenance in [../artifacts/provenance.md](../artifacts/provenance.md), and generated outputs under [../results/](../results/) or [../reports/](../reports/). Do not create separate artifact, coverage, or methodology policy docs; consolidate policy here and link to generated evidence.

## 1. Run Model

Curated evals are decentralized and host self-selecting. A runner starts from the current evals branch pin, runs the matrix driver, and lets the driver detect architecture, accelerator inventory, profile applicability, and bundle feature requirements. Each host runs what it can, skips or blocks what it cannot with a structured reason, and writes collision-free run directories under `evals/results/<snapshot>/<run-id>/`.

Artifact preflight is a pre-run gate. Run `evals preflight` against the canonical cache before matrix work; missing artifacts, missing submodules, native build gaps, and unsupported profiles must be recorded as structured outcomes instead of hidden by absence.

Coverage publication is PR-based. Per-host raw records are committed through results PRs, then `evals report --aggregate` produces the cycle report and accounting matrix from committed data.

## 2. Artifacts And Provenance

The `motlie-models` curated bundle registry is the source of truth for artifacts. The generated provenance document covers all 18 `CuratedBundle` descriptors and must stay in sync through the all-curated provenance `#[test]`: [../artifacts/provenance.md](../artifacts/provenance.md).

Every artifact is either downloaded or derived. Downloaded artifacts declare an HF repo, snapshot, and file rule. Derived artifacts declare the producing recipe and source artifact. Mystery local files, mislabeled files, and unprovenanced cache mutations are not accepted; switch to a known published source or add a reproducible derivation recipe before declaring the artifact provisionable.

The canonical per-bundle artifact tool is `motlie-models-download` (`libs/models/src/bin/download_artifacts.rs`). It downloads the registry-declared HF artifacts and then runs that bundle's declared derivations in the same reproducible path; for Kokoro, that means generating `tokens.txt` from `tokenizer.json` via `kokoro_82m::tokens_txt_from_tokenizer_json` and copying `model.onnx`, `voices.bin`, and `espeak-ng-data` from the downloaded streaming source. Future model-specific artifact processing belongs in this registry download path, declared in the registry and generated provenance doc, never as ad-hoc manual steps.

The canonical cache root is `$HOME/artifacts/hf-cache`, using the standard Hugging Face layout. Keep one canonical copy per repo/snapshot and deduplicate through that cache rather than copying bespoke per-worktree artifacts. Download from HF through the registry sync path, validate snapshot-hash parity before repointing a descriptor, and validate runtime behavior against the provenanced files. See [../RUNBOOK.md](../RUNBOOK.md#artifacts) for commands and [../artifacts/provenance.md](../artifacts/provenance.md) for per-bundle recipes.

HF tokens are environment-only. Results may record token presence as a boolean, never a token value. Scrub phone numbers and credentials from logs, comments, reports, and committed data.

## 3. Coverage And Metrics

Coverage is enum-keyed. The tuple `(bundle, quantization, capability, profile)` maps one-to-one to `CuratedBundle`, `QuantizationScheme`, `CapabilityKind`, and the closed eval `Profile` registry. Quantization is scheme-aware, not a bits-only rollup. Code labels and eval labels must match; freetext dimensions are not accepted.

Each tuple reconciles into exactly one of four states: `Validated`, `NotApplicable(reason)`, `BuildGap`, or `Gap`. Completeness tests are fail-closed. A missing tuple, unparsable tuple, or contradictory record is a failure. A gap must be declared with an explicit reason; silent absence is not a state.

Metrics are capability-specific. LLM rows report warmup/startup, two-number TTFT, tokens/sec, and peak memory where instrumented. ASR rows report RTF, WER, and time-to-first-partial. TTS rows report RTF and time-to-first-audio. Embeddings rows report vectors/sec or an explicit gap. Tool-use rows report precision, recall, and round-trip behavior.

Two-number TTFT is required for LLM/chat-style output: first generated token and first answer token. Thinking tokens before the answer are counted separately where available.

Do not compare performance across different hardware. Compare same-host, same-pin, same-profile runs only. Cross-host reports may show coverage state, but speedup claims require same-host evidence.

For design details see [DESIGN.md](DESIGN.md). For generated reports and indexes see [../results/](../results/) and [../reports/](../reports/).

## 4. Accelerator Honesty

Accelerator labels are observed, not requested. `resolved_accelerator` must reflect the device/backend mode actually observed by the backend. A model that starts without backend offload proof is `Blocked`, not `Validated`; no backend may get a false green from a requested CUDA or Metal label.

Offload verification is fail-closed for real runs. If a model launches and no backend observation proves the requested accelerator, record `backend_offload_unverified` or the backend-specific honest reason. Pre-run failures and child build/run failures carry the real primary reason, such as `artifact_unauthorized`, `feature_build_failed`, `native_link_failed`, or `timeout`; do not manufacture `backend_offload_unverified` for a model that never reached the observation point.

The CUDA audio finding remains part of the process record: small ORT audio models show no meaningful CUDA speedup on the DGX, so CPU can be the right deployment answer. LLM CUDA paths do show meaningful acceleration when offload is proven.

## 5. Methodology

Audio latency uses the cold/warm two-phase protocol. Cold runs measure one first-call path with zero warmup in a fresh process. Warm runs use configured warmup iterations and report mean and p95. Keep both phases; do not overwrite cold evidence with warm evidence.

Measure caps; do not guess them. Let generation run to natural EOS within the scenario context unless the cycle explicitly sets `--max-wall-time-secs` as a backstop. Record `thinking_tokens_to_answer` when answer-first behavior matters.

Do not fabricate metric rows. A timeout, build failure, artifact failure, or behavior failure commits the structured failure/block record with no invented metric values.

## 6. Data Integrity

Raw result records are immutable evidence. Committed raw run directories are never silently rewritten. Derived views and indexes must regenerate byte-stably from the raw directories.

Any data migration requires prior sign-off, an exact mapping proposal, and proof that only approved fields changed. Migrations must be fields-only unless explicitly approved otherwise; performance, resource, and acceptance data are not rewritten as a side effect.

## 7. Validation And Review

Independent review means a different account and identity from the author, not merely a different session. The #519 self-review gap is the standing cautionary example.

Review at the exact PR head. Before validating, fetch the branch and verify the checked-out SHA equals the PR head. Stale worktrees invalidate review evidence.

Streaming requires real streaming proof. For speech, first playable PCM must be produced before independent synthesis completion. A chunked buffer emitted after full synthesis is not streaming. The #531 streaming contract is the reference point for this distinction.

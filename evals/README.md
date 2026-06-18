# Motlie Curated Model Evals

`evals/` contains the curated model eval scenarios, matrix snapshots, raw run records, and generated reports used to decide whether local model bundles are deployable on the supported host classes.

## Start Here

- [RUNBOOK.md](RUNBOOK.md): operational how-to for running the matrix, preflighting artifacts, collecting host results, and generating aggregate reports.
- [docs/PROCESS.md](docs/PROCESS.md): authoritative policies and process rules for eval data, artifacts, coverage, metrics, acceleration, review, and migrations.
- [docs/DESIGN.md](docs/DESIGN.md): deeper coverage-ontology design context behind the generated coverage index and accounting matrix.
- [artifacts/provenance.md](artifacts/provenance.md): generated artifact provenance from the 18 curated bundle descriptors.
- [results/](results/): committed raw result records and generated coverage/index artifacts that are intentionally versioned.
- [reports/](reports/): generated report outputs when a cycle produces committed report artifacts.

## Layout

```text
evals/
  README.md        # this index
  RUNBOOK.md       # operator commands and cycle notes
  docs/
    PROCESS.md     # authoritative policies and process
    DESIGN.md      # deep ontology design notes
  artifacts/       # generated provenance over curated bundle descriptors
  scenarios/       # TOML scenario manifests
  datasets/        # prompts, images, audio, and expected outputs
  snapshots/       # pinned matrix manifests
  results/         # immutable raw records plus committed derived coverage artifacts
  reports/         # generated aggregate reports, when committed for a cycle
```

## Common Entry Points

Run artifact preflight before matrix work:

```sh
BINDGEN_EXTRA_CLANG_ARGS="-I$PWD/tools/clang-compat/include" \
  cargo run -p evals --features all-curated -- preflight
```

Run a host-selected matrix:

```sh
cargo run -p evals -- matrix --snapshot evals/snapshots/curated-v2-smoke.toml
```

Generate an aggregate report from committed raw results:

```sh
cargo run -p evals -- report --aggregate 'evals/results/**/results.jsonl' \
  --snapshot evals/snapshots/curated-v2-smoke.toml \
  --output evals/reports/curated-v2-smoke/coverage.md
```

Operational detail belongs in [RUNBOOK.md](RUNBOOK.md). Policy decisions belong in [docs/PROCESS.md](docs/PROCESS.md). Do not add separate artifact, coverage, or methodology policy docs; fold those updates into PROCESS and link out to generated artifacts or reports.

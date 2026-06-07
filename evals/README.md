# Motlie Curated Model Evals

`evals/` contains scenario manifests, datasets, and generated reports for
curated model acceptance.

Human-facing examples stay under `libs/models/examples`. The eval suite is for
repeatable curation evidence: bundle behavior, runtime configuration, platform
profile, performance, resources, and final acceptance status.

## Layout

```text
evals/
  scenarios/       # TOML scenario manifests
  datasets/        # prompts, images, audio, and expected outputs
  reports/         # generated outputs, ignored by default
```

## Scenario Conventions

- `id` is stable snake_case and matches the filename without `.toml`.
- `capability` names the model capability under test and serializes as snake_case, for example `embeddings`.
- `bundle_filter` selects compatible bundles by capability and backend.
- `input` describes prompts, audio, images, or text inputs.
- `assertions` defines behavioral pass/fail checks.
- `metrics` requests performance and resource capture.
- `profiles.<name>.gates` may define optional acceptance thresholds.

## Embeddings Exemplar

```sh
cargo run -p evals --features "model-google-gemma-300m model-qwen3-embedding-06b" -- run --bundle embeddinggemma_300m --scenario embeddings_similarity
```

The command parses the scenario TOML, starts one compiled embedding bundle from
local artifacts, emits a sectioned JSONL result record, and applies the
`similar_gt_dissimilar` assertion.

Generated JSONL results should be consumed by `evals report` once the reporting
command lands.

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
- `capability` names the model capability under test.
- `bundle_filter` selects compatible bundles by capability and backend.
- `input` describes prompts, audio, images, or text inputs.
- `assertions` defines behavioral pass/fail checks.
- `metrics` requests performance and resource capture.
- `profiles.<name>.gates` may define optional acceptance thresholds.

Generated JSONL results should be consumed by `evals report` once the CLI
lands.

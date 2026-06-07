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
`similar_gt_dissimilar` assertion. On GB10/Linux AArch64, the repo
`.cargo/config.toml` wires the required `+fp16,+fhm` target features, so no
manual `RUSTFLAGS` are needed for the default Cargo command.

For CUDA-class hosts, pass the matching profile:

```sh
cargo run -p evals --features "model-google-gemma-300m model-qwen3-embedding-06b" -- run --bundle embeddinggemma_300m --scenario embeddings_similarity --profile dgx-spark
```

When `nvidia-smi` is available, platform records include `gpu_backend =
"nvidia"`, GPU identity, and driver/CUDA metadata.

Generated JSONL results should be consumed by `evals report` once the reporting
command lands.

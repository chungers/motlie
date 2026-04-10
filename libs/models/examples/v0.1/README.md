# `motlie-models` v0.1 Example

This example demonstrates the curated embedding slice in `libs/models`.

It is intentionally built with exactly one curated embedding bundle feature enabled. Today that means either:

- `embeddinggemma_300m` (`google/embeddinggemma-300m`)
- `qwen3_embedding_06b` (`Qwen/Qwen3-Embedding-0.6B`)

The bundle returns normalized embedding vectors. In practice that means:

- cosine similarity is the intended comparison metric
- dot product is effectively equivalent for ranking because the vectors are normalized
- the example below computes cosine similarity directly so the retrieval semantics are visible in the output

## What It Demonstrates

1. direct curated enum selection through `EmbeddingModels::only_enabled()` for a single-bundle build
2. optional parser-driven selection through `--embedding=google/embeddinggemma_300m` or `--embedding=qwen/qwen3_embedding_06b`
3. optional curated artifact download into `artifacts/models/hf-cache`
4. descriptor/capability introspection
5. local-only startup through `ArtifactPolicy::LocalOnly`
6. optional quantization through `--precision=q4|q8|f32`, validated against the selected bundle's `QuantizationSupport`
7. one-shot embedding generation from command-line input
8. a semantically similar pair with their vectors and cosine similarity
9. a semantically dissimilar pair with their vectors and cosine similarity
10. latency for startup and each embedding computation
11. process/memory snapshots before startup, after startup, and after each embedding call
12. handle-level model metrics after startup and after each embedding phase

## Run

EmbeddingGemma build:

```sh
cargo run -p motlie-models --no-default-features --features model-google-gemma-300m --example models_v0_1 -- "motlie curated model bundle"
```

Qwen build:

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-embedding-06b --example models_v0_1 -- --precision=q8 "motlie curated model bundle"
```

The default path uses the direct enum selection for the single embedding model compiled into the build.

If you want the example to exercise the parser-driven selector path, use the selector that matches the enabled bundle:

```sh
cargo run -p motlie-models --no-default-features --features model-google-gemma-300m --example models_v0_1 -- --embedding=google/embeddinggemma_300m "motlie curated model bundle"
cargo run -p motlie-models --no-default-features --features model-qwen3-embedding-06b --example models_v0_1 -- --embedding=qwen/qwen3_embedding_06b --precision=q8 "motlie curated model bundle"
```

If you want the example to prefetch curated artifacts before startup:

```sh
cargo run -p motlie-models --no-default-features --features model-google-gemma-300m --example models_v0_1 -- --download-artifacts "motlie curated model bundle"
cargo run -p motlie-models --no-default-features --features model-qwen3-embedding-06b --example models_v0_1 -- --download-artifacts --precision=q8 "motlie curated model bundle"
```

## Preconditions

- network access is only required when `--download-artifacts` is used or when curated artifacts are not already present and you choose to prefetch them
- if the upstream model requires authentication, pre-download artifacts out of band with the downloader utility and an HF token
- the current machine can run the `mistralrs` embedding path used by the selected curated embedding bundle
- for regulated or offline validation, omit `--download-artifacts` and rely on the previously populated curated artifact root
- the example expects a single-bundle build and prints `catalog-entry-count: 1`; use `--no-default-features` plus exactly one embedding-bundle feature as shown above
- not every embedding bundle supports every precision:
  - `embeddinggemma_300m` currently supports only `f32`
  - `qwen3_embedding_06b` currently supports `q8`, with `f32` as the unquantized default

Example authenticated pre-download:

```sh
export HF_TOKEN=...
cargo run -p motlie-models --no-default-features --features model-google-gemma-300m --bin motlie-models-download -- --hf-token-env HF_TOKEN embeddinggemma_300m
cargo run -p motlie-models --no-default-features --features model-qwen3-embedding-06b --bin motlie-models-download -- --hf-token-env HF_TOKEN qwen3_embedding_06b
```

## Expected Output

The example prints:

- the bundle ID and artifact root
- whether bundle resolution happened through the direct enum path or parsed selector path
- whether the run downloaded curated files or intentionally skipped download and used existing local artifacts
- process snapshots including pid and RSS before startup, after startup, and after requests
- bundle metadata such as family and backend
- capability descriptor details including normalized input/output kinds and interaction style
- the embedding vector dimension and first few floats for the command-line input
- the latency for the command-line embedding request
- a similar text pair, each vector head, and the computed cosine similarity
- a dissimilar text pair, each vector head, and the computed cosine similarity
- latency for each pair computation

A successful run should show the similar pair with a noticeably higher cosine similarity than the dissimilar pair.

## Source

- example entrypoint: [main.rs](/Users/dchung/projects/claude-mistral/motlie/libs/models/examples/v0.1/main.rs)
- bundle definition: [google_gemma_300m.rs](/Users/dchung/projects/claude-mistral/motlie/libs/models/src/embeddings/google_gemma_300m.rs)
- bundle definition: [qwen3_embedding_06b.rs](/Users/dchung/projects/claude-mistral/motlie/libs/models/src/embeddings/qwen3_embedding_06b.rs)

# `motlie-models` v0.1 Example

This example demonstrates the current curated embedding bundle in `libs/models`.

Important: the current bundle is `embeddinggemma_300m` (`google/embeddinggemma-300m`), not a Gemma 3 chat bundle. This is the first curated embedding vertical slice currently implemented in the catalog.

The bundle returns normalized embedding vectors. In practice that means:

- cosine similarity is the intended comparison metric
- dot product is effectively equivalent for ranking because the vectors are normalized
- the example below computes cosine similarity directly so the retrieval semantics are visible in the output

## What It Demonstrates

1. direct curated enum selection through `EmbeddingModels::GoogleGemma300m`
2. optional parser-driven selection through `--embedding=google/embeddinggemma_300m`
3. optional curated artifact download into `artifacts/models/hf-cache`
4. descriptor/capability introspection
5. local-only startup through `ArtifactPolicy::LocalOnly`
6. one-shot embedding generation from command-line input
7. a semantically similar pair with their vectors and cosine similarity
8. a semantically dissimilar pair with their vectors and cosine similarity
9. latency for each embedding computation

## Run

```sh
cargo run -p motlie-models --example models_v0_1 -- "motlie curated model bundle"
```

This default path uses the direct enum selection:

```rust
EmbeddingModels::GoogleGemma300m
```

If you want the example to exercise the parser-driven selector path:

```sh
cargo run -p motlie-models --example models_v0_1 -- --embedding=google/embeddinggemma_300m "motlie curated model bundle"
```

If you want the example to prefetch curated artifacts before startup:

```sh
cargo run -p motlie-models --example models_v0_1 -- --download-artifacts "motlie curated model bundle"
```

## Preconditions

- network access is only required when `--download-artifacts` is used or when curated artifacts are not already present and you choose to prefetch them
- if the upstream model requires authentication, pre-download artifacts out of band with the downloader utility and an HF token
- the current machine can run the `mistralrs` embedding path used by `embeddinggemma_300m`
- for regulated or offline validation, omit `--download-artifacts` and rely on the previously populated curated artifact root

Example authenticated pre-download:

```sh
export HF_TOKEN=...
cargo run -p motlie-models --bin motlie-models-download -- --hf-token-env HF_TOKEN embeddinggemma_300m
```

## Expected Output

The example prints:

- the bundle ID and artifact root
- whether bundle resolution happened through the direct enum path or parsed selector path
- whether the run downloaded curated files or intentionally skipped download and used existing local artifacts
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

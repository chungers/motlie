# `motlie-models` v0.1 Example

This example demonstrates the current curated embedding bundle in `libs/models`.

Important: the current bundle is `embeddinggemma_300m` (`google/embeddinggemma-300m`), not a Gemma 3 chat bundle. This is the first curated embedding vertical slice currently implemented in the catalog.

## What It Demonstrates

1. explicit curated artifact download into `artifacts/models/hf-cache`
2. catalog lookup and descriptor/capability introspection
3. local-only startup through `ArtifactPolicy::LocalOnly`
4. one-shot embedding generation from command-line input

## Run

```sh
cargo run -p motlie-models --example models_v0_1 -- "motlie curated model bundle"
```

## Preconditions

- network access is available if the curated artifacts are not already present in the local artifact root
- if the upstream model requires authentication, pre-download artifacts out of band with the downloader utility and an HF token
- the current machine can run the `mistralrs` embedding path used by `embeddinggemma_300m`

Example authenticated pre-download:

```sh
export HF_TOKEN=...
cargo run -p motlie-models --bin motlie-models-download -- --hf-token-env HF_TOKEN embeddinggemma_300m
```

## Expected Output

The example prints:

- the bundle ID and artifact root
- the number of curated files downloaded or confirmed in cache
- bundle metadata such as family, backend, and packaging
- capability descriptor details including normalized input/output kinds and interaction style
- the embedding vector dimension
- the first few floats from the embedding vector

## Source

- example entrypoint: [main.rs](/Users/dchung/projects/claude-mistral/motlie/libs/models/examples/v0.1/main.rs)
- bundle definition: [embeddinggemma_300m.rs](/Users/dchung/projects/claude-mistral/motlie/libs/models/src/embeddinggemma_300m.rs)

# `motlie-models` `embeddings_basic` Example

`embeddings_basic` is the capability-first embedding example. It demonstrates
how to select a curated embedding bundle, start it from local artifacts, embed
text, and inspect cosine similarity for two fixed pairs.

The legacy `embeddings` example target remains registered and runs the same
implementation for compatibility.

## Run

```sh
cargo run -p motlie-models --no-default-features --features "model-google-gemma-300m model-qwen3-embedding-06b" --example embeddings_basic -- --bundle embeddinggemma_300m "motlie curated model bundle"
```

Selector form is also accepted:

```sh
cargo run -p motlie-models --no-default-features --features "model-google-gemma-300m model-qwen3-embedding-06b" --example embeddings_basic -- --selector embedding:qwen/qwen3_embedding_06b --precision=q8 "motlie curated model bundle"
```

The compatibility flag from the legacy target still works:

```sh
cargo run -p motlie-models --no-default-features --features "model-google-gemma-300m model-qwen3-embedding-06b" --example embeddings_basic -- --embedding=google/embeddinggemma_300m "motlie curated model bundle"
```

## Eval Counterpart

Repeatable acceptance evidence belongs in
[`../../../../evals/scenarios/embeddings_similarity.toml`](../../../../evals/scenarios/embeddings_similarity.toml)
and the runnable counterpart is:

```sh
cargo run -p evals --features "model-google-gemma-300m model-qwen3-embedding-06b" -- run --bundle embeddinggemma_300m --scenario embeddings_similarity
```

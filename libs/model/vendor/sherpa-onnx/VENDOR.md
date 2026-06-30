# sherpa-onnx Vendor Notes

Upstream: `sherpa-onnx` crate version `1.13.2` from crates.io / https://github.com/k2-fsa/sherpa-onnx.

Local reason: the high-level Rust wrapper did not expose the native online ASR token probability arrays already present in `SherpaOnnxGetOnlineStreamResultAsJson`; issue #480 needs those fields for the ASR confidence carrier while keeping the existing vendored `sherpa-onnx-sys` path.

Local delta from upstream 1.13.2:
- `Cargo.toml` is the upstream manifest shape with `sherpa-onnx-sys` pointed at sibling path `../sherpa-onnx-sys`; crates.io packaging metadata files are not vendored.
- `src/online_asr.rs`: `RecognizerResult` includes `ys_probs: Option<Vec<f32>>` and `lm_probs: Option<Vec<f32>>`, plus a JSON-deserialization regression test proving both native arrays are retained.
- `src/tts.rs`: the doctest `None` progress callback is typed as `None::<fn(&[f32], f32) -> bool>` so local doctests compile.
- `src/offline_asr.rs`: clippy-only Moonshine doc-comment reorder for `doc_lazy_continuation`.
- `src/speaker_embedding.rs`: clippy-only modernization of the embedding length divisibility check to `is_multiple_of`.

When upgrading, re-vendor from the new upstream crate, preserve the local `sherpa-onnx-sys` path dependency, and re-check whether upstream now exposes `ys_probs`/`lm_probs` before reapplying the wrapper delta.

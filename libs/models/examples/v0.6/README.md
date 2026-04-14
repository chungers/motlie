# v0.6 — Sherpa ONNX Streaming ASR

This example demonstrates the Phase 2 ASR vertical slice using the
`sherpa-onnx` Zipformer backend.

## Run

```bash
cargo run -p motlie-models --example models_v0_6 \
  --no-default-features --features model-sherpa-onnx-streaming \
  -- --wav /path/to/audio.wav
```

## Preconditions

- The curated sherpa-onnx artifacts must already be downloaded under the
  default artifact root or the path passed with `--artifact-root`.
- A compatible ONNX Runtime shared library must be available for the `ort`
  crate. If it is not on the system library path, set `ORT_DYLIB_PATH`.
- The `.wav` file must be PCM audio (`i16` or `f32`).

## Expected Behavior

- The example opens the curated `Sherpa ONNX Streaming Zipformer EN` bundle.
- Audio is streamed through the shared PCM transcription contract.
- Incremental transcript updates are printed as `[partial]` and `[final]`
  segments while the file is processed.

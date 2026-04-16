# tts_v0.3 — Fish Speech Vertical Slice

This example demonstrates the Phase 3 TTS vertical slice using the curated
Fish Speech 1.5 Rust backend.

## Run

```bash
cargo run -p motlie-models --example models_tts_v0_3 \
  --no-default-features --features model-fish-speech-1_5 \
  -- --text "Hello from Motlie." --wav /tmp/motlie-fish-speech.wav
```

## Preconditions

- The curated Fish Speech 1.5 safetensor artifacts must already be downloaded
  under the default artifact root or the path passed with `--artifact-root`.
- The env-gated test path uses `MOTLIE_TEST_FISH_SPEECH_ARTIFACT_ROOT` to
  point at the same Hugging Face cache root.
- The first vertical slice embeds the upstream default voice prompt assets in
  the backend crate, so no external voice directory setup is required.
- `MOTLIE_MODEL_FORCE_CPU=1` forces CPU startup even when the backend is built
  with `fish-speech-cuda` or `fish-speech-metal`.
- On this Apple Silicon host, validation required
  `RUSTFLAGS='-C target-cpu=native'` because the upstream Candle/GEMM stack
  emits FP16 instructions that are unavailable under the generic baseline
  target.

## Expected Behavior

- The example opens the curated `Fish Speech 1.5` bundle.
- Text is synthesized through the shared `SpeechModel` / `SpeechStream`
  contract and collected into a `.wav` file.
- In Phase 3 v0.3, Fish Speech performs whole-utterance synthesis in
  `open_stream()` and then emits buffered PCM chunks through `next_chunk()`.
- The resulting file uses the backend-reported sample rate and `S16Le` PCM
  encoding.
- This first implementation slice does not yet expose reference-audio cloning
  or named speaker selection on the Motlie contract surface.

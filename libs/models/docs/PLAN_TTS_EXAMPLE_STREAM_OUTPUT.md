# TTS Example Stream Output Plan

## Status: Proposed

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-19 | @codex-tts | Revised the plan to emit WAV on stdout by default when `--wav` is absent. Replaced the custom framing work with a dedicated non-seekable stdout WAV writer and validation against normal CLI consumers. |
| 2026-04-19 | @codex-tts | Initial plan for issue #208. Covers a shared CLI/sink helper and consistent stream-output behavior for all shipped TTS examples. |

Derived from [DESIGN_TTS_EXAMPLE_STREAM_OUTPUT.md](./DESIGN_TTS_EXAMPLE_STREAM_OUTPUT.md).

## Phase 1: Shared Example Helper

### 1.1 - Add common TTS example support module

- [ ] Add a shared helper module under `libs/models/examples/`.
  DESIGN reference: `Layering`
- [ ] Centralize parsing of common flags: `--text`, `--artifact-root`, `--wav`.
  DESIGN reference: `Proposed Shared CLI`
- [ ] Add stdin text loading for cases where `--text` is omitted.
  DESIGN reference: `Required Behavior`

### 1.2 - Add sink helpers

- [ ] Add a `.wav` sink helper that collects typed speech chunks into a wav
  writer.
  DESIGN reference: `Required Behavior`
- [ ] Add a stdout-safe WAV writer for non-seekable output streams.
  DESIGN reference: `Stdout WAV Contract`
- [ ] Route diagnostics to stderr in pipeline mode.
  DESIGN reference: `Error Handling`

## Phase 2: Migrate All Shipped TTS Examples

### 2.1 - Update `tts_piper`

- [ ] Replace ad hoc CLI parsing with the shared helper.
  DESIGN reference: `Affected Binaries`
- [ ] Support stdin text input in addition to `--text`.
  DESIGN reference: `Required Behavior`
- [ ] Support stdout WAV output when `--wav` is omitted.
  DESIGN reference: `Required Behavior`, `Stdout WAV Contract`

### 2.2 - Update `tts_qwen3_onnx`

- [ ] Replace ad hoc CLI parsing with the shared helper while preserving
  `--reference-audio` and `--reference-text`.
  DESIGN reference: `Affected Binaries`, `Proposed Shared CLI`
- [ ] Support stdin text input and stdout WAV output.
  DESIGN reference: `Required Behavior`

### 2.3 - Update `tts_qwen3_tts_cpp`

- [ ] Replace ad hoc CLI parsing with the shared helper while preserving
  `--reference-audio`.
  DESIGN reference: `Affected Binaries`, `Proposed Shared CLI`
- [ ] Support stdin text input and stdout WAV output.
  DESIGN reference: `Required Behavior`

## Phase 3: Documentation

### 3.1 - Update example READMEs

- [ ] Document file mode for all three binaries.
  DESIGN reference: `Required Behavior`
- [ ] Document pipeline mode with stdin text and stdout WAV examples.
  DESIGN reference: `Required Behavior`, `Stdout WAV Contract`
- [ ] Document backend-specific flags separately from the shared flags.
  DESIGN reference: `Proposed Shared CLI`

### 3.2 - Update `libs/models/docs/API.md`

- [ ] Add a short section that points at the standardized TTS example behavior.
  DESIGN reference: `Affected Binaries`

## Phase 4: Validation

### 4.1 - Build checks

- [ ] `cargo check -p motlie-models --example tts_piper --no-default-features --features model-piper-en-us-ljspeech-medium`
- [ ] `cargo check -p motlie-models --example tts_qwen3_onnx --no-default-features --features model-qwen3-tts-0_6b`
- [ ] `cargo check -p motlie-models --example tts_qwen3_tts_cpp --no-default-features --features model-qwen3-tts-cpp`

### 4.2 - Behavior checks

- [ ] Verify `.wav` output still works for each binary.
  DESIGN reference: `Validation`
- [ ] Verify stdin text works for each binary when `--text` is omitted.
  DESIGN reference: `Validation`
- [ ] Verify stdout emits a valid WAV stream for each binary.
  DESIGN reference: `Validation`
- [ ] Verify standard CLI consumers can read stdout WAV, for example `ffmpeg`
  or another stdin-based WAV consumer.
  DESIGN reference: `Validation`, `TTS to ASR Composition`

### 4.3 - Error-path checks

- [ ] Verify diagnostics stay off stdout in pipeline mode.
  DESIGN reference: `Error Handling`

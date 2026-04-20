# TTS Example Stream Output Plan

## Status: Proposed

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-19 | @codex-tts | Initial plan for issue #208. Covers a shared CLI/sink helper and consistent stream-output behavior for all shipped TTS examples. |

Derived from [DESIGN_TTS_EXAMPLE_STREAM_OUTPUT.md](./DESIGN_TTS_EXAMPLE_STREAM_OUTPUT.md).

## Phase 1: Shared Example Helper

### 1.1 - Add common TTS example support module

- [ ] Add a shared helper module under `libs/models/examples/`.
  DESIGN reference: `Layering`
- [ ] Centralize parsing of common flags: `--text`, `--artifact-root`, `--wav`,
  `--stdout-stream`.
  DESIGN reference: `Proposed Shared CLI`
- [ ] Add stdin text loading for cases where `--text` is omitted.
  DESIGN reference: `Required Behavior`

### 1.2 - Add sink helpers

- [ ] Add a `.wav` sink helper that collects typed speech chunks into a wav
  writer.
  DESIGN reference: `Required Behavior`
- [ ] Add a stdout framing helper that writes the JSON header plus chunk frames.
  DESIGN reference: `Stream Framing Contract`
- [ ] Route diagnostics to stderr in pipeline mode.
  DESIGN reference: `Error Handling`

## Phase 2: Migrate All Shipped TTS Examples

### 2.1 - Update `tts_piper`

- [ ] Replace ad hoc CLI parsing with the shared helper.
  DESIGN reference: `Affected Binaries`
- [ ] Support stdin text input in addition to `--text`.
  DESIGN reference: `Required Behavior`
- [ ] Support `--stdout-stream`.
  DESIGN reference: `Required Behavior`, `Stream Framing Contract`

### 2.2 - Update `tts_qwen3_onnx`

- [ ] Replace ad hoc CLI parsing with the shared helper while preserving
  `--reference-audio` and `--reference-text`.
  DESIGN reference: `Affected Binaries`, `Proposed Shared CLI`
- [ ] Support stdin text input and `--stdout-stream`.
  DESIGN reference: `Required Behavior`

### 2.3 - Update `tts_qwen3_tts_cpp`

- [ ] Replace ad hoc CLI parsing with the shared helper while preserving
  `--reference-audio`.
  DESIGN reference: `Affected Binaries`, `Proposed Shared CLI`
- [ ] Support stdin text input and `--stdout-stream`.
  DESIGN reference: `Required Behavior`

## Phase 3: Documentation

### 3.1 - Update example READMEs

- [ ] Document file mode for all three binaries.
  DESIGN reference: `Required Behavior`
- [ ] Document pipeline mode with stdin text and stdout stream examples.
  DESIGN reference: `Required Behavior`, `Stream Framing Contract`
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
- [ ] Verify stdout stream framing is emitted correctly for each binary.
  DESIGN reference: `Validation`
- [ ] Verify a simple stream-to-wav adapter can consume the stdout framing.
  DESIGN reference: `Validation`

### 4.3 - Error-path checks

- [ ] Verify the binaries reject missing sink selection.
  DESIGN reference: `Error Handling`
- [ ] Verify the binaries reject `--wav` plus `--stdout-stream` together in the
  first slice.
  DESIGN reference: `Required Behavior`
- [ ] Verify diagnostics stay off stdout in pipeline mode.
  DESIGN reference: `Error Handling`

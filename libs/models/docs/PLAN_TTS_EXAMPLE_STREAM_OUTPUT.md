# TTS Example Stream Output Plan

## Status: Proposed

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-21 | @codex-tts | Moved the generic stdout WAV and sample-conversion primitives into the new `libs/voice` crate, leaving only CLI/output-selection code in the example helpers. |
| 2026-04-21 | @codex-tts | Reworked the stdout WAV sink to write incrementally with an aligned indefinite-length header, added shutdown-safe handle cleanup, and paired it with a tolerant stdin-side ASR parser so `tts | asr` works without temp files. |
| 2026-04-20 | @codex-tts | Removed `tts_qwen3_onnx` from the shipped example set in this PR after reconfirming it is non-functional for real speech output. The implementation/validation scope now covers Piper and `qwen3-tts.cpp` only. |
| 2026-04-21 | @codex-tts | Implemented quiet-mode stderr redirection so backend-native logs are suppressed as well as example-layer diagnostics in pipe-heavy workflows. |
| 2026-04-20 | @codex-tts | Added the shared `--quiet` flag to the shipped TTS example contract so example-layer stderr logging can be suppressed in pipe-heavy workflows. |
| 2026-04-20 | @codex-tts | Implemented the shared TTS helper modules, migrated all shipped TTS examples to file-or-stdout WAV behavior with stdin text fallback, updated the READMEs/API docs, and validated compile/clippy plus direct stdout WAV and TTS→ASR pipeline behavior. |
| 2026-04-19 | @codex-tts | Revised the plan to emit WAV on stdout by default when `--wav` is absent. Replaced the custom framing work with a dedicated non-seekable stdout WAV writer and validation against normal CLI consumers. |
| 2026-04-19 | @codex-tts | Initial plan for issue #208. Covers a shared CLI/sink helper and consistent stream-output behavior for all shipped TTS examples. |

Derived from [DESIGN_TTS_EXAMPLE_STREAM_OUTPUT.md](./DESIGN_TTS_EXAMPLE_STREAM_OUTPUT.md).

## Phase 1: Shared Example Helper

### 1.1 - Add common TTS example support module

- [x] Add a shared helper module under `libs/models/examples/`.
  DESIGN reference: `Layering`
- [x] Centralize parsing of common flags: `--text`, `--artifact-root`, `--wav`.
  DESIGN reference: `Proposed Shared CLI`
- [x] Add stdin text loading for cases where `--text` is omitted.
  DESIGN reference: `Required Behavior`
- [x] Add `--quiet` for whole-process stderr suppression during quiet example execution.
  DESIGN reference: `Proposed Shared CLI`

### 1.2 - Add sink helpers

- [x] Add a `.wav` sink helper that collects typed speech chunks into a wav
  writer.
  DESIGN reference: `Required Behavior`
- [x] Add a stdout-safe WAV writer for non-seekable output streams.
  DESIGN reference: `Stdout WAV Contract`
- [x] Move the reusable WAV/signal helpers into `libs/voice` so `examples/`
  keep only CLI/source-selection logic.
  DESIGN reference: `Layering`
- [x] Make the stdout sink incremental rather than buffering the whole
  utterance before the first byte is written.
  DESIGN reference: `Stdout WAV Contract`
- [x] Route diagnostics to stderr in pipeline mode.
  DESIGN reference: `Error Handling`

## Phase 2: Migrate All Shipped TTS Examples

### 2.1 - Update `tts_piper`

- [x] Replace ad hoc CLI parsing with the shared helper.
  DESIGN reference: `Affected Binaries`
- [x] Support stdin text input in addition to `--text`.
  DESIGN reference: `Required Behavior`
- [x] Support stdout WAV output when `--wav` is omitted.
  DESIGN reference: `Required Behavior`, `Stdout WAV Contract`

### 2.2 - Update `tts_qwen3_tts_cpp`

- [x] Replace ad hoc CLI parsing with the shared helper while preserving
  `--reference-audio`.
  DESIGN reference: `Affected Binaries`, `Proposed Shared CLI`
- [x] Support stdin text input and stdout WAV output.
  DESIGN reference: `Required Behavior`

## Phase 3: Documentation

### 3.1 - Update example READMEs

- [x] Document file mode for both shipped binaries.
  DESIGN reference: `Required Behavior`
- [x] Document pipeline mode with stdin text and stdout WAV examples.
  DESIGN reference: `Required Behavior`, `Stdout WAV Contract`
- [x] Document backend-specific flags separately from the shared flags.
  DESIGN reference: `Proposed Shared CLI`

### 3.2 - Update `libs/models/docs/API.md`

- [x] Add a short section that points at the standardized TTS example behavior.
  DESIGN reference: `Affected Binaries`

## Phase 4: Validation

### 4.1 - Build checks

- [x] `cargo check -p motlie-models --example tts_piper --no-default-features --features model-piper-en-us-ljspeech-medium`
- [x] `cargo check -p motlie-models --example tts_qwen3_tts_cpp --no-default-features --features model-qwen3-tts-cpp`

### 4.2 - Behavior checks

- [ ] Verify `.wav` output still works for each binary.
- [x] Verify `.wav` output still works for each binary.
  DESIGN reference: `Validation`
- [x] Verify stdin text works for each binary when `--text` is omitted.
  DESIGN reference: `Validation`
- [x] Verify stdout emits a valid WAV stream for each binary.
  DESIGN reference: `Validation`
- [x] Verify standard CLI consumers can read stdout WAV, for example `ffmpeg`
  or another stdin-based WAV consumer.
  DESIGN reference: `Validation`, `TTS to ASR Composition`

### 4.3 - Error-path checks

- [x] Verify diagnostics stay off stdout in pipeline mode.
  DESIGN reference: `Error Handling`

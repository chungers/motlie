# ASR Example Stream Input Plan

## Status: Proposed

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-19 | @codex-tts | Initial plan for stdin WAV support in all shipped ASR examples so they can compose directly with the TTS stdout WAV path from issue #208. |

Derived from [DESIGN_ASR_EXAMPLE_STREAM_INPUT.md](./DESIGN_ASR_EXAMPLE_STREAM_INPUT.md).

## Phase 1: Shared Example Helper

### 1.1 - Add common ASR example support module

- [ ] Add a shared helper module under `libs/models/examples/`.
  DESIGN reference: `Layering`
- [ ] Centralize parsing of common flags: `--wav`, `--artifact-root`.
  DESIGN reference: `Proposed Shared CLI`
- [ ] Add WAV source selection logic: file path or stdin.
  DESIGN reference: `Required Behavior`

### 1.2 - Add pipeline-safe logging behavior

- [ ] Route diagnostics to stderr when the example is reading binary WAV from
  stdin.
  DESIGN reference: `Stdout Transcript Contract`
- [ ] Keep transcript text on stdout.
  DESIGN reference: `Stdout Transcript Contract`

## Phase 2: Migrate All Shipped ASR Examples

### 2.1 - Update `asr_whisper`

- [ ] Replace ad hoc CLI parsing with the shared helper while preserving
  `--language`.
  DESIGN reference: `Affected Binaries`, `Proposed Shared CLI`
- [ ] Support WAV input from stdin when `--wav` is omitted.
  DESIGN reference: `Required Behavior`
- [ ] Keep transcript text on stdout and diagnostics on stderr in pipeline mode.
  DESIGN reference: `Stdout Transcript Contract`

### 2.2 - Update `asr_sherpa_onnx`

- [ ] Replace ad hoc CLI parsing with the shared helper.
  DESIGN reference: `Affected Binaries`
- [ ] Support WAV input from stdin when `--wav` is omitted.
  DESIGN reference: `Required Behavior`
- [ ] Keep transcript text on stdout and diagnostics on stderr in pipeline mode.
  DESIGN reference: `Stdout Transcript Contract`

### 2.3 - Update `asr_moonshine`

- [ ] Replace ad hoc CLI parsing with the shared helper.
  DESIGN reference: `Affected Binaries`
- [ ] Support WAV input from stdin when `--wav` is omitted.
  DESIGN reference: `Required Behavior`
- [ ] Keep transcript text on stdout and diagnostics on stderr in pipeline mode.
  DESIGN reference: `Stdout Transcript Contract`

## Phase 3: Documentation

### 3.1 - Update example READMEs

- [ ] Document file mode for all three binaries.
  DESIGN reference: `Required Behavior`
- [ ] Document stdin pipeline mode for all three binaries.
  DESIGN reference: `Required Behavior`
- [ ] Document backend-specific flags separately from the shared flags.
  DESIGN reference: `Proposed Shared CLI`

### 3.2 - Cross-reference the TTS example work

- [ ] Add a short note showing `tts_example | asr_example` composition.
  DESIGN reference: `Validation`

## Phase 4: Validation

### 4.1 - Build checks

- [ ] `cargo check -p motlie-models --example asr_whisper --no-default-features --features model-whisper-base-en`
- [ ] `cargo check -p motlie-models --example asr_sherpa_onnx --no-default-features --features model-sherpa-onnx-streaming`
- [ ] `cargo check -p motlie-models --example asr_moonshine --no-default-features --features model-moonshine-streaming`

### 4.2 - Behavior checks

- [ ] Verify `--wav <path>` still works for each binary.
  DESIGN reference: `Validation`
- [ ] Verify stdin WAV works for each binary when `--wav` is omitted.
  DESIGN reference: `Validation`
- [ ] Verify transcript text stays on stdout.
  DESIGN reference: `Validation`
- [ ] Verify diagnostics move to stderr in pipeline mode.
  DESIGN reference: `Validation`
- [ ] Verify at least one end-to-end `tts_example | asr_example` pipeline.
  DESIGN reference: `Validation`

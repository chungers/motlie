# ASR Example Stream Input Plan

## Status: Proposed

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-21 | @codex-tts | Moved the reusable WAV decode/downmix/resample primitives into `libs/voice` and factored sherpa/moonshine through one shared streaming runner helper so the example layer no longer duplicates the same audio pipeline. |
| 2026-04-21 | @codex-tts | Replaced the plain stdin `hound` path with a tolerant example-layer WAV parser so the ASR examples accept the aligned indefinite-length headers emitted by the TTS stdout sink, and added regression tests for that contract. |
| 2026-04-21 | @codex-tts | Implemented quiet-mode stderr redirection so backend-native logs are suppressed as well as example-layer diagnostics. |
| 2026-04-20 | @codex-tts | Tightened the shipped stdout contract after live pipe validation. Streaming examples now default to one final plain-text transcript, add `--partials` for event-style output, and add `--quiet` for example-layer stderr suppression. |
| 2026-04-20 | @codex-tts | Implemented the shared ASR helper modules, migrated all shipped ASR examples to file-or-stdin WAV input with transcript stdout and diagnostics stderr, updated the READMEs, and validated compile/clippy plus stdin WAV and TTS→ASR pipeline behavior. |
| 2026-04-19 | @codex-tts | Initial plan for stdin WAV support in all shipped ASR examples so they can compose directly with the TTS stdout WAV path from issue #208. |

Derived from [DESIGN_ASR_EXAMPLE_STREAM_INPUT.md](./DESIGN_ASR_EXAMPLE_STREAM_INPUT.md).

## Phase 1: Shared Example Helper

### 1.1 - Add common ASR example support module

- [x] Add a shared helper module under `libs/models/examples/`.
  DESIGN reference: `Layering`
- [x] Centralize parsing of common flags: `--wav`, `--artifact-root`.
  DESIGN reference: `Proposed Shared CLI`
- [x] Add WAV source selection logic: file path or stdin.
  DESIGN reference: `Required Behavior`
- [x] Use a tolerant stdin-side WAV parser so the examples accept the aligned
  indefinite-length headers emitted by the TTS stdout contract.
  DESIGN reference: `Feasibility`
- [x] Add shared plain-transcript rendering for shell-safe stdout.
  DESIGN reference: `Stdout Transcript Contract`
- [x] Move reusable audio decode/downmix/resample primitives into `libs/voice`
  and keep only source-selection/CLI logic in `examples/`.
  DESIGN reference: `Layering`

### 1.2 - Add pipeline-safe logging behavior

- [x] Route diagnostics to stderr when the example is reading binary WAV from
  stdin.
  DESIGN reference: `Stdout Transcript Contract`
- [x] Keep transcript text on stdout.
  DESIGN reference: `Stdout Transcript Contract`
- [x] Add `--quiet` for whole-process stderr suppression during quiet example execution.
  DESIGN reference: `Stdout Transcript Contract`

## Phase 2: Migrate All Shipped ASR Examples

### 2.1 - Update `asr_whisper`

- [x] Replace ad hoc CLI parsing with the shared helper while preserving
  `--language`.
  DESIGN reference: `Affected Binaries`, `Proposed Shared CLI`
- [x] Support WAV input from stdin when `--wav` is omitted.
  DESIGN reference: `Required Behavior`
- [x] Keep transcript text on stdout and diagnostics on stderr in pipeline mode.
  DESIGN reference: `Stdout Transcript Contract`
- [x] Default stdout to one final plain-text transcript line.
  DESIGN reference: `Stdout Transcript Contract`

### 2.2 - Update `asr_sherpa_onnx`

- [x] Replace ad hoc CLI parsing with the shared helper.
  DESIGN reference: `Affected Binaries`
- [x] Support WAV input from stdin when `--wav` is omitted.
  DESIGN reference: `Required Behavior`
- [x] Keep transcript text on stdout and diagnostics on stderr in pipeline mode.
  DESIGN reference: `Stdout Transcript Contract`
- [x] Add `--partials` so streaming event output is opt-in instead of default.
  DESIGN reference: `Proposed Shared CLI`, `Stdout Transcript Contract`
- [x] Share the generic streaming runner with `asr_moonshine` instead of
  maintaining a second near-identical ingest loop.
  DESIGN reference: `Layering`

### 2.3 - Update `asr_moonshine`

- [x] Replace ad hoc CLI parsing with the shared helper.
  DESIGN reference: `Affected Binaries`
- [x] Support WAV input from stdin when `--wav` is omitted.
  DESIGN reference: `Required Behavior`
- [x] Keep transcript text on stdout and diagnostics on stderr in pipeline mode.
  DESIGN reference: `Stdout Transcript Contract`
- [x] Add `--partials` so streaming event output is opt-in instead of default.
  DESIGN reference: `Proposed Shared CLI`, `Stdout Transcript Contract`
- [x] Share the generic streaming runner with `asr_sherpa_onnx` instead of
  maintaining a second near-identical ingest loop.
  DESIGN reference: `Layering`

## Phase 3: Documentation

### 3.1 - Update example READMEs

- [x] Document file mode for all three binaries.
  DESIGN reference: `Required Behavior`
- [x] Document stdin pipeline mode for all three binaries.
  DESIGN reference: `Required Behavior`
- [x] Document backend-specific flags separately from the shared flags.
  DESIGN reference: `Proposed Shared CLI`

### 3.2 - Cross-reference the TTS example work

- [x] Add a short note showing `tts_example | asr_example` composition.
  DESIGN reference: `Validation`

## Phase 4: Validation

### 4.1 - Build checks

- [x] `cargo check -p motlie-models --example asr_whisper --no-default-features --features model-whisper-base-en`
- [x] `cargo check -p motlie-models --example asr_sherpa_onnx --no-default-features --features model-sherpa-onnx-streaming`
- [x] `cargo check -p motlie-models --example asr_moonshine --no-default-features --features model-moonshine-streaming`

### 4.2 - Behavior checks

- [x] Verify `--wav <path>` still works for each binary.
  DESIGN reference: `Validation`
- [x] Verify stdin WAV works for each binary when `--wav` is omitted.
  DESIGN reference: `Validation`
- [x] Verify transcript text stays on stdout.
  DESIGN reference: `Validation`
- [x] Verify diagnostics move to stderr in pipeline mode.
  DESIGN reference: `Validation`
- [x] Verify at least one end-to-end `tts_example | asr_example` pipeline.
  DESIGN reference: `Validation`

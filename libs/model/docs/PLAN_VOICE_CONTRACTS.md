## Changelog

| Date | Who | Summary |
|---|---|---|
| 2026-04-29 | @codex-tts | Initial plan for tightening speech model contracts and curated metadata across `libs/model` and `libs/models`. |
| 2026-04-29 | @codex-tts | Implementation update: completed the in-repo contract/catalog migration slice, validated buffered TTS backends, and recorded the external voice-skill follow-on explicitly. |
| 2026-04-30 | @codex-tts | Imported the voice-agent skill layer onto `feature/models`, migrated `bins/voice-agent` and the playbook to capability-driven buffered/batch/streaming composition, and removed the stale compatibility helper constructors that still implied the old semantics. |

# Plan: Voice Contracts

Derived from [DESIGN_VOICE_CONTRACTS.md](./DESIGN_VOICE_CONTRACTS.md).

This PLAN focuses on brownfield contract correction and migration. It covers the common speech layer in `libs/model`, the curated catalog in `libs/models`, and the composition surface used by the voice skills and examples.

## Phase 1. Correct Metadata Drift

- [x] 1.1 Audit all curated ASR bundles in `libs/models/src/asr/` and record the actual typed execution contract used by each bundle.
  DESIGN reference: `Current State`, `1. Keep ASR type split, but tighten metadata`
- [x] 1.2 Fix `whisper_base_en` metadata so it no longer advertises streaming-only semantics when the backend implements `BatchTranscriber`.
  DESIGN reference: `Current State`, `Migration Strategy / Phase 1`
- [x] 1.3 Add regression tests that fail if curated metadata claims streaming for a batch-only ASR backend.
  DESIGN reference: `Testing Scope for PLAN`
- [x] 1.4 Review current TTS bundle metadata in `libs/models/src/tts/` and document where `speech_stream_only` currently means buffered chunk output rather than true streaming generation.
  DESIGN reference: `Current State`, `3. Add explicit speech execution metadata`

## Phase 2. Add Explicit Execution Metadata

- [x] 2.1 Extend `libs/model` capability or metadata structures to represent transcription execution mode explicitly:
  `Batch`, `StreamingFinalOnly`, `StreamingWithPartials`.
  DESIGN reference: `1. Keep ASR type split, but tighten metadata`, `3. Add explicit speech execution metadata`
- [x] 2.2 Extend `libs/model` capability or metadata structures to represent speech execution mode explicitly:
  `Buffered`, `Streaming`.
  DESIGN reference: `2. Split TTS into buffered vs streaming contracts`, `3. Add explicit speech execution metadata`
- [x] 2.3 Encode voice-cloning support in structured metadata rather than requiring caller knowledge of backend identity.
  DESIGN reference: `Goals`, `3. Add explicit speech execution metadata`
- [x] 2.4 Update curated bundle descriptors in `libs/models` to populate the new metadata correctly.
  DESIGN reference: `3. Add explicit speech execution metadata`
- [x] 2.5 Add unit tests covering all curated speech bundles so execution metadata is validated per bundle.
  2026-04-29 @codex-tts -- Added a catalog-wide `libs/models` regression test for the curated speech bundle descriptors, alongside targeted backend and core capability tests.
  DESIGN reference: `Testing Scope for PLAN`

## Phase 3. Introduce Buffered TTS Contracts

- [x] 3.1 Add `BufferedSpeechSynthesizer` to `libs/model/src/typed.rs`.
  DESIGN reference: `2. Split TTS into buffered vs streaming contracts`
- [x] 3.2 Add `BufferedVoiceCloneSynthesizer` to `libs/model/src/typed.rs`.
  DESIGN reference: `2. Split TTS into buffered vs streaming contracts`
- [x] 3.3 Decide whether current `SpeechSynthesizer` naming remains as the true-streaming contract or whether it should be renamed for clarity during migration.
  DESIGN reference: `2. Split TTS into buffered vs streaming contracts`, `Migration Strategy / Phase 2`
- [x] 3.4 Keep migration-compatible shims long enough to avoid a flag day across backends and examples.
  DESIGN reference: `Migration Strategy`
- [ ] 3.5 Add compile-time tests or examples that demonstrate the intended split between buffered and streaming synthesis traits.
  2026-04-29 @codex-tts -- Backend unit tests now exercise the shared buffered stream helper, but there is not yet a dedicated example binary showing composition at the API boundary.
  DESIGN reference: `API Ergonomics Examples`, `Testing Scope for PLAN`

## Phase 4. Migrate Buffered TTS Backends

- [x] 4.1 Migrate Piper to `BufferedSpeechSynthesizer`.
  DESIGN reference: `Current State`, `Migration Strategy / Phase 3`
- [ ] 4.2 Migrate Qwen3 ONNX TTS to `BufferedSpeechSynthesizer` and buffered clone support.
  2026-04-29 @codex-tts -- Not applicable on the current `feature/models` base because the Qwen3 ONNX curated path has already been removed.
  DESIGN reference: `Current State`, `Migration Strategy / Phase 3`
- [x] 4.3 Migrate qwen3-tts.cpp to `BufferedSpeechSynthesizer` and buffered clone support.
  DESIGN reference: `Current State`, `Migration Strategy / Phase 3`
- [x] 4.4 Introduce a shared buffered-audio stream adapter in `libs/model` if the backend migrations reveal enough common structure to justify it.
  DESIGN reference: `4. Centralize buffered TTS wrapper logic`
- [x] 4.5 Keep backend-local code focused on runtime startup, artifact resolution, and synthesis, rather than repeated stream-buffer plumbing.
  DESIGN reference: `4. Centralize buffered TTS wrapper logic`

## Phase 5. Preserve and Clarify Streaming Backends

- [x] 5.1 Keep Moonshine and Sherpa aligned to `StreamingTranscriber` with correct partial/final-only metadata.
  DESIGN reference: `1. Keep ASR type split, but tighten metadata`
- [x] 5.2 Explicitly classify Whisper as batch ASR in both code and metadata.
  DESIGN reference: `Current State`
- [ ] 5.3 Reserve true streaming TTS metadata and traits for backends that can emit meaningful first audio before full synthesis completion.
  2026-04-29 @codex-tts -- Metadata now distinguishes `speech_buffered` from `speech_stream`; a dedicated `StreamingSpeechSynthesizer` trait remains deferred.
  DESIGN reference: `2. Split TTS into buffered vs streaming contracts`, `Proposed true streaming TTS path`

## Phase 6. Update Higher-Level Composition

- [ ] 6.1 Update `libs/models` examples so they demonstrate the right typed path for each backend family:
  2026-04-29 @codex-tts -- Not started in this repo slice.
  batch ASR, streaming ASR, buffered TTS, and streaming TTS if available.
  DESIGN reference: `API Ergonomics Examples`
- [x] 6.2 Update voice-skill selection and composition logic to choose behavior from typed capabilities and execution metadata rather than backend name switches.
  2026-04-30 @codex-tts -- `feature/models` now carries the imported voice skill tree. `bins/voice-agent` resolves the selected curated model, derives buffered-vs-streaming / batch-vs-streaming behavior from capability metadata, and the voice playbook documents the same execution semantics explicitly.
  DESIGN reference: `5. Treat composition as a first-class design goal`, `Expected Impact on Voice Skills`
- [ ] 6.3 Add example or integration coverage for:
  2026-04-29 @codex-tts -- Deferred with Phase 6.
  buffered TTS -> playback,
  streaming ASR -> partial/final transcript,
  and streaming TTS + streaming ASR turn-taking where supported.
  DESIGN reference: `Expected Impact on Voice Skills`, `Testing Scope for PLAN`

## Phase 7. Documentation and Cleanup

- [ ] 7.1 Update `libs/model/docs/API.md` so the public contract documentation reflects the new speech execution split accurately.
  2026-04-29 @codex-tts -- No `libs/model/docs/API.md` exists yet in this repo; this remains follow-up documentation work.
  DESIGN reference: `API Ergonomics Examples`
- [ ] 7.2 Update `libs/models/docs/DESIGN_ASR.md`, `PLAN_ASR.md`, `DESIGN_TTS.md`, and `PLAN_TTS.md` if their current language assumes streaming where the implementation is actually buffered or batch.
  2026-04-29 @codex-tts -- Not started; those docs were not part of this initial repo slice.
  DESIGN reference: `Migration Strategy`
- [x] 7.3 Remove obsolete or misleading helper constructors such as `transcription_stream_only` / `speech_stream_only` if they no longer describe the real contract surface.
  2026-04-30 @codex-tts -- Removed the stale aliases from `libs/model` once the voice-agent and catalog callers had moved to explicit execution-mode descriptors.
  DESIGN reference: `Migration Strategy / Phase 5`

## Validation Matrix

- [ ] V1 `cargo fmt` passes for `libs/model` and `libs/models`.
  2026-04-29 @codex-tts -- Targeted `rustfmt --edition 2024` passed on the touched files. Repo-wide `cargo fmt` is currently blocked by an unrelated missing example file outside this scope.
- [x] V2 `cargo clippy -- -D warnings` passes for the speech-related feature combinations already used by CI and the curated model bundle checks.
- [ ] V3 Batch ASR example compiles and runs against Whisper with the batch contract.
- [ ] V4 Streaming ASR examples compile and run against Moonshine and Sherpa with the streaming contract.
- [ ] V5 Buffered TTS examples compile and run against Piper and qwen3-tts.cpp.
  2026-04-29 @codex-tts -- Backend crates and typed wrappers are validated, but example-binary execution is still open.
- [x] V6 Voice-skill selection logic correctly distinguishes buffered vs streaming TTS and batch vs streaming ASR.
  2026-04-30 @codex-tts -- Validated on the imported voice-agent runtime by starting models through curated selectors and letting capability metadata choose batch vs streaming ASR and buffered TTS behavior.
- [x] V7 Curated metadata tests fail closed when a bundle advertises a capability that does not match its typed backend contract.

## Notes

- This effort is intentionally additive first, then migratory.
- The first correctness target is truthful metadata.
- The first ergonomics target is eliminating backend-name inference in higher-level composition.

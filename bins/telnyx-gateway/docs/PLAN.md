# Telnyx Real-Time Voice Integration - Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-06 13:40 PDT | @codex-366-impl | Made M3 echo replies opt-in for testing via `--conversation-smoke-test` or `conversation smoke-test on`, and wired outbound `dial` to auto-attach conversation like inbound `answer`; default attached conversations now remain transcription-only unless a test handler is enabled. |
| 2026-06-06 13:12 PDT | @codex-366-impl | Applied M3 manual acceptance feedback: default conversation attach/answer is auto-approved, `conversation disapprove` cancels active TTS and returns calls to transcription-only, and status listener formatting is operator-friendly. |
| 2026-06-06 08:48 PDT | @codex-366-impl | Addressed PR #400 review round 1: removed dead sink wording, added manual proposal approval commands, clarified source/backend behavior, documented final-transcript-triggered barge-in latency, and expanded core coverage. |
| 2026-06-05 23:34 PDT | @codex-366-impl | Implemented M3 full-duplex composition: conversation attach/detach/status/mode commands, final transcript forwarding, provider-neutral conversation command routing, Piper auto-say via the M2 speech queue, barge-in cancellation, selected-call chat state, and build/test/clippy verification with `sherpa piper`. |
| 2026-06-04 | @codex-369-rv | Removed the ASR startup-default flag/config path while keeping `kroko-2025` as the code default, and made the TTS factory boundary backend-neutral by normalizing synthesized output to signed 16-bit PCM plus source sample-rate metadata before packetization. |
| 2026-06-04 | @codex-369-rv | Aligned milestone 2 TTS operator UX with ASR: `tts list`, `tts status`, and `tts use piper` are shared TUI/socket commands, `tts status` reports clear availability, and `dial` tells operators to wait for media before running `speak`. |
| 2026-06-03 23:30 PDT | @codex-369-rv | Captured M2 live-test hardening: Piper eSpeak-ng data must be auto-detected or fail loudly, outbound speech is prebuffered as continuous utterance audio before resample/packetize, the media task owns 20 ms pacing, silence keepalive is withheld during active speech, and frame interval/underrun logs diagnose choppy playback. |
| 2026-06-03 21:49 PDT | @codex-369-rv | Locked milestone 2 outbound TTS to the single bidirectional RTP media WebSocket with an outbound frame queue, cancellable `speak`, always-live inbound ASR during playback, Telnyx `clear`/`mark`, `stream_track=inbound_track`, TUI/socket parity, and explicit Outbound Voice Profile live-test prerequisite. |
| 2026-06-02 | @codex-371-impl | Added Sherpa artifact A/B selection for current 2023 Zipformer vs the newer Kroko 2025 English Zipformer and recorded that actual WER/latency scoring is blocked until the private golden WAV/reference artifacts are copied onto this host. |
| 2026-06-01 | @codex-371-impl | Added M1.5 A/B replay infrastructure: a golden corpus manifest, selectable replay backends, and comparable WER/token-error/latency reports while leaving model integration and tuning sidecars pending. |
| 2026-06-01 | @codex-371-impl | Implemented the #371 R1 prerequisite and recorded the updated models-first M1.5 sequence: build backend A/B infrastructure and model comparisons before hotwords, endpointing, decoder tuning, or normalization. |
| 2026-06-01 | @codex-364-impl | Fixed capture WAV finalization, added a `replay-capture` WER harness over `asr-input-16khz.wav`, and split Sherpa-only ASR quality tuning into milestone 1.5 (#370) for hotwords, model/decoder A/B, and separately scored normalization. |
| 2026-06-01 | @codex-364-impl | Recorded the first outbound ASR-only live attempt blocker: Telnyx returned `403 D38` because the Call Control application has no Outbound Voice Profile assignment; live validation resumes after Telnyx account setup. |
| 2026-06-01 | @codex-364-impl | Added an ASR-only outbound live-test command, `test dial-transcribe <to> [--from <from>]`, that reuses Telnyx outbound dial setup, bidirectional RTP media, silence keepalive, capture, and Sherpa transcription while leaving TTS for milestone 2. |
| 2026-06-01 | @codex-364-impl | Recorded post-fix live validation for two `L16 16 kHz` inbound calls, with measured WER around `16-18%`, and fixed bare `answer` to prefer the single waiting inbound call when stale ended calls remain selected. |
| 2026-06-01 | @codex-364-impl | Corrected the milestone 1 Telnyx `L16` comparison path to decode observed WebSocket `L16` payloads as little-endian PCM, added provider-neutral little-endian L16 helpers, and added a fixed read-aloud reference for WER checks. |
| 2026-06-01 | @codex-364-impl | Added the post-live-test ASR quality loop for milestone 1: gateway media capture writes raw Telnyx JSONL, decoded inbound WAV, Sherpa input WAV, transcript JSONL, and manifest files, while replayable config can switch answer media between the known-good `PCMU 8 kHz` path and an `L16 16 kHz` comparison run. |
| 2026-06-01 | @codex-364-impl | Updated milestone 1 ASR quality work to use upstream `sherpa-onnx` for online recognition, endpointing, and static native archive linking; normal speech pauses now stay in one ASR session while repeated-token suppression remains a reset safety valve. |
| 2026-06-01 | @codex-364-impl | Added a milestone 1 agent-assisted live-test control path: `--socket` starts a local command socket with line-oriented Motlie driver commands, JSON command responses, `call show` transcript snapshots, and `shutdown`; milestone 4 still owns richer event polling, request IDs, socket/TUI mux validation, and appserver integration. |
| 2026-06-01 | @codex-364-impl | Added milestone 1 ASR-session reset work from live long-call logs: speech resumed after sustained silence opens a fresh Sherpa session, repeated-token hallucinations request a reset, and the TUI keeps one assembled call transcript because current Sherpa finals are word-level. |
| 2026-06-01 | @codex-364-impl | Updated milestone 1 live behavior to answer with bidirectional PCMU RTP and send outbound silence keepalive after Telnyx reported normal caller-side hangups in receive-only mode; added assembled transcript display above raw partial/final events. |
| 2026-06-01 | @codex-364-impl | Added milestone 1 live-call termination diagnostics: preserve Telnyx hangup/source/SIP fields in call state, selected-call detail, structured logs, and webhook tests. |
| 2026-06-01 | @codex-364-impl | Refined milestone 1 live handling: suppress initial silence and sustained low-energy tails before Sherpa, suppress repeated-token transcript hallucinations from the TUI transcript stream while logging them as suppressed events, and align the shell cursor by rendered rows when output wraps. |
| 2026-06-01 | @codex-364-impl | Merged the TUI command input and REPL history into one shell-style left pane. |
| 2026-06-01 | @codex-364-impl | Added milestone 1 ASR start-of-speech gating: low-energy initial telephony frames are logged and withheld from Sherpa until speech energy is detected. |
| 2026-06-01 | @codex-364-impl | Moved TUI-mode tracing away from the terminal by default; `--tui` writes structured logs to `telnyx-gateway.log` unless `--log-file` overrides the path. |
| 2026-06-01 | @codex-364-impl | Recorded the now-superseded receive-only `PCMU` live-test attempt; later live evidence moved milestone 1 to bidirectional `PCMU` RTP silence keepalive. |
| 2026-05-31 | @codex-364-impl | Clarified socket mode as the local agent-tooling surface: NDJSON command execution, structured snapshots, cursor-based event polling, and optional tmux/mstream wake-up notifications are socket responsibilities, while webhooks/Control API remain appserver integration surfaces. |
| 2026-05-31 | @codex-364-impl | Updated gateway ORT guidance to use `ort/download-binaries`: Cargo downloads and statically links `libonnxruntime.a`; no `ORT_LIB_PATH`, dynamic-link, vendored ORT, or source build. |
| 2026-05-31 | @codex-364-impl | Superseded earlier manual ONNX Runtime provisioning guidance with the `ort/download-binaries` static-link policy above. |
| 2026-05-31 | @codex-364-impl | Superseded earlier shared-library guidance with static linkage through the downloaded `libonnxruntime.a` archive. |
| 2026-05-30 | @codex-358-research | Addressed review round 1 by co-locating M4 with M1-M3, marking Control API/webhook Phase 6 work as M4 definition/stub work, adding M3 conversation webhook events, aligning M1 structured-log acceptance with #364, and updating operator module trees. |
| 2026-05-30 | @codex-358-research | Reviewed the design and split product tracking into four milestone issues: M1 inbound TUI transcription (#364), M2 outbound TUI dialer/TTS (#365), M3 full-duplex TUI chat conversation (#366), and M4 external socket/webhook/appserver integration (#367). |
| 2026-05-30 | @codex-358-research | Added startup-selected operator input modes: `--tui` for the local terminal UI, `--socket <path>` for headless Unix-domain command control, and a command-source mux that serializes both sources through one dispatcher when enabled together. |
| 2026-05-30 | @codex-358-research | Added explicit Control API work for app-server call discovery and per-call attachment: `GET /api/v1/calls`, `GET /api/v1/calls/{call_id}`, and call attachments that bind a registered webhook subscription to one call's events/transcripts. |
| 2026-05-30 | @codex-358-research | Refined the TUI work items around a right-side call roster plus selected-call detail pane; inbound calls create highlighted pending/waiting rows, preserve selection when another call is active, and expose transcript/status/action hints for the selected call. |
| 2026-05-30 | @codex-358-research | Switched operator examples to assume Tailscale Funnel public URLs by default; ngrok remains only an alternate tunnel option. |
| 2026-05-30 | @codex-358-research | Added replayable state persistence: `state dump [path]`, `shutdown [dump_path]`, and startup `--load <dump_path>` rehydrate durable Telnyx/app-server configuration by replaying idempotent gateway commands. |
| 2026-05-30 | @codex-358-research | Clarified application webhook delivery: gateway events are outbound HTTP `POST` requests to registered app-server URLs, acknowledged by `2xx`, retried on failure, and separate from app-server Control API calls back into the gateway. |
| 2026-05-30 | @codex-358-research | Added the external automation surface: gateway-emitted application webhooks plus an authenticated Gateway Control API for programmatic inbound answer/transcription and outbound dial/say TTS service flows. |
| 2026-05-30 | @codex-358-research | Reframed `bins/telnyx-gateway` as an operator-driven TUI/REPL app: it starts an idle listener, uses `motlie-driver` commands for Telnyx app/number setup, surfaces pending inbound calls in the right call roster/detail area, and only answers after explicit operator or inbound-mode selection. |
| 2026-05-30 | @codex-358-research | Reworked sequencing around three composable milestones: inbound transcription to `TranscriptSink`, outbound `motlie-driver` dial/say to TTS, opaque provider-neutral call handles, and a later conversation bridge connecting transcript events to outbound speech commands. |
| 2026-05-30 | @codex-358-research | Split provider-neutral application control from telephony vocabulary: `motlie_voice::app` owns `ConversationHandler`, `DtmfHandler`, `IvrNavigator`, and `ConversationContext`, while `motlie_voice::telephony` owns `DtmfDigit`, `CallAction`, provider-neutral events, and track/direction vocabulary. |
| 2026-05-30 | @codex-358-research | Moved the Telnyx PLAN under `bins/telnyx-gateway/docs/PLAN.md`, renamed its paired design link to `DESIGN.md`, and collapsed the former Telnyx adapter-crate tasks into `bins/telnyx-gateway` module and binary work. |
| 2026-05-30 | @codex-358-research | Aligned the remaining PLAN reorder tasks with the DESIGN boundary: Telnyx maps `media.chunk` to provider-neutral sequence metadata, `libs/voice` reorders generic sequenced frames before decode, and phase 3 now explicitly includes the i16 resampler wrapper plus missing `i16_to_f32` helper. |
| 2026-05-30 | @codex-358-research | Rebased implementation sequencing around the landed `motlie-voice` crate from PR #209. The Telnyx slice now extends existing `PcmFrame`, conversion, and resampling surfaces, adds missing codecs/packetization/stages, requires anti-aliased resampling before live calls, and records Piper buffered/CUDA caveats. |
| 2026-04-17 | @codex-macmini-telnyx | Tightened the execution policy further so no initial acceptance criteria, examples, or live validation steps can rely on Moonshine or Qwen3-TTS. The first complete Telnyx slice is Sherpa + Piper only. |
| 2026-04-17 | @codex-macmini-telnyx | Tightened the PLAN so Sherpa + Piper is the explicit first vertical slice. Follow-on pairings such as Sherpa + Qwen3-TTS and Moonshine + Qwen3-TTS are now deferred to a later phase instead of treated as peer initial targets. |
| 2026-04-16 | @codex-macmini-telnyx | Reworked the PLAN to match the hierarchy separating provider-agnostic `libs/voice` from Telnyx-specific gateway modules. |
| 2026-04-15 | @codex-macmini-telnyx | Initial Telnyx integration PLAN derived from `DESIGN.md`. Covers the transport gateway, codec normalization, inbound and outbound call flows, application orchestration, and validation. |

Derived from [DESIGN.md](./DESIGN.md). This PLAN assumes brownfield work against the existing `libs/model`, `libs/models`, and landed `libs/voice` (`motlie-voice`) speech stack and now explicitly separates provider-neutral voice infrastructure from the Telnyx-specific transport adapter.

Execution policy for the initial implementation:

- the gateway starts as an idle operator-controlled application with an HTTP/WebSocket listener; `--tui` enables the local TUI, `--socket <path>` enables headless Unix-domain command control, and the gateway does not answer inbound calls until an operator command source enables inbound mode or runs `answer`
- in milestone 4 (#367), local agents must be able to drive the gateway through the Unix-domain socket using NDJSON command requests, structured command results, call snapshots, and cursor-based event polling; authenticated Gateway Control API and gateway application webhooks remain separate appserver integration surfaces
- milestone 1 (#364) is inbound calls answered and managed in the TUI, with ASR transcription in the selected-call detail pane
- milestone 1.5 (#371) is Sherpa-only ASR quality tuning using captured audio replay, WER reports, hotwords/context bias, and model/decoder A/B before considering non-Sherpa ASR
- milestone 2 (#365) is outbound dialing plus TTS driven through the shared command engine from either the TUI or the Unix-domain socket; selected-call detail text input is a TUI affordance over the same `speak` command path
- milestone 3 (#366) is full-duplex conversation driven by a TUI chat interface over the selected call
- milestone 4 (#367) is external integration: local agent socket tooling, application webhooks, Gateway Control API call attachment/read flows, and a harness appserver
- `Sherpa + Piper` remains the first complete duplex backend pairing once milestone 3 exists
- `Moonshine` and `Qwen3-TTS` may influence generic pipeline design, but they must not add blocking requirements to phases 1 through 9
- all work needed only for those follow-on pairings belongs in phase 10 or later
- phases are capability layers; milestones #364-#367 are shippable increments gated in Phase 8, so Phase 6 API/webhook tasks define contracts and stubs until their milestone 4 implementation gate

---

## Phase 1: Workspace and Crate Skeleton

Extend the existing provider-neutral voice layer and add the Telnyx-specific deployable gateway under `bins/telnyx-gateway`.

### 1.1 - Workspace shape

- [ ] Confirm `libs/voice` (`motlie-voice`) from PR #209 is the provider-agnostic voice crate to extend; do not add a second voice crate.
  DESIGN reference: `Overview`, `Crate Hierarchy and API Surfaces`
- [ ] Add the deployable Telnyx gateway crate at `bins/telnyx-gateway`.
  DESIGN reference: `Overview`, `Crate Hierarchy and API Surfaces`
- [ ] Keep provider-neutral media, runtime, and application traits out of `bins/telnyx-gateway`.
  DESIGN reference: `Provider-Neutral API Rule`
- [ ] Keep Telnyx webhook, WebSocket, and REST schema types out of `libs/voice`.
  DESIGN reference: `Provider-Neutral API Rule`

### 1.2 - `libs/voice` module skeleton

- [ ] Preserve and reuse the landed `libs/voice/src/{frame.rs,wav.rs,pipeline/convert.rs,pipeline/resample.rs}` surfaces.
  DESIGN reference: `Crate Hierarchy and API Surfaces`, `Typed Media Values`
- [ ] Add missing Telnyx-slice modules `app/`, `runtime/`, `codec/`, and `telephony/` under `libs/voice/src/`, and extend `pipeline/` with stage composition, reorder, chunk, and packetize modules.
  DESIGN reference: `Crate Hierarchy and API Surfaces`
- [ ] Add crate-level `error.rs` and `config.rs`.
  DESIGN reference: `Crate Hierarchy and API Surfaces`

### 1.3 - `bins/telnyx-gateway` module skeleton

- [ ] Add `Cargo.toml`, `src/{main.rs,lib.rs,error.rs,cli.rs,serve.rs,logging.rs}`, `operator/`, `api/`, `events/`, and Telnyx-specific `webhook/`, `call_control/`, `media/`, and `adapter.rs` modules under `bins/telnyx-gateway/`.
  DESIGN reference: `Crate Hierarchy and API Surfaces`
- [ ] Add `operator/{commands.rs,state.rs,session.rs,persistence.rs,tui.rs,socket.rs,mux.rs}` for the `motlie-driver` command family, source-local operator sessions, optional TUI, optional Unix-domain command socket, command-source mux, shared gateway runtime state, replayable state dumps, and command/status routing.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Crate Hierarchy and API Surfaces`
- [ ] Add TUI view models for `CallRosterItem`, selected-call detail, unread event counts, and selection state derived from `SessionRegistry`.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Crate Hierarchy and API Surfaces`
- [ ] Add startup flags `--tui` and `--socket <path>`; allow either, both, or neither when the process relies only on `--load` and/or authenticated HTTP Control API, and route both live local input sources through one serialized command dispatcher when both are enabled.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Gateway Configuration Requirement`
- [ ] Add `api/{control.rs,auth.rs,subscriptions.rs}` and `events/{envelope.rs,dispatcher.rs,delivery.rs}` scaffolding for the external Control API and gateway application webhook delivery; implement and validate these surfaces in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Crate Hierarchy and API Surfaces`
- [ ] Start the gateway as an idle listener by default; do not register Telnyx apps, bind numbers, answer inbound calls, or dial outbound calls until a configured command source or authenticated Control API request does so.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Staged Build Strategy`
- [ ] Keep `bins/telnyx-gateway` focused on Telnyx protocol mapping, Telnyx call control, process configuration, and wiring.
  DESIGN reference: `Crate Hierarchy and API Surfaces`

## Phase 2: Provider-Neutral Application and Runtime Contracts

Define the reusable voice application surface in `libs/voice`.

### 2.1 - Application contracts and telephony vocabulary

- [ ] Add application-level control traits and state in `libs/voice/src/app/`: `ConversationContext`, `ConversationHandler`, `DtmfHandler`, and `IvrNavigator`.
  DESIGN reference: `Conversation Handler Contract`, `v1.1: DTMF and Call Control`, `Crate Hierarchy and API Surfaces`
- [ ] Add `TranscriptEvent` and `TranscriptSink` so milestone 1 can deliver ASR partials/finals to stdout, tests, tmux, or a later conversation bridge without requiring TTS.
  DESIGN reference: `Staged Build Strategy`, `Inbound Handler Surface`
- [ ] Add provider-neutral conversation outputs such as `ConversationCommand::Say` and `ConversationCommand::Call` so milestone 3 can route handler decisions into outbound speech or call-control actions.
  DESIGN reference: `Conversation Handler Contract`, `Staged Build Strategy`
- [ ] Add provider-neutral telephony vocabulary in `libs/voice/src/telephony/`: `DtmfDigit`, `CallAction`, `IvrEvent`, call/media lifecycle events, and track/direction markers.
  DESIGN reference: `motlie_voice::telephony Surface`, `v1.1: DTMF and Call Control`, `Crate Hierarchy and API Surfaces`
- [ ] Keep call handles and call context provider-neutral: raw Telnyx fields such as `call_control_id`, `call_session_id`, `stream_id`, and `connection_id` stay in `bins/telnyx-gateway` mapping/configuration.
  DESIGN reference: `Staged Build Strategy`, `Provider Adapter Boundary`, `motlie_voice::telephony Surface`
- [ ] Keep these contracts and vocabulary provider-neutral so they can be reused by future Twilio or SIP adapters.
  DESIGN reference: `Provider-Neutral API Rule`

### 2.2 - Static runtime selection

- [ ] Add provider-neutral static-dispatch ASR/TTS runtime wrappers in `libs/voice/src/runtime/`.
  DESIGN reference: `Static-Dispatch Model Injection`, `Crate Hierarchy and API Surfaces`
- [ ] Reuse `motlie_model::typed::{AudioBuf, StreamingTranscriber, TranscriptionSession, SpeechSynthesizer, SpeechStream, SynthesisRequest}` as the only model boundary for Telnyx ASR/TTS orchestration.
  DESIGN reference: `Overview`, `Recommended ASR/TTS Stack`
- [ ] Map compiled `motlie_models::AsrModels` and `motlie_models::TtsModels` selections into closed runtime enums or generic wrappers, reusing the capability-driven selection pattern from `bins/voice-agent`.
  DESIGN reference: `Closed-Enum Selection`, `Build-Time Model Surface`
- [ ] Document the required `motlie-models` feature flags in code comments or crate docs near the runtime selection layer.
  DESIGN reference: `Build-Time Model Surface`

### 2.3 - Pipeline controller contracts

- [ ] Add `InboundAsrPipeline` as the high-level controller consumed by inbound call handlers.
  DESIGN reference: `Pipeline Controller Contracts`, `Inbound Call Handler Design`
- [ ] Add `OutboundTtsPipeline` and `OutboundSpeechController` as the high-level controller consumed by operator commands, socket clients, mstream/broadcast bridges, tests, and conversation handlers.
  DESIGN reference: `Pipeline Controller Contracts`, `Driver REPL Dialer Surface`
- [ ] Keep these controllers source-agnostic: TUI commands, socket commands, mstream broadcast, fixture replay, and conversation handlers should all call the same outbound `speak()` path.
  DESIGN reference: `Staged Build Strategy`, `Driver REPL Dialer Surface`

## Phase 3: Typed Media Pipeline in `libs/voice`

Build the provider-neutral media adaptation pipeline with explicit stage contracts.

### 3.1 - Media values and markers

- [ ] Reuse the existing `motlie_voice::frame::PcmFrame<const RATE_HZ, C, E>` over `motlie_model::typed::AudioBuf`; do not define a duplicate `PcmFrame`.
  DESIGN reference: `Typed Media Values`
- [ ] Add `EncodedFrame<C>` and transport-specific metadata/envelope types without moving sequence or end-of-stream semantics into a parallel PCM frame type.
  DESIGN reference: `Typed Media Values`
- [ ] Add only the missing marker types and marker traits in `libs/voice/src/pipeline/markers.rs`, reusing `motlie_model::typed::{Mono,Stereo,ChannelLayout}` for channels and const generics for sample rate.
  DESIGN reference: `Marker Traits`
- [ ] Add stage traits in `libs/voice/src/pipeline/stage.rs`.
  DESIGN reference: `Stage API Surface`

### 3.2 - Stage builder and safety checks

- [ ] Add typed stage-composition helpers in `libs/voice/src/pipeline/builder.rs`.
  DESIGN reference: `Compile-Time Pipeline Assembly`
- [ ] Add compile-time or unit-level proofs that adjacent stages must agree on `In` and `Out`.
  DESIGN reference: `Compile-Time Pipeline Assembly`, `Recommended Safety Properties`

### 3.3 - Core pipeline stages

- [ ] Extend the existing `motlie_voice::pipeline::{convert,resample}` modules rather than copying their decode/downmix/sample-conversion helpers into Telnyx code.
  DESIGN reference: `Media Adaptation Pipeline`, `Required Concrete Stage Inventory`
- [ ] Implement reorder, chunk, and packetize stages in `libs/voice/src/pipeline/`.
  DESIGN reference: `Media Adaptation Pipeline`, `Required Concrete Stage Inventory`
- [ ] Replace or wrap the existing `LinearInterpolator` with an anti-aliased resampler (`rubato`, `dasp`, or a documented polyphase/windowed-sinc implementation) before live Telnyx validation.
  DESIGN reference: `Behavior Contracts`, `Resampling and Format Normalization`
- [ ] Add the provider-neutral `i16_to_f32` conversion helper in `motlie_voice::pipeline::convert` and implement i16 telephony resampler wrappers as `i16 -> f32 -> resample_f32 -> i16` over the f32 `Resampler` path until an i16-native resampler is justified.
  DESIGN reference: `Behavior Contracts`, `Resampling and Format Normalization`
- [ ] Implement provider-neutral G.711 and `L16` codecs in `libs/voice/src/codec/`, including explicit big-endian and little-endian L16 helpers so provider adapters choose the byte order proven by their media captures.
  DESIGN reference: `Codec and Container Gaps`, `Crate Hierarchy and API Surfaces`
- [ ] Keep stage responsibilities split; do not collapse decode, resample, and packetization into one opaque adapter.
  DESIGN reference: `Required Concrete Stage Inventory`

## Phase 4: Milestone 1 - Inbound Transcription

Assemble and validate the first useful Telnyx slice: inbound call handling plus Sherpa ONNX streaming ASR into a transcript sink. This phase deliberately does not require outbound TTS or a conversation handler.

### 4.1 - Sherpa inbound assembly

- [ ] Implement the Sherpa inbound path:
  Telnyx sequence-map -> provider-neutral reorder -> decode -> mono normalize -> `16 kHz` resample -> Sherpa ingest flow.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Require upstream `sherpa-onnx` static native linkage for the Sherpa live-test path: Cargo downloads and statically links the prebuilt `sherpa-onnx` native archive, including Sherpa's internal ONNX Runtime library; the runbook must not set `ORT_LIB_PATH`, `ORT_PREFER_DYNAMIC_LINK`, or `LD_LIBRARY_PATH`, and must not ask operators to build ONNX Runtime from source.
  DESIGN reference: `Recommended ASR/TTS Stack`

### 4.2 - Transcript sinks

- [ ] Implement `StdoutTranscriptSink` for the first live inbound validation.
  DESIGN reference: `Staged Build Strategy`, `Inbound Handler Surface`
- [ ] Implement gateway-local `TuiTranscriptSink` so milestone 1 can render transcripts to the selected-call detail pane; keep `WebhookTranscriptSink` for milestone 4 external integration.
  DESIGN reference: `Staged Build Strategy`, `Application Webhooks and Gateway Control API`
- [ ] Add a test sink that records transcript events for assertions.
  DESIGN reference: `Staged Build Strategy`, `Testing Scope for PLAN`
- [ ] Add a documented extension point for a future tmux sink that maps final transcripts to `motlie_tmux::KeySequence` and `Target::send_keys()`.
  DESIGN reference: `Staged Build Strategy`, `Inbound Handler Surface`

### 4.3 - First-milestone constraints

- [ ] Add explicit assembly coverage for Sherpa inbound ASR only in the first implementation slice.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Reject undeclared pairings cleanly instead of falling back to guessed pipeline behavior.
  DESIGN reference: `Concrete Backend Requirements Matrix`
- [ ] Keep outbound TTS, conversation handlers, Moonshine, and Qwen3-TTS support out of the first live inbound-transcription validation path even if the abstractions are designed broadly enough to support them later.
  DESIGN reference: `Recommended ASR/TTS Stack`, `Concrete Combination Requirements`
- [ ] Do not add first-milestone code paths, examples, or acceptance checks that require Piper, Moonshine-specific rechunking, or Qwen3-TTS-specific `AudioBuf<f32, 24_000, Mono>` conversion.
  DESIGN reference: `Concrete Combination Requirements`

## Phase 5: Telnyx Protocol and Adapter Layer

Wire the Telnyx-specific schema and gateway adapter modules on top of `libs/voice`.

### 5.1 - Telnyx protocol types

- [ ] Add Rust types for Telnyx webhook payloads needed by the first milestone:
  `call.initiated`,
  `call.answered`,
  `streaming.started`,
  `streaming.stopped`,
  `streaming.failed`.
  DESIGN reference: `Telnyx API Research`, `Crate Hierarchy and API Surfaces`
- [ ] Add Rust types for WebSocket events:
  `connected`,
  `start`,
  `media`,
  `mark`,
  `clear`,
  `dtmf`,
  `error`,
  `stop`.
  DESIGN reference: `WebSocket Event and Chunk Format`, `Crate Hierarchy and API Surfaces`
- [ ] Add parser tests against captured JSON fixtures.
  DESIGN reference: `Testing Scope for PLAN`

### 5.2 - Telnyx adapter mapping

- [ ] Map Telnyx `start.media_format` into provider-neutral typed transport frames.
  DESIGN reference: `Telnyx Media Schema Mapping`
- [ ] In `bins/telnyx-gateway`, map Telnyx `media.chunk` into provider-neutral per-track sequence metadata and create `EncodedFrame<C>` values.
  DESIGN reference: `Telnyx Media Schema Mapping`
- [ ] In `bins/telnyx-gateway`, map Telnyx call IDs to provider-neutral call/session/media-stream IDs before passing context into `motlie_voice::app` handlers.
  DESIGN reference: `Staged Build Strategy`, `Provider Adapter Boundary`
- [ ] In `libs/voice`, run `SequencedFrameReorder<C>` over generic sequenced frames before codec decode.
  DESIGN reference: `Required Concrete Stage Inventory`
- [ ] Emit one gateway-local monotonic sequence for provider-neutral frames and model chunks.
  DESIGN reference: `Telnyx Media Schema Mapping`
- [ ] Default the conversational path to `inbound_track` only.
  DESIGN reference: `Why stream_track=inbound_track`

### 5.3 - Session registry and lifecycle

- [ ] Add a session registry keyed by `call_control_id` and `stream_id`.
  DESIGN reference: `Recommended Integration Shape`
- [ ] Model `PendingCallSession` versus fully started media sessions so webhook order and WebSocket order can arrive independently.
  DESIGN reference: `Inbound Call Handler Design`, `Outbound Call Handler Design`
- [ ] Add timeout and cleanup behavior for abandoned sessions.
  DESIGN reference: `Reordering and Jitter Handling`

## Phase 6: Telnyx REST and Webhook Handling

Wire call-control operations for inbound and outbound calls.

### 6.1 - REST client

- [ ] Add a small Telnyx REST client for `answer`, `dial`, `hangup`, and `streaming_start`.
  DESIGN reference: `Call Control and Media Transport`
- [ ] Extend the Telnyx REST client with the provisioning calls needed by the operator command surface: list/create/select Call Control Applications, update the application webhook URL, list phone numbers, and bind a phone number to a connection ID.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Make the outbound `dial` request support:
  `stream_url`,
  `stream_track`,
  `stream_bidirectional_mode`,
  `stream_bidirectional_codec`,
  `stream_bidirectional_sampling_rate`,
  `stream_bidirectional_target_legs`,
  `stream_establish_before_call_originate`,
  `send_silence_when_idle`.
  DESIGN reference: `Outbound Call Handling`
- [ ] Make the inbound `answer` request support inline stream attachment so `answer` and media setup occur in one step.
  DESIGN reference: `Inbound Call Handling`

### 6.2 - Webhook server

- [ ] Add an HTTP endpoint that accepts Telnyx voice webhooks and returns `200 OK` deterministically.
  DESIGN reference: `Inbound Call Handling`
- [ ] Add optional webhook signature verification if available in the chosen Telnyx app configuration.
  DESIGN reference: `Gap Analysis`
- [ ] On `call.initiated`, create or update a `PendingInbound` / `WaitingForAnswer` session, render a highlighted row in the top call roster, increment its unread event count, append a global status event, and auto-select it only if no active call is selected.
  DESIGN reference: `Inbound Call Handler Design`, `Operator REPL and TUI Control Surface`
- [ ] Do not issue `answer` while inbound mode is disabled or manual without an operator `answer` command.
  DESIGN reference: `Inbound Call Handler Design`, `Operator REPL and TUI Control Surface`
- [ ] Implement inbound modes `Disabled`, `Manual`, and `AutoTranscribe`; make `Manual` the first milestone default after `inbound enable --manual`.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Inbound Call Handler Design`
- [ ] On streaming failure or terminal call events, tear down session state and release model resources.
  DESIGN reference: `Recommended Integration Shape`, `Open Concerns`

### 6.3 - Operator provisioning and call-control commands

- [ ] Add `motlie-driver` commands for gateway status, config, and persistence: `status`, `listener status`, `state dump [path]`, `shutdown [dump_path]`, `config show`, `config set webhook-url <https-url>`, `config set media-url <wss-url>`, `config set from-number <e164>`, and `config set state-path <path>`.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Define `motlie-driver` commands for external automation: `api token create`, `api token revoke`, `webhook subscription list`, `webhook subscription add <url> --events <event,...>`, `webhook subscription upsert <subscription-id> <url> --events <event,...> --secret-ref <secret-ref>`, `webhook subscription remove <subscription-id>`, and `webhook subscription test <subscription-id>`; implement and validate them in milestone 4 (#367).
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Application Webhooks and Gateway Control API`
- [ ] Add Telnyx provisioning commands: `telnyx app list`, `telnyx app create <name>`, `telnyx app use <connection-id>`, `telnyx app webhook set <https-url>`, `telnyx number list`, `telnyx number use <e164>`, and `telnyx number bind <e164> <connection-id>`.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Add inbound call commands: `inbound status`, `inbound enable --manual`, `inbound enable --auto-transcribe`, `inbound disable`, `calls`, `call use <call>`, `answer [call] [--sink tui|stdout|webhook:<subscription-id>]`, `reject [call]`, `hangup [call]`, and transcript follow/clear commands.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Inbound Call Handler Design`
- [ ] Route command results, Telnyx API responses, pending/active call state, media metadata, and transcript events to the global status stream, top call roster, or selected-call detail pane as appropriate.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Make `call use <call>` update source-local `selected_call`; allow `answer`, `reject`, `hangup`, and `speak <text>` to target that source-local selection when no call id is supplied, without changing another source's selection.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Define the Unix-domain socket command protocol as newline-delimited JSON frames containing `motlie-driver` command text, request IDs, structured command responses, and optional structured `data`; keep restrictive filesystem permissions, stale-socket safety, and optional peer credential checks where available. Implement and validate it in milestone 4.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Add socket-visible structured snapshots for agent polling: `calls --json`, `call get <call_id> --json`, and equivalent status/config resources that do not require parsing TUI text.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Add an in-memory gateway event journal with monotonic sequence cursors and a socket command `events poll --after <cursor> [--call <call_id>] [--types <event,...>] [--limit <n>]`; return structured `cursor_expired` errors when a client falls behind the retained ring buffer.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Add optional local agent wake-up notifiers, such as tmux send-keys or mstream channel notifications, that emit short hints telling an agent to poll the socket; treat these as opt-in hints rather than authoritative event delivery.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Define command-source mux ordering so TUI, socket, replay, and Control API commands enter the shared controller through one total-order dispatcher while responses route back to the source that submitted each command; implement and validate it in milestone 4.
  DESIGN reference: `Operator REPL and TUI Control Surface`

### 6.4 - Replayable gateway state

- [ ] Add startup `--load <dump_path>` support that replays a gateway command dump before enabling inbound handling or accepting live TUI, socket, or Control API mutations.
  DESIGN reference: `Replayable State Dumps`, `Gateway Configuration Requirement`
- [ ] Implement replayable dump generation as command text, not an opaque binary snapshot.
  DESIGN reference: `Replayable State Dumps`
- [ ] Include durable state in dumps: public webhook/media URLs, selected Telnyx app/connection, selected/bound phone number, default from number, inbound mode, webhook subscriptions/event filters, secret references, token metadata or token hash references, and selected ASR/TTS model names when configured through the operator command surface.
  DESIGN reference: `Replayable State Dumps`
- [ ] Exclude transient state from dumps: active calls, pending calls, media sessions, transcript history, in-flight TTS playback, and retry queues unless a later event journal explicitly persists them.
  DESIGN reference: `Replayable State Dumps`
- [ ] Ensure replay commands are idempotent and prefer `use`, `bind`, and `upsert` forms over create-only commands that would duplicate remote or local objects.
  DESIGN reference: `Replayable State Dumps`
- [ ] Never dump raw Telnyx API keys, application webhook HMAC secrets, or bearer tokens; persist only secret references or hashed token material.
  DESIGN reference: `Replayable State Dumps`

### 6.5 - Gateway Control API

Phase 6.5 defines the HTTP Control API contract and lightweight routing stubs only. Full implementation and validation belong to milestone 4 (#367), not M1, M2, or M3.

- [ ] Define authenticated call read endpoints: `GET /api/v1/calls` with state/direction filters and cursor pagination, plus `GET /api/v1/calls/{call_id}` with full call detail, media/transcript/TTS state, provider diagnostic IDs, attachments, last error, and recent timeline entries; implement and validate in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Define authenticated call attachment endpoints: `POST /api/v1/calls/{call_id}/attachments`, `GET /api/v1/calls/{call_id}/attachments`, and `DELETE /api/v1/calls/{call_id}/attachments/{attachment_id}`; implement and validate in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Define attachment semantics so an app server can bind an existing webhook subscription to one call's transcript/events before or after answer; deleting the attachment stops delivery for that subscription without hanging up the call. Implement and validate in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Define authenticated call-control endpoints: `POST /api/v1/calls/{call_id}/answer`, `POST /api/v1/calls/{call_id}/reject`, and `POST /api/v1/calls/{call_id}/hangup`; implement and validate the HTTP API in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Define outbound dialer/TTS endpoints: `POST /api/v1/calls`, `POST /api/v1/calls/{call_id}/say`, and `POST /api/v1/calls/{call_id}/interrupt`; implement and validate in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Outbound Call Handler Design`
- [ ] Define routing so Control API requests enter the same internal command/controller layer as operator commands and answer, dial, hangup, and say behavior cannot drift; implement and validate in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Operator REPL and TUI Control Surface`
- [ ] Define authentication and `Idempotency-Key` requirements for mutating Control API requests, including call attachment create/delete operations; implement and validate in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`

### 6.6 - Gateway application webhooks

Phase 6.6 defines the application webhook contract and event taxonomy only. Full subscription storage, delivery, retries, and harness validation belong to milestone 4 (#367).

- [ ] Define subscription CRUD endpoints: `POST /api/v1/webhook-subscriptions`, `GET /api/v1/webhook-subscriptions`, `GET /api/v1/webhook-subscriptions/{subscription_id}`, `DELETE /api/v1/webhook-subscriptions/{subscription_id}`, and `POST /api/v1/webhook-subscriptions/{subscription_id}/test`; implement and validate in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Define outbound application webhook delivery as HTTP `POST <subscription.url>` with `Content-Type: application/json`, the gateway event envelope as the body, and HMAC delivery headers; implement and validate in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Define delivery acknowledgement and retry semantics: treat any `2xx` webhook response as acknowledgement, and treat non-`2xx`, timeout, and connection failure as retryable delivery failure; implement and validate in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Define inbound service event types for milestone 4 delivery: `call.inbound.pending`, `call.answering`, `call.answered`, `media.started`, `transcript.partial`, `transcript.final`, `call.ended`, and `call.failed`.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Inbound Call Handler Design`
- [ ] Define outbound service event types for milestone 4 delivery: `call.outbound.created`, `call.dialing`, `call.ringing`, `call.answered`, `media.started`, `tts.playback.started`, `tts.playback.finished`, `tts.playback.failed`, `call.ended`, and `call.failed`.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Outbound Call Handler Design`
- [ ] Define milestone 3 conversation event types for milestone 4 delivery: `conversation.attached`, `conversation.command.created`, `conversation.command.completed`, and `conversation.detached`.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Conversation Handler Contract`
- [ ] Deliver application webhooks asynchronously with bounded retries; never block Telnyx webhook handling on subscriber delivery. Implement and validate in milestone 4 (#367).
  DESIGN reference: `Application Webhooks and Gateway Control API`

## Phase 7: Pipeline Orchestration

Connect the Telnyx adapter and the provider-neutral voice pipeline to the existing Motlie speech contracts.

### 7.1 - Inbound audio to ASR

- [ ] On Telnyx `start`, instantiate the correct typed inbound pipeline from `start.media_format` and the selected ASR runtime.
  DESIGN reference: `Telnyx Media Schema Mapping`, `Concrete Combination Requirements`
- [ ] Convert ordered inbound media frames into the selected ASR runtime's typed input, such as `AudioBuf<i16, 16_000, Mono>`, and call `TranscriptionSession::ingest()`.
  DESIGN reference: `Mapping Telnyx Frames to Typed ASR Input`
- [ ] Forward `TranscriptionUpdate` partials and finals as `TranscriptEvent` values to `TranscriptSink` without exposing Telnyx types.
  DESIGN reference: `Staged Build Strategy`, `Feeding the Conversation Handler`

### 7.2 - Outbound text to TTS

- [ ] Add a per-call outbound media channel, drained only by the WebSocket-owning media task, so TTS producers never write directly to the Telnyx socket.
  DESIGN reference: `M2-Safe Bidirectional Media Contract`, `Returning TTS Audio`
- [ ] Keep the inbound media read loop and ASR session active while outbound TTS frames are being synthesized, queued, and sent.
  DESIGN reference: `M2-Safe Bidirectional Media Contract`, `Why stream_track=inbound_track`
- [ ] Implement the Piper outbound path:
  `AudioBuf<i16, 22_050, Mono>` -> target-rate resample -> telephony packetize -> Telnyx encode.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Treat Piper as full-buffer synthesis followed by transport packetization; do not claim incremental TTS audio until the backend actually emits audio incrementally.
  DESIGN reference: `Transport Streaming vs Incremental TTS`, `Latency Budget`
- [ ] Split long `speak` text into sentence- or clause-sized `SynthesisRequest` jobs so milestone 2 does not enqueue one giant TTS buffer and milestone 3 can cancel between chunks.
  DESIGN reference: `Returning TTS Audio`, `M2-Safe Bidirectional Media Contract`
- [ ] Concatenate the returned Piper PCM chunks for one utterance before outbound resampling and packetization so resampler state is continuous across sentence/clause boundaries.
  DESIGN reference: `Returning TTS Audio`, `M2-Safe Bidirectional Media Contract`
- [ ] Pace outbound media writes from the WebSocket-owning media task at one fixed `20 ms` frame per `20 ms` tick; producers must fill the per-call queue, not write directly or burst frames to Telnyx.
  DESIGN reference: `M2-Safe Bidirectional Media Contract`
- [ ] Withhold silence keepalive while a speech job is active; log active-speech queue underruns instead of splicing silence into an utterance.
  DESIGN reference: `Returning TTS Audio`, `M2-Safe Bidirectional Media Contract`
- [ ] Record the Piper runtime mode used for validation. By default Piper is CPU-only because of #230; `MOTLIE_PIPER_ALLOW_CUDA=1` must be explicitly documented if enabled.
  DESIGN reference: `Recommended TTS: Piper`, `Open Concerns`
- [ ] Auto-detect common eSpeak-ng data paths for Piper phonemization and fail loudly with a clear operator error if the data is absent; do not silently send empty TTS audio.
  DESIGN reference: `Recommended TTS: Piper`
- [ ] Call the selected `SpeechSynthesizer` and drain its returned `SpeechStream` when `OutboundSpeechController::speak()` receives text.
  DESIGN reference: `Returning TTS Audio`, `Driver REPL Dialer Surface`
- [ ] Instantiate the correct typed outbound pipeline from the selected TTS runtime and Telnyx outbound codec.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Add interruption handling:
  stop active TTS generation,
  drop queued local outbound frames for the speech job,
  stop outbound packet send for that job,
  send Telnyx `clear` over the bidirectional RTP WebSocket.
  DESIGN reference: `Returning TTS Audio`
- [ ] Add Telnyx `mark` send/correlation so the selected-call detail pane and socket `status` can report queued, playing, completed, canceled, and failed speech jobs.
  DESIGN reference: `Returning TTS Audio`, `M2-Safe Bidirectional Media Contract`

### 7.3 - Conversation, DTMF, and IVR control

- [ ] Wire `ConversationHandler` as a bridge from selected `TranscriptEvent` values to `ConversationCommand` values.
  DESIGN reference: `Conversation Handler Contract`, `Staged Build Strategy`
- [ ] Wire `DtmfHandler` and provider-neutral `CallAction` mapping through the Telnyx adapter.
  DESIGN reference: `v1.1: DTMF and Call Control`
- [ ] Keep dialog and keypad policy out of `bins/telnyx-gateway`.
  DESIGN reference: `Provider-Neutral API Rule`
- [ ] Keep Telnyx-specific call-control translation in `bins/telnyx-gateway`; `motlie_voice::telephony::CallAction` is the provider-neutral command vocabulary, not a provider REST client.
  DESIGN reference: `motlie_voice::telephony Surface`, `Provider-Neutral API Rule`

## Phase 8: Product Milestones

Close the loop on independently useful product flows before combining them.

### 8.1 - Milestone 1 inbound TUI transcription flow (#364)

- [ ] Implement startup into an idle listener with `--tui`; the left pane is the command surface and the right side is split into call roster plus selected-call detail.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Keep the shell-style TUI prompt and cursor aligned to the rendered `>` prompt when command output spans multiple terminal rows.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Implement `inbound enable --manual` -> `call.initiated` -> highlighted pending/waiting call roster row -> optional `call use <call>` -> `answer [call]` -> `answer + streaming` -> WebSocket media -> ASR -> `TuiTranscriptSink` selected-call detail -> hangup flow.
  DESIGN reference: `Inbound Call Handler Design`, `Operator REPL and TUI Control Surface`
- [ ] Implement the milestone 1 local command-socket subset for agent-assisted live validation: headless `--socket <path>` startup, one command line in / one JSON response line out, `status`, `inbound enable --manual`, `calls`, `answer`, `call show`, and `shutdown`.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Getting Started: Local Deployment`
- [ ] Show call state, media metadata, assembled transcript text, recent partial/final transcript events, errors, and terminal call state in the selected-call detail pane.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Inbound Call Handler Design`
- [ ] Answer milestone 1 calls with bidirectional RTP and send outbound silence keepalive frames until milestone 2/M3 replaces that path with real TTS audio; default to the live-validated `PCMU 8 kHz` path, but allow operator config to request `L16 16 kHz` for measured ASR quality comparison.
  DESIGN reference: `Inbound Call Handler Design`, `Audio Codecs and Formats`
- [ ] Preserve Telnyx termination details from `streaming.stopped`, `streaming.failed`, `call.hangup`, and `call.ended` webhooks in selected-call detail and structured logs so live drops can be classified as caller hangup, provider timeout, media failure, or another carrier/SIP cause.
  DESIGN reference: `Inbound Call Handler Design`, `Operator REPL and TUI Control Surface`
- [ ] Gate ASR ingestion for milestone 1 by suppressing low-energy initial frames, allowing a seconds-scale post-speech low-energy hangover so upstream Sherpa endpointing can finalize utterances, suppressing sustained low-energy tails after that hangover, keeping the ASR session alive across normal resumed speech, and filtering pathological repeated-token Sherpa transcripts out of the TUI transcript stream while logging `transcript.suppressed_repeated_token` with call/stream/media metadata and opening a fresh ASR session only as a safety reset.
  DESIGN reference: `Inbound Call Handler Design`, `Testing Scope for PLAN`
- [ ] Keep inbound disabled by default at process startup; incoming webhooks must not be answered until the operator enables inbound handling.
  DESIGN reference: `Staged Build Strategy`, `Operator REPL and TUI Control Surface`
- [ ] Validate during live testing that `stream_bidirectional_target_legs=self` plus outbound PCMU silence keepalive prevents provider-normal early call termination in the initial single-leg ASR-only pattern.
  DESIGN reference: `Open Concerns`
- [ ] Add structured logs for the gateway call id, Telnyx diagnostic ids such as `call_control_id`, `call_session_id`, and `call_leg_id` when present, `stream_id`, observed codec, observed sample rate, and transcript partial/final events.
  DESIGN reference: `Open Concerns`
- [ ] Add operator-configurable capture for ASR quality debugging: raw Telnyx media JSONL, decoded inbound WAV at observed format, `16 kHz` Sherpa input WAV, transcript-event JSONL, and a manifest under a per-call/per-stream directory.
  DESIGN reference: `Inbound Call Handler Design`, `Testing Scope for PLAN`
- [ ] Finalize captured WAV files with finite RIFF/data sizes so standard readers can inspect duration/sample counts; keep the Motlie decoder tolerant of older indefinite-length streaming captures.
  DESIGN reference: `Inbound Call Handler Design`, `Testing Scope for PLAN`

### 8.1.5 - Milestone 1.5 Sherpa ASR quality tuning (#371)

- [x] Move Sherpa-specific repeated-token suppression and session-reset policy behind the ASR adapter before hotword/model work; keep Echo/non-Sherpa transcript events pass-through by default, and leave the shared media loop to record adapter-supplied decisions only. (@codex-371-impl, 2026-06-01 PDT)
  DESIGN reference: `Milestone 1.5: Sherpa ASR Quality Tuning`, `Inbound Call Handler Design`
- [x] Define the golden-WAV corpus manifest with reference text plus codec, sample rate, direction, capture path, and baseline fields; include the known `29.2%` L16 `16 kHz` capture and at least one PCMU `8 kHz` capture. (@codex-371-impl, 2026-06-01 PDT: added `bins/telnyx-gateway/corpus/asr-golden.json`; private WAVs and the exact outbound 65-word reference remain external artifacts to place at the manifest paths before scoring.)
  DESIGN reference: `Milestone 1.5: Sherpa ASR Quality Tuning`, `Testing Scope for PLAN`
- [x] Extend `replay-capture` into a backend-selectable A/B harness that runs captured WAVs through swappable `StreamingTranscriber` backends and reports transcript, WER, substitutions, deletions, insertions, token errors, and latency in a comparable report. (@codex-371-impl, 2026-06-01 PDT)
  DESIGN reference: `Milestone 1.5: Sherpa ASR Quality Tuning`, `Recommended ASR/TTS Stack`
- [x] Wire backend selection for replay A/B so candidates can be compared behind the existing typed ASR backbone without changing Telnyx media decode/capture behavior. (@codex-371-impl, 2026-06-01 PDT)
  DESIGN reference: `Recommended ASR/TTS Stack`
- [ ] Integrate and benchmark candidate models on the same golden corpus: newer Sherpa Zipformer first, then Nemotron/Parakeet candidates coordinated with #191/#369. (@codex-371-impl, 2026-06-02 PDT: `sherpa-zipformer-2023` and `sherpa-zipformer-kroko-2025` replay selectors are wired; WER/latency scoring is blocked until the L16 baseline and PCMU golden capture WAVs/reference are present. See `docs/ASR_ARTIFACT_AB.md`.)
  DESIGN reference: `Recommended ASR/TTS Stack`
- [ ] Defer Sherpa hotwords/context bias, endpointing tuning, decoder tuning, and post-ASR normalization until the model A/B results establish best-model WER and latency on the golden corpus.
  DESIGN reference: `Milestone 1.5: Sherpa ASR Quality Tuning`
- [ ] Keep raw ASR transcript, normalized transcript, and agent/LLM-corrected transcript as separate outputs if normalization is later implemented. WER acceptance must score raw ASR separately from any normalized-output metric.
  DESIGN reference: `Milestone 1.5: Sherpa ASR Quality Tuning`

### 8.2 - Milestone 2 outbound TUI/socket dialer/TTS flow (#365)

- [ ] Confirm live Telnyx outbound prerequisites before implementation validation: selected Call Control application/connection has an Outbound Voice Profile assignment, the `from` number is outbound-enabled for the account, and the gateway surfaces account-side errors such as `403 D38` instead of failing silently.
  DESIGN reference: `Outbound Call Handler Design`, `Getting Started: Local Deployment`
- [ ] Implement application-triggered `POST /v2/calls` with inline streaming parameters.
  DESIGN reference: `Outbound Call Handler Design`
- [ ] Use `stream_establish_before_call_originate=true` in the first outbound implementation.
  DESIGN reference: `Outbound Call Handler Design`
- [ ] Add outbound `motlie-driver` commands for `dial <phone-or-sip-uri> [--from <+e164>]`, `speak [call-id] <text...>`, `speak cancel [call-id]`, `hangup [call-id]`, `status [call-id]`, `tts list`, `tts status`, and `tts use piper`.
  DESIGN reference: `Driver REPL Dialer Surface`
- [ ] Route every M2 command through the shared command engine used by TUI, socket, and `--load` replay; source-local selected call and source-local command session state must remain isolated between TUI and each socket connection.
  DESIGN reference: `Driver REPL Dialer Surface`, `Operator REPL and TUI Control Surface`
- [ ] Add selected-call detail text input/composer for outbound TTS; submitting it must call the same `OutboundSpeechController::speak()` path as `speak [call-id] <text...>`.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Driver REPL Dialer Surface`
- [ ] Render outbound call state in the top roster and media session state, TTS synthesis status, and packet-send status in the selected-call detail pane.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Driver REPL Dialer Surface`
- [ ] Make socket `status [call-id]` report the same outbound call/media/TTS state needed for an agent to place an outbound call, drive TTS, cancel speech, and summarize the result without reading the TUI.
  DESIGN reference: `Driver REPL Dialer Surface`
- [ ] Keep Telnyx `connection_id`, stream URL, credentials, and bidirectional media flags in gateway configuration rather than adding them to provider-neutral outbound call requests.
  DESIGN reference: `Staged Build Strategy`, `Provider Adapter Boundary`
- [ ] Keep the TUI REPL and socket command surface as utterance sources only; mstream send-keys/broadcast bridges and fixture replay should call the same `OutboundSpeechController::speak()` path.
  DESIGN reference: `Staged Build Strategy`, `Driver REPL Dialer Surface`
- [ ] Support optional initial greeting once the media session is ready, implemented as a normal `speak()` call.
  DESIGN reference: `Outbound Call Handler Design`, `Driver REPL Dialer Surface`
- [ ] Document and exercise a manual M2 live-test runbook: start gateway with TUI and optional socket, load config, verify outbound profile/number, dial David, send one or more `speak` utterances, cancel one utterance with `speak cancel`, hang up, and inspect TUI/socket/log status.
  DESIGN reference: `Getting Started: Local Deployment`, `Driver REPL Dialer Surface`

### 8.3 - Milestone 3 full-duplex TUI chat conversation flow (#366)

- [x] Forward selected final transcript events from the media socket to `ConversationHandler`. (@codex-366-impl, 2026-06-06 08:48 PDT: corrected implementation note; forwarding is via `conversation::handle_final_transcript` on unsuppressed final transcript events for attached calls, not a separate sink adapter.)
  DESIGN reference: `Staged Build Strategy`, `Conversation Handler Contract`
- [x] Add conversation bridge commands `conversation status`, `conversation smoke-test <on|off>`, `conversation attach [call]`, `conversation detach [call]`, `conversation disapprove [call]`, `conversation approve [call]`, `conversation say [call]`, and `conversation mode <manual|auto>`. (@codex-366-impl, 2026-06-06 13:40 PDT: default attach/answer/dial uses auto-approved mode for the M3 acceptance path, but the built-in echo reply handler is disabled unless `--conversation-smoke-test` or `conversation smoke-test on` explicitly marks the session as a test; `conversation disapprove` cancels active TTS and detaches the handler so the call remains transcription-only, while manual proposal approval remains available after `conversation mode manual`.)
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Conversation Handler Contract`
- [x] Add selected-call chat interface state showing inbound transcript, assistant response lifecycle, outbound `speak` activity, playback status, and call lifecycle. (@codex-366-impl, 2026-06-05 23:34 PDT: added bounded conversation lines/status to call state, call list/show status, and selected-call TUI chat rendering.)
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Conversation Handler Contract`
- [x] Route `ConversationCommand::Say { text }` to `OutboundSpeechController::speak()`. (@codex-366-impl, 2026-06-06 08:48 PDT: extracted M2 speech queue/cancel into `speech.rs`; auto mode queues Piper speech through the same media registry and single bidirectional WebSocket, while manual approve/say queues the pending proposal through the command source's selected TTS backend.)
  DESIGN reference: `Staged Build Strategy`, `Returning TTS Audio`
- [x] Route `ConversationCommand::Call(action)` through the Telnyx call-control mapping. (@codex-366-impl, 2026-06-05 23:34 PDT: mapped `CallAction::Hangup`; unsupported future call actions fail closed and record conversation failure state.)
  DESIGN reference: `motlie_voice::telephony Surface`, `v1.1: DTMF and Call Control`
- [x] Add barge-in policy only after milestone 1 and milestone 2 are independently stable. (@codex-366-impl, 2026-06-06 08:48 PDT: implemented the #366 drop-and-regenerate policy by canceling active M2 speech with Telnyx `clear` before handling the new final transcript; VAD/partial-triggered low-latency barge-in remains deferred.)
  DESIGN reference: `Returning TTS Audio`, `Real-Time Latency Requirements`

Verification (@codex-366-impl, 2026-06-05 23:34 PDT): `cargo build -p motlie-telnyx-gateway --features "sherpa piper"`, `cargo test -p motlie-telnyx-gateway --features "sherpa piper"`, and `cargo clippy -p motlie-telnyx-gateway --features "sherpa piper" -- -D warnings` all pass. Workspace-wide `cargo fmt` is still blocked by unrelated missing `examples/vector2/app/benchmark.rs`; package-scoped `cargo fmt -p motlie-telnyx-gateway` passes.

### 8.4 - Milestone 4 external integration harness (#367)

- [ ] Implement startup with `--socket <path>` for headless Unix-domain command control; when both `--tui` and `--socket` are present, route both through the command-source mux.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Implement the local-agent milestone 4 flow over the socket only: configure gateway state, enable inbound, poll `events poll` until `call.inbound.pending`, inspect `call get`, run `answer <call_id>`, poll transcript events, and run `hangup <call_id>` without using appserver webhooks or the HTTP Control API.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Inbound Call Handler Design`
- [ ] Implement the programmatic milestone 1 flow: application webhook subscription -> `call.inbound.pending` event -> optional `GET /api/v1/calls/{call_id}` state reconciliation -> `POST /api/v1/calls/{call_id}/attachments` to bind transcript/event delivery -> `POST /api/v1/calls/{call_id}/answer` -> media start -> `transcript.partial` / `transcript.final` webhooks.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Inbound Call Handler Design`
- [ ] Implement the programmatic milestone 2 flow: `POST /api/v1/calls` -> outbound Telnyx dial with media -> `POST /api/v1/calls/{call_id}/say` -> outbound TTS media -> TTS playback webhooks.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Outbound Call Handler Design`
- [ ] Add a harness appserver that can register a webhook subscription, verify HMAC signatures, reconcile call state, attach to calls, answer inbound calls, drive outbound dial/say, and record observed events.
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Verify socket/TUI selection isolation and command-source mux ordering under concurrent command submissions.
  DESIGN reference: `Operator REPL and TUI Control Surface`

## Phase 9: Verification, Examples, and Operational Docs

Make each milestone reviewable and runnable independently before combining them.

### 9.1 - Local and simulated verification

- [ ] Add JSON fixtures for Telnyx webhook and WebSocket messages.
  DESIGN reference: `Testing Scope for PLAN`
- [ ] Add milestone 1 loopback harnesses that replay captured inbound media frames through the typed provider-neutral pipeline and verify:
  transcript updates,
  session teardown.
  DESIGN reference: `Testing Scope for PLAN`, `Media Adaptation Pipeline`
- [ ] Add replay/WER checks over finalized `asr-input-16khz.wav` capture artifacts using `replay-capture`, including transcript output, WER percentage, substitution/deletion/insertion counts, and token-level error rows.
  DESIGN reference: `Milestone 1.5: Sherpa ASR Quality Tuning`, `Testing Scope for PLAN`
- [ ] Add a provider-free adaptation test that feeds synthetic Telnyx-like `PCMU 8 kHz mono` payloads through G.711 decode, typed PCM, anti-aliased `8 kHz -> 16 kHz` resample, and a fake `StreamingTranscriber<Input = AudioBuf<i16, 16_000, Mono>>`.
  DESIGN reference: `Testing Scope for PLAN`, `Media Adaptation Pipeline`
- [ ] Add milestone 2 outbound loopback tests from Piper-shaped `AudioBuf<i16, 22_050, Mono>` through anti-aliased resample, packetization, and `L16` or `PCMU` transport encoding, verifying TTS packet emission through the outbound media channel and session teardown.
  DESIGN reference: `Testing Scope for PLAN`, `Recommended Telnyx v1 Pipelines`
- [ ] Add outbound TTS quality regression checks for fixed `20 ms` packet sizing, continuous utterance-level resampling, no keepalive-silence interleaving during active speech, and frame interval/underrun structured logging.
  DESIGN reference: `Testing Scope for PLAN`, `M2-Safe Bidirectional Media Contract`
- [ ] Add media-task tests proving inbound ASR frames are still consumed while outbound TTS frames are queued/sent over the same bidirectional WebSocket.
  DESIGN reference: `Testing Scope for PLAN`, `M2-Safe Bidirectional Media Contract`
- [ ] Add cancellation tests proving `speak cancel` stops synthesis/draining, drops local queued frames, sends Telnyx `clear`, and preserves the call/ASR session.
  DESIGN reference: `Testing Scope for PLAN`, `Returning TTS Audio`
- [ ] Add `mark` correlation tests proving speech jobs move through queued/sent/completed/canceled states and that TUI/socket `status` exposes those states.
  DESIGN reference: `Testing Scope for PLAN`, `Returning TTS Audio`
- [ ] Add command parity tests proving TUI, socket, and replayed `load` commands dispatch `dial`, `speak`, `speak cancel`, `hangup`, and `status` through the same source-local command engine without leaking selected calls between sources.
  DESIGN reference: `Testing Scope for PLAN`, `Driver REPL Dialer Surface`
- [ ] Add regression tests for out-of-order chunks, duplicate chunks, empty media, unsupported codecs, and invalid pipeline assemblies.
  DESIGN reference: `Testing Scope for PLAN`, `Recommended Safety Properties`
- [ ] Add regression tests for ASR low-energy gating before speech, sustained low-energy tail suppression after speech, ASR-session reopen on speech resume after sustained silence, repeated-token transcript suppression/reset signaling, and shell cursor alignment when command output wraps across multiple rendered rows.
  DESIGN reference: `Testing Scope for PLAN`, `Inbound Call Handler Design`, `Operator REPL and TUI Control Surface`
- [ ] Add operator-state tests for disabled inbound mode, manual inbound pending-call behavior, highlighted roster rows, unread event counts, auto-selection only when no active call is selected, source-local selected-call default command targets, TUI/socket selection isolation, command-source mux ordering, `answer` command transition, and selected-call detail event emission.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Inbound Call Handler Design`
- [ ] Add a milestone 1 structured-log check that verifies gateway call id, Telnyx diagnostic ids, stream id, observed codec, observed sample rate, transcript partial/final events, and Telnyx termination details are logged during an inbound transcription session.
  DESIGN reference: `Inbound Call Handler Design`, `Open Concerns`
- [ ] Add Control API tests for authenticated call list/detail reads, call attachment create/list/delete, attach-before-answer and attach-after-answer behavior, answer/reject/hangup, outbound dial/say, idempotency handling, and refusal of unauthenticated mutating requests.
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Add application webhook delivery tests for event envelope shape, HMAC signature verification, retry behavior, event filtering, and duplicate-event deduplication by event ID.
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Add persistence tests that dump state, restart with `--load <dump_path>`, and assert the replayed gateway state matches durable Telnyx/app-server configuration without resurrecting active calls or media sessions.
  DESIGN reference: `Replayable State Dumps`, `Gateway Configuration Requirement`
- [ ] Add secret-safety tests that generated dumps contain secret references or hashes, not raw Telnyx API keys, application webhook HMAC secrets, or bearer tokens.
  DESIGN reference: `Replayable State Dumps`
- [ ] Add compile-fail or equivalent type-level tests for wrong rate, wrong channel layout, wrong sample type, and wrong stage order.
  DESIGN reference: `Testing Scope for PLAN`, `Compile-Time Pipeline Assembly`

### 9.2 - Example and operator docs

- [ ] Add an operator-focused doc or example showing:
  required Telnyx app settings,
  Tailscale Funnel webhook URL,
  Tailscale Funnel WSS endpoint,
  environment variables,
  `--tui` and `--socket <path>` startup modes,
  TUI and headless command sequence for app creation/selection,
  phone-number binding,
  inbound enablement,
  `state dump`, `shutdown [dump_path]`, and `--load <dump_path>`,
  chosen ASR/TTS bundles.
  DESIGN reference: `Overview`, `Operator REPL and TUI Control Surface`, `Replayable State Dumps`, `Open Concerns`
- [ ] Add one milestone 1 inbound transcription example configuration for Sherpa ASR to `StdoutTranscriptSink`.
  DESIGN reference: `Staged Build Strategy`, `Concrete Combination Requirements`
- [ ] Add one milestone 1 TUI walkthrough: `telnyx app create/use`, `telnyx number bind`, `inbound enable --manual`, incoming call highlighted in the top roster, `call use <call>` or keyboard selection, `answer [call]`, and transcript in the selected-call detail pane.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Inbound Call Handler Design`
- [ ] Add one milestone 4 inbound service walkthrough: register a webhook subscription, receive `call.inbound.pending`, reconcile with `GET /api/v1/calls/{call_id}`, attach the subscription to the call with `POST /api/v1/calls/{call_id}/attachments`, call `POST /api/v1/calls/{call_id}/answer`, and receive transcript webhooks.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Inbound Call Handler Design`
- [ ] Add one milestone 2 outbound dial/speak example configuration for Piper TTS through `OutboundSpeechController::speak()`.
  DESIGN reference: `Staged Build Strategy`, `Driver REPL Dialer Surface`
- [ ] Add one milestone 4 outbound service walkthrough: call `POST /api/v1/calls`, then `POST /api/v1/calls/{call_id}/say`, and observe TTS playback webhooks.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Outbound Call Handler Design`
- [ ] Add one milestone 3 end-to-end example configuration for Sherpa + Piper only after the inbound ASR and outbound TTS milestones are independently stable.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Document the first live-call checklist, including observed codec logging, Telnyx termination reason logging, and bidirectional PCMU silence keepalive validation.
  DESIGN reference: `Open Concerns`

### 9.3 - Live validation

- [ ] Place at least one inbound test call for milestone 1 and one outbound test call for milestone 2 against a real Telnyx application after local simulation passes.
  DESIGN reference: `Testing Scope for PLAN`
- [ ] Capture and review the exact observed `start.media_format` values from Telnyx.
  DESIGN reference: `Open Concerns`
- [ ] Use captured WAV/JSONL artifacts and `replay-capture` WER reports to compare `PCMU 8 kHz` against `L16 16 kHz`, verify the observed L16 byte order, and tune Sherpa decoding/endpointing without requiring a fresh phone call for every ASR experiment.
  DESIGN reference: `Inbound Call Handler Design`, `Recommended ASR/TTS Stack`
- [ ] Verify conversational latency for the Sherpa + Piper duplex path once milestone 3 exists, and document measured numbers nearby in this PLAN or a follow-up note.
  DESIGN reference: `Real-Time Latency Requirements`
- [ ] Document first-audio latency separately from outbound transport packet cadence, because Piper currently produces a full buffer before packetized streaming begins.
  DESIGN reference: `Transport Streaming vs Incremental TTS`, `Real-Time Latency Requirements`
- [ ] During M2 live validation, review `tts.outbound.frame.sent` intervals, queue depth, `tts.outbound.underrun`, Telnyx `mark` receipt, and human-reported smoothness before accepting outbound TTS quality.
  DESIGN reference: `M2-Safe Bidirectional Media Contract`, `Testing Scope for PLAN`

## Phase 10: Follow-On Backend Pairings

After the Sherpa + Piper duplex milestone is implemented and validated end to end, broaden the backend menu.

### 10.1 - Sherpa + Qwen3-TTS

- [ ] Implement and validate the Qwen3-TTS outbound path after the Piper outbound milestone is stable.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Measure the added cost of `f32` to `i16` conversion plus `24 kHz` resampling in live telephony conditions.
  DESIGN reference: `Concrete Combination Requirements`, `Real-Time Latency Requirements`
- [ ] Preserve the buffered-vs-incremental TTS distinction for Qwen3-TTS.cpp; current chunked output drains a completed buffer rather than streaming model generation.
  DESIGN reference: `Transport Streaming vs Incremental TTS`

### 10.2 - Moonshine + Qwen3-TTS

- [ ] Implement the Moonshine inbound path:
  Telnyx sequence-map -> provider-neutral reorder -> decode -> mono normalize -> `16 kHz` resample -> fixed `1280`-sample rechunking.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Validate that Moonshine's fixed `1280`-sample cadence remains acceptable under Telnyx media pacing and jitter conditions.
  DESIGN reference: `Concrete Combination Requirements`

### 10.3 - Additional pairings and provider expansion

- [ ] Add any additional ASR/TTS pairings only after documenting their exact adaptation requirements alongside the existing matrix.
  DESIGN reference: `Concrete Backend Requirements Matrix`
- [ ] Reuse `libs/voice` for future provider adapters without backfilling provider-specific assumptions into the generic pipeline.
  DESIGN reference: `Provider-Neutral API Rule`
- [ ] Treat all non-`Sherpa + Piper` combinations as explicit follow-on slices with their own validation notes rather than opportunistic add-ons to the first duplex milestone.
  DESIGN reference: `Concrete Combination Requirements`

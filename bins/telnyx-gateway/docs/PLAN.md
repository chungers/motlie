# Telnyx Real-Time Voice Integration - Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
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

- the gateway starts as an idle TUI/REPL application with an HTTP/WebSocket listener; it does not answer inbound calls until the operator enables inbound mode or runs `answer`
- the same call-control actions must also be available through an authenticated Gateway Control API, and gateway application webhooks must allow external servers to receive call/transcript/TTS events
- milestone 1 is inbound calls with ASR-only transcription to a `TranscriptSink`
- milestone 2 is outbound dialing plus TTS driven by `motlie-driver` commands such as `dial <number>` and `say <text>`
- milestone 3 connects inbound transcript events to outbound speech through a `ConversationHandler`
- `Sherpa + Piper` remains the first complete duplex backend pairing once milestone 3 exists
- `Moonshine` and `Qwen3-TTS` may influence generic pipeline design, but they must not add blocking requirements to phases 1 through 9
- all work needed only for those follow-on pairings belongs in phase 10 or later

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
- [ ] Add `operator/{commands.rs,state.rs,persistence.rs,tui.rs}` for the `motlie-driver` command family, left REPL pane, right split call roster/detail TUI, shared gateway REPL state, replayable state dumps, and command/status routing.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Crate Hierarchy and API Surfaces`
- [ ] Add TUI view models for `CallRosterItem`, selected-call detail, unread event counts, and selection state derived from `SessionRegistry`.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Crate Hierarchy and API Surfaces`
- [ ] Add `api/{control.rs,auth.rs,subscriptions.rs}` and `events/{envelope.rs,dispatcher.rs,delivery.rs}` for the external Control API and gateway application webhook delivery.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Crate Hierarchy and API Surfaces`
- [ ] Start the gateway as an idle listener by default; do not register Telnyx apps, bind numbers, answer inbound calls, or dial outbound calls until a REPL command does so.
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
- [ ] Add `OutboundTtsPipeline` and `OutboundSpeechController` as the high-level controller consumed by REPL commands, mstream/broadcast bridges, tests, and conversation handlers.
  DESIGN reference: `Pipeline Controller Contracts`, `Driver REPL Dialer Surface`
- [ ] Keep these controllers source-agnostic: REPL commands, mstream broadcast, fixture replay, and conversation handlers should all call the same outbound `say()` path.
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
- [ ] Implement provider-neutral G.711 and `L16` codecs in `libs/voice/src/codec/`.
  DESIGN reference: `Codec and Container Gaps`, `Crate Hierarchy and API Surfaces`
- [ ] Keep stage responsibilities split; do not collapse decode, resample, and packetization into one opaque adapter.
  DESIGN reference: `Required Concrete Stage Inventory`

## Phase 4: Milestone 1 - Inbound Transcription

Assemble and validate the first useful Telnyx slice: inbound call handling plus Sherpa ONNX streaming ASR into a transcript sink. This phase deliberately does not require outbound TTS or a conversation handler.

### 4.1 - Sherpa inbound assembly

- [ ] Implement the Sherpa inbound path:
  Telnyx sequence-map -> provider-neutral reorder -> decode -> mono normalize -> `16 kHz` resample -> Sherpa ingest flow.
  DESIGN reference: `Concrete Combination Requirements`

### 4.2 - Transcript sinks

- [ ] Implement `StdoutTranscriptSink` for the first live inbound validation.
  DESIGN reference: `Staged Build Strategy`, `Inbound Handler Surface`
- [ ] Implement gateway-local `TuiTranscriptSink` and `WebhookTranscriptSink` adapters so milestone 1 can render transcripts to the selected-call detail pane or emit `transcript.partial` / `transcript.final` application webhooks.
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
- [ ] Extend the Telnyx REST client with the provisioning calls needed by the REPL: list/create/select Call Control Applications, update the application webhook URL, list phone numbers, and bind a phone number to a connection ID.
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
- [ ] Add `motlie-driver` commands for external automation: `api token create`, `api token revoke`, `webhook subscription list`, `webhook subscription add <url> --events <event,...>`, `webhook subscription upsert <subscription-id> <url> --events <event,...> --secret-ref <secret-ref>`, `webhook subscription remove <subscription-id>`, and `webhook subscription test <subscription-id>`.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Application Webhooks and Gateway Control API`
- [ ] Add Telnyx provisioning commands: `telnyx app list`, `telnyx app create <name>`, `telnyx app use <connection-id>`, `telnyx app webhook set <https-url>`, `telnyx number list`, `telnyx number use <e164>`, and `telnyx number bind <e164> <connection-id>`.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Add inbound call commands: `inbound status`, `inbound enable --manual`, `inbound enable --auto-transcribe`, `inbound disable`, `calls`, `call use <call>`, `answer [call] [--sink tui|stdout|webhook:<subscription-id>]`, `reject [call]`, `hangup [call]`, and transcript follow/clear commands.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Inbound Call Handler Design`
- [ ] Route command results, Telnyx API responses, pending/active call state, media metadata, and transcript events to the global status stream, top call roster, or selected-call detail pane as appropriate.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Make `call use <call>` update `selected_call`; allow `answer`, `reject`, `hangup`, and `say <text>` to target `selected_call` when no call id is supplied.
  DESIGN reference: `Operator REPL and TUI Control Surface`

### 6.4 - Replayable gateway state

- [ ] Add startup `--load <dump_path>` support that replays a gateway command dump before enabling inbound handling or accepting Control API mutations.
  DESIGN reference: `Replayable State Dumps`, `Gateway Configuration Requirement`
- [ ] Implement replayable dump generation as command text, not an opaque binary snapshot.
  DESIGN reference: `Replayable State Dumps`
- [ ] Include durable state in dumps: public webhook/media URLs, selected Telnyx app/connection, selected/bound phone number, default from number, inbound mode, webhook subscriptions/event filters, secret references, token metadata or token hash references, and selected ASR/TTS model names when configured through the REPL.
  DESIGN reference: `Replayable State Dumps`
- [ ] Exclude transient state from dumps: active calls, pending calls, media sessions, transcript history, in-flight TTS playback, and retry queues unless a later event journal explicitly persists them.
  DESIGN reference: `Replayable State Dumps`
- [ ] Ensure replay commands are idempotent and prefer `use`, `bind`, and `upsert` forms over create-only commands that would duplicate remote or local objects.
  DESIGN reference: `Replayable State Dumps`
- [ ] Never dump raw Telnyx API keys, application webhook HMAC secrets, or bearer tokens; persist only secret references or hashed token material.
  DESIGN reference: `Replayable State Dumps`

### 6.5 - Gateway Control API

- [ ] Add authenticated call-control endpoints: `GET /api/v1/calls`, `GET /api/v1/calls/{call_id}`, `POST /api/v1/calls/{call_id}/answer`, `POST /api/v1/calls/{call_id}/reject`, and `POST /api/v1/calls/{call_id}/hangup`.
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Add outbound dialer/TTS endpoints: `POST /api/v1/calls`, `POST /api/v1/calls/{call_id}/say`, and `POST /api/v1/calls/{call_id}/interrupt`.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Outbound Call Handler Design`
- [ ] Route Control API requests through the same internal command/controller layer as REPL commands so answer, dial, hangup, and say behavior cannot drift.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Operator REPL and TUI Control Surface`
- [ ] Require authentication for the Control API and support `Idempotency-Key` on mutating requests.
  DESIGN reference: `Application Webhooks and Gateway Control API`

### 6.6 - Gateway application webhooks

- [ ] Add subscription CRUD endpoints: `POST /api/v1/webhook-subscriptions`, `GET /api/v1/webhook-subscriptions`, `GET /api/v1/webhook-subscriptions/{subscription_id}`, `DELETE /api/v1/webhook-subscriptions/{subscription_id}`, and `POST /api/v1/webhook-subscriptions/{subscription_id}/test`.
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Implement outbound application webhook delivery as HTTP `POST <subscription.url>` with `Content-Type: application/json`, the gateway event envelope as the body, and HMAC delivery headers.
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Treat any `2xx` webhook response as delivery acknowledgement; treat non-`2xx`, timeout, and connection failure as retryable delivery failure.
  DESIGN reference: `Application Webhooks and Gateway Control API`
- [ ] Emit milestone 1 events: `call.inbound.pending`, `call.answering`, `call.answered`, `media.started`, `transcript.partial`, `transcript.final`, `call.ended`, and `call.failed`.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Inbound Call Handler Design`
- [ ] Emit milestone 2 events: `call.outbound.created`, `call.dialing`, `call.ringing`, `call.answered`, `media.started`, `tts.playback.started`, `tts.playback.finished`, `tts.playback.failed`, `call.ended`, and `call.failed`.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Outbound Call Handler Design`
- [ ] Deliver application webhooks asynchronously with bounded retries; never block Telnyx webhook handling on subscriber delivery.
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

- [ ] Implement the Piper outbound path:
  `AudioBuf<i16, 22_050, Mono>` -> target-rate resample -> telephony packetize -> Telnyx encode.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Treat Piper as full-buffer synthesis followed by transport packetization; do not claim incremental TTS audio until the backend actually emits audio incrementally.
  DESIGN reference: `Transport Streaming vs Incremental TTS`, `Latency Budget`
- [ ] Record the Piper runtime mode used for validation. By default Piper is CPU-only because of #230; `MOTLIE_PIPER_ALLOW_CUDA=1` must be explicitly documented if enabled.
  DESIGN reference: `Recommended TTS: Piper`, `Open Concerns`
- [ ] Call the selected `SpeechSynthesizer` and drain its returned `SpeechStream` when `OutboundSpeechController::say()` receives text.
  DESIGN reference: `Returning TTS Audio`, `Driver REPL Dialer Surface`
- [ ] Instantiate the correct typed outbound pipeline from the selected TTS runtime and Telnyx outbound codec.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Add interruption handling:
  stop active TTS generation,
  stop outbound packet send,
  send `clear` only for MP3 mode.
  DESIGN reference: `Returning TTS Audio`

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

### 8.1 - Milestone 1 inbound transcription flow

- [ ] Implement startup into an idle listener plus a TUI where the left pane is the REPL and the right side is split into a top call roster plus bottom selected-call detail; the roster shows pending, active, dialing, held, failed, and recently ended calls, while the detail pane shows status, timeline, media, transcript, TTS state, and action hints for the selected call.
  DESIGN reference: `Operator REPL and TUI Control Surface`
- [ ] Implement `inbound enable --manual` -> `call.initiated` -> highlighted pending/waiting call roster row -> optional `call use <call>` -> `answer [call]` -> `answer + streaming` -> WebSocket media -> ASR -> `TuiTranscriptSink` selected-call detail or `StdoutTranscriptSink` -> hangup flow.
  DESIGN reference: `Inbound Call Handler Design`, `Operator REPL and TUI Control Surface`
- [ ] Implement the programmatic milestone 1 flow: application webhook subscription -> `call.inbound.pending` event -> `POST /api/v1/calls/{call_id}/answer` -> media start -> `transcript.partial` / `transcript.final` webhooks.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Inbound Call Handler Design`
- [ ] Keep inbound disabled by default at process startup; incoming webhooks must not be answered until the operator enables inbound handling.
  DESIGN reference: `Staged Build Strategy`, `Operator REPL and TUI Control Surface`
- [ ] Validate that `stream_bidirectional_target_legs=self` is correct for the initial single-leg AI call pattern.
  DESIGN reference: `Open Concerns`
- [ ] Add structured logs for `call_control_id`, `stream_id`, codec, and observed sample rate.
  DESIGN reference: `Open Concerns`

### 8.2 - Milestone 2 outbound dialer/TTS flow

- [ ] Implement application-triggered `POST /v2/calls` with inline streaming parameters.
  DESIGN reference: `Outbound Call Handler Design`
- [ ] Use `stream_establish_before_call_originate=true` in the first outbound implementation.
  DESIGN reference: `Outbound Call Handler Design`
- [ ] Add outbound `motlie-driver` commands for `dial <phone-or-sip-uri>`, `say [call] <text...>`, `hangup [call]`, `tts status`, and `tts model use <model>`.
  DESIGN reference: `Driver REPL Dialer Surface`
- [ ] Implement the programmatic milestone 2 flow: `POST /api/v1/calls` -> outbound Telnyx dial with media -> `POST /api/v1/calls/{call_id}/say` -> outbound TTS media -> TTS playback webhooks.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Outbound Call Handler Design`
- [ ] Render outbound call state in the top roster and media session state, TTS synthesis status, and packet-send status in the selected-call detail pane.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Driver REPL Dialer Surface`
- [ ] Keep Telnyx `connection_id`, stream URL, credentials, and bidirectional media flags in gateway configuration rather than adding them to provider-neutral outbound call requests.
  DESIGN reference: `Staged Build Strategy`, `Provider Adapter Boundary`
- [ ] Keep the REPL as one utterance source only; mstream send-keys/broadcast bridges and fixture replay should call the same `OutboundSpeechController::say()` path.
  DESIGN reference: `Staged Build Strategy`, `Driver REPL Dialer Surface`
- [ ] Support optional initial greeting once the media session is ready, implemented as a normal `say()` call.
  DESIGN reference: `Outbound Call Handler Design`, `Driver REPL Dialer Surface`

### 8.3 - Milestone 3 connected conversation flow

- [ ] Implement `TranscriptSink` adapter that forwards selected final transcript events to `ConversationHandler`.
  DESIGN reference: `Staged Build Strategy`, `Conversation Handler Contract`
- [ ] Add conversation bridge commands `conversation status`, `conversation attach [call]`, `conversation detach [call]`, and `conversation mode <manual|auto>`.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Conversation Handler Contract`
- [ ] Route `ConversationCommand::Say { text }` to `OutboundSpeechController::say()`.
  DESIGN reference: `Staged Build Strategy`, `Returning TTS Audio`
- [ ] Route `ConversationCommand::Call(action)` through the Telnyx call-control mapping.
  DESIGN reference: `motlie_voice::telephony Surface`, `v1.1: DTMF and Call Control`
- [ ] Add barge-in policy only after milestone 1 and milestone 2 are independently stable.
  DESIGN reference: `Returning TTS Audio`, `Real-Time Latency Requirements`

## Phase 9: Verification, Examples, and Operational Docs

Make each milestone reviewable and runnable independently before combining them.

### 9.1 - Local and simulated verification

- [ ] Add JSON fixtures for Telnyx webhook and WebSocket messages.
  DESIGN reference: `Testing Scope for PLAN`
- [ ] Add milestone 1 loopback harnesses that replay captured inbound media frames through the typed provider-neutral pipeline and verify:
  transcript updates,
  session teardown.
  DESIGN reference: `Testing Scope for PLAN`, `Media Adaptation Pipeline`
- [ ] Add a provider-free adaptation test that feeds synthetic Telnyx-like `PCMU 8 kHz mono` payloads through G.711 decode, typed PCM, anti-aliased `8 kHz -> 16 kHz` resample, and a fake `StreamingTranscriber<Input = AudioBuf<i16, 16_000, Mono>>`.
  DESIGN reference: `Testing Scope for PLAN`, `Media Adaptation Pipeline`
- [ ] Add milestone 2 outbound loopback tests from Piper-shaped `AudioBuf<i16, 22_050, Mono>` through anti-aliased resample, packetization, and `L16` or `PCMU` transport encoding, verifying TTS packet emission and session teardown.
  DESIGN reference: `Testing Scope for PLAN`, `Recommended Telnyx v1 Pipelines`
- [ ] Add regression tests for out-of-order chunks, duplicate chunks, empty media, unsupported codecs, and invalid pipeline assemblies.
  DESIGN reference: `Testing Scope for PLAN`, `Recommended Safety Properties`
- [ ] Add operator-state tests for disabled inbound mode, manual inbound pending-call behavior, highlighted roster rows, unread event counts, auto-selection only when no active call is selected, selected-call default command targets, `answer` command transition, and selected-call detail event emission.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Inbound Call Handler Design`
- [ ] Add Control API tests for authenticated answer/reject/hangup, outbound dial/say, idempotency handling, and refusal of unauthenticated mutating requests.
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
  REPL command sequence for app creation/selection,
  phone-number binding,
  inbound enablement,
  `state dump`, `shutdown [dump_path]`, and `--load <dump_path>`,
  chosen ASR/TTS bundles.
  DESIGN reference: `Overview`, `Operator REPL and TUI Control Surface`, `Replayable State Dumps`, `Open Concerns`
- [ ] Add one milestone 1 inbound transcription example configuration for Sherpa ASR to `StdoutTranscriptSink`.
  DESIGN reference: `Staged Build Strategy`, `Concrete Combination Requirements`
- [ ] Add one milestone 1 TUI walkthrough: `telnyx app create/use`, `telnyx number bind`, `inbound enable --manual`, incoming call highlighted in the top roster, `call use <call>` or keyboard selection, `answer [call]`, and transcript in the selected-call detail pane.
  DESIGN reference: `Operator REPL and TUI Control Surface`, `Inbound Call Handler Design`
- [ ] Add one milestone 1 service walkthrough: register a webhook subscription, receive `call.inbound.pending`, call `POST /api/v1/calls/{call_id}/answer`, and receive transcript webhooks.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Inbound Call Handler Design`
- [ ] Add one milestone 2 outbound dial/say example configuration for Piper TTS through `OutboundSpeechController::say()`.
  DESIGN reference: `Staged Build Strategy`, `Driver REPL Dialer Surface`
- [ ] Add one milestone 2 service walkthrough: call `POST /api/v1/calls`, then `POST /api/v1/calls/{call_id}/say`, and observe TTS playback webhooks.
  DESIGN reference: `Application Webhooks and Gateway Control API`, `Outbound Call Handler Design`
- [ ] Add one milestone 3 end-to-end example configuration for Sherpa + Piper only after the inbound ASR and outbound TTS milestones are independently stable.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Document the first live-call checklist, including observed codec logging and `stream_bidirectional_target_legs` validation.
  DESIGN reference: `Open Concerns`

### 9.3 - Live validation

- [ ] Place at least one inbound test call for milestone 1 and one outbound test call for milestone 2 against a real Telnyx application after local simulation passes.
  DESIGN reference: `Testing Scope for PLAN`
- [ ] Capture and review the exact observed `start.media_format` values from Telnyx.
  DESIGN reference: `Open Concerns`
- [ ] Verify conversational latency for the Sherpa + Piper duplex path once milestone 3 exists, and document measured numbers nearby in this PLAN or a follow-up note.
  DESIGN reference: `Real-Time Latency Requirements`
- [ ] Document first-audio latency separately from outbound transport packet cadence, because Piper currently produces a full buffer before packetized streaming begins.
  DESIGN reference: `Transport Streaming vs Incremental TTS`, `Real-Time Latency Requirements`

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

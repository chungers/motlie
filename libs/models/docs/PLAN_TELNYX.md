# Telnyx Real-Time Voice Integration - Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-17 | @codex-macmini-telnyx | Tightened the execution policy further so no initial acceptance criteria, examples, or live validation steps can rely on Moonshine or Qwen3-TTS. The first complete Telnyx slice is Sherpa + Piper only. |
| 2026-04-17 | @codex-macmini-telnyx | Tightened the PLAN so Sherpa + Piper is the explicit first vertical slice. Follow-on pairings such as Sherpa + Qwen3-TTS and Moonshine + Qwen3-TTS are now deferred to a later phase instead of treated as peer initial targets. |
| 2026-04-16 | @codex-macmini-telnyx | Reworked the PLAN to match the new hierarchy: provider-agnostic `libs/voice`, Telnyx-specific `libs/voice_telnyx`, and a thin `bins/motlie-telnyx-gateway` binary. |
| 2026-04-15 | @codex-macmini-telnyx | Initial Telnyx integration PLAN derived from `DESIGN_TELNYX.md`. Covers the transport gateway, codec normalization, inbound and outbound call flows, application orchestration, and validation. |

Derived from [DESIGN_TELNYX.md](./DESIGN_TELNYX.md). This PLAN assumes brownfield work against the existing `libs/model` and `libs/models` speech stack and now explicitly separates provider-neutral voice infrastructure from the Telnyx-specific transport adapter.

Execution policy for the initial implementation:

- the first complete implementation, operator example, and live validation path is `Sherpa + Piper`
- `Moonshine` and `Qwen3-TTS` may influence generic pipeline design, but they must not add blocking requirements to phases 1 through 9
- all work needed only for those follow-on pairings belongs in phase 10 or later

---

## Phase 1: Workspace and Crate Skeleton

Create the provider-neutral voice layer, the Telnyx adapter layer, and the thin deployable binary.

### 1.1 - Workspace shape

- [ ] Add a new provider-agnostic voice crate at `libs/voice`.
  DESIGN reference: `Overview`, `Crate Hierarchy and API Surfaces`
- [ ] Add a new Telnyx adapter crate at `libs/voice_telnyx`.
  DESIGN reference: `Overview`, `Crate Hierarchy and API Surfaces`
- [ ] Add the deployable binary at `bins/motlie-telnyx-gateway`.
  DESIGN reference: `Overview`, `Crate Hierarchy and API Surfaces`
- [ ] Keep provider-neutral media, runtime, and application traits out of `libs/voice_telnyx`.
  DESIGN reference: `Provider-Neutral API Rule`
- [ ] Keep Telnyx webhook, WebSocket, and REST schema types out of `libs/voice`.
  DESIGN reference: `Provider-Neutral API Rule`

### 1.2 - `libs/voice` module skeleton

- [ ] Add `app/`, `runtime/`, `pipeline/`, `codec/`, and `telephony/` under `libs/voice/src/`.
  DESIGN reference: `Crate Hierarchy and API Surfaces`
- [ ] Add crate-level `error.rs` and `config.rs`.
  DESIGN reference: `Crate Hierarchy and API Surfaces`

### 1.3 - `libs/voice_telnyx` module skeleton

- [ ] Add `webhook/`, `call_control/`, `media/`, and `adapter.rs` under `libs/voice_telnyx/src/`.
  DESIGN reference: `Crate Hierarchy and API Surfaces`
- [ ] Keep `libs/voice_telnyx` focused on protocol mapping and Telnyx call control.
  DESIGN reference: `Crate Hierarchy and API Surfaces`

## Phase 2: Provider-Neutral Application and Runtime Contracts

Define the reusable voice application surface in `libs/voice`.

### 2.1 - Application contracts

- [ ] Add `ConversationContext`, `ConversationHandler`, `DtmfHandler`, `IvrNavigator`, `DtmfDigit`, and provider-neutral `CallAction` in `libs/voice/src/app/`.
  DESIGN reference: `Conversation Handler Contract`, `v1.1: DTMF and Call Control`, `Crate Hierarchy and API Surfaces`
- [ ] Keep these contracts provider-neutral so they can be reused by future Twilio or SIP adapters.
  DESIGN reference: `Provider-Neutral API Rule`

### 2.2 - Static runtime selection

- [ ] Add provider-neutral static-dispatch ASR/TTS runtime wrappers in `libs/voice/src/runtime/`.
  DESIGN reference: `Static-Dispatch Model Injection`, `Crate Hierarchy and API Surfaces`
- [ ] Map compiled `motlie_models::AsrModels` and `motlie_models::TtsModels` selections into closed runtime enums or generic wrappers.
  DESIGN reference: `Closed-Enum Selection`, `Build-Time Model Surface`
- [ ] Document the required `motlie-models` feature flags in code comments or crate docs near the runtime selection layer.
  DESIGN reference: `Build-Time Model Surface`

## Phase 3: Typed Media Pipeline in `libs/voice`

Build the provider-neutral media adaptation pipeline with explicit stage contracts.

### 3.1 - Media values and markers

- [ ] Add `EncodedFrame<C>` and `PcmFrame<R, Ch, E>` in `libs/voice/src/pipeline/types.rs`.
  DESIGN reference: `Typed Media Values`
- [ ] Add marker types and marker traits in `libs/voice/src/pipeline/markers.rs`.
  DESIGN reference: `Marker Traits`
- [ ] Add stage traits in `libs/voice/src/pipeline/stage.rs`.
  DESIGN reference: `Stage API Surface`

### 3.2 - Stage builder and safety checks

- [ ] Add typed stage-composition helpers in `libs/voice/src/pipeline/builder.rs`.
  DESIGN reference: `Compile-Time Pipeline Assembly`
- [ ] Add compile-time or unit-level proofs that adjacent stages must agree on `In` and `Out`.
  DESIGN reference: `Compile-Time Pipeline Assembly`, `Recommended Safety Properties`

### 3.3 - Core pipeline stages

- [ ] Implement reorder, convert, resample, chunk, and packetize stages in `libs/voice/src/pipeline/`.
  DESIGN reference: `Media Adaptation Pipeline`, `Required Concrete Stage Inventory`
- [ ] Implement provider-neutral G.711 and `L16` codecs in `libs/voice/src/codec/`.
  DESIGN reference: `Codec and Container Gaps`, `Crate Hierarchy and API Surfaces`
- [ ] Keep stage responsibilities split; do not collapse decode, resample, and packetization into one opaque adapter.
  DESIGN reference: `Required Concrete Stage Inventory`

## Phase 4: First Vertical Slice - Sherpa + Piper

Assemble and validate the first end-to-end Telnyx slice with the most proven real-time stack: Sherpa ONNX streaming ASR plus Piper TTS.

### 4.1 - Sherpa inbound assembly

- [ ] Implement the Sherpa inbound path:
  Telnyx decode -> mono normalize -> `16 kHz` resample -> Sherpa chunk flow.
  DESIGN reference: `Concrete Combination Requirements`

### 4.2 - Piper outbound assembly

- [ ] Implement the Piper outbound path:
  `22.05 kHz` `S16Le` -> target-rate resample -> telephony packetize -> Telnyx encode.
  DESIGN reference: `Concrete Combination Requirements`

### 4.3 - First-slice pairing constraints

- [ ] Add explicit assembly coverage for Sherpa + Piper only in the first implementation slice.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Reject undeclared pairings cleanly instead of falling back to guessed pipeline behavior.
  DESIGN reference: `Concrete Backend Requirements Matrix`
- [ ] Keep Moonshine and Qwen3-TTS support out of the first live Telnyx validation path even if the provider-neutral abstractions are designed broadly enough to support them later.
  DESIGN reference: `Recommended ASR/TTS Stack`, `Concrete Combination Requirements`
- [ ] Do not add first-slice code paths, examples, or acceptance checks that require Moonshine-specific rechunking or Qwen3-TTS-specific `F32Le 24 kHz` conversion.
  DESIGN reference: `Concrete Combination Requirements`

## Phase 5: Telnyx Protocol and Adapter Layer

Wire the Telnyx-specific schema and adapter crate on top of `libs/voice`.

### 5.1 - Telnyx protocol types

- [ ] Add Rust types for Telnyx webhook payloads needed by the first slice:
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
- [ ] Implement per-track reordering using Telnyx `media.chunk` before handing frames to `libs/voice` pipeline stages.
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
- [ ] On `call.initiated`, create a pending inbound session and issue `answer` with the recommended stream parameters.
  DESIGN reference: `Inbound Call Handler Design`
- [ ] On streaming failure or terminal call events, tear down session state and release model resources.
  DESIGN reference: `Recommended Integration Shape`, `Open Concerns`

## Phase 7: Voice Orchestration

Connect the Telnyx adapter and the provider-neutral voice pipeline to the existing Motlie speech contracts.

### 7.1 - Inbound audio to ASR

- [ ] On Telnyx `start`, instantiate the correct typed inbound pipeline from `start.media_format` and the selected ASR runtime.
  DESIGN reference: `Telnyx Media Schema Mapping`, `Concrete Combination Requirements`
- [ ] Convert ordered inbound media frames into `PcmChunk` values and call `push_chunk()`.
  DESIGN reference: `Mapping Telnyx Frames to TranscriptionStream`
- [ ] Forward `TranscriptionUpdate` partials and finals to application logic without exposing Telnyx types.
  DESIGN reference: `Feeding the Conversation Handler`

### 7.2 - Conversation, DTMF, and IVR control

- [ ] Wire `ConversationHandler` into the main voice loop.
  DESIGN reference: `Conversation Handler Contract`
- [ ] Wire `DtmfHandler` and provider-neutral `CallAction` mapping through the Telnyx adapter.
  DESIGN reference: `v1.1: DTMF and Call Control`
- [ ] Keep dialog and keypad policy out of `libs/voice_telnyx`.
  DESIGN reference: `Provider-Neutral API Rule`

### 7.3 - TTS to outbound media

- [ ] Open `SpeechStream` from the selected `SpeechModel` when the application requests speech.
  DESIGN reference: `Returning TTS Audio`
- [ ] Instantiate the correct typed outbound pipeline from the selected TTS runtime and Telnyx outbound codec.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Add interruption handling:
  stop active TTS generation,
  stop outbound packet send,
  send `clear` only for MP3 mode.
  DESIGN reference: `Returning TTS Audio`

## Phase 8: Inbound and Outbound Product Flows

Close the loop on complete call lifecycle behavior.

### 8.1 - Inbound voice agent flow

- [ ] Implement `call.initiated` -> `answer + streaming` -> WebSocket media -> ASR -> app -> TTS -> hangup flow.
  DESIGN reference: `Inbound Call Handler Design`
- [ ] Validate that `stream_bidirectional_target_legs=self` is correct for the initial single-leg AI call pattern.
  DESIGN reference: `Open Concerns`
- [ ] Add structured logs for `call_control_id`, `stream_id`, codec, and observed sample rate.
  DESIGN reference: `Open Concerns`

### 8.2 - Outbound voice agent flow

- [ ] Implement application-triggered `POST /v2/calls` with inline streaming parameters.
  DESIGN reference: `Outbound Call Handler Design`
- [ ] Use `stream_establish_before_call_originate=true` in the first outbound implementation.
  DESIGN reference: `Outbound Call Handler Design`
- [ ] Support optional initial greeting once the media session is ready.
  DESIGN reference: `Outbound Call Handler Design`

## Phase 9: Verification, Examples, and Operational Docs

Make the first slice reviewable and runnable.

### 9.1 - Local and simulated verification

- [ ] Add JSON fixtures for Telnyx webhook and WebSocket messages.
  DESIGN reference: `Testing Scope for PLAN`
- [ ] Add loopback harnesses that replay captured inbound media frames through the typed provider-neutral pipeline and verify:
  transcript updates,
  TTS packet emission,
  session teardown.
  DESIGN reference: `Testing Scope for PLAN`, `Media Adaptation Pipeline`
- [ ] Add regression tests for out-of-order chunks, duplicate chunks, empty media, unsupported codecs, and invalid pipeline assemblies.
  DESIGN reference: `Testing Scope for PLAN`, `Recommended Safety Properties`

### 9.2 - Example and operator docs

- [ ] Add an operator-focused doc or example showing:
  required Telnyx app settings,
  webhook URL,
  WSS endpoint,
  environment variables,
  chosen ASR/TTS bundles.
  DESIGN reference: `Overview`, `Open Concerns`
- [ ] Add one end-to-end example configuration for Sherpa + Piper as the first and only required example in the initial slice.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Document the first live-call checklist, including observed codec logging and `stream_bidirectional_target_legs` validation.
  DESIGN reference: `Open Concerns`

### 9.3 - Live validation

- [ ] Place at least one inbound test call and one outbound test call against a real Telnyx application after local simulation passes.
  DESIGN reference: `Testing Scope for PLAN`
- [ ] Capture and review the exact observed `start.media_format` values from Telnyx.
  DESIGN reference: `Open Concerns`
- [ ] Verify conversational latency is acceptable for the Sherpa + Piper first backend pair and document measured numbers nearby in this PLAN or a follow-up note.
  DESIGN reference: `Real-Time Latency Requirements`

## Phase 10: Follow-On Backend Pairings

After the Sherpa + Piper slice is implemented and validated end to end, broaden the backend menu.

### 10.1 - Sherpa + Qwen3-TTS

- [ ] Implement and validate the Qwen3-TTS outbound path after the first slice is stable.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Measure the added cost of `F32Le` conversion plus `24 kHz` resampling in live telephony conditions.
  DESIGN reference: `Concrete Combination Requirements`, `Real-Time Latency Requirements`

### 10.2 - Moonshine + Qwen3-TTS

- [ ] Implement the Moonshine inbound path:
  Telnyx decode -> mono normalize -> `16 kHz` resample -> fixed `1280`-sample rechunking.
  DESIGN reference: `Concrete Combination Requirements`
- [ ] Validate that Moonshine's fixed `1280`-sample cadence remains acceptable under Telnyx media pacing and jitter conditions.
  DESIGN reference: `Concrete Combination Requirements`

### 10.3 - Additional pairings and provider expansion

- [ ] Add any additional ASR/TTS pairings only after documenting their exact adaptation requirements alongside the existing matrix.
  DESIGN reference: `Concrete Backend Requirements Matrix`
- [ ] Reuse `libs/voice` for future provider adapters without backfilling provider-specific assumptions into the generic pipeline.
  DESIGN reference: `Provider-Neutral API Rule`
- [ ] Treat all non-`Sherpa + Piper` combinations as explicit follow-on slices with their own validation notes rather than opportunistic add-ons to the first slice.
  DESIGN reference: `Concrete Combination Requirements`

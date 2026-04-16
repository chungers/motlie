# Telnyx Real-Time Voice Integration - Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-15 | @codex-macmini-telnyx | Initial Telnyx integration PLAN derived from `DESIGN_TELNYX.md`. Covers the transport gateway, codec normalization, inbound and outbound call flows, application orchestration, and validation. |

Derived from [DESIGN_TELNYX.md](./DESIGN_TELNYX.md). This PLAN assumes brownfield work against the existing `libs/model` and `libs/models` speech stack and intentionally keeps the first slice focused on Telnyx WebSocket media streaming rather than generic SIP infrastructure.

---

## Phase 1: Gateway Skeleton and Transport Contracts

Create the deployable Telnyx integration layer without changing the core model contracts.

### 1.1 - Workspace shape

- [ ] Add a new reusable crate for Telnyx voice integration, recommended path `libs/voice_telnyx`.
  DESIGN reference: `Overview`, `Recommended Integration Shape`
- [ ] Add an optional service binary, recommended path `bins/motlie-telnyx-gateway`, if the deployable entrypoint does not belong in an existing binary.
  DESIGN reference: `Overview`, `Recommended Integration Shape`
- [ ] Keep Telnyx transport, codec, and webhook logic out of `libs/model` and `libs/models`.
  DESIGN reference: `Why This Belongs Above libs/model`

### 1.2 - Telnyx protocol types

- [ ] Add Rust types for Telnyx webhook payloads needed by the first slice:
  `call.initiated`,
  `call.answered`,
  `streaming.started`,
  `streaming.stopped`,
  `streaming.failed`.
  DESIGN reference: `Telnyx API Research`
- [ ] Add Rust types for WebSocket events:
  `connected`,
  `start`,
  `media`,
  `mark`,
  `clear`,
  `dtmf`,
  `error`,
  `stop`.
  DESIGN reference: `WebSocket Event and Chunk Format`
- [ ] Add parser tests against captured JSON fixtures.
  DESIGN reference: `Testing Scope for PLAN`

### 1.3 - Session registry and lifecycle

- [ ] Add a session registry keyed by `call_control_id` and `stream_id`.
  DESIGN reference: `Recommended Integration Shape`
- [ ] Model `PendingCallSession` versus fully started media sessions so webhook order and WebSocket order can arrive independently.
  DESIGN reference: `Inbound Call Handler Design`, `Outbound Call Handler Design`
- [ ] Add timeout and cleanup behavior for abandoned sessions.
  DESIGN reference: `Reordering and Jitter Handling`

## Phase 2: Telnyx REST and Webhook Handling

Wire call-control operations for inbound and outbound calls.

### 2.1 - REST client

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

### 2.2 - Webhook server

- [ ] Add an HTTP endpoint that accepts Telnyx voice webhooks and returns `200 OK` deterministically.
  DESIGN reference: `Inbound Call Handling`
- [ ] Add optional webhook signature verification if available in the chosen Telnyx app configuration.
  DESIGN reference: `Gap Analysis`
- [ ] On `call.initiated`, create a pending inbound session and issue `answer` with the recommended stream parameters.
  DESIGN reference: `Inbound Call Handler Design`
- [ ] On streaming failure or terminal call events, tear down session state and release model resources.
  DESIGN reference: `Recommended Integration Shape`, `Open Concerns`

## Phase 3: WebSocket Media Server

Accept real-time media from Telnyx and surface a stable gateway session API.

### 3.1 - WebSocket endpoint

- [ ] Add a `wss://.../telnyx/media` endpoint that upgrades incoming Telnyx connections.
  DESIGN reference: `Recommended Integration Shape`
- [ ] Parse the `connected` and `start` events before opening ASR resources.
  DESIGN reference: `WebSocket Event and Chunk Format`
- [ ] Persist `media_format`, `call_session_id`, and `stream_id` on the session.
  DESIGN reference: `WebSocket Event and Chunk Format`

### 3.2 - Reorder buffer and track handling

- [ ] Implement per-track reordering using Telnyx `media.chunk`.
  DESIGN reference: `Mapping Telnyx Frames to TranscriptionStream`
- [ ] Emit one gateway-local monotonic sequence for `PcmChunk`.
  DESIGN reference: `Mapping Telnyx Frames to TranscriptionStream`
- [ ] Add bounded jitter buffering and gap metrics.
  DESIGN reference: `Reordering and Jitter Handling`
- [ ] Default the conversational path to `inbound_track` only.
  DESIGN reference: `Why stream_track=inbound_track`

## Phase 4: Codec, Packetization, and Audio Normalization

Bridge Telnyx transport audio with Motlie PCM model contracts.

### 4.1 - Phase 1 codec support

- [ ] Implement `PCMU` decode/encode.
  DESIGN reference: `Codec and Container Gaps`
- [ ] Implement `PCMA` decode/encode.
  DESIGN reference: `Codec and Container Gaps`
- [ ] Implement `L16` payload encode/decode helpers for Telnyx RTP payloads.
  DESIGN reference: `Audio Codecs and Formats`
- [ ] Reject unsupported codecs with explicit errors and configurable call-failure policy.
  DESIGN reference: `Codec and Container Gaps`

### 4.2 - Resampling and sample conversion

- [ ] Add `8 kHz` mono to `16 kHz` mono normalization for inbound telephony audio.
  DESIGN reference: `Recommended Audio Normal Form`
- [ ] Add TTS-output normalization to the configured Telnyx outbound rate.
  DESIGN reference: `Recommended Audio Normal Form`
- [ ] Add `f32` <-> `s16le` conversion helpers where required by selected ASR/TTS backends.
  DESIGN reference: `Resampling and Format Normalization`
- [ ] Add unit tests that prove byte-stable `AudioSpec` to payload conversion for the supported codecs.
  DESIGN reference: `Testing Scope for PLAN`

## Phase 5: ASR and TTS Orchestration

Connect the gateway pipeline to the existing Motlie speech contracts.

### 5.1 - Inbound audio to ASR

- [ ] On Telnyx `start`, open a `TranscriptionStream` using normalized `AudioSpec { sample_rate_hz: 16_000, channels: 1, encoding: S16Le }`.
  DESIGN reference: `Recommended Audio Normal Form`, `Mapping Telnyx Frames to TranscriptionStream`
- [ ] Convert ordered inbound media frames into `PcmChunk` values and call `push_chunk()`.
  DESIGN reference: `Mapping Telnyx Frames to TranscriptionStream`
- [ ] Forward `TranscriptionUpdate` partials and finals to application logic without exposing Telnyx types.
  DESIGN reference: `Feeding Application Logic`

### 5.2 - Application turn handling

- [ ] Define a transport-agnostic application callback trait for transcript updates and speech actions.
  DESIGN reference: `Feeding Application Logic`
- [ ] Support at least:
  `SpeakText`,
  `ClearPlayback`,
  `Hangup`,
  `ContinueListening`.
  DESIGN reference: `Feeding Application Logic`
- [ ] Keep dialog policy outside the Telnyx transport crate.
  DESIGN reference: `Feeding Application Logic`

### 5.3 - TTS to outbound media

- [ ] Open `SpeechStream` from selected `SpeechModel` when the application requests speech.
  DESIGN reference: `Returning TTS Audio`
- [ ] Packetize outgoing audio into Telnyx `media` WebSocket frames using RTP mode.
  DESIGN reference: `MP3 Mode vs RTP Mode`, `Returning TTS Audio`
- [ ] Add interruption handling:
  stop active TTS generation,
  stop outbound packet send,
  send `clear` only for MP3 mode.
  DESIGN reference: `Returning TTS Audio`

## Phase 6: Inbound and Outbound Product Flows

Close the loop on complete call lifecycle behavior.

### 6.1 - Inbound voice agent flow

- [ ] Implement `call.initiated` -> `answer + streaming` -> WebSocket media -> ASR -> app -> TTS -> hangup flow.
  DESIGN reference: `Inbound Call Handler Design`
- [ ] Validate that `stream_bidirectional_target_legs=self` is correct for the initial single-leg AI call pattern.
  DESIGN reference: `Open Concerns`
- [ ] Add structured logs for `call_control_id`, `stream_id`, codec, and observed sample rate.
  DESIGN reference: `Open Concerns`

### 6.2 - Outbound voice agent flow

- [ ] Implement application-triggered `POST /v2/calls` with inline streaming parameters.
  DESIGN reference: `Outbound Call Handler Design`
- [ ] Use `stream_establish_before_call_originate=true` in the first outbound implementation.
  DESIGN reference: `Outbound Call Handler Design`
- [ ] Support optional initial greeting once the media session is ready.
  DESIGN reference: `Outbound Call Handler Design`

## Phase 7: Verification, Examples, and Operational Docs

Make the first slice reviewable and runnable.

### 7.1 - Local and simulated verification

- [ ] Add JSON fixtures for Telnyx webhook and WebSocket messages.
  DESIGN reference: `Testing Scope for PLAN`
- [ ] Add a loopback harness that replays captured inbound media frames into the gateway and verifies:
  transcript updates,
  TTS packet emission,
  session teardown.
  DESIGN reference: `Testing Scope for PLAN`
- [ ] Add regression tests for out-of-order chunks, duplicate chunks, empty media, and unsupported codecs.
  DESIGN reference: `Testing Scope for PLAN`

### 7.2 - Example and operator docs

- [ ] Add an operator-focused doc or example showing:
  required Telnyx app settings,
  webhook URL,
  WSS endpoint,
  environment variables,
  chosen ASR/TTS bundles.
  DESIGN reference: `Overview`, `Open Concerns`
- [ ] Add one end-to-end example configuration for:
  `sherpa-onnx` ASR + Piper TTS.
  DESIGN reference: `Recommended Audio Normal Form`
- [ ] Document the first live-call checklist, including observed codec logging and `stream_bidirectional_target_legs` validation.
  DESIGN reference: `Open Concerns`

### 7.3 - Live validation

- [ ] Place at least one inbound test call and one outbound test call against a real Telnyx application after local simulation passes.
  DESIGN reference: `Testing Scope for PLAN`
- [ ] Capture and review the exact observed `start.media_format` values from Telnyx.
  DESIGN reference: `Open Concerns`
- [ ] Verify conversational latency is acceptable for the first backend pair and document measured numbers nearby in this PLAN or a follow-up note.
  DESIGN reference: `Real-Time Latency Requirements`

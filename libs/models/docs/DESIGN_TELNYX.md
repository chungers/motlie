# Telnyx Real-Time Voice Integration Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-15 | @codex-macmini-telnyx: Initial Telnyx real-time voice integration design for Motlie. Documents Telnyx API research, recommends a WebSocket media gateway over the existing ASR/TTS contracts, and outlines brownfield gaps around codecs, resampling, duplex orchestration, and deployment. | All |

This document defines a brownfield design for integrating Telnyx programmable voice with Motlie's existing speech stack. The current session-specific product classification has not yet been confirmed by the user, but this proposal assumes brownfield work because Motlie already has established `libs/model` speech and transcription contracts plus curated backends in `libs/models`. The Telnyx integration should extend those seams instead of introducing a parallel speech subsystem.

## Table of Contents

- [Overview](#overview)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Telnyx API Research](#telnyx-api-research)
- [Recommended Integration Shape](#recommended-integration-shape)
- [Inbound Call Handler Design](#inbound-call-handler-design)
- [Outbound Call Handler Design](#outbound-call-handler-design)
- [Gap Analysis](#gap-analysis)
- [Alternatives Considered](#alternatives-considered)
- [Testing Scope for PLAN](#testing-scope-for-plan)
- [Open Concerns](#open-concerns)
- [References](#references)

---

## Overview

### Problem Statement

Motlie now has:

- TTS contracts and a working Piper backend
- ASR contracts plus `whisper.cpp` and `sherpa-onnx` backends
- shared PCM primitives: `AudioSpec`, `PcmChunk`, `PcmEncoding`

It does not yet have a telephony transport layer that can:

- receive live call audio from Telnyx
- normalize Telnyx media frames into `TranscriptionStream`
- route transcript updates into application logic
- synthesize TTS responses with `SpeechModel`
- send outbound call audio back over the same Telnyx media channel

### Solution

Add a Telnyx-facing voice gateway above `libs/model` and `libs/models`.

Recommended deployment shape:

1. `libs/model`
   Remains the stable ASR and TTS contract layer.
2. `libs/models`
   Remains the curated bundle layer for Piper, Fish Speech, `whisper.cpp`, and `sherpa-onnx`.
3. new Rust telephony integration crate, recommended name: `libs/voice_telnyx`
   Owns Telnyx webhook handling, WebSocket media protocol, codec conversion, resampling, duplex orchestration, and Telnyx REST commands.
4. optional deployable service binary, recommended name: `bins/motlie-telnyx-gateway`
   Hosts the HTTP webhook endpoint and WebSocket media endpoint and wires them to application logic.

This keeps telephony transport concerns out of the model contracts while preserving reuse of the existing speech backends.

## Goals and Non-Goals

### Goals

- Integrate Telnyx inbound and outbound calls with Motlie ASR/TTS
- Reuse `TranscriptionModel` / `TranscriptionStream` for inbound speech
- Reuse `SpeechModel` / `SpeechStream` for outbound speech
- Minimize telephony latency by preferring Telnyx bidirectional RTP streaming over queued MP3 playback
- Keep codec and transport adaptation outside `libs/model`
- Support interruption and barge-in at the gateway layer
- Keep room for multiple ASR/TTS backend combinations behind the same gateway

### Non-Goals

- Implementing a SIP endpoint inside Motlie in the first cut
- Using Telnyx-hosted AI assistant, hosted STT, or hosted TTS in the first cut
- Building a generic RTP or SIPREC platform before Telnyx is validated
- Extending `libs/model` into a network server framework
- Solving call-center features such as transfers, conferencing, queueing, or supervisor modes in the first cut

## Telnyx API Research

### Call Control and Media Transport

Telnyx programmable voice is call-control driven:

- call lifecycle events arrive via HTTP webhooks
- call actions are driven with REST commands
- real-time media streaming uses WebSockets

For the Motlie use case, the main path is:

1. receive a voice webhook such as `call.initiated`
2. answer the call or place an outbound `dial`
3. include streaming parameters in `answer` / `dial`, or issue `streaming_start`
4. Telnyx connects to Motlie's `wss://...` media endpoint
5. Telnyx sends JSON WebSocket events for stream start, media frames, DTMF, marks, errors, and stop

Inference from the current docs: Telnyx also supports SIP and RTP-oriented products, but the cleanest Motlie integration point is the WebSocket media streaming path because it already provides call-control hooks and a bidirectional media API without requiring Motlie to become a SIP user agent.

### Inbound Call Handling

Inbound calls are surfaced as webhooks. Telnyx documents `call.initiated` with:

- `direction: "incoming"`
- `state: "parked"`
- `call_control_id`
- `call_leg_id`
- `call_session_id`

That means the inbound call handler should treat the webhook as the trigger to decide whether to:

- reject the call
- answer the call without streaming
- answer the call and attach media streaming immediately

The lowest-friction path for Motlie is to answer and attach streaming in the same `answer` request so there is no extra control round trip before audio starts flowing.

### Outbound Call Handling

Outbound calls are initiated with `POST /v2/calls` using at least:

- `connection_id`
- `to`
- `from`

Telnyx allows streaming parameters to be provided in the same outbound `dial` request, including:

- `stream_url`
- `stream_track`
- `stream_codec`
- `stream_bidirectional_mode`
- `stream_bidirectional_codec`
- `stream_bidirectional_target_legs`
- `stream_bidirectional_sampling_rate`
- `stream_establish_before_call_originate`
- `send_silence_when_idle`

For an AI voice agent, the recommended outbound setup is:

- attach streaming in the original `dial` request
- set `stream_establish_before_call_originate=true`
- set bidirectional RTP mode rather than MP3

That reduces setup latency and avoids losing initial greeting audio while the media connection is still being established.

### Audio Codecs and Formats

Telnyx documents the following stream codec options:

- `PCMU`
- `PCMA`
- `G722`
- `OPUS`
- `AMR-WB`
- `L16`
- `default`

Important details:

- streamed inbound media is delivered as a base64-encoded RTP payload without RTP headers, wrapped in JSON
- bidirectional streaming defaults to `mp3` mode unless `stream_bidirectional_mode=rtp` is requested
- bidirectional RTP supports `PCMU`, `PCMA`, `G722`, `OPUS`, `AMR-WB`, and `L16`
- Telnyx documents `L16` as `16 kHz`
- Telnyx documents `PCMU` and `PCMA` as `8 kHz`
- bidirectional sampling rates can be `8000`, `16000`, `22050`, `24000`, or `48000`

Implications for Motlie:

- the most direct fit to existing ASR/TTS contracts is `L16` bidirectional RTP at `16 kHz`
- PSTN-originated inbound audio may still start as `PCMU` or `PCMA` at `8 kHz`
- the gateway must be able to decode G.711 and possibly other Telnyx codecs if the inbound stream format is not already linear PCM

### WebSocket Event and Chunk Format

Telnyx documents these WebSocket events:

- `connected`
- `start`
- `media`
- `mark`
- `clear`
- `dtmf`
- `error`
- `stop`

The `start` event includes:

- `stream_id`
- `call_control_id`
- `call_session_id`
- `from`
- `to`
- `client_state`
- `media_format.encoding`
- `media_format.sample_rate`
- `media_format.channels`

Inbound media frames arrive as JSON:

```json
{
  "event": "media",
  "sequence_number": "4",
  "media": {
    "track": "inbound",
    "chunk": "2",
    "timestamp": "5",
    "payload": "<base64 RTP payload bytes, no RTP headers>"
  },
  "stream_id": "<uuid>"
}
```

Documented transport semantics:

- the media payload is an RTP payload, not a full RTP packet
- event order is not guaranteed
- chunk numbers should be used to reorder
- bidirectional outbound RTP payloads may be sent back in `media.payload`
- outbound RTP chunks may be `20 ms` to `30 s`
- only one streaming or forking operation is supported per call
- one bidirectional RTP stream per call is supported

### MP3 Mode vs RTP Mode

Telnyx supports two ways to send audio back to the call:

1. `stream_bidirectional_mode=mp3`
   Outbound audio is sent as base64-encoded MP3 in `media` events and is queued for playback.
2. `stream_bidirectional_mode=rtp`
   Outbound audio is sent as base64-encoded RTP payload bytes in `media` events.

`mp3` is not a good default for Motlie voice agents because:

- it adds encode latency
- it introduces queue semantics that work against low-latency interruption
- it does not align with Motlie's chunked PCM speech streams

Recommended choice: `rtp` mode with `L16` when available.

### Telnyx Features We Should Not Use First

Telnyx also exposes:

- built-in transcription commands
- AI assistant attachment
- TTS / speak-text features

Those are useful products, but they bypass Motlie's local ASR/TTS stack. They should be treated as alternatives, not the first integration path.

## Recommended Integration Shape

### High-Level Architecture

```text
Telnyx webhook -> motlie-telnyx-gateway HTTP handler
               -> Telnyx REST answer/dial/streaming_start

Telnyx media WebSocket -> voice_telnyx session
                       -> codec decoder / jitter reorder / resampler
                       -> TranscriptionStream
                       -> application dialog logic
                       -> SpeechModel / SpeechStream
                       -> encoder / packetizer
                       -> Telnyx media WebSocket
```

Primary gateway subsystems:

1. `webhook`
   Validates webhook signatures, routes call events, answers inbound calls, and starts or stops sessions.
2. `call_control_client`
   Thin REST client for `answer`, `dial`, `hangup`, `streaming_start`, and optional metadata updates.
3. `media_server`
   WebSocket server that accepts Telnyx stream connections.
4. `session_registry`
   Maps `call_control_id` and `stream_id` to live session state.
5. `codec`
   Decode and encode G.711 and linear PCM; packetize outbound Telnyx RTP payload bytes.
6. `audio_pipeline`
   Reorder chunks, resample to model-native rates, and adapt into `PcmChunk`.
7. `conversation_orchestrator`
   Owns ASR stream, application turn logic, TTS stream, barge-in rules, and shutdown.

### Why This Belongs Above `libs/model`

`libs/model` contracts are already correct for model I/O:

- `TranscriptionStream::push_chunk()` accepts ordered PCM
- `SpeechStream::next_chunk()` yields ordered PCM

Telnyx integration adds:

- network transport
- JSON framing
- codec transcoding
- jitter and reordering
- call-control lifecycle

Those are telephony concerns, not model capability concerns.

### Recommended Audio Normal Form

Use a gateway-internal normal form of:

- mono
- `16_000 Hz`
- `PcmEncoding::S16Le`

Why:

- `sherpa-onnx` streaming ASR is naturally aligned with `16 kHz`
- `whisper.cpp` can also consume normalized mono PCM
- Telnyx `L16` supports `16 kHz`
- it minimizes surprises across the current ASR/TTS backends

If a TTS backend emits a different format, the gateway should resample before sending to Telnyx.

## Inbound Call Handler Design

### Control Flow

Recommended inbound flow:

1. Telnyx sends `call.initiated` webhook.
2. HTTP handler verifies signature and parses `call_control_id`.
3. The handler allocates a `PendingCallSession` keyed by `call_control_id`.
4. The handler answers the call with:
   - `stream_url=wss://.../telnyx/media`
   - `stream_track=inbound_track`
   - `stream_bidirectional_mode=rtp`
   - `stream_bidirectional_codec=L16`
   - `stream_bidirectional_sampling_rate=16000`
   - `stream_bidirectional_target_legs=self`
5. Telnyx opens the WebSocket.
6. On `start`, the gateway finalizes session media metadata and opens `TranscriptionStream`.
7. Each inbound `media` event is decoded, reordered, converted to normalized PCM, and pushed into the ASR stream.
8. ASR updates are sent to application logic.
9. Application logic produces response text.
10. TTS converts response text into PCM chunks.
11. Gateway encodes outbound audio into the configured Telnyx format and sends `media` events back on the same WebSocket.
12. On hangup or `stop`, the gateway finishes the ASR stream and tears down the session.

### Rust Session Types

Recommended shapes:

```rust
pub struct TelnyxGateway {
    sessions: Arc<SessionRegistry>,
    telnyx: TelnyxCallControlClient,
    app: Arc<dyn VoiceApplication>,
}

pub struct TelnyxCallSession {
    pub call_control_id: String,
    pub call_session_id: Option<String>,
    pub stream_id: Option<String>,
    pub inbound_track: TrackState,
    pub outbound_codec: TelnyxCodec,
    pub inbound_decoder: Box<dyn AudioDecoder>,
    pub outbound_encoder: Box<dyn AudioEncoder>,
    pub asr: Box<dyn TranscriptionStream>,
    pub current_tts: Option<Box<dyn SpeechStream>>,
}

#[async_trait]
pub trait VoiceApplication: Send + Sync {
    async fn on_transcription(
        &self,
        call: &CallContext,
        update: TranscriptionUpdate,
    ) -> Result<Vec<VoiceAction>, VoiceAppError>;
}
```

This keeps Telnyx-specific types outside the model crates while allowing application logic to remain transport-agnostic.

### Mapping Telnyx Frames to `TranscriptionStream`

Recommended mapping:

1. Parse the WebSocket `start` frame.
2. Read `media_format` and instantiate the matching decoder.
3. Create a reorder buffer keyed by Telnyx `media.chunk` within each track.
4. For each ordered audio payload:
   - base64 decode
   - decode from Telnyx codec to linear PCM
   - resample to `16 kHz` if needed
   - convert to `PcmChunk { data, sequence, end_of_stream: false }`
5. Push into `TranscriptionStream`.
6. If `push_chunk()` returns `Some(update)`, forward it to the dialog engine.
7. On WebSocket `stop`, call `finish()` with a final `end_of_stream` chunk if needed.

Sequence mapping detail:

- Telnyx `sequence_number` is at the event level
- Telnyx `media.chunk` is specific to media sequencing
- `TranscriptionStream` needs one monotonic sequence

Recommended gateway rule:

- reorder by `media.chunk` per track
- emit a gateway-local `u64` sequence counter after reordering

This isolates Telnyx transport quirks from the ASR contract.

### Feeding Application Logic

The gateway should not bake dialog policy into the Telnyx crate. Instead:

- ASR partials and finals are delivered to an application callback
- the callback returns actions such as:
  - continue listening
  - speak text
  - hang up
  - clear current playback

This is the cleanest seam for integrating Motlie-specific agent logic later.

### Returning TTS Audio

When the application wants to speak:

1. select a `SpeechModel`
2. open a `SpeechStream` with a `SpeechRequest`
3. read `next_chunk()` until exhaustion or interruption
4. normalize PCM to the configured outbound codec and sampling rate
5. packetize into Telnyx `media` WebSocket events

If the caller barges in while TTS is active:

- stop reading the current `SpeechStream`
- call `finish()` to release backend resources
- stop sending outbound RTP frames
- if MP3 mode is ever used, also send Telnyx `clear`

### Why `stream_track=inbound_track`

Motlie ASR should usually ingest only caller speech. Streaming both tracks by default would mix:

- remote caller audio
- Motlie-generated TTS audio

That would degrade ASR and complicate turn-taking. `both_tracks` should be reserved for diagnostics or analytics, not the default conversational path.

## Outbound Call Handler Design

### Control Flow

Recommended outbound flow:

1. Application requests a call to a destination number or SIP URI.
2. Gateway issues `POST /v2/calls`.
3. The dial request includes:
   - `connection_id`
   - `to`
   - `from`
   - `webhook_url` override if needed
   - `stream_url`
   - `stream_track=inbound_track`
   - `stream_bidirectional_mode=rtp`
   - `stream_bidirectional_codec=L16`
   - `stream_bidirectional_sampling_rate=16000`
   - `stream_establish_before_call_originate=true`
4. Telnyx establishes the WebSocket.
5. The gateway starts the conversation session on `start`.
6. Once the callee answers, the application may:
   - play an initial greeting via TTS
   - wait for the callee to speak first
7. The rest of the session uses the same duplex media path as inbound calls.

### Why Streaming Should Be Attached at Dial Time

Two valid patterns exist:

1. attach streaming in the original `dial`
2. dial first, then call `streaming_start`

Recommendation: use the first pattern for voice agents because:

- it removes an extra REST round trip
- it allows `stream_establish_before_call_originate=true`
- it makes early media setup deterministic

### Outbound Session API Sketch

```rust
pub struct OutboundCallRequest {
    pub to: String,
    pub from: String,
    pub connection_id: String,
    pub initial_prompt: Option<String>,
}

impl TelnyxGateway {
    pub async fn start_outbound_call(
        &self,
        request: OutboundCallRequest,
    ) -> Result<CallHandle, GatewayError> {
        // 1. allocate session id
        // 2. call Telnyx /v2/calls with inline streaming params
        // 3. wait for websocket start
        // 4. optionally queue initial TTS greeting
        todo!()
    }
}
```

The call-control client should remain explicit instead of hiding Telnyx behavior behind a generic telephony abstraction too early.

## Gap Analysis

### 1. Codec and Container Gaps

Current Motlie contracts operate on raw PCM only. Telnyx may deliver or require:

- `PCMU`
- `PCMA`
- `G722`
- `OPUS`
- `AMR-WB`
- `L16`

Required additions outside `libs/model`:

- G.711 mu-law decoder and encoder
- G.711 A-law decoder and encoder
- `L16` framing helpers
- optionally `G722` / `Opus` support if Telnyx account or call routing yields those codecs

Recommendation:

- Phase 1 support `PCMU`, `PCMA`, and `L16`
- explicitly reject unsupported codecs at session start with a clear error and hangup policy
- defer `G722`, `OPUS`, and `AMR-WB` until demand is proven

### 2. Resampling and Format Normalization

Current model contracts already carry `AudioSpec`, but they do not provide conversion utilities.

Required gateway utilities:

- `8 kHz` -> `16 kHz` upsampling for PSTN-originated `PCMU` / `PCMA`
- backend-output resampling from TTS-native rates to Telnyx outbound rate
- `f32` <-> `s16le` conversion as needed

This is a utility-layer gap, not a contract-design gap.

### 3. WebSocket and REST Infrastructure

Motlie has no Telnyx transport layer yet.

Needed:

- HTTP webhook server
- WebSocket server endpoint for Telnyx media streams
- Telnyx REST client
- session registry and cleanup logic
- optional webhook signature verification utilities

Important nuance: the media path needs a WebSocket server, not a client, because Telnyx connects to Motlie's `stream_url`.

### 4. Reordering and Jitter Handling

`TranscriptionStream` expects ordered chunks. Telnyx explicitly documents that media event order is not guaranteed.

Needed:

- per-track reorder buffer
- bounded jitter timeout
- gap detection metrics

This is a real-time transport gap that must be solved before audio is pushed into ASR.

### 5. Real-Time Latency Requirements

Inference, not a documented Telnyx contract:

- conversational UX needs steady `20 ms` frame handling
- ASR partials should ideally surface every `200-500 ms`
- end-of-turn detection should usually settle within `300-700 ms` after trailing silence
- first TTS audio should begin within roughly `400-800 ms` after response text is ready

Design implication:

- avoid MP3 mode for conversational audio
- avoid large internal buffering
- keep outbound RTP packets near Telnyx's minimum chunk size rather than multi-second payloads

### 6. Contract Gaps in `libs/model`

The existing speech and transcription traits are already good enough for a first Telnyx slice. The main missing pieces are adapters and orchestration, not core model capabilities.

Potential follow-on improvements:

- a shared audio utility module for resampling and sample-format conversion
- a small cancellation helper around `SpeechStream` to make barge-in easier to express
- optional timestamp metadata on `PcmChunk` if future transports need precise wall-clock alignment

These are useful, but they should not block Phase 1.

### 7. Bundle and Backend Selection Gaps

The gateway needs a clean way to choose:

- ASR backend and bundle
- TTS backend and bundle
- voice and language defaults

Recommendation:

- keep this as gateway config rather than extending `libs/models` selectors again right now
- reference existing bundle enums and start options directly

## Alternatives Considered

### Alternative 1: Telnyx WebSocket Streaming + Motlie Local ASR/TTS

Decision: recommended.

Pros:

- direct reuse of current Motlie speech contracts
- lowest-latency documented Telnyx path for custom AI media
- no SIP stack required in Motlie
- keeps ASR/TTS model control inside Motlie

Cons:

- requires codec and transport work
- requires running a public HTTPS/WSS service

### Alternative 2: Telnyx MP3 Bidirectional Playback

Decision: reject for real-time voice agent use.

Pros:

- simpler outbound path
- queued playback semantics are easy for fixed prompts

Cons:

- encode latency
- queue semantics hurt interruption
- poorer fit to chunked PCM contracts

### Alternative 3: Telnyx Hosted AI / Hosted Speech Features

Decision: reject for the first Motlie integration.

Pros:

- less engineering work
- fewer transport details to manage

Cons:

- bypasses Motlie's local model stack
- reduces control over bundle selection and speech quality
- does not validate the core Motlie voice architecture the repo is already building

### Alternative 4: Direct SIP or RTP Integration First

Decision: defer.

Pros:

- potentially more telephony-portable long term
- avoids some WebSocket framing work

Cons:

- much larger implementation surface
- not necessary to validate Telnyx integration
- conflicts with the goal of a surgical brownfield slice

## Testing Scope for PLAN

- webhook parsing and signature verification tests
- Telnyx WebSocket protocol parsing tests for `connected`, `start`, `media`, `dtmf`, `error`, `stop`
- codec tests for `PCMU`, `PCMA`, and `L16`
- reorder-buffer tests for out-of-order `media.chunk`
- resampler tests for `8 kHz` inbound to `16 kHz` normalized PCM
- end-to-end simulated call tests that feed media frames into `TranscriptionStream`
- loopback tests that synthesize TTS and verify outbound `media` frames
- env-gated integration tests against a real Telnyx test number after local simulation passes

## Open Concerns

- The user still needs to confirm whether this work should be treated as greenfield or brownfield in the product sense. This document currently assumes brownfield.
- Telnyx documents codec options broadly, but actual inbound codec selection may depend on carrier, destination, and account configuration. Phase 1 should log observed `media_format` values in real calls before broadening codec support.
- `stream_bidirectional_target_legs=self` is the best current inference for single-leg AI-agent calls, but this should be validated on the first live call because Telnyx defaults to `opposite`.
- If Fish Speech becomes the preferred TTS backend, its native sample rate and chunk cadence need to be measured against Telnyx's RTP pacing requirements before promotion.

## References

- Telnyx media streaming docs: https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming
- Telnyx `streaming_start` API: https://developers.telnyx.com/api-reference/call-commands/streaming-start
- Telnyx `dial` API: https://developers.telnyx.com/api-reference/call-commands/dial
- Telnyx voice webhooks overview: https://developers.telnyx.com/docs/voice/programmable-voice/voice-api-webhooks
- Telnyx `call.initiated` webhook: https://developers.telnyx.com/api-reference/callbacks/call-initiated
- Existing ASR design: [DESIGN_ASR.md](/Users/dchung/sessions/codex-macmini-telnyx/motlie/libs/models/docs/DESIGN_ASR.md)
- Existing TTS design: [DESIGN_TTS.md](/Users/dchung/sessions/codex-macmini-telnyx/motlie/libs/models/docs/DESIGN_TTS.md)
- Existing API contracts: [transcription.rs](/Users/dchung/sessions/codex-macmini-telnyx/motlie/libs/model/src/transcription.rs), [speech.rs](/Users/dchung/sessions/codex-macmini-telnyx/motlie/libs/model/src/speech.rs)

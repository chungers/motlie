# Telnyx Real-Time Voice Integration Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-16 | @codex-macmini-telnyx: Added an explicit media-adaptation pipeline design with typed stage contracts, marker types, and compile-time pipeline assembly guidance for codecs, resampling, framing, and model-specific normalization. | Recommended Integration Shape, Media Adaptation Pipeline, Inbound Call Handler Design, Gap Analysis |
| 2026-04-15 | @codex-macmini-telnyx: Added the pluggable `ConversationHandler` stage, made ASR/TTS injection explicit, and documented private-host deployment plus a local getting-started flow using ngrok or Tailscale Funnel. | Overview, Recommended Integration Shape, Recommended ASR/TTS Stack, Inbound Call Handler Design, Deployment: Private Host, Getting Started: Local Deployment, Open Concerns |
| 2026-04-15 | @codex-macmini-telnyx: Added the recommended v1 ASR/TTS stack, concrete telephony pipeline, and latency budget based on current Motlie benchmark results and backend readiness. | Recommended ASR/TTS Stack, Gap Analysis, Open Concerns |
| 2026-04-15 | @codex-macmini-telnyx: Initial Telnyx real-time voice integration design for Motlie. Documents Telnyx API research, recommends a WebSocket media gateway over the existing ASR/TTS contracts, and outlines brownfield gaps around codecs, resampling, duplex orchestration, and deployment. | All |

This document defines a brownfield design for integrating Telnyx programmable voice with Motlie's existing speech stack. The current session-specific product classification has not yet been confirmed by the user, but this proposal assumes brownfield work because Motlie already has established `libs/model` speech and transcription contracts plus curated backends in `libs/models`. The Telnyx integration should extend those seams instead of introducing a parallel speech subsystem.

## Table of Contents

- [Overview](#overview)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Telnyx API Research](#telnyx-api-research)
- [Recommended Integration Shape](#recommended-integration-shape)
- [Media Adaptation Pipeline](#media-adaptation-pipeline)
- [Recommended ASR/TTS Stack](#recommended-asrtts-stack)
- [Inbound Call Handler Design](#inbound-call-handler-design)
- [Outbound Call Handler Design](#outbound-call-handler-design)
- [Deployment: Private Host](#deployment-private-host)
- [Getting Started: Local Deployment](#getting-started-local-deployment)
- [v1.1: DTMF and Call Control](#v11-dtmf-and-call-control)
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
- route transcript updates into application logic through a pluggable conversation stage
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
- Make the conversation stage explicitly pluggable for application-specific logic
- Reuse `TranscriptionModel` / `TranscriptionStream` for inbound speech
- Reuse `SpeechModel` / `SpeechStream` for outbound speech
- Allow any `TranscriptionModel` and `SpeechModel` implementation to be injected without changing gateway code
- Minimize telephony latency by preferring Telnyx bidirectional RTP streaming over queued MP3 playback
- Keep codec and transport adaptation outside `libs/model`
- Support interruption and barge-in at the gateway layer
- Keep room for multiple ASR/TTS backend combinations behind the same gateway
- Support private-host deployment behind ngrok or Tailscale Funnel for v1

### Non-Goals

- Implementing a SIP endpoint inside Motlie in the first cut
- Using Telnyx-hosted AI assistant, hosted STT, or hosted TTS in the first cut
- Building a generic RTP or SIPREC platform before Telnyx is validated
- Extending `libs/model` into a network server framework
- Solving call-center features such as transfers, conferencing, queueing, or supervisor modes in the first cut
- Assuming cloud deployment, public IPs, or load balancers in v1

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
                       -> ConversationHandler
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
   Owns ASR stream, `ConversationHandler`, TTS stream, barge-in rules, and shutdown.
8. `conversation_handler`
   The application-specific async text-processing stage where LLM calls, business logic, RAG, policy checks, or workflow actions run.

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
- application conversation orchestration

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

## Media Adaptation Pipeline

### Why This Must Be Explicit

The speech traits in `libs/model` intentionally normalize capability shape, but they do not remove transport and model-specific media adaptation work.

The Telnyx gateway still has to reconcile:

- transport codecs such as `PCMU`, `PCMA`, and `L16`
- transport pacing and chunk boundaries
- jitter and reorder behavior
- model-specific preferred sample rates
- model-specific sample encodings
- output packetization for outbound telephony pacing

So the Telnyx design must not treat the audio path as “just a stream.” It should be assembled from explicit stages with typed inputs and outputs.

### Stage Breakdown

Recommended inbound stages:

```text
Telnyx media payload
-> transport decode
-> reorder / jitter normalization
-> sample-format conversion
-> resampling
-> ASR framing adapter
-> ASR model
```

Recommended outbound stages:

```text
TTS model
-> TTS chunk drain
-> sample-format conversion
-> resampling
-> telephony pacing / packetization
-> transport encode
-> Telnyx media payload
```

### Stage API Surface

Recommended common stage shape:

```rust
pub trait Stage {
    type In;
    type Out;
    type Error;

    fn transform(&mut self, input: Self::In) -> Result<Self::Out, Self::Error>;
}
```

For async boundaries such as model calls or external orchestration:

```rust
#[async_trait]
pub trait AsyncStage {
    type In;
    type Out;
    type Error;

    async fn transform(&mut self, input: Self::In) -> Result<Self::Out, Self::Error>;
}
```

Design rule:

- pure media transforms such as decode, resample, and rechunk should prefer `Stage`
- model or network boundaries should use `AsyncStage`
- stages should transform typed media values rather than raw `Vec<u8>` wherever possible

### Typed Media Values

The gateway should wrap payloads in domain-specific types so a stage cannot accidentally accept “some bytes” without knowing what they represent.

Recommended shapes:

```rust
use core::marker::PhantomData;

pub struct Mono;
pub struct Stereo;

pub struct Hz8000;
pub struct Hz16000;
pub struct Hz22050;
pub struct Hz24000;

pub struct S16Le;
pub struct F32Le;
pub struct Pcmu;
pub struct Pcma;
pub struct L16;

pub struct EncodedFrame<C> {
    pub bytes: Vec<u8>,
    pub sequence: u64,
    pub end_of_stream: bool,
    _codec: PhantomData<C>,
}

pub struct PcmFrame<R, Ch, E> {
    pub bytes: Vec<u8>,
    pub sequence: u64,
    pub end_of_stream: bool,
    _rate: PhantomData<R>,
    _channels: PhantomData<Ch>,
    _encoding: PhantomData<E>,
}
```

This makes it impossible to confuse:

- G.711 payload bytes with linear PCM
- `8 kHz` PCM with `16 kHz` PCM
- `S16Le` with `F32Le`

without writing an explicit conversion stage.

### Marker Traits

Marker traits should describe what a stage consumes or produces.

Recommended examples:

```rust
pub trait TelephonyCodec {}
impl TelephonyCodec for Pcmu {}
impl TelephonyCodec for Pcma {}
impl TelephonyCodec for L16 {}

pub trait LinearPcmEncoding {}
impl LinearPcmEncoding for S16Le {}
impl LinearPcmEncoding for F32Le {}

pub trait SampleRateTag {}
impl SampleRateTag for Hz8000 {}
impl SampleRateTag for Hz16000 {}
impl SampleRateTag for Hz22050 {}
impl SampleRateTag for Hz24000 {}

pub trait ChannelLayoutTag {}
impl ChannelLayoutTag for Mono {}
impl ChannelLayoutTag for Stereo {}
```

These are zero-cost type-level markers. They are there to make legal assembly obvious and illegal assembly hard.

### Common Stage Contracts

Recommended transport codec stage:

```rust
pub trait Decoder<C, R, Ch, E>
where
    C: TelephonyCodec,
    R: SampleRateTag,
    Ch: ChannelLayoutTag,
    E: LinearPcmEncoding,
{
    fn decode(&mut self, input: EncodedFrame<C>) -> Result<PcmFrame<R, Ch, E>, MediaError>;
}

pub trait Encoder<C, R, Ch, E>
where
    C: TelephonyCodec,
    R: SampleRateTag,
    Ch: ChannelLayoutTag,
    E: LinearPcmEncoding,
{
    fn encode(&mut self, input: PcmFrame<R, Ch, E>) -> Result<EncodedFrame<C>, MediaError>;
}
```

Recommended resampler stage:

```rust
pub trait Resampler<InRate, OutRate, Ch, E>
where
    InRate: SampleRateTag,
    OutRate: SampleRateTag,
    Ch: ChannelLayoutTag,
    E: LinearPcmEncoding,
{
    fn resample(
        &mut self,
        input: PcmFrame<InRate, Ch, E>,
    ) -> Result<PcmFrame<OutRate, Ch, E>, MediaError>;
}
```

Recommended sample-format conversion stage:

```rust
pub trait SampleConverter<R, Ch, InEnc, OutEnc>
where
    R: SampleRateTag,
    Ch: ChannelLayoutTag,
    InEnc: LinearPcmEncoding,
    OutEnc: LinearPcmEncoding,
{
    fn convert(
        &mut self,
        input: PcmFrame<R, Ch, InEnc>,
    ) -> Result<PcmFrame<R, Ch, OutEnc>, MediaError>;
}
```

Recommended framing stage for ASR:

```rust
pub trait AsrChunker<R, Ch, E>
where
    R: SampleRateTag,
    Ch: ChannelLayoutTag,
    E: LinearPcmEncoding,
{
    fn push_frame(
        &mut self,
        input: PcmFrame<R, Ch, E>,
    ) -> Result<Vec<motlie_model::PcmChunk>, MediaError>;
}
```

Recommended pacing stage for telephony output:

```rust
pub trait Packetizer<R, Ch, E, C>
where
    R: SampleRateTag,
    Ch: ChannelLayoutTag,
    E: LinearPcmEncoding,
    C: TelephonyCodec,
{
    fn push_frame(
        &mut self,
        input: PcmFrame<R, Ch, E>,
    ) -> Result<Vec<EncodedFrame<C>>, MediaError>;
}
```

### Behavior Contracts

Each stage should document strict behavior.

Decoder and encoder:

- must preserve monotonic `sequence`
- must not silently reinterpret one codec as another
- must reject unsupported payload lengths or invalid framing with a typed error

Resampler:

- must preserve `end_of_stream`
- must not change channel layout or encoding
- must state whether it buffers internally and when buffered audio is flushed

Sample converter:

- must not change sample rate or channels
- must document clipping, rounding, and saturation behavior

Chunker:

- must produce deterministic `PcmChunk.sequence`
- must state whether it emits fixed frame sizes or model-tuned windows
- must flush trailing buffered audio on end-of-stream

Packetizer:

- must produce telephony-sized packets at a documented cadence
- must not emit oversized outbound payloads that increase interactive latency
- must preserve ordering and end-of-stream semantics

### Model-Specific Adaptation

This design should make model requirements explicit instead of hiding them in the transport loop.

Examples:

- Sherpa ONNX streaming may want mono `16 kHz` PCM frames with a chunk cadence aligned to its streaming decoder.
- Moonshine streaming may also be telephony-friendly, but it should still get its own chunker configuration if its best frame sizing differs.
- Whisper can satisfy the `TranscriptionModel` interface, but it likely needs a different buffering policy and should not reuse the low-latency streaming defaults blindly.
- Piper may emit one `AudioSpec`, while Qwen3-TTS may emit another; the outbound side must adapt from the actual emitted type, not from a hardcoded assumption.

Design implication:

- model selection determines which normalization and framing stages are assembled
- the surrounding transport pipeline remains structurally the same

### Compile-Time Pipeline Assembly

The recommended assembly pattern is a typed builder or typed pipeline chain, not ad hoc wiring.

Sketch:

```rust
pub struct Pipeline<S1, S2, S3> {
    s1: S1,
    s2: S2,
    s3: S3,
}

impl<S1, S2, S3> Pipeline<S1, S2, S3>
where
    S1: Stage,
    S2: Stage<In = S1::Out>,
    S3: Stage<In = S2::Out>,
{
    pub fn run(&mut self, input: S1::In) -> Result<S3::Out, PipelineError> {
        let a = self.s1.transform(input)?;
        let b = self.s2.transform(a)?;
        let c = self.s3.transform(b)?;
        Ok(c)
    }
}
```

That enforces stage compatibility through `S2::In = S1::Out` and `S3::In = S2::Out`.

### Recommended Safety Properties

The pipeline design should make these bugs impossible or very obvious:

- feeding encoded G.711 bytes directly into an ASR chunker
- resampling a payload that is still compressed instead of decoded
- encoding outbound telephony packets from the wrong PCM rate
- forgetting whether a given frame is `S16Le` or `F32Le`
- guessing whether a stage is operating on `8 kHz` or `16 kHz`

The type system does not replace runtime validation, but it should prevent “everything is bytes” assembly mistakes.

### Static-Dispatch Model Injection

The gateway must not hardcode `sherpa-onnx`, `whisper.cpp`, Piper, Fish Speech, or any other concrete backend. It also should not use `dyn` at the gateway boundary, because the supported model universe is known at build time through Cargo features.

Recommended construction rule:

- make the Telnyx gateway generic over ASR, TTS, and conversation-handler types
- choose concrete model bundles at build time through `motlie-models` feature flags
- use closed enums for operator-facing selection when more than one compiled model is present

Recommended generic shape:

```rust
pub struct TelnyxGateway<A, T, H> {
    asr: A,
    tts: T,
    handler: H,
    telnyx: TelnyxCallControlClient,
    sessions: SessionRegistry,
}
```

Where:

- `A` is the concrete ASR capability handle or wrapper
- `T` is the concrete TTS capability handle or wrapper
- `H` is the concrete `ConversationHandler`

This keeps the transport reusable while letting callers swap:

- `sherpa-onnx` for Moonshine or `whisper.cpp`
- Piper for Qwen3-TTS
- a trivial echo handler for an LLM-backed assistant

without changing gateway code and without introducing gateway-level trait-object dispatch.

### Build-Time Model Surface

The relevant `motlie-models` feature flags are:

- ASR:
  - `model-sherpa-onnx-streaming`
  - `model-moonshine-streaming`
  - `model-whisper-base-en`
- TTS:
  - `model-piper-en-us-ljspeech-medium`
  - `model-qwen3-tts-0_6b`

Examples:

```bash
cargo build --release -p motlie-telnyx-gateway \
  --no-default-features \
  --features "model-sherpa-onnx-streaming model-piper-en-us-ljspeech-medium"

cargo build --release -p motlie-telnyx-gateway \
  --no-default-features \
  --features "model-moonshine-streaming model-qwen3-tts-0_6b"
```

Design implication:

- a Telnyx deployment should compile only the ASR/TTS bundles it intends to operate
- the gateway binary should fail at compile time if the required model feature set is not enabled
- when multiple ASR or TTS bundles are compiled in, selection should be done via closed enums from `motlie_models::AsrModels` and `motlie_models::TtsModels`, not via open-ended runtime plugin loading

### Closed-Enum Selection

For builds that include more than one ASR or TTS backend, the gateway should prefer closed enums over trait objects.

Sketch:

```rust
pub enum TelnyxAsrSelection {
    #[cfg(feature = "model-sherpa-onnx-streaming")]
    SherpaOnnxStreaming,
    #[cfg(feature = "model-moonshine-streaming")]
    MoonshineStreaming,
    #[cfg(feature = "model-whisper-base-en")]
    WhisperBaseEn,
}

pub enum TelnyxTtsSelection {
    #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
    PiperEnUsLjspeechMedium,
    #[cfg(feature = "model-qwen3-tts-0_6b")]
    Qwen3Tts12Hz0_6B,
}
```

These enums map cleanly onto the existing `motlie_models::AsrModels` and `motlie_models::TtsModels` surfaces.

### Current Lower-Layer Constraint

The current `libs/model` and `libs/models` internals still use dynamic capability return types in `ModelBundle` and `BundleHandle`. So the design can remove `dyn` from the gateway architecture immediately, but a fully static-dispatch implementation all the way down would require follow-on work in the model layer itself.

That should be documented explicitly so the design does not pretend the existing lower-level loader surface is already fully monomorphized.

### Conversation Handler Contract

Between ASR transcript output and TTS response input, the gateway should expose a dedicated pluggable processing stage.

Recommended conversational pipeline:

```text
Inbound audio
-> ASR
-> transcript text
-> ConversationHandler
-> response text
-> TTS
-> outbound audio
```

Recommended Rust shapes:

```rust
use async_trait::async_trait;
use std::collections::BTreeMap;

pub struct ConversationTurn {
    pub speaker: TurnSpeaker,
    pub text: String,
}

pub enum TurnSpeaker {
    Caller,
    Assistant,
    System,
}

pub struct ConversationContext {
    pub call_id: String,
    pub turn_history: Vec<ConversationTurn>,
    pub custom_state: BTreeMap<String, String>,
}

pub struct ConversationError {
    pub message: String,
}

#[async_trait]
pub trait ConversationHandler: Send + Sync {
    async fn handle(
        &self,
        transcript: &str,
        context: &mut ConversationContext,
    ) -> Result<String, ConversationError>;
}
```

This is the main extensibility point for:

- LLM inference
- business logic
- RAG lookups
- workflow integrations
- deterministic rule engines

The transport and speech stack stay reusable infrastructure. The handler is the application-specific component.

### Streaming Conversation Responses

The handler should also support a streaming mode for lower latency so partial text can begin TTS before the full response is ready.

Recommended extension path:

```rust
use core::pin::Pin;
use futures_core::Stream;

pub type TextChunkStream =
    Pin<Box<dyn Stream<Item = Result<String, ConversationError>> + Send>>;

#[async_trait]
pub trait ConversationHandler: Send + Sync {
    async fn handle(
        &self,
        transcript: &str,
        context: &mut ConversationContext,
    ) -> Result<String, ConversationError>;

    async fn handle_streaming(
        &self,
        transcript: &str,
        context: &mut ConversationContext,
    ) -> Result<TextChunkStream, ConversationError>;
}
```

The exact chunking policy should remain gateway-local so the handler returns text fragments and the gateway decides when those fragments are stable enough to synthesize.

## Recommended ASR/TTS Stack

### Recommended v1 Stack

For Telnyx real-time voice, the recommended Motlie v1 stack is:

- ASR: `sherpa-onnx` streaming Zipformer2
- TTS: Piper

This recommendation is based on the current Motlie benchmark and implementation status, not on long-term model ambition.
It is guidance for v1 defaults, not a hard dependency of the gateway architecture.

### Recommended ASR: `sherpa-onnx`

`sherpa-onnx` is the only viable ASR backend for Telnyx live telephony in the current Motlie stack.

Current evidence from Motlie's benchmarks and implementation status:

- `sherpa-onnx` streaming Zipformer2: about `5.2 s` mean latency on CPU, `0.30` WER, true chunk-by-chunk streaming
- `whisper.cpp`: about `19.7 s` mean latency, batch-oriented in Motlie today, not suitable for live telephony turn handling

Implications:

- telephony needs incremental processing and sub-second chunk handling
- `sherpa-onnx` is the only backend that satisfies that shape today
- `whisper.cpp` remains useful for offline or file-oriented ASR, but it should not be selected for the Telnyx conversational path

Recommended design rule:

- Telnyx v1 should treat `sherpa-onnx` as the default recommended ASR backend for the real-time conversational flow
- the gateway itself should still accept any injected `TranscriptionModel`

### Recommended TTS: Piper

Piper is the recommended Telnyx v1 TTS backend because it is the only TTS backend currently producing working speech in Motlie.

Current evidence from Motlie's implementation status:

- short utterance generation: about `70 ms`
- paragraph generation: about `3.6 s`
- stable working output today

Why Piper is good enough for Telnyx v1:

- short telephony responses are the common case
- `~70 ms` generation for short responses is fast enough for conversational turn taking
- output is already PCM, which fits the gateway normalization and re-encoding pipeline cleanly

Current Piper limitations:

- no voice cloning
- one pre-trained voice per model
- less flexibility than future expressive or clone-capable TTS stacks

Phase 3 status:

- Fish Speech is not yet producing speech in Motlie
- Qwen3-TTS via GGUF is not yet producing speech in Motlie

Recommended design rule:

- Telnyx v1 should standardize on Piper as the default recommended backend and defer richer voice-selection work until a second TTS backend is actually operational
- the gateway itself should still accept any injected `SpeechModel`

### Recommended Telnyx v1 Pipeline

The recommended end-to-end voice path is:

```text
Inbound Telnyx audio
-> G.711 decode
-> resample 8 kHz to 16 kHz
-> sherpa-onnx streaming ASR
-> transcript
-> ConversationHandler
-> response text
-> Piper TTS
-> PCM at about 22 kHz
-> resample to 16 kHz
-> encode G.711 or L16
-> outbound Telnyx media
```

This pipeline aligns with what currently works instead of over-optimizing for future model combinations that are not yet production-viable.

### Latency Budget

Recommended conversational target:

- Telnyx inbound audio chunk cadence: about `20 ms` for G.711 at `8 kHz`
- ASR processing: `sherpa-onnx` chunk-by-chunk streaming
- TTS generation: about `70 ms` for short Piper responses
- total round-trip target: under `500 ms`

Inference from the current stack:

- the main latency risks are not model startup or raw short-utterance TTS generation
- the main risks are transport buffering, resampling, end-of-turn detection, and oversized outbound packetization

Design implications:

- prefer `sherpa-onnx` over `whisper.cpp` for any real-time Telnyx path
- keep inbound jitter buffers small
- keep TTS responses concise when possible
- packetize outbound audio at telephony-friendly intervals instead of large queued chunks
- prefer RTP mode over MP3 mode to avoid avoidable playback delay

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
8. ASR updates are sent to the `ConversationHandler`.
9. The `ConversationHandler` produces response text.
10. TTS converts response text into PCM chunks.
11. Gateway encodes outbound audio into the configured Telnyx format and sends `media` events back on the same WebSocket.
12. On hangup or `stop`, the gateway finishes the ASR stream and tears down the session.

### Rust Session Types

Recommended shapes:

```rust
pub struct TelnyxGateway {
    sessions: Arc<SessionRegistry>,
    telnyx: TelnyxCallControlClient,
    asr: TelnyxAsrRuntime,
    tts: TelnyxTtsRuntime,
    handler: H,
}

pub struct TelnyxCallSession {
    pub call_control_id: String,
    pub call_session_id: Option<String>,
    pub stream_id: Option<String>,
    pub inbound_track: TrackState,
    pub outbound_codec: TelnyxCodec,
    pub inbound_pipeline: InboundPipeline,
    pub outbound_pipeline: OutboundPipeline,
    pub asr_stream: AsrStreamRuntime,
    pub current_tts: Option<TtsStreamRuntime>,
    pub conversation: ConversationContext,
}
```

Where:

- `TelnyxAsrRuntime` and `TelnyxTtsRuntime` are concrete, feature-gated runtime wrappers selected from the compiled model set rather than open-ended trait-object slots
- `InboundPipeline` is a typed composition such as decode -> reorder -> normalize -> chunk
- `OutboundPipeline` is a typed composition such as convert -> resample -> packetize -> encode
- `AsrStreamRuntime` and `TtsStreamRuntime` are concrete wrappers around the currently selected runtime handles

This keeps Telnyx-specific types outside the model crates while allowing ASR, TTS, and conversation logic to remain transport-agnostic.

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

### Feeding the Conversation Handler

The gateway should not bake dialog policy into the Telnyx crate. Instead:

- ASR partials and finals are converted into transcript text events
- transcript text is delivered to an injected `ConversationHandler`
- the handler reads and mutates `ConversationContext`
- the handler returns response text or a streaming text response
- the gateway turns that text into TTS audio

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

## v1.1: DTMF and Call Control

### DTMF During WebSocket Media Streaming

Telnyx documents a dedicated WebSocket `dtmf` event during media streaming. For the gateway design, this means DTMF should be treated as an out-of-band control signal, not only as audio embedded in the media stream.

Practical implication:

- primary DTMF path: consume Telnyx `dtmf` WebSocket events
- fallback path: detect tones from audio only if required by an edge case or carrier behavior

This is the preferred design because it keeps keypad navigation separate from ASR and avoids confusing tone bursts with speech.

### In-Band vs Out-of-Band DTMF

Current Telnyx evidence:

- media streaming sends a dedicated `dtmf` WebSocket event when DTMF occurs on the call
- Call Control Applications also expose a configurable `dtmf_type` field for how DTMF is sent from Telnyx to the connection

Inference from these sources:

- for the Motlie WebSocket media path, DTMF should usually be modeled as out-of-band because Telnyx already surfaces it separately
- the inbound audio stream may still contain audible DTMF energy depending on remote endpoint behavior, carrier path, or chosen call-control features

Design rule:

- trust the WebSocket `dtmf` event as authoritative when present
- keep optional in-band detection and suppression as a defensive fallback rather than the main control path

### Can We Detect DTMF From Audio Ourselves?

Yes. If the audio stream contains in-band tones, the gateway can detect them with classic DSP approaches such as:

- Goertzel detectors for the standard DTMF frequency pairs
- narrow bandpass analysis around the DTMF rows and columns

This should be treated as a fallback path only.

Reasons not to make audio-only DTMF detection the default:

- Telnyx already provides a control event
- ASR and DTMF detection compete for the same audio frames
- false positives are possible with noisy telephony audio

### Sending DTMF Outbound

Telnyx exposes a `send_dtmf` call-control command for active calls. That is the right mechanism for:

- navigating IVR menus
- entering voicemail PINs
- acknowledging automated systems

The gateway should use the command API rather than synthesizing DTMF tones into the outbound audio stream whenever the goal is remote IVR control.

### Call Control Commands During an Active Call

Telnyx Call Control supports a broad set of active-call commands. Relevant families for Motlie include:

- `answer`, `hangup`, `reject`
- `streaming_start`, `streaming_stop`
- `gather_using_audio`
- `send_dtmf`
- `playback_start`, `playback_stop`
- `speak`
- `enqueue`
- `transfer` / refer-style redirection and bridge-style workflows, depending on the call topology
- recording commands
- conference commands for join, hold, mute, and participant control when the call is moved into a conference

Inference from the command model:

- active-call commands operate on `call_control_id`
- media streaming also operates on the same active call
- therefore call-control commands can be issued while WebSocket media is active

Operational caveat:

- commands like `gather_using_audio`, `playback_start`, and `speak` can overlap with the custom media loop and create competing audio sources
- v1 should avoid mixing Telnyx-hosted prompt playback with Motlie-generated TTS unless there is a clear reason

### How Telnyx Gather Works

Telnyx exposes `gather_using_audio` to play an audio prompt and collect digits from the caller.

That feature is useful for:

- legacy IVR flows
- hybrid applications where some branches use keypad input instead of speech

For the Motlie gateway, `gather_using_audio` should be treated as an optional call-control tool, not the main conversational path, because:

- Motlie already has a custom media loop
- the gateway can receive DTMF events directly
- a local `DtmfHandler` is often simpler than temporarily delegating prompt-plus-digit control back to Telnyx

Recommended rule:

- use direct WebSocket `dtmf` events plus local application logic by default
- reserve `gather_using_audio` for flows where Telnyx-hosted prompt playback is intentionally desired

### Filtering DTMF From Voice Before ASR

If DTMF is present in the audio path, the gateway should keep those tones from degrading ASR.

Recommended handling:

1. consume WebSocket `dtmf` events as the canonical keypad signal
2. when a `dtmf` event arrives, mark the corresponding time window in session state
3. suppress or attenuate the matching inbound audio window before feeding it to ASR

If explicit event timing is insufficient or absent, use a fallback detector such as a Goertzel filter to find the DTMF tone pair and blank or attenuate those frames.

Recommended v1.1 DSP policy:

- zero or attenuate short DTMF windows before ASR
- keep the original event in control flow
- do not send DTMF-derived frames into `ConversationHandler`

### Correlating DTMF Events With Audio Timeline

If DTMF is surfaced out-of-band, the gateway still needs to correlate it with audio timing for filtering and observability.

Recommended approach:

- store arrival metadata from the `dtmf` event
- align it to the session's media chunk clock using the latest ordered chunk index and timestamp
- record a short mute window in normalized PCM time

This does not need sample-perfect accuracy for v1.1. The practical goal is to keep tone bursts out of ASR and expose consistent keypad events to the application layer.

### Proposed `DtmfHandler` Trait

DTMF handling should sit alongside `ConversationHandler`.

Voice path:

```text
audio -> ASR -> ConversationHandler -> TTS
```

DTMF path:

```text
dtmf event -> DtmfHandler -> CallAction
```

Recommended Rust shapes:

```rust
use async_trait::async_trait;

pub enum DtmfDigit {
    D0,
    D1,
    D2,
    D3,
    D4,
    D5,
    D6,
    D7,
    D8,
    D9,
    Star,
    Hash,
    A,
    B,
    C,
    D,
}

pub enum CallAction {
    Continue,
    Transfer { destination: String },
    Hold,
    Hangup,
    PlayAudio { path: String },
    GatherMore { timeout_ms: u64 },
}

#[async_trait]
pub trait DtmfHandler: Send + Sync {
    async fn on_dtmf(
        &self,
        digit: DtmfDigit,
        context: &mut ConversationContext,
    ) -> Result<CallAction, ConversationError>;
}
```

Design intent:

- `ConversationHandler` owns spoken language
- `DtmfHandler` owns keypad control
- both can read and mutate shared call state

### Scripted Outbound IVR Navigation

For outbound calls that must navigate an IVR before reaching a human, the gateway should support a scripted sequence controller.

Example behavior:

- wait for prompt
- send DTMF `1`
- wait for next prompt
- send DTMF `3`
- wait for human pickup

Recommended abstraction:

```rust
#[async_trait]
pub trait IvrNavigator: Send + Sync {
    async fn on_event(
        &self,
        event: IvrEvent,
        context: &mut ConversationContext,
    ) -> Result<Vec<CallAction>, ConversationError>;
}
```

Useful `IvrEvent` inputs could include:

- prompt started
- prompt finished
- DTMF acknowledged
- silence timeout
- transcript matched phrase
- human detected

This keeps outbound IVR automation separate from the human-conversation path while reusing the same call-control client and session state.

## Deployment: Private Host

### v1 Deployment Constraint

The initial Telnyx integration must support running on a private host behind a tunnel, not just on the open internet.

Supported v1 deployment shapes:

- a private DGX host with an ngrok tunnel
- a private mac mini with an ngrok tunnel
- a private Tailscale-connected host using Tailscale Funnel

Not assumed in v1:

- public cloud deployment
- public IP addresses
- load balancers
- Kubernetes ingress

### Why Tunneling Is Required

Telnyx requires reachable external URLs for:

- voice webhooks over HTTPS
- media WebSocket connections over `wss://`

If the gateway runs on a private host, an external tunnel must expose those endpoints.

### Option 1: ngrok

ngrok is the simpler and more disposable option.

Current documented behavior from ngrok:

- `ngrok http <port>` creates a public HTTPS endpoint to a local HTTP service
- HTTP/S endpoints support WebSockets out of the box
- the agent establishes outbound TLS connections; no inbound port opening is required

Recommended v1 ngrok workflow:

1. run the gateway locally on a private host
2. start `ngrok http <gateway-port>`
3. use the generated `https://...ngrok.app` URL for Telnyx webhooks
4. use the matching `wss://...ngrok.app/...` URL for Telnyx media streaming
5. update the Telnyx application webhook URL whenever the ngrok hostname changes

### Option 2: Tailscale Funnel

Tailscale Funnel is the more persistent mesh-native option.

Current documented behavior from Tailscale:

- Funnel exposes a local service on a public `https://<node>.<tailnet>.ts.net` URL
- it requires MagicDNS, HTTPS certificates, and Funnel permission in the tailnet
- it only supports public HTTPS exposure on ports `443`, `8443`, and `10000`

Recommended v1 Funnel workflow:

1. run the gateway locally on the private host
2. expose it with `tailscale funnel`
3. use the resulting `https://<node>.<tailnet>.ts.net` URL for Telnyx webhooks
4. use the matching `wss://<node>.<tailnet>.ts.net/...` URL for Telnyx media streaming

### Gateway Configuration Requirement

The gateway binary should accept a `--webhook-url` flag so the external public URL can be configured independently of the local listen address.

Example shape:

```text
motlie-telnyx-gateway \
  --listen 127.0.0.1:8080 \
  --webhook-url https://example.ngrok.app \
  --media-path /telnyx/media \
  --webhook-path /telnyx/webhooks
```

This matters because the process may listen on `127.0.0.1:8080` while Telnyx needs the externally reachable tunnel URL, not the private bind address.

### Recommended v1 Dev Workflow

1. start the gateway on the private host
2. start ngrok or Tailscale Funnel
3. configure the Telnyx webhook URL to the tunnel URL
4. make or receive calls

## Getting Started: Local Deployment

This section is intentionally operator-oriented. It describes how to go from zero to a first inbound and outbound call without assuming public cloud deployment.

### Telnyx Account Setup

#### 1. Create a Telnyx account and API key

Create a Telnyx account in Mission Control, then create an API key for Voice API calls. All programmable voice requests use:

```text
Authorization: Bearer $TELNYX_API_KEY
```

#### 2. Purchase a phone number

You can do this in the portal or via API.

Example search and purchase flow:

```bash
curl -X GET \
  --header "Authorization: Bearer $TELNYX_API_KEY" \
  "https://api.telnyx.com/v2/available_phone_numbers?filter[country_code]=US&filter[locality]=Chicago"

curl -X POST \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer $TELNYX_API_KEY" \
  --data '{
    "phone_numbers": [
      { "phone_number": "+13125551234" }
    ]
  }' \
  "https://api.telnyx.com/v2/number_orders"
```

#### 3. Create a Call Control Application

For Motlie's Telnyx gateway, prefer a Call Control Application over TeXML because the design depends on programmable voice webhooks plus media streaming commands.

Current Telnyx create call control application request:

```bash
curl -X POST \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer $TELNYX_API_KEY" \
  --data '{
    "application_name": "motlie-telnyx-local",
    "webhook_event_url": "https://example.ngrok.app/telnyx/webhooks",
    "webhook_api_version": "2"
  }' \
  "https://api.telnyx.com/v2/call_control_applications"
```

This returns the application `id`, which is also the `connection_id` used for outbound calls.

#### 4. Configure the webhook URL

If your tunnel URL changes, update the application:

```bash
curl -X PATCH \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer $TELNYX_API_KEY" \
  --data '{
    "webhook_event_url": "https://example.ngrok.app/telnyx/webhooks",
    "webhook_api_version": "2"
  }' \
  "https://api.telnyx.com/v2/call_control_applications/$TELNYX_CONNECTION_ID"
```

#### 5. Assign the phone number to the application

Telnyx phone numbers carry a `connection_id`. Set that to the Call Control Application ID.

```bash
curl -X PATCH \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer $TELNYX_API_KEY" \
  --data "{
    \"connection_id\": \"$TELNYX_CONNECTION_ID\"
  }" \
  "https://api.telnyx.com/v2/phone_numbers/$TELNYX_PHONE_NUMBER_ID"
```

You can verify the current voice settings with:

```bash
curl -X GET \
  --header "Authorization: Bearer $TELNYX_API_KEY" \
  "https://api.telnyx.com/v2/phone_numbers/$TELNYX_PHONE_NUMBER_ID/voice"
```

### Local Host Setup

#### 6. Build the gateway binary

```bash
cargo build --release -p motlie-telnyx-gateway
```

#### 7. Download model artifacts

Download the artifacts for the models you want to run.

Recommended v1 defaults:

- Piper voice model
- `sherpa-onnx` Zipformer2 streaming model

#### 8. Start the gateway with model paths and listen address

```bash
./target/release/motlie-telnyx-gateway \
  --listen 127.0.0.1:8080 \
  --webhook-url https://example.ngrok.app \
  --webhook-path /telnyx/webhooks \
  --media-path /telnyx/media \
  --telnyx-api-key "$TELNYX_API_KEY" \
  --connection-id "$TELNYX_CONNECTION_ID" \
  --phone-number "+13125551234" \
  --asr-model sherpa-onnx \
  --tts-model piper
```

The exact model configuration flags can evolve, but the binary should always separate:

- local listen address
- externally reachable webhook URL
- injected ASR/TTS backend choice

#### 9. Start ngrok or Tailscale Funnel

ngrok example:

```bash
ngrok http 8080
```

Tailscale Funnel example:

```bash
tailscale funnel 8080
```

#### 10. Update the Telnyx webhook URL if using ngrok

If you use a randomly assigned ngrok hostname, repeat step 4 every time ngrok restarts with a different public URL.

### First Call Test

#### 11. Call the Telnyx number from a phone

Place an inbound call to the purchased number assigned to the Call Control Application.

#### 12. Verify webhook received and WebSocket media connected

Confirm in gateway logs that:

- `call.initiated` arrived
- the call was answered
- the Telnyx WebSocket `start` event connected

#### 13. Speak and verify ASR transcript appears in gateway logs

Speak into the call and verify transcript text appears in gateway logs or tracing output.

#### 14. Verify TTS response plays back through the phone

Confirm the response audio plays back to the phone and that the transcript-to-response path is working.

### Outbound Call Test

#### 15. Use the gateway CLI or REST to initiate an outbound call

If calling Telnyx directly for a manual smoke test, the documented outbound API shape is:

```bash
curl --location 'https://api.telnyx.com/v2/calls' \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer $TELNYX_API_KEY" \
  --data "{
    \"connection_id\": \"$TELNYX_CONNECTION_ID\",
    \"to\": \"+1234567890\",
    \"from\": \"+13125551234\"
  }"
```

For the Motlie gateway, the preferred implementation is that the gateway issues this request itself and attaches the media streaming parameters described earlier in this design.

#### 16. Same verification as inbound

Confirm:

- outbound call connects
- WebSocket media starts
- ASR transcripts appear
- `ConversationHandler` generates text
- TTS audio is sent back over the call

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
- typed `EncodedFrame<C>` transport wrappers so codecs are explicit in stage signatures
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
- typed `PcmFrame<R, Ch, E>` wrappers so rate, channels, and encoding are explicit during assembly

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
- typed stage-builder helpers in the model-adjacent layer so common decode -> normalize -> chunk and drain -> normalize -> packetize pipelines are assembled consistently

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
- The exact interaction between simultaneous WebSocket media streaming and specific hosted call-control features such as `gather_using_audio` or hosted prompt playback should be validated on a live test call before those mixed modes are promoted.
- If some carriers deliver audible in-band DTMF energy even when Telnyx emits a separate `dtmf` event, the gateway should log and measure the overlap before finalizing the v1.1 suppression strategy.

## References

- Telnyx media streaming docs: https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming
- Telnyx `streaming_start` API: https://developers.telnyx.com/api-reference/call-commands/streaming-start
- Telnyx `dial` API: https://developers.telnyx.com/api-reference/call-commands/dial
- Telnyx create call control application API: https://developers.telnyx.com/api-reference/call-control-applications/create-a-call-control-application
- Telnyx `gather_using_audio` API: https://developers.telnyx.com/api-reference/call-commands/gather-using-audio
- Telnyx `send_dtmf` API: https://developers.telnyx.com/api-reference/call-commands/send-dtmf
- Telnyx voice webhooks overview: https://developers.telnyx.com/docs/voice/programmable-voice/voice-api-webhooks
- Telnyx `call.initiated` webhook: https://developers.telnyx.com/api-reference/callbacks/call-initiated
- Tailscale Funnel docs: https://tailscale.com/kb/1223/tailscale-funnel/
- ngrok localhost sharing docs: https://ngrok.com/docs/guides/share-localhost/overview
- Existing ASR design: [DESIGN_ASR.md](/Users/dchung/sessions/codex-macmini-telnyx/motlie/libs/models/docs/DESIGN_ASR.md)
- Existing TTS design: [DESIGN_TTS.md](/Users/dchung/sessions/codex-macmini-telnyx/motlie/libs/models/docs/DESIGN_TTS.md)
- Existing API contracts: [transcription.rs](/Users/dchung/sessions/codex-macmini-telnyx/motlie/libs/model/src/transcription.rs), [speech.rs](/Users/dchung/sessions/codex-macmini-telnyx/motlie/libs/model/src/speech.rs)

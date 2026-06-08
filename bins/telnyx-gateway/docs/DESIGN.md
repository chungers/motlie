# Telnyx Real-Time Voice Integration Design

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-06-07 23:08 PDT | @codex-366-impl: Added M5 support for conversational text-agent handoff: speech enqueue can opt into cancel-and-replace instead of treating an active TTS slot as a protocol failure, and M4 text-call playback terminal frames should expose completed/canceled/failed/superseded status. | Milestone 5: Conversational Realism Latency Improvements, Milestone 4: External Integration Harness, Returning TTS Audio |
| 2026-06-07 18:16 PDT | @codex-366-impl: Expanded M5 #402 conversational realism scope into PR #417: local ASR endpoint finalization, partial/final/speech-onset barge-in behind the shared `conversation barge-in` toggle, and chunked TTS enqueue after each synthesized text chunk while preserving backend audio-chunk continuity inside that text chunk. | Milestone 5: Conversational Realism Latency Improvements, Milestone 3: Full-Duplex TUI Chat Conversation, Returning TTS Audio |
| 2026-06-07 | @codex-366-impl: Fixed live ASR end-of-turn latency by replacing indefinite low-energy tail suppression with replay-sized local endpoint finalization: after the trailing silence pad, the gateway finishes the active ASR session and waits for new speech, so final transcripts do not depend on a later utterance. | Inbound Call Handler Design, Testing Scope |
| 2026-06-07 | @codex-366-impl: Added a gateway-wide `conversation barge-in on|off|status` toggle for TUI/socket live tests; default remains on, while off prevents transcript-triggered TTS clear during smoke-test echo validation. | Milestone 3: Full-Duplex TUI Chat Conversation, Operator REPL and TUI Control Surface |
| 2026-06-06 15:00 PDT | @codex-366-impl: Folded partial-ASR-triggered barge-in into M3: unsuppressed meaningful partial transcripts for attached calls now cancel active playback through the existing clear/cancel path, while final transcripts still drive regeneration; frame-level speech-onset barge-in was later folded into M5 #402 / PR #417. | Milestone 3: Full-Duplex TUI Chat Conversation, Operator REPL and TUI Control Surface |
| 2026-06-06 13:40 PDT | @codex-366-impl: Made the M3 echo conversation handler explicitly test-only and wired outbound `dial` into the same auto-attach conversation path: default attached conversations are transcription-only, `--conversation-smoke-test` or `conversation smoke-test on` enables the echo reply loop, and status surfaces the active handler mode. | Milestone 3: Full-Duplex TUI Chat Conversation, Operator REPL and TUI Control Surface |
| 2026-06-06 13:12 PDT | @codex-366-impl: Updated M3 operator UX after manual acceptance feedback: inbound answer and plain conversation attach now default to auto/approved mode, `conversation disapprove` cancels active TTS and returns the call to transcription-only, and status prints listener addresses without `Option` debug formatting. | Milestone 3: Full-Duplex TUI Chat Conversation, Operator REPL and TUI Control Surface |
| 2026-06-06 08:48 PDT | @codex-366-impl: Addressed PR #400 review round 1 by documenting final-transcript-triggered barge-in latency, adding manual approval commands, and clarifying that auto conversation is Piper-locked for the first Sherpa+Piper pairing while manual approval uses source-selected TTS. | Milestone 3: Full-Duplex TUI Chat Conversation, Operator REPL and TUI Control Surface |
| 2026-06-05 23:34 PDT | @codex-366-impl: Implemented milestone 3 composition using gateway-local `ConversationRuntime`, final-transcript forwarding for attached calls, shared TUI/socket `conversation` commands, M2 speech queue reuse for `Say`, and drop-and-regenerate barge-in via the existing `speak cancel`/Telnyx `clear` path. | Milestone 3: Full-Duplex TUI Chat Conversation, Recommended Telnyx v1 Pipelines, Operator REPL and TUI Control Surface |
| 2026-06-04 | @codex-369-rv: Removed ASR startup-default configuration in favor of code defaults plus source-local `asr use`, and normalized gateway TTS output to backend-neutral signed 16-bit PCM plus sample-rate metadata before Telnyx packetization so Piper and future Qwen3-TTS.cpp share one media path. | Operator REPL and TUI Control Surface, Returning TTS Audio, Recommended TTS: Piper |
| 2026-06-04 | @codex-369-rv: Aligned TTS operator commands with the ASR command pattern: `tts list`, `tts status`, and `tts use piper` are available through the shared TUI/socket command engine, and `dial` output now tells operators to wait for media before running `speak`. | Operator REPL and TUI Control Surface, Driver REPL Dialer Surface, Recommended TTS: Piper |
| 2026-06-03 | @codex-369-rv: Recorded M2 live-test findings for Piper: eSpeak-ng phonemization data must be present or auto-discovered, outbound TTS is packetized from continuous utterance audio, frames are paced by the media task, silence keepalive is not injected during active speech, and frame interval/underrun telemetry is required for live diagnosis. | Milestone 2: Outbound Dialer and TTS, Returning TTS Audio, M2-Safe Bidirectional Media Contract, Recommended TTS: Piper, Testing Scope |
| 2026-06-03 | @codex-369-rv: Locked the milestone 2 outbound TTS shape around one bidirectional RTP media WebSocket, an outbound frame queue owned by the socket task, cancellable `speak`, live inbound ASR during playback, `clear`/`mark`, `stream_track=inbound_track`, and TUI/socket command parity so milestone 3 can add barge-in policy without replacing transport. | Staged Build Strategy, Outbound Call Handler Design, Returning TTS Audio, Driver REPL Dialer Surface, Testing Scope |
| 2026-06-03 | @codex-369-rv: Made the intentional live ASR default explicit: the gateway defaults to `kroko-2025` because it is the balanced choice across call-center and PM/technical corpora, while `sherpa-2023` remains recommended for call-center-only deployments. Operators can switch per source between calls with `asr use`. | Recommended ASR/TTS Stack, Milestone 1.5: Sherpa ASR Quality Tuning |
| 2026-06-01 | @codex-371-impl: Implemented the #371 R1 prerequisite by moving Sherpa repeated-token transcript suppression and session-reset decisions behind the ASR adapter; recorded the updated models-first M1.5 sequence where A/B infrastructure and model comparison precede hotwords, endpointing, decoder tuning, and normalization. | Inbound Call Handler Design, Milestone 1.5: Sherpa ASR Quality Tuning, Recommended ASR/TTS Stack |
| 2026-06-01 | @codex-364-impl: Fixed media capture WAV finalization so standard readers see finite durations, added `replay-capture` for deterministic captured-audio WER checks, and split Sherpa-only ASR quality tuning into milestone 1.5 (#370) covering hotwords, model/decoder A/B, and optional marked normalization. | Staged Build Strategy, Inbound Call Handler Design, Recommended ASR/TTS Stack, Testing Scope, Getting Started: Local Deployment |
| 2026-06-01 | @codex-364-impl: Recorded the first outbound ASR-only dial attempt blocker: Telnyx rejected `POST /v2/calls` with `403 D38` because the Call Control application has no Outbound Voice Profile assignment; the gateway command is ready, but live outbound validation requires Telnyx account setup. | Outbound Call Handler Design, Testing Scope, Getting Started: Local Deployment |
| 2026-06-01 | @codex-364-impl: Added an ASR-only outbound live-test command, `test dial-transcribe <to> [--from <from>]`, that reuses the Telnyx dial API, bidirectional RTP media attach, silence keepalive, capture, and Sherpa transcription paths without implementing milestone 2 outbound TTS. | Outbound Call Handler Design, Inbound Call Handler Design, Operator REPL and TUI Control Surface, Testing Scope |
| 2026-06-01 | @codex-364-impl: Recorded post-fix live validation for two `L16 16 kHz` inbound calls: transcription is now usable enough to measure, with reference-section WER around `16-18%`, but remaining errors concentrate on proper nouns and telephony homophones; also tightened bare `answer` behavior so it selects the single waiting inbound call instead of a stale ended selection. | Inbound Call Handler Design, Operator REPL and TUI Control Surface, Testing Scope |
| 2026-06-01 | @codex-364-impl: Updated the milestone 1 `L16` comparison path after capture analysis showed Telnyx WebSocket `L16` payloads must be decoded as little-endian PCM for the tested account/carrier path; big-endian interpretation produced clipped, unusable ASR input. | Audio Codecs and Formats, Inbound Call Handler Design, Testing Scope |
| 2026-06-01 | @codex-364-impl: Added the post-live-test ASR quality loop for milestone 1: keep `PCMU 8 kHz` as the safe default, make the Telnyx answer media request configurable for `L16 16 kHz` comparison runs, and capture raw Telnyx JSONL, decoded WAV, Sherpa input WAV, transcript JSONL, and manifests for offline replay/tuning. | Audio Codecs and Formats, Inbound Call Handler Design, Testing Scope, Getting Started: Local Deployment |
| 2026-06-01 | @codex-364-impl: Updated the milestone 1 ASR design to prefer upstream `sherpa-onnx` over Motlie's former hand-rolled ONNX decoder; the gateway now preserves normal speech pauses for Sherpa endpointing and uses the upstream crate's static native archive download/link path instead of the workspace Pyke `ort` path. | Recommended ASR/TTS Stack, Inbound Call Handler Design, Getting Started: Local Deployment |
| 2026-06-01 | @codex-364-impl: Added a milestone 1 local command-socket subset for agent-assisted live testing: headless startup can load known config, accept line-oriented Motlie driver commands, return JSON command responses, answer inbound calls, expose `call show` transcript snapshots, and shut down without scraping the TUI; richer event polling and mux validation remain milestone 4. | Staged Build Strategy, Operator REPL and TUI Control Surface, Application Webhooks and Gateway Control API, Getting Started: Local Deployment |
| 2026-06-01 | @codex-364-impl: Updated milestone 1 live ASR handling after long-call logs showed continuous Telnyx media delivery but Sherpa getting stuck in repeated-token output; the gateway now treats speech gaps and repeated-token detection as ASR-session reset boundaries while keeping one assembled call transcript for the operator. | Inbound Call Handler Design, Operator REPL and TUI Control Surface, Testing Scope |
| 2026-06-01 | @codex-364-impl: Updated milestone 1 live-call behavior from receive-only PCMU to bidirectional PCMU RTP with outbound silence keepalive after Telnyx reported provider-normal caller hangups while the gateway was only listening; added assembled transcript display so raw partial/final fragments do not replace readable call-level transcript text. | Audio Codecs and Formats, Inbound Call Handler Design, Operator REPL and TUI Control Surface |
| 2026-06-01 | @codex-364-impl: Added live-call termination diagnostics to milestone 1 so Telnyx `call.hangup`/`call.ended`/stream termination fields are preserved in structured logs and selected-call detail instead of reducing every provider-side close to `ended`. | Inbound Call Handler Design, Operator REPL and TUI Control Surface, Testing Scope |
| 2026-06-01 | @codex-364-impl: Refined live milestone 1 handling so the ASR path gates both initial silence and sustained low-energy tails, suppresses repeated-token Sherpa hallucinations from the operator transcript stream while logging them as suppressed events, and keeps the shell prompt cursor aligned by rendered rows when command output wraps. | Inbound Call Handler Design, Operator REPL and TUI Control Surface, Testing Scope |
| 2026-06-01 | @codex-364-impl: Merged the TUI command input and REPL history into one shell-style left pane so operator commands, output, and the active prompt share one terminal-like surface. | Operator REPL and TUI Control Surface |
| 2026-06-01 | @codex-364-impl: Added an explicit ASR start-of-speech gate for milestone 1 so low-energy initial telephony frames are logged but not fed into Sherpa, preventing silence-driven partial transcript growth during live inbound tests. | Inbound Call Handler Design, Testing Scope |
| 2026-06-01 | @codex-364-impl: Moved TUI-mode tracing away from the terminal by default; `--tui` writes structured logs to `telnyx-gateway.log` unless `--log-file` overrides the path, so logs cannot corrupt the alternate-screen TUI. | Operator REPL and TUI Control Surface, Getting Started: Local Deployment |
| 2026-06-01 | @codex-364-impl: Recorded the now-superseded receive-only `PCMU` live-test attempt after the first live `L16` ASR output was unusable; later live evidence required bidirectional `PCMU` RTP silence keepalive for milestone 1. | Audio Codecs and Formats, Inbound Call Handler Design |
| 2026-05-31 | @codex-364-impl: Clarified that the Unix-domain socket is the local agent-tooling interface: it can provide command execution, call snapshots, cursor-based event polling, and optional tmux/mstream-style wake-up notifications without depending on appserver webhooks or the HTTP Control API. | Operator REPL and TUI Control Surface, Application Webhooks and Gateway Control API |
| 2026-05-31 | @codex-364-impl: Updated the gateway ORT policy to use Cargo's `ort/download-binaries` static `libonnxruntime.a` path with no `ORT_LIB_PATH`, dynamic-link, vendored ORT, or source-build runbook. | Recommended ASR/TTS Stack, Getting Started: Local Deployment |
| 2026-05-31 | @codex-364-impl: Clarified that static ONNX Runtime builds should build ORT's own C/C++ dependencies from source through FetchContent; operators only provision host build tools, not ORT internal libraries. | Recommended ASR/TTS Stack |
| 2026-05-31 | @codex-364-impl: Made static ONNX Runtime linkage the required Sherpa/Piper operational policy for gateway live tests and deployments; dynamic `ORT_PREFER_DYNAMIC_LINK` runbooks are explicitly rejected. | Recommended ASR/TTS Stack, Getting Started: Local Deployment |
| 2026-05-30 | @codex-358-research: Addressed PR #363 review round 1 by moving M4, the design-quality assessment, and the recommended v1 pipelines under the staged build strategy; marking Control API and application webhook flows as milestone 4 work; aligning M1 structured logging with #364; adding operator session/socket/mux modules; and matching M3 prose to `ConversationCommand::{Say, Call, Noop}`. | Staged Build Strategy, Application Webhooks and Gateway Control API, Operator REPL and TUI Control Surface, Inbound Call Handler Design, Crate Hierarchy and API Surfaces |
| 2026-05-30 | @codex-358-research: Reviewed the overall design quality and split tracking into four milestone issues: M1 inbound TUI transcription (#364), M2 outbound TUI dialer/TTS (#365), M3 full-duplex TUI chat conversation (#366), and M4 external socket/webhook/appserver integration (#367). | Staged Build Strategy, Recommended Telnyx v1 Pipelines, Operator REPL and TUI Control Surface |
| 2026-05-30 | @codex-358-research: Added startup-selected operator input modes: `--tui` enables the local TUI, `--socket <path>` enables a Unix-domain command socket for headless control, and both sources are multiplexed through one command dispatcher when enabled together. | Operator REPL and TUI Control Surface, Staged Build Strategy, Gateway Configuration Requirement |
| 2026-05-30 | @codex-358-research: Added the app-server call read/attach API: `GET /api/v1/calls` and `GET /api/v1/calls/{call_id}` expose current gateway call state, while per-call attachments bind an app webhook subscription to call transcripts/events without replacing webhook delivery. | Application Webhooks and Gateway Control API |
| 2026-05-30 | @codex-358-research: Refined the TUI layout so the right side is split into a top call roster and bottom selected-call detail pane; inbound calls appear as highlighted pending rows, can become the selected call, and drive detail/transcript/action hints without stealing REPL input. | Operator REPL and TUI Control Surface |
| 2026-05-30 | @codex-358-research: Switched local deployment examples to assume Tailscale Funnel as the default public URL path, keeping ngrok only as an alternate tunnel option. | Deployment: Private Host, Getting Started: Local Deployment, Replayable State Dumps |
| 2026-05-30 | @codex-358-research: Added replayable gateway state dumps: `shutdown [dump_path]` writes durable Telnyx/app-server configuration as idempotent REPL commands, and startup `--load <dump_path>` replays that file to rehydrate configuration before accepting work. | Operator REPL and TUI Control Surface, Gateway Configuration Requirement, Getting Started: Local Deployment |
| 2026-05-30 | @codex-358-research: Clarified that "gateway emits" means an outbound HTTP `POST` from the gateway to a registered application webhook URL; event delivery is acknowledgement-based, asynchronous, retried on failure, and separate from the application server calling the Gateway Control API back into the gateway. | Application Webhooks and Gateway Control API |
| 2026-05-30 | @codex-358-research: Added an external automation surface: gateway-emitted application webhooks plus an authenticated Gateway Control API so another server can receive inbound-call events, answer calls programmatically, receive transcript events, and use the gateway as an outbound dialer/TTS service. | Application Webhooks and Gateway Control API, Staged Build Strategy, Inbound Call Handler Design, Outbound Call Handler Design |
| 2026-05-30 | @codex-358-research: Reframed `bins/telnyx-gateway` as an operator-driven TUI/REPL application that starts an idle HTTP/WebSocket listener, uses `motlie-driver` commands for Telnyx provisioning and call control, and only answers inbound calls after explicit operator or mode selection. | Operator REPL and TUI Control Surface, Staged Build Strategy, Inbound Call Handler Design, Outbound Call Handler Design |
| 2026-05-30 | @codex-358-research: Added staged implementation milestones and composable contracts for inbound transcription sinks, outbound dial/say control through `motlie-driver`, reusable ASR/TTS pipelines, opaque provider-neutral call handles, and the later conversation-handler bridge. | Staged Build Strategy, Provider Adapter Boundary, Conversation Handler Contract, Inbound Call Handler Design, Outbound Call Handler Design |
| 2026-05-30 | @codex-358-research: Clarified that `motlie_voice::app` owns application-level voice control traits while `motlie_voice::telephony` owns provider-neutral telephony vocabulary and events, not provider-specific transports or call-control clients. | Crate Hierarchy and API Surfaces, Provider-Neutral API Rule, v1.1: DTMF and Call Control |
| 2026-05-30 | @codex-358-research: Moved the Telnyx design docs under `bins/telnyx-gateway/docs`, collapsed the previously separate Telnyx adapter crate into Telnyx-specific modules owned by `bins/telnyx-gateway`, and standardized the gateway path on `bins/telnyx-gateway`. | Overview, Crate Hierarchy and API Surfaces, Media Adaptation Pipeline, Inbound Call Handler Design, Deployment, Getting Started, References |
| 2026-05-30 | @codex-358-research: Replaced stale absolute local references with repo-relative links to the current ASR/TTS design docs and model contract source files. | References |
| 2026-05-30 | @codex-358-research: Addressed PR review by making the reorder boundary Telnyx-adapter mapping plus provider-neutral sequenced-frame reorder, correcting Sherpa's current typed input contract to ordered `AudioBuf<i16, 16_000, Mono>`, demoting Fish Speech to a historical rejected candidate, and documenting the missing `i16_to_f32` helper needed by the i16 telephony resampler wrapper. | Media Adaptation Pipeline, Recommended ASR/TTS Stack, Inbound Call Handler Design, Gap Analysis, Open Concerns |
| 2026-05-30 | @codex-358-research: Rebased the design around the landed `motlie-voice` crate from PR #209, so Telnyx extends existing `motlie_model::typed` and `motlie_voice` media primitives instead of rebuilding them. Added anti-aliased resampling, buffered-vs-transport streaming, and Piper CUDA caveats. | Overview, Goals and Non-Goals, Crate Hierarchy and API Surfaces, Media Adaptation Pipeline, Recommended ASR/TTS Stack, Gap Analysis, Testing Scope for PLAN |
| 2026-04-17 | @codex-macmini-telnyx: Removed the remaining `Box<dyn>` streaming-response sketch from `ConversationHandler` and replaced it with an associated `TextStream` type so the design remains static-dispatch at the gateway surface. | Closed-Enum Selection, Conversation Handler Contract, Streaming Conversation Responses |
| 2026-04-17 | @codex-macmini-telnyx: Tightened the execution policy so the first live Telnyx implementation and validation slice is explicitly `Sherpa + Piper`. Other pairings remain documented as follow-on combinations, not peer initial targets. | Overview, Goals and Non-Goals, Concrete Combination Requirements |
| 2026-04-16 | @codex-macmini-telnyx: Split the proposed implementation into provider-agnostic `libs/voice` and Telnyx-specific `bins/telnyx-gateway`, and documented the crate hierarchy plus API surfaces explicitly in both DESIGN and PLAN. | Overview, Recommended Integration Shape, Crate Hierarchy and API Surfaces, Deployment: Private Host, Getting Started: Local Deployment |
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
- [Staged Build Strategy](#staged-build-strategy)
- [Operator REPL and TUI Control Surface](#operator-repl-and-tui-control-surface)
- [Application Webhooks and Gateway Control API](#application-webhooks-and-gateway-control-api)
- [Crate Hierarchy and API Surfaces](#crate-hierarchy-and-api-surfaces)
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

- typed TTS and ASR contracts in `motlie_model::typed`, including `AudioBuf`, `StreamingTranscriber`, `TranscriptionSession`, `SpeechSynthesizer`, and `SpeechStream`
- a working Piper backend and streaming ASR backends such as `sherpa-onnx` and Moonshine
- a landed provider-neutral `motlie-voice` crate from PR #209 with `PcmFrame<const RATE_HZ, C, E>`, WAV stream helpers, sample conversion helpers, downmixing, and a placeholder `Resampler`

It does not yet have a telephony transport layer that can:

- receive live call audio from Telnyx
- normalize Telnyx media frames into the typed ASR input expected by `StreamingTranscriber`
- route transcript updates into application logic through a pluggable conversation stage
- synthesize TTS responses through `SpeechSynthesizer`
- send outbound call audio back over the same Telnyx media channel

### Solution

Extend the existing provider-agnostic `motlie-voice` layer and add a Telnyx-facing adapter above `libs/model` and `libs/models`.

Recommended deployment shape:

1. `libs/model`
   Remains the stable typed ASR and TTS contract layer. Telnyx uses `motlie_model::typed` instead of defining a gateway-specific ASR/TTS contract.
2. `libs/models`
   Remains the curated bundle layer for Piper, Qwen3-TTS, `whisper.cpp`, `sherpa-onnx`, and Moonshine.
3. existing provider-agnostic Rust voice crate: `libs/voice` (`motlie-voice`)
   Already owns `PcmFrame`, WAV stream helpers, conversion helpers, downmixing, and a placeholder resampler. The Telnyx slice extends it with provider-neutral codecs, packetization, stage composition, conversation and DTMF handler traits, and runtime orchestration.
4. Telnyx-specific gateway crate and deployable binary: `bins/telnyx-gateway`
   Owns Telnyx webhook handling, WebSocket message schema, call-control client, mapping between Telnyx transport messages and `libs/voice` types, HTTP/WebSocket serving, configuration, and process wiring.

This keeps telephony transport concerns out of the model contracts while also preventing Telnyx-specific protocol types from leaking into the reusable media pipeline.

Execution policy for v1:

- milestone 1 is inbound calls answered and managed in the TUI, with Sherpa ASR transcription in the selected-call detail pane
- milestone 2 is outbound dialing plus Piper TTS driven by the TUI command surface and selected-call detail text input
- milestone 3 is full-duplex conversation driven by a TUI chat interface over the selected call
- milestone 4 is external integration: Unix-domain socket mux, application webhooks, Gateway Control API call attachment/read flows, and a harness appserver
- `Sherpa + Piper` remains the first complete duplex backend pairing once milestone 3 exists
- `Moonshine` and `Qwen3-TTS` remain supported design targets for follow-on slices, not peer initial implementation targets
- provider-neutral abstractions in `libs/voice` must be broad enough to support those later pairings without redesign, but milestone 1 and milestone 2 acceptance must not depend on them

## Goals and Non-Goals

### Goals

- Integrate Telnyx inbound and outbound calls with Motlie ASR/TTS
- Make the conversation stage explicitly pluggable for application-specific logic
- Reuse `motlie_model::typed::StreamingTranscriber` / `TranscriptionSession` for inbound speech
- Reuse `motlie_model::typed::SpeechSynthesizer` / `SpeechStream` for outbound speech
- Reuse existing `motlie_voice::frame::PcmFrame` and `motlie_voice::pipeline::{convert,resample}` rather than defining a parallel voice/media layer
- Allow any compatible typed ASR and TTS implementation to be injected without changing gateway code
- Minimize telephony latency by preferring Telnyx bidirectional RTP streaming over queued MP3 playback
- Keep codec and transport adaptation outside `libs/model`
- Support interruption and barge-in at the gateway layer
- Keep room for multiple ASR/TTS backend combinations behind the same gateway
- Support private-host deployment behind Tailscale Funnel for v1, with ngrok as an alternate tunnel option
- Make Sherpa inbound ASR the only required live milestone 1 backend, Piper the only required live milestone 2 TTS backend, and `Sherpa + Piper` the only required complete duplex backend pairing for milestone 3
- Defer external socket/webhook/appserver integration validation to milestone 4 so the first three TUI product slices stay narrow

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

- the most direct fit to the later duplex ASR/TTS contracts is `L16` bidirectional RTP at `16 kHz`
- milestone 1 live inbound transcription defaults to `PCMU` at `8 kHz` for inbound media and enables bidirectional RTP in the same codec so the gateway can send outbound silence keepalive while it is otherwise only transcribing
- after the first successful upstream Sherpa live test, the gateway should expose an operator-controlled `L16 16 kHz` comparison mode; `PCMU 8 kHz` remains the rollback/default path until captured evidence shows `L16` is both stable and higher quality on the active Telnyx account/carrier path
- milestone 1 live capture showed Telnyx WebSocket `L16` payloads decoding correctly as little-endian PCM on the tested account/carrier path; provider-neutral codec helpers should support both byte orders, but the Telnyx gateway must use the observed Telnyx byte order for inbound `L16`
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
Telnyx webhook -> telnyx-gateway HTTP handler
               -> telnyx_gateway::call_control
               -> Telnyx REST answer/dial/streaming_start

Telnyx media WebSocket -> telnyx_gateway::media
                       -> libs/voice pipeline stages
                       -> StreamingTranscriber session
                       -> ConversationHandler
                       -> SpeechSynthesizer / SpeechStream
                       -> libs/voice pipeline stages
                       -> telnyx_gateway::media
                       -> Telnyx media WebSocket
```

Primary gateway subsystems:

1. `webhook`
   Validates webhook signatures, routes call events, records pending inbound calls, and starts or stops sessions. It must not answer inbound calls unless the operator has enabled an inbound mode or issued `answer`.
2. `call_control_client`
   Thin REST client for `answer`, `dial`, `hangup`, `streaming_start`, and optional metadata updates.
3. `media_server`
   HTTP/WebSocket server that listens on the configured address, defaulting to `127.0.0.1:8080` for local development. The bind address is runtime configuration; the public webhook/media URL should usually be a Tailscale Funnel URL for v1, with other tunnels or deployed URLs as alternatives.
4. `session_registry`
   Maps `call_control_id` and `stream_id` to live session state.
5. `control_api`
   Authenticated HTTP API for external application servers to list calls, answer/reject/hang up, dial outbound calls, and request TTS playback.
6. `event_dispatcher`
   Delivers gateway application webhooks for call, media, transcript, TTS, and conversation events without blocking Telnyx webhook handling.
7. `voice transport adapter`
   Translates Telnyx protocol messages into provider-neutral typed frames and call actions.
8. `codec and media pipeline`
   Decode, reorder, resample, convert, chunk, and packetize provider-neutral media frames.
9. `conversation_orchestrator`
   Owns ASR stream, `ConversationHandler`, TTS stream, barge-in rules, and shutdown.
10. `conversation_handler`
   The application-specific async text-processing stage where LLM calls, business logic, RAG, policy checks, or workflow actions run.
11. `operator_control`
   Owns the `motlie-driver` command set, startup-selected command sources, command-source mux, optional TUI, optional Unix-domain command socket, command status, and operator-visible event/transcript log.

## Staged Build Strategy

Build the gateway in four independently useful milestones. Each milestone should expose a narrow control surface that can later compose with the others. Tracking issues:

- M1 inbound TUI transcription: #364
- M1.5 Sherpa ASR quality tuning: #371
- M2 outbound TUI dialer/TTS: #365
- M3 full-duplex TUI chat conversation: #366
- M4 external integration socket/webhook/appserver harness: #367

The gateway process itself starts before any milestone-specific call behavior is enabled:

```text
telnyx-gateway
-> bind configured HTTP/WebSocket listener, default 127.0.0.1:8080
-> enable configured operator inputs, such as --tui and/or --socket <path>
-> command-source mux waits for motlie-driver commands
```

Startup state is intentionally idle. The process may accept health checks, webhook delivery, and media WebSocket connections at the configured address, but it does not register Telnyx applications, bind numbers, answer inbound calls, or dial outbound calls until the operator drives those actions from a configured command source such as the TUI, Unix-domain socket, replay file, or authenticated Control API.

### Milestone 1: Inbound TUI Transcription

The first runnable milestone is inbound call handling plus ASR in the operator TUI only:

```text
operator provisions Telnyx application and phone number
-> inbound enable --manual
-> Telnyx inbound call
-> pending call appears in the TUI call roster
-> operator selects the call
-> answer [call]
-> answer with media stream
-> Telnyx media WebSocket
-> inbound ASR pipeline
-> TuiTranscriptSink
-> selected-call detail pane transcript
```

No TTS, outbound audio, external appserver, or `ConversationHandler` is required for this milestone. The gateway should listen, surface pending calls in the roster, answer on operator command, stream, normalize, transcribe, and render transcript events in the selected-call detail pane. The TUI remains the primary M1 operator UI, but a minimal local command socket may drive the same command set for agent-assisted live tests where an agent starts the gateway, asks the human to dial, answers the pending call, and polls transcript state without scraping terminal UI output.

Milestone 1 ASR input should suppress low-energy initial media until speech is detected, pass a replay-sized trailing-silence pad after speech, then finish the active ASR session locally so live final transcripts do not wait for a later utterance. Because the current Sherpa backend has known repeated-token failure modes, the gateway should suppress pathological repeated-token transcript text from the operator transcript stream, emit a structured `transcript.suppressed_repeated_token` log with the same call, stream, codec, and sample-rate metadata, and reset the ASR session before feeding subsequent speech. Short pauses below the local endpoint threshold stay in the same ASR session; sustained silence finalizes the previous session and resumed speech opens a fresh one.

Current Sherpa `final` events are word-level rather than sentence-level. The selected-call detail should therefore keep an assembled call transcript that appends accepted final fragments and overlays the current partial, while the raw recent partial/final event list remains available for debugging.

Milestone 1 structured logs must include the gateway call id, Telnyx diagnostic ids such as `call_control_id`, `call_session_id`, and `call_leg_id` when present, `stream_id`, observed codec, observed sample rate, transcript partial/final events, and Telnyx termination details such as hangup cause/source or SIP cause when the provider sends them.

### Milestone 1.5: Sherpa ASR Quality Tuning

Milestone 1.5 (#371) is a follow-on quality milestone for the already-runnable M1 ASR path. It must stay inside the Sherpa ecosystem: improve the current Sherpa-based transcript quality before considering non-Sherpa ASR replacements.

The latest prepared outbound `L16 16 kHz` capture measured `29.2%` WER against a `65`-word reference. The failure pattern is mostly phonetic and domain-specific rather than random media corruption: `outbound` became `ALBAN`/`ALBOW`, `Telnyx` became `TAL NICHS`, `Sherpa` became `SHARPA`, `Motlie` became `MOTLEY`, `voice` became `BOYS`, and `id` became `IDEED`. That suggests the recognizer hears the approximate phonemes but lacks the desired domain/context prior.

The validated Qwen3-TTS golden A/B after the #378 trailing-silence pad fix changed the live default decision from a call-center-only pick to a balanced profile pick:

- `sherpa-2023`: `10.5% / 12.0%` call-center WER, `19.8% / 22.1%` PM/technical WER (`L16-16k / PCMU-8k`)
- `kroko-2025`: `14.1% / 13.9%` call-center WER, `14.0% / 14.0%` PM/technical WER (`L16-16k / PCMU-8k`)

Therefore the live gateway default is intentionally `kroko-2025`, while `sherpa-2023` is a recommended operator-selected backend for call-center-only runs. This is not a hidden default flip: the CLI default, `asr status`, and `asr use` operator commands must make the selected backend observable and switchable between calls.

M1.5 should proceed in this order:

1. Keep backend-specific transcript suppression behind the ASR adapter before model or tuning work. The shared media loop records adapter-supplied suppression/reset decisions, while Echo and other non-Sherpa backends pass transcript events through by default.
2. Build model A/B infrastructure before tuning sidecars: define a golden-WAV corpus manifest with reference text, codec/rate/direction metadata, the known `29.2%` L16 `16 kHz` capture, and a PCMU `8 kHz` capture.
3. Extend `replay-capture` into a backend-selectable A/B harness that runs captured WAVs through swappable `StreamingTranscriber` backends and reports WER, substitutions, deletions, insertions, and latency in a comparable form.
4. Integrate and A/B candidate models behind the same backbone, starting with newer Sherpa Zipformer artifacts and then Nemotron/Parakeet candidates coordinated through #191/#369.
5. Defer Sherpa hotwords/context bias, endpointing tuning, decoder tuning, and post-ASR normalization until the best-model WER and latency baseline are measured on the golden corpus. Raw ASR events and WER reports must remain available; agent/LLM normalization is not raw ASR improvement.

An agent-operated gateway can use its LLM context for transcript normalization and intent recovery after ASR. It does not replace decoder-time context bias. Decoder-time hotwords improve the actual ASR token choice before final transcript emission; LLM normalization can clean up displayed/service text after the fact but may hallucinate, hide model regressions, and should not be used for WER acceptance unless explicitly scored as a separate normalized-output metric.

Recommended inbound surface:

```rust
use async_trait::async_trait;
use motlie_model::typed::TranscriptionUpdate;
use motlie_voice::telephony::CallAction;

pub struct CallIds {
    pub provider_call_id: String,
    pub provider_session_id: Option<String>,
    pub media_stream_id: Option<String>,
}

pub struct CallContext {
    pub ids: CallIds,
    pub custom_state: std::collections::BTreeMap<String, String>,
}

pub enum TranscriptEvent {
    Partial {
        text: String,
        update: TranscriptionUpdate,
    },
    Final {
        text: String,
        update: TranscriptionUpdate,
    },
}

pub struct VoiceAppError {
    pub message: String,
}

#[async_trait]
pub trait TranscriptSink: Send + Sync {
    async fn on_transcript(
        &self,
        event: TranscriptEvent,
        context: &mut CallContext,
    ) -> Result<Vec<CallAction>, VoiceAppError>;
}
```

The first sink can be `StdoutTranscriptSink`, which logs partials/finals. A later `TmuxTranscriptSink` can map final transcript text to `motlie_tmux::KeySequence` and call `Target::send_keys()`. This keeps the ASR call path the same while swapping what consumes transcripts.

For milestone 1, `TuiTranscriptSink` should append partial/final transcript events to the selected-call detail pane. `StdoutTranscriptSink` remains useful for non-TUI tests and log-only runs. `WebhookTranscriptSink` belongs to milestone 4 external integration, where the same transcript events are delivered to subscribed application servers.

For Telnyx, `provider_call_id` maps to `call_control_id`, `provider_session_id` maps to `call_session_id`, and `media_stream_id` maps to Telnyx `stream_id`. Those raw Telnyx field names should stay in `bins/telnyx-gateway`.

### Milestone 2: Outbound Dialer and TTS

The second milestone is outbound call control plus TTS from the shared operator command engine. The TUI shell, selected-call detail composer, startup `--load` replay, and Unix-domain socket all dispatch the same typed commands with source-local selected call and model state:

```text
motlie-driver command source (TUI or socket)
-> dial <number>
-> Telnyx outbound call with media stream
-> selected-call detail pane shows outbound state
-> operator or agent enters speak text through shell/socket/detail composer
-> outbound TTS pipeline enqueues outbound media frames
-> existing Telnyx bidirectional media WebSocket writes those frames
```

This should use the existing `motlie-driver` pattern: define a Telnyx command family implementing `CommandSet<TelnyxDialerState>`, then run it through the operator command dispatcher. The minimum command surface is:

```text
dial <phone-or-sip-uri> [--from <+e164>]
speak [call-id] <text...>
speak cancel [call-id]
hangup [call-id]
status [call-id]
```

The command names above are the M2 operator-facing surface. If the implementation retains an internal `say()` method for compatibility with `ConversationCommand::Say`, `speak` must still dispatch to that exact same controller path rather than introducing a second TTS route.

Recommended controller surface:

```rust
use async_trait::async_trait;

pub struct VoiceDialRequest {
    pub target: VoiceAddress,
    pub caller_id: Option<VoiceAddress>,
    pub metadata: std::collections::BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct CallHandle {
    pub provider_call_id: String,
}

#[derive(Clone, Debug)]
pub enum VoiceAddress {
    PhoneE164(String),
    SipUri(String),
}

pub struct SpeechPlaybackHandle {
    pub sequence: u64,
}

pub struct VoiceControlError {
    pub message: String,
}

#[async_trait]
pub trait OutboundSpeechController: Send {
    async fn dial(&mut self, request: VoiceDialRequest) -> Result<CallHandle, VoiceControlError>;
    async fn speak(
        &mut self,
        call: &CallHandle,
        text: String,
    ) -> Result<SpeechPlaybackHandle, VoiceControlError>;
    async fn cancel_speech(&mut self, call: &CallHandle) -> Result<(), VoiceControlError>;
    async fn hangup(&mut self, call: &CallHandle) -> Result<(), VoiceControlError>;
}
```

The TUI REPL, selected-call detail text composer, and Unix-domain socket are milestone 2 utterance sources. Any later source that can produce text, including mstream send-keys, a broadcast bridge, fixture replay, or the milestone 4 Control API, should be able to call the same `speak()` surface:

```rust
#[async_trait]
pub trait OutboundUtteranceSource: Send {
    async fn next_utterance(&mut self) -> Result<Option<String>, VoiceControlError>;
}
```

This makes the outbound TTS path testable without needing a conversation loop: a TUI REPL, selected-call detail composer, socket client, mstream bridge, fixture replay, or scripted test can all feed text into the same outbound speech controller.

Telnyx-specific dial configuration such as `connection_id`, stream URL, credentials, and bidirectional media flags belongs in `bins/telnyx-gateway` configuration and adapter code, not in the provider-neutral `VoiceDialRequest`.

### Milestone 3: Full-Duplex TUI Chat Conversation

The third milestone composes milestone 1 and milestone 2 into a TUI-driven conversation experience:

```text
selected-call chat interface
-> media final-transcript forwarding
-> ConversationHandler
-> ConversationCommand::Say { text }
-> OutboundSpeechController::speak(...)
-> assistant response and playback status in the chat/detail pane
```

In this milestone, `ConversationHandler` is a bridge between transcript events and outbound commands, surfaced through a selected-call chat interface. The chat interface should show user transcript, assistant response state, outbound `say` activity, playback status, and call lifecycle. `ConversationHandler` should not own Telnyx REST details, ASR model details, or TTS packetization. Those remain in the inbound/outbound controllers and media pipelines.

Recommended composition rule:

- inbound handlers emit `TranscriptEvent`
- transcript consumers decide whether to log, send to tmux, or call a conversation handler
- conversation handlers emit provider-neutral commands such as `Say { text }`, `Call(CallAction::Hangup)`, `Call(CallAction::Transfer { ... })`, or `Noop`
- `bins/telnyx-gateway` maps those commands to Telnyx call-control and outbound media operations

Implementation note (@codex-366-impl, 2026-06-06 13:40 PDT): the first M3 implementation keeps `ConversationHandler` provider-neutral and gateway-local. The Telnyx media socket forwards only unsuppressed final transcript events for calls whose conversation state is attached. `answer` for inbound calls, outbound `dial`, and plain `conversation attach [call]` attach in auto/approved mode by default, but the built-in echo reply handler is disabled by default so normal live sessions are transcription-only after attach. Operators must start with `--conversation-smoke-test` or run `conversation smoke-test on` from the TUI/socket to enable the `I heard: ...` smoke-test loop. Operators can run `conversation disapprove [call]` mid-call to cancel active conversation TTS through the M2 clear/cancel path and leave the call in transcription-only mode. Manual mode remains available through `conversation mode manual [call]`; it records the assistant proposal in the selected-call chat state, and `conversation approve [call]` / `conversation say [call]` speaks the pending proposal through that command source's selected TTS backend when a non-smoke handler proposes a response. Auto mode routes `ConversationCommand::Say` to the extracted M2 `speech::queue_speech` path with the gateway default live TTS backend, currently `kokoro-82m` with Piper fallback; it intentionally does not use source-local `tts use` because media-triggered turns are not associated with a command source. Barge-in defaults to drop-and-regenerate when a conversation handler is enabled: meaningful partial ASR events and frame-level speech-onset detection for attached calls trigger the drop/cancel path when playback is active, while final ASR events still drive regeneration through the handler. `conversation barge-in off` disables partial/final/speech-onset clear for live smoke-test validation; if another final turn arrives while TTS is already active, the generated assistant text is retained as a proposal instead of interrupting playback. `CallAction::Hangup` maps to Telnyx hangup; future call actions fail closed and record conversation failure state until implemented.

### Milestone 5: Conversational Realism Latency Improvements (#402)

M5 is a post-M3 hardening slice for live turn feel, not a replacement for the `ConversationHandler` contract. The expanded scope in PR #417 keeps response generation turn-based on final transcript events, while reducing the time spent waiting for endpointing and first outbound audio:

- local ASR endpoint finalization finishes the active ASR session after the replay-sized trailing-silence pad, so final transcripts do not depend on a later utterance
- barge-in can be triggered by meaningful partial ASR or by frame-level speech onset while conversation playback is active; speech onset fires on initial speech or resumed speech after at least 120 ms of low energy; both reuse the same M2 cancel/clear path and remain gated by attached conversation state plus `conversation barge-in on|off`
- final transcripts still drive handler invocation and regenerated `ConversationCommand::Say`; speech-onset barge-in only interrupts active playback
- outbound TTS is queued per synthesized text chunk instead of after the whole utterance is synthesized; backend audio chunks inside one text chunk are concatenated before resampling/packetization so Piper-style short backend chunks do not reset the resampler every 40 ms
- the design remains compatible with future agent-human bidirectional ASR/TTS text streams because interruption policy is gateway-local and handler turn semantics stay explicit

Coordination note (@codex-366-impl, 2026-06-07 23:08 PDT): M5 exposes the media-safe primitive M4 needs for more conversational app-agent text streams: callers can keep the existing reject-on-overlap behavior or opt into cancel-and-replace when enqueueing outbound speech. Cancel-and-replace requests the same Telnyx `clear` path, drops queued old audio before the next frames are sent, returns the replaced playback id to the caller, and keeps stale clear/mark acknowledgements from demoting a newer active playback. M4 text-call protocol should treat stale or superseded `agent.turn` messages as normal conversation outcomes, not socket/protocol errors, and should include a terminal playback status such as `completed`, `canceled`, `failed`, or `superseded` in `playback.finished`.

The remaining realism work is measurement and tuning: live first-audio latency, frame pacing smoothness, and any audible sentence-boundary artifacts should be captured in M5 validation before expanding to non-smoke-test agents.

### Milestone 4: External Integration Harness

The fourth milestone proves that non-TUI integrations can drive the same gateway safely:

```text
Unix-domain socket client and harness appserver
-> command-source mux and Gateway Control API
-> application webhook subscription
-> call list/detail and per-call attachment APIs
-> inbound answer/transcript service flow
-> outbound dial/speak service flow
```

This milestone owns richer Unix-domain socket event polling, command-source mux validation, application webhooks, authenticated Gateway Control API read/attach flows, and a harness appserver. It should not be required for milestone 1 inbound TUI transcription, milestone 2 outbound TUI/socket TTS, or milestone 3 TUI chat conversation. The harness appserver should register a webhook subscription, verify signatures, reconcile call state through `GET /api/v1/calls/{call_id}`, attach to a call, answer inbound calls, drive outbound dial/speak through the gateway command layer or outbound dial/say through the Control API, and record observed events.

### Design Quality Assessment

Overall quality: viable and coherent, provided the four milestone gates are enforced. The strongest parts of the design are the reuse-first boundary around `motlie_model::typed` and `motlie_voice`, the separation of Telnyx schema/control into `bins/telnyx-gateway`, and the staged ASR-only -> TTS-only -> conversation -> external integration progression.

Primary design risks:

- scope creep: socket, webhooks, Control API, and harness appserver should stay in milestone 4 instead of blocking the first TUI slices
- media quality: the placeholder linear resampler must be replaced with an anti-aliased resampler before live telephony validation
- UI state complexity: call roster, selected-call detail, TTS composer, and chat mode need source-local selection and clear state transitions
- latency expectations: transport streaming is available before model-level incremental TTS; M5 queues audio after each synthesized text chunk, but first-audio latency is still bounded by the selected backend returning that first chunk
- operational safety: socket permissions, webhook HMAC, Control API auth, replayable state dumps, and secret redaction need explicit tests

### Recommended Telnyx v1 Pipelines

The recommended build order is four narrower milestone slices.

Milestone 1 inbound TUI transcription (#364):

```text
Inbound Telnyx audio
-> G.711 decode
-> resample 8 kHz to 16 kHz
-> sherpa-onnx streaming ASR
-> TuiTranscriptSink
-> selected-call detail transcript
```

Milestone 2 outbound dialer/TTS (#365):

```text
TUI/socket dial/speak command or selected-call detail text composer
-> OutboundSpeechController
-> Piper TTS
-> PCM at about 22 kHz
-> resample to 16 kHz
-> packetize to ~20 ms frames
-> outbound mpsc
-> single bidirectional Telnyx media WebSocket
```

Milestone 3 full-duplex TUI chat conversation (#366):

```text
selected-call chat interface
-> media final-transcript forwarding
-> ConversationHandler
-> ConversationCommand::Say { text }
-> OutboundSpeechController
-> selected-call chat/detail playback status
```

Milestone 4 external integration harness (#367):

```text
Unix-domain socket and harness appserver
-> command-source mux / Gateway Control API
-> application webhooks
-> call list/detail and per-call attachments
-> inbound answer/transcript service flow
-> outbound dial/speak service flow
```

These pipelines align with what currently works while allowing useful validation before the full duplex conversation path exists.

## Operator REPL and TUI Control Surface

`bins/telnyx-gateway` should be an operator-controlled gateway, not a daemon that immediately activates all call flows at startup. The operator chooses the local control mode at process start.

### Operator Input Modes

The gateway should expose the same `motlie-driver` command family through startup-selected input sources:

- `--tui` enables the local terminal UI with a left REPL pane and right call roster/detail area
- `--socket <path>` enables a Unix-domain command socket for headless local agent/tooling control
- `--load <dump_path>` replays durable configuration commands during startup before live command sources are accepted
- the authenticated HTTP Control API remains the remote application-server control surface for call reads, attachments, answer/reject/hangup, dial, and say

`--tui` and `--socket` are independent. If both are present, the gateway runs both and multiplexes their commands through one command dispatcher. If only `--socket` is present, the gateway is headless but still locally agent-controllable. If only `--tui` is present, behavior matches the local interactive flow. If neither is present, no local operator command input is started; the process must rely on `--load` and/or the authenticated HTTP Control API for configuration and control.

The Unix-domain socket is the preferred interface for local agent tooling. It should be sufficient for an agent to configure the gateway, poll call/event state, answer or hang up calls, dial outbound calls, and send `say` text without using application webhooks or the HTTP Control API. Appserver webhooks and the Control API remain separate integration surfaces for other processes and services.

The milestone 1 socket subset can be line-oriented for immediate local testing: each input line is Motlie driver command text such as `status`, `inbound enable --manual`, `calls`, `answer`, `call show`, or `shutdown`, and each output line is a JSON command result with `ok`, `lines`, `effects`, and `error`.

Milestone 4 should generalize this into newline-delimited JSON request/response frames around the same `motlie-driver` command grammar as the TUI REPL:

```json
{"id":"req-1","command":"status"}
{"id":"req-1","ok":true,"command_id":"cmd_01HZ...","lines":["listener ready"],"data":{"listener":"ready"}}
{"id":"req-2","command":"answer call_01HZ..."}
{"id":"req-2","ok":true,"command_id":"cmd_01HZ...","lines":["answering call_01HZ..."]}
```

Each response must carry the request `id`, a stable success/error shape, command output lines for human-readable parity with the TUI, and structured `data` for machine use when the command has a natural resource representation. Errors should include a machine-readable code and a human-readable message.

The socket is a local privileged control surface. It should bind only to a filesystem path chosen by `--socket`, set restrictive permissions by default, unlink stale socket files only after verifying they are sockets, and avoid accepting remote TCP traffic. Authentication can be filesystem-permission based for v1, with optional peer credential checks where available.

Agent event access should be polling-first. Some agents cannot receive pushed events; they can only run commands and inspect responses. The socket therefore needs command-level snapshots plus cursor polling over the gateway event journal:

```text
calls --json
call get <call_id> --json
events poll --after <cursor> [--call <call_id>] [--types <event,...>] [--limit <n>]
```

Recommended polling response:

```json
{
  "id": "req-3",
  "ok": true,
  "data": {
    "events": [
      {
        "seq": 101,
        "type": "call.inbound.pending",
        "call_id": "call_01HZ...",
        "created_at": "2026-05-31T21:08:12Z",
        "summary": "inbound call waiting"
      }
    ],
    "next_cursor": 101
  }
}
```

The event journal should be an in-memory ring buffer at minimum. Every event needs a monotonic `seq`, stable `type`, timestamp, gateway call id when applicable, and enough provider diagnostic fields for troubleshooting. If an agent polls with a cursor older than the retained journal, the socket should return a structured `cursor_expired` error and the agent should recover by calling `calls --json` and resuming from the latest cursor.

Socket event polling is the authoritative agent event path. Optional wake-up notifications may be layered on top for agents that sit inside a tmux session or mstream-driven prompt:

```text
agent notify tmux --target <target> --types call.inbound.pending,transcript.final
agent notify mstream --channel <channel> --types call.inbound.pending,call.ended
agent notify disable
```

These notifications are hints, not delivery guarantees. They should carry only a short wake-up message such as "motlie event: call.inbound.pending call_01HZ..." so the agent can decide to poll the socket. They must be opt-in, local-only, rate-limited, and avoid secrets or raw transcript text unless explicitly enabled by the operator.

The command-source mux should convert every command into a common envelope:

```rust
pub enum OperatorCommandSource {
    Tui,
    UnixSocket { client_id: OperatorClientId, path: PathBuf },
    ReplayFile { path: PathBuf },
    ControlApi { principal: ApiPrincipal },
}

pub struct OperatorCommandEnvelope {
    pub id: CommandId,
    pub source: OperatorCommandSource,
    pub received_at: SystemTime,
    pub text: String,
}
```

Mux rules:

- all command sources dispatch to the same command parser and controller layer
- commands receive a total order at the mux before mutating shared gateway state
- each command response is routed back to the source that submitted it
- shared state changes are published to the gateway event/status stream so the TUI reflects socket-originated changes
- source-local interactive state, such as `selected_call`, must not leak across sources

### TUI Layout

When `--tui` is enabled, the TUI should have one command pane and one call/status area:

- left pane: shell-style `motlie-driver` REPL surface combining command history, command output, and the active prompt in one pane; prompt placement and cursor position must be based on rendered terminal rows so wrapped or multi-line command output cannot leave the cursor detached from the `>` prompt
- right area: split vertically into a top call roster and a bottom selected-call detail pane

Right top call roster:

- lists every pending, active, dialing, held, failed, and recently ended call
- highlights new inbound calls as `PENDING INBOUND` / `WAITING`
- shows compact columns: short call id, direction, from, to, state, media state, transcript state, TTS state, age, and unread event count
- supports selection through `call use <call>` and keyboard navigation in the TUI

Right bottom selected-call detail:

- shows the currently selected call's full state, provider IDs, webhook/media status, codec, sample rate, and latest errors
- shows Telnyx termination reason/cause when the call ends or the media stream stops
- shows timeline events for that call
- shows milestone 1 ASR as an assembled transcript plus recent raw partial/final transcript events for debugging
- shows milestone 2 TTS playback state and the latest `speak` requests
- shows context-aware action hints such as `answer`, `reject`, `hangup`, `speak <text>`, `transcript follow`, or `conversation attach`

Inbound call notification behavior:

- when `call.initiated` arrives, insert or update a call roster row in `PendingInbound`
- mark the row highlighted and increment its unread event count
- append a global status event so the operator sees that a call arrived even if another call is selected
- if no call is selected, or the selected call is ended, auto-select the new pending call and show its detail pane
- if another active call is selected, do not steal selection; leave the new call highlighted until the operator selects it
- optionally emit a terminal bell or desktop notification when enabled, but the roster highlight must be sufficient by itself

The right bottom pane is not a second command input by default. The left REPL remains the TUI input path. Selected-call context should make TUI commands ergonomic: if a call is selected in that TUI session, `answer`, `reject`, `hangup`, and `speak <text>` apply to that call unless the command names a different call id. Socket and Control API clients should prefer explicit call IDs; if they support `call use <call>`, that selection is scoped to that socket session or API principal and must not change the TUI selection.

The right panes are the first visual validation surface. During milestone 1, an answered inbound call should show codec/media metadata plus live transcript updates in the selected-call detail pane.

### Gateway Runtime State

The command dispatcher should mutate one shared gateway runtime state object. Interactive cursor state belongs to the command source session, not the shared gateway state:

```rust
pub enum InboundMode {
    Disabled,
    Manual,
    AutoTranscribe { sink: TranscriptSinkKind },
    AutoConversation,
}

pub enum CallState {
    PendingInbound,
    WaitingForAnswer,
    Answering,
    Answered,
    MediaStarting,
    Transcribing,
    Dialing,
    Ringing,
    Speaking,
    Held,
    Active,
    Failed,
    Ended,
}

pub struct GatewayRuntimeState<C> {
    pub listener: ListenerStatus,
    pub telnyx: TelnyxProvisioningState,
    pub webhook_subscriptions: WebhookSubscriptionStore,
    pub persistence: GatewayPersistenceState,
    pub inbound_mode: InboundMode,
    pub calls: SessionRegistry,
    pub operator_sessions: OperatorSessionRegistry,
    pub outbound: C,
}

pub struct OperatorSessionState {
    pub source: OperatorCommandSource,
    pub selected_call: Option<CallHandle>,
    pub last_command_id: Option<CommandId>,
}
```

This state is gateway-local. `TelnyxProvisioningState` can store Telnyx application IDs, connection IDs, selected phone numbers, public webhook URL, public media WebSocket URL, and credential source. Those values should not leak into provider-neutral `motlie_voice::app` traits.

The TUI should derive its call roster from `SessionRegistry` plus a small view model:

```rust
pub struct CallRosterItem {
    pub call: CallHandle,
    pub direction: CallDirection,
    pub from: Option<VoiceAddress>,
    pub to: Option<VoiceAddress>,
    pub state: CallState,
    pub media_state: MediaState,
    pub transcript_state: TranscriptState,
    pub tts_state: TtsState,
    pub unread_events: u32,
}
```

The selected-call detail pane should be a view over the selected `CallHandle`, not a separate source of truth.

### Command Groups

The command set should be grouped by operator intent.

General gateway commands:

```text
help [command]
status
listener status
state dump [path]
shutdown [dump_path]
config show
config set webhook-url <https-url>
config set media-url <wss-url>
config set from-number <e164>
config set state-path <path>
log clear
log level <trace|debug|info|warn|error>
```

External automation commands:

```text
api token create <name>
api token revoke <token-id>
webhook subscription list
webhook subscription add <url> --events <event,...>
webhook subscription upsert <subscription-id> <url> --events <event,...> --secret-ref <secret-ref>
webhook subscription remove <subscription-id>
webhook subscription test <subscription-id>
```

Telnyx provisioning commands:

```text
telnyx app list
telnyx app create <name>
telnyx app use <connection-id>
telnyx app show
telnyx app webhook set <https-url>
telnyx number list
telnyx number use <e164>
telnyx number bind <e164> <connection-id>
telnyx number unbind <e164>
```

Inbound call commands:

```text
inbound status
inbound enable --manual
inbound enable --auto-transcribe
inbound disable
calls
call use <call>
answer [call] [--sink tui|stdout|webhook:<subscription-id>]
reject [call]
hangup [call]
transcript follow [call]
transcript clear [call]
```

Milestone 1 should implement `inbound enable --manual` first. In that mode, `call.initiated` creates a `PendingInbound` session, renders a highlighted row in the top call roster, and shows detail/transcript state in the bottom pane when selected. The gateway answers only after the operator runs `answer`.

Outbound call commands:

```text
dial <phone-or-sip-uri> [--from <e164>]
speak [call] <text...>
speak cancel [call]
hangup [call]
status [call]
tts status
tts list
tts use piper
```

The `dial` command creates or selects the active outbound call for the command source that ran it and should tell the operator or agent to wait for media before running `speak`. The `speak` command routes text through `OutboundSpeechController::speak()`. Non-REPL sources such as mstream broadcast, scripted fixtures, socket clients, or tmux-driven tests should call the same controller instead of creating another TTS pathway. `tts list`, `tts status`, and `tts use piper` mirror the ASR switching surface even though milestone 2 currently has one live TTS backend; this keeps future Piper/Qwen3-TTS A/B work additive.

Conversation bridge commands:

```text
conversation status
conversation smoke-test <on|off>
conversation barge-in [on|off|status]
conversation attach [call]
conversation detach [call]
conversation disapprove [call]
conversation approve [call]
conversation say [call]
conversation mode <manual|auto>
```

These belong after milestone 1 and milestone 2 are independently useful. `answer`, `dial`, and `conversation attach` wire selected media final transcript events to `ConversationHandler` in auto mode by default; they should not change the Telnyx media or model contracts. Built-in echo replies are test-only and require `--conversation-smoke-test` or `conversation smoke-test on`. Conversation barge-in defaults on and can be triggered by meaningful partial ASR, final overlap checks, or frame-level speech onset; operators can run `conversation barge-in off` from either the TUI or socket when validating the smoke-test echo loop and wanting active TTS to finish instead of being cleared by speech-onset/partial/final ASR. `conversation disapprove` cancels active conversation TTS and detaches the handler so the call remains transcription-only. Manual proposals are spoken only after `conversation approve` / `conversation say`.

### Replayable State Dumps

The gateway should persist durable configuration as a replayable command script, not as an opaque binary snapshot. This makes restart behavior auditable, editable, and compatible with the same `motlie-driver` command dispatcher used by the TUI, socket, and Control API surfaces.

Supported entry points:

```text
state dump [path]
shutdown [dump_path]
telnyx-gateway --load <dump_path>
```

`state dump [path]` writes the current durable gateway state and keeps running. `shutdown [dump_path]` writes the same dump, then exits cleanly after refusing new work and draining or terminating active sessions according to normal shutdown policy. If the path is omitted, `shutdown` uses `config state-path` if set; otherwise it should use a documented default such as `./telnyx-gateway.state.repl`.

`--load <dump_path>` replays the command file during startup before the gateway enables inbound handling or accepts Control API mutations. Replay should be deterministic and should fail closed: if a command fails, the gateway should report the failing line, leave inbound mode disabled, and require operator action.

Recommended dump format:

```text
# motlie telnyx-gateway state v1
# generated_at 2026-05-30T18:35:00Z
config set webhook-url https://motlie-gateway.example.ts.net/telnyx/webhooks
config set media-url wss://motlie-gateway.example.ts.net/telnyx/media
config set from-number +15557654321
config set state-path ./telnyx-gateway.state.repl
telnyx app use <connection-id>
telnyx app webhook set https://motlie-gateway.example.ts.net/telnyx/webhooks
telnyx number use +15557654321
telnyx number bind +15557654321 <connection-id>
webhook subscription upsert whsub_01HZ... https://app.example.com/motlie/events --events call.inbound.pending,transcript.final,call.ended --secret-ref env:MOTLIE_APP_WEBHOOK_SECRET
inbound enable --manual
```

The dump should include durable configuration:

- public webhook and media URLs
- selected Telnyx application / connection ID
- selected and bound Telnyx phone number
- default caller ID / from number
- inbound mode
- webhook subscriptions and event filters
- application webhook secret references
- Control API token metadata or secret references
- selected ASR/TTS model names if they are set through the operator command surface

The dump should not include transient runtime state:

- active calls
- pending calls
- media WebSocket sessions
- transcript history
- in-flight TTS playback
- retry queues, except optionally durable undelivered webhook events in a separate event journal

Secret handling rule:

- do not dump raw Telnyx API keys, application webhook HMAC secrets, or bearer tokens
- dump secret references such as `env:TELNYX_API_KEY`, `env:MOTLIE_APP_WEBHOOK_SECRET`, or `file:/run/secrets/...`
- if token persistence is required, dump only hashed token material or a command that imports a hash/reference, never the bearer token itself

Replay commands should be idempotent wherever possible. In particular, dump output should prefer `use`, `bind`, and `upsert` forms over commands that blindly create duplicates. Commands that touch Telnyx remotely, such as `telnyx app webhook set` or `telnyx number bind`, should tolerate replay when the remote object is already in the desired state.

### Example Milestone 1 Operator Flow

```text
status
config set webhook-url https://motlie-gateway.example.ts.net/telnyx/webhooks
config set media-url wss://motlie-gateway.example.ts.net/telnyx/media
telnyx app create motlie-local
telnyx app webhook set https://motlie-gateway.example.ts.net/telnyx/webhooks
telnyx number bind +15551234567 <connection-id>
inbound enable --manual
calls
answer <call>
transcript follow <call>
shutdown ./telnyx-gateway.state.repl
```

The gateway should log every API request, selected app/number, webhook update, inbound call event, answer action, media `start`, gateway call id, Telnyx diagnostic ids, stream id, observed codec/sample rate, and transcript partial/final event into the global status stream, call roster, or selected-call detail pane as appropriate.

### Why This Belongs Above `libs/model`

`libs/model` contracts are already correct for model I/O:

- `StreamingTranscriber::open_session()` creates a typed ASR session
- `TranscriptionSession::ingest()` accepts typed `AudioBuf` input
- `SpeechSynthesizer::synthesize()` returns a typed `SpeechStream`
- `SpeechStream::next_chunk()` yields typed `AudioBuf` output

Telnyx integration adds:

- network transport
- JSON framing
- codec transcoding
- jitter and reordering
- call-control lifecycle
- application conversation orchestration

Those are telephony concerns, not model capability concerns.

## Application Webhooks and Gateway Control API

The gateway should support external automation in addition to the operator TUI. This surface belongs to milestone 4 (#367); it is designed here so contracts are clear, but implementation and validation must not block the M1 inbound TUI, M2 outbound TUI, or M3 TUI chat milestones. This is a separate surface from Telnyx voice webhooks:

- Telnyx webhooks come into `bins/telnyx-gateway` from Telnyx.
- Gateway application webhooks go out from `bins/telnyx-gateway` to a user's application server.
- The Gateway Control API is called by that application server to list current calls, inspect one call, attach to a call's event/transcript stream, answer, reject, hang up, dial, and say text.
- The Unix-domain socket is not an appserver webhook/control API substitute; it is the local agent-tooling interface and can provide command execution, snapshots, event polling, and optional local wake-up notifications without exposing HTTP control routes.

This lets the gateway act as a telephony edge service. Another server can subscribe to call events, decide what to do, reconcile current state through `GET` endpoints, then call back into the gateway without embedding Telnyx protocol logic, media adaptation, ASR, or TTS.

### Event Envelope

All gateway-emitted webhooks should use one stable envelope:

```json
{
  "id": "evt_01HZ...",
  "type": "call.inbound.pending",
  "created_at": "2026-05-30T18:30:00Z",
  "gateway_id": "gw_local",
  "call": {
    "id": "call_01HZ...",
    "direction": "inbound",
    "state": "pending",
    "from": "+15551234567",
    "to": "+15557654321"
  },
  "provider": {
    "name": "telnyx",
    "call_control_id": "..."
  },
  "data": {}
}
```

`call.id` is the gateway's stable opaque call ID. External servers should use that ID with the Gateway Control API. Provider fields are diagnostic and should not be required for normal application control.

Recommended webhook headers:

```text
X-Motlie-Event-Id: evt_01HZ...
X-Motlie-Timestamp: 2026-05-30T18:30:00Z
X-Motlie-Signature: v1=<hmac-sha256>
```

The signature should cover `timestamp + "." + raw_body` with a per-subscription secret. Webhook receivers should treat event delivery as at-least-once and deduplicate by `X-Motlie-Event-Id`.

### Event Types

Inbound milestone 1 events:

```text
call.inbound.pending
call.answering
call.answered
media.started
transcript.partial
transcript.final
call.ended
call.failed
```

Outbound milestone 2 events:

```text
call.outbound.created
call.dialing
call.ringing
call.answered
media.started
tts.playback.started
tts.playback.finished
tts.playback.failed
call.ended
call.failed
```

Conversation milestone 3 events can add:

```text
conversation.attached
conversation.command.created
conversation.command.completed
conversation.detached
```

Transcript event payloads should include the text, whether it is partial or final, the ASR backend, and a gateway-local monotonic sequence:

```json
{
  "id": "evt_01HZ...",
  "type": "transcript.final",
  "call": { "id": "call_01HZ...", "direction": "inbound", "state": "transcribing" },
  "data": {
    "sequence": 42,
    "text": "hello this is a test",
    "is_final": true,
    "asr_backend": "sherpa-onnx"
  }
}
```

### Webhook Subscription API

The TUI should be able to configure subscriptions, but external automation needs an HTTP API too.

"Gateway emits an event" means the gateway makes an outbound HTTP `POST` to the subscription URL with the JSON event envelope as the request body. It should not use `GET` for event delivery because the event has a body, signature, retry identity, and acknowledgement semantics. The receiving application server acknowledges delivery by returning any `2xx` response. Non-`2xx`, timeout, or connection failure means delivery failed and should be retried according to policy.

Recommended endpoints:

```text
POST   /api/v1/webhook-subscriptions
GET    /api/v1/webhook-subscriptions
GET    /api/v1/webhook-subscriptions/{subscription_id}
DELETE /api/v1/webhook-subscriptions/{subscription_id}
POST   /api/v1/webhook-subscriptions/{subscription_id}/test
```

Create request:

```json
{
  "url": "https://app.example.com/motlie/events",
  "events": [
    "call.inbound.pending",
    "transcript.partial",
    "transcript.final",
    "call.ended"
  ],
  "secret_ref": "env:MOTLIE_APP_WEBHOOK_SECRET"
}
```

Delivery policy:

- deliver events with `POST <subscription.url>` and `Content-Type: application/json`
- treat `2xx` as acknowledgement; treat non-`2xx`, timeout, or connection failure as retryable delivery failure
- retry non-2xx responses with bounded exponential backoff
- disable or quarantine a subscription after repeated failures
- never block Telnyx webhook handling on application webhook delivery
- include enough IDs for the receiving server to call the Gateway Control API later

Example delivery request:

```http
POST /motlie/events HTTP/1.1
Host: app.example.com
Content-Type: application/json
X-Motlie-Event-Id: evt_01HZ...
X-Motlie-Timestamp: 2026-05-30T18:30:00Z
X-Motlie-Signature: v1=<hmac-sha256>

{
  "id": "evt_01HZ...",
  "type": "call.inbound.pending",
  "call": {
    "id": "call_01HZ...",
    "direction": "inbound",
    "state": "pending",
    "from": "+15551234567",
    "to": "+15557654321"
  },
  "data": {}
}
```

The app server should verify the HMAC signature, persist or enqueue the event idempotently by `id`, then return `204 No Content` or another `2xx`. If it wants to answer the call, that is a second HTTP request in the opposite direction: the app server calls `POST /api/v1/calls/{call_id}/answer` on the gateway.

### Gateway Control API

The Gateway Control API should expose the same actions as the operator command surface. The TUI, Unix-domain socket, replay loader, and Control API should dispatch into the same internal command/controller layer so they cannot drift.

Recommended call read endpoints:

```text
GET  /api/v1/calls
GET  /api/v1/calls/{call_id}
```

`GET /api/v1/calls` returns the gateway's current in-memory call registry. It is the app server's reconciliation API, not an event stream. It should support narrow filters and stable pagination:

```text
GET /api/v1/calls?state=pending,active&direction=inbound&limit=50&cursor=...
```

Example response:

```json
{
  "calls": [
    {
      "id": "call_01HZ...",
      "direction": "inbound",
      "state": "pending",
      "from": "+15551234567",
      "to": "+15557654321",
      "media": { "state": "not_started" },
      "transcription": { "state": "not_started" },
      "tts": { "state": "idle" },
      "attachments": [],
      "created_at": "2026-05-30T18:30:00Z",
      "updated_at": "2026-05-30T18:30:02Z"
    }
  ],
  "next_cursor": null
}
```

`GET /api/v1/calls/{call_id}` returns the same resource at full detail, including provider diagnostic IDs, last error, currently attached webhook subscriptions, media state, transcript state, TTS state, and recent timeline entries. Provider IDs remain diagnostic; app servers should use the gateway call ID for control.

Recommended call attachment endpoints:

```text
POST   /api/v1/calls/{call_id}/attachments
GET    /api/v1/calls/{call_id}/attachments
DELETE /api/v1/calls/{call_id}/attachments/{attachment_id}
```

A call attachment binds an application-side consumer to one gateway call. In v1 this should usually mean "send this call's selected events/transcripts to this already-registered webhook subscription." It does not replace application webhooks; it scopes webhook delivery to a specific call and records which app server is observing or driving the call.

Attach request:

```json
{
  "subscription_id": "whsub_01HZ...",
  "role": "observer",
  "events": [
    "call.answered",
    "media.started",
    "transcript.partial",
    "transcript.final",
    "call.ended",
    "call.failed"
  ],
  "transcription": {
    "enabled": true,
    "partials": true,
    "finals": true
  }
}
```

Attach response:

```json
{
  "attachment_id": "att_01HZ...",
  "call_id": "call_01HZ...",
  "subscription_id": "whsub_01HZ...",
  "role": "observer",
  "state": "attached",
  "events": [
    "transcript.partial",
    "transcript.final",
    "call.ended",
    "call.failed"
  ]
}
```

Attachment semantics:

- attaching before `answer` stores the requested event/transcript sinks and applies them when media starts
- attaching after media has started adds the sinks to the active call session without restarting Telnyx streaming
- deleting an attachment stops delivery to that subscription for that call, but does not hang up the call
- attachments must be idempotent under `Idempotency-Key`; duplicate requests for the same call/subscription/role should return the existing attachment
- initial roles should include `observer`; milestone 3 can add `conversation_controller` when app-server-driven conversation handling is ready

Recommended call-control endpoints:

```text
POST /api/v1/calls/{call_id}/answer
POST /api/v1/calls/{call_id}/reject
POST /api/v1/calls/{call_id}/hangup
```

Recommended outbound dialer/TTS endpoints:

```text
POST /api/v1/calls
POST /api/v1/calls/{call_id}/say
POST /api/v1/calls/{call_id}/interrupt
```

Inbound answer request:

```json
{
  "transcription": {
    "enabled": true,
    "sink": {
      "type": "webhook",
      "subscription_id": "whsub_01HZ..."
    }
  }
}
```

The answer request may either include a sink directly, as above, or rely on a prior call attachment. The prior-attachment form is better when an app server wants to acknowledge a pending inbound webhook, attach its transcript/event subscription, then answer as a separate, retryable control step.

Outbound dial request:

```json
{
  "to": "+15551234567",
  "from": "+15557654321",
  "events": {
    "subscription_id": "whsub_01HZ..."
  },
  "streaming": {
    "establish_before_originate": true
  }
}
```

TTS request:

```json
{
  "text": "hello from motlie",
  "interrupt_current": false
}
```

All mutating control API requests should support `Idempotency-Key` so external servers can retry safely.

GET endpoints complement, but do not replace, application webhooks:

- webhooks are push notifications for transitions and transcript/TTS events
- `GET /api/v1/calls` and `GET /api/v1/calls/{call_id}` are pull-based snapshots for startup, reconnect, missed-webhook recovery, and UI reconciliation
- `POST /api/v1/calls/{call_id}/attachments` converts a global webhook subscription into a per-call consumer relationship
- call-control mutations such as `answer`, `hangup`, and `say` remain explicit API calls; receiving or attaching to webhooks must not implicitly answer a call

### Programmatic Inbound Flow

```text
external app registers webhook subscription
-> inbound call reaches Telnyx
-> Telnyx sends call.initiated to gateway
-> gateway records PendingInbound and emits call.inbound.pending
-> external app receives event
-> external app GETs /api/v1/calls/{call_id} to reconcile current state
-> external app POSTs /api/v1/calls/{call_id}/attachments to receive that call's transcript/events
-> external app POSTs /api/v1/calls/{call_id}/answer
-> gateway answers with media stream
-> gateway emits media.started
-> gateway emits transcript.partial / transcript.final events
```

This is the milestone 4 service analogue of the milestone 1 inbound TUI flow. The TUI can still show the same call and transcript stream, but Control API, webhook delivery, and app-server attachment are milestone 4 integration work and must not block milestone 1.

### Programmatic Outbound Flow

```text
external app POSTs /api/v1/calls
-> gateway dials through Telnyx with media streaming attached
-> gateway emits call.outbound.created / call.answered / media.started
-> external app POSTs /api/v1/calls/{call_id}/say
-> gateway synthesizes TTS and sends outbound media
-> gateway emits tts.playback.started / tts.playback.finished
```

This is the milestone 4 service analogue of the milestone 2 outbound TUI flow. The gateway becomes a dialer and TTS transport service while still hiding Telnyx call-control details and media adaptation from the calling application, but the Control API and webhook service path must not block milestone 2.

### Auth and Exposure

The Control API and application webhook subscription API are powerful. For local development they can bind to the same listener as the gateway server, but production deployment should require:

- bearer token or mTLS for the Control API
- HMAC signatures for emitted webhooks
- per-subscription event filters
- optional allowlist for webhook destination hosts
- audit logging in the global status stream and structured logs

Do not expose the Control API publicly without authentication. If the Telnyx webhook/media listener is exposed with Tailscale Funnel or another tunnel, the Control API should either remain bound to localhost, require a separate authenticated route, or be disabled on that public listener.

## Crate Hierarchy and API Surfaces

### Crate Split

The implementation should extend the landed provider-agnostic crate and keep Telnyx-specific adapter modules inside the `bins/telnyx-gateway` executable crate. The rule is reuse first: do not create a second voice/media hierarchy when `motlie-voice` already owns the shared primitives from PR #209.

Recommended hierarchy:

```text
libs/
  voice/                         # existing motlie-voice crate from PR #209
    Cargo.toml                   # extend, do not recreate
    src/
      lib.rs
      error.rs
      config.rs
      frame.rs                   # already defines PcmFrame<const RATE_HZ, C, E>
      wav.rs                     # already defines streaming WAV helpers
      app/
      runtime/
      pipeline/
        convert.rs               # already has decode/downmix/sample conversion helpers
        resample.rs              # already has Resampler + LinearInterpolator placeholder
      codec/
      telephony/
bins/
  telnyx-gateway/
    Cargo.toml
    docs/
      DESIGN.md
      PLAN.md
    src/
      lib.rs
      error.rs
      main.rs
      cli.rs
      serve.rs
      logging.rs
      operator/
        mod.rs
        commands.rs
        state.rs
        session.rs
        persistence.rs
        socket.rs
        mux.rs
        tui.rs
      api/
      events/
      webhook/
      call_control/
      media/
      adapter.rs
```

### `libs/voice` Surface

`libs/voice` already exists as `motlie-voice`. It should continue to contain only provider-neutral abstractions and reusable media infrastructure. Telnyx work extends this crate; it must not duplicate its existing `PcmFrame`, WAV, conversion, or resampling surfaces.

Recommended modules:

```text
libs/voice/src/
  lib.rs
  error.rs
  config.rs
  frame.rs        # landed in PR #209
  wav.rs          # landed in PR #209

  app/
    mod.rs
    context.rs
    conversation.rs
    dtmf.rs
    ivr.rs

  runtime/
    mod.rs
    asr.rs
    tts.rs
    session.rs

  pipeline/
    mod.rs
    stage.rs
    builder.rs
    types.rs      # new Telnyx slice types such as EncodedFrame<C>
    markers.rs
    reorder.rs
    convert.rs    # extend existing helpers; do not redefine them elsewhere
    resample.rs   # replace/wrap LinearInterpolator before live telephony
    chunk.rs
    packetize.rs

  codec/
    mod.rs
    g711.rs
    l16.rs

  telephony/
    mod.rs
    digits.rs
    actions.rs
    events.rs
    tracks.rs
```

`libs/voice` owns:

- application-level control traits and state in `motlie_voice::app`, including `ConversationHandler`, `DtmfHandler`, `IvrNavigator`, and `ConversationContext`
- provider-neutral telephony vocabulary in `motlie_voice::telephony`, including `DtmfDigit`, `CallAction`, call/media events, and track/direction markers
- typed media values such as new `EncodedFrame<C>` and the existing `PcmFrame<const RATE_HZ, C, E>`
- stage traits and compile-time stage composition
- provider-neutral codecs such as `PCMU`, `PCMA`, and `L16`
- provider-neutral telephony packetization and outbound pacing primitives
- an anti-aliased resampler implementation or wrapper suitable for telephony
- provider-neutral runtime wrappers for selected ASR/TTS model combinations

`libs/voice` must not own:

- Telnyx webhook payload structs
- Telnyx `call_control_id`
- Telnyx WebSocket message enums
- Telnyx REST command payloads

### `motlie_voice::telephony` Surface

`motlie_voice::telephony` is the provider-neutral telephony vocabulary layer. It is not a `TelephonyProvider` abstraction and it should not hide Telnyx, Twilio, SIP, or any other provider behind a generic call-control client.

It owns reusable telephony concepts that application logic and provider adapters can share:

- `DtmfDigit`
- `CallAction`
- provider-neutral call and media lifecycle events
- track, direction, leg, and pacing vocabulary where the concept is not provider-specific
- correlation metadata needed by provider-neutral tests, logs, and jitter/reorder policy

It must not own:

- provider webhook schemas
- provider WebSocket message enums
- provider REST command payloads or clients
- provider credentials, connection IDs, webhook URLs, or transport servers
- application policy about what to say or which action to take

The intended layering is:

```text
provider adapter, e.g. bins/telnyx-gateway
-> motlie_voice::telephony events/actions
-> motlie_voice::app handlers
-> provider adapter maps CallAction back to provider commands
```

### Provider Adapter Boundary

Do not introduce a broad `TelephonyProvider` trait in the first implementation. The provider boundary should be modeled by a small set of composable surfaces:

- Telnyx event/schema mapping in `bins/telnyx-gateway`, which translates webhook/WebSocket/REST payloads into provider-neutral `motlie_voice::telephony` events, actions, track markers, and sequence metadata
- inbound call handling in `bins/telnyx-gateway`, which owns answer/stream lifecycle and feeds a provider-neutral `InboundAsrPipeline`
- outbound call control in `bins/telnyx-gateway`, which owns Telnyx `dial`, `hangup`, streaming setup, and provider-specific IDs while implementing or delegating to the provider-neutral `OutboundSpeechController`
- application policy in `motlie_voice::app`, including `TranscriptSink`, `ConversationHandler`, `DtmfHandler`, `IvrNavigator`, `ConversationContext`, and provider-neutral conversation commands

This avoids an over-wide provider abstraction while still making the useful pieces testable and reusable. A future Twilio or SIP adapter should reuse the same `telephony` vocabulary, `app` traits, and media pipeline stages, but it can own its provider-specific transport and call-control mapping in its own gateway crate.

### `bins/telnyx-gateway` Surface

`bins/telnyx-gateway` should be the Telnyx-specific executable crate over `libs/voice`. It owns Telnyx protocol/control modules and process wiring, but it must not become a second provider-neutral voice library.

Recommended modules:

```text
bins/telnyx-gateway/src/
  lib.rs
  error.rs

  webhook/
    mod.rs
    payloads.rs
    verify.rs
    server.rs

  call_control/
    mod.rs
    client.rs
    commands.rs
    callbacks.rs

  media/
    mod.rs
    protocol.rs
    websocket.rs
    session.rs

  operator/
    mod.rs
    commands.rs
    state.rs
    session.rs
    persistence.rs
    socket.rs
    mux.rs
    tui.rs

  api/
    mod.rs
    control.rs
    auth.rs
    subscriptions.rs

  events/
    mod.rs
    envelope.rs
    dispatcher.rs
    delivery.rs

  adapter.rs
```

`bins/telnyx-gateway` owns:

- CLI parsing
- `motlie-driver` command registration and execution
- optional left REPL pane plus right call roster/detail TUI state, rendering, and command/status routing
- Unix-domain command socket serving, client session tracking, and command-source mux routing
- replayable command dump generation and startup `--load` replay
- Gateway Control API routing and authentication
- application webhook subscription storage, event envelope construction, and delivery retries
- environment loading
- model selection flags
- listen address, webhook path, media path, command socket path, TUI enablement, and operator-configured public URLs
- tunnel-friendly startup wiring
- logging and tracing
- Telnyx webhook JSON schema
- Telnyx WebSocket event schema
- Telnyx REST request and response types
- gateway application webhook event schema and delivery policy
- translation between Telnyx `start.media_format` and `libs/voice` typed transport frames
- translation from Telnyx `media.chunk` into provider-neutral per-track frame sequence numbers before handing frames to `libs/voice`
- translation between provider-neutral call actions and Telnyx call-control commands
- Telnyx-specific hangup/error policy for unsupported codecs or protocol violations

It should assemble:

- concrete `libs/models` bundle selections
- concrete `libs/voice` runtime and pipeline selections
- Telnyx protocol modules that map transport events into provider-neutral voice stages

`bins/telnyx-gateway` must not own:

- generic resamplers
- generic PCM frame types
- generic chunkers or packetizers
- provider-neutral telephony codecs such as G.711 or L16
- application conversation traits
- reusable runtime abstractions

### Provider-Neutral API Rule

If a type or module could be reused for Twilio or another telephony provider, it belongs in `libs/voice`, not `bins/telnyx-gateway`.

Examples that belong in `libs/voice`:

- `EncodedFrame<C>`
- existing `PcmFrame<const RATE_HZ, C, E>`
- `motlie_voice::app::ConversationHandler`
- `motlie_voice::app::DtmfHandler`
- `motlie_voice::app::IvrNavigator`
- `motlie_voice::telephony::DtmfDigit`
- `motlie_voice::telephony::CallAction`
- `Resampler`
- `Packetizer`

Examples that belong in `bins/telnyx-gateway`:

- `TelnyxWebhookEvent`
- `TelnyxWsMessage`
- `TelnyxCallControlClient`
- `TelnyxStreamStart`
- `TelnyxMediaPayload`

### Recommended Audio Normal Form

Use a gateway-internal normal form of:

- mono
- `16_000 Hz`
- `AudioBuf<i16, 16_000, Mono>` / `PcmFrame<16_000, Mono, i16>` at the media boundary

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
-> Telnyx base64/protocol mapping
-> EncodedFrame<C> with provider-neutral sequence
-> reorder / jitter normalization
-> transport decode
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

### Telnyx Media Schema Mapping

The gateway should map Telnyx WebSocket fields into typed transport values explicitly:

- `start.media_format.encoding` selects the inbound transport codec marker, for example `Pcmu`, `Pcma`, or `L16`
- `start.media_format.sample_rate` selects the inbound transport sample-rate const, for example `8_000` or `16_000`
- `start.media_format.channels` selects the inbound channel-layout marker; current telephony expectation is mono
- each `media.payload` becomes one `EncodedFrame<C>` after base64 decoding and Telnyx protocol mapping
- `media.chunk` stays in `bins/telnyx-gateway`; that adapter maps it to a provider-neutral per-track sequence before a frame enters `libs/voice`

Design rule:

- do not bypass `start.media_format`
- do not infer codec or sample rate from payload length alone
- instantiate inbound decode and normalization stages only after the `start` frame is received
- provider-neutral `libs/voice` reorder stages operate on generic frame sequence metadata, never directly on Telnyx `media.chunk`

### Concrete Backend Requirements Matrix

The implementation must not leave ASR/TTS media requirements to backend-specific experimentation. The current concrete backends imply the following adaptation targets.

| Backend | Capability | Runtime-facing Input / Output | Required Gateway Adaptation | Concrete Notes |
|---------|------------|-------------------------------|-----------------------------|----------------|
| `sherpa-onnx` streaming Zipformer | ASR | Ordered `AudioBuf<i16, 16_000, Mono>` values passed to `TranscriptionSession::ingest()` | map Telnyx chunk to provider-neutral sequence -> reorder -> decode Telnyx transport -> mono mixdown if needed -> resample to `16 kHz` -> feed ordered `AudioBuf<i16, 16_000, Mono>` values into `TranscriptionSession::ingest()` | current `StreamingTranscriber::Input` is `AudioBuf<i16, 16_000, Mono>`; there is no typed `f32` input and no internal resampler contract; Sherpa converts i16 samples to f32 internally for feature extraction |
| Moonshine streaming | ASR | Input stream is normalized to mono `16 kHz`; decode loop processes fixed `1280`-sample chunks | map Telnyx chunk to provider-neutral sequence -> reorder -> decode Telnyx transport -> mono mixdown if needed -> resample to `16 kHz` -> rechunk into `1280`-sample windows before runtime inference | `CHUNK_SIZE = 1280` samples, which is `80 ms` at `16 kHz`; chunk sequence is strict equality, not just monotonic increase |
| Piper `en_US ljspeech medium` | TTS | Output PCM is mono `S16Le`; current curated bundle audio spec is `22.05 kHz` | drain TTS chunks -> resample `22.05 kHz` to Telnyx outbound rate -> packetize -> encode transport codec | emits buffered PCM after full synthesis; output chunks are `40 ms` of PCM at model rate |
| Qwen3-TTS.cpp `0.6B` | TTS | Output PCM is mono `f32` at config sample rate; current default config is `24 kHz` | drain TTS chunks -> if needed convert `f32` to outbound working PCM format -> resample `24 kHz` to Telnyx outbound rate -> packetize -> encode transport codec | current config default is `24_000 Hz`; current typed stream drains a full synthesized buffer in `40 ms` chunks; reference-audio conditioning path also resamples inbound reference audio to model rate |

### Concrete Combination Requirements

The design documents multiple pairings because the media adaptation requirements differ materially across them. That does not make them equal implementation targets.

Execution rule:

- Sherpa inbound ASR is the first required live milestone
- Piper outbound TTS is the second required live milestone
- `Sherpa + Piper` is the first and only required complete duplex backend pairing once the conversation bridge exists
- all other pairings in this section are specified so the generic pipeline can be designed correctly up front
- no first-milestone acceptance criteria, examples, or operator docs may depend on `Moonshine` or `Qwen3-TTS`

#### Sherpa + Piper

Inbound:

- Telnyx preferred inbound codec: `L16` at `16 kHz`, fallback `PCMU` / `PCMA` at `8 kHz`
- gateway inbound target before ASR push: mono `16 kHz`
- if Telnyx starts in `PCMU` or `PCMA`, decode G.711 and resample `8 kHz -> 16 kHz`
- emit typed `AudioBuf<i16, 16_000, Mono>` values in strict arrival order after reorder buffering

ASR behavior:

- the gateway delivers ordered `AudioBuf<i16, 16_000, Mono>` values to Sherpa
- Sherpa does not currently expose a typed `f32` streaming input or an internal resampler contract
- Sherpa converts i16 samples to f32 internally for feature extraction after the gateway has already normalized to `16 kHz`
- Sherpa decode cadence is governed by ONNX metadata plus a `40 ms` frame shift

Outbound:

- Piper emits mono `S16Le` at `22.05 kHz`
- gateway must resample `22.05 kHz -> 16 kHz` if Telnyx outbound uses `L16 16 kHz`
- then packetize into telephony-sized outbound frames and encode as `L16` or G.711

#### Sherpa + Qwen3-TTS

Status:

- follow-on pairing, not part of milestone 1 inbound ASR, milestone 2 outbound TTS, or the first required duplex milestone

Inbound:

- same Sherpa inbound path as above: normalized mono `16 kHz`

Outbound:

- Qwen3-TTS.cpp emits mono `f32` at `24 kHz`
- gateway must convert `f32` to the chosen outbound working PCM representation if the packetizer expects signed 16-bit PCM
- gateway must resample `24 kHz -> 16 kHz` for Telnyx `L16 16 kHz`, or `24 kHz -> 8 kHz` if forced onto G.711
- packetizer must preserve the `40 ms` chunk cadence or split it into smaller telephony packets if lower playback latency is needed

Design implication:

- this combination requires both sample-format conversion and sample-rate conversion on the outbound side

#### Moonshine + Qwen3-TTS

Status:

- follow-on pairing, not part of milestone 1 inbound ASR, milestone 2 outbound TTS, or the first required duplex milestone

Inbound:

- Telnyx preferred inbound codec remains `L16 16 kHz`, fallback `PCMU` / `PCMA 8 kHz`
- gateway inbound target before Moonshine runtime: mono `16 kHz`
- gateway must rechunk normalized PCM into fixed `1280`-sample blocks, which is `80 ms` at `16 kHz`
- sequence numbers must be gap-free because the Moonshine stream expects exact next sequence values

Outbound:

- same Qwen3-TTS.cpp outbound adaptation as above: mono `f32 24 kHz` -> convert -> resample -> packetize -> encode

Design implication:

- Moonshine is less tolerant of arbitrary telephony frame sizes than Sherpa because the runtime loop is structured around fixed `1280`-sample chunk inference
- the inbound chunker for Moonshine should therefore be a dedicated stage, not a parameter tweak on the Sherpa path

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

`PcmFrame` already exists in `motlie-voice` and must be reused:

```rust
pub struct PcmFrame<const RATE_HZ: u32, C: ChannelLayout, E> {
    audio: AudioBuf<E, RATE_HZ, C>,
}
```

That type composes directly over `motlie_model::typed::AudioBuf<E, RATE_HZ, C>`. Telnyx must not introduce a second `PcmFrame` that also tries to model rate, channels, or sample type. If the live transport path needs sequence numbers, end-of-stream markers, or jitter metadata, add a separate envelope or stage-local metadata type around `PcmFrame` rather than duplicating the frame itself.

New Telnyx work should add typed encoded transport values and codec markers:

```rust
use core::marker::PhantomData;

pub struct Pcmu;
pub struct Pcma;
pub struct L16;

pub struct EncodedFrame<C> {
    pub bytes: Vec<u8>,
    pub sequence: u64,
    pub end_of_stream: bool,
    _codec: PhantomData<C>,
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
use motlie_model::typed::{ChannelLayout, Mono, Stereo};

pub trait TelephonyCodec {}
impl TelephonyCodec for Pcmu {}
impl TelephonyCodec for Pcma {}
impl TelephonyCodec for L16 {}

pub trait LinearPcmSample {}
impl LinearPcmSample for i16 {}
impl LinearPcmSample for f32 {}
```

The existing channel markers come from `motlie_model::typed`. Sample rate is carried by `PcmFrame` as a const generic. The current landed `PcmFrame` uses its third type parameter as the sample type (`i16`, `f32`, and so on). Encoding markers such as `S16Le` and `F32Le` may be added later if the implementation needs a clearer distinction between sample type and wire encoding, but Telnyx should not redefine `PcmFrame` to get that marker.

### Common Stage Contracts

Recommended transport codec stage:

```rust
pub trait Decoder<C, const RATE_HZ: u32, Ch, S>
where
    C: TelephonyCodec,
    Ch: ChannelLayout,
    S: LinearPcmSample,
{
    fn decode(&mut self, input: EncodedFrame<C>)
        -> Result<PcmFrame<RATE_HZ, Ch, S>, MediaError>;
}

pub trait Encoder<C, const RATE_HZ: u32, Ch, S>
where
    C: TelephonyCodec,
    Ch: ChannelLayout,
    S: LinearPcmSample,
{
    fn encode(&mut self, input: PcmFrame<RATE_HZ, Ch, S>)
        -> Result<EncodedFrame<C>, MediaError>;
}
```

Recommended resampler stage:

```rust
pub trait TelephonyResampler<const IN_RATE_HZ: u32, const OUT_RATE_HZ: u32, Ch, S>
where
    Ch: ChannelLayout,
    S: LinearPcmSample,
{
    fn resample(
        &mut self,
        input: PcmFrame<IN_RATE_HZ, Ch, S>,
    ) -> Result<PcmFrame<OUT_RATE_HZ, Ch, S>, MediaError>;
}
```

The existing `motlie_voice::pipeline::resample::Resampler` trait is a starting point, but its current `LinearInterpolator` implementation is example-grade. The Telnyx live path must use an anti-aliased implementation before live-call acceptance.

Recommended sample-format conversion stage:

```rust
pub trait SampleConverter<const RATE_HZ: u32, Ch, InSample, OutSample>
where
    Ch: ChannelLayout,
    InSample: LinearPcmSample,
    OutSample: LinearPcmSample,
{
    fn convert(&mut self, input: PcmFrame<RATE_HZ, Ch, InSample>)
        -> Result<PcmFrame<RATE_HZ, Ch, OutSample>, MediaError>;
}
```

Recommended framing stage for ASR:

```rust
pub trait AsrChunker<const RATE_HZ: u32, Ch, S>
where
    Ch: ChannelLayout,
    S: LinearPcmSample,
{
    fn push_frame(
        &mut self,
        input: PcmFrame<RATE_HZ, Ch, S>,
    ) -> Result<Vec<AudioBuf<S, RATE_HZ, Ch>>, MediaError>;
}
```

Recommended pacing stage for telephony output:

```rust
pub trait Packetizer<const RATE_HZ: u32, Ch, S, C>
where
    Ch: ChannelLayout,
    S: LinearPcmSample,
    C: TelephonyCodec,
{
    fn push_frame(
        &mut self,
        input: PcmFrame<RATE_HZ, Ch, S>,
    ) -> Result<Vec<EncodedFrame<C>>, MediaError>;
}
```

### Required Concrete Stage Inventory

To avoid implementation drift, the first Telnyx slice should build the following concrete stages explicitly.

Inbound mandatory stages:

1. `SequencedFrameReorder<C>`
   Input: `EncodedFrame<C>`
   Output: `EncodedFrame<C>`
   Contract: reorder by provider-neutral per-track sequence, reject unbounded gaps after timeout. Telnyx `media.chunk` is consumed earlier by `bins/telnyx-gateway` and must not appear in this provider-neutral stage.
2. `G711Decoder<Pcmu, 8_000, Mono, i16>` and `G711Decoder<Pcma, 8_000, Mono, i16>`
   Contract: decode encoded RTP payload bytes into linear PCM, no base64 or RTP header parsing inside this stage.
3. `L16Decoder<16_000, Mono, i16>`
   Contract: reinterpret Telnyx `L16` payload bytes as linear PCM without codec compression loss.
4. `MonoNormalizer<R, Ch, E>`
   Contract: collapse multi-channel PCM to mono if ever required; current telephony expectation is mono passthrough.
5. `PcmResampler<8_000, 16_000, Mono, i16>` and related variants
   Contract: produce deterministic resampled output, preserve end-of-stream, document buffering. The i16 telephony stage should wrap the existing f32 `Resampler` path internally until an i16-native resampler exists.
6. `SherpaChunker` or `MoonshineChunker`
   Contract:
   Sherpa path emits ordered `AudioBuf<i16, 16_000, Mono>` values suitable for immediate `TranscriptionSession::ingest()`;
   Moonshine path emits fixed `1280`-sample windows.

Outbound mandatory stages:

1. `TtsDrain<R, Mono, E>`
   Contract: consume `SpeechStream::next_chunk()` until exhaustion or interruption and expose typed PCM frames.
2. `PcmFormatConverter<R, Mono, f32, i16>` where needed
   Contract: used for Qwen3-TTS outbound adaptation before transport encoding if the packetizer or codec path expects signed 16-bit PCM.
3. `PcmResampler<22_050, 16_000, Mono, i16>` for Piper and `PcmResampler<24_000, 16_000, Mono, _>` for Qwen3-TTS
   Contract: make outbound target rate explicit instead of inferred.
4. `TelephonyPacketizer<16_000, Mono, i16, L16>` or G.711 variant
   Contract: emit telephony-sized packets rather than full TTS chunks.
5. `G711Encoder` or `L16Encoder`
   Contract: produce outbound Telnyx payload bytes only after packetization.

The first implementation should not merge these responsibilities into one opaque “audio adapter” component.

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
- must use anti-alias filtering for telephony live paths, especially downsampling and non-integer-rate conversions such as `22_050 Hz -> 16_000 Hz`
- must replace or wrap the existing `motlie_voice::pipeline::resample::LinearInterpolator` before any live Telnyx acceptance test
- i16 telephony resampler stages should wrap the existing f32 `Resampler` implementation as `i16 -> f32 -> resample_f32 -> i16` until the crate has a dedicated i16 implementation
- `motlie_voice::pipeline::convert` already has `f32_to_i16_clamped`; add the missing provider-neutral `i16_to_f32` helper before wiring the i16 wrapper
- may use a mature external crate such as `rubato` or `dasp`, or a documented in-tree polyphase/windowed-sinc implementation

Sample converter:

- must not change sample rate or channels
- must document clipping, rounding, and saturation behavior

Chunker:

- must produce deterministic ASR input ordering and expose sequence metadata where the gateway needs logs or jitter metrics
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
- Whisper can satisfy the typed batch ASR interface, but it likely needs a different buffering policy and should not reuse the low-latency streaming defaults blindly.
- Piper emits `AudioBuf<i16, 22_050, Mono>`, while Qwen3-TTS.cpp emits `AudioBuf<f32, 24_000, Mono>`; the outbound side must adapt from the actual emitted type, not from a hardcoded assumption.

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

### Pipeline Controller Contracts

The low-level `Stage` contracts should compose into two higher-level controllers. These controllers are what the call handlers use; they should hide codec/reorder/resample details without hiding provider identity.

Inbound ASR pipeline:

```rust
#[async_trait]
pub trait InboundAsrPipeline: Send {
    type TransportFrame;

    async fn ingest_transport_frame(
        &mut self,
        frame: Self::TransportFrame,
    ) -> Result<Vec<TranscriptEvent>, PipelineError>;

    async fn finish(self) -> Result<Option<TranscriptEvent>, PipelineError>;
}
```

Outbound TTS pipeline:

```rust
#[async_trait]
pub trait OutboundTtsPipeline: Send {
    async fn speak_text(
        &mut self,
        text: String,
    ) -> Result<SpeechPlaybackHandle, PipelineError>;

    async fn interrupt(&mut self) -> Result<(), PipelineError>;
}
```

For Telnyx, `TransportFrame` is created in `bins/telnyx-gateway` from WebSocket `media` events after base64 decoding and `media.chunk` to generic sequence mapping. The reusable pipeline stages after that point remain in `libs/voice`.

### Recommended Safety Properties

The pipeline design should make these bugs impossible or very obvious:

- feeding encoded G.711 bytes directly into an ASR chunker
- resampling a payload that is still compressed instead of decoded
- encoding outbound telephony packets from the wrong PCM rate
- forgetting whether a given frame is `S16Le` or `F32Le`
- guessing whether a stage is operating on `8 kHz` or `16 kHz`

The type system does not replace runtime validation, but it should prevent “everything is bytes” assembly mistakes.

### Static-Dispatch Model Injection

The gateway must not hardcode `sherpa-onnx`, `whisper.cpp`, Piper, Qwen3-TTS.cpp, or any other concrete backend. It also should not use `dyn` at the gateway boundary, because the supported model universe is known at build time through Cargo features.

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
  - `model-qwen3-tts-cpp`

Examples:

```bash
cargo build --release -p telnyx-gateway \
  --no-default-features \
  --features "model-sherpa-onnx-streaming model-piper-en-us-ljspeech-medium"

cargo build --release -p telnyx-gateway \
  --no-default-features \
  --features "model-moonshine-streaming model-qwen3-tts-cpp"
```

Design implication:

- a Telnyx deployment should compile only the ASR/TTS bundles it intends to operate
- the gateway binary should fail at compile time if the required model feature set is not enabled
- when multiple ASR or TTS bundles are compiled in, selection should be done via closed enums from `motlie_models::AsrModels` and `motlie_models::TtsModels`, not via open-ended runtime plugin loading

### Closed-Enum Selection

For builds that include more than one ASR or TTS backend, the gateway must use closed enums rather than trait objects.

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
    #[cfg(feature = "model-qwen3-tts-cpp")]
    Qwen3TtsCpp0_6B,
}
```

These enums map cleanly onto the existing `motlie_models::AsrModels` and `motlie_models::TtsModels` surfaces.

### Current Lower-Layer Shape

The current `libs/model` typed ASR/TTS contracts and `libs/models` curated model selectors are already sufficient for the gateway surface to stay statically dispatched. The gateway should use concrete typed handles, `AsrModels` / `TtsModels`, and capability metadata for compatibility checks, following the `bins/voice-agent` pattern. It should not reopen the erased ASR/TTS adapter path that PR #201 removed.

That should be documented explicitly so the design does not pretend the existing lower-level loader surface is already fully monomorphized.

### Conversation Handler Contract

Between ASR transcript output and TTS response input, the gateway should expose a dedicated pluggable processing stage.

Recommended conversational pipeline:

```text
Inbound audio
-> ASR
-> TranscriptEvent
-> ConversationHandler
-> ConversationCommand
-> OutboundSpeechController / CallAction mapping
```

Recommended Rust shapes:

```rust
use async_trait::async_trait;
use futures_core::Stream;
use std::collections::BTreeMap;

use motlie_voice::telephony::CallAction;

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

pub enum ConversationCommand {
    Say { text: String },
    Call(CallAction),
    Noop,
}

#[async_trait]
pub trait ConversationHandler: Send + Sync {
    type CommandStream: Stream<Item = Result<ConversationCommand, ConversationError>> + Send;

    async fn on_transcript(
        &mut self,
        event: TranscriptEvent,
        context: &mut ConversationContext,
    ) -> Result<Vec<ConversationCommand>, ConversationError>;

    async fn on_transcript_streaming(
        &mut self,
        event: TranscriptEvent,
        context: &mut ConversationContext,
    ) -> Self::CommandStream;
}
```

This is the main extensibility point for:

- LLM inference
- business logic
- RAG lookups
- workflow integrations
- deterministic rule engines

The transport and speech stack stay reusable infrastructure. The handler is the application-specific component. A simple handler may emit `ConversationCommand::Say { text }`; a deterministic test handler may emit canned responses; a tmux-backed handler may turn transcript text into `motlie_tmux::KeySequence` send-keys operations instead of speech.

### Streaming Conversation Responses

The handler should also support a streaming mode for lower latency so partial text can begin TTS before the full response is ready.

Recommended extension path:

```rust
use futures_core::Stream;

#[async_trait]
pub trait ConversationHandler: Send + Sync {
    type CommandStream: Stream<Item = Result<ConversationCommand, ConversationError>> + Send;

    async fn on_transcript(
        &mut self,
        event: TranscriptEvent,
        context: &mut ConversationContext,
    ) -> Result<Vec<ConversationCommand>, ConversationError>;

    async fn on_transcript_streaming(
        &mut self,
        event: TranscriptEvent,
        context: &mut ConversationContext,
    ) -> Self::CommandStream;
}
```

The exact chunking policy should remain gateway-local. The handler returns commands or command fragments; the gateway decides when text is stable enough to synthesize and how to pace outbound telephony packets.

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

- Telnyx v1 should treat upstream `sherpa-onnx` streaming Zipformer models as the default recommended ASR family for the real-time conversational flow
- the live gateway default is `kroko-2025` (`sherpa-zipformer-en-kroko-2025-08-06`) because the M1.5 A/B results show it is the balanced cross-domain choice: call-center WER is `14.1% / 13.9%` on `L16-16k / PCMU-8k`, and PM/technical WER is `14.0% / 14.0%`
- `sherpa-2023` (`sherpa-zipformer-en-2023-06-26`) remains the recommended profile for call-center-only deployments: it measured `10.5% / 12.0%` on the call-center corpus but `19.8% / 22.1%` on the PM/technical corpus
- operators and agents can switch the next-call backend between `kroko-2025` and `sherpa-2023` with `asr use <backend>`; switching is source-local and takes effect only for calls answered or dialed after the command
- the gateway itself should still accept any injected typed ASR that implements `StreamingTranscriber`
- gateway live-test and deployment runbooks must use the upstream `sherpa-onnx` Rust crate for Sherpa; Cargo downloads and statically links the upstream prebuilt `sherpa-onnx` native archive, including the ONNX Runtime library Sherpa uses internally
- gateway runbooks must not set `ORT_LIB_PATH`, `ORT_PREFER_DYNAMIC_LINK`, or `LD_LIBRARY_PATH`, and must not require building ONNX Runtime from source or vendoring ONNX Runtime

Milestone 1.5 (#371) keeps ASR quality work inside the Sherpa ecosystem. Current Sherpa model docs list multiple English-capable online transducer candidates, including `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26`, `2023-06-21`, `2023-02-21`, `en-20M-2023-02-17`, and an English LSTM transducer. They also list a newer English `sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25` path under Nemotron ASR Streaming. M1.5 should add curated catalog entries only after replay WER shows a candidate is worth keeping.

Recommended Sherpa-only quality order:

1. Keep the current upstream Rust `sherpa-onnx` streaming Zipformer path as the baseline because it is already integrated and live-call validated.
2. Add hotword/context-bias configuration for transducer models first. Sherpa documents hotwords as transducer-only; expose hotwords file/buffer and score without hiding raw ASR output.
3. Benchmark current Zipformer, alternate English Zipformer variants, and the English Nemotron streaming model on the same capture corpus before changing the default.
4. Treat online Paraformer and online CTC families as secondary for the initial English telephony target unless the curated model catalog identifies a live English candidate with equal or better streaming support.
5. Use offline Sherpa/NeMo models, including Parakeet-family artifacts, only for post-call second-pass comparison or review until a live streaming configuration is selected and proven.

### Recommended TTS: Piper

Piper is the recommended Telnyx v1 TTS backend because it is the simplest stable outbound TTS target in Motlie and provides the TTS half of the first complete Sherpa + Piper duplex pairing.

Current evidence from Motlie's implementation status:

- short utterance generation: about `70 ms`
- paragraph generation: about `3.6 s`
- stable working output today
- current typed Piper synthesis produces a full PCM buffer before the returned `SpeechStream` exposes chunked output

Why Piper is good enough for Telnyx v1:

- short telephony responses are the common case
- `~70 ms` generation for short responses is fast enough for conversational turn taking
- output is already PCM, which fits the gateway normalization and re-encoding pipeline cleanly

Current Piper limitations:

- no voice cloning
- one pre-trained voice per model
- less flexibility than future expressive or clone-capable TTS stacks
- requires a system `libespeak-ng` installation plus phonemization data; the gateway should auto-discover common `espeak-ng-data` paths and otherwise fail loudly before sending silent or empty TTS audio
- CUDA execution is currently gated by issue #230: Piper defaults to CPU-only ONNX Runtime for shutdown stability on the GB10 validation host, and advanced users can opt back into CUDA probing with `MOTLIE_PIPER_ALLOW_CUDA=1`
- gateway live-test and deployment runbooks must use the same static ONNX Runtime linkage policy for Piper once outbound TTS is enabled; dynamic ONNX Runtime linkage is not an accepted operator path

Current TTS status:

- Piper is the required milestone 2 outbound TTS backend
- Qwen3-TTS.cpp is available through the typed TTS surface, but it emits `f32` `24 kHz` audio and remains a follow-on Telnyx pairing rather than part of the first Sherpa + Piper duplex milestone
- Fish Speech was a historical rejected candidate and is not part of the current curated Telnyx plan unless a future backend is reintroduced and validated against the typed TTS contracts

Recommended design rule:

- Telnyx v1 should standardize on Piper as the default recommended TTS backend and defer richer voice-selection work until the outbound TTS milestone is stable
- the operator command surface should still expose `tts list`, `tts status`, and `tts use piper` so TTS follows the same discoverable pattern as ASR; `tts status` must clearly report `unavailable` when the binary was built without the `piper` feature
- the gateway itself should still accept any injected typed TTS that implements `SpeechSynthesizer`

### Transport Streaming vs Incremental TTS

The Telnyx gateway streams media over the transport, but the current Motlie TTS backends should not be treated as incremental audio generators.

Current backend behavior:

- Piper synthesizes a full `AudioBuf<i16, 22_050, Mono>` first, then exposes chunked output through the typed stream adapter.
- Qwen3-TTS.cpp similarly produces a full `AudioBuf<f32, 24_000, Mono>` before chunked draining.

Design implications:

- outbound Telnyx packetization can stream chunks after synthesis completes, but first audio cannot leave the gateway until the backend has produced the initial buffer
- `ConversationHandler` streaming text can reduce text-decision latency, but it does not by itself make Piper or Qwen3-TTS.cpp produce incremental audio
- latency validation must measure the actual first-audio time of the selected backend, not only WebSocket packet cadence

### Latency Budget

Recommended conversational target:

- Telnyx inbound audio chunk cadence: about `20 ms` for G.711 at `8 kHz`
- ASR processing: `sherpa-onnx` chunk-by-chunk streaming
- TTS generation: about `70 ms` for short Piper responses, with full-buffer synthesis before transport packetization
- total round-trip target: under `500 ms`

Inference from the current stack:

- the main latency risks are not model startup or raw short-utterance TTS generation
- the main risks are transport buffering, resampling, end-of-turn detection, and oversized outbound packetization

Design implications:

- prefer `sherpa-onnx` over `whisper.cpp` for any real-time Telnyx path
- keep inbound jitter buffers small
- keep TTS responses concise when possible
- packetize outbound audio at telephony-friendly intervals instead of large queued chunks
- measure Piper in the actual deployment mode; by default it is CPU-only until #230 is resolved or `MOTLIE_PIPER_ALLOW_CUDA=1` is deliberately enabled
- prefer RTP mode over MP3 mode to avoid avoidable playback delay

## Inbound Call Handler Design

### Control Flow

Recommended inbound flow:

1. Gateway starts and listens on the configured HTTP/WebSocket bind address, defaulting to `127.0.0.1:8080` for local development.
2. For milestone 1, the operator uses the TUI command source to provision/select the Telnyx application, set the public webhook/media URLs, bind a phone number, and enable the desired inbound mode. Milestone 4 may expose the same setup/control path through the socket, Control API, and harness appserver.
3. Telnyx sends `call.initiated` webhook.
4. HTTP handler verifies signature, parses `call_control_id`, allocates a `PendingCallSession`, renders the pending call as a highlighted row in the top call roster, and updates the selected-call detail pane when selected. In milestone 4, the same state transition also emits `call.inbound.pending` to matching application webhook subscriptions.
5. If inbound mode is `Disabled`, the gateway must not answer. It may log the event and return `200 OK`, or reject the call if the operator configured reject-on-disabled behavior.
6. If inbound mode is `Manual`, the gateway waits for `answer <call>` from a configured operator command source. In milestone 4, an authenticated application server may request the same transition with `POST /api/v1/calls/{call_id}/answer`.
7. If inbound mode is `AutoTranscribe`, the gateway behaves as if a trusted controller immediately requested answer with a configured transcript sink.
8. The answer action attaches media streaming with:
   - `stream_url=wss://.../telnyx/media`
   - `stream_track=inbound_track`
   - `stream_codec=PCMU` for the milestone 1 default live path, or `L16` for an explicit ASR quality comparison run
   - `stream_bidirectional_mode=rtp`
   - `stream_bidirectional_codec` matching the requested stream codec
   - `stream_bidirectional_sampling_rate=8000` for `PCMU`, or `16000` for `L16`
   - `stream_bidirectional_target_legs=self`
9. Telnyx opens the WebSocket.
10. On `start`, the gateway finalizes session media metadata and opens a typed `StreamingTranscriber` session.
11. Until milestone 2/M3 supplies real outbound TTS audio, the gateway sends silence `media` frames matching the observed/requested bidirectional RTP codec back over the WebSocket as a keepalive so the single-leg PSTN call does not terminate while Motlie is receive-only at the application level.
12. Each inbound `media` event is mapped to provider-neutral sequence metadata, reordered, decoded, converted to normalized PCM, and pushed into the ASR stream.
13. The gateway passes the replay-sized trailing-silence pad through the active ASR session, then locally finishes the session and waits for the next speech-energy frame; short pauses below that threshold do not force a new ASR session.
14. If Sherpa emits a repeated-token hallucination such as a growing `Q` run, the gateway suppresses that event from operator-visible transcripts, logs it, resets the ASR session, and waits for the next speech-energy frame before feeding the backend again.
15. ASR updates are converted to `TranscriptEvent` values and sent to the configured `TranscriptSink`.

Milestone 1 quality debugging should capture replayable artifacts for accepted media streams when the operator enables a capture directory. Each call/stream directory should contain raw Telnyx media JSONL, decoded inbound WAV at the observed codec/sample rate, the `16 kHz` WAV actually fed into Sherpa after gating/resampling, transcript-event JSONL, and a manifest with call ids, stream id, observed codec, and sample rate. WAV captures must be finalized with finite RIFF/data sizes so normal tooling can read duration and sample count, while Motlie's permissive streaming decoder may still read old indefinite-length captures. Capture failures should warn but must not fail the live call.
16. In milestone 1, a sink such as `TuiTranscriptSink` or `StdoutTranscriptSink` emits transcripts and returns no call-control actions; `WebhookTranscriptSink` is milestone 4 external-integration work.
17. In milestone 3, a sink can forward final transcript events to a `ConversationHandler`, then route resulting `ConversationCommand::Say { text }` values to outbound TTS.
18. On hangup or `stop`, the gateway finishes the ASR stream and tears down the session.

The inbound handler should therefore be ASR-first. It must not require outbound TTS or a conversation handler to be useful.

### Inbound Handler Surface

Recommended minimal inbound handler shape:

```rust
pub struct InboundCallConfig<Sink> {
    pub transcript_sink: Sink,
    pub stream_track: StreamTrack,
    pub preferred_codec: TelnyxCodec,
    pub preferred_sample_rate_hz: u32,
}

pub struct InboundCallHandler<A, Sink> {
    pub asr: A,
    pub transcript_sink: Sink,
    pub sessions: SessionRegistry,
}
```

The handler owns the inbound call lifecycle and ASR session. `TranscriptSink` owns what happens to transcript updates. That allows:

- stdout logging for milestone 1
- tmux `send_keys` integration for interactive workflows
- fixture sinks for tests
- a conversation bridge for milestone 3

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
- `InboundPipeline` is a typed composition such as sequence-map -> reorder -> decode -> normalize -> chunk
- `OutboundPipeline` is a typed composition such as convert -> resample -> packetize -> encode
- `AsrStreamRuntime` and `TtsStreamRuntime` are concrete wrappers around the currently selected runtime handles

This keeps Telnyx-specific types outside the model crates while allowing ASR, TTS, and conversation logic to remain transport-agnostic.

### Mapping Telnyx Frames to Typed ASR Input

Recommended mapping:

1. Parse the WebSocket `start` frame.
2. Read `media_format` and instantiate the matching decoder.
3. In `bins/telnyx-gateway`, base64 decode each `media.payload`, map Telnyx `media.chunk` to provider-neutral per-track sequence metadata, and create `EncodedFrame<C>` values.
4. In `libs/voice`, reorder those sequenced encoded frames before codec decode.
5. For each ordered audio payload:
   - decode from Telnyx codec to linear PCM
   - resample to `16 kHz` if needed
   - convert to the ASR runtime's typed input, such as `AudioBuf<i16, 16_000, Mono>`
6. Call `TranscriptionSession::ingest()` with each ordered typed audio frame.
7. If `ingest()` returns `Some(update)`, convert it to `TranscriptEvent` and forward it to the configured `TranscriptSink`.
8. On WebSocket `stop`, call `finish()`.

Sequence mapping detail:

- Telnyx `sequence_number` is at the event level
- Telnyx `media.chunk` is specific to media sequencing
- the gateway needs provider-neutral sequence metadata before reorder and a monotonic post-reorder frame counter for logs, jitter metrics, and deterministic test fixtures even though the current typed ASR input is an `AudioBuf`

Recommended gateway rule:

- `bins/telnyx-gateway` maps `media.chunk` per track to provider-neutral sequence metadata
- `libs/voice` reorders by that generic sequence before codec decode
- emit a gateway-local `u64` sequence counter after reordering for logs and deterministic fixtures

This isolates Telnyx transport quirks from the ASR contract.

### Feeding the Conversation Handler

The gateway should not bake dialog policy into the Telnyx crate. Instead:

- ASR partials and finals are converted into transcript text events
- transcript events are delivered to an injected `TranscriptSink`
- simple sinks log or forward transcripts without TTS
- a conversation sink forwards selected transcript events to a `ConversationHandler`
- the handler reads and mutates `ConversationContext`
- the handler returns provider-neutral conversation commands
- the gateway maps `ConversationCommand::Say` to TTS and `ConversationCommand::Call` to call-control actions

This is the cleanest seam for integrating Motlie-specific agent logic later.

### Returning TTS Audio

When the application wants to speak:

1. select a typed `SpeechSynthesizer`
2. allocate a per-call speech job with a cancellation token and monotonically increasing playback/mark id
3. split long text into sentence- or clause-sized synthesis requests so one large buffer does not monopolize the call
4. call `synthesize()` with each `SynthesisRequest`
5. read `next_chunk()` until exhaustion or interruption
6. concatenate the returned Piper PCM chunks for the utterance before resampling so the `22.05 kHz -> telephony-rate` resampler state is continuous across sentence/clause boundaries
7. normalize PCM to the configured outbound codec and sampling rate
8. packetize into fixed `20 ms` Telnyx RTP media frames, padding only the final frame if needed
9. enqueue frames onto the per-call outbound media `mpsc`

The WebSocket-owning media task is the only task that reads from or writes to the Telnyx media socket. Its loop must concurrently:

- read inbound `media` frames and feed ASR
- drain outbound TTS frames from the per-call `mpsc` at real-time packet cadence, one `20 ms` frame per `20 ms` wall-clock tick
- send Telnyx `mark` events when a speech job has fully queued
- handle Telnyx `mark`, `clear`, `stop`, and `dtmf` events
- send silence keepalive frames only when no speech job is active; do not inject silence into a momentary active-TTS queue dip
- log outbound frame send intervals, queue depth, and active-speech underrun ticks so live breakup can be diagnosed from structured logs

`speak` must therefore be non-blocking from the operator command perspective: the command starts or enqueues a cancellable speech job and returns a playback id/status while the socket task continues inbound ASR. Milestone 2 may ignore the inbound transcript during playback, but it must not pause the read loop or close the ASR session while TTS is active.

If the caller barges in or the operator runs `speak cancel` while TTS is active:

- stop reading the current `SpeechStream`
- call `finish()` to release backend resources
- drop queued local outbound frames for that speech job
- send Telnyx `clear` over the same bidirectional media WebSocket so provider-side buffered media is flushed
- keep inbound ASR active and continue the call unless a separate hangup action was requested

### Why `stream_track=inbound_track`

Motlie ASR should usually ingest only caller speech. Streaming both tracks by default would mix:

- remote caller audio
- Motlie-generated TTS audio

That would degrade ASR and complicate turn-taking. `both_tracks` should be reserved for diagnostics or analytics, not the default conversational path.

For milestone 2 and milestone 3, `stream_track=inbound_track` is also the guard that lets inbound ASR stay live during TTS playback without Motlie transcribing its own synthesized audio.

### M2-Safe Bidirectional Media Contract

Milestone 2 outbound TTS must use the single bidirectional Media Streaming WebSocket already attached to the call. It must not use Telnyx Call Control `playback_start`, hosted `speak`, or any separate playback API as the primary Motlie TTS path.

The per-call media task owns:

- the Telnyx WebSocket read half
- the Telnyx WebSocket write half
- the outbound frame receiver
- the currently active speech job metadata
- Telnyx `mark` correlation state

The TTS synthesis task owns:

- the selected typed TTS backend
- text chunking into sentence/clause requests
- conversion from backend audio to outbound telephony frames, with all chunks for one utterance concatenated before resampling so boundaries do not reset the resampler
- cancellation checks before and while enqueuing frames

The media task must be the pacer. Producers may fill the per-call outbound queue quickly, but only the WebSocket-owning task sends media frames to Telnyx, and it sends them on the fixed `20 ms` cadence. When a speech job is active and the queue is empty, the task records an underrun and withholds keepalive silence instead of splicing silence into the utterance. Silence keepalive resumes only after the speech job completes, fails, or is canceled.

This split keeps milestone 3 safe: barge-in policy can decide when to call `speak cancel`, but it does not need to replace the transport, ASR read loop, or outbound packet writer.

## Outbound Call Handler Design

### Control Flow

Recommended outbound flow:

1. For milestone 2, the operator or local agent runs `dial <phone-or-sip-uri>` from the TUI shell or Unix-domain socket command source. In milestone 4, an external application may request the same transition with `POST /api/v1/calls`, optionally after selecting a default `from` number.
2. Gateway looks up the selected Telnyx application/connection, public media URL, caller ID, and outbound stream defaults from `GatewayRuntimeState`.
3. Gateway verifies that a Telnyx outbound caller ID and outbound-capable Voice Profile are configured. If Telnyx returns an account-side outbound error such as `403 D38`, the operator-visible status and logs must surface that prerequisite clearly.
4. Gateway issues `POST /v2/calls`.
5. The Telnyx dial request includes:
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
6. Telnyx establishes the WebSocket.
7. The gateway starts the outbound media session on `start`, opens inbound ASR for caller speech, creates the outbound frame channel, and renders call state in the roster/detail panes.
8. Once the callee answers, the operator or socket client may run `speak <text...>` to synthesize outbound audio. In milestone 4, an external application may request the same action with `POST /api/v1/calls/{call_id}/say`.
9. The rest of the session uses the same media path as inbound calls: ASR read loop live, outbound TTS frames written by the socket task, and cancellation using local queue flush plus Telnyx `clear`.

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
pub struct TelnyxOutboundDial {
    pub to: String,
    pub from: String,
    pub connection_id: String,
    pub stream_url: String,
    pub media: TelnyxMediaConfig,
}

impl TelnyxGateway {
    pub async fn start_outbound_call(
        &self,
        request: TelnyxOutboundDial,
    ) -> Result<CallHandle, GatewayError> {
        // 1. allocate session id
        // 2. call Telnyx /v2/calls with inline streaming params
        // 3. wait for websocket start
        // 4. select the returned call for the command source that requested it
        todo!()
    }
}
```

### Driver REPL Dialer Surface

The outbound product surface should be the shared operator `motlie-driver` command family described above, not a separate dialer REPL. The command family owns parsing and source-local operator session state; the gateway owns call and media execution. Every M2 command must work through both the TUI shell and the Unix-domain agent socket.

Recommended command family:

```text
dial <phone-or-sip-uri> [--from <+e164>]
speak [call-id] <text...>
speak cancel [call-id]
hangup [call-id]
status [call-id]
```

Recommended driver state:

```rust
pub struct TelnyxDialerState<C> {
    pub controller: C,
    pub active_call: Option<CallHandle>,
}
```

`dial` stores the active `CallHandle` in the command source that requested it. `speak` sends text to the active call through `OutboundSpeechController::speak()`. `speak cancel` interrupts the active speech job through the same controller and WebSocket `clear` path. `hangup` terminates the active call. `status` returns the source-local selected call plus call/media/TTS state so a socket-driven agent can operate without scraping the TUI.

The same controller should also accept text from non-REPL sources. The Gateway Control API, mstream bridge, broadcast command, fixture replay, or tmux-driven test should not need a second TTS pathway; each source should produce text and call the same outbound controller.

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
use motlie_voice::telephony::{CallAction, DtmfDigit};

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
use motlie_voice::telephony::{CallAction, IvrEvent};

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

- a private Tailscale-connected DGX host using Tailscale Funnel
- a private Tailscale-connected mac mini using Tailscale Funnel
- an ngrok tunnel as an alternate local-development path

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

### Option 1: Tailscale Funnel

Tailscale Funnel is the default documented v1 option because it fits private-host development and can provide a stable public HTTPS endpoint.

Current documented behavior from Tailscale:

- Funnel exposes a local service on a public `https://<node>.<tailnet>.ts.net` URL
- it requires MagicDNS, HTTPS certificates, and Funnel permission in the tailnet
- it only supports public HTTPS exposure on ports `443`, `8443`, and `10000`

Recommended v1 Funnel workflow:

1. run the gateway locally on the private host
2. expose it with `tailscale funnel`
3. use the resulting `https://<node>.<tailnet>.ts.net` URL for Telnyx webhooks
4. use the matching `wss://<node>.<tailnet>.ts.net/...` URL for Telnyx media streaming

### Option 2: ngrok

ngrok remains a useful disposable fallback, but examples in this design assume Tailscale Funnel.

Current documented behavior from ngrok:

- `ngrok http <port>` creates a public HTTPS endpoint to a local HTTP service
- HTTP/S endpoints support WebSockets out of the box
- the agent establishes outbound TLS connections; no inbound port opening is required

Recommended fallback ngrok workflow:

1. run the gateway locally on a private host
2. start `ngrok http <gateway-port>`
3. use the generated `https://...ngrok.app` URL for Telnyx webhooks
4. use the matching `wss://...ngrok.app/...` URL for Telnyx media streaming
5. update the Telnyx application webhook URL whenever the ngrok hostname changes

### Gateway Configuration Requirement

The gateway binary should accept a `--listen` flag for the local bind address, a `--load <dump_path>` flag for replaying durable gateway state, a `--tui` flag for the local TUI, and a `--socket <path>` flag for a Unix-domain operator command socket. The external public URLs can be provided either as startup flags or through any configured operator command source after a tunnel starts, and then persisted with `state dump` or `shutdown <dump_path>`.

Example shape:

```text
telnyx-gateway \
  --listen 127.0.0.1:8080 \
  --media-path /telnyx/media \
  --webhook-path /telnyx/webhooks \
  --tui \
  --socket /tmp/motlie-telnyx-gateway.sock \
  --load ./telnyx-gateway.state.repl
```

For headless operation, omit `--tui` and keep `--socket`:

```text
telnyx-gateway \
  --listen 127.0.0.1:8080 \
  --media-path /telnyx/media \
  --webhook-path /telnyx/webhooks \
  --socket /run/motlie/telnyx-gateway.sock \
  --load ./telnyx-gateway.state.repl
```

Then use either the TUI REPL or a local socket client to send the same command text:

```text
config set webhook-url https://motlie-gateway.example.ts.net/telnyx/webhooks
config set media-url wss://motlie-gateway.example.ts.net/telnyx/media
config set state-path ./telnyx-gateway.state.repl
```

This matters because the process may listen on `127.0.0.1:8080` while Telnyx needs the externally reachable tunnel URL, not the private bind address.

### Recommended v1 Dev Workflow

1. start the gateway on the private host
2. start Tailscale Funnel
3. use the TUI REPL or local command socket to set the public URLs
4. use the TUI REPL or local command socket to create/select the Telnyx application and bind the phone number
5. run `inbound enable --manual`, then make or receive calls
6. exit with `shutdown ./telnyx-gateway.state.repl`
7. restart later with `telnyx-gateway --load ./telnyx-gateway.state.repl ...`

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

#### 3. Create or select a Call Control Application from an operator command source

For Motlie's Telnyx gateway, prefer a Call Control Application over TeXML because the design depends on programmable voice webhooks plus media streaming commands.

Preferred gateway command flow:

```text
telnyx app create motlie-telnyx-local
telnyx app use <connection-id>
```

This returns the application `id`, which is also the `connection_id` used for outbound calls.

#### 4. Configure public webhook and media URLs from an operator command source

With Tailscale Funnel, use your public Funnel hostname:

```text
config set webhook-url https://motlie-gateway.example.ts.net/telnyx/webhooks
config set media-url wss://motlie-gateway.example.ts.net/telnyx/media
telnyx app webhook set https://motlie-gateway.example.ts.net/telnyx/webhooks
```

#### 5. Assign the phone number to the application from an operator command source

Telnyx phone numbers carry a `connection_id`. Set that to the Call Control Application ID.

```text
telnyx number list
telnyx number use +13125551234
telnyx number bind +13125551234 <connection-id>
```

The right status area should show the Telnyx API request status and the currently selected application/number.

### Local Host Setup

#### 6. Build the gateway binary

```bash
cargo build --release -p telnyx-gateway
```

#### 7. Download model artifacts

Download the artifacts for the models you want to run.

Recommended v1 defaults:

- Piper voice model
- `sherpa-onnx` Zipformer2 streaming model

#### 8. Start the gateway with model paths and listen address

```bash
./target/release/telnyx-gateway \
  --listen 127.0.0.1:8080 \
  --webhook-path /telnyx/webhooks \
  --media-path /telnyx/media \
  --tui \
  --socket /tmp/motlie-telnyx-gateway.sock \
  --load ./telnyx-gateway.state.repl \
  --telnyx-api-key "$TELNYX_API_KEY" \
  --asr-model sherpa-onnx \
  --tts-model piper
```

The exact model configuration flags can evolve, but the binary should always separate:

- local listen address
- externally reachable webhook and media URLs
- injected ASR/TTS backend choice
- operator input mode: `--tui`, `--socket <path>`, both, or neither when only HTTP Control API and loaded configuration are desired
- Telnyx application/number selection, which can be driven from any configured operator command source
- optional replayable state file loaded with `--load`

#### 9. Start Tailscale Funnel

```bash
tailscale funnel 8080
```

#### 10. Configure the Tailscale Funnel public URLs through an operator command source

Use the Funnel hostname assigned to this node:

```text
config set webhook-url https://motlie-gateway.example.ts.net/telnyx/webhooks
config set media-url wss://motlie-gateway.example.ts.net/telnyx/media
telnyx app webhook set https://motlie-gateway.example.ts.net/telnyx/webhooks
```

### First Call Test

#### 11. Enable inbound manual mode and call the Telnyx number from a phone

```text
inbound enable --manual
```

Place an inbound call to the purchased number assigned to the Call Control Application. The call should appear as a highlighted pending row in the top call roster.

#### 12. Answer and verify WebSocket media connected

```text
calls
answer <call>
```

Confirm in gateway logs that:

- `call.initiated` arrived
- the pending call appeared in the TUI
- the call was answered
- the Telnyx WebSocket `start` event connected

#### 13. Speak and verify ASR transcript appears in the selected-call detail pane

Speak into the call and verify transcript text appears in gateway logs, tracing output, or the selected-call detail pane.

#### 14. Optional: attach conversation or TTS after later milestones

Milestone 1 stops at transcription. TTS playback belongs to milestone 2 outbound calls and milestone 3 connected conversation.

### Outbound Call Test

#### 15. Use the gateway command surface to initiate an outbound call

```text
dial +1234567890
say hello from motlie
```

#### 16. Same verification as inbound

Confirm:

- outbound call connects
- WebSocket media starts
- `say` generates TTS audio
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

Current Motlie already has the beginning of this layer in `motlie-voice`: `PcmFrame<const RATE_HZ, C, E>`, WAV sample decoding, downmixing, `f32_to_i16_clamped`, and a f32-oriented `Resampler` trait with a `LinearInterpolator` placeholder. The Telnyx gap is not ownership of a new crate; it is production-grade telephony adaptation inside the existing crate.

Required gateway utilities:

- `8 kHz` -> `16 kHz` upsampling for PSTN-originated `PCMU` / `PCMA`
- backend-output resampling from TTS-native rates to Telnyx outbound rate
- `f32` <-> `s16le` conversion as needed, including a new provider-neutral `i16_to_f32` helper to pair with the existing `f32_to_i16_clamped`
- reuse of `motlie_voice::frame::PcmFrame<const RATE_HZ, C, E>` so rate, channels, and sample type remain explicit during assembly
- replacement or wrapping of `LinearInterpolator` with anti-aliased resampling before live telephony
- i16 telephony resampler wrappers that preserve i16 stage boundaries while using the existing f32 resampler internally until an i16-native implementation is justified

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

Typed ASR sessions expect ordered audio. Telnyx explicitly documents that media event order is not guaranteed.

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
- optional timestamp metadata around typed `AudioBuf` frames if future transports need precise wall-clock alignment
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

### Alternative 2b: Telnyx Call Control `playback_start` / Hosted `speak`

Decision: reject for milestone 2 Motlie-generated TTS.

Pros:

- simple REST-level command path
- useful for fixed hosted prompts outside the Motlie media loop

Cons:

- introduces a second audio source outside the gateway's bidirectional media socket
- makes `speak` blocking or queue-oriented unless extra cancellation glue is added
- prevents the inbound ASR read loop from being the single always-live source of caller speech during playback
- would force milestone 3 to replace transport before adding barge-in policy

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
- Telnyx adapter tests that map out-of-order `media.chunk` values into provider-neutral sequence metadata
- provider-neutral reorder-buffer tests over generic sequenced frames
- resampler tests for `8 kHz` inbound to `16 kHz` normalized PCM and `22_050 Hz` Piper output to the selected outbound telephony rate, using the anti-aliased Telnyx resampler implementation
- provider-free media adaptation tests that feed generic sequenced `PCMU` / `L16` payloads through reorder, decode, normalize, resample, and a fake `StreamingTranscriber<Input = AudioBuf<i16, 16_000, Mono>>`
- compile-fail tests proving wrong rate, wrong channel layout, wrong sample type, and wrong stage order fail before live testing
- end-to-end simulated call tests that feed media frames into the typed ASR session path
- loopback tests that synthesize TTS and verify outbound `media` frames
- env-gated integration tests against a real Telnyx test number after local simulation passes

## Open Concerns

- The user still needs to confirm whether this work should be treated as greenfield or brownfield in the product sense. This document currently assumes brownfield.
- Telnyx documents codec options broadly, but actual inbound codec selection may depend on carrier, destination, and account configuration. Phase 1 should log observed `media_format` values in real calls before broadening codec support.
- `stream_bidirectional_target_legs=self` is the best current inference for single-leg AI-agent calls, but this should be validated on the first live call because Telnyx defaults to `opposite`.
- Piper currently defaults to CPU-only ONNX Runtime because of issue #230. Live latency numbers should record whether `MOTLIE_PIPER_ALLOW_CUDA=1` was enabled.
- Fish Speech remains a historical rejected candidate for this Telnyx plan. If a future curated backend revives that path, its native sample rate and chunk cadence must be measured against Telnyx's RTP pacing requirements before promotion.
- The exact interaction between simultaneous WebSocket media streaming and specific hosted call-control features such as `gather_using_audio` or hosted prompt playback should be validated on a live test call before those mixed modes are promoted.
- If some carriers deliver audible in-band DTMF energy even when Telnyx emits a separate `dtmf` event, the gateway should log and measure the overlap before finalizing the v1.1 suppression strategy.

## References

- Motlie `motlie-voice` seed PR: https://github.com/chungers/motlie/pull/209
- Piper CUDA teardown follow-up: https://github.com/chungers/motlie/issues/230
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
- Sherpa ONNX Rust crate docs: https://docs.rs/sherpa-onnx
- Sherpa ONNX hotwords/contextual biasing: https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html
- Sherpa ONNX online transducer models: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
- Sherpa ONNX NeMo and Nemotron models: https://k2-fsa.github.io/sherpa/onnx/nemo/index.html
- Existing ASR design: [DESIGN_ASR.md](../../../libs/models/docs/DESIGN_ASR.md)
- Existing TTS design: [DESIGN_TTS.md](../../../libs/models/docs/DESIGN_TTS.md)
- Existing transcription contracts: [transcription.rs](../../../libs/model/src/transcription.rs)
- Existing speech contracts: [speech.rs](../../../libs/model/src/speech.rs)

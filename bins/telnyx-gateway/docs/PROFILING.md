# Telnyx M6 Profiling and Turn Quality Analysis

## Status

Draft for Telnyx M6 (#418), authored by @codex-367-design.

This document specifies the profiling, logging, and tuning framework needed to analyze and improve live Telnyx voice conversation quality after M4 text-call integration and M5 conversational realism. Implementers should be able to derive event schemas, span boundaries, CLI/REPL/socket surfaces, test cases, and acceptance criteria directly from this document.

Related issues:

- Telnyx umbrella: #358
- M4 external text-call integration: #367
- M5 conversational realism: #402
- M6 turn logging, quality config, and LLM judge: #418

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-11 PDT | @codex-366-impl | Added live smoke-call recovery details: smoke-test final coalescing now has a 900 ms settle floor and active-playback hold, and outbound pacing rollups separate true underrun, append starvation, post-mark wait, and first-frame idle gaps. |
| 2026-06-11 PDT | @codex-366-impl | Captured live-call audio stabilization for PR #464: live TTS synthesis is isolated onto blocking threads, chunked TTS honors the one-chunk prebuffer default for first-audio latency, and smoke-test enablement turns barge-in off for deterministic echo validation. |
| 2026-06-11 | @codex-366-impl | Captured live-call quality fixes: deterministic assistant-echo transcript suppression before `caller.turn`, text-call ownership gating for manual `speak`, and call/TUI diagnostics for first-audio, buffer, underrun, and echo-suppression counters. |
| 2026-06-09 | @codex-m6-ds-rv | Resolved #427 pluggability follow-up: separated generic handler dispatch from smoke final coalescing, added `tts.first_chunk_max_chars` for sentence-boundary first-audio ramp experiments, and documented streaming-agent partial/voice-response contract notes. |
| 2026-06-09 | @codex-m6-ds-rv | Resolved #427 review: smoke-test final coalescing is handler-local and keyed by the ASR-session config snapshot, ASR finish padding is a separate `asr.finish_pad_ms` knob, and first-audio critical-path spans include handler/TTS time. |
| 2026-06-09 | @codex-m6-ds-rv | Updated live-call tuned defaults and TTS chunking guidance after M6 smoke-call trials: 650 ms endpoint tail, stricter speech gate, 90-char sentence-packed TTS chunks, and one-chunk prebuffer. |
| 2026-06-09 | @codex-367-design | Added M6 gap implementation notes: deferred ASR/TTS/media/barge-in spans, inbound/outbound transport rollups, live call-bound tuning commands, and operator TUI command-history recall through `motlie-driver::HistoryBuffer`. |
| 2026-06-08 | @codex-367-design | Revised after the first M6 review pass: added call-level config snapshots, monotonic/non-blocking span emission, critical-path latency accounting, pre-turn ASR join keys, privacy defaults, typed config validation, existing line-oriented socket reuse, normalized artifacts, and judge reproducibility requirements. |
| 2026-06-08 | @codex-367-design | Added the M6 profiling specification: span boundaries, deterministic `I heard` harnesses, structured turn logs, latency categories, LLM judge inputs/outputs, tunable config surfaces, and acceptance criteria. |

## Purpose

M6 answers one practical question: when a live voice turn feels slow, choppy, incomplete, or unrealistic, which layer caused it?

A single metric such as `caller_turn_to_playback_started_ms = 1800` is not enough. The gateway must distinguish:

- acoustic caller speaking time
- endpointing wait after the caller stops talking
- ASR finalization and transcript quality
- inbound RTP loss, stale frames, reordering, and jitter
- gateway dispatch overhead
- text-call WebSocket/app transport overhead
- app or LLM generation latency
- intentional harness delays such as tmux quiet-window waits
- queue waits such as media-not-ready or active playback replacement
- TTS first-audio latency
- media packetization and outbound pacing/underrun behavior
- barge-in cancel latency and playback terminal status

M6 reports must separate live critical-path latency from overlapping diagnostic spans. A report should be able to say:

```text
This turn was slow because the serial speech-to-first-audio critical path was dominated by endpointing wait and TTS first-chunk synthesis. App generation was not the bottleneck.
```

Or:

```text
This agent-tmux run was slow because the configured input quiet-window intentionally waited before injecting the caller text. Tune the bridge wait before changing ASR or TTS.
```

## Scope

M6 covers two related but distinct models.

| Model | Description | Why it matters |
|---|---|---|
| Strict text-call turn model | M4 app-agent protocol: final ASR text becomes `caller.turn`; app replies with `agent.turn`; gateway emits `playback.started` and `playback.finished`. | This is the stable app integration contract. |
| Full-duplex realism model | M5 behavior: partial ASR and frame-level speech onset can cancel active playback before a final `caller.turn` exists. | This is how live conversation feels responsive and interruptible. |

M6 must profile both. Reports must not force barge-in events into a purely request/response shape. Barge-in spans can overlap the prior playback span and the next caller turn, but overlap must be marked as concurrent and excluded from serial critical-path percentages.

## Future Streaming-Agent Contract Notes

A pluggable streaming `ConversationHandler` is the target real-agent path; the gateway-local smoke harness must not leak coalescing or deterministic echo assumptions into that path. Handler enablement and smoke-test final coalescing are separate controls. David scoped the streaming protocol work to follow-on issue #428, not PR #419.

Contract requirements for that future streaming path (#428):

- The agent contract must deliver advisory partial ASR events for early-commit and barge-in decisions while keeping final `caller.turn` emission as the stable app-visible turn boundary.
- Streamed agent replies should use `agent.turn.partial { turn_id, text, append: true }` plus a terminal `agent.turn`, and the gateway should feed appended fragments into a future `SpeechConflictPolicy::Append` path upstream of the sentence packer. This avoids per-sentence cancel-and-replace self-interruption and lets first audio start from the agent's first complete sentence instead of the last token.
- `bins/telnyx-agent` should adopt `libs/agent` Channel (#421) for delivery once `feature/telnyx-voice` syncs/cherry-picks the mainline crate. Channel should own quiet guard, dedup/coalescing, submit settle/verify/retry, and UI profile composer detection; the Telnyx agent may keep marker-based reply extraction as the response watcher.
- Voice and TTS backend selection belong on the typed conversation response, for example `Say { text, voice: Option<...>, tts_backend: Option<...> }`, so per-response voice policy is explicit and does not inherit transient operator-session state.

## Non-Goals

- Do not change the M4 app-agent protocol to emit partial ASR by default.
- Do not let analysis knobs change live `caller.turn` emission or merge/suppress turns on the wire.
- Do not require LLM judging for normal live call operation.
- Do not let an LLM judge mutate live gateway configuration.
- Do not use LLM output as the only source of truth when golden references or WER data exist.
- Do not introduce a second gateway control socket or a JSON command envelope; reuse the existing line-oriented command dispatcher/socket mux.
- Do not log raw phone numbers or use them as IDs, keys, subscription IDs, filenames, or report grouping values.

## Current Behavior Summary

The gateway already emits several useful signals, but M6 requires a coherent lifecycle log around turns and spans.

| Current signal | Current behavior | Usefulness | M6 gap |
|---|---|---|---|
| `media.speech.detected` | Emitted when speech energy first appears after quiet. | Start-of-speech and speech threshold tuning. | Needs an `asr_session_id`/`utterance_id` before `turn_id` exists. |
| `media.frame.local_endpoint` | Debug-level low-energy tail logging after speech. | Endpoint tail diagnosis. | Needs span boundaries and `config_id` correlation. |
| `asr.local_endpoint.finalizing` | Local endpoint gate decides to finish the ASR session. | Endpoint decision timing. | Needs link to final transcript and caller turn. |
| `transcript.partial` | ASR partial transcript. | Partial quality and barge-in analysis. | Not normally sent to M4 app-agent protocol; can be logged only when privacy mode allows text. |
| `transcript.final` | Final ASR transcript. | Best current proxy for caller-turn text. | Does not include the future `turn_id`; needs `asr_session_id` mapping. |
| `transcript.suppressed_repeated_token` | ASR adapter suppression for repeated-token hallucination. | ASR stability signal. | Suppression thresholds should be config-backed and logged by `config_id`. |
| `transcript.suppressed_assistant_echo` | Deterministic normalized token/substring match against active or just-finished assistant TTS. | Prevents acoustic assistant echo from becoming transcript history, `caller.turn`, or conversation input without LLM/NLP matching. | Needs live-call false-positive labeling in M6 reports. |
| `text_call.caller_turn.forward_failed` | Failure to send a caller turn to the app stream. | Reliability. | Successful caller turns need structured events. |
| `playback.finished.status` | Terminal app-agent playback state. | Completed/canceled/failed/superseded lifecycle. | Needs latency span correlation. |
| `tts.speak.chunk_queued` | TTS chunk packetization and queueing. | TTS startup and chunking behavior. | Needs turn-level aggregation and outbound pacing rollups. |

## Definitions

| Term | Definition |
|---|---|
| Caller utterance | Acoustic speech segment from first speech-energy frame until low-energy tail begins. |
| Caller speaking time | Non-tunable time the caller spent talking, measured by `utterance.speech_to_low_energy`. |
| Endpoint wait | Tunable/backend-driven wait after low-energy tail starts until finalization. |
| ASR session ID | Opaque ID minted when a new ASR session/utterance begins, before a `turn_id` exists. |
| Utterance ID | Opaque acoustic utterance ID linked to speech onset, endpointing, ASR finalization, and the eventual `turn_id`. It can be the same value as `asr_session_id` if the implementation has one active ASR session per utterance. |
| Final transcript | ASR event considered final by backend or local endpoint finish. |
| Caller turn | M4 protocol text unit sent to the app as `caller.turn`; created from final transcript text. |
| Agent turn | App response text sent as `agent.turn` and correlated by `turn_id`. In `bins/telnyx-agent`, tmux reply extraction accepts a turn-end marker only as the last standalone trimmed marker line, so wrapped prompt echoes are not returned as spoken agent text. |
| Playback | TTS/media job created from an accepted `agent.turn`. |
| Barge-in | Caller speech/partial/final transcript during active playback that cancels or replaces speech. |
| Harness mode | A controlled profiling mode such as gateway echo, text-call echo, tmux echo, or real agent. |
| Intentional delay | A deliberate wait such as endpoint trailing silence, tmux quiet-window, or trailing Enter delay. |
| Queue wait | A wait caused by resource state, such as media not ready or active playback conflict. |
| Config snapshot | `call.config.snapshot` event carrying resolved `VoiceQualityConfig` values and a stable `config_id` hash. |
| Critical path | Ordered, non-overlapping spans that sum exactly to a named round-trip total such as `speech_to_first_audio`. |
| Concurrent span | Diagnostic span that overlaps the critical path and is not included in percentage attribution. |
| Report-only knob | Analysis/tuning knob used by reports and judges only; it must not change live `caller.turn` emission. |

## Event Envelope and Versioning

Every emitted M6 record must include enough metadata to be joined and replayed across calls, hosts, and analyzer runs.

Common event envelope:

```json
{
  "quality_schema_version": 1,
  "event_sequence": 42,
  "event": "voice.span.finished",
  "run_id": "run_...",
  "gateway_call_id": "gwc_...",
  "host_id": "host_...",
  "git_sha": "git_...",
  "config_id": "cfg_...",
  "redaction_mode": "metrics_only",
  "created_at": "2026-06-08T21:00:00.000Z"
}
```

Requirements:

| Field | Requirement |
|---|---|
| `quality_schema_version` | Required on every event, report, and judge output. Increment on incompatible schema changes. |
| `run_id` | Required for live, replay, and harness runs. It groups all artifacts from one run. |
| `gateway_call_id` | Required when an event belongs to a call. |
| `event_sequence` | Monotonic per call where possible; reports must detect gaps. |
| `host_id` | Gateway instance or host identity, never a phone number. |
| `git_sha` | Build/revision identity for reproducibility. |
| `config_id` | Hash of the active resolved `VoiceQualityConfig`, never inline config per span. |
| `redaction_mode` | Required value such as `metrics_only`, `hashed_text`, `redacted_text`, or `sensitive_plaintext`. |
| `created_at` | Wall-clock timestamp for ordering and operator inspection. Durations must not be derived from wall-clock timestamps. |

Artifacts and reports additionally need `analyzer_version`, `input_event_hash`, and `report_hash` so offline analysis can be reproduced.

## Config Snapshot Event

Spans must not inline `config { ... }`. Instead, the gateway emits one `call.config.snapshot` event at stream start and one additional snapshot whenever a live config change is accepted and scheduled or applied. Spans carry `config_id` only.

```json
{
  "quality_schema_version": 1,
  "event": "call.config.snapshot",
  "run_id": "run_...",
  "gateway_call_id": "gwc_...",
  "config_id": "cfg_sha256_...",
  "config_hash_algorithm": "sha256-canonical-json",
  "snapshot_reason": "stream_start",
  "effective_scope": "new_asr_sessions",
  "effective_after_asr_session_id": null,
  "resolved_config": {
    "profile": "balanced",
    "speech": {
      "rms_threshold": 220.0,
      "peak_threshold": 1100,
      "onset_min_silence_ms": 180
    },
    "endpoint": {
      "trailing_silence_ms": 650,
      "min_turn_words": 2,
      "min_turn_chars": 6,
      "merge_window_ms": 350,
      "max_turn_words": 80,
      "max_turn_duration_ms": 12000
    },
    "tts": {
      "chunking_enabled": true,
      "max_text_chunk_chars": 90,
      "first_chunk_max_chars": 0,
      "prebuffer_chunks": 1
    },
    "asr": {
      "finish_pad_ms": 160,
      "repeated_token_run_threshold": 16,
      "repeated_q_run_threshold": 8
    },
    "logging": {
      "include_transcript_text": false,
      "redaction_mode": "metrics_only"
    }
  }
}
```

Live config changes are staged. A new `config_id` may be recorded immediately, but ASR/endpointing changes apply only at the next ASR session/utterance boundary, never mid-utterance. Text-call and playback changes apply only to new turns or new playback requests unless explicitly marked immediate and safe.

## Span Event Model

A span is a named timed interval. Every M6 latency measurement should be expressed as a span or as an aggregate of spans.

Required span event shape:

```json
{
  "quality_schema_version": 1,
  "event": "voice.span.finished",
  "span_id": "span_...",
  "parent_span_id": "span_...",
  "gateway_call_id": "gwc_...",
  "asr_session_id": "asr_...",
  "utterance_id": "utt_...",
  "turn_id": "turn_...",
  "playback_id": "tts_...",
  "config_id": "cfg_sha256_...",
  "name": "asr.endpoint_wait",
  "category": "endpointing",
  "critical_path": "speech_to_first_audio",
  "critical_path_order": 2,
  "concurrent": false,
  "started_at": "2026-06-08T21:00:00.000Z",
  "finished_at": "2026-06-08T21:00:00.820Z",
  "duration_ms": 820,
  "status": "completed",
  "wait_reason": "trailing_silence_threshold"
}
```

Required fields:

| Field | Requirement |
|---|---|
| `event` | Stable event name, usually `voice.span.finished`. |
| `span_id` | Opaque ID for this span. |
| `parent_span_id` | Optional parent span. |
| `gateway_call_id` | Required call correlation key. |
| `asr_session_id` | Required for acoustic, ASR, endpointing, and pre-turn events. |
| `utterance_id` | Required for acoustic/endpointing spans; can match `asr_session_id` if appropriate. |
| `turn_id` | Required when a span belongs to a caller/agent turn. Null before the mapping event exists. |
| `playback_id` | Required when a span belongs to TTS/media playback. |
| `config_id` | Required; references the latest applicable `call.config.snapshot`. |
| `name` | Stable span name from the tables below. |
| `category` | One M6 latency category. Multi-category analysis must be done by report aggregation, not by putting multiple categories on one span. |
| `critical_path` | Null for purely diagnostic spans; otherwise the round-trip model this span contributes to. |
| `critical_path_order` | Stable order among serial spans in the chosen critical path. |
| `concurrent` | `true` for overlapping diagnostic spans excluded from serial percentage math. |
| `started_at` / `finished_at` | Wall-clock timestamps for ordering and human inspection. |
| `duration_ms` | Derived from monotonic `Instant` deltas, not `SystemTime`. |
| `status` | `completed`, `canceled`, `failed`, `superseded`, or `timeout`. |
| `wait_reason` | Null except for intentional delay or queue wait spans. |

Privacy requirements:

- No raw phone numbers.
- No routing values as IDs or grouping keys.
- `include_transcript_text` defaults to `false` in live and config examples.
- Transcript text is an explicit sensitive opt-in and must be stored only in local/operator-controlled artifacts unless a separate upload approval exists.
- Redacted mode must preserve metrics while dropping, hashing, or redacting raw text.

## Non-Blocking Emission Contract

M6 logging must not perturb live media behavior.

| Requirement | Spec |
|---|---|
| Realtime path | Media-frame and ASR ingest tasks must never await on log I/O, judge calls, report generation, or disk/network writes. |
| Event sink | Use a bounded MPSC queue from realtime producers to a non-realtime writer. Producers use non-blocking `try_send` or equivalent. |
| Drop behavior | If the queue is full, drop the profiling event, increment a dropped-event counter by event kind, and continue media processing. |
| Writer | Batch and serialize off the media path. Writer failures become structured `quality_log.writer_error` events when possible. |
| Per-frame spans | `asr.ingest_frame` spans are off by default and must be sampled because per-frame logging can dominate the path being measured. |
| Duration source | Store wall-clock timestamps for ordering, but compute `duration_ms` from monotonic `Instant` boundaries captured in-process. |
| Non-perturbation test | A forced-full queue test must prove media processing does not block and the dropped-event counter increments. |

## Latency Categories

Categories are used in two ways:

1. Critical-path categories form a serial decomposition whose durations sum exactly to the requested round-trip total.
2. Concurrent diagnostic categories explain overlapping behavior and confounders. They can be reported, but they are excluded from critical-path percentages.

| Category | Meaning | Tunable? | Examples |
|---|---|---|---|
| `caller_speech` | Time the caller spent speaking before low-energy tail. | No | `utterance.speech_to_low_energy` |
| `endpointing` | Waiting for the caller turn boundary after speech has ended. | Yes | trailing silence after speech, backend final wait |
| `asr_generation` | ASR backend work. | Indirect | frame ingest, ASR session finish |
| `gateway_overhead` | Gateway-local dispatch and state work. | Usually no | transcript to turn, frame serialization, turn correlation |
| `network_transport` | HTTP/WebSocket transport outside model work. | Indirect | app WebSocket frame send/receive |
| `model_generation` | LLM/app/agent text generation. | App-owned | real agent response, handler call |
| `harness` | Deterministic test harness behavior. | Test-only | echo response construction |
| `intentional_delay` | Configured deliberate wait. | Yes | tmux quiet-window, trailing Enter delay |
| `queue_wait` | Resource or ordering wait. | Sometimes | media not ready, active playback conflict |
| `tts_generation` | TTS model work. | Backend/config | first audio chunk, full synthesis |
| `media_packetization` | Resample, packetize, chunk queue. | Indirect | first packet prep |
| `playback_transport` | Outbound media send and playback terminal behavior. | Indirect | first frame sent, mark/clear terminal |
| `barge_in` | Interrupt detection and cancellation path. | Yes | speech onset to clear request |
| `transport_confounder` | Network/media quality rollup that explains degraded results but is not part of turn latency percentage math. | No | RTP loss, stale frames, jitter, outbound underruns |
| `turn_taking` | End-to-end turn envelope or rollup span. | No direct knob; inspect child/component spans. | `turn.finalize_to_first_audio` |

The analyzer must never classify caller speaking time as endpointing. If a caller talks for a long time, that is a user utterance characteristic, not a recommendation to cut trailing silence. Rollup categories such as `turn_taking` are for direct p50/p95 tracking and must not be included in per-category percentage attribution when component spans are also present.

## ASR Session and Turn Join Events

Early events often happen before a `turn_id` exists. M6 therefore requires a pre-turn join key.

```json
{
  "event": "asr.session.started",
  "asr_session_id": "asr_...",
  "utterance_id": "utt_...",
  "gateway_call_id": "gwc_...",
  "config_id": "cfg_..."
}
```

When a final transcript becomes a M4 caller turn, emit the mapping:

```json
{
  "event": "asr.turn_mapped",
  "asr_session_id": "asr_...",
  "utterance_id": "utt_...",
  "turn_id": "turn_...",
  "final_transcript_event_id": "trn_...",
  "caller_turn_sent": true
}
```

This mapping is required for endpointing, barge-in, suppressed-token, and transport events to be joined with the eventual `caller.turn`.

## Acoustic, ASR, and Endpointing Spans

| Span | Start | End | Category | Critical path? | Purpose |
|---|---|---|---|---|---|
| `utterance.speech_to_low_energy` | first speech-energy frame after quiet | first low-energy frame after speech | `caller_speech` | yes | Approximate actual spoken portion; not tunable endpointing. |
| `asr.endpoint_wait` | first low-energy tail frame after speech | local endpoint decision or backend final decision | `endpointing` | yes | Separate deliberate endpoint delay from model work. |
| `asr.finish_pad` | local endpoint decision or stream stop | configured finish-pad audio fed to ASR | `asr_generation` | yes if it delays final transcript | Flush a small contiguous pad without reusing the full endpoint tail. |
| `asr.local_finish` | call to finish active ASR session | final transcript events returned | `asr_generation` | yes | Measure ASR flush/finalization cost. |
| `asr.ingest_frame` | frame ingest starts | frame ingest returns transcript events | `asr_generation` | no by default | Optional sampled frame-level backend cost; off by default. |
| `asr.final_to_turn` | final transcript recorded | `caller.turn` created/sent | `gateway_overhead` | yes | Measure gateway handoff overhead. |

Boundary rules:

- The start of `asr.endpoint_wait` is the first low-energy frame after speech began.
- The end is the finalization decision, not the later transcript-to-turn handoff.
- Hot-reloaded ASR/endpoint config applies only when the next `asr.session.started` event is created.
- `asr.finish_pad_ms` is separate from endpoint trailing silence. It is a short ASR flush pad after the endpoint decision, not another full endpoint wait.
- `endpoint.merge_window_ms` may label adjacent turns as likely merge candidates in reports. The gateway-local smoke-test handler may also use it as a handler-local final-fragment debounce before producing its deterministic echo, but it must not merge live M4 `caller.turn` events on the app-agent protocol.

## Text-Call and App-Agent Spans

| Span | Start | End | Category | Critical path? | Purpose |
|---|---|---|---|---|---|
| `text_call.send_caller_turn` | caller turn frame serialization begins | WebSocket frame send completes | `network_transport` | yes | M4 outbound text protocol overhead including local serialization. |
| `app.agent_turn_wait` | `caller.turn` sent | `agent.turn` received | `model_generation` or `harness` | yes | App response latency envelope. Implemented for the M4 WebSocket path by stamping the caller-turn send time and recovering it when the correlated `agent.turn` arrives. Echo handlers must classify deterministic work as `harness`, not model generation. |
| `app.generation` | app receives turn | app creates response text | `model_generation` or `harness` | concurrent unless app instrumentation provides exact boundary | App-side detail when available. |
| `text_call.agent_turn_accept` | `agent.turn` received | TTS queue request begins | `gateway_overhead` | yes | Turn correlation and policy work. |
| `text_call.supersede_stale_turn` | stale valid `agent.turn` received | `playback.finished status=superseded` sent | `gateway_overhead` | concurrent | Validates stale-but-valid behavior without hanging up. |

The strict M4 protocol remains turn-based. The gateway should not require apps to stream tokens to satisfy M6.

## TTS and Playback Spans

| Span | Start | End | Category | Critical path? | Purpose |
|---|---|---|---|---|---|
| `tts.media_ready_wait` | queue requested while media is not ready | media becomes ready or timeout | `queue_wait` | yes if it delays the first response audio | Distinguish setup waiting from TTS generation. |
| `tts.conflict_cancel_replace` | new speech replaces active playback | prior playback terminal `canceled` emitted | `barge_in` | concurrent | Measure latest-response-wins behavior. |
| `conversation.smoke_final_debounce` | first smoke-test final transcript | handler-local debounce expires or adjacent final merges | `intentional_delay` | yes in `gateway-echo` only | Exposes deterministic harness coalescing instead of hiding it in handler latency. |
| `tts.synthesis_first_chunk` | TTS synthesis starts | first audio chunk available | `tts_generation` | yes | Time to first audio. |
| `tts.synthesis_full` | TTS synthesis starts | all chunks available | `tts_generation` | concurrent after first chunk | Full generation cost. |
| `tts.packetize_first_chunk` | first audio chunk available | first media packet queued | `media_packetization` | yes | Resample/packetization overhead. |
| `media.first_frame_send` | first media packet queued | first outbound media frame sent | `playback_transport` | yes | Queue-to-wire latency. |
| `tts.request_to_first_audio` | TTS queue request accepted | first outbound media frame sent | `tts_generation` | yes | End-to-end request-to-first-audio envelope, including synthesis, packetization, and prebuffer. |
| `turn.finalize_to_first_audio` | final transcript/turn boundary captured | first outbound media frame sent | `turn_taking` rollup | yes, but not a category-attribution bucket | Full post-finalize critical-path envelope; includes handler dispatch, deterministic debounce, app-agent round trip, TTS, prebuffer, and first outbound frame. Text-call sessions pass the final transcript `Instant` through the turn tracker into `SpeechQueueRequest` so this fires for real agent calls as well as smoke calls. Reports use child/component spans for percentage attribution. |
| `media.playback_terminal` | playback started | mark/clear/failure terminal status | `playback_transport` | concurrent after first frame | Completion/cancel/failure timing. |

## Barge-In and Full-Duplex Spans

M5 barge-in is not strictly turn-bound. The caller may start speaking while the prior `agent.turn` playback is active, before a new final transcript exists. M6 must preserve this causality.

| Span | Start | End | Category | Critical path? | Purpose |
|---|---|---|---|---|---|
| `barge_in.speech_onset_to_cancel_request` | speech onset while playback active | cancel/clear requested | `barge_in` | concurrent | Perceived interruption responsiveness. |
| `barge_in.cancel_request_to_terminal` | cancel/clear requested | playback terminal `canceled` observed | `playback_transport` | concurrent | How fast playback actually stops. |
| `barge_in.partial_to_cancel_request` | meaningful partial ASR while playback active | cancel/clear requested | `barge_in` | concurrent | Partial-ASR interruption path. |
| `barge_in.final_to_regenerate` | final transcript after interruption | new response queued | `gateway_overhead` | yes for the new response path | Regeneration latency. |

Use span links for overlapping causality:

```json
{
  "event": "voice.span.link",
  "from_span": "barge_in.speech_onset_to_cancel_request",
  "to_span": "media.playback_terminal",
  "relationship": "cancels_playback",
  "playback_id": "tts_...",
  "concurrent": true
}
```

## Transport Quality Rollups

Transport defects can make ASR, endpointing, and playback look bad even when the logic is correct. M6 therefore requires aggregate rollups that are joined to calls/turns but excluded from serial critical-path percentages.

Inbound RTP/media rollup:

```json
{
  "event": "media.inbound_transport.rollup",
  "gateway_call_id": "gwc_...",
  "asr_session_id": "asr_...",
  "packets_total": 2400,
  "lost_packets": 3,
  "stale_frames": 1,
  "reordered_frames": 2,
  "jitter_ms_p50": 3,
  "jitter_ms_p95": 18,
  "jitter_ms_max": 42,
  "codec": "pcmu",
  "sample_rate_hz": 8000,
  "frame_ms": 20
}
```

Outbound pacing rollup:

```json
{
  "event": "media.outbound_pacing.rollup",
  "gateway_call_id": "gwc_...",
  "playback_id": "tts_...",
  "frames_sent": 180,
  "underrun_count": 0,
  "inter_frame_gap_ms_p50": 20,
  "inter_frame_gap_ms_p95": 24,
  "inter_frame_gap_ms_max": 37,
  "queue_depth_p50": 4,
  "queue_depth_p95": 9
}
```

Reports must surface these fields as confounders and exclusion reasons, for example `excluded_reason = high_inbound_jitter` or `confounder = outbound_underrun`.

## Deterministic `I Heard` Harness

The deterministic `I heard: <caller text>` repeat handler is a required M6 profiling harness. Its response generation is predictable, so it isolates infrastructure latency before real LLM/agent generation enters the loop.

Required harness modes:

| Mode | Handler | What it isolates |
|---|---|---|
| `gateway-echo` | Gateway-local conversation smoke-test handler. | ASR endpointing + gateway conversation path + TTS + media, with no M4 app WebSocket. |
| `text-call-echo` | External app immediately returns `I heard: <text>` over the M4 text WebSocket. | M4 text-call protocol overhead plus baseline TTS/media. |
| `agent-tmux-echo` | `bins/telnyx-agent` bridges to a deterministic local tmux echo prompt. | motlie-tmux injection, activity wait, trailing Enter, monitor/history scrape overhead. |
| `real-agent` | Actual app/agent. | Full app/model/tmux generation latency. |

Comparison rules:

- Do not subtract whole-call totals from unrelated live calls; live audio, network, and caller behavior make that noisy.
- Whole-run subtraction is valid only for fixed/replayed audio or a paired harness corpus where the same media input is used.
- Preferred analysis is per-category deltas: compare endpointing to endpointing, TTS to TTS, network transport to network transport, and app/model generation to deterministic harness spans.
- The deterministic echo handler must report near-zero `model_generation` unless the implementation intentionally routes through a model.

Paired replay formulas:

```text
text-call protocol delta = text-call-echo.network_transport - gateway-echo.network_transport
agent bridge delta       = agent-tmux-echo.intentional_delay + agent-tmux-echo.gateway_overhead - text-call-echo.gateway_overhead
real generation delta    = real-agent.model_generation - agent-tmux-echo.harness
```

Harness metadata:

```json
{
  "harness_mode": "text-call-echo",
  "response_policy": "i_heard_repeat",
  "deterministic_response": true,
  "expected_response_prefix": "I heard:"
}
```

## Harness-Level Spans

| Span | Applies to | Category | Critical path? | Purpose |
|---|---|---|---|---|
| `harness.echo_generate` | gateway/text-call echo | `harness` | yes if it delays response | Deterministic response construction; should be near-zero. |
| `harness.http_ws_round_trip` | text-call echo | `network_transport` | yes | App protocol loop overhead. |
| `harness.tmux_input_wait` | agent-tmux echo | `intentional_delay` | yes | Quiet-window wait before keystroke injection. |
| `harness.tmux_backoff_wait` | agent-tmux echo | `intentional_delay` | yes | Backoff while target is active. |
| `harness.tmux_inject` | agent-tmux echo | `gateway_overhead` | yes | Send keystrokes through motlie-tmux. |
| `harness.trailing_enter_delay` | agent-tmux echo | `intentional_delay` | yes | Configured submit delay. |
| `harness.tmux_reply_wait` | agent-tmux echo | `harness` | yes | Wait for deterministic reply to render. Real agents should use `model_generation`. |
| `harness.tmux_scrape` | agent-tmux echo | `gateway_overhead` | yes | Monitor/history scrape and reply extraction. |

## Round-Trip and Critical-Path Definitions

Different teams use "round trip" differently. M6 reports must name the exact boundary.

| Round trip | Start | End | Use |
|---|---|---|---|
| `speech_to_first_audio` | first speech-energy frame | first outbound assistant media frame sent | User-perceived full turn latency. |
| `low_energy_to_first_audio` | first low-energy frame after speech | first outbound assistant media frame sent | Endpointing plus response stack. |
| `final_transcript_to_first_audio` | final transcript event | first outbound assistant media frame sent | Excludes caller speech and endpointing wait; implemented by `turn.finalize_to_first_audio` when the handler path has a final transcript boundary. |
| `caller_turn_to_playback_started` | `caller.turn` sent | `playback.started` sent | M4 text-call responsiveness. |
| `agent_turn_to_first_audio` | `agent.turn` received | first outbound assistant media frame sent | TTS/media path. |
| `barge_in_to_silence` | caller speech onset during playback | prior playback terminal/canceled | Interruption responsiveness. |

The default M6 summary should report at least:

- `speech_to_first_audio`
- `low_energy_to_first_audio`
- `caller_turn_to_playback_started`
- `agent_turn_to_first_audio`
- `barge_in_to_silence` when applicable

Critical-path rule:

```text
sum(duration_ms for spans where critical_path == X and concurrent == false) == total_X_ms
```

Only those serial spans may be used for percentage attribution. Concurrent spans and transport confounder rollups can explain behavior but must not be included in the denominator.

## Example `I Heard` Report

This worked example is internally reconciled: critical-path category totals sum to `total_speech_to_first_audio_ms`.

```json
{
  "quality_schema_version": 1,
  "harness_mode": "text-call-echo",
  "turn_id": "turn_...",
  "transcript_text_included": false,
  "text_metrics": {
    "caller_text_words": 3,
    "caller_text_chars": 17,
    "agent_text_words": 4,
    "agent_text_chars": 26
  },
  "total_speech_to_first_audio_ms": 2053,
  "critical_path_breakdown_ms": {
    "utterance_speech_to_low_energy": 720,
    "asr_endpoint_wait": 820,
    "asr_local_finish": 35,
    "asr_final_to_caller_turn": 4,
    "text_call_send_caller_turn": 8,
    "app_agent_turn_wait": 22,
    "text_call_agent_turn_accept": 4,
    "tts_synthesis_first_chunk": 410,
    "tts_packetize_first_chunk": 18,
    "media_first_frame_send": 12
  },
  "critical_path_by_category_ms": {
    "caller_speech": 720,
    "endpointing": 820,
    "asr_generation": 35,
    "gateway_overhead": 8,
    "network_transport": 30,
    "model_generation": 0,
    "harness": 0,
    "intentional_delay": 0,
    "queue_wait": 0,
    "tts_generation": 410,
    "media_packetization": 18,
    "playback_transport": 12
  },
  "excluded_concurrent_ms": {
    "tts_synthesis_full": 890,
    "media_playback_terminal": 3600
  },
  "dominant_tunable_categories": ["endpointing", "tts_generation"],
  "recommendation": "Endpointing and TTS first chunk dominate this deterministic baseline; do not tune app generation."
}
```

Example agent-tmux echo bottleneck:

```json
{
  "harness_mode": "agent-tmux-echo",
  "total_speech_to_first_audio_ms": 12093,
  "critical_path_by_category_ms": {
    "caller_speech": 720,
    "endpointing": 820,
    "asr_generation": 35,
    "gateway_overhead": 52,
    "network_transport": 26,
    "intentional_delay": 10000,
    "harness": 10,
    "tts_generation": 410,
    "media_packetization": 18,
    "playback_transport": 2
  },
  "recommendation": "The primary latency is agent bridge intentional delay; tune the agent bridge quiet-window before changing ASR or TTS."
}
```

## Turn Quality Analysis

M6 must combine deterministic metrics and semantic judging.

Per-turn deterministic metrics:

| Metric | Meaning | Quality interpretation |
|---|---|---|
| `text_words` | Words in caller turn. | Very low can mean premature endpointing unless acknowledgement. |
| `text_chars` | Characters in caller turn. | Useful for empty or low-information turns. |
| `speech_start_to_final_ms` | Speech plus endpoint and finalization time. | High values can indicate long utterance or late endpointing. |
| `caller_speech_ms` | Approximate caller speaking time. | Non-tunable; separates long speech from late endpointing. |
| `trailing_silence_ms` | Local endpoint tail. | High p95 means slow turn-taking. |
| `final_to_turn_sent_ms` | Gateway dispatch overhead. | Should be small. |
| `caller_turn_to_agent_turn_ms` | App/agent response time. | Impacts responsiveness. |
| `agent_turn_to_playback_started_ms` | TTS queue/start time. | Impacts perceived speed. |
| `finalize_to_first_audio_ms` | Final transcript to first outbound frame. | Confirms whether delay is handler/debounce/TTS/media rather than endpointing. |
| `playback_finished.status` | Terminal outcome. | High canceled/superseded rates can indicate churn. |
| `partial_count_before_final` | ASR stability. | High counts with poor final quality can indicate decoder instability. |
| `suppressed_count` | Suppressed hallucination-like output. | Backend/model quality signal. |
| `transport_confounders` | RTP loss/jitter/underrun rollups. | Exclude or stratify before tuning endpointing/ASR. |

Aggregate metrics:

| Aggregate | Use |
|---|---|
| p50/p95 `trailing_silence_ms` | Endpoint latency budget. |
| p50/p95 `speech_to_first_audio_ms` | Live perceived latency. |
| p50/p95 `caller_turn_to_playback_started_ms` | Text-call protocol responsiveness. |
| short-turn rate | Premature endpointing / low information. |
| overlong-turn rate | Late endpointing / merged intents. |
| incomplete-turn rate from labels | Semantic endpoint quality. |
| overmerged-turn rate from labels | Semantic endpoint quality. |
| garbled-turn rate from labels | ASR text quality. |
| canceled playback rate | Barge-in aggressiveness. |
| superseded turn rate | Caller overlap / app-agent staleness. |
| invalid turn rate | Protocol correctness. |
| repeated-token suppression rate | ASR hallucination pressure. |
| dropped profiling event rate | Logging non-perturbation and report completeness. |

Report-only/advisory analysis knobs:

| Knob | Report use | Live protocol effect |
|---|---|---|
| `endpoint.min_turn_words` | Label possible low-information fragments. | None; must not suppress `caller.turn`. |
| `endpoint.min_turn_chars` | Label tiny text fragments. | None; must not suppress `caller.turn`. |
| `endpoint.merge_window_ms` | Suggest adjacent-turn merge candidates in reports; handler-local smoke-test debounce for deterministic echo only. | None for M4 app-agent `caller.turn`; smoke-test handler may delay/merge only its local echo input. |
| `endpoint.max_turn_words` | Flag likely overmerged turns. | None; must not split live turns. |
| `endpoint.max_turn_duration_ms` | Flag long utterance or late endpointing. | None; must not force live endpointing. |

## Normalized Artifacts

M6 reports must emit normalized JSONL records in addition to human-readable Markdown. These files are append-friendly and analysis-ready.

| Artifact | Required records | Notes |
|---|---|---|
| `calls.jsonl` | call lifecycle, direction, harness mode, codec/sample rate, redaction mode, final outcome, exclusion reason | No raw phone numbers. |
| `turns.jsonl` | `turn_id`, `asr_session_id`, timing rollups, text metrics, optional redacted/sensitive text, label joins | `turn_id` created only after final transcript. |
| `spans.jsonl` | all `voice.span.finished` records and links | Include `config_id`, `critical_path`, `concurrent`. |
| `playbacks.jsonl` | playback started/finished, status, cancel/supersede reason, first-frame timings | Join by `playback_id` and `turn_id`. |
| `barge_in_events.jsonl` | speech onset/partial/final triggers, cancel request, clear/terminal status, false-positive labels when available | Can reference previous playback and next ASR session. |
| `judge_labels.jsonl` | human, WER scorer, or LLM labels | Labels are observations, not truth unless source is a golden reference scorer. |
| `reports.jsonl` | aggregate run/report summaries | Include denominators, exclusions, confidence intervals, input/report hashes. |

Every aggregate report must include denominators and exclusion counts, for example:

```json
{
  "event": "quality.report.summary",
  "run_id": "run_...",
  "input_event_hash": "sha256_...",
  "report_hash": "sha256_...",
  "turns_total": 120,
  "turns_analyzed": 110,
  "turns_excluded": 10,
  "exclusions": {
    "missing_final_transcript": 2,
    "high_inbound_jitter": 5,
    "dropped_required_events": 3
  }
}
```

## Reference, WER, and Label Schema

Logs alone cannot compute WER. Golden/replay runs must carry reference metadata when available.

```json
{
  "event": "quality.label",
  "label_source": "wer_scorer",
  "reference_id": "ref_...",
  "reference_manifest_id": "manifest_...",
  "turn_id": "turn_...",
  "scorer_version": "wer_scorer_...",
  "wer": 0.08,
  "cer": 0.03,
  "alignment_ref": "align_..."
}
```

Human and LLM labels use the same label artifact with different `label_source` values. Required label metadata:

| Field | Requirement |
|---|---|
| `label_source` | `human`, `llm`, `wer_scorer`, or `deterministic_rule`. |
| `labeler_id` | Required for human labels; opaque and non-sensitive. |
| `judge_run_id` | Required for LLM labels. |
| `rubric_version` | Required for human and LLM labels. |
| `confidence` | `0.0..1.0`; labels may abstain. |
| `evidence_refs` | Span IDs, turn IDs, or playback IDs used as evidence. |
| `reviewed_at` | Wall-clock timestamp for label creation. |

LLM output is a label source, not ground truth. Reports should prefer WER/golden references when available, then calibrated human labels, then LLM labels with confidence and uncertainty.

## LLM Judge

The LLM judge consumes report artifacts and emits labels/recommendations only. It never mutates live gateway config and must never run on the realtime media path. Live sampling, if enabled, samples at the log/report layer after events are already written.

Judge input:

```json
{
  "quality_schema_version": 1,
  "run_id": "run_...",
  "turn_id": "turn_...",
  "asr_session_id": "asr_...",
  "harness_mode": "text-call-echo",
  "redaction_mode": "sensitive_plaintext",
  "reference_id": null,
  "previous_turn": {
    "text": "I need to",
    "text_words": 3,
    "trailing_silence_ms": 220
  },
  "current_turn": {
    "text": "reset my password",
    "text_words": 3,
    "trailing_silence_ms": 260,
    "playback_status": "completed"
  },
  "next_turn": {
    "text": "and update my email",
    "text_words": 4,
    "trailing_silence_ms": 230
  },
  "critical_path_by_category_ms": {
    "caller_speech": 720,
    "endpointing": 820,
    "asr_generation": 35,
    "gateway_overhead": 8,
    "network_transport": 30,
    "model_generation": 0,
    "intentional_delay": 0,
    "queue_wait": 0,
    "tts_generation": 410,
    "playback_transport": 12
  },
  "transport_confounders": {
    "inbound_jitter_ms_p95": 18,
    "lost_packets": 0,
    "outbound_underrun_count": 0
  }
}
```

Judge output:

```json
{
  "quality_schema_version": 1,
  "event": "quality.judge_label",
  "judge_run_id": "judge_...",
  "turn_id": "turn_...",
  "label_source": "llm",
  "asr_text_quality": 4,
  "endpointing_quality": 2,
  "conversation_realism_quality": 3,
  "actionability_for_agent": 3,
  "too_short": true,
  "too_long": false,
  "likely_incomplete": false,
  "likely_overmerged": false,
  "likely_premature_cut": true,
  "likely_late_endpoint": false,
  "likely_asr_garbled": false,
  "should_merge_with_previous": true,
  "should_merge_with_next": false,
  "dominant_latency_category": "endpointing",
  "generation_latency_suspected": false,
  "intentional_delay_suspected": false,
  "queueing_suspected": false,
  "recommended_next_probe": "Run the same utterance in gateway-echo and text-call-echo to isolate app protocol overhead.",
  "confidence": 0.72,
  "uncertain": false,
  "insufficient_context": false,
  "evidence_refs": ["turn_prev", "turn_...", "span_endpoint_wait"],
  "reason": "The previous turn is an incomplete prefix and this turn completes it."
}
```

Judge reproducibility metadata:

| Field | Requirement |
|---|---|
| `judge_model` | Provider/model name used. |
| `judge_model_version` | Required when provider exposes it; otherwise record `unversioned`. |
| `prompt_version` | Stable prompt template version. |
| `rubric_version` | Stable rubric version. |
| `temperature` | Numeric setting. |
| `seed` | Required when supported, null otherwise. |
| `input_hash` | Hash of judge input after redaction. |
| `output_hash` | Hash of raw judge output. |
| `cache_key` | Key used for deterministic offline caching. |
| `transcript_redaction_mode` | Must match or be stricter than the source artifact. |

Anchored rubric:

| Score | ASR text quality | Endpointing quality | Conversation realism | Actionability |
|---:|---|---|---|---|
| 1 | Unusable or mostly wrong. | Clearly cut off or severely overmerged. | Breaks turn-taking badly. | Agent cannot act. |
| 2 | Major errors or missing key words. | Likely premature or late endpointing. | Noticeably awkward. | Agent must ask clarification. |
| 3 | Understandable with some errors. | Acceptable but imperfect boundary. | Usable but not natural. | Agent can proceed with caution. |
| 4 | Minor errors only. | Good boundary. | Natural enough for live use. | Agent can proceed. |
| 5 | Clean and faithful. | Excellent boundary. | Smooth, realistic interaction. | Fully actionable. |

Prompt requirements:

- Return strict JSON only.
- Use the anchored `1..5` rubric.
- Include `confidence`, `uncertain`, `insufficient_context`, and `evidence_refs`.
- Distinguish ASR text quality from endpointing quality.
- Treat short acknowledgements as valid when context supports them.
- Flag overmerged turns when one text contains multiple intents or topic shifts.
- Flag late endpointing when trailing silence and response latency are high.
- Do not blame model generation when `harness_mode` is deterministic and `model_generation_ms` is zero or near-zero.
- Abstain rather than infer when transcript context, reference text, or adjacent turns are unavailable.
- Do not infer sensitive identity details from transcript text.

## Current and Proposed Tunables

### Implemented or Present Today

| Area | Knob | Current status | Default/current value | What logs optimize |
|---|---|---:|---:|---|
| Speech detection | `SPEECH_RMS_THRESHOLD` | hard-coded | `220.0` | false starts, missed speech |
| Speech detection | `SPEECH_PEAK_THRESHOLD` | hard-coded | `1100` | noise vs speech distinction |
| Endpointing | `ASR_LOCAL_ENDPOINT_TRAILING_SILENCE_MS` | hard-coded live, replay default | `650 ms` | premature vs late endpointing |
| Barge-in onset | `ASR_SPEECH_ONSET_MIN_SILENCE_MS` | hard-coded | `180 ms` | barge-in sensitivity |
| Replay ASR | `--chunk-ms` | CLI implemented | `20` | streaming stability |
| Replay ASR | `--trailing-silence-pad-ms` | CLI implemented | `800` | finalization behavior |
| ASR backend | `--backend`, `asr use` | CLI/REPL implemented | selected backend | backend comparison |
| Codec eval | `--codec` in golden A/B | CLI implemented | selected matrix | Telnyx format comparison |
| Conversation | `conversation barge-in on|off|status` | REPL/socket implemented | `on` for normal conversation; smoke-test enablement sets `off` and uses a 900 ms final settle floor with active-playback hold | interruption realism vs deterministic echo validation |
| Text-call | `quality text-call max-active-turns <n>` | REPL/socket/TUI implemented | `32` | runaway app-agent lag |
| Text-call | `quality text-call media-ready-timeout-ms <ms>` | REPL/socket/TUI implemented | `20000 ms` | setup reliability |
| Text-call | `quality text-call playback-wait-timeout-ms <ms>` | REPL/socket/TUI implemented | `180000 ms` | hung playback detection |
| Text-call | `quality text-call latest-response-wins <on|off>` | REPL/socket/TUI implemented | `on` | stale response policy |
| Inbound/outbound callbacks | `quality text-call callback-timeout-ms <ms>` | REPL/socket/TUI implemented | `5000 ms` | subscriber responsiveness |
| Outbound API | `timeout_ms` | request field implemented | `45000 ms` | outbound setup latency |
| Agent bridge | `reply_timeout_ms` | CLI implemented | `120000 ms` | local agent responsiveness |
| Agent bridge | `input_quiet_for_ms` | CLI implemented | `10000 ms` | non-barge-in tmux UX |
| Agent bridge | `input_delivery_timeout_ms` | CLI implemented | `30000 ms` | maximum caller silence while waiting for target idle |
| Agent bridge | input backoff initial/max | CLI implemented | `250 ms` / `5000 ms` | queued transcription delay |
| Agent bridge | trailing Enter delay | CLI implemented | `750 ms` | prompt submission reliability |
| Agent bridge | trailing Enter enabled | CLI implemented | default on | prompt submission reliability |
| ASR finish pad | `quality asr finish-pad-ms <ms>` | REPL/socket/TUI implemented | `160 ms` | ASR final flush without doubling endpoint tail |
| ASR suppression | `quality asr repeated-token-run-threshold <n>`, `quality asr repeated-q-run-threshold <n>` | REPL/socket/TUI implemented | run `16`, q-run `8` | hallucination suppression |
| TTS chunking | `quality tts chunking on|off` | REPL/socket/TUI implemented | default on | first-audio latency vs smoothness |
| TTS first chunk ramp | `quality tts first-chunk-max-chars <n>` | REPL/socket/TUI implemented | `0` disabled | first-audio latency vs sentence-complete audio |

### Gateway-Owned `VoiceQualityConfig` Knobs

`VoiceQualityConfig` is the single resolved source of truth for gateway-owned quality behavior. Every knob has a typed domain, min/max or enum values, a live-safe default, validation behavior, and live-apply boundary.

| Key | Domain type | Min / max or values | Live-safe default | Validation | Live apply boundary | Effect |
|---|---|---|---:|---|---|---|
| `profile` | `QualityProfile` enum | `fast`, `balanced`, `complete`, `noisy` | `balanced` | reject unknown | next ASR session for derived ASR values | Selects defaults. |
| `speech.rms_threshold` | `RmsThreshold(f32)` | `0.0..20000.0` | `220.0` | reject NaN, clamp to range with warning | next ASR session | Live speech gate. |
| `speech.peak_threshold` | `PeakThreshold(i32)` | `0..32767` | `1100` | clamp to range | next ASR session | Live speech gate. |
| `speech.onset_min_silence_ms` | `DurationMs` | `0..2000` | `180` | clamp to range | next ASR session | Barge-in onset sensitivity. |
| `endpoint.trailing_silence_ms` | `DurationMs` | `100..5000` | `650` | clamp to range | next ASR session | Live endpointing. |
| `endpoint.min_turn_words` | `ReportOnlyCount` | `0..50` | `2` | clamp to range | report only | Short-turn label threshold only. |
| `endpoint.min_turn_chars` | `ReportOnlyCount` | `0..200` | `6` | clamp to range | report only | Tiny-turn label threshold only. |
| `endpoint.merge_window_ms` | `ReportOnlyDurationMs` | `0..5000` | `350` | clamp to range | report only; smoke-test handler reads the ASR-session snapshot | Adjacent-turn recommendation and deterministic smoke-test final-fragment debounce only. |
| `endpoint.max_turn_words` | `ReportOnlyCount` | `1..500` | `80` | clamp to range | report only | Overmerged-turn label threshold only. |
| `endpoint.max_turn_duration_ms` | `ReportOnlyDurationMs` | `1000..120000` | `12000` | clamp to range | report only | Long-turn label threshold only. |
| `asr.finish_pad_ms` | `DurationMs` | `0..2000` | `160` | clamp to range | next ASR session | Short final ASR flush pad after endpoint decision; separate from endpoint tail. |
| `asr.repeated_token_run_threshold` | `Count` | `2..128` | `16` | clamp to range | next ASR session | Suppression policy. |
| `asr.repeated_q_run_threshold` | `Count` | `2..64` | `8` | clamp to range | next ASR session | Suppression policy. |
| `text_call.max_active_turns` | `Count` | `1..1024` | `32` | reject zero, clamp high | new text-call session or new turn | Backpressure cap. |
| `text_call.media_ready_timeout_ms` | `DurationMs` | `1000..120000` | `20000` | clamp to range | new playback request | Setup wait policy. |
| `text_call.playback_wait_timeout_ms` | `DurationMs` | `1000..600000` | `180000` | clamp to range | new playback request | Hung playback detection. |
| `text_call.latest_response_wins` | `bool` | `true,false` | `true` | reject non-bool | new agent turn | Cancel-and-replace policy. |
| `text_call.callback_timeout_ms` | `DurationMs` | `100..60000` | `5000` | clamp to range | new callback attempt | Subscriber responsiveness. |
| `tts.chunking_enabled` | `bool` | `true,false` | `true` | reject non-bool | new playback request | Enables sentence-packed text splitting before TTS; off synthesizes the full response as one chunk. |
| `tts.max_text_chunk_chars` | `Count` | `40..500` | `90` | clamp to range | new playback request | Packs complete sentence segments up to this size before falling back to word splits for oversized segments. |
| `tts.first_chunk_max_chars` | `Count` | `0` or `40..500` | `0` | `0` disables, otherwise clamp to range | new playback request | Optional sentence-boundary first-chunk ramp for pipelining streaming LLM output into TTS. |
| `tts.prebuffer_chunks` | `Count` | `1..64` | `1` | clamp to range | new playback request | Configured prepared text chunks required before playback starts; the live default starts after one prepared chunk for lower first-audio latency, with mid-utterance starvation tracked by outbound pacing rollups. |
| `barge_in.enabled` | `bool` | `true,false` | `true` | reject non-bool | next ASR session | Enables barge-in path. |
| `barge_in.speech_onset_cancel_enabled` | `bool` | `true,false` | `true` | reject non-bool | next ASR session | Speech onset cancel path; while assistant playback is active, media-level cancellation defers to partial/final ASR confirmation so assistant echo can be suppressed instead of cutting itself off. |
| `barge_in.partial_asr_cancel_enabled` | `bool` | `true,false` | `true` | reject non-bool | next ASR session | Partial ASR cancel path. |
| `barge_in.final_asr_cancel_enabled` | `bool` | `true,false` | `true` | reject non-bool | next ASR session | Final ASR cancel path. |
| `barge_in.clear_timeout_ms` | `DurationMs` | `100..10000` | `1000` | clamp to range | new cancel request | Clear/terminal wait. |
| `logging.enabled` | `bool` | `true,false` | `false` | reject non-bool | immediate | Enables writer, not media blocking. |
| `logging.queue_capacity` | `Count` | `1024..1048576` | `65536` | clamp to range | next writer start | Bounded event queue size. |
| `logging.per_frame_sample_rate` | `Ratio` | `0.0..1.0` | `0.0` | reject NaN, clamp to range | immediate | `asr.ingest_frame` sampling. |
| `logging.include_transcript_text` | `bool` | `true,false` | `false` | reject non-bool | next event | Sensitive transcript opt-in. |
| `logging.redaction_mode` | `RedactionMode` enum | `metrics_only`, `hashed_text`, `redacted_text`, `sensitive_plaintext` | `metrics_only` | reject unsafe upload combos | next event | Text privacy mode. |
| `quality_judge.enabled` | `bool` | `true,false` | `false` | reject non-bool | report layer only | Enables offline/live-sampled judging. |
| `quality_judge.mode` | `JudgeMode` enum | `offline`, `live_sample` | `offline` | reject unknown | report layer only | Judge scheduling. |
| `quality_judge.sample_rate` | `Ratio` | `0.0..1.0` | `0.0` | reject NaN, clamp to range | report layer only | Live sample fraction. |
| `quality_judge.model` | `String` | non-empty if enabled | `default` | reject empty when enabled | next judge job | Judge model. |
| `quality_judge.batch_size` | `Count` | `1..1000` | `20` | clamp to range | next judge job | Offline throughput. |
| `quality_judge.timeout_ms` | `DurationMs` | `1000..120000` | `30000` | clamp to range | next judge job | Judge request timeout. |
| `targets.*` | threshold types | per target | profile-defined | reject impossible rates, clamp durations | report only | Pass/fail and recommendations. |

Report-only keys are advisory. They may affect labels, recommendations, and Markdown/JSON reports, but they must not alter live ASR finalization, live `caller.turn` emission, M4 WebSocket messages, playback ordering, or hangup behavior.

### Telnyx Agent-Owned Bridge Config

The gateway must not become the second owner of telnyx-agent bridge behavior. The agent daemon owns bridge knobs; the gateway may proxy status or include them in reports when the agent exposes them.

| Agent bridge key | Owner | Report use |
|---|---|---|
| `reply_timeout_ms` | `bins/telnyx-agent` | local agent responsiveness. |
| `input_quiet_for_ms` | `bins/telnyx-agent` | intentional tmux quiet-window delay. |
| `input_delivery_timeout_ms` | `bins/telnyx-agent` | bounded admission/delivery failure instead of indefinite caller silence while the target stays active. |
| `input_backoff_initial_ms` / `input_backoff_max_ms` | `bins/telnyx-agent` | queued transcription backoff delay. |
| `trailing_enter_enabled` | `bins/telnyx-agent` | prompt submission behavior. |
| `trailing_enter_delay_ms` | `bins/telnyx-agent` | intentional submit delay. |

## Coherent Config Framework

Proposed TOML:

```toml
[voice_quality]
profile = "balanced"

[voice_quality.speech]
rms_threshold = 220.0
peak_threshold = 1100
onset_min_silence_ms = 180

[voice_quality.endpoint]
trailing_silence_ms = 650
min_turn_words = 2
min_turn_chars = 6
merge_window_ms = 350
max_turn_words = 80
max_turn_duration_ms = 12000

[voice_quality.asr]
finish_pad_ms = 160
repeated_token_run_threshold = 16
repeated_q_run_threshold = 8

[voice_quality.tts]
chunking_enabled = true
max_text_chunk_chars = 90
first_chunk_max_chars = 0
prebuffer_chunks = 1

[voice_quality.text_call]
max_active_turns = 32
media_ready_timeout_ms = 20000
playback_wait_timeout_ms = 180000
latest_response_wins = true
callback_timeout_ms = 5000

[voice_quality.barge_in]
enabled = true
speech_onset_cancel_enabled = true
partial_asr_cancel_enabled = true
final_asr_cancel_enabled = true
clear_timeout_ms = 1000

[voice_quality.logging]
enabled = false
queue_capacity = 65536
per_frame_sample_rate = 0.0
include_transcript_text = false
redaction_mode = "metrics_only"

[voice_quality.quality_judge]
enabled = false
mode = "offline"
sample_rate = 0.0
model = "default"
batch_size = 20
timeout_ms = 30000

[voice_quality.targets]
p50_endpoint_trailing_silence_ms = 900
p95_endpoint_trailing_silence_ms = 1300
p50_turn_to_playback_started_ms = 1200
p95_turn_to_playback_started_ms = 2500
max_incomplete_turn_rate = 0.05
max_overmerged_turn_rate = 0.05
max_garbled_turn_rate = 0.03
max_inappropriate_cancel_rate = 0.03
```

Bootstrap precedence:

1. Code defaults.
2. Profile defaults selected by `--quality-profile` or TOML profile.
3. `--quality-config <path>` TOML overrides.
4. `--load <state>` replayed state lines.
5. CLI flag overrides.

Post-boot precedence:

- The process has one resolved `VoiceQualityConfig` plus a current `config_id`.
- `quality ...` REPL/socket mutations validate against typed domains, then produce a new resolved config and `config_id`.
- Mutations are applied at the live-safe boundary declared in the knob table.
- Accepted mutations emit `call.config.snapshot` for active calls where the config will apply.
- Rejected mutations return a typed validation error and do not change the current config.

Config hashing:

- `config_id` is a hash of canonical serialized resolved config.
- Hash input excludes transient paths such as output log filenames unless the path changes behavior.
- Reports must join every span to exactly one `config_id`; missing config snapshots are report errors.

## Persistence and Parity Tests

State dump/load must use replayable line-oriented commands, not a separate private config format.

Example dump lines:

```text
quality profile balanced
quality speech rms-threshold 220
quality speech peak-threshold 1100
quality speech onset-min-silence-ms 180
quality endpoint trailing-silence-ms 650
quality endpoint min-turn-words 2
quality endpoint min-turn-chars 6
quality endpoint merge-window-ms 350
quality endpoint max-turn-words 80
quality endpoint max-turn-duration-ms 12000
quality asr finish-pad-ms 160
quality asr repeated-token-run-threshold 16
quality asr repeated-q-run-threshold 8
quality tts first-chunk-max-chars 0
quality text-call max-active-turns 32
quality text-call latest-response-wins on
quality logging include-transcript-text off
quality logging redaction-mode metrics-only
quality judge off
```

Required round-trip tests:

| Test | Requirement |
|---|---|
| CLI to resolved config | CLI flags produce the same `VoiceQualityConfig` as equivalent TOML plus state lines. |
| REPL to state dump | Mutating via REPL then dumping state emits replay lines that recreate the same `config_id`. |
| Socket to state dump | Mutating via socket command then dumping state emits equivalent replay lines. |
| State load parity | Loading dumped lines after TOML and before CLI overrides follows the bootstrap precedence exactly. |
| Validation parity | Invalid values are rejected or clamped the same way from CLI, REPL, socket, and state load. |
| Boundary parity | Hot reload does not affect the active ASR session; it takes effect on the next `asr.session.started`. |

## Proposed CLI Surface

Gateway startup:

```text
telnyx-gateway \
  --quality-config ./telnyx-quality.toml \
  --quality-profile balanced \
  --endpoint-trailing-silence-ms 650 \
  --speech-rms-threshold 220 \
  --speech-peak-threshold 1100 \
  --speech-onset-min-silence-ms 180 \
  --turn-log-jsonl ./turns.jsonl \
  --quality-report-json ./quality-report.json
```

Offline analysis:

```text
telnyx-gateway turn-report ./turns.jsonl \
  --output-json ./turn-report.json \
  --output-md ./turn-report.md

telnyx-gateway turn-judge ./turns.jsonl \
  --model <judge-model> \
  --sample-rate 1.0 \
  --output-json ./turn-judge.json

telnyx-gateway tune-endpoint ./turns.jsonl \
  --candidate-trailing-silence-ms 500,650,800,950,1100 \
  --candidate-rms-threshold 140,180,220 \
  --candidate-peak-threshold 700,900,1200 \
  --objective balanced \
  --output-json ./tuning-recommendations.json
```

Harness profiling:

```text
telnyx-gateway profile-roundtrip \
  --harness gateway-echo \
  --turn-log-jsonl ./gateway-echo-turns.jsonl \
  --report-json ./gateway-echo-report.json

telnyx-gateway profile-roundtrip \
  --harness text-call-echo \
  --app-url <websocket-url> \
  --turn-log-jsonl ./text-call-echo-turns.jsonl \
  --report-json ./text-call-echo-report.json

telnyx-gateway profile-roundtrip \
  --harness agent-tmux-echo \
  --agent-socket <path> \
  --turn-log-jsonl ./agent-tmux-echo-turns.jsonl \
  --report-json ./agent-tmux-echo-report.json
```

Replay/corpus extension:

```text
telnyx-gateway replay-capture <capture-dir> \
  --backend kroko-2025 \
  --chunk-ms 20 \
  --trailing-silence-pad-ms 800 \
  --turn-log-jsonl ./replay-turns.jsonl

telnyx-gateway asr-golden-ab <manifest> \
  --backend sherpa-2023 \
  --backend kroko-2025 \
  --codec pcmu-8k \
  --codec l16-16k \
  --chunk-ms 20 \
  --trailing-silence-pad-ms 800 \
  --turn-judge \
  --output-json ./asr-ab-quality.json
```

## Proposed REPL Commands

`quality ...` is a normal gateway command namespace handled by the existing command dispatcher.

```text
quality status
quality profile fast|balanced|complete|noisy
quality endpoint status
quality endpoint trailing-silence-ms <ms>
quality endpoint min-turn-words <n>
quality endpoint min-turn-chars <n>
quality endpoint merge-window-ms <ms>
quality endpoint max-turn-words <n>
quality endpoint max-turn-duration-ms <ms>
quality speech status
quality speech rms-threshold <value>
quality speech peak-threshold <value>
quality speech onset-min-silence-ms <ms>
quality asr status
quality asr finish-pad-ms <ms>
quality asr repeated-token-run-threshold <n>
quality asr repeated-q-run-threshold <n>
quality text-call status
quality text-call max-active-turns <n>
quality text-call media-ready-timeout-ms <ms>
quality text-call playback-wait-timeout-ms <ms>
quality text-call latest-response-wins on|off
quality text-call callback-timeout-ms <ms>
quality tts status
quality tts chunking on|off
quality tts max-text-chunk-chars <n>
quality tts first-chunk-max-chars <n>
quality tts prebuffer-chunks <n>
quality logging status
quality logging on <path>
quality logging off
quality logging include-transcript-text on|off
quality logging redaction-mode metrics-only|hashed-text|redacted-text|sensitive-plaintext
quality judge status
quality judge on --sample-rate <rate> --model <model>
quality judge off
quality report [path]
quality recommendations [path]
quality profile-roundtrip gateway-echo [path]
quality profile-roundtrip text-call-echo [path]
quality profile-roundtrip agent-tmux-echo [path]
```

Existing command to preserve and integrate into reports:

```text
conversation barge-in on|off|status
```

Proposed extension:

```text
quality barge-in status
quality barge-in on|off
quality barge-in speech-onset on|off
quality barge-in partial-asr on|off
quality barge-in final-asr on|off
quality barge-in clear-timeout-ms <ms>
```

## Operator TUI Command History

Live M6 tuning is iterative: operators often adjust one parameter, place another call, then adjust the same command again. The operator TUI shell must support command-input recall for this workflow.

Implementation requirements:

- Reuse `motlie-driver::HistoryBuffer` for command storage and recall cursor behavior.
- Bind `KeyCode::Up` and `KeyCode::Down` in the gateway TUI shell input to `HistoryBuffer::prev()` and `HistoryBuffer::next()`.
- Do not maintain a parallel history cursor or bespoke recall buffer in `operator/tui.rs`.
- Record only submitted non-empty command lines.
- Reset recall when the operator edits the input after recalling a command.
- Leave REPL and socket command dispatch line-oriented and unchanged.

This supports call-to-call tuning loops such as:

```text
quality endpoint trailing-silence-ms 650
quality speech rms-threshold 220
quality barge-in clear-timeout-ms 750
quality tts chunking off
```

## Proposed Socket Commands

The gateway command socket remains line-oriented. Clients send the same command line accepted by the REPL, terminated by newline. Do not add a JSON `{id, command}` envelope for M6.

Gateway socket input examples:

```text
quality status
quality endpoint trailing-silence-ms 950
quality speech rms-threshold 220
quality barge-in clear-timeout-ms 750
quality tts chunking off
quality logging on /tmp/motlie-turns.jsonl
quality report /tmp/motlie-quality-report.json
quality recommendations /tmp/motlie-tuning.json
quality profile-roundtrip gateway-echo /tmp/gateway-echo-report.json
```

Expected typed outputs are normal `GatewayCommand` results, for example:

```text
QualityStatus { profile: "balanced", config_id: "cfg_...", logging_enabled: true }
QualityConfigChanged { key: "endpoint.trailing_silence_ms", value: 950, config_id: "cfg_...", applies: "next_asr_session" }
QualityValidationError { key: "speech.rms_threshold", reason: "not a finite number" }
QualityReportStarted { path: "/tmp/motlie-quality-report.json" }
```

Agent daemon socket examples remain line-oriented and agent-owned:

```text
status
bridge input-quiet-for-ms 8000
bridge backoff initial-ms 250 max-ms 5000
bridge trailing-enter delay-ms 750
bridge trailing-enter off
gateway quality status
gateway quality endpoint trailing-silence-ms 950
```

Expected agent status fields:

```json
{
  "bridge": {
    "reply_timeout_ms": 120000,
    "input_quiet_for_ms": 10000,
    "input_backoff_initial_ms": 250,
    "input_backoff_max_ms": 5000,
    "trailing_enter_enabled": true,
    "trailing_enter_delay_ms": 750
  },
  "gateway_quality": {
    "profile": "balanced",
    "config_id": "cfg_...",
    "endpoint_trailing_silence_ms": 800
  }
}
```

The gateway may proxy `gateway quality ...` from the agent daemon to the gateway socket, but ownership remains separated: bridge config is agent-owned; voice quality config is gateway-owned.

## Analysis Workflow

Offline replay:

1. Capture or replay calls.
2. Emit normalized JSONL artifacts for calls, turns, spans, playbacks, barge-in events, labels, and reports.
3. Run deterministic `turn-report`.
4. Run LLM `turn-judge` if transcript text may be analyzed.
5. Run tuning recommendation generator.
6. Compare candidate configs on fixed/replayed audio or per-category paired deltas.
7. Promote a profile only when it improves quality without violating latency targets.

Live sampling:

1. Enable turn logging locally.
2. Keep `include_transcript_text = false` unless a sensitive local artifact is explicitly approved.
3. Optionally enable sampled LLM judging at the report layer.
4. Review daily or weekly reports.
5. Apply runtime config changes through REPL/socket.
6. Persist state as replayable `quality ...` lines.
7. Compare before/after windows with denominators, exclusions, and confounders.

Example live flow:

```text
quality logging on ./logs/telnyx-turns.jsonl
quality judge on --sample-rate 0.05 --model <judge-model>
quality report ./reports/telnyx-quality-current.json
quality recommendations ./reports/telnyx-quality-recommendations.json
quality endpoint trailing-silence-ms 950
state dump ./state-after-quality-tune.repl
```

## Optimization Strategy

Candidate objective function:

```text
score =
  0.30 * endpointing_quality
+ 0.25 * asr_text_quality
+ 0.20 * conversation_realism_quality
+ 0.15 * latency_score
+ 0.10 * stability_score
```

Where:

- `endpointing_quality` comes from labels and deterministic premature/overmerged rates.
- `asr_text_quality` comes from WER when references exist, otherwise labels and suppression rates.
- `conversation_realism_quality` uses barge-in appropriateness, playback cancel/supersede rates, and response start latency.
- `latency_score` uses critical-path categories and excludes concurrent diagnostic spans from percentages.
- `stability_score` penalizes invalid turns, failures, dropped required events, and high suppression/hallucination rates.
- Transport confounders stratify or exclude samples before attributing poor quality to ASR/endpointing.

Recommended report output:

```json
{
  "recommended_profile": "balanced",
  "recommendations_only": true,
  "changes": [
    {
      "knob": "endpoint.trailing_silence_ms",
      "from": 800,
      "to": 950,
      "confidence": "medium",
      "expected_effect": "reduce premature cuts with modest p95 latency increase",
      "apply_boundary": "next_asr_session"
    },
    {
      "knob": "speech.rms_threshold",
      "from": 180,
      "to": 220,
      "confidence": "low",
      "expected_effect": "reduce false speech starts on noisy calls; validate on more samples",
      "apply_boundary": "next_asr_session"
    }
  ]
}
```

Reports and judges emit recommendations. Operators or automation outside the judge may choose to run `quality ...` commands, but the judge must not directly mutate live config.

## Implementation Phases

### Phase 1 - Span and Turn Logging

- [ ] Add stable event envelope/versioning and JSON serialization.
- [ ] Add `call.config.snapshot` at stream start and on accepted live config changes; spans carry `config_id` only.
- [ ] Add `asr_session_id`/`utterance_id` and emit `asr.turn_mapped` when `caller.turn` is created.
- [ ] Add non-blocking bounded event sink with dropped-event counters and writer error events.
- [ ] Emit `text_call.caller_turn.sent` on successful caller turn send.
- [ ] Emit app-agent, playback, endpointing, barge-in, inbound transport, and outbound pacing events.
- [ ] Add span links for barge-in canceling prior playback.
- [ ] Ensure logs do not contain raw phone numbers and transcript text defaults off.

### Phase 2 - Deterministic Harnesses

- [ ] Add `gateway-echo` profile report.
- [ ] Add `text-call-echo` app harness.
- [ ] Add `agent-tmux-echo` deterministic tmux harness.
- [ ] Compare harness modes only on fixed/replayed audio or per-category paired deltas.
- [ ] Ensure worked reports reconcile critical-path sums.

### Phase 3 - Config Framework

- [ ] Add `VoiceQualityConfig` sections for speech, endpoint, ASR, text-call, barge-in, logging, judge, and targets.
- [ ] Implement typed domains, min/max validation, clamp/reject behavior, and live-safe apply boundaries.
- [ ] Wire CLI overrides using bootstrap precedence.
- [ ] Wire REPL/socket `quality ...` commands through the existing line-oriented command dispatcher.
- [ ] Persist config through replayable state dump/load lines.
- [ ] Keep gateway-owned quality config separate from agent-owned bridge config.
- [ ] Replace live hard-coded constants with config-backed values while preserving defaults and M4 protocol behavior.

### Phase 4 - Reports and LLM Judge

- [ ] Add deterministic `turn-report` over normalized JSONL artifacts.
- [ ] Add strict JSON `turn-judge` that emits labels and recommendations only.
- [ ] Add span-aware critical-path latency attribution in judge input/output.
- [ ] Add WER/reference hooks and label schema.
- [ ] Add tuning recommendations with denominators, exclusions, confounders, and confidence.
- [ ] Support redacted/no-transcript privacy modes.

## Acceptance Criteria

M6 profiling is complete when:

- [ ] A live or replayed call can produce normalized structured JSONL artifacts for calls, turns, spans, playbacks, barge-in events, labels, and reports.
- [ ] Every event/report/judge output includes schema/run/config/version metadata needed for replay.
- [ ] A `call.config.snapshot` is emitted at stream start and on live config changes; spans reference `config_id` only.
- [ ] Hot-reloaded ASR/endpointing config applies only at the next ASR session/utterance boundary.
- [ ] `include_transcript_text` defaults to `false`; transcript text requires explicit sensitive opt-in and records redaction mode.
- [ ] `asr_session_id`/`utterance_id` join pre-turn events to the eventual `turn_id` through `asr.turn_mapped`.
- [ ] Span durations are computed from monotonic `Instant` deltas, not wall-clock timestamps.
- [ ] The event sink is bounded and non-blocking on realtime media paths; forced-full queue tests prove drops are counted and media processing does not block.
- [ ] Per-frame `asr.ingest_frame` spans are off by default and sampled when enabled.
- [ ] Critical-path reports satisfy `sum(serial span durations) == named round-trip total`; concurrent spans are flagged and excluded from percentages.
- [ ] Caller speaking time is reported as `caller_speech`, not endpointing, and is not treated as a tunable endpoint delay.
- [ ] Inbound RTP loss/stale/reorder/jitter and outbound pacing/underrun rollups are emitted and surfaced as confounders/exclusions.
- [ ] `gateway-echo` profiling separates caller speech, endpointing, ASR final flush, TTS first chunk, packetization, and first media frame timings.
- [ ] Operator TUI command input supports Up/Down recall through `motlie-driver::HistoryBuffer`, so live tuning commands can be repeated without retyping.
- [ ] `text-call-echo` profiling separates M4 WebSocket/app protocol overhead from deterministic harness work, and real text-call sessions emit `app.agent_turn_wait` plus `turn.finalize_to_first_audio` using the final transcript boundary.
- [ ] `agent-tmux-echo` profiling separates activity quiet-window wait, bounded delivery timeout, backoff, trailing Enter delay, tmux injection, reply wait, and scrape latency.
- [ ] `real-agent` profiling attributes incremental latency using fixed/replayed audio or per-category paired deltas, not noisy live whole-call subtraction.
- [ ] Barge-in reports link speech onset/partial/final ASR to cancel request and playback terminal `canceled` status.
- [ ] LLM judge output includes reproducibility metadata, anchored rubric scores, confidence, uncertainty/abstention, evidence references, dominant latency category, and recommended next probe.
- [ ] WER/reference runs can store reference IDs, scorer version, and alignment metadata; LLM labels are stored as labels, not truth.
- [ ] `quality ...` REPL/socket commands use the existing line-oriented dispatcher and typed outputs, not a new JSON command envelope.
- [ ] CLI, REPL, socket, and state dump/load paths produce equivalent validated `VoiceQualityConfig` and `config_id` values.
- [ ] Gateway-owned quality config remains separate from telnyx-agent-owned bridge config; gateway may proxy/report agent bridge status but does not own it.
- [ ] Reports can answer: "Was this turn slow because of caller speech duration, ASR endpointing, agent generation, intentional quiet-window delay, queueing, TTS, media transport, or a transport confounder?"
- [ ] All logs and reports obey phone-number redaction requirements.

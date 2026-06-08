# Telnyx M6 Profiling and Turn Quality Analysis

## Status

Draft for Telnyx M6 (#418), authored by @codex-367-design.

This document specifies the profiling, logging, and tuning framework needed to analyze and improve live Telnyx voice conversation quality after M4 text-call integration and M5 conversational realism. It is intentionally more detailed than a checklist: implementers should be able to derive event schemas, span boundaries, CLI/REPL/socket surfaces, test cases, and acceptance criteria directly from this document.

Related issues:

- Telnyx umbrella: #358
- M4 external text-call integration: #367
- M5 conversational realism: #402
- M6 turn logging, quality config, and LLM judge: #418

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-08 | @codex-367-design | Added the M6 profiling specification: span boundaries, deterministic `I heard` harnesses, structured turn logs, latency categories, LLM judge inputs/outputs, tunable config surfaces, and acceptance criteria. |

## Purpose

M6 answers one practical question: when a live voice turn feels slow, choppy, incomplete, or unrealistic, which layer caused it?

A single metric such as `caller_turn_to_playback_started_ms = 1800` is not enough. The gateway must distinguish:

- acoustic speech duration
- endpointing wait after the caller stops talking
- ASR finalization and transcript quality
- gateway dispatch overhead
- text-call WebSocket/app transport overhead
- app or LLM generation latency
- intentional harness delays such as tmux quiet-window waits
- queue waits such as media-not-ready or active playback replacement
- TTS first-audio latency
- media packetization and outbound frame send latency
- barge-in cancel latency and playback terminal status

M6 should produce reports that say, for example:

```text
This turn was slow because 61% of the latency was endpointing wait and 27% was TTS first-chunk synthesis. App generation was not the bottleneck.
```

Or:

```text
This agent-tmux run was slow because the configured input quiet-window intentionally waited 10000 ms before injecting the caller text.
```

## Scope

M6 covers two related but distinct models.

| Model | Description | Why it matters |
|---|---|---|
| Strict text-call turn model | M4 app-agent protocol: final ASR text becomes `caller.turn`; app replies with `agent.turn`; gateway emits `playback.started` and `playback.finished`. | This is the stable app integration contract. |
| Full-duplex realism model | M5 behavior: partial ASR and frame-level speech onset can cancel active playback before a final `caller.turn` exists. | This is how live conversation feels responsive and interruptible. |

M6 must profile both. Reports must not force barge-in events into a purely request/response shape. Barge-in spans can overlap the prior playback span and the next caller turn.

## Non-Goals

- Do not change the M4 app-agent protocol to emit partial ASR by default.
- Do not require LLM judging for normal live call operation.
- Do not use LLM output as the only source of truth when golden references or WER data exist.
- Do not introduce a second gateway control socket; reuse the existing command dispatcher/socket mux.
- Do not log raw phone numbers or use them as IDs, keys, subscription IDs, filenames, or report grouping values.

## Current Behavior Summary

The gateway already emits several useful signals, but M6 requires a coherent lifecycle log around turns and spans.

| Current signal | Current behavior | Usefulness | M6 gap |
|---|---|---|---|
| `media.speech.detected` | Emitted when speech energy first appears after quiet. | Start-of-speech and speech threshold tuning. | Not linked to a `turn_id` yet. |
| `media.frame.local_endpoint` | Debug-level low-energy tail logging after speech. | Endpoint tail diagnosis. | Needs span boundaries and config values. |
| `asr.local_endpoint.finalizing` | Local endpoint gate decides to finish the ASR session. | Endpoint decision timing. | Needs link to final transcript and caller turn. |
| `transcript.partial` | ASR partial transcript. | Partial quality and barge-in analysis. | Not normally sent to M4 app-agent protocol. |
| `transcript.final` | Final ASR transcript. | Best current proxy for caller-turn text. | Does not include the future `turn_id`. |
| `transcript.suppressed_repeated_token` | ASR adapter suppression for repeated-token hallucination. | ASR stability signal. | Suppression thresholds are not config-backed. |
| `text_call.caller_turn.forward_failed` | Failure to send a caller turn to the app stream. | Reliability. | Successful caller turns need structured events. |
| `playback.finished.status` | Terminal app-agent playback state. | Completed/canceled/failed/superseded lifecycle. | Needs latency span correlation. |
| `tts.speak.chunk_queued` | TTS chunk packetization and queueing. | TTS startup and chunking behavior. | Needs turn-level aggregation. |

## Definitions

| Term | Definition |
|---|---|
| Caller utterance | Acoustic speech segment from first speech-energy frame until low-energy tail begins. |
| Endpoint wait | Intentional or backend-driven wait after low-energy tail starts until finalization. |
| Final transcript | ASR event considered final by backend or local endpoint finish. |
| Caller turn | M4 protocol text unit sent to the app as `caller.turn`; created from final transcript text. |
| Agent turn | App response text sent as `agent.turn` and correlated by `turn_id`. |
| Playback | TTS/media job created from an accepted `agent.turn`. |
| Barge-in | Caller speech/partial/final transcript during active playback that cancels or replaces speech. |
| Harness mode | A controlled profiling mode such as gateway echo, text-call echo, tmux echo, or real agent. |
| Intentional delay | A deliberate wait such as endpoint trailing silence, tmux quiet-window, or trailing Enter delay. |
| Queue wait | A wait caused by resource state, such as media not ready or active playback conflict. |

## Span Event Model

A span is a named timed interval. Every M6 latency measurement should be expressed as a span or as an aggregate of spans.

Required span event shape:

```json
{
  "event": "voice.span.finished",
  "span_id": "span_...",
  "parent_span_id": "span_...",
  "gateway_call_id": "gwc_...",
  "turn_id": "turn_...",
  "playback_id": "tts_...",
  "name": "asr.endpoint_wait",
  "category": "endpointing",
  "started_at": "2026-06-08T20:44:00.000Z",
  "finished_at": "2026-06-08T20:44:00.820Z",
  "duration_ms": 820,
  "status": "completed",
  "wait_reason": "trailing_silence_threshold",
  "config": {
    "endpoint_trailing_silence_ms": 800,
    "speech_rms_threshold": 180.0,
    "speech_peak_threshold": 900
  }
}
```

Required fields:

| Field | Requirement |
|---|---|
| `event` | Stable event name, usually `voice.span.finished`. |
| `span_id` | Opaque ID for this span. |
| `parent_span_id` | Optional parent span. |
| `gateway_call_id` | Required call correlation key. |
| `turn_id` | Required when a span belongs to a caller/agent turn. |
| `playback_id` | Required when a span belongs to TTS/media playback. |
| `name` | Stable span name from the tables below. |
| `category` | One of the M6 latency categories. |
| `started_at` / `finished_at` | Wall-clock timestamps. |
| `duration_ms` | Derived duration. |
| `status` | `completed`, `canceled`, `failed`, `superseded`, or `timeout`. |
| `wait_reason` | Null except for intentional delay or queue wait spans. |
| `config` | Relevant tunable values active during the span. |

Privacy requirements:

- No raw phone numbers.
- No routing values as IDs or grouping keys.
- Transcript text logs are sensitive and must be explicitly enabled or stored in local/operator-controlled artifacts.
- Redacted mode must preserve metrics while dropping or hashing raw text.

## Latency Categories

| Category | Meaning | Examples |
|---|---|---|
| `endpointing` | Waiting for the caller turn boundary. | trailing silence after speech, backend final wait |
| `asr_generation` | ASR backend work. | frame ingest, ASR session finish |
| `gateway_overhead` | Gateway-local dispatch and state work. | transcript to turn, frame serialization, turn correlation |
| `network_transport` | HTTP/WebSocket transport outside model work. | app WebSocket round trip |
| `model_generation` | LLM/app/agent text generation. | real agent response, handler call |
| `intentional_delay` | Configured deliberate wait. | tmux quiet-window, trailing Enter delay |
| `queue_wait` | Resource or ordering wait. | media not ready, active playback replacement |
| `tts_generation` | TTS model work. | first audio chunk, full synthesis |
| `media_packetization` | Resample, packetize, chunk queue. | first packet prep |
| `playback_transport` | Outbound media send and playback terminal behavior. | first frame sent, mark/clear terminal |
| `barge_in` | Interrupt detection and cancellation path. | speech onset to clear request |
| `harness` | Deterministic test harness behavior. | echo response construction |

## Acoustic, ASR, and Endpointing Spans

| Span | Start | End | Category | Purpose |
|---|---|---|---|---|
| `utterance.speech_to_low_energy` | first speech-energy frame after quiet | first low-energy frame after speech | `endpointing` | Approximate actual spoken portion. |
| `asr.endpoint_wait` | first low-energy tail frame after speech | local endpoint decision or backend final decision | `endpointing` / `intentional_delay` | Separate deliberate endpoint delay from model work. |
| `asr.local_finish` | call to finish active ASR session | final transcript events returned | `asr_generation` | Measure ASR flush/finalization cost. |
| `asr.ingest_frame` | frame ingest starts | frame ingest returns transcript events | `asr_generation` | Optional sampled frame-level backend cost. |
| `asr.final_to_turn` | final transcript recorded | `caller.turn` created/sent | `gateway_overhead` | Measure gateway handoff overhead. |

Important boundary rule:

- The start of `asr.endpoint_wait` is the first low-energy frame after speech began.
- The end is the finalization decision, not the later transcript-to-turn handoff.
- This makes endpointing wait separable from ASR final flush and gateway dispatch.

## Text-Call and App-Agent Spans

| Span | Start | End | Category | Purpose |
|---|---|---|---|---|
| `text_call.send_caller_turn` | caller turn frame serialization begins | WebSocket frame send completes | `gateway_overhead` / `network_transport` | M4 outbound text protocol overhead. |
| `app.agent_turn_wait` | `caller.turn` sent | `agent.turn` received | `model_generation` / `harness` | App response latency envelope. |
| `app.generation` | app receives turn | app creates response text | `model_generation` / `harness` | Requires app/agent instrumentation when available. |
| `text_call.agent_turn_accept` | `agent.turn` received | TTS queue request begins | `gateway_overhead` | Turn correlation and policy work. |
| `text_call.supersede_stale_turn` | stale valid `agent.turn` received | `playback.finished status=superseded` sent | `gateway_overhead` | Validates stale-but-valid behavior. |

The strict M4 protocol remains turn-based. The gateway should not require apps to stream tokens to satisfy M6.

## TTS and Playback Spans

| Span | Start | End | Category | Purpose |
|---|---|---|---|---|
| `tts.media_ready_wait` | queue requested while media is not ready | media becomes ready or timeout | `queue_wait` | Distinguish setup waiting from TTS generation. |
| `tts.conflict_cancel_replace` | new speech replaces active playback | prior playback terminal `canceled` emitted | `barge_in` / `queue_wait` | Measure latest-response-wins behavior. |
| `tts.synthesis_first_chunk` | TTS synthesis starts | first audio chunk available | `tts_generation` | Time to first audio. |
| `tts.synthesis_full` | TTS synthesis starts | all chunks available | `tts_generation` | Full generation cost. |
| `tts.packetize_first_chunk` | first audio chunk available | first media packet queued | `media_packetization` | Resample/packetization overhead. |
| `media.first_frame_send` | first media packet queued | first outbound media frame sent | `playback_transport` | Queue-to-wire latency. |
| `media.playback_terminal` | playback started | mark/clear/failure terminal status | `playback_transport` | Completion/cancel/failure timing. |

## Barge-In and Full-Duplex Spans

M5 barge-in is not strictly turn-bound. The caller may start speaking while the prior `agent.turn` playback is active, before a new final transcript exists. M6 must preserve this causality.

| Span | Start | End | Category | Purpose |
|---|---|---|---|---|
| `barge_in.speech_onset_to_cancel_request` | speech onset while playback active | cancel/clear requested | `barge_in` | Perceived interruption responsiveness. |
| `barge_in.cancel_request_to_terminal` | cancel/clear requested | playback terminal `canceled` observed | `playback_transport` | How fast playback actually stops. |
| `barge_in.partial_to_cancel_request` | meaningful partial ASR while playback active | cancel/clear requested | `barge_in` | Partial-ASR interruption path. |
| `barge_in.final_to_regenerate` | final transcript after interruption | new response queued | `model_generation` / `gateway_overhead` | Regeneration latency. |

Use span links for overlapping causality:

```json
{
  "event": "voice.span.link",
  "from_span": "barge_in.speech_onset_to_cancel_request",
  "to_span": "media.playback_terminal",
  "relationship": "cancels_playback",
  "playback_id": "tts_..."
}
```

## Deterministic `I Heard` Harness

The deterministic `I heard: <caller text>` repeat handler is a required M6 profiling harness. Its response generation is predictable, so it isolates infrastructure latency before real LLM/agent generation enters the loop.

Required harness modes:

| Mode | Handler | What it isolates |
|---|---|---|
| `gateway-echo` | Gateway-local conversation smoke-test handler. | ASR endpointing + gateway conversation path + TTS + media, with no M4 app WebSocket. |
| `text-call-echo` | External app immediately returns `I heard: <text>` over the M4 text WebSocket. | M4 text-call protocol overhead plus baseline TTS/media. |
| `agent-tmux-echo` | `bins/telnyx-agent` bridges to a deterministic local tmux echo prompt. | motlie-tmux injection, activity wait, trailing Enter, monitor/history scrape overhead. |
| `real-agent` | Actual app/agent. | Full app/model/tmux generation latency. |

Profiling formulas:

```text
text-call protocol overhead = text-call-echo - gateway-echo
agent bridge overhead       = agent-tmux-echo - text-call-echo
real generation overhead    = real-agent - agent-tmux-echo
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

| Span | Applies to | Category | Purpose |
|---|---|---|---|
| `harness.echo_generate` | gateway/text-call echo | `harness` | Deterministic response construction; should be near-zero. |
| `harness.http_ws_round_trip` | text-call echo | `network_transport` | App protocol loop overhead. |
| `harness.tmux_input_wait` | agent-tmux echo | `intentional_delay` | Quiet-window wait before keystroke injection. |
| `harness.tmux_backoff_wait` | agent-tmux echo | `intentional_delay` | Backoff while target is active. |
| `harness.tmux_inject` | agent-tmux echo | `gateway_overhead` | Send keystrokes through motlie-tmux. |
| `harness.trailing_enter_delay` | agent-tmux echo | `intentional_delay` | Configured submit delay. |
| `harness.tmux_reply_wait` | agent-tmux echo | `model_generation` / `harness` | Wait for deterministic reply to render. |
| `harness.tmux_scrape` | agent-tmux echo | `gateway_overhead` | Monitor/history scrape and reply extraction. |

## Round-Trip Definitions

Different teams use “round trip” differently. M6 reports must name the exact boundary.

| Round trip | Start | End | Use |
|---|---|---|---|
| `speech_to_first_audio` | first speech-energy frame | first outbound assistant media frame sent | User-perceived full turn latency. |
| `low_energy_to_first_audio` | first low-energy frame after speech | first outbound assistant media frame sent | Endpointing plus response stack. |
| `final_transcript_to_first_audio` | final transcript event | first outbound assistant media frame sent | Excludes endpointing wait. |
| `caller_turn_to_playback_started` | `caller.turn` sent | `playback.started` sent | M4 text-call responsiveness. |
| `agent_turn_to_first_audio` | `agent.turn` received | first outbound assistant media frame sent | TTS/media path. |
| `barge_in_to_silence` | caller speech onset during playback | prior playback terminal/canceled | Interruption responsiveness. |

The default M6 summary should report at least:

- `speech_to_first_audio`
- `low_energy_to_first_audio`
- `caller_turn_to_playback_started`
- `agent_turn_to_first_audio`
- `barge_in_to_silence` when applicable

## Example `I Heard` Report

```json
{
  "harness_mode": "text-call-echo",
  "turn_id": "turn_...",
  "caller_text": "reset my password",
  "agent_text": "I heard: reset my password",
  "total_speech_to_first_audio_ms": 1640,
  "breakdown": {
    "utterance_speech_to_low_energy_ms": 720,
    "asr_endpoint_wait_ms": 820,
    "asr_local_finish_ms": 35,
    "asr_final_to_caller_turn_ms": 4,
    "text_call_send_caller_turn_ms": 8,
    "app_agent_turn_wait_ms": 22,
    "tts_synthesis_first_chunk_ms": 410,
    "tts_packetize_first_chunk_ms": 18,
    "media_first_frame_send_ms": 12
  },
  "classified_latency": {
    "endpointing_ms": 820,
    "asr_generation_ms": 35,
    "gateway_overhead_ms": 30,
    "network_transport_ms": 22,
    "model_generation_ms": 0,
    "intentional_delay_ms": 0,
    "queue_wait_ms": 0,
    "tts_generation_ms": 410,
    "media_packetization_ms": 18,
    "playback_transport_ms": 12
  },
  "recommendation": "Endpointing and TTS first chunk dominate this deterministic baseline; do not tune app generation."
}
```

Example agent-tmux echo bottleneck:

```json
{
  "harness_mode": "agent-tmux-echo",
  "total_speech_to_first_audio_ms": 11840,
  "classified_latency": {
    "endpointing_ms": 820,
    "intentional_delay_ms": 10000,
    "model_generation_ms": 40,
    "tts_generation_ms": 420
  },
  "recommendation": "The primary latency is input_quiet_for_ms; tune the agent bridge quiet-window before changing ASR or TTS."
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
| `trailing_silence_ms` | Local endpoint tail. | High p95 means slow turn-taking. |
| `final_to_turn_sent_ms` | Gateway dispatch overhead. | Should be small. |
| `caller_turn_to_agent_turn_ms` | App/agent response time. | Impacts responsiveness. |
| `agent_turn_to_playback_started_ms` | TTS queue/start time. | Impacts perceived speed. |
| `playback_finished.status` | Terminal outcome. | High canceled/superseded rates can indicate churn. |
| `partial_count_before_final` | ASR stability. | High counts with poor final quality can indicate decoder instability. |
| `suppressed_count` | Suppressed hallucination-like output. | Backend/model quality signal. |

Aggregate metrics:

| Aggregate | Use |
|---|---|
| p50/p95 `trailing_silence_ms` | Endpoint latency budget. |
| p50/p95 `speech_to_first_audio_ms` | Live perceived latency. |
| p50/p95 `caller_turn_to_playback_started_ms` | Text-call protocol responsiveness. |
| short-turn rate | Premature endpointing / low information. |
| overlong-turn rate | Late endpointing / merged intents. |
| incomplete-turn rate from LLM judge | Semantic endpoint quality. |
| overmerged-turn rate from LLM judge | Semantic endpoint quality. |
| garbled-turn rate from LLM judge | ASR text quality. |
| canceled playback rate | Barge-in aggressiveness. |
| superseded turn rate | Caller overlap / app-agent staleness. |
| invalid turn rate | Protocol correctness. |
| repeated-token suppression rate | ASR hallucination pressure. |

## LLM Judge

The LLM judge must consume text, adjacent turn context, deterministic metrics, and span breakdowns. It should not replace WER when references exist.

Judge input:

```json
{
  "turn_id": "turn_...",
  "harness_mode": "text-call-echo",
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
  "span_breakdown_ms": {
    "endpointing": 820,
    "asr_generation": 35,
    "gateway_overhead": 30,
    "network_transport": 22,
    "intentional_delay": 0,
    "queue_wait": 0,
    "model_generation": 0,
    "tts_generation": 410,
    "playback_transport": 12
  }
}
```

Judge output:

```json
{
  "turn_id": "turn_...",
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
  "reason": "The previous turn is an incomplete prefix and this turn completes it."
}
```

Prompt requirements:

- Return strict JSON only.
- Score `1..5`, where `5` is best.
- Distinguish ASR text quality from endpointing quality.
- Treat short acknowledgements as valid when context supports them.
- Flag overmerged turns when one text contains multiple intents or topic shifts.
- Flag late endpointing when trailing silence and response latency are high.
- Do not blame model generation when `harness_mode` is deterministic and `model_generation_ms` is zero or near-zero.
- Do not infer sensitive identity details from transcript text.

## Current and Proposed Tunables

### Implemented or Present Today

| Area | Knob | Current status | Default/current value | What logs optimize |
|---|---|---:|---:|---|
| Speech detection | `SPEECH_RMS_THRESHOLD` | hard-coded | `180.0` | false starts, missed speech |
| Speech detection | `SPEECH_PEAK_THRESHOLD` | hard-coded | `900` | noise vs speech distinction |
| Endpointing | `ASR_LOCAL_ENDPOINT_TRAILING_SILENCE_MS` | hard-coded live, replay default | `800 ms` | premature vs late endpointing |
| Barge-in onset | `ASR_SPEECH_ONSET_MIN_SILENCE_MS` | hard-coded | `120 ms` | barge-in sensitivity |
| Replay ASR | `--chunk-ms` | CLI implemented | `20` | streaming stability |
| Replay ASR | `--trailing-silence-pad-ms` | CLI implemented | `800` | finalization behavior |
| ASR backend | `--backend`, `asr use` | CLI/REPL implemented | selected backend | backend comparison |
| Codec eval | `--codec` in golden A/B | CLI implemented | selected matrix | Telnyx format comparison |
| Conversation | `conversation barge-in on|off|status` | REPL/socket implemented | `on` | interruption realism |
| Text-call | `MAX_ACTIVE_TEXT_CALL_TURNS` | hard-coded | `32` | runaway app-agent lag |
| Text-call | `MEDIA_READY_TIMEOUT` | hard-coded | `20 s` | setup reliability |
| Text-call | `PLAYBACK_WAIT_TIMEOUT` | hard-coded | `180 s` | hung playback detection |
| Inbound offer | callback timeout | hard-coded | `5 s` | subscriber responsiveness |
| Outbound API | `timeout_ms` | request field implemented | `45000 ms` | outbound setup latency |
| Agent bridge | `reply_timeout_ms` | CLI implemented | `120000 ms` | local agent responsiveness |
| Agent bridge | `input_quiet_for_ms` | CLI implemented | `10000 ms` | non-barge-in tmux UX |
| Agent bridge | input backoff initial/max | CLI implemented | `250 ms` / `5000 ms` | queued transcription delay |
| Agent bridge | trailing Enter delay | CLI implemented | `750 ms` | prompt submission reliability |
| Agent bridge | trailing Enter enabled | CLI implemented | default on | prompt submission reliability |
| ASR suppression | repeated-token thresholds | hard-coded | run `16`, q-run `8` | hallucination suppression |
| TTS chunking | punctuation split | hard-coded | sentence/clause punctuation | first-audio latency vs smoothness |

### Proposed M6 Config Knobs

| Area | Proposed key | Type | Why |
|---|---|---:|---|
| Endpointing | `endpoint.trailing_silence_ms` | integer ms | Primary latency/completeness knob. |
| Endpointing | `endpoint.min_turn_words` | integer | Mark or merge low-information fragments. |
| Endpointing | `endpoint.min_turn_chars` | integer | Mark or merge tiny fragments. |
| Endpointing | `endpoint.merge_window_ms` | integer ms | Merge adjacent incomplete final turns. |
| Endpointing | `endpoint.max_turn_words` | integer | Flag overmerged turns. |
| Endpointing | `endpoint.max_turn_duration_ms` | integer ms | Flag late endpointing / long utterances. |
| Endpointing | `endpoint.profile` | enum | `fast`, `balanced`, `complete`, `noisy` presets. |
| Speech gate | `speech.rms_threshold` | float | Environment-specific VAD tuning. |
| Speech gate | `speech.peak_threshold` | integer | Environment-specific VAD tuning. |
| Speech gate | `speech.onset_min_silence_ms` | integer ms | Barge-in onset sensitivity. |
| ASR suppression | `asr.repeated_token_run_threshold` | integer | Hallucination policy tuning. |
| ASR suppression | `asr.repeated_q_run_threshold` | integer | q-run special case. |
| Text-call | `text_call.max_active_turns` | integer | Backpressure and memory cap. |
| Text-call | `text_call.media_ready_timeout_ms` | integer ms | Setup wait policy. |
| Text-call | `text_call.playback_wait_timeout_ms` | integer ms | Hung playback detection. |
| Text-call | `text_call.latest_response_wins` | bool | Cancel-and-replace policy toggle. |
| Inbound offers | `text_call.callback_timeout_ms` | integer ms | Subscriber responsiveness. |
| LLM judge | `quality_judge.enabled` | bool | Live/offline analysis. |
| LLM judge | `quality_judge.sample_rate` | float | Production sampling. |
| LLM judge | `quality_judge.model` | string | Model selection. |
| LLM judge | `quality_judge.batch_size` | integer | Offline throughput. |
| LLM judge | `quality_judge.include_transcript_text` | bool | Privacy mode. |
| Reports | `quality_report.targets.*` | thresholds | Pass/fail and recommendations. |

## Coherent Config Framework

Proposed TOML:

```toml
[voice_quality]
profile = "balanced"

[voice_quality.speech]
rms_threshold = 180.0
peak_threshold = 900
onset_min_silence_ms = 120

[voice_quality.endpoint]
trailing_silence_ms = 800
min_turn_words = 2
min_turn_chars = 6
merge_window_ms = 350
max_turn_words = 80
max_turn_duration_ms = 12000

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

[voice_quality.llm_judge]
enabled = false
mode = "offline"
sample_rate = 0.05
model = "default"
batch_size = 20
include_transcript_text = true
redact_before_upload = false

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

Precedence:

1. CLI flags
2. REPL/socket runtime commands
3. loaded state/config file
4. profile defaults
5. code defaults

## Proposed CLI Surface

Gateway startup:

```text
telnyx-gateway \
  --quality-config ./telnyx-quality.toml \
  --quality-profile balanced \
  --endpoint-trailing-silence-ms 800 \
  --speech-rms-threshold 180 \
  --speech-peak-threshold 900 \
  --speech-onset-min-silence-ms 120 \
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

```text
quality status
quality profile fast|balanced|complete|noisy
quality endpoint status
quality endpoint trailing-silence-ms <ms>
quality endpoint min-turn-words <n>
quality endpoint merge-window-ms <ms>
quality speech status
quality speech rms-threshold <value>
quality speech peak-threshold <value>
quality speech onset-min-silence-ms <ms>
quality text-call status
quality text-call max-active-turns <n>
quality text-call latest-response-wins on|off
quality turn-log on <path>
quality turn-log off
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
```

## Proposed Socket Commands

Gateway command socket examples:

```json
{"id":"q1","command":"quality status"}
{"id":"q2","command":"quality endpoint trailing-silence-ms 950"}
{"id":"q3","command":"quality speech rms-threshold 220"}
{"id":"q4","command":"quality turn-log on /tmp/motlie-turns.jsonl"}
{"id":"q5","command":"quality report /tmp/motlie-quality-report.json"}
{"id":"q6","command":"quality recommendations /tmp/motlie-tuning.json"}
{"id":"q7","command":"quality profile-roundtrip gateway-echo /tmp/gateway-echo-report.json"}
```

Expected response shape:

```json
{
  "id": "q2",
  "ok": true,
  "data": {
    "quality": {
      "endpoint": {
        "trailing_silence_ms": 950
      }
    }
  }
}
```

Agent daemon socket examples:

```json
{"id":"a1","command":"status"}
{"id":"a2","command":"bridge input-quiet-for-ms 8000"}
{"id":"a3","command":"bridge backoff initial-ms 250 max-ms 5000"}
{"id":"a4","command":"bridge trailing-enter delay-ms 750"}
{"id":"a5","command":"bridge trailing-enter off"}
{"id":"a6","command":"gateway quality status"}
{"id":"a7","command":"gateway quality endpoint trailing-silence-ms 950"}
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
    "endpoint_trailing_silence_ms": 800
  }
}
```

## Analysis Workflow

Offline replay:

1. Capture or replay calls.
2. Emit turn JSONL and span JSONL.
3. Run deterministic `turn-report`.
4. Run LLM `turn-judge` if transcript text may be analyzed.
5. Run tuning recommendation generator.
6. Compare candidate configs.
7. Promote a profile only when it improves quality without violating latency targets.

Live sampling:

1. Enable turn logging locally.
2. Optionally enable sampled LLM judging.
3. Review daily or weekly reports.
4. Apply runtime config changes through REPL/socket.
5. Persist state.
6. Compare before/after windows.

Example live flow:

```text
quality turn-log on ./logs/telnyx-turns.jsonl
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

- `endpointing_quality` comes from LLM judge and deterministic premature/overmerged rates.
- `asr_text_quality` comes from WER when references exist, otherwise LLM judge and suppression rates.
- `conversation_realism_quality` uses barge-in appropriateness, playback cancel/supersede rates, and response start latency.
- `latency_score` penalizes p95 endpoint and turn-to-playback latency.
- `stability_score` penalizes invalid turns, failures, and high suppression/hallucination rates.

Recommended report output:

```json
{
  "recommended_profile": "balanced",
  "changes": [
    {
      "knob": "endpoint.trailing_silence_ms",
      "from": 800,
      "to": 950,
      "confidence": "medium",
      "expected_effect": "reduce premature cuts with modest p95 latency increase"
    },
    {
      "knob": "speech.rms_threshold",
      "from": 180,
      "to": 220,
      "confidence": "low",
      "expected_effect": "reduce false speech starts on noisy calls; validate on more samples"
    }
  ]
}
```

## Implementation Phases

### Phase 1 - Span and Turn Logging

- [ ] Add stable span event types and JSON serialization.
- [ ] Emit `text_call.caller_turn.sent` on successful caller turn send.
- [ ] Emit app-agent, playback, endpointing, and barge-in span events.
- [ ] Add span links for barge-in canceling prior playback.
- [ ] Ensure logs do not contain raw phone numbers.

### Phase 2 - Deterministic Harnesses

- [ ] Add `gateway-echo` profile report.
- [ ] Add `text-call-echo` app harness.
- [ ] Add `agent-tmux-echo` deterministic tmux harness.
- [ ] Compare harness modes using the overhead formulas above.

### Phase 3 - Config Framework

- [ ] Add `VoiceQualityConfig` sections for speech, endpoint, text-call, barge-in, judge, and targets.
- [ ] Wire CLI overrides.
- [ ] Wire REPL/socket commands.
- [ ] Persist config through state dump/load.
- [ ] Replace live hard-coded constants with config-backed values while preserving defaults.

### Phase 4 - Reports and LLM Judge

- [ ] Add deterministic `turn-report`.
- [ ] Add strict JSON `turn-judge`.
- [ ] Add span-aware latency attribution in judge input/output.
- [ ] Add tuning recommendations.
- [ ] Support redacted/no-transcript privacy modes.

## Acceptance Criteria

M6 profiling is complete when:

- [ ] A live or replayed call can produce structured turn and span JSONL.
- [ ] `gateway-echo` profiling separates endpointing, ASR final flush, TTS first chunk, packetization, and first media frame timings.
- [ ] `text-call-echo` profiling separates M4 WebSocket/app protocol overhead from model generation.
- [ ] `agent-tmux-echo` profiling separates activity quiet-window wait, backoff, trailing Enter delay, tmux injection, reply wait, and scrape latency.
- [ ] `real-agent` profiling can attribute incremental latency over deterministic echo baselines to app/model generation or harness waits.
- [ ] Barge-in reports link speech onset/partial/final ASR to cancel request and playback terminal `canceled` status.
- [ ] LLM judge output includes `dominant_latency_category`, `generation_latency_suspected`, `intentional_delay_suspected`, `queueing_suspected`, and `recommended_next_probe`.
- [ ] Reports can answer: "Was this turn slow because of ASR endpointing, agent generation, intentional quiet-window delay, queueing, TTS, or media transport?"
- [ ] All logs and reports obey phone-number redaction requirements.
